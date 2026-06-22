from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from funasr import AutoModel

import config
from leader_store import LeaderStore, cosine_similarity
from text_normalizer import normalize_asr_text


logger = logging.getLogger(__name__)


def detect_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def embedding_to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    return [float(v) for v in value]


class FunASRService:
    def __init__(self) -> None:
        self.device = detect_device()
        self.model: AutoModel | None = None

    def load(self) -> None:
        logger.info("Loading FunASR models")
        logger.info("ASR=%s", config.ASR_MODEL)
        logger.info("VAD=%s", config.VAD_MODEL)
        logger.info("PUNC=%s", config.PUNC_MODEL)
        logger.info("SPK=%s", config.SPK_MODEL)
        logger.info("DEVICE=%s", self.device)
        logger.info("VAD_MAX_SINGLE_SEGMENT_TIME_MS=%s", config.VAD_MAX_SINGLE_SEGMENT_TIME_MS)
        logger.info("BATCH_SIZE_S=%s", config.BATCH_SIZE_S)
        if str(self.device).startswith("npu"):
            import torch
            import torch_npu  # noqa: F401

            if not hasattr(torch, "npu"):
                raise RuntimeError("torch_npu was imported but torch.npu is unavailable")
            device_count = torch.npu.device_count()
            if device_count < 1:
                raise RuntimeError("no Ascend NPU is visible to the ASR process")
            logger.info("Ascend NPU backend is ready: devices=%s", device_count)
        self.model = AutoModel(
            model=config.ASR_MODEL,
            vad_model=config.VAD_MODEL,
            punc_model=config.PUNC_MODEL,
            spk_model=config.SPK_MODEL,
            device=self.device,
            disable_update=True,
            vad_kwargs=self._vad_kwargs(),
        )
        logger.info("FunASR models are ready")

    @staticmethod
    def _vad_kwargs() -> dict[str, Any]:
        if config.VAD_MAX_SINGLE_SEGMENT_TIME_MS <= 0:
            return {}
        return {"max_single_segment_time": config.VAD_MAX_SINGLE_SEGMENT_TIME_MS}

    def transcribe(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        hotwords: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        kwargs: dict[str, Any] = {
            "input": audio_path,
            "batch_size_s": config.BATCH_SIZE_S,
        }
        selected_hotwords = config.ASR_HOTWORDS if hotwords is None else hotwords.strip()
        if selected_hotwords:
            kwargs["hotword"] = selected_hotwords
        if num_speakers and num_speakers > 0:
            kwargs["preset_spk_num"] = num_speakers

        raw_results = self.model.generate(**kwargs)
        full_text: list[str] = []
        segments: list[dict[str, Any]] = []

        for result in raw_results or []:
            full_text.append(result.get("text", ""))
            for item in result.get("sentence_info", []):
                start = round(float(item.get("start", 0)) / 1000.0, 3)
                end = round(float(item.get("end", 0)) / 1000.0, 3)
                raw_text = str(item.get("text", ""))
                text = normalize_asr_text(raw_text)
                segment = {
                    "speaker": str(item.get("spk", 0)),
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "text": text,
                    "is_leader": False,
                    "leader_id": None,
                }
                if text != raw_text:
                    segment["raw_text"] = raw_text
                    segment["rule_normalized"] = True
                segments.append(segment)

        return self._refine_speaker_labels(audio_path, segments, num_speakers)

    @staticmethod
    def transcript_quality(
        segments: list[dict[str, Any]],
        audio_duration: float | None = None,
    ) -> dict[str, Any]:
        if not segments:
            return {
                "status": "empty",
                "warnings": ["no_segments"],
                "segment_count": 0,
                "speaker_count": 0,
            }

        durations = [
            max(0.0, float(segment["end_time"]) - float(segment["start_time"]))
            for segment in segments
        ]
        total_segment_seconds = sum(durations)
        if audio_duration is None:
            audio_duration = max(float(segment["end_time"]) for segment in segments)

        speaker_seconds: dict[str, float] = {}
        speaker_segments: dict[str, int] = {}
        for segment, duration in zip(segments, durations):
            speaker = str(segment.get("speaker", ""))
            speaker_seconds[speaker] = speaker_seconds.get(speaker, 0.0) + duration
            speaker_segments[speaker] = speaker_segments.get(speaker, 0) + 1

        dominant_speaker = max(speaker_seconds, key=speaker_seconds.get)
        dominant_ratio = speaker_seconds[dominant_speaker] / max(total_segment_seconds, 0.001)
        max_segment_seconds = max(durations)
        segment_count = len(segments)
        speaker_count = len(speaker_seconds)
        segments_per_hour = segment_count / max(audio_duration / 3600.0, 0.001)
        coverage_ratio = total_segment_seconds / max(audio_duration, 0.001)

        warnings: list[str] = []
        if max_segment_seconds > config.ASR_QUALITY_MAX_SEGMENT_SECONDS:
            warnings.append("segment_too_long")
        if speaker_count < config.ASR_QUALITY_MIN_EXPECTED_SPEAKERS and audio_duration >= 300:
            warnings.append("too_few_speakers")
        if dominant_ratio > config.ASR_QUALITY_DOMINANT_SPEAKER_RATIO and audio_duration >= 300:
            warnings.append("dominant_speaker_too_large")
        if segments_per_hour < config.ASR_QUALITY_MIN_SEGMENTS_PER_HOUR and audio_duration >= 300:
            warnings.append("too_few_segments")

        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "segment_count": segment_count,
            "speaker_count": speaker_count,
            "audio_duration_seconds": round(audio_duration, 2),
            "total_segment_seconds": round(total_segment_seconds, 2),
            "coverage_ratio": round(coverage_ratio, 4),
            "max_segment_seconds": round(max_segment_seconds, 2),
            "average_segment_seconds": round(total_segment_seconds / segment_count, 2),
            "segments_per_hour": round(segments_per_hour, 2),
            "dominant_speaker": dominant_speaker,
            "dominant_speaker_ratio": round(dominant_ratio, 4),
            "speaker_seconds": {
                speaker: round(seconds, 2)
                for speaker, seconds in sorted(
                    speaker_seconds.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "speaker_segments": dict(sorted(speaker_segments.items())),
        }

    @staticmethod
    def should_retry_diarization(
        quality: dict[str, Any],
        requested_speakers: int | None,
        audio_duration: float | None,
    ) -> bool:
        if requested_speakers or not config.ASR_AUTO_RETRY_BAD_DIARIZATION:
            return False
        if config.ASR_FALLBACK_SPEAKER_NUM <= 0:
            return False
        if audio_duration is None or audio_duration < config.ASR_RETRY_MIN_AUDIO_SECONDS:
            return False
        warnings = set(quality.get("warnings") or [])
        max_segment = float(quality.get("max_segment_seconds") or 0.0)
        dominant_ratio = float(quality.get("dominant_speaker_ratio") or 0.0)
        speaker_count = int(quality.get("speaker_count") or 0)
        if "segment_too_long" in warnings and max_segment >= config.ASR_RETRY_MAX_SEGMENT_SECONDS:
            return True
        return (
            "dominant_speaker_too_large" in warnings
            and dominant_ratio >= config.ASR_RETRY_DOMINANT_SPEAKER_RATIO
            and speaker_count <= config.ASR_QUALITY_MIN_EXPECTED_SPEAKERS
        )

    def extract_voiceprint(self, audio_path: str) -> list[float]:
        self._ensure_loaded()
        if self.model.spk_model is None:
            raise RuntimeError("speaker model is not loaded")
        spk_kwargs = dict(getattr(self.model, "spk_kwargs", {}) or {})
        spk_kwargs["device"] = self.device
        results = self.model.inference(audio_path, model=self.model.spk_model, kwargs=spk_kwargs)
        if not results or "spk_embedding" not in results[0]:
            raise RuntimeError("could not extract speaker embedding")
        return embedding_to_list(results[0]["spk_embedding"])

    def annotate_leaders(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        leader_store: LeaderStore,
        threshold: float,
        return_scores: bool = False,
    ) -> list[dict[str, Any]]:
        if not leader_store.has_samples():
            return segments

        segment_scores: dict[int, list[dict[str, Any]]] = {}
        for index, segment in enumerate(segments):
            scores = self._score_segment(audio_path, segment, leader_store)
            segment_scores[index] = scores
            if return_scores:
                segment["leader_candidates"] = scores

        speaker_matches = self._identify_speakers(
            audio_path,
            segments,
            leader_store,
            segment_scores,
            threshold,
        )
        for segment in segments:
            match = speaker_matches.get(segment["speaker"])
            if match:
                self._mark_leader(segment, match)
        if config.LEADER_MERGE_MATCHED_SPEAKERS:
            self._merge_matched_leader_speakers(segments, speaker_matches)
        return segments

    def _identify_speakers(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        leader_store: LeaderStore,
        segment_scores: dict[int, list[dict[str, Any]]],
        threshold: float,
    ) -> dict[str, dict[str, Any]]:
        speaker_segments: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, segment in enumerate(segments):
            if segment["end_time"] - segment["start_time"] >= config.LEADER_MIN_SEGMENT_SECONDS:
                speaker_segments.setdefault(segment["speaker"], []).append((index, segment))

        leader_candidates: dict[str, list[dict[str, Any]]] = {}
        for speaker, indexed_items in speaker_segments.items():
            items = [item for _, item in indexed_items]
            combined_path = self._extract_speaker_audio(audio_path, items)
            try:
                embedding = self.extract_voiceprint(combined_path)
                speaker_scores = leader_store.score_all(embedding)
                segment_evidence = self._speaker_segment_evidence(indexed_items, segment_scores)
                for score in speaker_scores:
                    leader_id = score["leader_id"]
                    evidence = max(score["score"], segment_evidence.get(leader_id, {}).get("top_avg", -1.0))
                    next_leader_score = self._next_score(speaker_scores, leader_id)
                    candidate = {
                        "speaker": speaker,
                        "leader_id": leader_id,
                        "score": round(evidence, 5),
                        "speaker_score": score["score"],
                        "segment_top_avg": segment_evidence.get(leader_id, {}).get("top_avg"),
                        "support_segments": segment_evidence.get(leader_id, {}).get("support", 0),
                        "speaker_margin": round(score["score"] - next_leader_score, 5),
                        "speaker_rank": next(
                            index
                            for index, item in enumerate(speaker_scores)
                            if item["leader_id"] == leader_id
                        ),
                    }
                    if self._candidate_has_evidence(candidate, threshold):
                        leader_candidates.setdefault(leader_id, []).append(candidate)
            finally:
                try:
                    os.remove(combined_path)
                except OSError:
                    pass

        return self._select_leader_matches(leader_candidates)

    def _select_leader_matches(
        self,
        leader_candidates: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        candidates = [
            candidate
            for leader_items in leader_candidates.values()
            for candidate in leader_items
        ]
        candidates.sort(
            key=lambda item: (
                item["score"],
                item["speaker_score"],
                item["support_segments"],
            ),
            reverse=True,
        )

        matches: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            speaker = candidate["speaker"]
            if speaker in matches:
                continue
            matches[speaker] = {
                "leader_id": candidate["leader_id"],
                "score": candidate["score"],
                "confidence": self._confidence_label(candidate),
            }
        return matches

    def _speaker_segment_evidence(
        self,
        indexed_segments: list[tuple[int, dict[str, Any]]],
        segment_scores: dict[int, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        by_leader: dict[str, list[float]] = {}
        for index, _ in indexed_segments:
            scores = segment_scores.get(index, [])
            if not scores:
                continue
            best = scores[0]
            runner_up = scores[1]["score"] if len(scores) > 1 else -1.0
            if best["score"] - runner_up >= config.LEADER_SCORE_MARGIN:
                by_leader.setdefault(best["leader_id"], []).append(best["score"])

        evidence: dict[str, dict[str, Any]] = {}
        for leader_id, values in by_leader.items():
            top_values = sorted(values, reverse=True)[:3]
            evidence[leader_id] = {
                "top_avg": round(sum(top_values) / len(top_values), 5),
                "support": len(values),
            }
        return evidence

    def _score_segment(
        self,
        audio_path: str,
        segment: dict[str, Any],
        leader_store: LeaderStore,
    ) -> list[dict[str, Any]]:
        if segment["end_time"] - segment["start_time"] < config.LEADER_MIN_SEGMENT_SECONDS:
            return []
        clip_path = self._extract_clip(audio_path, segment["start_time"], segment["end_time"])
        try:
            embedding = self.extract_voiceprint(clip_path)
            return leader_store.score_all(embedding)
        finally:
            try:
                os.remove(clip_path)
            except OSError:
                pass

    @staticmethod
    def _next_score(scores: list[dict[str, Any]], leader_id: str) -> float:
        for score in scores:
            if score["leader_id"] != leader_id:
                return score["score"]
        return -1.0

    def _candidate_has_evidence(self, candidate: dict[str, Any], threshold: float) -> bool:
        if candidate["score"] >= threshold:
            if candidate["speaker_score"] >= 0.50:
                return True
            if candidate["speaker_margin"] < config.LEADER_SCORE_MARGIN:
                return False
            if candidate["speaker_score"] >= config.LEADER_SPEAKER_THRESHOLD:
                return True
            if candidate["support_segments"] >= 2 and candidate["score"] >= config.LEADER_SEGMENT_THRESHOLD:
                return True

        if (
            candidate["speaker_rank"] != 0
            or candidate["speaker_score"] < config.LEADER_RELATIVE_MIN_SCORE
        ):
            return False

        if (
            candidate["speaker_margin"] >= config.LEADER_RELATIVE_MARGIN
            and candidate["support_segments"] >= config.LEADER_RELATIVE_SUPPORT_SEGMENTS
        ):
            return True

        if (
            candidate["speaker_score"] >= config.LEADER_SPEAKER_THRESHOLD
            and candidate["speaker_margin"] >= config.LEADER_RELATIVE_MARGIN
            and candidate["support_segments"] >= config.LEADER_SHORT_SUPPORT_SEGMENTS
        ):
            return True

        return (
            candidate["score"] >= config.LEADER_STRONG_SUPPORT_MIN_SCORE
            and candidate["speaker_margin"] >= config.LEADER_STRONG_SUPPORT_MARGIN
            and candidate["support_segments"] >= config.LEADER_STRONG_SUPPORT_SEGMENTS
        )

    @staticmethod
    def _confidence_label(candidate: dict[str, Any]) -> str:
        if candidate["score"] >= 0.55 and candidate["speaker_margin"] >= 0.08:
            return "high"
        if candidate["score"] >= 0.40 and candidate["speaker_margin"] >= 0.04:
            return "medium"
        return "low"

    def _mark_leader(self, segment: dict[str, Any], match: dict[str, Any]) -> None:
        segment["is_leader"] = True
        segment["leader_id"] = match["leader_id"]

    @staticmethod
    def _merge_matched_leader_speakers(
        segments: list[dict[str, Any]],
        speaker_matches: dict[str, dict[str, Any]],
    ) -> None:
        speakers_by_leader: dict[str, list[str]] = {}
        for speaker, match in speaker_matches.items():
            speakers_by_leader.setdefault(match["leader_id"], []).append(speaker)

        aliases: dict[str, str] = {}
        for speakers in speakers_by_leader.values():
            if len(speakers) < 2:
                continue
            canonical = max(
                speakers,
                key=lambda speaker: speaker_matches[speaker]["score"],
            )
            for speaker in speakers:
                aliases[speaker] = canonical

        if not aliases:
            return

        roots_in_order: list[str] = []
        for segment in segments:
            speaker = aliases.get(segment["speaker"], segment["speaker"])
            if speaker not in roots_in_order:
                roots_in_order.append(speaker)
            segment["speaker"] = str(roots_in_order.index(speaker))

    def _extract_clip(self, audio_path: str, start: float, end: float) -> str:
        start = max(0.0, start - config.SEGMENT_PADDING_MS / 1000.0)
        duration = max(0.2, end - start + config.SEGMENT_PADDING_MS / 1000.0)
        out_path = str(Path(tempfile.gettempdir()) / f"leader_clip_{uuid.uuid4().hex}.wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path

    def _extract_speaker_audio(self, audio_path: str, segments: list[dict[str, Any]]) -> str:
        clip_paths = []
        list_path = str(Path(tempfile.gettempdir()) / f"leader_concat_{uuid.uuid4().hex}.txt")
        out_path = str(Path(tempfile.gettempdir()) / f"leader_speaker_{uuid.uuid4().hex}.wav")
        try:
            longest = sorted(segments, key=lambda item: item["end_time"] - item["start_time"], reverse=True)[:8]
            for segment in sorted(longest, key=lambda item: item["start_time"]):
                clip_paths.append(self._extract_clip(audio_path, segment["start_time"], segment["end_time"]))
            with open(list_path, "w", encoding="utf-8") as f:
                for path in clip_paths:
                    escaped = path.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            return out_path
        finally:
            for path in clip_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.remove(list_path)
            except OSError:
                pass

    def _refine_speaker_labels(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        target_count: int | None,
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for segment in segments:
            if segment["end_time"] - segment["start_time"] >= config.LEADER_MIN_SEGMENT_SECONDS:
                groups.setdefault(segment["speaker"], []).append(segment)
        if len(groups) < 2:
            return segments

        profiles: dict[str, dict[str, Any]] = {}
        for speaker, items in groups.items():
            combined_path = self._extract_speaker_audio(audio_path, items)
            try:
                profiles[speaker] = {
                    "embedding": self.extract_voiceprint(combined_path),
                    "duration": sum(item["end_time"] - item["start_time"] for item in items),
                }
            finally:
                try:
                    os.remove(combined_path)
                except OSError:
                    pass

        aliases = {speaker: speaker for speaker in groups}
        if target_count and target_count > 0:
            while len({self._speaker_root(aliases, speaker) for speaker in groups}) > target_count:
                pair = self._most_similar_active_speakers(profiles, aliases)
                if pair is None:
                    break
                left, right, score = pair
                self._merge_speaker_profiles(left, right, profiles, aliases)
                logger.info(
                    "Merged speaker %s into %s to satisfy requested count, similarity=%.5f",
                    left,
                    right,
                    score,
                )
        else:
            total_duration = sum(profile["duration"] for profile in profiles.values())
            for speaker in sorted(profiles, key=lambda item: profiles[item]["duration"]):
                if self._speaker_root(aliases, speaker) != speaker:
                    continue
                duration = profiles[speaker]["duration"]
                is_minor = (
                    duration <= config.SPEAKER_AUTO_MERGE_MAX_SECONDS
                    or duration / max(total_duration, 0.001) <= config.SPEAKER_AUTO_MERGE_MAX_RATIO
                )
                if not is_minor:
                    continue
                candidates = []
                for other in profiles:
                    if other == speaker or self._speaker_root(aliases, other) != other:
                        continue
                    score = cosine_similarity(
                        profiles[speaker]["embedding"],
                        profiles[other]["embedding"],
                    )
                    candidates.append((score, other))
                candidates.sort(reverse=True)
                if not candidates:
                    continue
                best_score, best_speaker = candidates[0]
                runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
                if (
                    best_score >= config.SPEAKER_AUTO_MERGE_THRESHOLD
                    and best_score - runner_up >= config.SPEAKER_AUTO_MERGE_MARGIN
                ):
                    self._merge_speaker_profiles(speaker, best_speaker, profiles, aliases)
                    logger.info(
                        "Auto-merged minor speaker %s into %s, similarity=%.5f, margin=%.5f",
                        speaker,
                        best_speaker,
                        best_score,
                        best_score - runner_up,
                    )

        for segment in segments:
            segment["speaker"] = self._speaker_root(aliases, segment["speaker"])
        self._smooth_short_speaker_switches(audio_path, segments, profiles)

        roots_in_order: list[str] = []
        for segment in segments:
            root = segment["speaker"]
            if root not in roots_in_order:
                roots_in_order.append(root)
            segment["speaker"] = str(roots_in_order.index(root))
        return segments

    def _smooth_short_speaker_switches(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        for index in range(1, len(segments) - 1):
            previous = segments[index - 1]
            current = segments[index]
            following = segments[index + 1]
            neighbor_speaker = previous["speaker"]
            current_speaker = current["speaker"]
            if neighbor_speaker != following["speaker"] or current_speaker == neighbor_speaker:
                continue

            duration = current["end_time"] - current["start_time"]
            previous_gap = max(0.0, current["start_time"] - previous["end_time"])
            following_gap = max(0.0, following["start_time"] - current["end_time"])
            if (
                duration < config.LEADER_MIN_SEGMENT_SECONDS
                or duration > config.SPEAKER_JITTER_MAX_SECONDS
                or previous_gap > config.SPEAKER_JITTER_MAX_GAP_SECONDS
                or following_gap > config.SPEAKER_JITTER_MAX_GAP_SECONDS
                or neighbor_speaker not in profiles
                or current_speaker not in profiles
            ):
                continue

            clip_path = self._extract_clip(audio_path, current["start_time"], current["end_time"])
            try:
                try:
                    embedding = self.extract_voiceprint(clip_path)
                    neighbor_score = cosine_similarity(
                        embedding,
                        profiles[neighbor_speaker]["embedding"],
                    )
                    current_score = cosine_similarity(
                        embedding,
                        profiles[current_speaker]["embedding"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not score short speaker switch at %.2f-%.2f: %s",
                        current["start_time"],
                        current["end_time"],
                        exc,
                    )
                    continue
            finally:
                try:
                    os.remove(clip_path)
                except OSError:
                    pass

            if (
                neighbor_score >= config.SPEAKER_JITTER_MIN_SIMILARITY
                and neighbor_score - current_score >= config.SPEAKER_JITTER_MARGIN
            ):
                current["speaker"] = neighbor_speaker
                logger.info(
                    "Smoothed short speaker switch %s into %s at %.2f-%.2f, "
                    "similarity=%.5f, margin=%.5f",
                    current_speaker,
                    neighbor_speaker,
                    current["start_time"],
                    current["end_time"],
                    neighbor_score,
                    neighbor_score - current_score,
                )

    @staticmethod
    def _speaker_root(aliases: dict[str, str], speaker: str) -> str:
        while aliases[speaker] != speaker:
            aliases[speaker] = aliases[aliases[speaker]]
            speaker = aliases[speaker]
        return speaker

    def _most_similar_active_speakers(
        self,
        profiles: dict[str, dict[str, Any]],
        aliases: dict[str, str],
    ) -> tuple[str, str, float] | None:
        active = [speaker for speaker in profiles if self._speaker_root(aliases, speaker) == speaker]
        best: tuple[str, str, float] | None = None
        for index, left in enumerate(active):
            for right in active[index + 1 :]:
                score = cosine_similarity(
                    profiles[left]["embedding"],
                    profiles[right]["embedding"],
                )
                if best is None or score > best[2]:
                    best = (left, right, score)
        return best

    @staticmethod
    def _merge_speaker_profiles(
        source: str,
        target: str,
        profiles: dict[str, dict[str, Any]],
        aliases: dict[str, str],
    ) -> None:
        if profiles[source]["duration"] > profiles[target]["duration"]:
            source, target = target, source
        source_duration = profiles[source]["duration"]
        target_duration = profiles[target]["duration"]
        total_duration = source_duration + target_duration
        profiles[target]["embedding"] = [
            (left * target_duration + right * source_duration) / total_duration
            for left, right in zip(
                profiles[target]["embedding"],
                profiles[source]["embedding"],
            )
        ]
        profiles[target]["duration"] = total_duration
        aliases[source] = target

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("model is not loaded")

    @staticmethod
    def _merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for segment in segments:
            gap = (
                max(0.0, segment["start_time"] - merged[-1]["end_time"])
                if merged
                else 0.0
            )
            if (
                merged
                and merged[-1]["speaker"] == segment["speaker"]
                and FunASRService._leader_id(merged[-1]) == FunASRService._leader_id(segment)
                and gap <= config.SEGMENT_MERGE_MAX_GAP_SECONDS
            ):
                left_text = merged[-1].get("text", "")
                right_text = segment.get("text", "")
                left_raw = merged[-1].get("raw_text", left_text)
                right_raw = segment.get("raw_text", right_text)
                merged[-1]["text"] = FunASRService._join_segment_text(left_text, right_text)
                if merged[-1].get("rule_normalized") or segment.get("rule_normalized"):
                    merged[-1]["rule_normalized"] = True
                    merged[-1]["raw_text"] = FunASRService._join_segment_text(left_raw, right_raw)
                merged[-1]["end_time"] = max(merged[-1]["end_time"], segment["end_time"])
                merged[-1]["is_leader"] = merged[-1].get("is_leader", False) or segment.get("is_leader", False)
                if segment.get("leader_id"):
                    merged[-1]["leader_id"] = segment["leader_id"]
            else:
                segment.setdefault("is_leader", False)
                segment.setdefault("leader_id", None)
                segment.pop("leader", None)
                merged.append(segment)
        return merged

    @staticmethod
    def _join_segment_text(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        needs_space = (
            left[-1].isascii()
            and right[0].isascii()
            and left[-1].isalnum()
            and right[0].isalnum()
        )
        return f"{left} {right}" if needs_space else f"{left}{right}"

    @staticmethod
    def _leader_id(segment: dict[str, Any]) -> str | None:
        return segment.get("leader_id")
