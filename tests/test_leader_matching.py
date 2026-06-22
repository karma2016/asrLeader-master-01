from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.modules.setdefault("funasr", SimpleNamespace(AutoModel=object))

import config
from model_service import FunASRService
from post_processor import TranscriptPostProcessor
from text_normalizer import normalize_asr_text, repair_mojibake


def candidate(
    speaker: str,
    leader_id: str,
    *,
    score: float,
    speaker_score: float,
    support: int,
    margin: float,
    rank: int = 0,
) -> dict:
    return {
        "speaker": speaker,
        "leader_id": leader_id,
        "score": score,
        "speaker_score": speaker_score,
        "segment_top_avg": score,
        "support_segments": support,
        "speaker_margin": margin,
        "speaker_rank": rank,
    }


class LeaderMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FunASRService.__new__(FunASRService)

    def test_same_leader_can_match_multiple_split_speakers(self) -> None:
        matches = self.service._select_leader_matches(
            {
                "liu": [
                    candidate(
                        "2",
                        "liu",
                        score=0.62,
                        speaker_score=0.60,
                        support=12,
                        margin=0.30,
                    ),
                    candidate(
                        "3",
                        "liu",
                        score=0.51,
                        speaker_score=0.49,
                        support=4,
                        margin=0.22,
                    ),
                    candidate(
                        "4",
                        "liu",
                        score=0.47,
                        speaker_score=0.45,
                        support=3,
                        margin=0.18,
                    ),
                ]
            }
        )

        self.assertEqual({"2", "3", "4"}, set(matches))
        self.assertTrue(all(match["leader_id"] == "liu" for match in matches.values()))

    def test_best_leader_wins_when_one_speaker_has_multiple_candidates(self) -> None:
        matches = self.service._select_leader_matches(
            {
                "liu": [
                    candidate(
                        "2",
                        "liu",
                        score=0.48,
                        speaker_score=0.46,
                        support=4,
                        margin=0.15,
                    )
                ],
                "zhu": [
                    candidate(
                        "2",
                        "zhu",
                        score=0.56,
                        speaker_score=0.52,
                        support=6,
                        margin=0.20,
                    )
                ],
            }
        )

        self.assertEqual("zhu", matches["2"]["leader_id"])

    def test_consistent_segments_accept_lower_margin(self) -> None:
        zhu = candidate(
            "4",
            "zhu",
            score=0.26335,
            speaker_score=0.25542,
            support=8,
            margin=0.04079,
        )

        with (
            patch.object(config, "LEADER_STRONG_SUPPORT_SEGMENTS", 6),
            patch.object(config, "LEADER_STRONG_SUPPORT_MIN_SCORE", 0.20),
            patch.object(config, "LEADER_STRONG_SUPPORT_MARGIN", 0.04),
        ):
            self.assertTrue(self.service._candidate_has_evidence(zhu, 0.45))

    def test_weak_consistent_segments_still_fail(self) -> None:
        weak = candidate(
            "5",
            "zhu",
            score=0.11452,
            speaker_score=0.03826,
            support=8,
            margin=0.05039,
        )

        self.assertFalse(self.service._candidate_has_evidence(weak, 0.45))

    def test_short_split_cluster_can_match_with_strong_speaker_score(self) -> None:
        short_split = candidate(
            "3",
            "liu",
            score=0.37,
            speaker_score=0.37,
            support=2,
            margin=0.25,
        )

        self.assertTrue(self.service._candidate_has_evidence(short_split, 0.45))

    def test_matched_speaker_labels_are_merged_and_renumbered(self) -> None:
        segments = [
            {"speaker": "2", "leader_id": "liu"},
            {"speaker": "5", "leader_id": None},
            {"speaker": "3", "leader_id": "liu"},
            {"speaker": "4", "leader_id": "liu"},
        ]
        matches = {
            "2": {"leader_id": "liu", "score": 0.62},
            "3": {"leader_id": "liu", "score": 0.51},
            "4": {"leader_id": "liu", "score": 0.47},
        }

        self.service._merge_matched_leader_speakers(segments, matches)

        self.assertEqual(segments[0]["speaker"], segments[2]["speaker"])
        self.assertEqual(segments[0]["speaker"], segments[3]["speaker"])
        self.assertNotEqual(segments[0]["speaker"], segments[1]["speaker"])

    def test_merge_adjacent_segments_preserves_rule_normalized_raw_text(self) -> None:
        segments = [
            {
                "speaker": "0",
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "协同办公",
                "raw_text": "协动办公",
                "rule_normalized": True,
                "leader_id": None,
            },
            {
                "speaker": "0",
                "start_time": 1.2,
                "end_time": 2.0,
                "text": "可以接入。",
                "leader_id": None,
            },
        ]

        merged = FunASRService._merge_adjacent_segments(segments)

        self.assertEqual(1, len(merged))
        self.assertTrue(merged[0]["rule_normalized"])
        self.assertEqual("协动办公可以接入。", merged[0]["raw_text"])
        self.assertEqual("协同办公可以接入。", merged[0]["text"])

    def test_transcript_quality_flags_long_mixed_speaker_segments(self) -> None:
        segments = [
            {"speaker": "0", "start_time": 0.0, "end_time": 1545.0},
            {"speaker": "1", "start_time": 1545.0, "end_time": 1600.0},
        ]

        with (
            patch.object(config, "ASR_QUALITY_MAX_SEGMENT_SECONDS", 120.0),
            patch.object(config, "ASR_QUALITY_DOMINANT_SPEAKER_RATIO", 0.70),
            patch.object(config, "ASR_QUALITY_MIN_EXPECTED_SPEAKERS", 2),
            patch.object(config, "ASR_QUALITY_MIN_SEGMENTS_PER_HOUR", 60.0),
        ):
            quality = FunASRService.transcript_quality(segments, audio_duration=2114.7)

        self.assertEqual("warning", quality["status"])
        self.assertIn("segment_too_long", quality["warnings"])
        self.assertIn("dominant_speaker_too_large", quality["warnings"])
        self.assertIn("too_few_segments", quality["warnings"])
        self.assertEqual("0", quality["dominant_speaker"])

    def test_transcript_quality_accepts_balanced_segments(self) -> None:
        segments = []
        for index in range(120):
            speaker = str(index % 3)
            start = index * 10.0
            segments.append(
                {
                    "speaker": speaker,
                    "start_time": start,
                    "end_time": start + 8.0,
                }
            )

        quality = FunASRService.transcript_quality(segments, audio_duration=1200.0)

        self.assertEqual("ok", quality["status"])
        self.assertEqual([], quality["warnings"])
        self.assertEqual(3, quality["speaker_count"])

    def test_bad_diarization_quality_triggers_auto_retry(self) -> None:
        quality = {
            "warnings": ["segment_too_long", "dominant_speaker_too_large"],
            "max_segment_seconds": 353.28,
            "dominant_speaker_ratio": 0.8657,
            "speaker_count": 2,
        }

        with (
            patch.object(config, "ASR_AUTO_RETRY_BAD_DIARIZATION", True),
            patch.object(config, "ASR_FALLBACK_SPEAKER_NUM", 6),
            patch.object(config, "ASR_RETRY_MIN_AUDIO_SECONDS", 300.0),
            patch.object(config, "ASR_RETRY_MAX_SEGMENT_SECONDS", 180.0),
            patch.object(config, "ASR_RETRY_DOMINANT_SPEAKER_RATIO", 0.80),
        ):
            should_retry = FunASRService.should_retry_diarization(
                quality,
                requested_speakers=None,
                audio_duration=2114.72,
            )

        self.assertTrue(should_retry)

    def test_requested_speaker_count_disables_auto_retry(self) -> None:
        quality = {
            "warnings": ["segment_too_long", "dominant_speaker_too_large"],
            "max_segment_seconds": 353.28,
            "dominant_speaker_ratio": 0.8657,
            "speaker_count": 2,
        }

        self.assertFalse(
            FunASRService.should_retry_diarization(
                quality,
                requested_speakers=6,
                audio_duration=2114.72,
            )
        )

    def test_rescue_candidates_prioritize_suspect_terms(self) -> None:
        segments = [
            {"speaker": "0", "start_time": 0.0, "end_time": 2.0, "text": "正常文本"},
            {"speaker": "0", "start_time": 3.0, "end_time": 5.0, "text": "非国产机的是候可以的"},
        ]

        candidates = FunASRService._rescue_candidates(segments)

        self.assertEqual(1, candidates[0][0])
        self.assertIn("suspect_terms", candidates[0][1])

    def test_rescue_replacement_accepts_reduced_bad_terms(self) -> None:
        self.assertTrue(
            FunASRService._accept_rescue_text(
                "非国产机的是候可以的",
                "非国产机的时候可以的",
                2.0,
            )
        )

    def test_rescue_replacement_rejects_digit_changes(self) -> None:
        self.assertFalse(
            FunASRService._accept_rescue_text(
                "每天调用50次",
                "每天调用80次",
                2.0,
            )
        )

    def test_rescue_replacement_rejects_padding_drift(self) -> None:
        self.assertFalse(
            FunASRService._accept_rescue_text(
                "主子分号了吧。",
                "朱朱子芬号上吧，然后后面又多识别了一大段。",
                1.2,
            )
        )

    def test_rescue_replacement_rejects_negation_changes(self) -> None:
        self.assertFalse(
            FunASRService._accept_rescue_text(
                "否则谁都可以调一个问题",
                "否则谁都不可以调一个问题",
                2.0,
            )
        )

    def test_rescue_replacement_rejects_remaining_suspect_terms(self) -> None:
        self.assertFalse(
            FunASRService._accept_rescue_text(
                "他要我天天找我要那个真能体压衣啊",
                "他要我天天找我要那个智能体压计啊",
                4.0,
            )
        )

    def test_contextual_fallback_normalizes_domain_terms(self) -> None:
        text = "随身办的智智能体接入下量库，孙小猪会讲骂死和pass。"

        fixed = TranscriptPostProcessor._contextual_fallback(text, "")

        self.assertIn("随申办", fixed)
        self.assertIn("智能体", fixed)
        self.assertIn("向量库", fixed)
        self.assertIn("申小助", fixed)
        self.assertIn("MaaS", fixed)
        self.assertIn("PaaS", fixed)

    def test_deepseek_provider_uses_json_object_response(self) -> None:
        with (
            patch.object(config, "POSTPROCESS_PROVIDER", "deepseek"),
            patch.object(config, "DEEPSEEK_API_KEY", "test-key"),
        ):
            processor = TranscriptPostProcessor(use_worker_pool=False)
            with patch.object(
                processor,
                "_call_deepseek",
                return_value='{"items":[{"id":0,"text":"hello world"}]}',
            ) as deepseek_call:
                result = processor._correct_batch([{"text": "hello word"}])

        self.assertEqual(["hello world"], result)
        messages = deepseek_call.call_args.args[0]
        self.assertIn('"items"', messages[0]["content"])

    def test_deepseek_provider_requires_api_key(self) -> None:
        with (
            patch.object(config, "POSTPROCESS_PROVIDER", "deepseek"),
            patch.object(config, "DEEPSEEK_API_KEY", ""),
        ):
            processor = TranscriptPostProcessor(use_worker_pool=False)
            self.assertFalse(processor._ensure_model())

    def test_deepseek_batch_prompt_includes_surrounding_context(self) -> None:
        with patch.object(config, "DEEPSEEK_CONTEXT_CHARS", 4):
            context = TranscriptPostProcessor._batch_context("abcdHELLOefgh", "HELLO")
            messages = TranscriptPostProcessor._deepseek_messages([{"id": 0, "text": "HELLO"}], context)

        self.assertEqual("abcdHELLOefgh", context)
        self.assertIn("前后文", messages[1]["content"])
        self.assertIn('"items"', messages[1]["content"])

    def test_text_normalizer_repairs_mojibake_response_text(self) -> None:
        text = "灏变細鏈夋潈闄愩€傛潈闄愯繖鍧楀憿"

        fixed = repair_mojibake(text)

        self.assertIn("就会有权限", fixed)
        self.assertIn("权限这块呢", fixed)

    def test_text_normalizer_normalizes_latest_domain_errors(self) -> None:
        text = (
            "系动办公和协动办公里统一组织架构数要接入生小猪，"
            "公共公共知识库可以通过接口申请 k，能力开给他以后就能做搅合，"
            "否则每次掉接口要我 KK 是不是，还要控制几点直接钓鱼。"
        )

        fixed = normalize_asr_text(text)

        self.assertIn("协同办公", fixed)
        self.assertIn("统一组织架构树", fixed)
        self.assertIn("申小助", fixed)
        self.assertIn("共性知识库", fixed)
        self.assertIn("申请 key", fixed)
        self.assertIn("做校核", fixed)
        self.assertIn("调接口", fixed)
        self.assertIn("key 是不是", fixed)
        self.assertIn("直接调用", fixed)

    def test_text_normalizer_repairs_meeting_specific_suishenban_phrases(self) -> None:
        text = "至少随身带你们一部，不要现在随申办你面一个，用户体系不就是随申办的可吗？都是通过随申办的工。"

        fixed = normalize_asr_text(text)

        self.assertIn("随申办里面有一部分", fixed)
        self.assertIn("随申办里面有", fixed)
        self.assertIn("随申办的卡吗", fixed)
        self.assertIn("通过随申办登录", fixed)

    def test_text_normalizer_repairs_llm_meeting_specific_regressions(self) -> None:
        text = "主子分好了吧。主子分号了吧。这个只是主子的。但是分中心除外，分中心范围是配齐的。掉一个问题。整个车试。真能体。真能家。"

        fixed = normalize_asr_text(text)

        self.assertIn("主旨分好了", fixed)
        self.assertIn("只是主旨的", fixed)
        self.assertIn("四分中心除外，四分中心范围", fixed)
        self.assertIn("调一个问题", fixed)
        self.assertIn("整个测试", fixed)
        self.assertIn("智能体", fixed)


if __name__ == "__main__":
    unittest.main()
