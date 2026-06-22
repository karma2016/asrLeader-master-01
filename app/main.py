from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

import config
from leader_store import LeaderAlreadyExistsError, LeaderStore, cosine_similarity
from model_service import FunASRService
from post_processor import TranscriptPostProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
correction_executor = ThreadPoolExecutor(max_workers=config.CORRECTION_TASK_WORKERS)
correction_tasks: dict[str, dict[str, Any]] = {}
correction_tasks_lock = threading.Lock()
asr_service = FunASRService()
leader_store = LeaderStore(config.LEADER_DB_PATH)
post_processor = TranscriptPostProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, asr_service.load)
    if config.POSTPROCESS_ENABLED and config.POSTPROCESS_PRELOAD:
        await loop.run_in_executor(executor, post_processor.load)
    yield


app = FastAPI(
    title="FunASR Leader ASR",
    description="Speech recognition, speaker diarization, and leader voiceprint identification.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = schema.get("properties", {})
        for prop in properties.values():
            items = prop.get("items")
            if isinstance(items, dict) and items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def ok(data: Any = None) -> dict[str, Any]:
    return {"code": 0, "message": "success", "data": data}


def transcribe_ok(data: Any = None, **extra: Any) -> dict[str, Any]:
    response = {"code": 200, "message": "success", "data": data}
    response.update(extra)
    return response


def now_ts() -> float:
    return round(time.time(), 3)


def public_correction_task(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != "future"}


def get_correction_task(task_id: str) -> dict[str, Any] | None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is None:
            return None
        return public_correction_task(task)


def segment_text_chars(segments: list[dict[str, Any]]) -> int:
    return sum(len(str(segment.get("text", ""))) for segment in segments)


def run_correction_task(task_id: str, segments: list[dict[str, Any]]) -> None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is not None:
            task["status"] = "running"
            task["updated_at"] = now_ts()
    try:
        result = post_processor.correct_segments([dict(segment) for segment in segments])
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "completed"
                task["updated_at"] = now_ts()
                task["result"] = result
    except Exception as exc:
        logger.exception("ASR correction task failed: %s", task_id)
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "failed"
                task["updated_at"] = now_ts()
                task["error"] = str(exc)


def submit_correction_task(segments: list[dict[str, Any]], mode: str) -> tuple[str, Any]:
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "queued",
        "mode": mode,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "result": None,
        "error": None,
    }
    with correction_tasks_lock:
        correction_tasks[task_id] = task
    future = correction_executor.submit(run_correction_task, task_id, [dict(segment) for segment in segments])
    with correction_tasks_lock:
        correction_tasks[task_id]["future"] = future
    return task_id, future


def correction_pending_fields(task_id: str, mode: str) -> dict[str, Any]:
    return {
        "correction_status": "running",
        "correction_mode": mode,
        "correction_task_id": task_id,
        "poll_url": f"/corrections/{task_id}",
    }


def normalize_correction_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in {"", "auto"}:
        return "auto"
    if mode in {"sync", "async", "none"}:
        return mode
    raise HTTPException(status_code=400, detail="correction_mode must be auto, sync, async, or none")


def validate_similarity_threshold(value: float, field_name: str = "threshold") -> float:
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail=f"{field_name} must be between 0 and 1")
    return value


def normalize_leader_id(value: str) -> str:
    leader_id = value.strip()
    if not leader_id:
        raise HTTPException(status_code=400, detail="id不能为空")
    if len(leader_id) > 128:
        raise HTTPException(status_code=400, detail="id长度不能超过128个字符")
    if any(ord(char) < 32 for char in leader_id):
        raise HTTPException(status_code=400, detail="id不能包含控制字符")
    return leader_id


def effective_speech_seconds(path: str) -> float:
    chunk_ms = 20
    active_chunks = 0
    with wave.open(path, "rb") as audio:
        sample_rate = audio.getframerate()
        frames_per_chunk = max(1, sample_rate * chunk_ms // 1000)
        while data := audio.readframes(frames_per_chunk):
            samples = array("h")
            samples.frombytes(data)
            if not samples:
                continue
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            dbfs = 20 * math.log10(max(rms, 1.0) / 32768.0)
            if dbfs >= config.LEADER_ENROLLMENT_SPEECH_DBFS:
                active_chunks += 1
    return round(active_chunks * chunk_ms / 1000.0, 2)


def audio_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as audio:
        return round(audio.getnframes() / max(audio.getframerate(), 1), 2)


def validate_enrollment_samples(
    samples: list[tuple[list[float], str | None, float]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    short_samples = [
        {"source": source, "speech_seconds": speech_seconds}
        for _, source, speech_seconds in samples
        if speech_seconds < config.LEADER_ENROLLMENT_MIN_SPEECH_SECONDS
    ]
    if short_samples:
        issues.append(
            {
                "message": "声纹样本中的有效语音时长不足",
                "minimum_speech_seconds": config.LEADER_ENROLLMENT_MIN_SPEECH_SECONDS,
                "samples": short_samples,
            }
        )

    pair_scores: list[dict[str, Any]] = []
    best_by_sample = [-1.0] * len(samples)
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            score = cosine_similarity(samples[left][0], samples[right][0])
            best_by_sample[left] = max(best_by_sample[left], score)
            best_by_sample[right] = max(best_by_sample[right], score)
            pair_scores.append(
                {
                    "left": samples[left][1],
                    "right": samples[right][1],
                    "score": round(score, 5),
                }
            )

    minimum_best_score = min(best_by_sample)
    if minimum_best_score < config.LEADER_ENROLLMENT_MIN_SIMILARITY:
        issues.append(
            {
                "message": "上传的声纹样本相似度过低，可能不是同一个人，或录音设备、环境差异过大",
                "minimum_similarity": config.LEADER_ENROLLMENT_MIN_SIMILARITY,
                "minimum_best_score": round(minimum_best_score, 5),
                "pairs": pair_scores,
            }
        )
    if issues:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "声纹注册质量校验失败",
                "issues": issues,
            },
        )
    return {
        "minimum_best_score": round(minimum_best_score, 5),
        "pairs": pair_scores,
    }


async def save_upload(file: UploadFile, prefix: str) -> str:
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "不支持的音频格式",
                "file_name": filename or None,
                "allowed_formats": config.ALLOWED_AUDIO_FORMATS,
            },
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传的音频文件为空")
    if len(content) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"音频文件超过大小限制，最大允许 {config.MAX_FILE_SIZE} 字节",
        )
    upload_id = uuid.uuid4().hex
    source_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{upload_id}{ext}")
    normalized_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{upload_id}_16k.wav")
    with open(source_path, "wb") as f:
        f.write(content)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        normalized_path,
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if completed.stderr:
            logger.warning("Audio normalization warning for %s: %s", file.filename, completed.stderr.strip())
        return normalized_path
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="服务器未安装 ffmpeg，无法处理音频") from exc
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or "unsupported or damaged audio").strip()
        logger.warning("Audio normalization failed for %s: %s", file.filename, error)
        try:
            os.remove(normalized_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="音频文件无效、已损坏或内容与扩展名不一致") from exc
    finally:
        try:
            os.remove(source_path)
        except OSError:
            pass


def parse_embedding(raw: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="embedding must be valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("embedding")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="embedding must be a numeric array")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="embedding contains non-numeric values") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": asr_service.model is not None,
        "device": asr_service.device,
        "leader_count": len(leader_store.list()),
        "models_dir": str(config.MODELS_DIR),
        "postprocess_enabled": config.POSTPROCESS_ENABLED,
        "postprocess_provider": config.POSTPROCESS_PROVIDER if config.POSTPROCESS_ENABLED else None,
        "postprocess_model": config.POSTPROCESS_MODEL if config.POSTPROCESS_ENABLED else None,
        "postprocess_device": config.POSTPROCESS_DEVICE if config.POSTPROCESS_ENABLED else None,
        "postprocess_workers": config.POSTPROCESS_WORKERS if config.POSTPROCESS_ENABLED else 0,
        "postprocess_sync_max_chars": config.POSTPROCESS_SYNC_MAX_CHARS if config.POSTPROCESS_ENABLED else 0,
        "postprocess_sync_timeout_seconds": config.POSTPROCESS_SYNC_TIMEOUT_SECONDS if config.POSTPROCESS_ENABLED else 0,
        "postprocess_min_similarity": config.POSTPROCESS_MIN_SIMILARITY if config.POSTPROCESS_ENABLED else None,
        "postprocess_max_length_ratio": config.POSTPROCESS_MAX_LENGTH_RATIO if config.POSTPROCESS_ENABLED else None,
        "postprocess_max_length_extra": config.POSTPROCESS_MAX_LENGTH_EXTRA if config.POSTPROCESS_ENABLED else None,
        "deepseek_configured": bool(config.DEEPSEEK_API_KEY) if config.POSTPROCESS_ENABLED else False,
        "deepseek_model": config.DEEPSEEK_MODEL if config.POSTPROCESS_ENABLED and config.POSTPROCESS_PROVIDER == "deepseek" else None,
        "deepseek_context_chars": config.DEEPSEEK_CONTEXT_CHARS if config.POSTPROCESS_ENABLED and config.POSTPROCESS_PROVIDER == "deepseek" else 0,
        "correction_task_workers": config.CORRECTION_TASK_WORKERS if config.POSTPROCESS_ENABLED else 0,
        "batch_size_s": config.BATCH_SIZE_S,
        "vad_max_single_segment_time_ms": config.VAD_MAX_SINGLE_SEGMENT_TIME_MS,
        "default_hotwords_enabled": bool(config.ASR_HOTWORDS.strip()),
        "quality_thresholds": {
            "max_segment_seconds": config.ASR_QUALITY_MAX_SEGMENT_SECONDS,
            "dominant_speaker_ratio": config.ASR_QUALITY_DOMINANT_SPEAKER_RATIO,
            "min_expected_speakers": config.ASR_QUALITY_MIN_EXPECTED_SPEAKERS,
            "min_segments_per_hour": config.ASR_QUALITY_MIN_SEGMENTS_PER_HOUR,
        },
        "auto_retry_bad_diarization": config.ASR_AUTO_RETRY_BAD_DIARIZATION,
        "fallback_speaker_num": config.ASR_FALLBACK_SPEAKER_NUM,
        "rescue_asr": {
            "enabled": config.ASR_RESCUE_ENABLED,
            "max_segments": config.ASR_RESCUE_MAX_SEGMENTS,
            "max_total_audio_seconds": config.ASR_RESCUE_MAX_TOTAL_AUDIO_SECONDS,
            "max_segment_seconds": config.ASR_RESCUE_MAX_SEGMENT_SECONDS,
            "min_text_chars_per_second": config.ASR_RESCUE_MIN_TEXT_CHARS_PER_SECOND,
            "long_segment_seconds": config.ASR_RESCUE_LONG_SEGMENT_SECONDS,
            "padding_ms": config.ASR_RESCUE_PADDING_MS,
            "preset_speaker_num": config.ASR_RESCUE_PRESET_SPEAKER_NUM,
            "min_similarity": config.ASR_RESCUE_MIN_SIMILARITY,
            "noise_min_similarity": config.ASR_RESCUE_NOISE_MIN_SIMILARITY,
            "provider": config.ASR_RESCUE_PROVIDER,
            "model": config.ASR_RESCUE_MODEL if config.ASR_RESCUE_PROVIDER == "sensevoice" else None,
            "language": config.ASR_RESCUE_LANGUAGE,
            "audit_samples": config.ASR_RESCUE_AUDIT_SAMPLES,
        },
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    identify_leaders: bool = Form(default=True),
    leader_threshold: float = Form(default=config.LEADER_THRESHOLD),
    return_leader_scores: bool = Form(default=False),
    post_process: bool = Form(default=config.POSTPROCESS_ENABLED),
    correct_text: Optional[bool] = Form(default=None),
    correction_mode: str = Form(default="auto"),
):
    leader_threshold = validate_similarity_threshold(leader_threshold, "leader_threshold")
    path = await save_upload(file, "asr")
    try:
        loop = asyncio.get_event_loop()
        audio_duration = audio_duration_seconds(path)
        result = await loop.run_in_executor(
            executor,
            asr_service.transcribe,
            path,
            num_speakers,
            hotwords,
        )
        if identify_leaders:
            result = await loop.run_in_executor(
                executor,
                asr_service.annotate_leaders,
                path,
                result,
                leader_store,
                leader_threshold,
                return_leader_scores,
            )
        result = asr_service._merge_adjacent_segments(result)
        asr_quality = asr_service.transcript_quality(result, audio_duration)
        if asr_service.should_retry_diarization(asr_quality, num_speakers, audio_duration):
            initial_quality = asr_quality
            fallback_speakers = config.ASR_FALLBACK_SPEAKER_NUM
            logger.warning(
                "Retrying ASR with fallback speaker count %s after diarization warning: %s",
                fallback_speakers,
                initial_quality,
            )
            result = await loop.run_in_executor(
                executor,
                asr_service.transcribe,
                path,
                fallback_speakers,
                hotwords,
            )
            if identify_leaders:
                result = await loop.run_in_executor(
                    executor,
                    asr_service.annotate_leaders,
                    path,
                    result,
                    leader_store,
                    leader_threshold,
                    return_leader_scores,
                )
            result = asr_service._merge_adjacent_segments(result)
            asr_quality = asr_service.transcript_quality(result, audio_duration)
            asr_quality["retry"] = {
                "reason": "bad_diarization",
                "fallback_speaker_num": fallback_speakers,
                "initial_quality": initial_quality,
            }
        if config.ASR_RESCUE_ENABLED:
            result, rescue_summary = await loop.run_in_executor(
                executor,
                asr_service.rescue_segments,
                path,
                result,
                hotwords,
            )
            asr_quality["rescue"] = rescue_summary
        if asr_quality["status"] != "ok":
            logger.warning("ASR quality warnings: %s", asr_quality)
        should_correct_text = post_process if correct_text is None else correct_text
        mode = normalize_correction_mode(correction_mode)
        if mode == "none":
            should_correct_text = False
        if should_correct_text:
            if mode == "async" or (
                mode == "auto" and segment_text_chars(result) > config.POSTPROCESS_SYNC_MAX_CHARS
            ):
                task_id, _ = submit_correction_task(result, mode)
                return transcribe_ok(
                    result,
                    asr_quality=asr_quality,
                    **correction_pending_fields(task_id, mode),
                )

            task_id, future = submit_correction_task(result, mode)
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=config.POSTPROCESS_SYNC_TIMEOUT_SECONDS)
                task = get_correction_task(task_id)
                if task and task.get("status") == "completed":
                    return transcribe_ok(task.get("result", result), asr_quality=asr_quality)
                if task and task.get("status") == "failed":
                    logger.warning("ASR correction task failed, returning raw ASR: %s", task.get("error"))
                    return transcribe_ok(result, asr_quality=asr_quality)
            except asyncio.TimeoutError:
                return transcribe_ok(
                    result,
                    asr_quality=asr_quality,
                    **correction_pending_fields(task_id, "timeout"),
                )
        return transcribe_ok(result, asr_quality=asr_quality)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/corrections/{task_id}")
async def correction_result(task_id: str):
    task = get_correction_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="correction task not found")
    return ok(task)


@app.post("/leaders/enroll")
async def enroll_leader(
    leader_id: str = Form(..., alias="id", description="业务系统中的用户唯一ID。"),
    files: List[UploadFile] = File(..., description="请至少上传两个声纹音频文件。"),
):
    leader_id = normalize_leader_id(leader_id)
    if leader_store.exists(leader_id):
        raise HTTPException(status_code=409, detail="当前用户的声纹信息已存在")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="至少需要上传两个声纹音频文件")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        samples: list[tuple[list[float], str | None, float]] = []
        for file in files:
            path = await save_upload(file, "leader")
            paths.append(path)
            speech_seconds = effective_speech_seconds(path)
            embedding = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path)
            samples.append((embedding, file.filename, speech_seconds))
        quality = validate_enrollment_samples(samples)
        embeddings = [(embedding, source) for embedding, source, _ in samples]
        try:
            item = leader_store.create_leader(leader_id, embeddings)
        except LeaderAlreadyExistsError as exc:
            raise HTTPException(status_code=409, detail="当前用户的声纹信息已存在") from exc
        item["quality"] = {
            "samples": [
                {"source": source, "speech_seconds": speech_seconds}
                for _, source, speech_seconds in samples
            ],
            **quality,
        }
        return ok(item)
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/leaders")
async def list_leaders():
    return ok(leader_store.list())


@app.delete("/leaders/{leader_id}")
async def delete_leader(leader_id: str):
    deleted = leader_store.delete(leader_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="leader not found")
    return ok({"deleted": True, "leader_id": leader_id})


@app.post("/voiceprint/extract")
async def voiceprint_extract(file: UploadFile = File(...)):
    path = await save_upload(file, "voiceprint")
    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path)
        return ok({"embedding": embedding, "dimension": len(embedding)})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/voiceprint/verify")
async def voiceprint_verify(
    audio_a: UploadFile = File(...),
    audio_b: Optional[UploadFile] = File(default=None),
    embedding: Optional[str] = Form(default=None),
    threshold: float = Form(default=config.LEADER_THRESHOLD),
):
    threshold = validate_similarity_threshold(threshold)
    if audio_b is None and embedding is None:
        raise HTTPException(status_code=400, detail="provide audio_b or embedding")
    if audio_b is not None and embedding is not None:
        raise HTTPException(status_code=400, detail="audio_b and embedding cannot both be provided")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        path_a = await save_upload(audio_a, "voiceprint")
        paths.append(path_a)
        embedding_a = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path_a)

        if audio_b is not None:
            path_b = await save_upload(audio_b, "voiceprint")
            paths.append(path_b)
            embedding_b = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path_b)
        else:
            embedding_b = parse_embedding(embedding or "")

        score = cosine_similarity(embedding_a, embedding_b)
        return ok(
            {
                "score": round(score, 5),
                "threshold": threshold,
                "same_speaker": score >= threshold,
                "dimension": len(embedding_a),
            }
        )
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, workers=1)
