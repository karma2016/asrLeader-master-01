import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))).resolve()


def _bool_env(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


RESOLVE_LOCAL_ASR_MODELS = _bool_env("ASR_RESOLVE_LOCAL_MODELS", True)

ASR_MODEL_ALIAS = os.getenv("ASR_MODEL_ALIAS", "paraformer-zh")
VAD_MODEL_ALIAS = os.getenv("VAD_MODEL_ALIAS", "fsmn-vad")
PUNC_MODEL_ALIAS = os.getenv("PUNC_MODEL_ALIAS", "ct-punc")
SPK_MODEL_ALIAS = os.getenv("SPK_MODEL_ALIAS", "cam++")


def _resolve_model(env_key: str, local_name: str) -> str:
    value = os.getenv(env_key)
    if value:
        return value
    if RESOLVE_LOCAL_ASR_MODELS:
        local_path = MODELS_DIR / local_name
        if local_path.is_dir():
            return str(local_path)
    return local_name


ASR_MODEL = _resolve_model("ASR_MODEL", ASR_MODEL_ALIAS)
VAD_MODEL = _resolve_model("VAD_MODEL", VAD_MODEL_ALIAS)
PUNC_MODEL = _resolve_model("PUNC_MODEL", PUNC_MODEL_ALIAS)
SPK_MODEL = _resolve_model("SPK_MODEL", SPK_MODEL_ALIAS)

HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "8000"))
DEVICE = os.getenv("ASR_DEVICE", "auto")
BATCH_SIZE_S = int(os.getenv("BATCH_SIZE_S", "120"))
VAD_MAX_SINGLE_SEGMENT_TIME_MS = int(os.getenv("VAD_MAX_SINGLE_SEGMENT_TIME_MS", "30000"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(1024 * 1024 * 1024)))
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}
ALLOWED_AUDIO_FORMATS = ["WAV", "MP3", "M4A", "AAC"]
DEFAULT_ASR_HOTWORDS = " ".join(
    [
        "随申办",
        "申小助",
        "MaaS",
        "PaaS",
        "OA",
        "ETL",
        "WPS",
        "共性能力",
        "共性智能体",
        "共性知识库",
        "共享空间",
        "电子公文库",
        "公文校核",
        "向量库",
        "向量化",
        "一体化协同",
        "协同办公",
        "委办局",
        "大数据中心",
        "数据局",
        "政务公开",
        "知识库",
        "数据库",
        "统一用户",
        "用户体系",
        "组织架构",
        "组织架构树",
        "子树",
        "协同办公",
        "测试环境",
        "生产环境",
        "标杆场景",
        "智能化场景",
        "项目空间",
        "委办局管理员",
        "分中心",
        "看板",
        "下钻",
        "接口",
        "key",
        "API",
        "调用次数",
        "开通系统",
    ]
)
ASR_HOTWORDS = os.getenv("ASR_HOTWORDS") or DEFAULT_ASR_HOTWORDS
ASR_QUALITY_MAX_SEGMENT_SECONDS = float(os.getenv("ASR_QUALITY_MAX_SEGMENT_SECONDS", "120"))
ASR_QUALITY_DOMINANT_SPEAKER_RATIO = float(os.getenv("ASR_QUALITY_DOMINANT_SPEAKER_RATIO", "0.70"))
ASR_QUALITY_MIN_EXPECTED_SPEAKERS = int(os.getenv("ASR_QUALITY_MIN_EXPECTED_SPEAKERS", "5"))
ASR_QUALITY_MIN_SEGMENTS_PER_HOUR = float(os.getenv("ASR_QUALITY_MIN_SEGMENTS_PER_HOUR", "60"))
ASR_AUTO_RETRY_BAD_DIARIZATION = _bool_env("ASR_AUTO_RETRY_BAD_DIARIZATION", True)
ASR_FALLBACK_SPEAKER_NUM = int(os.getenv("ASR_FALLBACK_SPEAKER_NUM", "6"))
ASR_RETRY_MIN_AUDIO_SECONDS = float(os.getenv("ASR_RETRY_MIN_AUDIO_SECONDS", "300"))
ASR_RETRY_MAX_SEGMENT_SECONDS = float(os.getenv("ASR_RETRY_MAX_SEGMENT_SECONDS", "180"))
ASR_RETRY_DOMINANT_SPEAKER_RATIO = float(os.getenv("ASR_RETRY_DOMINANT_SPEAKER_RATIO", "0.80"))
ASR_RESCUE_ENABLED = _bool_env("ASR_RESCUE_ENABLED", True)
ASR_RESCUE_MAX_SEGMENTS = int(os.getenv("ASR_RESCUE_MAX_SEGMENTS", "24"))
ASR_RESCUE_MAX_TOTAL_AUDIO_SECONDS = float(os.getenv("ASR_RESCUE_MAX_TOTAL_AUDIO_SECONDS", "180"))
ASR_RESCUE_MIN_SEGMENT_SECONDS = float(os.getenv("ASR_RESCUE_MIN_SEGMENT_SECONDS", "0.6"))
ASR_RESCUE_MAX_SEGMENT_SECONDS = float(os.getenv("ASR_RESCUE_MAX_SEGMENT_SECONDS", "60"))
ASR_RESCUE_LONG_SEGMENT_SECONDS = float(os.getenv("ASR_RESCUE_LONG_SEGMENT_SECONDS", "10"))
ASR_RESCUE_MIN_TEXT_CHARS_PER_SECOND = float(os.getenv("ASR_RESCUE_MIN_TEXT_CHARS_PER_SECOND", "2.0"))
ASR_RESCUE_PADDING_MS = int(os.getenv("ASR_RESCUE_PADDING_MS", "180"))
ASR_RESCUE_BATCH_SIZE_S = int(os.getenv("ASR_RESCUE_BATCH_SIZE_S", "30"))
ASR_RESCUE_PRESET_SPEAKER_NUM = int(os.getenv("ASR_RESCUE_PRESET_SPEAKER_NUM", "1"))
ASR_RESCUE_MIN_SIMILARITY = float(os.getenv("ASR_RESCUE_MIN_SIMILARITY", "0.62"))
ASR_RESCUE_NOISE_MIN_SIMILARITY = float(os.getenv("ASR_RESCUE_NOISE_MIN_SIMILARITY", "0.72"))
ASR_RESCUE_AUDIO_FILTER = os.getenv("ASR_RESCUE_AUDIO_FILTER", "").strip()
ASR_RESCUE_PROVIDER = os.getenv("ASR_RESCUE_PROVIDER", "sensevoice").strip().lower() or "sensevoice"
ASR_RESCUE_MODEL_ALIAS = os.getenv("ASR_RESCUE_MODEL_ALIAS", "iic/SenseVoiceSmall")
ASR_RESCUE_MODEL = os.getenv("ASR_RESCUE_MODEL", "").strip()
if not ASR_RESCUE_MODEL:
    local_rescue_model = MODELS_DIR / "SenseVoiceSmall"
    ASR_RESCUE_MODEL = str(local_rescue_model) if local_rescue_model.is_dir() else ASR_RESCUE_MODEL_ALIAS
ASR_QWEN_RESCUE_MODEL_ALIAS = os.getenv("ASR_QWEN_RESCUE_MODEL_ALIAS", "Qwen/Qwen3-ASR-0.6B")
ASR_QWEN_RESCUE_MODEL = os.getenv("ASR_QWEN_RESCUE_MODEL", "").strip()
if not ASR_QWEN_RESCUE_MODEL:
    local_qwen_model = MODELS_DIR / "Qwen3-ASR-0.6B"
    data_qwen_model = DATA_DIR / "qwen3-asr-0.6b"
    if local_qwen_model.is_dir():
        ASR_QWEN_RESCUE_MODEL = str(local_qwen_model)
    elif data_qwen_model.is_dir():
        ASR_QWEN_RESCUE_MODEL = str(data_qwen_model)
    else:
        ASR_QWEN_RESCUE_MODEL = ASR_QWEN_RESCUE_MODEL_ALIAS
ASR_QWEN_RESCUE_RUNTIME_PATH = os.getenv(
    "ASR_QWEN_RESCUE_RUNTIME_PATH",
    str(DATA_DIR / "qwen_asr_runtime"),
).strip()
ASR_QWEN_RESCUE_LANGUAGE = os.getenv("ASR_QWEN_RESCUE_LANGUAGE", "Chinese").strip() or "Chinese"
ASR_QWEN_RESCUE_DTYPE = os.getenv("ASR_QWEN_RESCUE_DTYPE", "bfloat16").strip().lower() or "bfloat16"
ASR_QWEN_RESCUE_DEVICE_MAP = os.getenv("ASR_QWEN_RESCUE_DEVICE_MAP", "").strip()
ASR_QWEN_RESCUE_MAX_NEW_TOKENS = int(os.getenv("ASR_QWEN_RESCUE_MAX_NEW_TOKENS", "768"))
ASR_QWEN_RESCUE_BATCH_SIZE = max(1, int(os.getenv("ASR_QWEN_RESCUE_BATCH_SIZE", "1")))
ASR_QWEN_RESCUE_MIN_SEGMENT_SECONDS = float(os.getenv("ASR_QWEN_RESCUE_MIN_SEGMENT_SECONDS", "8.0"))
ASR_QWEN_RESCUE_ALLOW_DIRECT_REPLACE = _bool_env("ASR_QWEN_RESCUE_ALLOW_DIRECT_REPLACE", False)
ASR_QWEN_RESCUE_FORBIDDEN_TERMS = tuple(
    item.strip()
    for item in os.getenv("ASR_QWEN_RESCUE_FORBIDDEN_TERMS", "").split(",")
    if item.strip()
)
ASR_QWEN_RESCUE_HINT_MIN_SIMILARITY = float(os.getenv("ASR_QWEN_RESCUE_HINT_MIN_SIMILARITY", "0.72"))
ASR_QWEN_RESCUE_HINT_MIN_LENGTH_RATIO = float(os.getenv("ASR_QWEN_RESCUE_HINT_MIN_LENGTH_RATIO", "0.80"))
ASR_QWEN_RESCUE_HINT_MAX_LENGTH_RATIO = float(os.getenv("ASR_QWEN_RESCUE_HINT_MAX_LENGTH_RATIO", "1.25"))
ASR_QWEN_RESCUE_REASON_FILTER = {
    item.strip()
    for item in os.getenv(
        "ASR_QWEN_RESCUE_REASON_FILTER",
        "suspect_terms,noise_terms,low_text_density,long_segment",
    ).split(",")
    if item.strip()
}
DEFAULT_QWEN_RESCUE_CONTEXT = (
    "会议领域词：随申办、申小助、协同办公、组织架构树、共性知识库、共性能力、"
    "共性智能体、智能体、向量库、向量化、接口、调用、key、AKSK、MaaS、PaaS、"
    "OA、WPS、公文校核、委办局、开发商、厂商、项目空间、智能化场景、tab页、平铺。"
    "请按音频逐字转写，不要总结，不要润色。"
)
ASR_QWEN_RESCUE_CONTEXT = os.getenv("ASR_QWEN_RESCUE_CONTEXT", "").strip() or DEFAULT_QWEN_RESCUE_CONTEXT
ASR_RESCUE_LANGUAGE = os.getenv("ASR_RESCUE_LANGUAGE", "zh").strip() or "zh"
ASR_RESCUE_USE_ITN = _bool_env("ASR_RESCUE_USE_ITN", True)
ASR_RESCUE_MERGE_VAD = _bool_env("ASR_RESCUE_MERGE_VAD", True)
ASR_RESCUE_MERGE_LENGTH_S = int(os.getenv("ASR_RESCUE_MERGE_LENGTH_S", "15"))
ASR_RESCUE_TRUST_REMOTE_CODE = _bool_env("ASR_RESCUE_TRUST_REMOTE_CODE", True)
ASR_RESCUE_DISABLE_UPDATE = _bool_env("ASR_RESCUE_DISABLE_UPDATE", False)
ASR_RESCUE_AUDIT_SAMPLES = int(os.getenv("ASR_RESCUE_AUDIT_SAMPLES", "8"))
LEADER_DB_PATH = Path(os.getenv("LEADER_DB_PATH", str(DATA_DIR / "leaders.json")))
LEADER_THRESHOLD = float(os.getenv("LEADER_THRESHOLD", "0.45"))
LEADER_SPEAKER_THRESHOLD = float(os.getenv("LEADER_SPEAKER_THRESHOLD", "0.35"))
LEADER_SEGMENT_THRESHOLD = float(os.getenv("LEADER_SEGMENT_THRESHOLD", "0.50"))
LEADER_SCORE_MARGIN = float(os.getenv("LEADER_SCORE_MARGIN", "0.03"))
LEADER_MIN_SEGMENT_SECONDS = float(os.getenv("LEADER_MIN_SEGMENT_SECONDS", "1.0"))
SEGMENT_PADDING_MS = int(os.getenv("SEGMENT_PADDING_MS", "250"))
SPEAKER_AUTO_MERGE_THRESHOLD = float(os.getenv("SPEAKER_AUTO_MERGE_THRESHOLD", "0.50"))
SPEAKER_AUTO_MERGE_MARGIN = float(os.getenv("SPEAKER_AUTO_MERGE_MARGIN", "0.10"))
SPEAKER_AUTO_MERGE_MAX_SECONDS = float(os.getenv("SPEAKER_AUTO_MERGE_MAX_SECONDS", "30.0"))
SPEAKER_AUTO_MERGE_MAX_RATIO = float(os.getenv("SPEAKER_AUTO_MERGE_MAX_RATIO", "0.08"))
SPEAKER_JITTER_MAX_SECONDS = float(os.getenv("SPEAKER_JITTER_MAX_SECONDS", "3.0"))
SPEAKER_JITTER_MAX_GAP_SECONDS = float(os.getenv("SPEAKER_JITTER_MAX_GAP_SECONDS", "0.8"))
SPEAKER_JITTER_MIN_SIMILARITY = float(os.getenv("SPEAKER_JITTER_MIN_SIMILARITY", "0.45"))
SPEAKER_JITTER_MARGIN = float(os.getenv("SPEAKER_JITTER_MARGIN", "0.08"))
SEGMENT_MERGE_MAX_GAP_SECONDS = float(os.getenv("SEGMENT_MERGE_MAX_GAP_SECONDS", "2.0"))
LEADER_RELATIVE_MIN_SCORE = float(os.getenv("LEADER_RELATIVE_MIN_SCORE", "0.12"))
LEADER_RELATIVE_MARGIN = float(os.getenv("LEADER_RELATIVE_MARGIN", "0.10"))
LEADER_RELATIVE_SUPPORT_SEGMENTS = int(os.getenv("LEADER_RELATIVE_SUPPORT_SEGMENTS", "3"))
LEADER_SHORT_SUPPORT_SEGMENTS = int(os.getenv("LEADER_SHORT_SUPPORT_SEGMENTS", "2"))
LEADER_STRONG_SUPPORT_SEGMENTS = int(os.getenv("LEADER_STRONG_SUPPORT_SEGMENTS", "6"))
LEADER_STRONG_SUPPORT_MIN_SCORE = float(os.getenv("LEADER_STRONG_SUPPORT_MIN_SCORE", "0.20"))
LEADER_STRONG_SUPPORT_MARGIN = float(os.getenv("LEADER_STRONG_SUPPORT_MARGIN", "0.04"))
LEADER_MERGE_MATCHED_SPEAKERS = _bool_env("LEADER_MERGE_MATCHED_SPEAKERS", True)
LEADER_ENROLLMENT_MIN_SPEECH_SECONDS = float(os.getenv("LEADER_ENROLLMENT_MIN_SPEECH_SECONDS", "5.0"))
LEADER_ENROLLMENT_MIN_SIMILARITY = float(os.getenv("LEADER_ENROLLMENT_MIN_SIMILARITY", "0.60"))
LEADER_ENROLLMENT_SPEECH_DBFS = float(os.getenv("LEADER_ENROLLMENT_SPEECH_DBFS", "-50"))

POSTPROCESS_ENABLED = _bool_env("POSTPROCESS_ENABLED", True)
POSTPROCESS_PRELOAD = _bool_env("POSTPROCESS_PRELOAD", False)
POSTPROCESS_MODEL = os.getenv("POSTPROCESS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
POSTPROCESS_MODEL_DIR = os.getenv("POSTPROCESS_MODEL_DIR", "")
POSTPROCESS_DEVICE = os.getenv("POSTPROCESS_DEVICE", os.getenv("ASR_DEVICE", "cpu"))
POSTPROCESS_MAX_CHARS = int(os.getenv("POSTPROCESS_MAX_CHARS", "1200"))
POSTPROCESS_MAX_NEW_TOKENS = int(os.getenv("POSTPROCESS_MAX_NEW_TOKENS", "512"))
POSTPROCESS_WORKERS = max(1, int(os.getenv("POSTPROCESS_WORKERS", "1")))
POSTPROCESS_PROVIDER = os.getenv("POSTPROCESS_PROVIDER", "local").strip().lower() or "local"
POSTPROCESS_MIN_SIMILARITY = float(os.getenv("POSTPROCESS_MIN_SIMILARITY", "0.45"))
POSTPROCESS_MAX_LENGTH_RATIO = float(os.getenv("POSTPROCESS_MAX_LENGTH_RATIO", "1.80"))
POSTPROCESS_MAX_LENGTH_EXTRA = int(os.getenv("POSTPROCESS_MAX_LENGTH_EXTRA", "40"))
POSTPROCESS_SYNC_MAX_CHARS = int(os.getenv("POSTPROCESS_SYNC_MAX_CHARS", "1200"))
POSTPROCESS_SYNC_TIMEOUT_SECONDS = int(os.getenv("POSTPROCESS_SYNC_TIMEOUT_SECONDS", "120"))
CORRECTION_TASK_WORKERS = max(1, int(os.getenv("CORRECTION_TASK_WORKERS", "1")))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"
DEEPSEEK_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", str(POSTPROCESS_MAX_NEW_TOKENS)))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0"))
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
DEEPSEEK_CONTEXT_CHARS = int(os.getenv("DEEPSEEK_CONTEXT_CHARS", "400"))
if DEEPSEEK_THINKING not in {"enabled", "disabled"}:
    DEEPSEEK_THINKING = "disabled"
