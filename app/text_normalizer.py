from __future__ import annotations

import re


MOJIBAKE_MARKERS = (
    "锛",
    "銆",
    "涓",
    "鐨",
    "杩",
    "浣",
    "瀛",
    "鏄",
    "鍚",
    "闅",
    "鑳",
    "淇",
    "寮",
    "紝",
    "噺",
    "搴",
    "灏",
    "鏈",
    "潈",
    "闄",
    "繖",
    "憿",
)


def repair_mojibake(text: str) -> str:
    if not text or _mojibake_score(text) < 2:
        return text
    try:
        repaired = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        try:
            repaired = text.encode("gb18030", errors="ignore").decode("utf-8", errors="ignore")
        except UnicodeError:
            return text
    return repaired if _mojibake_score(repaired) < _mojibake_score(text) else text


def normalize_asr_text(text: str, full_text: str | None = None) -> str:
    del full_text
    if not text:
        return text
    text = repair_mojibake(text)
    text = _replace_domain_terms(text)
    text = _normalize_key_terms(text)
    return text


def _replace_domain_terms(text: str) -> str:
    replacements = {
        "随身办": "随申办",
        "随身带你们一部": "随申办里面有一部分",
        "随身带你面一个": "随申办里面有",
        "随申办你们一部": "随申办里面有一部分",
        "随申办你面一个": "随申办里面有",
        "随申办的可吗": "随申办的卡吗",
        "通过随申办的工": "通过随申办登录",
        "系动办公": "协同办公",
        "协动办公": "协同办公",
        "全协同办公": "协同办公",
        "组织架构数": "组织架构树",
        "统一组织架构数": "统一组织架构树",
        "智智能体": "智能体",
        "自能体": "智能体",
        "职能体": "智能体",
        "真能体": "智能体",
        "真能家": "智能体",
        "主子分好了": "主旨分好了",
        "主子分号了": "主旨分好了",
        "主子分号": "主旨分好",
        "主子的": "主旨的",
        "只是主子的": "只是主旨的",
        "共信智能体": "共性智能体",
        "公性智能体": "共性智能体",
        "个性智能体": "共性智能体",
        "共信知识库": "共性知识库",
        "公应知识库": "共性知识库",
        "公共公共知识库": "共性知识库",
        "公共知识库": "共性知识库",
        "供给支出库": "共性知识库",
        "共给知识库": "共性知识库",
        "下量库": "向量库",
        "下量图": "向量库",
        "相量库": "向量库",
        "像量库": "向量库",
        "下量化": "向量化",
        "相量化": "向量化",
        "像量化": "向量化",
        "片面化": "向量化",
        "教核": "校核",
        "搅核": "校核",
        "公文教核": "公文校核",
        "公文搅核": "公文校核",
        "做搅合": "做校核",
        "能力去做搅合": "能力去做校核",
        "的是候": "的时候",
        "空制": "控制",
        "知士": "知识",
        "虚洗": "续写",
        "酷写": "扩写",
        "主界界面": "主界面",
        "口口文档": "接口文档",
        "掉一个问题": "调一个问题",
        "整个车试": "整个测试",
        "肾小柱": "申小助",
        "生小猪": "申小助",
        "孙小猪": "申小助",
        "声小注": "申小助",
        "深小柱": "申小助",
        "称小猪": "申小助",
        "生产度里面": "生产库里面",
        "开动器": "开通 key",
        "开通器": "开通 key",
        "掉接口": "调接口",
        "直接钓鱼": "直接调用",
        "能钓多少次": "能调用多少次",
        "钓多少次": "调用多少次",
        "调动的时候": "调用的时候",
        "但是分中心除外，分中心范围": "但是四分中心除外，四分中心范围",
        "骂死": "MaaS",
        "马斯": "MaaS",
        "妈是": "MaaS",
        "OOA": "OA",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"(?<![A-Za-z])(?:mast|mass)(?![A-Za-z])", "MaaS", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![A-Za-z])pass(?![A-Za-z])", "PaaS", text, flags=re.IGNORECASE)
    text = re.sub(r"共性知识或共性[，,、]?\s*智能体", "共性知识库和共性智能体", text)
    return text


def _normalize_key_terms(text: str) -> str:
    text = re.sub(r"申请\s*[kK](?=([\s的了，,。]|$))", "申请 key", text)
    text = re.sub(
        r"开(了|通)?\s*[kK](?=([\s的时侯候，,。]|$))",
        lambda match: f"开{match.group(1) or ''} key",
        text,
    )
    text = re.sub(r"有+\s*[kK]\s*的话", "有 key 的话", text)
    text = re.sub(r"开了\s*key", "开了 key", text, flags=re.IGNORECASE)
    text = re.sub(r"开通\s*key", "开通 key", text, flags=re.IGNORECASE)
    text = re.sub(r"KK\s*是不是", "key 是不是", text)
    text = re.sub(r"每次调接口要我\s*[kK]+(?![A-Za-z])", "每次调接口要我 key", text)
    return text


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
