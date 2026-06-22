from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET


DOMAIN_TERMS = [
    "随申办",
    "协同办公",
    "组织架构树",
    "申小助",
    "共性知识库",
    "共性能力",
    "共性智能体",
    "公文校核",
    "向量库",
    "向量化",
    "智能体",
    "接口",
    "调用",
    "key",
    "OA",
    "MaaS",
    "PaaS",
    "WPS",
    "ETL",
]

LIKELY_BAD_TERMS = [
    "行动办公",
    "协动办公",
    "组织架构数",
    "生小猪",
    "孙小猪",
    "公共知识库",
    "公共公共知识库",
    "搅和",
    "下量库",
    "片面化",
    "开通器",
    "钓鱼",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a FunASR JSON response against a Xunfei transcript docx.")
    parser.add_argument("funasr_json", type=Path)
    parser.add_argument("xunfei_docx", type=Path)
    parser.add_argument("--label", default="funasr")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_funasr(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        raise ValueError("FunASR JSON must contain a segment list")
    text = "".join(str(segment.get("text", "")) for segment in segments)
    speakers = Counter(str(segment.get("speaker", "")) for segment in segments)
    return {
        "segments": segments,
        "text": text,
        "speaker_count": len([speaker for speaker in speakers if speaker]),
        "speaker_segments": dict(speakers),
        "asr_quality": payload.get("asr_quality") if isinstance(payload, dict) else None,
        "postprocess_summary": payload.get("postprocess_summary") if isinstance(payload, dict) else None,
    }


def load_docx_text(path: Path) -> dict[str, Any]:
    with ZipFile(path) as docx:
        xml = docx.read("word/document.xml")
    root = ET.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        if text:
            paragraphs.append(text)
    speakers = Counter(re.findall(r"说话人(\d+)", "\n".join(paragraphs)))
    body_paragraphs = paragraphs[1:] if paragraphs and re.search(r"\d+月\d+日", paragraphs[0]) else paragraphs
    text = "\n".join(body_paragraphs)
    text = re.sub(r"说话人\d+\s+\d{2}:\d{2}", "", text)
    return {
        "paragraphs": paragraphs,
        "text": text,
        "speaker_count": len(speakers),
        "speaker_segments": dict(speakers),
    }


def comparable(text: str) -> str:
    text = text.replace("ＯＡ", "OA").replace("ｋｅｙ", "key")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.lower()


def count_terms(text: str, terms: list[str]) -> dict[str, int]:
    lower = text.lower()
    counts: dict[str, int] = {}
    for term in terms:
        haystack = lower if re.search(r"[A-Za-z]", term) else text
        needle = term.lower() if re.search(r"[A-Za-z]", term) else term
        counts[term] = haystack.count(needle)
    return counts


def nonzero(values: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in values.items() if value}


def summary(label: str, funasr: dict[str, Any], xunfei: dict[str, Any]) -> dict[str, Any]:
    fun_text = funasr["text"]
    xf_text = xunfei["text"]
    fun_comp = comparable(fun_text)
    xf_comp = comparable(xf_text)
    good_terms = count_terms(fun_text, DOMAIN_TERMS)
    bad_terms = count_terms(fun_text, LIKELY_BAD_TERMS)
    xf_good_terms = count_terms(xf_text, DOMAIN_TERMS)
    xf_bad_terms = count_terms(xf_text, LIKELY_BAD_TERMS)
    return {
        "label": label,
        "funasr_chars": len(fun_text),
        "xunfei_chars": len(xf_text),
        "similarity_to_xunfei": round(SequenceMatcher(None, fun_comp, xf_comp).ratio(), 4),
        "funasr_segments": len(funasr["segments"]),
        "funasr_speaker_count": funasr["speaker_count"],
        "xunfei_speaker_count": xunfei["speaker_count"],
        "funasr_speaker_segments": funasr["speaker_segments"],
        "xunfei_speaker_segments": xunfei["speaker_segments"],
        "domain_hits": {
            "funasr": nonzero(good_terms),
            "xunfei": nonzero(xf_good_terms),
        },
        "likely_bad_terms": {
            "funasr": nonzero(bad_terms),
            "xunfei": nonzero(xf_bad_terms),
        },
        "asr_quality": funasr["asr_quality"],
        "postprocess_summary": funasr["postprocess_summary"],
    }


def main() -> None:
    args = parse_args()
    result = summary(args.label, load_funasr(args.funasr_json), load_docx_text(args.xunfei_docx))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
