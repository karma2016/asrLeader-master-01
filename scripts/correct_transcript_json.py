from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
sys.path.insert(0, str(APP_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply transcript post-processing to an existing ASR JSON response.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--provider", choices=["local", "deepseek"], default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--deepseek-context-chars", type=int, default=None)
    return parser.parse_args()


def response_segments(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError("input JSON must be a segment list or an API response with a data list")


def main() -> None:
    args = parse_args()
    if args.provider:
        os.environ["POSTPROCESS_PROVIDER"] = args.provider
    if args.max_chars is not None:
        os.environ["POSTPROCESS_MAX_CHARS"] = str(args.max_chars)
    if args.deepseek_context_chars is not None:
        os.environ["DEEPSEEK_CONTEXT_CHARS"] = str(args.deepseek_context_chars)

    import config
    from post_processor import TranscriptPostProcessor

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = response_segments(payload)
    processor = TranscriptPostProcessor(use_worker_pool=False)
    corrected = processor.correct_segments([dict(segment) for segment in segments])

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        previous_summary = payload.get("postprocess_summary")
        payload = dict(payload)
        payload["data"] = corrected
        summary = {
            "provider": config.POSTPROCESS_PROVIDER,
            "changed_segments": sum(
                1 for before, after in zip(segments, corrected) if before.get("text") != after.get("text")
            ),
            "segment_count": len(corrected),
            "max_chars": config.POSTPROCESS_MAX_CHARS,
            "deepseek_context_chars": config.DEEPSEEK_CONTEXT_CHARS,
        }
        if previous_summary:
            summary["previous"] = previous_summary
        payload["postprocess_summary"] = summary
        output = payload
    else:
        output = corrected

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output.get("postprocess_summary", {"segment_count": len(corrected)}), ensure_ascii=False))


if __name__ == "__main__":
    main()
