#!/usr/bin/env python3
"""Convert ExDark JSON annotations to normalized grounding prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: float,
    image_height: float,
    scale: int = 1000,
) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width/height must be positive.")
    nx_min = round((x_min / image_width) * scale)
    ny_min = round((y_min / image_height) * scale)
    nx_max = round((x_max / image_width) * scale)
    ny_max = round((y_max / image_height) * scale)
    return (
        _clamp(nx_min, 0, scale),
        _clamp(ny_min, 0, scale),
        _clamp(nx_max, 0, scale),
        _clamp(ny_max, 0, scale),
    )


def format_grounding_prompt(detection: dict[str, Any], image_width: float, image_height: float) -> str:
    bbox = detection.get("bounding_box", {})
    x1, y1, x2, y2 = normalize_box(
        bbox.get("x_min", 0),
        bbox.get("y_min", 0),
        bbox.get("x_max", 0),
        bbox.get("y_max", 0),
        image_width=image_width,
        image_height=image_height,
    )
    label = str(detection.get("label", "unknown"))
    luminance_level = str(detection.get("luminance_level", "unknown"))
    confidence = float(detection.get("confidence", 0.0))
    reasoning = str(detection.get("reasoning", ""))
    return (
        "<|object_ref_start|>"
        f"{label}"
        "<|object_ref_end|>"
        "<|box_start|>"
        f"({x1},{y1}),({x2},{y2})"
        "<|box_end|>"
        f" luminance={luminance_level}; confidence={confidence:.4f}; reasoning={reasoning}"
    )


def _infer_dimension(data: dict[str, Any], explicit: int | None, keys: list[str]) -> int:
    if explicit is not None:
        return explicit
    for key in keys:
        value = data.get(key)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Missing required dimension. Provide one of {keys} in JSON or use --image-width/--image-height."
    )


def convert_annotations(input_path: Path, output_path: Path, image_width: int | None, image_height: int | None) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    width = _infer_dimension(payload, image_width, ["image_width", "width", "img_width"])
    height = _infer_dimension(payload, image_height, ["image_height", "height", "img_height"])

    detections = payload.get("detections", [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for detection in detections:
            record = {
                "prompt": format_grounding_prompt(detection, image_width=width, image_height=height),
                "label": detection.get("label", "unknown"),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(detections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to ExDark-style annotation JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file.")
    parser.add_argument("--image-width", type=int, help="Optional image width override.")
    parser.add_argument("--image-height", type=int, help="Optional image height override.")
    args = parser.parse_args()

    written = convert_annotations(
        input_path=args.input,
        output_path=args.output,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    print(f"Wrote {written} normalized detection prompts to {args.output}")


if __name__ == "__main__":
    main()
