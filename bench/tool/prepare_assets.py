#!/usr/bin/env python3
"""Extract video frames and copy shared bench assets into every bench app.

Run from anywhere: python3 bench/tool/prepare_assets.py
Requires opencv-python for the frame extraction and fps probe.
"""

import json
import shutil
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
ASSETS = BENCH / "assets"
FRAMES = ASSETS / "frames"
VIDEO = ASSETS / "bench_face_10s.mp4"
APPS = ["mine", "mlkit", "fdt"]
SHARED = ["portrait.jpg"]
JPEG_QUALITY = 90


def extract_frames() -> None:
    if FRAMES.is_dir() and any(FRAMES.glob("frame_*.jpg")):
        print(f"frames already extracted in {FRAMES}, skipping")
        return
    import cv2

    FRAMES.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(
            str(FRAMES / f"frame_{n:03d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        n += 1
    cap.release()
    print(f"extracted {n} frames to {FRAMES}")


def write_meta() -> None:
    """Record the source video fps so tests use real frame timestamps."""
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise SystemExit(f"cannot read fps from {VIDEO}")
    meta = {"fps": round(fps, 3), "frames": len(list(FRAMES.glob("frame_*.jpg")))}
    (FRAMES / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"wrote {FRAMES / 'meta.json'}: {meta}")


def copy_into_apps() -> None:
    for app in APPS:
        dst = BENCH / app / "assets"
        dst.mkdir(exist_ok=True)
        for name in SHARED:
            shutil.copy2(ASSETS / name, dst / name)
        dst_frames = dst / "frames"
        if dst_frames.is_dir():
            shutil.rmtree(dst_frames)
        shutil.copytree(FRAMES, dst_frames)
        print(f"copied assets into {dst}")


if __name__ == "__main__":
    extract_frames()
    write_meta()
    copy_into_apps()
