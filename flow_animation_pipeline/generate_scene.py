#!/usr/bin/env python3
"""Generate a 24-frame stickman dumbbell-lift sequence with the Gemini API.

Every request sends two images:
  1. the reference image (style and structure anchor), sent with every frame
  2. the last accepted frame (positional anchor), so the lines stay put

Each result is checked against the previous frame for colour drift and broken
or shifted lines. A failed frame is discarded and retried, up to 3 times, with
stronger "match the reference" settings each time.

Note: the Gemini API has no numeric "guidance" or "similarity strength" setting
for image output. In this script, "strength" means a lower sampling temperature
plus stricter wording in the prompt. See STRENGTH_LEVELS below.

Usage:
    export GEMINI_API_KEY=...            # or GOOGLE_API_KEY
    pip install google-genai pillow numpy
    python3 generate_scene.py            # 24 frames into frames/gym_lift/
    python3 generate_scene.py --dry-run  # check setup and validator, no API calls
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "references" / "05_gym_simple.jpg"
DEFAULT_OUT = ROOT / "frames" / "gym_lift"
LOG_DIR = ROOT / "logs"
DEFAULT_MODEL = "gemini-2.5-flash-image"

# Retry ladder: attempt 0 is the normal try; attempts 1-3 are the retries.
STRENGTH_LEVELS = [
    {"temperature": 0.6, "wording": "Match the reference style and layout closely."},
    {"temperature": 0.4, "wording": "Match the reference style and layout very closely. "
                                    "Do not change any colours or line weights."},
    {"temperature": 0.25, "wording": "STRICT: copy the previous frame pixel-for-pixel except "
                                     "the lifting arm and dumbbell. Same colours, same line "
                                     "thickness, same positions."},
    {"temperature": 0.1, "wording": "MAXIMUM STRICTNESS: the output must be identical to the "
                                    "previous frame in every way except the lifting arm and "
                                    "dumbbell. Any change in colour, line weight, camera, "
                                    "background or other characters is an error."},
]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    max_color_drift: float = 0.18     # histogram L1 distance (0 = identical, 2 = disjoint)
    min_line_overlap: float = 0.88    # share of ink lines that line up with the previous frame
    max_ink_change: float = 0.30      # allowed relative change in amount of dark line pixels


@dataclass
class Check:
    ok: bool
    color_drift: float
    line_recall: float
    line_precision: float
    ink_change: float
    reasons: list


def _color_hist(img: np.ndarray, bins: int = 8) -> np.ndarray:
    q = (img // (256 // bins)).astype(np.int32)
    idx = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    h = np.bincount(idx.ravel(), minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def _ink_mask(img: np.ndarray, threshold: int = 70) -> np.ndarray:
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return luma < threshold


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def validate(candidate: Image.Image, previous: Image.Image, th: Thresholds) -> Check:
    """Compare a new frame with the previous accepted frame (or the reference)."""
    size = (512, int(512 * previous.height / previous.width))
    a = np.asarray(candidate.convert("RGB").resize(size, Image.LANCZOS))
    b = np.asarray(previous.convert("RGB").resize(size, Image.LANCZOS))

    color_drift = float(np.abs(_color_hist(a) - _color_hist(b)).sum())

    ink_a, ink_b = _ink_mask(a), _ink_mask(b)
    n_a, n_b = int(ink_a.sum()), int(ink_b.sum())
    ink_change = abs(n_a - n_b) / max(n_b, 1)
    # Lines may shift a few pixels; only the arm should really move.
    recall = float((ink_b & _dilate(ink_a, 3)).sum() / max(n_b, 1))
    precision = float((ink_a & _dilate(ink_b, 3)).sum() / max(n_a, 1))

    reasons = []
    if color_drift > th.max_color_drift:
        reasons.append(f"colour drift {color_drift:.3f} > {th.max_color_drift}")
    if min(recall, precision) < th.min_line_overlap:
        reasons.append(f"line overlap {min(recall, precision):.3f} < {th.min_line_overlap}")
    if ink_change > th.max_ink_change:
        reasons.append(f"ink change {ink_change:.3f} > {th.max_ink_change}")

    return Check(not reasons, round(color_drift, 4), round(recall, 4),
                 round(precision, 4), round(ink_change, 4), reasons)


def score(c: Check) -> float:
    """Lower is better; used to pick the best rejected attempt with --keep-going."""
    return c.color_drift + (1 - min(c.line_recall, c.line_precision)) + c.ink_change


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def pose_for(i: int, n: int) -> str:
    """Bicep curl: arm down -> fully up -> back down over n frames."""
    t = i / (n - 1) if n > 1 else 0
    lift = math.sin(math.pi * t)                  # 0 -> 1 -> 0
    elbow = round(170 - 130 * lift)               # 170 deg (straight) to 40 deg (curled)
    if lift < 0.05:
        stage = "arm hanging straight down at the side, dumbbell at hip height"
    elif lift > 0.95:
        stage = "arm fully curled, dumbbell up near the shoulder"
    elif t < 0.5:
        stage = "lifting the dumbbell upward"
    else:
        stage = "lowering the dumbbell back down"
    return (f"Frame {i + 1} of {n}. The stickman who is lifting the dumbbell: {stage}, "
            f"elbow angle about {elbow} degrees, upper arm stays still against the body.")


def build_prompt(i: int, n: int, has_previous: bool, level: int) -> str:
    images = ("Image 1 is the STYLE and STRUCTURE reference: copy its art style, colour palette, "
              "line thickness, paper texture, camera angle, background and every object and "
              "character exactly.")
    if has_previous:
        images += (" Image 2 is the PREVIOUS FRAME of this animation and is a hard positional "
                   "reference: keep every line, character and object exactly where it is in "
                   "Image 2.")
    return (
        f"{images}\n\n"
        f"Draw the next frame of a smooth 2D animation of the same scene. "
        f"Only the dumbbell-lifting arm and the dumbbell may move.\n"
        f"{pose_for(i, n)}\n\n"
        f"{STRENGTH_LEVELS[level]['wording']} Output a single image with the same aspect "
        f"ratio as the reference. No text, no borders, no captions."
    )


def to_part(img: Image.Image, types):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def generate(client, types, model: str, reference: Image.Image, previous: Image.Image | None,
             prompt: str, level: int, seed: int, aspect: str) -> Image.Image:
    contents = [to_part(reference, types)]
    if previous is not None:
        contents.append(to_part(previous, types))
    contents.append(prompt)

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            temperature=STRENGTH_LEVELS[level]["temperature"],
            seed=seed,
            image_config=types.ImageConfig(aspect_ratio=aspect),
        ),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if part.inline_data and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
    raise RuntimeError(f"API returned no image (finish reason: {reason})")


def nearest_aspect(img: Image.Image) -> str:
    options = {"1:1": 1, "3:4": 3 / 4, "4:3": 4 / 3, "9:16": 9 / 16, "16:9": 16 / 9,
               "2:3": 2 / 3, "3:2": 3 / 2, "4:5": 4 / 5, "5:4": 5 / 4, "21:9": 21 / 9}
    r = img.width / img.height
    return min(options, key=lambda k: abs(math.log(options[k] / r)))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(args) -> int:
    reference = Image.open(args.reference).convert("RGB")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = LOG_DIR / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"generate_scene_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    th = Thresholds()
    aspect = nearest_aspect(reference)

    def log(event: dict):
        event["time"] = time.strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    print(f"Reference: {args.reference} ({reference.width}x{reference.height}, {aspect})")
    print(f"Output:    {out_dir}")
    print(f"Log:       {log_path}")

    if args.dry_run:
        c = validate(reference, reference, th)
        print(f"Dry run: reference vs itself -> {'PASS' if c.ok else 'FAIL'} {asdict(c)}")
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        print(f"API key in environment: {'yes' if key else 'NO'}")
        return 0

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Missing SDK: pip install google-genai", file=sys.stderr)
        return 2
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) first.", file=sys.stderr)
        return 2
    client = genai.Client()  # reads the key from the environment

    previous: Image.Image | None = None
    for i in range(args.frames):
        frame_path = out_dir / f"frame_{i + 1:03d}.png"
        if args.resume and frame_path.exists():
            previous = Image.open(frame_path).convert("RGB")
            print(f"[{i + 1:02d}/{args.frames}] exists, skipping")
            continue

        compare_to = previous if previous is not None else reference
        best: tuple[float, Image.Image, Check] | None = None
        accepted = None

        for attempt in range(args.max_retries + 1):
            level = min(attempt, len(STRENGTH_LEVELS) - 1)
            prompt = build_prompt(i, args.frames, previous is not None, level)
            try:
                img = generate(client, types, args.model, reference, previous, prompt,
                               level, args.seed + i, aspect)
            except Exception as e:  # network error, safety block, no image, etc.
                print(f"[{i + 1:02d}/{args.frames}] attempt {attempt + 1}: API error: {e}")
                log({"frame": i + 1, "attempt": attempt + 1, "level": level, "error": str(e)})
                time.sleep(2 ** attempt)
                continue

            img = img.resize(reference.size, Image.LANCZOS)
            check = validate(img, compare_to, th)
            log({"frame": i + 1, "attempt": attempt + 1, "level": level,
                 "temperature": STRENGTH_LEVELS[level]["temperature"], **asdict(check)})

            if check.ok:
                accepted = img
                print(f"[{i + 1:02d}/{args.frames}] attempt {attempt + 1}: OK "
                      f"(drift {check.color_drift}, lines "
                      f"{min(check.line_recall, check.line_precision)})")
                break

            print(f"[{i + 1:02d}/{args.frames}] attempt {attempt + 1}: rejected: "
                  f"{'; '.join(check.reasons)}")
            img.save(rejected_dir / f"frame_{i + 1:03d}_attempt{attempt + 1}.png")
            if best is None or score(check) < best[0]:
                best = (score(check), img, check)

        if accepted is None:
            if args.keep_going and best is not None:
                accepted = best[1]
                print(f"[{i + 1:02d}/{args.frames}] all attempts failed; using best attempt "
                      f"(--keep-going)")
                log({"frame": i + 1, "used_best_rejected": True, **asdict(best[2])})
            else:
                print(f"Frame {i + 1} failed after {args.max_retries + 1} attempts. Stopping so a "
                      f"bad frame isn't chained into the rest. Rejected attempts are in "
                      f"{rejected_dir}. Re-run with --resume to continue.", file=sys.stderr)
                return 1

        accepted.save(frame_path)
        previous = accepted

    print(f"Done: {args.frames} frames in {out_dir}")
    print(f"Preview: ffmpeg -framerate 12 -i {out_dir}/frame_%03d.png "
          f"-pix_fmt yuv420p {out_dir}/preview.mp4")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                   help="style + structure reference image")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="folder for accepted frames")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true", help="skip frames that already exist")
    p.add_argument("--keep-going", action="store_true",
                   help="if every attempt fails, save the best one instead of stopping")
    p.add_argument("--dry-run", action="store_true",
                   help="check reference, key and validator without calling the API")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
