#!/usr/bin/env python3
"""Sprite-sheet pipeline: animate CHAR_01 doing a dumbbell curl over the locked gym background.

Inputs (never modified), all in universal_asset_library/:
  1_stages/indoor_rooms/perfect_gym.png       background, used as-is for every frame
  2_puppets/base_bodies/char_01_lead.png      character turnaround + expression sheet
  2_puppets/base_bodies/char_config.json      where the body, arm and heads sit on that sheet

What it does:
  1. Cuts sprites out of the sheet: the 3/4-view body and the expression heads.
     Transparency comes from the sheet itself: paper connected to the edge of a
     crop becomes transparent; ink keeps its anti-aliased edge; the inside of each
     head stays opaque.
  2. Splits the body's front forearm off as its own sprite, so it can rotate
     around the elbow (cut-out / puppet animation). Nothing is redrawn: every
     pixel of the character comes from the sheet.
  3. Takes the dumbbell sprite from the floor dumbbell in perfect_gym.png, so it
     matches the scene's art.
  4. For each of 24 frames: rotates the forearm, puts the dumbbell in the hand,
     swaps in an expression head, scales the character and layers it over the
     untouched background. Writes PNG frames and preview.mp4.

Usage:
    python3 sprite_animator.py                   # flow_animation_pipeline/frames/gym_sprite/ + preview.mp4
    python3 sprite_animator.py --x 1250 --y 1270 --scale 0.62
    python3 sprite_animator.py --debug-sprites   # also saves the cut-out sprites to flow_animation_pipeline/logs/sprites/
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LIBRARY = Path(__file__).resolve().parent.parent          # universal_asset_library/
WORKSPACE = LIBRARY.parent
BG_PATH = LIBRARY / "1_stages" / "indoor_rooms" / "perfect_gym.png"
PUPPET_DIR = LIBRARY / "2_puppets" / "base_bodies"
CHAR_CONFIG = PUPPET_DIR / "char_config.json"
CHARACTER = "char_01_lead"
OUT_DIR = WORKSPACE / "flow_animation_pipeline" / "frames" / "gym_sprite"
SPRITE_DEBUG = WORKSPACE / "flow_animation_pipeline" / "logs" / "sprites"

# --------------------------------------------------------------------------- sheet layout
# Read from 2_puppets/base_bodies/char_config.json (pixel coordinates in the character sheet).
_cfg = json.loads(CHAR_CONFIG.read_text())[CHARACTER]
_view = _cfg["views"]["three_quarter"]
SHEET_PATH = PUPPET_DIR / _cfg["sheet"]
BODY_BOX = tuple(_view["box"])                              # 3/4 view, head to feet
BODY_HEAD = (*_view["head"]["centre"], _view["head"]["radius"])
BODY_FEET_Y = _view["feet_y"]                               # bottom of the feet (anchor point)
BODY_CENTRE_X = _view["centre_x"]
ELBOW = tuple(_view["front_arm"]["elbow"])                  # front arm: elbow joint
HAND_END = tuple(_view["front_arm"]["hand_end"])            # front arm: end of the hand
ARM_HALF_WIDTH = _view["front_arm"]["half_width"]
EXPRESSION_BOXES = {k: tuple(v) for k, v in _cfg["expressions"].items() if not k.startswith("_")}

# Dumbbell lying on the gym floor in perfect_gym.png (2576 x 1438): two plates and a bar.
DUMBBELL_BOX = (2008, 994, 2136, 1078)
DUMBBELL_PARTS = {
    "ellipses": [((2036, 1031), (24, 34)), ((2106, 1037), (26, 37))],
    "rects": [(2040, 1024, 2088, 1043)],
}
DUMBBELL_GRIP = (2063, 1034)              # middle of the bar

# --------------------------------------------------------------------------- animation
CURL_DEGREES = 148                        # forearm swing from hanging to fully curled


def lift_amount(i: int, n: int) -> float:
    """0 = arm down, 1 = fully curled; down -> up -> down, loops seamlessly."""
    return (1 - math.cos(2 * math.pi * i / n)) / 2


def expression_for(i: int, n: int, lift: float) -> str | None:
    """None keeps the body's own 3/4 head (relaxed). Otherwise an expression from the sheet."""
    if lift < 0.25:
        return None
    if lift > 0.85:
        return "nervous"          # straining at the top of the curl
    return "determined"


# --------------------------------------------------------------------------- sprite cutting
def cut_sprite(rgb: np.ndarray, box, paper_min=150) -> np.ndarray:
    """Return an RGBA sprite for the drawing inside `box` of a paper-and-ink sheet."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    luma = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Paper = light pixels connected to the crop's border. (Guide lines are mid-grey,
    # so they count as paper and disappear.) Light pixels enclosed by ink, such as the
    # inside of a head, are kept.
    light = (luma > paper_min).astype(np.uint8)
    n, labels = cv2.connectedComponents(light, connectivity=4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    border.discard(0)
    paper = np.isin(labels, list(border)) & (light > 0)

    # Ink next to the paper keeps a soft edge; everything else is opaque.
    near_paper = cv2.dilate(paper.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ink_alpha = np.clip((235 - luma) / 195, 0, 1)
    alpha = np.where(paper, 0.0, np.where(near_paper, ink_alpha, 1.0))

    out = np.zeros((*crop.shape[:2], 4), np.uint8)
    out[..., :3] = crop
    edge = near_paper & ~paper
    out[edge, :3] = 18                     # edge pixels are pure ink; the alpha carries the softness
    out[..., 3] = (alpha * 255).round().astype(np.uint8)
    return out


def cut_dumbbell(rgb: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = DUMBBELL_BOX
    crop = rgb[y0:y1, x0:x1]
    shape = np.zeros(crop.shape[:2], np.uint8)
    for (cx, cy), (rx, ry) in DUMBBELL_PARTS["ellipses"]:
        cv2.ellipse(shape, (cx - x0, cy - y0), (rx, ry), 0, 0, 360, 255, -1)
    for ax0, ay0, ax1, ay1 in DUMBBELL_PARTS["rects"]:
        cv2.rectangle(shape, (ax0 - x0, ay0 - y0), (ax1 - x0, ay1 - y0), 255, -1)
    # Drop the floor colour around the parts, then soften the edge by one pixel.
    floor = np.array([178, 184, 195], np.float32)
    not_floor = np.abs(crop.astype(np.float32) - floor).max(2) > 22
    alpha = cv2.GaussianBlur(((shape > 0) & not_floor).astype(np.float32), (3, 3), 0)
    out = np.dstack([crop, (alpha * 255).astype(np.uint8)])
    return out


def keep_largest(sprite: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected shape (the head), dropping stray bits such as label text."""
    solid = (sprite[..., 3] > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 2:
        return sprite
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = sprite.copy()
    out[labels != biggest, 3] = 0
    return out


def find_head(sprite: np.ndarray):
    """Centre and outer radius of the head drawn in an expression sprite."""
    ys, xs = np.nonzero(sprite[..., 3] > 128)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return cx, cy, r


def split_forearm(body: np.ndarray, origin):
    """Move the front forearm (elbow to hand) out of the body sprite into its own sprite."""
    ox, oy = origin
    ex, ey = ELBOW[0] - ox, ELBOW[1] - oy
    hx, hy = HAND_END[0] - ox, HAND_END[1] - oy
    h, w = body.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vx, vy = hx - ex, hy - ey
    length2 = vx * vx + vy * vy
    t = ((xx - ex) * vx + (yy - ey) * vy) / length2
    px, py = ex + t.clip(0, 1) * vx, ey + t.clip(0, 1) * vy
    dist = np.hypot(xx - px, yy - py)
    sel = (t >= 0) & (t <= 1.3) & (dist <= ARM_HALF_WIDTH)

    forearm = np.zeros_like(body)
    forearm[sel] = body[sel]
    rest = body.copy()
    rest[sel, 3] = 0

    # Elbow cap: the round cross-section of ink at the joint, laid over the seam.
    cap_r = ARM_HALF_WIDTH - 6
    cap = np.zeros_like(body)
    joint = np.hypot(xx - ex, yy - ey) <= cap_r
    cap[joint] = body[joint]
    return rest, forearm, cap


# --------------------------------------------------------------------------- compositing
def rgba(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a, "RGBA")


def build_character(parts, lift: float, expression: str | None, canvas_box) -> Image.Image:
    """Assemble one pose in sheet pixel space on a transparent canvas."""
    cx0, cy0, cx1, cy1 = canvas_box
    canvas = Image.new("RGBA", (cx1 - cx0, cy1 - cy0), (0, 0, 0, 0))
    bx, by = BODY_BOX[0] - cx0, BODY_BOX[1] - cy0
    ex, ey = ELBOW[0] - cx0, ELBOW[1] - cy0

    canvas.alpha_composite(parts["body"], (bx, by))

    # Forearm: rotate around the elbow (positive = counter-clockwise on screen: down -> forward -> up).
    angle = CURL_DEGREES * lift
    fore_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fore_layer.alpha_composite(parts["forearm"], (bx, by))
    fore_layer = fore_layer.rotate(angle, resample=Image.BICUBIC, center=(ex, ey))

    # Hand position after rotation; the dumbbell sits under the hand.
    hx, hy = HAND_END[0] - cx0, HAND_END[1] - cy0
    grip_back = 14                                       # grip the bar just inside the hand's end
    vx, vy = hx - ex, hy - ey
    L = math.hypot(vx, vy)
    gx, gy = ex + vx * (L - grip_back) / L, ey + vy * (L - grip_back) / L
    a = math.radians(-angle)                             # screen y points down
    rx = ex + (gx - ex) * math.cos(a) - (gy - ey) * math.sin(a)
    ry = ey + (gx - ex) * math.sin(a) + (gy - ey) * math.cos(a)
    db = parts["dumbbell"]
    gdx, gdy = parts["dumbbell_grip"]
    canvas.alpha_composite(db, (round(rx - gdx), round(ry - gdy)))

    canvas.alpha_composite(fore_layer)
    canvas.alpha_composite(parts["cap"], (bx, by))

    if expression:
        head = parts["heads"][expression]
        hcx, hcy = parts["head_centres"][expression]
        canvas.alpha_composite(head, (round(BODY_HEAD[0] - cx0 - hcx), round(BODY_HEAD[1] - cy0 - hcy)))
    return canvas


def load_parts(scale: float, debug: bool):
    sheet = np.asarray(Image.open(SHEET_PATH).convert("RGB"))
    gym = np.asarray(Image.open(BG_PATH).convert("RGB"))

    body = cut_sprite(sheet, BODY_BOX)
    body, forearm, cap = split_forearm(body, BODY_BOX[:2])

    heads, centres = {}, {}
    for name, box in EXPRESSION_BOXES.items():
        spr = keep_largest(cut_sprite(sheet, box))
        hcx, hcy, hr = find_head(spr)
        k = BODY_HEAD[2] / hr                               # match the body's head size
        img = rgba(spr).resize((round(spr.shape[1] * k), round(spr.shape[0] * k)), Image.LANCZOS)
        heads[name] = img
        centres[name] = (hcx * k, hcy * k)

    # The dumbbell is cut at scene scale; bring it into sheet space so it scales with the
    # character. 1.1x because the character stands nearer the camera than the floor dumbbell.
    db = cut_dumbbell(gym)
    k = 1.1 / scale
    db_img = rgba(db).resize((round(db.shape[1] * k), round(db.shape[0] * k)), Image.LANCZOS)
    grip = ((DUMBBELL_GRIP[0] - DUMBBELL_BOX[0]) * k, (DUMBBELL_GRIP[1] - DUMBBELL_BOX[1]) * k)

    if debug:
        SPRITE_DEBUG.mkdir(parents=True, exist_ok=True)
        rgba(body).save(SPRITE_DEBUG / "body_without_forearm.png")
        rgba(forearm).save(SPRITE_DEBUG / "forearm.png")
        rgba(db).save(SPRITE_DEBUG / "dumbbell.png")
        for name, img in heads.items():
            img.save(SPRITE_DEBUG / f"head_{name}.png")
        print(f"Sprites saved to {SPRITE_DEBUG}")

    return {"body": rgba(body), "forearm": rgba(forearm), "cap": rgba(cap),
            "heads": heads, "head_centres": centres,
            "dumbbell": db_img, "dumbbell_grip": grip}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--x", type=float, default=1380, help="feet position in the background, x")
    ap.add_argument("--y", type=float, default=1270, help="feet position in the background, y")
    ap.add_argument("--scale", type=float, default=0.62, help="sheet pixels -> background pixels")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--debug-sprites", action="store_true")
    args = ap.parse_args()

    bg = Image.open(BG_PATH).convert("RGBA")              # locked: only ever copied
    parts = load_parts(args.scale, args.debug_sprites)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Canvas in sheet space: the body plus room for the curled arm and dumbbell.
    canvas_box = (BODY_BOX[0] - 40, BODY_BOX[1] - 10, BODY_BOX[2] + 300, BODY_BOX[3] + 10)
    cw, ch = canvas_box[2] - canvas_box[0], canvas_box[3] - canvas_box[1]
    dst_w, dst_h = round(cw * args.scale), round(ch * args.scale)
    # Where the canvas lands so the feet sit at (--x, --y).
    left = round(args.x - (BODY_CENTRE_X - canvas_box[0]) * args.scale)
    top = round(args.y - (BODY_FEET_Y - canvas_box[1]) * args.scale)

    for i in range(args.frames):
        lift = lift_amount(i, args.frames)
        char = build_character(parts, lift, expression_for(i, args.frames, lift), canvas_box)
        char = char.resize((dst_w, dst_h), Image.LANCZOS)
        frame = bg.copy()
        frame.alpha_composite(char, (left, top))
        frame.convert("RGB").save(out / f"frame_{i + 1:03d}.png")
    print(f"Wrote {args.frames} frames to {out}")

    if not args.no_preview:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not found; skipping preview.mp4")
            return 0
        mp4 = out / "preview.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "3",
                        "-framerate", str(args.fps), "-i", str(out / "frame_%03d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", "-crf", "18", str(mp4)], check=True)
        print(f"Preview: {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
