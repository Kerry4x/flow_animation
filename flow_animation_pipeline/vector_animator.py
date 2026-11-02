#!/usr/bin/env python3
"""Draw a stickman curling a dumbbell over the gym background, frame by frame.

No AI and no API: the background is loaded once and a stickman is drawn on top
with Pillow. Only the curling forearm and dumbbell are moved, using simple
trigonometry, so every other pixel is identical in every frame.

Usage:
    python3 vector_animator.py                      # 24 frames -> frames/gym_vector/
    python3 vector_animator.py --preview            # also writes preview.mp4 (needs ffmpeg)
    python3 vector_animator.py --x 700 --y 745 --scale 1.2   # move / resize the figure
    python3 vector_animator.py --background references/05_gym_simple.jpg   # original busy gym
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DEFAULT_BG = ROOT / "references" / "empty_gym.png"      # made by clean_background.py
DEFAULT_OUT = ROOT / "frames" / "gym_vector"

SS = 4  # supersampling factor for smooth (anti-aliased) lines

INK = (17, 17, 17, 255)
HEAD_FILL = (238, 238, 236, 255)
PLATE = (62, 62, 64, 255)
PLATE_HUB = (150, 150, 152, 255)
BAR = (120, 120, 122, 255)


@dataclass
class Rig:
    """Figure measurements in background pixels at scale 1.0 (feet = anchor point)."""
    head_r: float = 46
    neck: float = 6
    torso: float = 120
    leg: float = 128
    hip_width: float = 26
    upper_arm: float = 66
    forearm: float = 62
    line: float = 7
    dumbbell_bar: float = 44
    plate_len: float = 26
    plate_thick: float = 17


def polar(origin, length, angle_deg):
    """Point at `length` from `origin`. Angles: 0 = right, 90 = straight down (screen coords)."""
    a = math.radians(angle_deg)
    return origin[0] + length * math.cos(a), origin[1] + length * math.sin(a)


def lift_amount(i: int, n: int) -> float:
    """0 = arm down, 1 = fully curled. Down -> up -> down; frame n+1 would equal frame 1, so it loops."""
    return (1 - math.cos(2 * math.pi * i / n)) / 2


class Canvas:
    """Supersampled RGBA layer covering only the figure's area, composited onto the background."""

    def __init__(self, box, ss=SS):
        self.x0, self.y0, x1, y1 = box
        self.ss = ss
        self.img = Image.new("RGBA", ((x1 - self.x0) * ss, (y1 - self.y0) * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def p(self, pt):
        return (pt[0] - self.x0) * self.ss, (pt[1] - self.y0) * self.ss

    def line(self, a, b, width, color=INK):
        w = width * self.ss
        pa, pb = self.p(a), self.p(b)
        self.d.line([pa, pb], fill=color, width=round(w))
        for c in (pa, pb):  # round caps
            self.d.ellipse([c[0] - w / 2, c[1] - w / 2, c[0] + w / 2, c[1] + w / 2], fill=color)

    def circle(self, c, r, fill=None, outline=INK, width=0):
        pc = self.p(c)
        R = r * self.ss
        self.d.ellipse([pc[0] - R, pc[1] - R, pc[0] + R, pc[1] + R], fill=fill,
                       outline=outline if width else None, width=round(width * self.ss))

    def arc(self, c, r, start, end, width):
        pc = self.p(c)
        R = r * self.ss
        self.d.arc([pc[0] - R, pc[1] - R, pc[0] + R, pc[1] + R], start, end, fill=INK,
                   width=round(width * self.ss))

    def composite(self, base: Image.Image):
        w, h = self.img.width // self.ss, self.img.height // self.ss
        small = self.img.resize((w, h), Image.LANCZOS)
        base.alpha_composite(small, (self.x0, self.y0))


def draw_dumbbell(cv: Canvas, hand, forearm_angle, r: Rig, s: float):
    # The hand grips the middle of the bar. The bar stays roughly level (as in the reference
    # art) and tilts only slightly with the forearm, so it reads as a real curl.
    tilt = (forearm_angle - 90) * 0.15
    half = r.dumbbell_bar * s / 2
    a = polar(hand, half, tilt)
    b = polar(hand, half, tilt + 180)
    plate_dir = tilt + 90
    cv.line(a, b, r.line * s * 0.9 + 4 * s, INK)
    cv.line(a, b, r.line * s * 0.9, BAR)
    for end in (a, b):
        # Each plate is a short thick slab across the bar, outlined in ink.
        p1 = polar(end, r.plate_len * s / 2, plate_dir)
        p2 = polar(end, r.plate_len * s / 2, plate_dir + 180)
        cv.line(p1, p2, r.plate_thick * s + 5 * s, INK)
        cv.line(p1, p2, r.plate_thick * s, PLATE)
        cv.circle(end, 3.2 * s, fill=PLATE_HUB)


def draw_frame(bg: Image.Image, feet, s: float, lift: float, r: Rig, flip: bool) -> Image.Image:
    frame = bg.copy()
    fx, fy = feet
    lw = r.line * s
    side = -1 if flip else 1  # which side the curling arm is on

    hip = (fx, fy - r.leg * s)
    neck = (fx, hip[1] - r.torso * s)
    head = (fx, neck[1] - (r.neck + r.head_r) * s)
    shoulder = (fx, neck[1] + 12 * s)

    def ang(a):  # mirror an angle horizontally when flipped
        return 180 - a if flip else a

    # Static arm: hangs down and slightly out.
    s_elbow = polar(shoulder, r.upper_arm * s, ang(118))
    s_hand = polar(s_elbow, r.forearm * s, ang(100))

    # Curling arm: upper arm fixed, forearm swings from down (80 deg) to up-and-out (-62 deg).
    upper_angle = ang(76)
    c_elbow = polar(shoulder, r.upper_arm * s, upper_angle)
    forearm_angle = ang(80 - 142 * lift)
    c_hand = polar(c_elbow, r.forearm * s, forearm_angle)

    reach = (r.head_r + r.upper_arm + r.forearm + r.dumbbell_bar) * s
    box = (int(fx - reach - 20), int(head[1] - r.head_r * s - 20),
           int(fx + reach + 20), int(fy + 20))
    box = (max(box[0], 0), max(box[1], 0), min(box[2], bg.width), min(box[3], bg.height))
    cv = Canvas(box)

    # Legs and body.
    cv.line(hip, (fx - r.hip_width * s, fy), lw)
    cv.line(hip, (fx + r.hip_width * s, fy), lw)
    cv.line(hip, neck, lw)

    # Static arm, then curling arm with dumbbell (dumbbell drawn under the hand line end).
    cv.line(shoulder, s_elbow, lw)
    cv.line(s_elbow, s_hand, lw)
    cv.line(shoulder, c_elbow, lw)
    draw_dumbbell(cv, c_hand, forearm_angle, r, s)
    cv.line(c_elbow, c_hand, lw)

    # Head: filled circle with ink outline, face looks toward the dumbbell.
    cv.circle(head, r.head_r * s, fill=HEAD_FILL, outline=INK, width=lw)
    look = side * 9 * s
    eye_y = head[1] - 6 * s
    for ex in (-15, 15):
        cv.circle((head[0] + ex * s + look, eye_y), 5 * s, fill=INK)
    # Mouth: neutral line at rest, tight downturned "effort" mouth near the top of the lift.
    mouth_y = head[1] + 20 * s
    mx = head[0] + look
    if lift < 0.6:
        cv.line((mx - 10 * s, mouth_y), (mx + 10 * s, mouth_y), 3.2 * s)
    else:
        cv.arc((mx, mouth_y + 10 * s), 11 * s, 220, 320, 3.2 * s)

    cv.composite(frame)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--background", default=str(DEFAULT_BG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--x", type=float, default=1195, help="feet position, x (background pixels)")
    ap.add_argument("--y", type=float, default=700, help="feet position, y (background pixels)")
    ap.add_argument("--scale", type=float, default=1.1)
    ap.add_argument("--flip", action="store_true", help="curl with the other arm")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--preview", action="store_true", help="also write preview.mp4 with ffmpeg")
    args = ap.parse_args()

    bg = Image.open(args.background).convert("RGBA")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = Rig()

    for i in range(args.frames):
        f = draw_frame(bg, (args.x, args.y), args.scale, lift_amount(i, args.frames), rig, args.flip)
        f.convert("RGB").save(out / f"frame_{i + 1:03d}.png")
    print(f"Wrote {args.frames} frames to {out}")

    if args.preview:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not found; skipping preview")
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
