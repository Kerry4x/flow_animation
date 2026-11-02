#!/usr/bin/env python3
"""Erase the stick figures from references/05_gym_simple.jpg -> references/empty_gym.png.

Three tools, used where each works best:

  1. Patch search (most figures). Each figure is traced by hand below as thick
     lines, head circles and polygons. The traced area is filled in small tiles,
     working inward from its edge; for each tile the script searches along rows
     and columns of the clean picture for the patch whose surroundings match
     best, and copies it in. Walls, floor, rails, the mat edge and the repeating
     dumbbell rack all line up because they come from the same picture.

  2. Surface rebuild (bench and bike). The bench presser, spotter and cyclist are
     wrapped around their equipment, so there is nothing clean to copy from.
     These areas are rebuilt from the room's geometry instead: each pixel is
     labelled back wall, side wall or floor, filled with texture copied from the
     same surface in the same row, and the room's edge lines are redrawn. The bench
     pad and the bike go with their riders; the barbell rack stays, and its
     hidden left upright and foot are rebuilt by copying the right ones.

  3. A light OpenCV inpaint smooths any seams.

All coordinates are specific to 05_gym_simple.jpg (1376x768).

Usage:
    pip install opencv-python-headless numpy
    python3 clean_background.py           # writes references/empty_gym.png
    python3 clean_background.py --debug   # also writes logs/clean_mask_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "references" / "05_gym_simple.jpg"
DST = ROOT / "references" / "empty_gym.png"
DEBUG = ROOT / "logs" / "clean_mask_debug.png"

# ----------------------------------------------------------------------------
# 1. Figures removed by patch search.
#   lines: polylines (stroke width = "width"), circles: (cx, cy, r) heads, polys: filled areas
# ----------------------------------------------------------------------------
FIGURES = {
    "runner": {
        "width": 20,
        "circles": [(185, 322, 62)],
        "lines": [
            [(198, 380), (232, 508)],
            [(232, 508), (182, 560), (170, 636), (145, 638)],
            [(232, 508), (275, 572), (335, 552), (342, 568)],
            [(202, 384), (262, 398), (273, 457)],
            [(202, 384), (167, 438), (116, 404)],
        ],
    },
    "bench_arms_on_bar": {  # spotter's and presser's arms where they cross the barbell
        "width": 24,
        "lines": [
            [(538, 342), (545, 394)],
            [(538, 352), (520, 372), (490, 392), (488, 394)],
            [(538, 352), (562, 372), (566, 394)],
            [(458, 378), (466, 394)],
            [(586, 382), (580, 394)],
        ],
    },
}

# ----------------------------------------------------------------------------
# 1b. The dumbbell guy stands in front of the window and the dumbbell rack. Those
# are rebuilt by direct copying, which is more exact than searching:
#   - window glass and frame: copied sideways from clean glass in the same row
#   - the window's centre post: copied down from the clean part above
#   - the rack: copied from two dumbbells (81 px) further right, which repeats exactly
# ----------------------------------------------------------------------------
DUMBBELL_GUY = {
    "width": 20,
    "circles": [(932, 309, 50)],
    "lines": [
        [(932, 352), (928, 490)],
        [(928, 488), (900, 560), (888, 563)],
        [(928, 488), (966, 560), (980, 563)],
        [(931, 366), (972, 397), (986, 356)],
        [(931, 366), (902, 466)],
    ],
    "polys": [  # the two dumbbells he is holding
        [(964, 328), (1006, 322), (1038, 348), (1032, 386), (990, 384), (962, 364)],
        [(864, 442), (928, 444), (930, 494), (866, 494)],
    ],
}

GLASS_LEFT_SRC = (821, 878)      # clean strip of the left pane, wall and frame below it
GLASS_RIGHT_SRC = (1040, 1065)   # clean strip of the right pane
MULLION_X = (975, 990)           # window centre post
WINDOW_BOTTOM = 363              # bottom of the window frame
RACK_PERIOD = 81                 # two dumbbells
FLOOR_SRC = (1000, 1060)         # clean floor strip below the rack

# ----------------------------------------------------------------------------
# 2. Areas rebuilt from room geometry (wall / floor / edge lines).
# ----------------------------------------------------------------------------
FLOOR_Y = 509.5                       # back wall meets floor (line is y 508-511)
CORNER_X = 1250                       # back wall meets right-hand wall
SIDE_SLOPE = 0.52                     # right-hand floor edge: y = FLOOR_Y + slope * (x - CORNER_X)
MAT = [(551, 632), (885, 632), (916, 694), (512, 694)]   # blue mat outline corners


def _circle(cx, cy, r):
    return [tuple(p) for p in cv2.ellipse2Poly((cx, cy), (r, r), 0, 0, 360, 10)]


SURFACE_AREAS = {
    # Bench pad, presser and spotter below the barbell. Leaves the right upright and its foot.
    "bench": {
        "keep": [[(580, 512), (652, 512), (652, 570), (580, 570)]],      # right upright's foot
        "area": [[(352, 418), (428, 418), (428, 390), (627, 390), (627, 418), (660, 418),
                  (660, 602), (352, 602)]],
    },
    "spotter_head": {"keep": [], "area": [_circle(536, 300, 57)]},
    "runner": {
        "keep": [],
        "area": [_circle(185, 322, 66),
                 [(188, 384), (262, 386), (288, 398), (290, 470), (252, 516), (206, 516),
                  (206, 432), (190, 410)],
                 [(222, 500), (300, 540), (352, 540), (354, 582), (270, 590), (218, 524)]],
    },
    "situp": {
        "keep": [],
        "area": [[(700, 470), (860, 470), (868, 600), (845, 612), (845, 640), (772, 690),
                  (586, 690), (586, 626), (656, 576), (700, 556)]],
    },
    "pullup": {
        "keep": [],
        "area": [[(1222, 158), (1316, 158), (1316, 250), (1354, 262), (1358, 438),
                  (1262, 438), (1228, 330)]],
    },
    # Exercise bike and cyclist. Leaves the rack's right leg and the dumbbell on the floor.
    "bike": {
        "keep": [],
        "area": [[(1136, 296), (1262, 296), (1262, 438), (1300, 438), (1352, 500),
                  (1352, 728), (1092, 728), (1092, 596), (1140, 592), (1140, 528), (1132, 524),
                  (1116, 500), (1116, 440), (1132, 400), (1132, 366), (1136, 366)]],
    },
}

# Final touch-ups: copy a source box to a destination box, repeating it to fill the box.
PATCHES = [
    # bench rack: redraw both uprights below the bar from a clean stretch of the right one,
    # and rebuild the hidden left foot from the right foot
    {"src": (594, 494, 627, 506), "dst": (594, 386, 627, 506)},
    {"src": (594, 494, 627, 506), "dst": (428, 386, 461, 506)},
    {"src": (580, 506, 652, 570), "dst": (414, 506, 486, 570)},
    # barbell: the bar between the uprights was hidden by hands and arms
    {"src": (370, 368, 371, 390), "dst": (453, 368, 602, 390)},
    # window centre post behind the dumbbell guy
    {"src": (975, 262, 990, 300), "dst": (975, 300, 990, 352)},
    # window's right-hand frame behind the cyclist's head, and its bottom corner
    # (the top corner, flipped upside down)
    {"src": (1136, 250, 1156, 300), "dst": (1136, 296, 1156, 352)},
    {"src": (1136, 157, 1156, 171), "dst": (1136, 352, 1156, 366), "flip": True},
    # treadmill handrail under the runner's elbow (shifted 2 px down to follow its slope)
    {"src": (130, 440, 160, 470), "dst": (158, 442, 188, 472)},
    # two small leftover specks: floor beside the old bike, right-hand wall under the pull-up bar
    # (their sources are areas this script already rebuilt, so they're read from the result)
    {"src": (1120, 600, 1146, 610), "dst": (1120, 584, 1146, 594), "from_result": True},
    {"src": (1312, 262, 1324, 270), "dst": (1312, 243, 1324, 251), "from_result": True},
    # dumbbell rack's left frame, where the held dumbbell covered it
    {"src": (844, 404, 864, 440), "dst": (844, 440, 864, 476)},
]

# Pull-up bar: it slopes down to the left and the hands covered most of it. It is redrawn
# from one clean column of the bar, shifted down a little per pixel to follow the slope.
PULLUP_BAR = {"src_x": 1340, "rows": (163, 184), "x": (1229, 1316), "slope": 0.137}

MASK_GROW = 6
TILE = 10
SEARCH = 260
CROSS = 6
RING = 6
LINE_BGR = (20, 20, 22)


def _draw_shapes(m, fig):
    w = fig.get("width", 18)
    for pts in fig.get("lines", []):
        cv2.polylines(m, [np.array(pts, np.int32)], False, 255, w, cv2.LINE_AA)
        for p in pts:
            cv2.circle(m, p, w // 2 + 1, 255, -1)
    for cx, cy, r in fig.get("circles", []):
        cv2.circle(m, (cx, cy), r, 255, -1)
    for poly in fig.get("polys", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)


def figure_mask(shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for fig in FIGURES.values():
        _draw_shapes(m, fig)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_GROW + 1,) * 2)
    return cv2.dilate((m > 0).astype(np.uint8), k).astype(bool)


def surface_mask(shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for a in SURFACE_AREAS.values():
        cv2.fillPoly(m, [np.array(p, np.int32) for p in a["area"]], 255)
        for p in a["keep"]:
            cv2.fillPoly(m, [np.array(p, np.int32)], 0)
    return m > 0


# ---------------------------------------------------------------------------- patch search
def patch_fill(img: np.ndarray, hole: np.ndarray, clean: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32).copy()
    h, w = hole.shape
    hole = hole.copy()

    # The room is made of horizontal bands and vertical posts, so search along rows and columns.
    offsets = {(dy, dx) for dy in range(-CROSS, CROSS + 1, 2) for dx in range(-SEARCH, SEARCH + 1)}
    offsets |= {(dy, dx) for dx in range(-CROSS, CROSS + 1, 2) for dy in range(-SEARCH, SEARCH + 1)}
    offsets.discard((0, 0))
    offsets = np.array(sorted(offsets))
    # Prefer sideways copies (rows hold the same surface), then nearby ones.
    penalty = 1 + np.abs(offsets[:, 0]) / 25 + np.abs(offsets[:, 1]) / (8 * SEARCH)

    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1,) * 2)
    while hole.any():
        inner = cv2.erode(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.nonzero(hole & ~inner)
        progressed = False
        for ty, tx in sorted(set(zip(ys // TILE, xs // TILE))):
            y0, x0 = ty * TILE, tx * TILE
            tile = np.zeros_like(hole)
            tile[y0:y0 + TILE, x0:x0 + TILE] = hole[y0:y0 + TILE, x0:x0 + TILE]
            if not tile.any():
                continue
            ring = cv2.dilate(tile.astype(np.uint8), ring_k).astype(bool) & ~hole
            ry, rx = np.nonzero(ring)
            hy, hx = np.nonzero(tile)
            if len(ry) < 8:
                continue
            sy, sx = ry[None] + offsets[:, :1], rx[None] + offsets[:, 1:]
            ty_, tx_ = hy[None] + offsets[:, :1], hx[None] + offsets[:, 1:]
            ok = ((sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)).all(1)
            ok &= ((ty_ >= 0) & (ty_ < h) & (tx_ >= 0) & (tx_ < w)).all(1)
            idx = np.nonzero(ok)[0]
            if not len(idx):
                continue
            good = clean[ty_[idx], tx_[idx]].all(1) & clean[sy[idx], sx[idx]].all(1)
            idx = idx[good]
            if not len(idx):
                continue
            ssd = ((out[sy[idx], sx[idx]] - out[ry, rx][None]) ** 2).sum((1, 2))
            best = idx[np.argmin(ssd * penalty[idx])]
            out[hy, hx] = out[ty_[best], tx_[best]]
            hole[hy, hx] = False
            progressed = True
        if not progressed:
            break
    out = out.clip(0, 255).astype(np.uint8)
    if hole.any():
        out = cv2.inpaint(out, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    return out


# ---------------------------------------------------------------------------- surface rebuild
def surface_of(x, y):
    """0 = back wall, 1 = right-hand wall, 2 = floor, 3 = mat (by room geometry)."""
    side_floor = FLOOR_Y + SIDE_SLOPE * (x - CORNER_X)
    floor = np.where(x < CORNER_X, y > FLOOR_Y, y > side_floor)
    geo = np.where(floor, 2, np.where(x < CORNER_X, 0, 1))
    mat = np.zeros(x.shape, np.uint8)
    cv2.fillPoly(mat, [np.array(MAT, np.int32)], 1)
    return np.where(mat > 0, 3, geo)


def surface_fill(img: np.ndarray, hole: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = hole.shape
    yy, xx = np.mgrid[0:h, 0:w]
    geo = surface_of(xx, yy)

    # Which clean pixels really show plain wall or floor (not equipment, not edge lines)?
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = {0: (slice(120, 160), slice(420, 600)),       # back wall
               1: (slice(300, 400), slice(1350, 1376)),     # right-hand wall
               2: (slice(720, 760), slice(420, 820)),       # floor
               3: (slice(674, 688), slice(752, 830))}       # mat
    plain = np.zeros((4, h, w), bool)
    for s, (sy, sx) in samples.items():
        ref = lab[sy, sx].reshape(-1, 3)
        mu, sd = ref.mean(0), ref.std(0) + 1.0
        dist = np.abs((lab - mu) / sd).max(2)
        plain[s] = (dist < 2.5) & (geo == s) & ~blocked

    # Fill each hole pixel with texture from the same surface in the same (or nearest) row.
    for y in np.unique(np.nonzero(hole)[0]):
        for s in (0, 1, 2, 3):
            tx = np.nonzero(hole[y] & (geo[y] == s))[0]
            if not len(tx):
                continue
            for dy in range(0, 200):
                rows = [r for r in (y - dy, y + dy) if 0 <= r < h and surface_ok(geo, r, s, y)]
                src = [(r, np.nonzero(plain[s, r])[0]) for r in rows]
                src = [(r, xs) for r, xs in src if len(xs) >= 8]
                if src:
                    r, xs = src[0]
                    out[y, tx] = img[r, nearby_sources(tx, xs, w)]
                    break

    # Redraw the room's edge lines through the rebuilt areas.
    lines = np.zeros((h, w), np.uint8)
    cv2.line(lines, (0, round(FLOOR_Y)), (CORNER_X, round(FLOOR_Y)), 255, 4)
    end_x = w - 1
    cv2.line(lines, (CORNER_X, round(FLOOR_Y)),
             (end_x, round(FLOOR_Y + SIDE_SLOPE * (end_x - CORNER_X))), 255, 5, cv2.LINE_AA)
    cv2.line(lines, (CORNER_X + 1, 85), (CORNER_X + 1, round(FLOOR_Y)), 255, 4)
    cv2.polylines(lines, [np.array(MAT, np.int32)], True, 255, 4, cv2.LINE_AA)
    sel = (lines > 0) & hole
    alpha = (lines[sel].astype(np.float32) / 255)[:, None]
    out[sel] = (alpha * np.array(LINE_BGR) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def nearby_sources(tx: np.ndarray, xs: np.ndarray, w: int) -> np.ndarray:
    """For each gap in the row, borrow the closest real surface pixels (one distinct source
    pixel per target, kept in order), so brightness matches the surroundings and the
    paper texture is copied rather than smeared."""
    src = np.empty_like(tx)
    breaks = np.nonzero(np.diff(tx) > 1)[0] + 1
    for run in np.split(tx, breaks):
        centre = (run[0] + run[-1]) / 2
        near = np.sort(xs[np.argsort(np.abs(xs - centre), kind="stable")[:len(run)]])
        src[np.searchsorted(tx, run)] = near[np.arange(len(run)) % len(near)]
    return src


def surface_ok(geo, r, s, y):
    # Only borrow from rows on the same side of the floor line, so bands don't bleed.
    return (r <= FLOOR_Y) == (y <= FLOOR_Y) or s == 1


def apply_patches(img: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Copy each PATCHES source box (taken from the untouched original) into its destination."""
    out = img.copy()
    for p in PATCHES:
        x0, y0, x1, y1 = p["src"]
        dx0, dy0, dx1, dy1 = p["dst"]
        src = (img if p.get("from_result") else original)[y0:y1, x0:x1]
        if p.get("flip"):
            src = src[::-1]
        reps = (-(-(dy1 - dy0) // src.shape[0]), -(-(dx1 - dx0) // src.shape[1]), 1)
        out[dy0:dy1, dx0:dx1] = np.tile(src, reps)[:dy1 - dy0, :dx1 - dx0]
    return out


def draw_pullup_bar(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    b = PULLUP_BAR
    r0, r1 = b["rows"]
    col = img[r0:r1, b["src_x"]]
    for x in range(b["x"][0], b["x"][1]):
        shift = round((b["src_x"] - x) * b["slope"])
        out[r0 + shift:r1 + shift, x] = col
    # Close the bar's left end with an outline.
    xl = b["x"][0]
    top = r0 + round((b["src_x"] - xl) * b["slope"])
    cv2.line(out, (xl, top + 4), (xl, top + 17), LINE_BGR, 3)
    return out


def dumbbell_guy_mask(shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    _draw_shapes(m, DUMBBELL_GUY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_GROW + 1,) * 2)
    return cv2.dilate((m > 0).astype(np.uint8), k).astype(bool)


def copy_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Rebuild the window and rack behind the dumbbell guy by direct copying."""
    out = img.copy()
    h, w = hole.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m0, m1 = MULLION_X
    floor = hole & (yy > FLOOR_Y + 2)
    rack = hole & (yy >= WINDOW_BOTTOM + 3) & ~floor
    glass = hole & ~rack & ~floor

    def rows(sel, src):
        ys, xs = np.nonzero(sel)
        width = src[1] - src[0]
        out[ys, xs] = img[ys, src[0] + (xs - src[0]) % width]

    rows(glass & (xx < m0), GLASS_LEFT_SRC)
    rows(glass & (xx >= m0), GLASS_RIGHT_SRC)   # mullion part is redrawn by PATCHES
    rows(floor, FLOOR_SRC)
    # Rack: copy from RACK_PERIOD px to the right, sweeping right to left so every source is clean.
    for x in sorted(set(np.nonzero(rack)[1]), reverse=True):
        ys = np.nonzero(rack[:, x])[0]
        out[ys, x] = out[ys, min(x + RACK_PERIOD, w - 1)]
    return out


def smooth_seams(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    edge = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return cv2.inpaint(img, edge.astype(np.uint8) * 255, 2, cv2.INPAINT_TELEA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--debug", action="store_true", help="also save the mask overlay")
    args = ap.parse_args()

    img = cv2.imread(args.src)
    if img is None:
        raise SystemExit(f"Can't read {args.src}")
    surf = surface_mask(img.shape)
    dbg_guy = dumbbell_guy_mask(img.shape) & ~surf
    figs = figure_mask(img.shape) & ~surf & ~dbg_guy
    removed = figs | surf | dbg_guy

    if args.debug:
        dbg = img.copy()
        dbg[figs] = (0.4 * dbg[figs] + [0, 0, 150]).astype(np.uint8)
        dbg[surf] = (0.4 * dbg[surf] + [150, 0, 0]).astype(np.uint8)
        dbg[dbg_guy] = (0.4 * dbg[dbg_guy] + [0, 150, 0]).astype(np.uint8)
        DEBUG.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG), dbg)
        print(f"Mask overlay (red = patch search, blue = surface rebuild, green = direct copy): {DEBUG}")

    out = surface_fill(img, surf, blocked=removed)
    out = copy_fill(out, dbg_guy)
    out = patch_fill(out, figs, clean=~removed)
    out = apply_patches(out, img)
    out = draw_pullup_bar(out)
    out = smooth_seams(out, removed)
    cv2.imwrite(args.dst, out)
    print(f"Removed {int(removed.sum())} px -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
