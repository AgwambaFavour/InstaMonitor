"""Censor: find marked words, brand names and logos in a video and cover them with your own images.

What it finds (all listed in config.json under "censor"):
  * words / brands  - text read from the frames by OCR (fuzzy, case-insensitive), and the same
                      terms in the post caption text.
  * logos           - every image in `logos_dir` (PNG/JPG crops of the logos you want gone).
                      Matched with SIFT features, so a logo is found at other sizes and angles.

How it covers them:
  Each hit becomes a region with a time window. The region is filled white (fill_color), then the replacement
  image is drawn on top, fitted inside it. The replacement is picked from `replacements`
  (keyed by the term, or by the logo file name without extension), else `default_replacement`,
  else the region is just filled (or boxed in box_color, if fallback is "box").

What it leaves alone:
  The source account's handle and `protected_terms` from the pipeline settings, unless the
  account is listed in `credit_removal_allowed` (accounts you have permission from to repost
  without their credit). A hit that overlaps a protected credit is skipped and logged.

Standalone test:
    python censor.py path/to/video.mp4 source_account     # prints what it would cover
"""
import json
import logging
import re
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pipeline

log = logging.getLogger("monitor")
BASE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ EDIT ME
# Text written over a censored name that has NO profile picture beside it
# (e.g. the "@UMBRA_HQ" watermark in the middle of a video).
#   REPLACEMENT_TEXT = "Manny"        -> writes "Manny"
#   REPLACEMENT_TEXT = "@uglymannyy"  -> writes "@uglymannyy"
#   REPLACEMENT_TEXT = ""             -> no text; the replacement image is used instead
REPLACEMENT_TEXT = "Manny"
REPLACEMENT_TEXT_COLOR = "black"
# ---------------------------------------------------------------------------

DEFAULTS = {
    "enabled": False,
    "words": [],                    # words/phrases to hide on screen and in captions
    "brands": [],                   # brand names; handled the same way as words
    "logos_dir": "censor/logos",    # reference logo images to look for
    "replacements": {},             # term or logo-file-stem -> image path
    "default_replacement": "",      # image used when a hit has no specific replacement
    "fallback": "fill",             # "fill" (fill_color) or "box" (box_color) when no replacement image is available
    "box_color": "black",
    "fill_color": "white",          # colour painted over each hit, under any replacement image
    "fill_only": [],                # terms that only get the fill, never an image or text
    "text_replacement": REPLACEMENT_TEXT,  # set at the top of this file
    "text_color": REPLACEMENT_TEXT_COLOR,
    "caption_mask": "",             # replacement text in captions; "" -> asterisks of same length
    "fuzzy": 0.85,                  # 0-1, how close OCR text must be to a term
    "logo_min_matches": 12,         # SIFT inliers needed to accept a logo hit
    "sample_fps": 2,                # frames per second scanned for hits
    "max_frames": 90,               # cap on scanned frames (long videos get sparser sampling)
    "pad": 6,                       # pixels added around each hit
    "cover_avatars": True,          # also cover a round profile picture beside a matched name
    "credit_removal_allowed": [],   # accounts whose credit you have permission to remove
}

_logo_cache = {}


def settings(cfg):
    s = dict(DEFAULTS)
    s.update(cfg.get("censor", {}))
    return s


def terms_of(s):
    return [t for t in s["words"] + s["brands"] if pipeline.norm(t)]


def credit_removal_allowed(cfg, account):
    s = settings(cfg)
    return bool(account) and pipeline.norm(account) in {pipeline.norm(a) for a in s["credit_removal_allowed"]}


def credit_terms(cfg, account=""):
    """Credits that stay protected: the source account and protected_terms, minus any
    account listed in credit_removal_allowed."""
    allowed = {pipeline.norm(a) for a in settings(cfg)["credit_removal_allowed"]}
    terms = [account] + list(pipeline.settings(cfg)["protected_terms"])
    return [t for t in terms if t and pipeline.norm(t) not in allowed]


def active_terms(cfg, account=""):
    """Terms the censor will act on: listed words/brands minus the protected source credits."""
    s = settings(cfg)
    if not s["enabled"]:
        return []
    credits = {pipeline.norm(c) for c in credit_terms(cfg, account)}
    out, dropped = [], []
    for t in terms_of(s):
        (dropped if pipeline.norm(t) in credits else out).append(t)
    if dropped:
        log.warning("Censor: not censoring %s - source credits without permission "
                    "(add the account to credit_removal_allowed if you have it)", sorted(set(dropped)))
    return out


# ---------------------------------------------------------------- caption text

def censor_text(text, cfg, account=""):
    """Mask listed words/brands in plain text (post captions, reworded on-screen captions)."""
    s = settings(cfg)
    if not s["enabled"] or not text:
        return text
    for term in sorted(active_terms(cfg, account), key=len, reverse=True):
        # Letters of the term with optional separators between them, so "Bet King",
        # "bet-king", "#betking" and "@BetKing" are all caught.
        letters = [re.escape(c) for c in term if c.isalnum()]
        pattern = r"(?<![A-Za-z0-9])[#@]?" + r"[\s._\-]?".join(letters) + r"(?![A-Za-z0-9])"
        mask = s["caption_mask"]
        text = re.sub(pattern, lambda m: mask or "*" * len(m.group(0)), text, flags=re.I)
    return text


# ---------------------------------------------------------------- sampling

def duration(video):
    r = subprocess.run([pipeline.FFMPEG, "-i", str(video)], capture_output=True, text=True, errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def sample(video, s, workdir):
    """Return [(time_seconds, frame_path)] spread over the whole video, and the sample step."""
    dur = duration(video) or 1.0
    fps = min(float(s["sample_fps"]), s["max_frames"] / dur)
    pattern = str(Path(workdir) / "c_%04d.png")
    subprocess.run([pipeline.FFMPEG, "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps:.4f}",
                    "-frames:v", str(int(s["max_frames"])), pattern], check=True)
    frames = sorted(Path(workdir).glob("c_*.png"))
    # ffmpeg's fps filter centres samples, so frame i sits at roughly (i + 0.5) / fps.
    return [((i + 0.5) / fps, f) for i, f in enumerate(frames)], 1.0 / fps


# ---------------------------------------------------------------- text hits

def span_box(text, box, start, end):
    """Approximate the pixel box of characters [start, end) inside an OCR line box."""
    x0, y0, x1, y1 = box
    n = max(1, len(text))
    return (int(x0 + (x1 - x0) * start / n), y0, int(x0 + (x1 - x0) * end / n + 0.999), y1)


def find_term_spans(text, term, fuzzy):
    """Return the distinct (start, end) character spans of `text` whose words match `term`."""
    spans = []
    for sp in _term_spans(text, term, fuzzy):
        if not any(sp[0] < b and a < sp[1] for a, b in spans):   # drop overlapping repeats
            spans.append(sp)
    return spans


def _term_spans(text, term, fuzzy):
    target = pipeline.norm(term)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    n_words = max(1, len(term.split()))
    for size in {n_words, n_words + 1, max(1, n_words - 1)}:
        for i in range(len(words) - size + 1):
            a, b = words[i][0], words[i + size - 1][1]
            cand = pipeline.norm(text[a:b])
            if not cand:
                continue
            if target in cand and len(cand) <= len(target) * 2.5:
                # Narrow to the matching part, e.g. "#BetKingNG" -> just "BetKing".
                lo = text[a:b].lower()
                hit = re.search(r"[\W_]*".join(re.escape(c) for c in target), lo)
                if hit:
                    yield a + hit.start(), a + hit.end()
                    continue
                yield a, b
            elif SequenceMatcher(None, cand, target).ratio() >= fuzzy:
                yield a, b


def joined_neighbours(items):
    """Extra pseudo-lines made by joining OCR pieces that sit side by side on one row.

    Animated or stylised text is often read in pieces ("bet" + "way"); joining them lets
    the term match. Both the spaced and unspaced joins are tried.
    """
    out = []
    for a in items:
        for b in items:
            if a is b:
                continue
            ax0, ay0, ax1, ay1 = a["box"]
            bx0, by0, bx1, by1 = b["box"]
            h = max(ay1 - ay0, by1 - by0)
            same_row = abs((ay0 + ay1) - (by0 + by1)) / 2 <= h * 0.5
            if same_row and 0 <= bx0 - ax1 <= h * 3:
                box = (ax0, min(ay0, by0), bx1, max(ay1, by1))
                out.append({"text": a["text"] + b["text"], "box": box})
    return out


def text_hits(items, terms, credit_terms, fuzzy):
    """Return (hits, credit_boxes) for one frame's OCR items."""
    hits, credits = [], []
    for it in items:
        for c in credit_terms:
            for a, b in find_term_spans(it["text"], c, 1.0):
                credits.append(span_box(it["text"], it["box"], a, b))
        for term in terms:
            for a, b in find_term_spans(it["text"], term, fuzzy):
                hits.append({"label": term, "kind": "text", "box": span_box(it["text"], it["box"], a, b),
                             "seen": it["text"]})
    for it in joined_neighbours(items):
        for term in terms:
            if any(pipeline.norm(term) in pipeline.norm(x["text"]) for x in items if inside(x["box"], it["box"])):
                continue  # one piece already matched on its own
            if find_term_spans(it["text"], term, fuzzy):
                hits.append({"label": term, "kind": "text", "box": it["box"], "seen": it["text"]})
    return hits, credits


# ---------------------------------------------------------------- logo hits

def load_logos(folder):
    folder = (BASE / folder) if not Path(folder).is_absolute() else Path(folder)
    key = str(folder)
    if key in _logo_cache:
        return _logo_cache[key]
    sift = cv2.SIFT_create()
    logos = []
    for p in sorted(folder.glob("*")) if folder.is_dir() else []:
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            continue
        img = Image.open(p)
        if img.mode in ("RGBA", "LA", "P"):
            # Put transparent logos on mid-grey so their edges still produce features.
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
            img = Image.alpha_composite(bg, img)
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        # Small crops give few features; upscale them.
        if max(gray.shape) < 300:
            f = 300 / max(gray.shape)
            gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            log.warning("Logo %s has too little detail to match reliably; skipping", p.name)
            continue
        logos.append({"name": p.stem, "kp": kp, "des": des, "shape": gray.shape})
    _logo_cache[key] = logos
    return logos


def logo_hits(frame_path, logos, min_matches):
    if not logos:
        return []
    sift = cv2.SIFT_create()
    gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    h_img, w_img = gray.shape
    hits = []
    for logo in logos:
        pairs = matcher.knnMatch(logo["des"], des, k=2)
        good = [m for m, *rest in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * rest[0].distance]
        if len(good) < min_matches:
            continue
        src = np.float32([logo["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or int(mask.sum()) < min_matches:
            continue
        h, w = logo["shape"]
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
        pts = corners.reshape(-1, 2)
        if not cv2.isContourConvex(pts.astype(np.int32)):
            continue  # a twisted quad means a bad match
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        if bw < 8 or bh < 8 or bw * bh > 0.6 * w_img * h_img:
            continue
        hits.append({"label": logo["name"], "kind": "logo", "inliers": int(mask.sum()),
                     "box": (int(max(0, x0)), int(max(0, y0)), int(min(w_img, x1)), int(min(h_img, y1)))})
    return hits


# ---------------------------------------------------------------- grouping

def find(video, account, cfg, workdir):
    """Scan the video and return regions to cover: [{label, kind, box, start, end}]."""
    s = settings(cfg)
    if not s["enabled"]:
        return []
    terms = active_terms(cfg, account)
    logos = load_logos(s["logos_dir"])
    if not terms and not logos:
        return []
    credits_kept = credit_terms(cfg, account)
    frames, step = sample(video, s, workdir)

    # Follow each hit from sample to sample as a track (same label, nearby, no long gap).
    tracks = []
    for t, f in frames:
        items = pipeline.ocr_frame(f) if terms else []
        texts, credits = text_hits(items, terms, credits_kept, s["fuzzy"])
        logo_list = logo_hits(f, logos, s["logo_min_matches"])
        # OCR often reads the name printed on a logo; the logo hit already covers it.
        texts = [x for x in texts if not any(inside(x["box"], lg["box"]) for lg in logo_list)]
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if texts and s["cover_avatars"] else None
        for h in texts + logo_list:
            if any(pipeline.overlaps(h["box"], c, pad=2) for c in credits):
                log.info("Censor: skipped %r at %.1fs - it overlaps the source credit", h["label"], t)
                continue
            if gray is not None and h["kind"] == "text":
                av = avatar_beside(gray, h["box"])
                if av:
                    h = dict(h, box=union(h["box"], av), avatar=True)
            for tr in tracks:
                lt, lb = tr["pts"][-1]
                if tr["label"] == h["label"] and lt < t and t - lt <= step * 1.6 and near(lb, h["box"]):
                    tr["pts"].append((t, h["box"]))
                    tr["avatar"] |= h.get("avatar", False)
                    break
            else:
                tracks.append({"label": h["label"], "kind": h["kind"], "pts": [(t, h["box"])],
                               "avatar": h.get("avatar", False)})

    # Turn each track into short timed boxes. Between two samples the box is the union of
    # both positions, so a moving logo stays covered without blanking its whole path.
    w, h = pipeline.video_size(video)
    pad, lead = s["pad"], step * 0.75
    regions = []
    for tr in tracks:
        pts = tr["pts"]
        head, tail = pts[0][1], pts[-1][1]
        if len(pts) > 1:
            # Keep moving the box the way the object was moving before/after the first/last sample.
            head = union(head, drift(pts[1], pts[0], lead))
            tail = union(tail, drift(pts[-2], pts[-1], lead))
        segs = [(max(0.0, pts[0][0] - lead), pts[0][0], head)]
        segs += [(a[0], b[0], union(a[1], b[1])) for a, b in zip(pts, pts[1:])]
        segs.append((pts[-1][0], pts[-1][0] + lead, tail))
        merged = []
        for start, end, box in segs:
            if merged and pipeline.iou(merged[-1]["box"], box) > 0.85:   # static: extend
                merged[-1]["end"] = end
                merged[-1]["box"] = union(merged[-1]["box"], box)
            else:
                merged.append({"label": tr["label"], "kind": tr["kind"], "box": box, "start": start, "end": end,
                               "avatar": tr["avatar"]})
        regions += merged
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        r["box"] = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    return regions


def avatar_beside(gray, box):
    """Box of a round profile picture just left of a text box (e.g. the avatar next to a
    username in a screenshotted post), or None if there isn't one."""
    x0, y0, x1, y1 = box
    th = max(6, y1 - y0)
    sx0, sy0, sy1 = max(0, x0 - 5 * th), max(0, y0 - 2 * th), min(gray.shape[0], y1 + 2 * th)
    if x0 - sx0 < th:
        return None
    roi = cv2.GaussianBlur(gray[sy0:sy1, sx0:x0], (3, 3), 0)
    found = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=th, param1=100, param2=18,
                             minRadius=int(th * 0.7), maxRadius=int(th * 2.2))
    if found is None:
        return None
    text_cy = (y0 + y1) / 2
    for cx, cy, r in found[0]:
        cx, cy, r = cx + sx0, cy + sy0, r
        # Must sit right beside the text and level with it; anything else is unrelated.
        if 0 <= x0 - (cx + r) <= max(8, th) and abs(cy - text_cy) <= r:
            return (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
    return None


def drift(p_from, p_to, seconds):
    """Box p_to moved further by `seconds` of the motion from p_from to p_to (each is (t, box))."""
    dt = (p_to[0] - p_from[0]) or 1e-6
    k = seconds / abs(dt)
    return tuple(int(b + (b - a) * k) for a, b in zip(p_from[1], p_to[1]))


def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def inside(a, b, frac=0.7):
    """True if at least `frac` of box a lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area >= frac


def near(a, b):
    """Same object in the next sample: overlapping, or centres within one box size."""
    if pipeline.overlaps(a, b):
        return True
    dx = abs((a[0] + a[2]) - (b[0] + b[2])) / 2
    dy = abs((a[1] + a[3]) - (b[1] + b[3])) / 2
    return dx <= max(a[2] - a[0], b[2] - b[0]) and dy <= max(a[3] - a[1], b[3] - b[1])


# ---------------------------------------------------------------- rendering

def replacement_for(label, s):
    for key, path in s["replacements"].items():
        if pipeline.norm(key) == pipeline.norm(label) and path:
            return path
    return s["default_replacement"]


def fit_image(path, bw, bh, out_png):
    """Fit an image inside bw x bh keeping its shape, centred on a transparent canvas."""
    img = Image.open(path).convert("RGBA")
    scale = min(bw / img.width, bh / img.height)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    canvas.alpha_composite(img, ((bw - img.width) // 2, (bh - img.height) // 2))
    canvas.save(out_png)


def fit_label(text, font_file, color, bw, bh, out_png):
    """Write `text` as large as fits inside bw x bh, centred on a transparent canvas."""
    canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    size = bh
    while True:
        font = ImageFont.truetype(font_file, max(6, size))
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        if (r - l <= bw * 0.92 and b - t <= bh * 0.8) or size <= 6:
            break
        size -= 1
    d.text(((bw - (r - l)) / 2 - l, (bh - (b - t)) / 2 - t), text, font=font, fill=color)
    canvas.save(out_png)


def ffmpeg_graph(regions, cfg, workdir, first_input=1):
    """Build the ffmpeg pieces that apply the regions to input [0:v].

    Returns (extra_input_args, filter_chain, output_label). With no regions the chain is
    empty and the label is "0:v", so callers can always splice it in.
    """
    s = settings(cfg)
    # Detection often reports the same box over the same time range more than once.
    seen, unique = set(), []
    for r in regions:
        key = (r["label"], tuple(r["box"]), round(r["start"], 2), round(r["end"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    regions = unique
    if not regions:
        return [], "", "0:v"

    # Every region is painted solid (fill_color, white by default, so it blends with the
    # background), then the replacement image is laid over it. All fills go down before
    # any image, so an overlapping fill can never cover a replacement.
    fills, overlays, inputs, idx = [], [], [], first_input
    for i, r in enumerate(regions):
        x0, y0, x1, y1 = r["box"]
        # Even sizes keep yuv420p happy.
        bw, bh = max(2, (x1 - x0) // 2 * 2), max(2, (y1 - y0) // 2 * 2)
        when = f"enable='between(t,{r['start']:.2f},{r['end']:.2f})'"
        png = Path(workdir) / f"rep_{i}.png"
        if pipeline.norm(r["label"]) in {pipeline.norm(t) for t in s["fill_only"]}:
            rep = ""                                     # just the white fill
        elif r["kind"] == "text" and not r.get("avatar") and s["text_replacement"]:
            rep = "text"                                 # a bare name: write the text instead
        else:
            rep = replacement_for(r["label"], s)
        color = s["box_color"] if s["fallback"] == "box" and not rep else s["fill_color"]
        fills.append(f"drawbox=x={x0}:y={y0}:w={bw}:h={bh}:color={color}:t=fill:{when}")
        if not rep:
            continue
        if rep == "text":
            fit_label(s["text_replacement"], pipeline.settings(cfg)["font_file"], s["text_color"], bw, bh, png)
        else:
            rep_path = Path(rep) if Path(rep).is_absolute() else pipeline.BASE / rep
            if not rep_path.is_file():
                log.warning("Censor: replacement image %s not found; filling %r instead", rep_path, r["label"])
                continue
            fit_image(rep_path, bw, bh, png)
        inputs += ["-i", str(png)]
        overlays.append((idx, x0, y0, when))
        idx += 1

    parts, cur = [f"[0:v]{','.join(fills)}[cfill]"], "cfill"
    for n, (k, x0, y0, when) in enumerate(overlays):
        parts.append(f"[{cur}][{k}:v]overlay={x0}:{y0}:{when}[cr{n}]")
        cur = f"cr{n}"
    return inputs, ";".join(parts), cur


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    config.setdefault("censor", {})["enabled"] = True
    with tempfile.TemporaryDirectory() as tmp:
        for reg in find(sys.argv[1], sys.argv[2].lstrip("@"), config, tmp):
            print(f"{reg['kind']:5} {reg['label']!r:20} box={reg['box']}  {reg['start']:.1f}s-{reg['end']:.1f}s")
