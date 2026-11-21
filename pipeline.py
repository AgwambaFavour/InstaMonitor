"""Video pipeline: re-caption a downloaded video, add your watermark, send to Telegram.

For each video:
  1. Sample a few frames and OCR them.
  2. Split the text found into *protected* text (the source account's handle, anything
     with "@", and any `protected_terms` from config) and *caption* text (the rest of
     the on-screen text that stays put across frames).
  3. Cover only the caption area with white and draw the reworded caption on it,
     smaller. Protected text is never covered: caption boxes that overlap it are dropped.
  4. Add your watermark in a free corner that doesn't cover the source credit.
  5. Send the result to Telegram.

Standalone test:
    python pipeline.py path/to/video.mp4 source_account [--no-send]
"""
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("monitor")
BASE = Path(__file__).resolve().parent
SECRETS_FILE = BASE / "secrets.json"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DEFAULTS = {
    "enabled": False,
    "font_file": "C:/Windows/Fonts/arialbd.ttf",
    "caption_scale": 0.8,          # new caption text size relative to the original line height
    "caption_color": "black",
    "cover_color": "white",
    "watermark_text": "",
    "watermark_image": "",          # PNG with transparency; used instead of text if set
    "watermark_opacity": 0.85,
    "watermark_scale": 0.035,       # text height as a fraction of video height
    "reword": "claude",            # "claude" or "none"
    "protected_terms": [],          # extra words that must never be covered (blog names etc.)
    "output_dir": "processed",
    "send_to_telegram": True,
    "sample_frames": 6,
}

_ocr = None


def settings(cfg):
    s = dict(DEFAULTS)
    s.update(cfg.get("pipeline", {}))
    return s


def secret(name):
    """Read a secret from the environment, then secrets.json (git-ignored)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8")).get(name, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


# ---------------------------------------------------------------- OCR

def get_ocr():
    global _ocr
    if _ocr is None:
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        from rapidocr import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def sample_frames(video, n, workdir):
    """Grab up to n frames spread over the first part of the video."""
    pattern = str(Path(workdir) / "f_%02d.png")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-vf", "fps=1",
                    "-frames:v", str(n), pattern], check=True)
    return sorted(Path(workdir).glob("f_*.png"))


def ocr_frame(path):
    r = get_ocr()(str(path))
    out = []
    if r.boxes is None:
        return out
    for box, txt, score in zip(r.boxes, r.txts, r.scores):
        if score < 0.6 or not txt.strip():
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": txt.strip(), "box": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))})
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def overlaps(a, b, pad=0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_protected(text, terms):
    if "@" in text:
        return True
    t = norm(text)
    return any(term and term in t for term in terms)


def analyse(frames, account, extra_terms):
    """Return (caption_items, protected_boxes) from OCR over several frames."""
    terms = [norm(account)] + [norm(x) for x in extra_terms]
    per_frame = [ocr_frame(f) for f in frames]
    protected = [i["box"] for items in per_frame for i in items if is_protected(i["text"], terms)]

    # Caption = unprotected text that sits in the same place in at least half the frames.
    need = max(1, len(per_frame) // 2)
    best = max(per_frame, key=len) if per_frame else []
    caption = []
    for item in best:
        if is_protected(item["text"], terms):
            continue
        hits = sum(any(iou(item["box"], o["box"]) > 0.5 for o in items) for items in per_frame)
        if hits >= need:
            caption.append(item)

    # Never cover protected text: drop any caption line that touches it.
    caption = [c for c in caption if not any(overlaps(c["box"], p, pad=6) for p in protected)]
    caption.sort(key=lambda c: (c["box"][1], c["box"][0]))
    return caption, protected


# ---------------------------------------------------------------- rewording

def reword(text, mode):
    if mode != "claude" or not text.strip():
        return text
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY") or None)
        resp = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            output_config={"effort": "low"},
            system=("You rewrite short on-screen video captions. Keep the same meaning and tone, "
                    "use different wording, keep it about the same length or shorter. "
                    "Do not add hashtags, account names or quotes. Reply with the caption only."),
            messages=[{"role": "user", "content": text}],
        )
        if resp.stop_reason == "refusal":
            log.warning("Reword declined; keeping original caption text")
            return text
        out = "".join(b.text for b in resp.content if b.type == "text").strip()
        return out or text
    except Exception as e:
        log.warning("Reword failed (%s); keeping original caption text", e)
        return text


# ---------------------------------------------------------------- drawing

def video_size(video):
    r = subprocess.run([FFMPEG, "-i", str(video)], capture_output=True, text=True, errors="replace")
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    if not m:
        raise RuntimeError("could not read video size")
    return int(m.group(1)), int(m.group(2))


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def fit_text(draw, text, font_file, start_size, box_w, box_h):
    size = max(10, int(start_size))
    while size > 10:
        font = ImageFont.truetype(font_file, size)
        lines = wrap(draw, text, font, box_w)
        line_h = int(size * 1.25)
        if line_h * len(lines) <= box_h and all(draw.textlength(l, font=font) <= box_w for l in lines):
            return font, lines, line_h
        size -= 2
    font = ImageFont.truetype(font_file, 10)
    return font, wrap(draw, text, font, box_w), 13


def pick_corner(w, h, mw, mh, avoid):
    m = int(min(w, h) * 0.03)
    corners = [(w - mw - m, h - mh - m), (m, h - mh - m), (w - mw - m, m), (m, m)]
    for x, y in corners:
        if not any(overlaps((x, y, x + mw, y + mh), a, pad=10) for a in avoid):
            return x, y
    return corners[0]


def build_overlay(size, caption, new_text, protected, s, out_png):
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    avoid = list(protected)

    if caption:
        x0 = min(c["box"][0] for c in caption); y0 = min(c["box"][1] for c in caption)
        x1 = max(c["box"][2] for c in caption); y1 = max(c["box"][3] for c in caption)
        pad = 8
        cover = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        if any(overlaps(cover, p) for p in protected):
            cover = (x0, y0, x1, y1)  # tighter box if the padding would touch the credit
        d.rectangle(cover, fill=s["cover_color"])
        orig_line_h = sum(c["box"][3] - c["box"][1] for c in caption) / len(caption)
        bw, bh = cover[2] - cover[0] - 2 * pad, cover[3] - cover[1] - 2 * pad
        font, lines, line_h = fit_text(d, new_text, s["font_file"], orig_line_h * s["caption_scale"], bw, bh)
        top = cover[1] + (cover[3] - cover[1] - line_h * len(lines)) // 2
        for i, line in enumerate(lines):
            lw = d.textlength(line, font=font)
            d.text((cover[0] + (cover[2] - cover[0] - lw) / 2, top + i * line_h), line,
                   font=font, fill=s["caption_color"])
        avoid.append(cover)

    alpha = int(255 * s["watermark_opacity"])
    if s["watermark_image"]:
        wm = Image.open(s["watermark_image"]).convert("RGBA")
        target_w = int(w * 0.22)
        wm = wm.resize((target_w, int(wm.height * target_w / wm.width)))
        wm.putalpha(wm.getchannel("A").point(lambda a: a * alpha // 255))
        x, y = pick_corner(w, h, wm.width, wm.height, avoid)
        img.alpha_composite(wm, (x, y))
    elif s["watermark_text"]:
        font = ImageFont.truetype(s["font_file"], max(14, int(h * s["watermark_scale"])))
        tb = d.textbbox((0, 0), s["watermark_text"], font=font, stroke_width=2)
        x, y = pick_corner(w, h, tb[2] - tb[0], tb[3] - tb[1], avoid)
        d.text((x - tb[0], y - tb[1]), s["watermark_text"], font=font, fill=(255, 255, 255, alpha),
               stroke_width=2, stroke_fill=(0, 0, 0, alpha))
    img.save(out_png)


def render(video, overlay_png, out):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(overlay_png),
                    "-filter_complex", "[0:v][1:v]overlay=0:0", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                    str(out)], check=True)


# ---------------------------------------------------------------- Telegram

def send_telegram(video, caption):
    token, chat = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); skipping send")
        return False
    if Path(video).stat().st_size > 50 * 1024 * 1024:
        log.warning("%s is over Telegram's 50 MB bot limit; skipping send", Path(video).name)
        return False
    with open(video, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendVideo",
                          data={"chat_id": chat, "caption": caption[:1024], "supports_streaming": "true"},
                          files={"video": f}, timeout=300)
    if not r.ok:
        log.warning("Telegram send failed: %s %s", r.status_code, r.text[:200])
        return False
    return True


# ---------------------------------------------------------------- entry points

def process_video(video, account, cfg, post_caption="", send=True):
    s = settings(cfg)
    video = Path(video)
    out_dir = BASE / s["output_dir"] / account
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / video.name

    with tempfile.TemporaryDirectory() as tmp:
        frames = sample_frames(video, s["sample_frames"], tmp)
        caption, protected = analyse(frames, account, s["protected_terms"])
        old_text = " ".join(c["text"] for c in caption)
        new_text = reword(old_text, s["reword"]) if old_text else ""
        log.info("[%s] %s: caption %r -> %r (%d protected text box(es) kept)",
                 account, video.name, old_text, new_text, len(protected))
        overlay = Path(tmp) / "overlay.png"
        build_overlay(video_size(video), caption, new_text, protected, s, overlay)
        render(video, overlay, out)

    if send and s["send_to_telegram"]:
        text = reword(post_caption, s["reword"]) if post_caption else new_text
        text = f"{text}\n\nSource: @{account}".strip()
        if send_telegram(out, text):
            log.info("[%s] sent %s to Telegram", account, out.name)
    return out


def process_post(post_files, account, cfg):
    """Called by monitor.py after a post downloads. Never raises."""
    if not settings(cfg)["enabled"]:
        return
    videos = [p for p in post_files if p.suffix.lower() == ".mp4"]
    txt = next((p for p in post_files if p.suffix.lower() == ".txt"), None)
    post_caption = txt.read_text(encoding="utf-8", errors="replace").strip() if txt else ""
    for v in videos:
        try:
            process_video(v, account, cfg, post_caption)
        except Exception as e:
            log.exception("[%s] pipeline failed on %s: %s", account, v.name, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    print(process_video(sys.argv[1], sys.argv[2].lstrip("@"), cfg, send="--no-send" not in sys.argv))
