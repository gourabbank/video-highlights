"""Highlight scoring + ffmpeg cut / drawtext / concat."""
import textwrap
from pathlib import Path

from .media import run

W_MOTION, W_PEOPLE, W_VLM = 0.45, 0.25, 0.30
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]


def score_shots(shots):
    """score = 0.45*motion/max + 0.25*people/max + 0.30*VLM_yes  (each term in [0,1])."""
    mm = max((s["motion_score"] for s in shots), default=0) or 1
    pm = max((s["avg_people"] for s in shots), default=0) or 1
    for s in shots:
        s["score"] = round(W_MOTION * s["motion_score"] / mm
                           + W_PEOPLE * s["avg_people"] / pm
                           + W_VLM * (1.0 if s["exciting"] else 0.0), 4)
    return shots


def _esc(p):
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_highlights(video, shots, out_path, work_dir, top_k=3, max_clip=6.0, has_audio=True):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    top = sorted(sorted(shots, key=lambda s: s["score"], reverse=True)[:top_k],
                 key=lambda s: s["start"])  # keep chronological order in the reel
    font = next((f for f in FONTS if Path(f).exists()), None)
    clips = []
    for s in top:
        s["selected"] = True
        dur = min(s["end"] - s["start"], max_clip)
        clip = work / f"clip_{s['shot_id']:02d}.mp4"
        vf = "scale=-2:720"
        cap = s.get("caption", "")
        if cap:
            tf = work / f"caption_{s['shot_id']:02d}.txt"
            tf.write_text(textwrap.fill(cap, 55), encoding="utf-8")
            vf += (f",drawtext=textfile='{_esc(tf)}'"
                   + (f":fontfile='{_esc(font)}'" if font else "")
                   + ":fontcolor=white:fontsize=26:box=1:boxcolor=black@0.6:boxborderw=10"
                     ":x=(w-text_w)/2:y=h-text_h-40")
        cmd = ["ffmpeg", "-y", "-ss", str(s["start"]), "-i", str(video), "-t", f"{dur:.2f}",
               "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-ar", "44100", "-ac", "2"] if has_audio else ["-an"]
        run(cmd + [str(clip)])
        clips.append(clip)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)])
    return top
