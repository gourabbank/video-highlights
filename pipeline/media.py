"""FFmpeg / ffprobe helpers: probing, 720p resize, frame + audio extraction."""
import json
import subprocess
from pathlib import Path


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)])
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    info = {
        "duration_s": round(float(data["format"]["duration"]), 2),
        "fps": round(float(num) / float(den), 3),
        "width": v["width"], "height": v["height"],
        "video_codec": v["codec_name"],
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }
    print("[ffprobe] duration={duration_s}s fps={fps} res={width}x{height} codec={video_codec} audio={has_audio}".format(**info))
    return info


def resize_720p(src, dst):
    # -2 keeps aspect ratio with an even width; never upscale small inputs
    run(["ffmpeg", "-y", "-i", str(src), "-vf", "scale=-2:'min(720,ih)'",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", str(dst)])
    return dst


def extract_frames(src, frames_dir, fps=2):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(src), "-vf", f"fps={fps}", "-q:v", "2",
         str(frames_dir / "frame_%05d.jpg")])
    return sorted(frames_dir.glob("frame_*.jpg"))


def extract_audio(src, dst):
    """16 kHz mono PCM WAV. Returns None if the clip has no audio track."""
    try:
        run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(dst)])
        return dst
    except RuntimeError:
        print("[audio] no audio stream, skipping")
        return None
