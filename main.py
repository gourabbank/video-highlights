"""python main.py --input video.mp4 --out results/"""
import argparse
import json
import time
from pathlib import Path

from pipeline import media, shots as shotmod, detect, highlights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--hist-thr", type=float, default=0.5, help="shot cut threshold (Bhattacharyya)")
    ap.add_argument("--min-shot", type=float, default=1.0)
    ap.add_argument("--max-shot", type=float, default=8.0)
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--vlm-model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM (debug)")
    ap.add_argument("--top-k", type=int, default=3)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    def timed(name, fn, *args, **kw):
        t = time.time()
        r = fn(*args, **kw)
        timings[name] = round(time.time() - t, 1)
        print(f"[time] {name}: {timings[name]}s")
        return r

    info = media.probe(a.input)
    v720 = out / "video_720p.mp4"
    timed("resize_720p", media.resize_720p, a.input, v720)
    frame_paths = timed("extract_frames", media.extract_frames, v720, out / "frames", 2)
    timed("extract_audio", media.extract_audio, v720, out / "audio_16k_mono.wav")

    shots = timed("shots_motion", shotmod.detect_shots, v720, a.hist_thr, a.min_shot, a.max_shot)
    frames = timed("yolo", detect.run_detector, frame_paths, 2, a.yolo)
    detect.aggregate(shots, frames, out / "keyframes")

    if a.no_vlm:
        from pipeline.vlm import skip_result
        for s in shots:
            r = skip_result()
            s.update(caption=r["caption"], exciting=r["exciting"], reason=r["reason"], vlm_source=r["source"])
    else:
        from pipeline.vlm import VLM
        vlm = timed("vlm_load", VLM, a.vlm_model)
        t = time.time()
        for s in shots:
            r = vlm.analyze(s["keyframe"])
            s.update(caption=r["caption"], exciting=r["exciting"], reason=r["reason"], vlm_source=r["source"])
        timings["vlm_inference"] = round(time.time() - t, 1)

    highlights.score_shots(shots)
    for s in shots:
        s.setdefault("selected", False)
    timed("highlights", highlights.build_highlights, v720, shots, out / "highlights.mp4",
          out / "_tmp", a.top_k, 6.0, info["has_audio"])

    summary = {"video": info, "timings_s": timings,
               "score_formula": "0.45*motion/max_motion + 0.25*avg_people/max_people + 0.30*vlm_yes",
               "shots": shots}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {out/'summary.json'} | {out/'highlights.mp4'} | total {sum(timings.values()):.1f}s")


if __name__ == "__main__":
    main()
