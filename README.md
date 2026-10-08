# Smart Video Highlight & Insight Pipeline

One command turns a video into a 3-shot highlight reel plus a structured per-shot summary: FFmpeg, OpenCV, YOLOv8n and a small VLM (SmolVLM-256M).

**Input video:** `https://www.pexels.com/video/a-soccer-team-training-in-a-park-10719816/`. 5 min 6 s, 3840x2160, 29.97 fps, H.264, **no audio track**.

## Setup & run
Docker is the only requirement (weights are baked into the image):
```bash
docker build -t video-highlights .
docker run --rm --user $(id -u):$(id -g) -v "$PWD":/data video-highlights \
    --input /data/video.mp4 --out /data/results
```
Without Docker: install ffmpeg, `pip install -r requirements.txt`, then `python main.py --input video.mp4 --out results/`.
Flags: `--hist-thr --min-shot --max-shot --yolo --vlm-model --top-k --no-vlm`.

**Outputs:** `results/summary.json` (per shot: timestamps, motion, object stats, caption, VLM verdict, score, selected), `highlights.mp4`, `keyframes/` (annotated), `frames/` (2 FPS, regenerated each run), plus `video_720p.mp4` and `audio_16k_mono.wav` when audio exists.

## Pipeline & design choices
1. **FFmpeg** (`media.py`): ffprobe JSON (duration, fps, resolution, codec); 720p H.264 resize; 2 FPS JPG frames; 16 kHz mono PCM WAV (skipped cleanly if no audio).
2. **Shots + motion** (`shots.py`): H-S histogram (50x60 bins, 320 px wide); cut if Bhattacharyya distance between consecutive frames **> 0.5** and shot **>= 1 s**. Motion = mean Farneback flow magnitude at ~5 FPS, never computed across a cut. Shots longer than **8 s** are split evenly, so single-take clips still give enough candidates.
3. **Detection** (`detect.py`): YOLOv8n, conf 0.3, batches of 16, on the 2 FPS frames. Per shot: average people and top-3 classes. Keyframe = frame with most people (tie: nearest shot centre), boxes drawn with OpenCV.
4. **VLM** (`vlm.py`): SmolVLM-256M on CPU, one 512 px view per keyframe. Prompt asks for `{caption, exciting, reason}`, assistant turn prefilled with `{"caption": "`. Validated; on failure: (a) regex-salvage unclosed JSON, (b) guided field-by-field JSON generation, (c) two plain-text calls + yes/no regex, (d) safe default (`exciting=false`). Each shot stores `vlm_source`.
5. **Highlights** (`highlights.py`): `score = 0.45*motion/max + 0.25*people/max + 0.30*VLM_yes` (each term in [0,1]; motion weighted highest as the most objective signal, VLM lowest as the noisiest). Top 3 shots, max 6 s each, cut and re-encoded uniformly, caption burned with `drawtext` (`textfile=`, boxed), joined with the concat demuxer, chronological order, audio kept if present.

## Runtime
CPU only, Docker on MacBook Pro (`<APPLE M3 Pro 18 GB RAM>`), 306 s 4K clip: **218 s total**: resize 40 s, frames 4.7 s, shots+motion 12.9 s, YOLO 40.5 s, VLM ~114 s (39 keyframes), highlights 1.3 s.

## Known limitations
- This clip is one continuous take: the 39 "shots" are 8 s splits, not real cuts, so the top 3 can be neighbouring segments. The hist threshold (0.5) is untuned and may miss dissolves or fire on flashes.
- Global flow magnitude mixes camera motion with subject motion.
- A 256M VLM gives generic captions and a weak yes/no, hence its low weight; Qwen2.5-VL-3B would be better but slower on CPU.
- 2 FPS detection can miss brief events; people count is per frame, not unique (no tracking).
- Source clip has no audio, so the reel is silent and the loudness bonus signal is not used.
- Image is CPU-only; not optimised for GPU.