"""YOLO detection on extracted frames (batched), per-shot aggregation, annotated keyframes."""
from collections import Counter
from pathlib import Path

import cv2


def run_detector(frame_paths, fps=2, model_name="yolov8n.pt", conf=0.3, batch=16):
    from ultralytics import YOLO
    model = YOLO(model_name)
    frames = []
    for i in range(0, len(frame_paths), batch):
        chunk = [str(p) for p in frame_paths[i:i + batch]]
        for j, r in enumerate(model.predict(chunk, conf=conf, verbose=False)):
            dets = [(r.names[int(c)], float(s), [int(v) for v in b])
                    for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())]
            idx = i + j
            frames.append({"t": idx / fps, "path": chunk[j], "dets": dets})
    return frames


def aggregate(shots, frames, keyframe_dir):
    keyframe_dir = Path(keyframe_dir)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    for s in shots:
        fs = [f for f in frames if s["start"] <= f["t"] < s["end"]]
        if not fs:  # very short shot: take frame nearest the midpoint
            mid = (s["start"] + s["end"]) / 2
            fs = [min(frames, key=lambda f: abs(f["t"] - mid))]
        people = [sum(1 for d in f["dets"] if d[0] == "person") for f in fs]
        classes = Counter(d[0] for f in fs for d in f["dets"])
        s["avg_people"] = round(sum(people) / len(people), 2)
        s["top_classes"] = [{"class": c, "count": n} for c, n in classes.most_common(3)]
        mid = (s["start"] + s["end"]) / 2
        key = max(zip(fs, people), key=lambda x: (x[1], -abs(x[0]["t"] - mid)))[0]
        img = cv2.imread(key["path"])
        for name, score, (x1, y1, x2, y2) in key["dets"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{name} {score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        out = keyframe_dir / f"shot_{s['shot_id']:02d}.jpg"
        cv2.imwrite(str(out), img)
        s["keyframe"] = str(out)
        s["keyframe_time"] = round(key["t"], 2)
    return shots
