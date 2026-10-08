"""Shot detection (HSV histogram distance) + motion scoring (Farneback optical flow)."""
import cv2
import numpy as np


def detect_shots(video, hist_thr=0.5, min_len=1.0, max_len=8.0,
                 flow_fps=5, proc_width=320):
    """
    Cut = Bhattacharyya distance between consecutive H-S histograms > hist_thr
    (0 = identical, 1 = disjoint), with a minimum shot length to suppress flicker.
    Motion = mean Farneback flow magnitude (px, at proc_width) sampled at ~flow_fps;
    flow is never computed across a cut. Shots longer than max_len are split into
    equal sub-shots so continuous single-take clips still yield several candidates.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / flow_fps))
    cuts = [0.0]
    flow = []  # (time, magnitude)
    prev_hist = prev_gray = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (proc_width, int(h * proc_width / w)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = (hist / max(hist.sum(), 1e-9)).astype(np.float32)
        t = i / fps
        is_cut = False
        if prev_hist is not None:
            d = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > hist_thr and (t - cuts[-1]) >= min_len:
                cuts.append(t)
                is_cut = True
        prev_hist = hist
        if i % step == 0:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and not is_cut:
                f = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                 0.5, 3, 15, 3, 5, 1.2, 0)
                flow.append((t, float(np.linalg.norm(f, axis=2).mean())))
            prev_gray = gray
        elif is_cut:
            prev_gray = None  # re-seed on next sampled frame
        i += 1
    cap.release()
    duration = i / fps

    bounds = cuts + [duration]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = max(1, int(np.ceil((b - a) / max_len)))
        edges = np.linspace(a, b, n + 1)
        for s, e in zip(edges[:-1], edges[1:]):
            shots.append({"start": round(float(s), 2), "end": round(float(e), 2)})
    for k, s in enumerate(shots):
        s["shot_id"] = k
        m = [v for t, v in flow if s["start"] <= t < s["end"]]
        s["motion_score"] = round(float(np.mean(m)), 4) if m else 0.0
    print(f"[shots] {len(cuts)} cut-shots -> {len(shots)} shots after max_len split")
    return shots
