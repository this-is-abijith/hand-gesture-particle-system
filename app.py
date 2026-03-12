# app.py — Hand Gesture Particle System
# MediaPipe 0.10+ | 8 precise gestures with debounce + confidence

from flask import Flask, Response, jsonify, render_template
import cv2, threading, time, math, collections
import mediapipe as mp

app = Flask(__name__)

# ════════════════════════════════════════════════════════
#  MEDIAPIPE SETUP  (0.10+ Tasks API)
# ════════════════════════════════════════════════════════
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import urllib.request, os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand landmark model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")

_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55,
    running_mode=mp_vision.RunningMode.VIDEO,
)
hand_detector = mp_vision.HandLandmarker.create_from_options(_options)
print(f"✅ MediaPipe {mp.__version__} ready")


# ════════════════════════════════════════════════════════
#  GESTURE ENGINE  — precise, 8 gestures
# ════════════════════════════════════════════════════════
class GestureEngine:
    """
    Landmark reference (mirrored feed):
      0=wrist  1-4=thumb  5-8=index  9-12=middle  13-16=ring  17-20=pinky
      TIP ids: thumb=4, index=8, middle=12, ring=16, pinky=20
      PIP ids: thumb=3, index=6, middle=10, ring=14, pinky=18
      MCP ids: index=5, middle=9, ring=13, pinky=17
    """

    # ── Precision helpers ────────────────────────────────

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def _angle(self, a, b, c):
        """Angle at point b formed by a-b-c (degrees)"""
        v1 = (a.x - b.x, a.y - b.y)
        v2 = (c.x - b.x, c.y - b.y)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag = (math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)) or 1e-6
        return math.degrees(math.acos(max(-1, min(1, dot / mag))))

    def _finger_extended(self, lm, tip, pip, mcp):
        """
        A finger is extended when:
        1. Tip is above PIP (Y-axis, image coords)
        2. The tip-pip-mcp angle is > 150° (finger is straight)
        Using BOTH conditions avoids false positives from bent fingers.
        """
        tip_above_pip = lm[tip].y < lm[pip].y
        straightness  = self._angle(lm[tip], lm[pip], lm[mcp])
        return tip_above_pip and straightness > 140

    def _thumb_extended(self, lm):
        """
        Thumb is trickier — it moves sideways.
        Use distance from thumb tip to index MCP vs thumb base width.
        """
        tip_to_index_mcp = self._dist(lm[4], lm[5])
        thumb_base_width = self._dist(lm[1], lm[2])
        return tip_to_index_mcp > thumb_base_width * 1.8

    def finger_states(self, lm):
        """
        Returns dict: thumb, index, middle, ring, pinky → True/False
        Uses angle-based detection for precision.
        """
        return {
            "thumb":  self._thumb_extended(lm),
            "index":  self._finger_extended(lm, 8,  6,  5),
            "middle": self._finger_extended(lm, 12, 10, 9),
            "ring":   self._finger_extended(lm, 16, 14, 13),
            "pinky":  self._finger_extended(lm, 20, 18, 17),
        }

    def classify(self, lm):
        """
        8 gestures — ordered from most-specific to least-specific
        to avoid misclassification.
        """
        f = self.finger_states(lm)
        t = f["thumb"]
        i = f["index"]
        m = f["middle"]
        r = f["ring"]
        p = f["pinky"]

        pinch_d    = self._dist(lm[4], lm[8])
        mid_pinch  = self._dist(lm[4], lm[12])

        # ── 1. PINCH — thumb + index very close, others don't matter
        if pinch_d < 0.055:
            return "PINCH"

        # ── 2. DOUBLE PINCH — thumb + middle close (alt pinch)
        if mid_pinch < 0.055:
            return "DOUBLE_PINCH"

        # ── 3. FIST — all fingers curled
        if not i and not m and not r and not p:
            return "FIST"

        # ── 4. OPEN — all 4 fingers up
        if i and m and r and p:
            return "OPEN"

        # ── 5. POINT — only index up
        if i and not m and not r and not p:
            return "POINT"

        # ── 6. PEACE / SCISSORS — index + middle up, rest down
        if i and m and not r and not p:
            return "PEACE"

        # ── 7. THREE — index + middle + ring up
        if i and m and r and not p:
            return "THREE"

        # ── 8. PINKY — only pinky up (call me gesture)
        if not i and not m and not r and p:
            return "PINKY"

        # ── 9. THUMBS UP — only thumb extended
        if t and not i and not m and not r and not p:
            return "THUMBS_UP"

        # ── 10. GUN — thumb + index up, rest down
        if t and i and not m and not r and not p:
            return "GUN"

        return "NONE"

    def palm_center(self, lm, w, h):
        """Center of palm = average of wrist + 4 MCP joints"""
        xs = [lm[i].x for i in [0, 5, 9, 13, 17]]
        ys = [lm[i].y for i in [0, 5, 9, 13, 17]]
        return int(sum(xs)/5 * w), int(sum(ys)/5 * h)

    def index_tip(self, lm, w, h):
        return int(lm[8].x * w), int(lm[8].y * h)

    def pinch_dist(self, lm):
        return self._dist(lm[4], lm[8])

    def hand_openness(self, lm):
        """0.0=fist  1.0=fully open — used for OPEN gesture intensity"""
        tips = [8, 12, 16, 20]
        mcps = [5,  9, 13, 17]
        avg  = sum(self._dist(lm[t], lm[m]) for t, m in zip(tips, mcps)) / 4
        return min(1.0, avg / 0.25)


# ── Debounce buffer — prevents jitter between gestures ──
class GestureDebouncer:
    """
    Requires a gesture to appear N consecutive frames
    before reporting it as confirmed.
    """
    def __init__(self, window=5, threshold=4):
        self.window    = window
        self.threshold = threshold
        self.buffer    = collections.deque(maxlen=window)
        self.current   = "NONE"

    def update(self, raw_gesture):
        self.buffer.append(raw_gesture)
        counts = collections.Counter(self.buffer)
        best, count = counts.most_common(1)[0]
        if count >= self.threshold:
            self.current = best
        return self.current


gesture_engine   = GestureEngine()
gesture_debounce = GestureDebouncer(window=6, threshold=4)


# ════════════════════════════════════════════════════════
#  SHARED STATE
# ════════════════════════════════════════════════════════
state = {
    "gesture":   "NONE",
    "raw":       "NONE",
    "x":         0.5,
    "y":         0.5,
    "pinch":     1.0,
    "openness":  0.0,
    "frame":     None,
}
lock = threading.Lock()


# ════════════════════════════════════════════════════════
#  LANDMARK DRAWING
# ════════════════════════════════════════════════════════
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]

GESTURE_COLORS = {
    "POINT":       (200, 200, 200),
    "FIST":        (180, 180, 180),
    "OPEN":        (220, 220, 220),
    "PINCH":       (160, 160, 160),
    "DOUBLE_PINCH":(160, 160, 160),
    "PEACE":       (200, 200, 200),
    "THREE":       (200, 200, 200),
    "PINKY":       (200, 200, 200),
    "THUMBS_UP":   (220, 220, 220),
    "GUN":         (200, 200, 200),
    "NONE":        (80,  80,  80),
}

def draw_landmarks(frame, lm, w, h, gesture):
    color = GESTURE_COLORS.get(gesture, (120, 120, 120))
    pts   = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for (px, py) in pts:
        cv2.circle(frame, (px, py), 3, color, -1, cv2.LINE_AA)
    # Label gesture on frame
    cx, cy = gesture_engine.palm_center(lm, w, h)
    cv2.putText(frame, gesture, (cx - 40, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════
#  BACKGROUND FRAME PROCESSING
# ════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)

def process_frames():
    ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts    += 33
        result = hand_detector.detect_for_video(mp_img, ts)
        lm     = result.hand_landmarks[0] if result.hand_landmarks else None

        if lm:
            raw_gesture = gesture_engine.classify(lm)
            gesture     = gesture_debounce.update(raw_gesture)
            ix, iy      = gesture_engine.index_tip(lm, w, h)
            pinch       = gesture_engine.pinch_dist(lm)
            openness    = gesture_engine.hand_openness(lm)

            draw_landmarks(frame, lm, w, h, gesture)

            with lock:
                state["gesture"]  = gesture
                state["raw"]      = raw_gesture
                state["x"]        = ix / w
                state["y"]        = iy / h
                state["pinch"]    = round(pinch, 4)
                state["openness"] = round(openness, 3)
        else:
            gesture_debounce.update("NONE")
            with lock:
                state["gesture"] = "NONE"
                state["raw"]     = "NONE"

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with lock:
            state["frame"] = buf.tobytes()

        time.sleep(0.01)


# ════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════
def generate():
    while True:
        with lock:
            frame = state.get("frame")
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture')
def get_gesture():
    with lock:
        data = {k: state[k] for k in ("gesture","raw","x","y","pinch","openness")}
    return jsonify(data)


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    print("\n🚀  Open → http://localhost:5000\n")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)