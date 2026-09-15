

# ──────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
import time
import datetime
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  MEDIAPIPE EYE LANDMARK INDICES
#  Source: e-candeloro/Driver-State-Detection
#  These are the 6 points around each eye in the 478-point mesh.
# ──────────────────────────────────────────────────────────────
LEFT_EYE_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33,  160, 158, 133, 153, 144]

# ──────────────────────────────────────────────────────────────
#  3-D FACE MODEL FOR HEAD POSE ESTIMATION
#  Source: e-candeloro/Driver-State-Detection
#  Real-world coordinates (mm) of 6 key facial points.
# ──────────────────────────────────────────────────────────────
FACE_3D_POINTS = np.array([
    [  0.0,    0.0,    0.0],   # Nose tip         (landmark 1)
    [  0.0, -330.0,  -65.0],   # Chin             (landmark 152)
    [-225.0,  170.0, -135.0],  # Left eye corner  (landmark 263)
    [ 225.0,  170.0, -135.0],  # Right eye corner (landmark 33)
    [-150.0, -150.0, -125.0],  # Left mouth       (landmark 287)
    [ 150.0, -150.0, -125.0],  # Right mouth      (landmark 57)
], dtype=np.float64)

FACE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# ──────────────────────────────────────────────────────────────
#  THRESHOLDS  (tuned for classroom environment)
# ──────────────────────────────────────────────────────────────
EAR_CLOSED_THRESHOLD = 0.20   # EAR below this = eyes closed
HEAD_PITCH_LIMIT     = 20     # degrees: looking too far down
HEAD_YAW_LIMIT       = 25     # degrees: turning too far left/right
DROWSY_SECONDS       = 3.5    # seconds of closed eyes = Drowsy
HISTORY_SIZE         = 15     # frames averaged for smooth score


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _compute_ear(landmarks, eye_points, img_w, img_h):
    """
    Compute Eye Aspect Ratio (EAR) for one eye.

    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Reference:
        Soukupova & Cech (2016), "Real-Time Eye Blink Detection
        using Facial Landmarks."
    Adapted from:
        e-candeloro/Driver-State-Detection, eye_detector.py

    Parameters
    ----------
    landmarks    : MediaPipe landmark list
    eye_points   : 6 landmark indices for this eye
    img_w, img_h : frame width and height in pixels

    Returns
    -------
    float : EAR value. Higher = more open. ~0 = fully closed.
    """
    pts = []
    for idx in eye_points:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h]))

    vertical_a = np.linalg.norm(pts[1] - pts[5])
    vertical_b = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])

    if horizontal == 0:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def _compute_head_pose(landmarks, img_w, img_h):
    image_pts = []
    for idx in FACE_LANDMARK_IDS:
        lm = landmarks[idx]
        image_pts.append([lm.x * img_w, lm.y * img_h])
    image_pts = np.array(image_pts, dtype=np.float64)

    focal_len = float(img_w)
    camera_matrix = np.array([
        [focal_len, 0, img_w / 2.0],
        [0, focal_len, img_h / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, image_pts,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = float(angles[0])
    yaw   = float(angles[1])
    roll  = float(angles[2])

    pitch = max(-90.0, min(90.0, pitch))
    yaw   = max(-90.0, min(90.0, yaw))
    roll  = max(-90.0, min(90.0, roll))

    return pitch, yaw, roll


# ══════════════════════════════════════════════════════════════
#  MAIN CLASS
# ══════════════════════════════════════════════════════════════

class AttentionDetector:
    """
    Real-time student attention detector.

    Detects whether a student is Attentive, Distracted, or Drowsy
    by analysing their face through the webcam using MediaPipe.

    Quick start
    -----------
        detector = AttentionDetector()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            result = detector.process_frame(frame)
            cv2.imshow("Attention", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        detector.close()
    """

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._closed_since    = None  # timestamp when eyes first closed
        self._score_history   = []    # sliding window of raw scores

    # ──────────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Analyse one BGR frame from the camera.

        Parameters
        ----------
        frame : np.ndarray  BGR image

        Returns
        -------
        dict with keys:
            score  (float)      attention 0–100 %
            status (str)        'Attentive' | 'Distracted' | 'Drowsy' | 'No Face'
            ear    (float)      average Eye Aspect Ratio
            pitch  (float)      head pitch in degrees
            yaw    (float)      head yaw in degrees
            frame  (np.ndarray) annotated frame
        """
        empty = dict(score=0.0, status="Face Not Detected",
                    ear=0.0, pitch=0.0, yaw=0.0, frame=frame)

        if frame is None:
            return empty

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res  = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_face_landmarks:
            self._closed_since = None
            cv2.putText(frame, "Face Not Detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (150, 150, 150), 2, cv2.LINE_AA)
            return {**empty, "frame": frame}

        lms = res.multi_face_landmarks[0].landmark

        ear = (_compute_ear(lms, LEFT_EYE_POINTS,  w, h) +
            _compute_ear(lms, RIGHT_EYE_POINTS, w, h)) / 2.0

        pitch, yaw, roll = _compute_head_pose(lms, w, h)

        eyes_closed = ear   < EAR_CLOSED_THRESHOLD
        head_turned = (abs(pitch) > HEAD_PITCH_LIMIT or
                    abs(yaw)   > HEAD_YAW_LIMIT)

        now = time.time()
        if eyes_closed:
            if self._closed_since is None:
                self._closed_since = now
            closed_for = now - self._closed_since
        else:
            self._closed_since = None
            closed_for = 0.0

        if   closed_for >= DROWSY_SECONDS: status = "Very Sleepy"
        elif eyes_closed or head_turned:   status = "Needs Attention"
        else:                              status = "Focused"

        s_ear   = min(ear / 0.35,  1.0)
        s_yaw   = max(0.0, 1.0 - abs(yaw)   / HEAD_YAW_LIMIT)
        s_pitch = max(0.0, 1.0 - abs(pitch) / HEAD_PITCH_LIMIT)
        raw     = (s_ear * 0.50 + s_yaw * 0.30 + s_pitch * 0.20) * 10.0

        self._score_history.append(raw)
        if len(self._score_history) > HISTORY_SIZE:
            self._score_history.pop(0)
        score = round(float(np.mean(self._score_history)), 1)

        self._draw(frame, status, score, ear, pitch, yaw)

        return dict(score=score, status=status,
                    ear=round(ear, 3),
                    pitch=round(pitch, 1),
                    yaw=round(yaw, 1),
                    frame=frame)
    # ──────────────────────────────────────────────────────────
    def reset(self):
        """Reset state between sessions."""
        self._closed_since  = None
        self._score_history = []

    def close(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()

    # ──────────────────────────────────────────────────────────
    def _draw(self, frame, status, score, ear, pitch, yaw):
        """Draw status overlay on the frame."""
        h = frame.shape[0]
        colour = {"Focused": (0, 210, 0),
                "Needs Attention": (0, 140, 255),
                "Very Sleepy": (0, 0, 220)}.get(status, (180, 180, 180))

        cv2.rectangle(frame, (0, 0), (430, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"{status}   Score: {score:.1f}/10",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, colour, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, h - 55), (330, h), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"EAR:{ear:.2f}  Pitch:{pitch:.1f}  Yaw:{yaw:.1f}",
                    (8, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  DATABASE  FUNCTIONS
# ══════════════════════════════════════════════════════════════

def save_attention_record(student_id, score, status, duration_seconds=0):
    """
    Save an attention record to the SQLite database.

    Parameters
    ----------
    student_id       : int
    score            : float  (0–100)
    status           : str
    duration_seconds : int

    Returns True on success.
    """
    try:
        from utils.db_utils import get_connection
        conn = get_connection()
        conn.execute(
            """INSERT INTO attention_records
               (student_id, session_date, attention_score,
                status, duration_seconds, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (student_id,
             datetime.date.today().isoformat(),
             round(score, 1),
             status,
             int(duration_seconds),
             datetime.datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        logger.error(f"save_attention_record: {exc}")
        return False


def get_attention_summary(date=None):
    """
    Return average attention score per student.

    Parameters
    ----------
    date : str | None   'YYYY-MM-DD'  –  None means all dates

    Returns
    -------
    list[dict]  keys: id, roll_no, name, avg_score, last_status
    """
    try:
        from utils.db_utils import get_connection
        conn = get_connection()
        if date:
            rows = conn.execute(
                """SELECT s.id, s.roll_no, s.name,
                          ROUND(AVG(ar.attention_score),1) AS avg_score,
                          ar.status AS last_status
                   FROM students s
                   LEFT JOIN attention_records ar
                          ON s.id = ar.student_id
                         AND ar.session_date = ?
                   GROUP BY s.id ORDER BY s.roll_no""",
                (date,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT s.id, s.roll_no, s.name,
                          ROUND(AVG(ar.attention_score),1) AS avg_score,
                          ar.status AS last_status
                   FROM students s
                   LEFT JOIN attention_records ar ON s.id = ar.student_id
                   GROUP BY s.id ORDER BY s.roll_no"""
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error(f"get_attention_summary: {exc}")
        return []