"""
rPPG Engine — Remote Photoplethysmography via webcam
Implements POS algorithm with low-light enhancements
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
from collections import deque
import time


class LowLightEnhancer:
    """Pre-processing pipeline for low-light video frames."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE + bilateral filter for low-light improvement."""
        # Bilateral filter: smooth noise but preserve skin edges
        denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # CLAHE on the L channel of LAB colorspace
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced


class FaceROIExtractor:
    """Extract forehead + cheek ROIs using MediaPipe Face Mesh."""
    def __init__(self):
        # Always initialize Haar cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        try:
            from mediapipe.python.solutions.face_mesh import FaceMesh
            self.face_mesh = FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            print("[FaceROI] MediaPipe loaded OK")
        except Exception as e:
            print(f"[Warning] MediaPipe unavailable ({e}), using Haar cascade fallback.")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.use_mediapipe = False

    def get_roi_means(self, frame: np.ndarray):
        """
        Returns RGB mean values from forehead + cheek patches.
        Falls back to full face region if landmarks unavailable.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_mediapipe:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Forehead: between eyebrows and hairline (approx landmark indices)
                forehead_pts = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                                361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                                162, 21, 54, 103, 67, 109]

                def landmark_mean(indices, frame):
                    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices])
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                    pixels = frame[mask == 255]
                    return pixels.mean(axis=0) if len(pixels) > 0 else None

                # Key ROI landmark groups
                forehead_indices = [10, 67, 109, 338, 297, 332]
                left_cheek_indices = [234, 93, 132, 58, 172]
                right_cheek_indices = [454, 323, 361, 288, 397]

                regions = []
                for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
                    mean = landmark_mean(indices, rgb_frame)
                    if mean is not None:
                        regions.append(mean)

                if regions:
                    return np.mean(regions, axis=0)  # [R, G, B]

        # Haar cascade fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            # Use upper-middle third of face (forehead region)
            forehead = rgb_frame[y:y + fh // 3, x + fw // 4: x + 3 * fw // 4]
            if forehead.size > 0:
                return forehead.reshape(-1, 3).mean(axis=0)

        return None


class POSFilter:
    """
    Plane-Orthogonal-to-Skin (POS) rPPG algorithm.
    Wang et al., 2017 — robust to illumination changes.
    """

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.rgb_buffer = deque(maxlen=window_size)

    def update(self, rgb_mean: np.ndarray) -> float | None:
        self.rgb_buffer.append(rgb_mean)
        if len(self.rgb_buffer) < self.window_size:
            return None

        C = np.array(self.rgb_buffer, dtype=float)  # (N, 3)

        # Normalize by temporal mean
        mean_C = C.mean(axis=0)
        if np.any(mean_C == 0):
            return None
        Cn = C / mean_C  # (N, 3)

        # POS projection matrix
        S_matrix = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
        P = S_matrix @ Cn.T  # (2, N)

        std0 = np.std(P[0])
        std1 = np.std(P[1])
        if std1 == 0:
            return None

        alpha = std0 / std1
        pulse = P[0] + alpha * P[1]
        return float(pulse[-1])


class BandpassFilter:
    """Butterworth bandpass for 42–210 BPM (0.7–3.5 Hz)."""

    def __init__(self, fps: float = 30.0, low_hz: float = 0.7, high_hz: float = 3.5, order: int = 4):
        self.fps = fps
        self.low = low_hz
        self.high = high_hz
        self.order = order

    def filter(self, signal: np.ndarray) -> np.ndarray:
        nyq = self.fps / 2.0
        low = self.low / nyq
        high = self.high / nyq
        if high >= 1.0:
            high = 0.99
        b, a = butter(self.order, [low, high], btype='band')
        if len(signal) < 28:
            return signal
        try:
            return filtfilt(b, a, signal)
        except ValueError:
            return signal


class HeartRateEstimator:
    """Estimate BPM from filtered rPPG signal using Welch's method."""

    def __init__(self, fps: float = 30.0, min_bpm: float = 42, max_bpm: float = 210):
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def estimate(self, signal: np.ndarray) -> float | None:
        if len(signal) < int(self.fps * 3):  # Need at least 3 seconds
            return None

        freqs, psd = welch(signal, fs=self.fps, nperseg=min(len(signal), int(self.fps * 4)))

        # Restrict to valid HR frequency range
        min_hz = self.min_bpm / 60.0
        max_hz = self.max_bpm / 60.0
        mask = (freqs >= min_hz) & (freqs <= max_hz)

        if not np.any(mask):
            return None

        peak_freq = freqs[mask][np.argmax(psd[mask])]
        return round(peak_freq * 60.0, 1)


class RPPGPipeline:
    """Full rPPG pipeline: capture → enhance → ROI → POS → filter → BPM."""

    def __init__(self, camera_index: int = 0, fps_target: float = 30.0, buffer_seconds: int = 10):
        self.fps_target = fps_target
        self.buffer_size = int(fps_target * buffer_seconds)

        self.enhancer = LowLightEnhancer()
        self.roi_extractor = FaceROIExtractor()
        self.pos_filter = POSFilter(window_size=32)
        self.bp_filter = BandpassFilter(fps=fps_target)
        self.hr_estimator = HeartRateEstimator(fps=fps_target)

        self.raw_signal = deque(maxlen=self.buffer_size)
        self.filtered_signal = deque(maxlen=self.buffer_size)
        self.bpm_history = deque(maxlen=30)

        self.cap = cv2.VideoCapture(camera_index)
        self._configure_camera()

        self.frame_count = 0
        self.last_bpm = None
        self.start_time = time.time()

    def _configure_camera(self):
        """Lock camera settings to prevent auto-exposure drift."""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        # Disable auto-exposure (critical for rPPG stability)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # Lock white balance
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        print(f"[Camera] FPS: {self.cap.get(cv2.CAP_PROP_FPS):.0f} | "
              f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    def process_frame(self, frame: np.ndarray):
        """Process one frame through the pipeline. Returns annotated frame + metrics."""
        enhanced = self.enhancer.enhance(frame)
        rgb_mean = self.roi_extractor.get_roi_means(enhanced)

        metrics = {
            "bpm": None,
            "signal_quality": 0.0,
            "face_detected": rgb_mean is not None,
            "frame": frame.copy(),
        }

        if rgb_mean is not None:
            pos_val = self.pos_filter.update(rgb_mean)
            if pos_val is not None:
                self.raw_signal.append(pos_val)

                if len(self.raw_signal) >= 32:
                    sig_arr = np.array(self.raw_signal)
                    filtered = self.bp_filter.filter(sig_arr)
                    self.filtered_signal.extend(filtered[-1:])

                    # Signal quality: ratio of AC to DC component
                    if len(filtered) > 1:
                        ac_power = np.std(filtered)
                        dc_level = np.mean(rgb_mean)
                        metrics["signal_quality"] = min(1.0, ac_power / (dc_level + 1e-6) * 500)

                    # BPM estimation every 15 frames
                    if self.frame_count % 15 == 0 and len(self.filtered_signal) > 30:
                        bpm = self.hr_estimator.estimate(np.array(self.filtered_signal))
                        if bpm:
                            self.bpm_history.append(bpm)
                            # Smoothed BPM
                            self.last_bpm = round(np.median(list(self.bpm_history)), 1)

        metrics["bpm"] = self.last_bpm
        metrics["raw_signal"] = list(self.raw_signal)[-60:]
        metrics["filtered_signal"] = list(self.filtered_signal)[-60:]

        self.frame_count += 1
        return metrics

    def get_annotated_frame(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw BPM overlay and status indicators on frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Status bar background
        cv2.rectangle(out, (0, 0), (w, 50), (0, 0, 0), -1)

        # Face detection indicator
        dot_color = (0, 255, 80) if metrics["face_detected"] else (0, 80, 255)
        cv2.circle(out, (20, 25), 8, dot_color, -1)

        face_text = "Face detected" if metrics["face_detected"] else "No face"
        cv2.putText(out, face_text, (36, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # BPM display
        if metrics["bpm"]:
            bpm_str = f"{metrics['bpm']:.0f} BPM"
            cv2.putText(out, bpm_str, (w - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 180), 2)
        else:
            cv2.putText(out, "Calibrating...", (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 80), 1)

        # Signal quality bar
        sq = metrics.get("signal_quality", 0)
        bar_w = int(120 * min(sq, 1.0))
        cv2.rectangle(out, (10, h - 20), (130, h - 8), (60, 60, 60), -1)
        bar_color = (0, 200, 100) if sq > 0.3 else (0, 120, 255)
        cv2.rectangle(out, (10, h - 20), (10 + bar_w, h - 8), bar_color, -1)
        cv2.putText(out, "Signal quality", (10, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        return out

    def run(self):
        """Main capture loop with live OpenCV display."""
        print("\n[rPPG Dashboard] Press 'q' to quit, 'e' to toggle enhancement\n")
        show_enhanced = True

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Error] Could not read frame.")
                break

            metrics = self.process_frame(frame)
            display = self.get_annotated_frame(
                self.enhancer.enhance(frame) if show_enhanced else frame,
                metrics
            )

            cv2.imshow("rPPG Dashboard — low-light vitals", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                show_enhanced = not show_enhanced
                mode = "ON" if show_enhanced else "OFF"
                print(f"[Enhancement] {mode}")

            if self.frame_count % 30 == 0 and metrics["bpm"]:
                elapsed = time.time() - self.start_time
                print(f"[{elapsed:.0f}s] BPM: {metrics['bpm']} | "
                      f"Quality: {metrics['signal_quality']:.2f} | "
                      f"Frames: {self.frame_count}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = RPPGPipeline(camera_index=0, fps_target=30.0)
    pipeline.run()

