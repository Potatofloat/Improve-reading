"""
rPPG WebSocket server — streams live metrics to the browser dashboard
Run with: python server.py
Then open dashboard.html in your browser (or VS Code Live Preview)
"""

import json
import time
import threading
import math
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# Try to import real pipeline; fall back to simulation for demo
try:
    import cv2
    import numpy as np
    from rppg_engine import RPPGPipeline
    REAL_CAMERA = True
except ImportError:
    REAL_CAMERA = False
    print("[Info] OpenCV not found — running in simulation mode")

# WebSocket-lite via SSE (Server-Sent Events) — no extra deps needed
class SSEHandler(SimpleHTTPRequestHandler):
    """Serves dashboard.html and streams rPPG metrics via SSE."""

    shared_metrics = {
        "bpm": None,
        "signal_quality": 0.0,
        "face_detected": False,
        "raw_signal": [],
        "filtered_signal": [],
        "timestamp": 0,
    }
    lock = threading.Lock()

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with SSEHandler.lock:
                        data = json.dumps(SSEHandler.shared_metrics)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def simulate_rppg():
    """Generate synthetic rPPG signal for demo/testing without a camera."""
    t = 0
    bpm_base = 72.0
    raw_buf = []
    filt_buf = []
    quality = 0.0
    warmup = 0

    while True:
        # Simulated heart rate variation
        bpm = bpm_base + 3 * math.sin(t / 30)
        hz = bpm / 60.0
        raw_val = math.sin(2 * math.pi * hz * t / 30) + random.gauss(0, 0.15)
        filt_val = math.sin(2 * math.pi * hz * t / 30) * 0.9

        raw_buf.append(raw_val)
        filt_buf.append(filt_val)
        if len(raw_buf) > 120:
            raw_buf.pop(0)
        if len(filt_buf) > 120:
            filt_buf.pop(0)

        warmup = min(warmup + 1, 90)
        quality = min(1.0, warmup / 90 * (0.6 + 0.2 * abs(math.sin(t / 50))))

        with SSEHandler.lock:
            SSEHandler.shared_metrics = {
                "bpm": round(bpm, 1) if warmup > 60 else None,
                "signal_quality": round(quality, 3),
                "face_detected": True,
                "raw_signal": [round(v, 4) for v in raw_buf[-60:]],
                "filtered_signal": [round(v, 4) for v in filt_buf[-60:]],
                "timestamp": round(time.time() * 1000),
                "mode": "simulation"
            }

        t += 1
        time.sleep(1 / 30)


def run_real_camera():
    """Run the actual rPPG pipeline and push metrics to SSE."""
    pipeline = RPPGPipeline(camera_index=0, fps_target=30.0)

    while True:
        ret, frame = pipeline.cap.read()
        if not ret:
            break

        metrics = pipeline.process_frame(frame)

        with SSEHandler.lock:
            SSEHandler.shared_metrics = {
                "bpm": metrics.get("bpm"),
                "signal_quality": round(metrics.get("signal_quality", 0), 3),
                "face_detected": metrics.get("face_detected", False),
                "raw_signal": [round(v, 4) for v in metrics.get("raw_signal", [])],
                "filtered_signal": [round(v, 4) for v in metrics.get("filtered_signal", [])],
                "timestamp": round(time.time() * 1000),
                "mode": "live"
            }

    pipeline.cap.release()


if __name__ == "__main__":
    PORT = 5500

    # Start data producer in background thread
    producer = threading.Thread(
        target=run_real_camera if REAL_CAMERA else simulate_rppg,
        daemon=True
    )
    producer.start()

    # Serve from dashboard directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with socketserver.TCPServer(("", PORT), SSEHandler) as httpd:
        print(f"\n[rPPG Server] Running at http://localhost:{PORT}")
        print(f"[rPPG Server] Open dashboard.html in VS Code Live Preview")
        print(f"[rPPG Server] Mode: {'Live camera' if REAL_CAMERA else 'Simulation'}\n")
        httpd.serve_forever()
