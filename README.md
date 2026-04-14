# rPPG Dashboard — Webcam Vital Signs in Low Light

Remote Photoplethysmography (rPPG) pipeline using the POS algorithm with
low-light enhancements. Works in VS Code with or without a webcam.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python server.py
```
Or press **F5** in VS Code with the "Run rPPG Dashboard Server" launch config.

### 3. Open the dashboard
Open `dashboard.html` with the **Live Preview** extension in VS Code,
or navigate to `http://localhost:5500/dashboard.html` in your browser.

If no webcam is available, the server runs in **simulation mode** automatically.

---

## Files

| File | Purpose |
|------|---------|
| `rppg_engine.py` | Core pipeline: enhance → ROI → POS → filter → BPM |
| `server.py` | SSE server that streams metrics to the browser |
| `dashboard.html` | Live dashboard with signal charts |
| `.vscode/launch.json` | VS Code run configurations |

---

## Low-Light Methods Implemented

### Signal Processing
- **POS algorithm** — Plane-Orthogonal-to-Skin; robust to illumination drift
- **Butterworth bandpass** — 0.7–3.5 Hz (42–210 BPM), order 4
- **Welch's method** — PSD-based heart rate frequency estimation
- **Temporal smoothing** — median over last 30 BPM estimates

### Image Enhancement
- **CLAHE** — Contrast Limited Adaptive Histogram Equalization on L channel
- **Bilateral filter** — noise reduction preserving skin edge structure
- **Camera lock** — auto-exposure and white balance disabled to prevent drift

### ROI Strategy
- **MediaPipe Face Mesh** — forehead + bilateral cheek patches
- **Haar cascade fallback** — upper-third face region if MediaPipe unavailable

---

## Keyboard Shortcuts (OpenCV window)
| Key | Action |
|-----|--------|
| `e` | Toggle CLAHE + bilateral enhancement |
| `q` | Quit |

---

## Tuning for Darker Environments

```python
# In rppg_engine.py — LowLightEnhancer.enhance()
self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  # More aggressive

# In RPPGPipeline._configure_camera()
cap.set(cv2.CAP_PROP_EXPOSURE, -4)   # Less negative = brighter (try -3 to -7)
cap.set(cv2.CAP_PROP_GAIN, 80)       # Increase gain (trade-off: more noise)
```

## Upgrading to Deep Learning

For best low-light performance, replace the POS block with a pretrained model:
```bash
git clone https://github.com/ubicomplab/rPPG-Toolbox
# Models: PhysNet, EfficientPhys, BigSmall
```
