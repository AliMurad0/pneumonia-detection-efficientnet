# ══════════════════════════════════════════════════════════════════════
# config.py — Edit these settings before running app.py
# ══════════════════════════════════════════════════════════════════════

import os

# ── Step 1: Put your model file inside the 'models/' folder ───────────
# Supported formats: .keras  or  .h5
# Example: models/efficientnetb3_pneumonia_final.keras
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')

# ── Step 2: Set the decision threshold ────────────────────────────────
# From your V1 results: 0.8714
# From V3 results: check model_config_v3.json → evaluation → optimal_threshold
THRESHOLD  = 0.8714

# ── Step 3: Image size ────────────────────────────────────────────────
# EfficientNetB3 = 300   |   EfficientNetB4 = 380
# Leave as None to auto-detect from the model's input shape
IMG_SIZE   = None   # auto-detect

# ── Step 4: Dashboard settings ────────────────────────────────────────
HOST       = '127.0.0.1'   # localhost only (safe)
PORT       = 7860           # default Gradio port
SHARE      = False          # True = creates a public ngrok link (optional)

# ── Advanced: TTA passes (set 1 for fastest, 7+ for better accuracy) ──
TTA_PASSES = 1              # 1 = single fast inference, no TTA
