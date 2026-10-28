# ══════════════════════════════════════════════════════════════════════
# model_loader.py — Finds, loads, and manages the TF model
# ══════════════════════════════════════════════════════════════════════

import os, json
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2
from config import MODEL_DIR, THRESHOLD, IMG_SIZE


class ModelLoader:
    def __init__(self):
        self.model      = None
        self.threshold  = THRESHOLD
        self.img_size   = IMG_SIZE or 300
        self.model_name = 'Not loaded'
        self.model_path = None
        self.status     = 'not_loaded'   # not_loaded | loaded | demo

    # ── Find & load ────────────────────────────────────────────────────
    def load(self):
        if not os.path.isdir(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f'[ModelLoader] Created models/ folder at: {MODEL_DIR}')
            print('[ModelLoader] Put your .keras or .h5 model file there, then restart.')
            self.status = 'demo'
            return

        # Scan for model files
        candidates = []
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(('.keras', '.h5')):
                candidates.append(os.path.join(MODEL_DIR, fname))

        if not candidates:
            print(f'[ModelLoader] No .keras or .h5 file found in: {MODEL_DIR}')
            print('[ModelLoader] Running in DEMO MODE (random predictions).')
            self.status = 'demo'
            return

        # Prefer .keras over .h5
        candidates.sort(key=lambda p: (0 if p.endswith('.keras') else 1))
        model_path = candidates[0]

        print(f'[ModelLoader] Loading: {os.path.basename(model_path)}')
        try:
            self.model      = tf.keras.models.load_model(model_path, compile=False)
            self.model_name = os.path.basename(model_path)
            self.model_path = model_path
            self.img_size   = IMG_SIZE or self.model.input_shape[1]
            self.status     = 'loaded'
            print(f'[ModelLoader] ✅ Loaded  input={self.model.input_shape}  params={self.model.count_params():,}')
        except Exception as e:
            print(f'[ModelLoader] ❌ Failed to load model: {e}')
            self.status = 'demo'
            return

        # Load threshold from config JSON if present
        for cfg_name in ['model_config_v3.json', 'model_config_and_results.json']:
            cfg_path = os.path.join(MODEL_DIR, cfg_name)
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                ev = cfg.get('evaluation', {})
                t  = ev.get('optimal_threshold', ev.get('tta_accuracy'))
                if t and 0 < float(t) < 1:
                    self.threshold = float(t)
                    print(f'[ModelLoader] Threshold from config: {self.threshold:.4f}')
                break

    # ── Preprocessing ──────────────────────────────────────────────────
    def _preprocess(self, pil_img):
        """PIL image → preprocessed numpy array ready for the model."""
        img = np.array(pil_img.convert('RGB'))

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LANCZOS4)

        # CLAHE — X-ray contrast enhancement (same as training pipeline)
        clahe = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
        img   = clahe.apply(img)

        # EfficientNet preprocessing: scales to [-1, 1]
        img = tf.keras.applications.efficientnet.preprocess_input(
            img.astype(np.float32)
        )
        return img[np.newaxis, ...]    # Add batch dim → (1, H, W, 3)

    # ── Predict ────────────────────────────────────────────────────────
    def predict(self, pil_img):
        """
        Run inference on a PIL image.
        Returns dict with: label, prob, confidence, label_color, status
        """
        import time, random

        if self.status == 'demo' or self.model is None:
            # Demo mode — return plausible random result
            time.sleep(0.2)
            prob  = random.uniform(0.05, 0.98)
            label = 'PNEUMONIA' if prob >= self.threshold else 'NORMAL'
            return self._build_result(prob, label, elapsed=0.2, demo=True)

        start     = time.perf_counter()
        img_input = self._preprocess(pil_img)
        raw_prob  = float(self.model.predict(img_input, verbose=0)[0][0])
        elapsed   = time.perf_counter() - start

        label = 'PNEUMONIA' if raw_prob >= self.threshold else 'NORMAL'
        return self._build_result(raw_prob, label, elapsed)

    # ── Build result dict ──────────────────────────────────────────────
    def _build_result(self, prob, label, elapsed, demo=False):
        confidence = prob if label == 'PNEUMONIA' else (1.0 - prob)
        conf_pct   = confidence * 100

        if conf_pct >= 90:
            certainty, cert_color = 'HIGH CONFIDENCE',      '#4ade80'
        elif conf_pct >= 75:
            certainty, cert_color = 'MODERATE CONFIDENCE',  '#facc15'
        else:
            certainty, cert_color = 'LOW — REVIEW RECOMMENDED', '#f97316'

        label_color = '#ef4444' if label == 'PNEUMONIA' else '#22c55e'
        icon        = '🔴' if label == 'PNEUMONIA' else '🟢'

        return {
            'label'      : label,
            'icon'       : icon,
            'prob'       : prob,
            'prob_pct'   : prob * 100,
            'confidence' : confidence,
            'conf_pct'   : conf_pct,
            'certainty'  : certainty,
            'cert_color' : cert_color,
            'label_color': label_color,
            'elapsed_ms' : elapsed * 1000,
            'threshold'  : self.threshold,
            'demo'       : demo,
        }


# Singleton — one model instance shared across the app
loader = ModelLoader()
