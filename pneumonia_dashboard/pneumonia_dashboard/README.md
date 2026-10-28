# 🫁 Pneumonia Detection Dashboard — Local Setup

## Folder Structure

```
pneumonia_dashboard/
├── app.py              ← Main dashboard (run this)
├── model_loader.py     ← Model loading & prediction logic
├── config.py           ← Settings (model path, threshold, port)
├── requirements.txt    ← Python packages
├── setup.bat           ← Windows setup (double-click once)
├── setup.sh            ← Mac/Linux setup (run once)
├── run.bat             ← Windows: start dashboard
└── models/             ← PUT YOUR MODEL FILE HERE
    └── (your .keras or .h5 file goes here)
```

---

## Step 1 — Download Your Model from Google Drive

Your trained model is saved in Google Drive. Download it to your computer:

1. Open **Google Drive** in your browser
2. Go to `My Drive → EfficientNetB3_Pneumonia/`
   (or `EfficientNetB4_Pneumonia_V3/` if you trained V3)
3. Right-click `efficientnetb3_pneumonia_final.keras` → **Download**
4. Also download `model_config_and_results.json` (for the threshold)
5. Copy both files into the `models/` folder in this project

Your `models/` folder should look like:
```
models/
├── efficientnetb3_pneumonia_final.keras   ← the model
└── model_config_and_results.json          ← threshold config
```

---

## Step 2 — Run Setup (Once)

### Windows
```
Double-click setup.bat
```
Or in VS Code terminal:
```bat
setup.bat
```

### Mac / Linux
```bash
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment and installs all packages.

---

## Step 3 — Start the Dashboard

### Windows
```
Double-click run.bat
```
Or in VS Code terminal (after activating venv):
```bat
.venv\Scripts\activate
python app.py
```

### Mac / Linux
```bash
source .venv/bin/activate
python app.py
```

The dashboard opens automatically at **http://127.0.0.1:7860**

---

## VS Code — Recommended Setup

1. Open the `pneumonia_dashboard/` folder in VS Code:
   `File → Open Folder → pneumonia_dashboard`

2. Open the terminal: `Ctrl+` ` (backtick)

3. Run setup once:
   ```
   setup.bat        (Windows)
   ./setup.sh       (Mac/Linux)
   ```

4. Select the Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type: `Python: Select Interpreter`
   - Choose `.venv` (the one inside this folder)

5. Run the app:
   ```
   python app.py
   ```

---

## Config Options (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `THRESHOLD` | `0.8714` | Decision cutoff (from your V1 results) |
| `IMG_SIZE` | `None` | Auto-detected from model. Set to `300` or `380` to override |
| `HOST` | `127.0.0.1` | Localhost only. Change to `0.0.0.0` to allow LAN access |
| `PORT` | `7860` | Dashboard port |
| `SHARE` | `False` | `True` creates a public URL via ngrok |
| `TTA_PASSES` | `1` | `1` = fast single pass. `7`+ = more accurate (slower) |

---

## Troubleshooting

**`RuntimeError: Physical devices cannot be modified after initialized`**
→ Already handled. If it still appears, just ignore it — it doesn't affect predictions.

**`No module named 'gradio'`**
→ Virtual environment not activated. Run:
```bat
.venv\Scripts\activate    (Windows)
source .venv/bin/activate  (Mac/Linux)
```

**Model not loading / Demo Mode**
→ Check that your `.keras` or `.h5` file is inside the `models/` folder.
→ File must be directly in `models/`, not in a subfolder.

**Port already in use**
→ Change `PORT = 7860` to any free port (e.g. `7861`) in `config.py`

**Slow predictions**
→ TensorFlow will use GPU automatically if CUDA is installed.
→ First prediction is always slower (model warmup). Subsequent ones are fast.
