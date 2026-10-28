#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# setup.sh — Mac / Linux Setup Script
# Run:  chmod +x setup.sh && ./setup.sh
# ══════════════════════════════════════════════════════════════════════

set -e

echo ""
echo "  ============================================================"
echo "   Pneumonia Detection Dashboard — Mac/Linux Setup"
echo "  ============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "  [ERROR] python3 not found."
    echo "  Install: sudo apt install python3 python3-venv  (Ubuntu)"
    echo "        or: brew install python                    (Mac)"
    exit 1
fi

echo "  [1/4] Python found: $(python3 --version)"
echo ""

# Create venv
echo "  [2/4] Creating virtual environment (.venv) ..."
python3 -m venv .venv
echo "  Done."
echo ""

# Activate and install
echo "  [3/4] Installing packages (3-5 minutes) ..."
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "  Done."
echo ""

# Create models folder
mkdir -p models
echo "  [4/4] models/ folder ready."

echo ""
echo "  ============================================================"
echo "   Setup complete!"
echo "  ============================================================"
echo ""
echo "  NEXT STEPS:"
echo "  1. Copy your model file (.keras or .h5) into the models/ folder"
echo "  2. Run the dashboard:"
echo "       source .venv/bin/activate"
echo "       python app.py"
echo ""
