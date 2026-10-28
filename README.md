🫁 Pneumonia Detection from Chest X-Rays

Deep-learning system that classifies chest X-ray images as NORMAL or PNEUMONIA using transfer learning on ImageNet-pretrained CNNs, with two ready-to-run interfaces:

Streamlit app (app.py): lightweight web app with the trained MobileNetV2 model (pneumonia_model.h5) included.
Clinical-style dashboard (dashboard/): Gradio dashboard for EfficientNet models with CLAHE contrast enhancement, a tuned decision threshold, confidence levels, session statistics and a prediction log.

⚠️ Disclaimer: This is a research/educational project and not a medical device. Predictions must not be used for diagnosis.

Streamlit app

✨ Features
Binary classification of chest X-rays: Normal vs Pneumonia
Transfer learning (MobileNetV2 / EfficientNet backbones)
CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing to enhance lung detail
Tuned decision threshold instead of a fixed 0.5 cut-off
Confidence levels: high, moderate, or low (review recommended)
Session statistics and prediction log in the dashboard
Optional test-time augmentation (TTA) passes
evaluate.py to reproduce test-set metrics, a confusion matrix and an ROC curve
🧠 Pipeline
yes
no
Chest X-ray image
Resize224 / 300 px
CLAHE contrastenhancement(EfficientNet dashboard)
Pretrained CNN backboneMobileNetV2 / EfficientNet
Global average pooling+ sigmoid output
Probability = threshold?
PNEUMONIA
NORMAL
Component	MobileNetV2 app (app.py)	EfficientNet dashboard (dashboard/)
Input size	224 × 224 RGB	300 × 300 (B3) / 380 × 380 (B4), auto-detected
Preprocessing	Rescale to [0, 1]	CLAHE + EfficientNet preprocess_input
Output	Sigmoid probability	Sigmoid probability
Decision threshold	0.50	0.8714 (tuned on validation set)
UI	Streamlit	Gradio
📊 Results

Evaluated on the held-out test set with evaluate.py.

Metric	Value
Accuracy	—
Precision	—
Recall (Sensitivity)	—
Specificity	—
F1-score	—
ROC-AUC	—
<p align="center"> <img src="images/confusion_matrix.png" width="45%" alt="Confusion matrix"/> &nbsp; <img src="images/roc_curve.png" width="45%" alt="ROC curve"/> </p>
🖥️ Dashboard

Gradio dashboard

📁 Project Structure
pneumonia-detection-efficientnet/
├── app.py                 # Streamlit app (MobileNetV2)
├── pneumonia_model.h5     # Trained MobileNetV2 model
├── evaluate.py            # Test-set metrics + figures
├── requirements.txt
├── images/                # README screenshots and result figures
└── dashboard/             # Gradio dashboard (EfficientNet)
    ├── app.py
    ├── model_loader.py
    ├── config.py          # model path, threshold, port, TTA
    ├── requirements.txt
    ├── setup.bat / setup.sh / run.bat
    └── models/            # put your .keras / .h5 model here
🚀 Getting Started
1. Clone
bash
git clone https://github.com/AliMurad0/pneumonia-detection-efficientnet.git
cd pneumonia-detection-efficientnet
2. Run the Streamlit app
bash
pip install -r requirements.txt
streamlit run app.py

Upload a chest X-ray (.jpg / .png) to get a prediction with a confidence score.

3. Run the EfficientNet dashboard
bash
cd dashboard
# Windows: setup.bat then run.bat
# Mac/Linux:
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
python app.py

Place your trained EfficientNet model (.keras or .h5) in dashboard/models/. The dashboard opens at http://127.0.0.1:7860. Settings such as the threshold, image size, port and TTA passes are in dashboard/config.py.

4. Reproduce the results

Download the Chest X-Ray Images (Pneumonia) dataset, then:

bash
python evaluate.py --model pneumonia_model.h5 --test-dir chest_xray/test

This writes images/confusion_matrix.png, images/roc_curve.png and images/metrics.json, and prints a metrics table for this README.

🛠️ Tech Stack

Python · TensorFlow / Keras · MobileNetV2 · EfficientNet · OpenCV / Albumentations (CLAHE) · scikit-learn · Streamlit · Gradio