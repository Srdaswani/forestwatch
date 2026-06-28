# ForestWatch — Deforestation Detection Dashboard

Detects deforestation from Sentinel-2 satellite imagery using a trained U-Net model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Authenticate Earth Engine:
```bash
earthengine authenticate
```

3. Run locally:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Done — your app is live at a public URL

## Usage

1. Upload your `best_model_v2.pth` checkpoint in the sidebar
2. Enter your Earth Engine project ID
3. Select a preset region or draw a custom one
4. Click "Run Detection Scan"
5. Red areas on the map = predicted forest loss

## Model

- Architecture: U-Net + ResNet34 encoder
- Training data: 1000 chips (Rondônia + Pará, Brazil)
- Validation IoU: 0.4044
- Input: 5-band Sentinel-2 (Red, Green, Blue, NIR, NDVI)
- Note: Trained on tropical deforestation — results in NC are indicative
