import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import rasterio
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
import requests
import tempfile
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForestWatch — Deforestation Detection",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  .main { background: #0a0f0a; }
  .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  .fw-hero {
    background: linear-gradient(135deg, #0a1a0a 0%, #0f2b0f 50%, #0a1a0a 100%);
    border: 1px solid #1a3a1a;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .fw-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #4ade80, transparent);
  }
  .fw-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0fdf0;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
  }
  .fw-hero .subtitle {
    color: #86efac;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .fw-hero .model-badge {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-top: 0.8rem;
    border: 1px solid #166534;
  }

  .fw-metric {
    background: #0f1f0f;
    border: 1px solid #1a3a1a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .fw-metric .val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4ade80;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }
  .fw-metric .label {
    font-size: 0.72rem;
    color: #86efac;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
  }

  .fw-status {
    background: #0f1f0f;
    border-left: 3px solid #4ade80;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #86efac;
    margin: 0.5rem 0;
  }
  .fw-status.warn { border-color: #fbbf24; color: #fde68a; }
  .fw-status.error { border-color: #f87171; color: #fca5a5; }

  .fw-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a3a1a;
  }

  .result-card {
    background: #0f1f0f;
    border: 1px solid #1a3a1a;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
  }
  .result-card h3 {
    color: #f0fdf0;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.8rem 0;
  }

  .stButton > button {
    background: #15803d !important;
    color: #f0fdf0 !important;
    border: 1px solid #166534 !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #166534 !important;
    border-color: #4ade80 !important;
  }

  .stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label {
    color: #86efac !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }

  [data-testid="stSidebar"] {
    background: #060e06 !important;
    border-right: 1px solid #1a3a1a !important;
  }
  [data-testid="stSidebar"] .stMarkdown { color: #86efac; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHIP_SIZE   = 256
BANDS       = 5
SCALE_M     = 30
THRESHOLD   = 0.35   # prediction confidence threshold
CHIP_DEG    = CHIP_SIZE * SCALE_M / 111320


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=BANDS,
        classes=1,
        activation=None,
    )
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

# ── Earth Engine ───────────────────────────────────────────────────────────────
@st.cache_resource
def init_ee(project_id: str):
    try:
        if "GEE_CREDENTIALS" in st.secrets:
            creds_json = json.loads(st.secrets["GEE_CREDENTIALS"])
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(creds_json, f)
                key_file = f.name
            service_account = "forestwatch-service@ee-deforestation-499600.iam.gserviceaccount.com"
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials=credentials, project=project_id)
            os.unlink(key_file)
        else:
            ee.Initialize(project=project_id)
        return True
    except Exception as e:
        return str(e)

# ── Core pipeline ──────────────────────────────────────────────────────────────
def fetch_chip(roi_coords, year_start, year_end):
    """Fetch a single 5-band chip from Earth Engine."""
    west, south, east, north = roi_coords
    roi = ee.Geometry.Rectangle([west, south, east, north])

    def mask_s2(img):
        scl = img.select('SCL')
        clear = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(clear)

    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(f'{year_start}-06-01', f'{year_end}-09-30')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(mask_s2)
        .map(lambda img: img.addBands(img.normalizedDifference(['B8','B4']).rename('NDVI')))
        .median()
        .select(['B4','B3','B2','B8','NDVI'])
        .clip(roi)
    )

    url = s2.getDownloadURL({
        'bands': ['B4','B3','B2','B8','NDVI'],
        'region': roi,
        'scale': SCALE_M,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        return None
    return resp.content

def preprocess_chip(tif_bytes):
    """Read GeoTIFF bytes → normalized numpy array (5, H, W)."""
    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            data = src.read().astype(np.float32)
    for i in range(data.shape[0]):
        band = data[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            data[i] = 0
            continue
        p2, p98 = np.percentile(finite, [2, 98])
        denom = (p98 - p2) if (p98 - p2) > 1e-6 else 1.0
        data[i] = np.clip((band - p2) / denom, 0, 1)
    return np.nan_to_num(data, nan=0.0)

def run_inference(model, chip_array):
    """Run model on a single chip, return prediction mask (H, W)."""
    tensor = torch.tensor(chip_array).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.sigmoid(logits).squeeze().numpy()
    return pred

def tile_region(bbox, max_chips=64):
    """Split bounding box into 256×256 chip tiles."""
    west, south, east, north = bbox
    cols = max(1, int((east - west) / CHIP_DEG))
    rows = max(1, int((north - south) / CHIP_DEG))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            w = west  + c * CHIP_DEG
            e = w + CHIP_DEG
            s = south + r * CHIP_DEG
            n = s + CHIP_DEG
            tiles.append((w, s, e, n))
            if len(tiles) >= max_chips:
                return tiles
    return tiles

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fw-section-label">Model</div>', unsafe_allow_html=True)
    model_file = st.file_uploader(
        "Upload best_model_v2.pth",
        type=['pth'],
        help="Upload your trained model checkpoint"
    )

    st.markdown('<div class="fw-section-label" style="margin-top:1.2rem">Earth Engine</div>', unsafe_allow_html=True)
    ee_project = st.text_input(
        "GEE Project ID",
        value="ee-deforestation-499600",
        help="Your Google Earth Engine project ID"
    )

    st.markdown('<div class="fw-section-label" style="margin-top:1.2rem">Detection Settings</div>', unsafe_allow_html=True)
    year_baseline = st.number_input("Baseline year", min_value=2015, max_value=2025, value=2020, step=1)
    year_recent   = st.number_input("Recent year",   min_value=2015, max_value=2025, value=2024, step=1)
    max_chips_ui  = st.number_input("Max tiles to scan", min_value=1, max_value=200, value=16, step=1,
                                    help="More tiles = wider coverage but slower")
    conf_threshold = st.number_input("Detection threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05,
                                     format="%.2f", help="Higher = fewer but more confident detections")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4ade80; font-family: JetBrains Mono, monospace;'>
    ForestWatch v1.0<br>
    Model: U-Net + ResNet34<br>
    Val IoU: 0.4044<br>
    Training: 1000 chips<br>
    Rondônia + Pará, Brazil
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fw-hero">
  <div class="subtitle">Satellite Intelligence</div>
  <h1>🌲 ForestWatch</h1>
  <div style="color:#a7f3d0; font-size:1rem; margin-top:0.3rem;">
    Detect deforestation from Sentinel-2 satellite imagery using deep learning.
    Select a region, run the scan, and see where forest has been lost.
  </div>
  <div class="model-badge">U-Net · ResNet34 · Val IoU 0.4044 · 1000 chip training set</div>
</div>
""", unsafe_allow_html=True)

# ── Region selector ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown('<div class="fw-section-label">Select Region</div>', unsafe_allow_html=True)

    bbox_col_a, bbox_col_b = st.columns(2)
    with bbox_col_a:
        west  = st.number_input("West longitude",  min_value=-180.0, max_value=180.0, value=-79.2, step=0.1, format="%.4f", help="Western longitude in decimal degrees")
        east  = st.number_input("East longitude",  min_value=-180.0, max_value=180.0, value=-78.2, step=0.1, format="%.4f", help="Eastern longitude in decimal degrees")
    with bbox_col_b:
        south = st.number_input("South latitude", min_value=-90.0,  max_value=90.0,  value=35.6,  step=0.1, format="%.4f", help="Southern latitude in decimal degrees")
        north = st.number_input("North latitude", min_value=-90.0,  max_value=90.0,  value=36.1,  step=0.1, format="%.4f", help="Northern latitude in decimal degrees")

    if west < east and south < north:
        bbox = [west, south, east, north]
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2

        tiles = tile_region(bbox, max_chips_ui)
        chip_count = len(tiles)
        area_km2 = (east - west) * 111 * (north - south) * 111

        st.markdown(f"""
        <div class="fw-metric" style="margin-top:0.8rem">
          <div class="val">{chip_count}</div>
          <div class="label">Tiles to scan</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="fw-metric" style="margin-top:0.5rem">
              <div class="val">{area_km2:.0f}</div>
              <div class="label">km² coverage</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="fw-metric" style="margin-top:0.5rem">
              <div class="val">{chip_count * 2 // 60 + 1}</div>
              <div class="label">Est. minutes</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="fw-status warn">West must be less than East, and South less than North</div>', unsafe_allow_html=True)
        bbox = None
        center_lat, center_lon = 35.8, -78.6

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶ Run Detection Scan", disabled=(bbox is None or model_file is None))

    if model_file is None:
        st.markdown('<div class="fw-status warn">Upload model checkpoint in sidebar to enable scanning</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="fw-section-label">Region Map</div>', unsafe_allow_html=True)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='CartoDB dark_matter'
    )

    if bbox is not None:
        west, south, east, north = bbox
        m.fit_bounds([[south, west], [north, east]])
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color='#4ade80',
            weight=2,
            fill=True,
            fill_color='#4ade80',
            fill_opacity=0.05,
            tooltip=f"Scan region: {chip_count} tiles"
        ).add_to(m)

        tiles_to_show = tile_region(bbox, max_chips_ui)
        for (w, s, e, n) in tiles_to_show:
            folium.Rectangle(
                bounds=[[s, w], [n, e]],
                color='#4ade80',
                weight=0.5,
                fill=False,
                opacity=0.3
            ).add_to(m)

    map_result = st_folium(m, width=None, height=420, returned_objects=[])

# ── Run detection ──────────────────────────────────────────────────────────────
if run_btn:
    if model_file is None:
        st.error("Upload your model checkpoint first.")
        st.stop()

    # Save uploaded model to temp file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp.write(model_file.read())
        tmp_model_path = tmp.name

    # Initialize EE
    ee_status = init_ee(str(ee_project).strip())
    if ee_status is not True:
        st.markdown(f'<div class="fw-status error">Earth Engine error: {ee_status}</div>', unsafe_allow_html=True)
        st.stop()

    # Load model
    try:
        model = load_model(tmp_model_path)
    except Exception as e:
        st.markdown(f'<div class="fw-status error">Model load error: {e}</div>', unsafe_allow_html=True)
        st.stop()

    tiles = tile_region(bbox, max_chips_ui)
    loss_tiles = []
    total_loss_pixels = 0
    total_pixels = 0

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for i, (w, s, e, n) in enumerate(tiles):
        status_text.markdown(
            f'<div class="fw-status">Scanning tile {i+1}/{len(tiles)} — [{w:.3f}, {s:.3f}]</div>',
            unsafe_allow_html=True
        )
        try:
            tif_bytes = fetch_chip([w, s, e, n], year_baseline, year_recent)
            if tif_bytes is None:
                continue
            chip_arr = preprocess_chip(tif_bytes)
            pred_mask = run_inference(model, chip_arr)
            binary = (pred_mask > conf_threshold).astype(np.float32)
            loss_frac = float(binary.mean())
            total_loss_pixels += int(binary.sum())
            total_pixels += binary.size
            if loss_frac > 0.005:
                loss_tiles.append({
                    'bounds': [w, s, e, n],
                    'loss_frac': loss_frac,
                    'pred': pred_mask,
                    'binary': binary,
                })
        except Exception:
            pass
        progress_bar.progress((i + 1) / len(tiles))

    status_text.empty()
    progress_bar.empty()

    # ── Results ────────────────────────────────────────────────────────────────
    loss_pct = (total_loss_pixels / max(total_pixels, 1)) * 100
    area_loss_km2 = total_loss_pixels * (SCALE_M / 1000) ** 2

    st.markdown(f"""
    <div class="result-card">
      <h3>Detection Results</h3>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="fw-metric"><div class="val">{len(tiles)}</div><div class="label">Tiles scanned</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="fw-metric"><div class="val">{len(loss_tiles)}</div><div class="label">Tiles with loss</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="fw-metric"><div class="val">{loss_pct:.1f}%</div><div class="label">Loss detected</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="fw-metric"><div class="val">{area_loss_km2:.1f}</div><div class="label">km² flagged</div></div>', unsafe_allow_html=True)

    # ── Results map ────────────────────────────────────────────────────────────
    if loss_tiles:
        st.markdown('<div class="fw-section-label" style="margin-top:1.5rem">Deforestation Map</div>', unsafe_allow_html=True)

        result_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB dark_matter'
        )

        for tile in loss_tiles:
            w, s, e, n = tile['bounds']
            intensity = min(tile['loss_frac'] * 5, 0.85)

            folium.Rectangle(
                bounds=[[s, w], [n, e]],
                color='#ef4444',
                weight=1,
                fill=True,
                fill_color='#ef4444',
                fill_opacity=intensity,
                tooltip=f"Forest loss: {tile['loss_frac']*100:.1f}% of tile"
            ).add_to(result_map)

        # Region outline
        west, south, east, north = bbox
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color='#4ade80',
            weight=2,
            fill=False,
        ).add_to(result_map)

        folium.LayerControl().add_to(result_map)
        st_folium(result_map, width=None, height=500, returned_objects=[])

        st.markdown(f"""
        <div class="fw-status" style="margin-top:1rem">
          ● Red areas = predicted forest loss detected above {conf_threshold:.0%} confidence threshold
        </div>
        <div class="fw-status warn">
          ⚠ Model trained on tropical Brazil — NC detections are indicative, not ground-truth verified
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="fw-status warn" style="margin-top:1rem">
          No significant forest loss detected in this region at current threshold.
          Try lowering the detection threshold in the sidebar, or select a different region.
        </div>
        """, unsafe_allow_html=True)

    os.unlink(tmp_model_path)
