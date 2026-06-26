# 🏆 TANAHI — AI Pipeline for Product Recognition from Social Media Images
### The 2nd URA Hackathon 2026 | Team TANAHI | Ho Chi Minh City University of Technology

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7.3-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![VietOCR](https://img.shields.io/badge/VietOCR-vgg__seq2seq-green)](https://github.com/pbcquoc/vietocr)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20CPU-lightblue)](https://kaggle.com)

> **Competition Score Formula:**  
> `Score = 0.40 × F1_brand + 0.35 × (1 − CER_ocr) + 0.25 × F1_product`

---

## 📖 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Our Approach & Key Ideas](#-our-approach--key-ideas)
3. [System Architecture](#-system-architecture)
4. [Notebook Overview](#-notebook-overview)
5. [How to Run on Kaggle](#-how-to-run-on-kaggle)
6. [How to Run Locally](#-how-to-run-locally)
7. [File Structure](#-file-structure)
8. [Technical Details](#-technical-details)
9. [Results](#-results)
10. [References](#-references)

---

## 🎯 Problem Statement

**Goal:** Build a **lightweight, GPU-free AI pipeline** that automatically extracts structured product information from social media images (TikTok, Instagram, etc.).

**Given:** A collection of product images from social media posts.

**Predict:**

| Column | Description | Example |
|---|---|---|
| `image_id` | Image identifier | `priv_h_0001` |
| `ocr_text` | All visible text in the image | `"Sữa tươi Vinamilk Flex 180ml không đường"` |
| `brand_name` | Brand / manufacturer name | `"Vinamilk"` |
| `product_name` | Product line / model name | `"Flex"` |

**Key Challenges:**
- 🌀 Social media noise: watermarks, filters, overlapping text, skewed angles
- 🇻🇳🇬🇧 Mixed Vietnamese–English text on product packaging
- 🎬 Input can be static images, animated GIFs, or video clips
- ⚙️ **CPU-only constraint** — no GPU allowed (competition rule)
- ⚡ Must be lightweight and fast enough for 1,200+ images in under 45 minutes

---

## 💡 Our Approach & Key Ideas

### Idea 1: Dual-Engine OCR (Detect + Recognize separately)

Standard PaddleOCR misreads Vietnamese diacritics (e.g., `"Sữa"` → `"Sua"`). VietOCR reads Vietnamese accurately but needs bounding boxes as input.

**Our solution: split the OCR task into two specialized engines:**

```
PaddleOCR PP-OCRv4  →  Detects WHERE text is (bounding boxes)
VietOCR vgg_seq2seq →  Reads WHAT the text says (per-crop recognition)
```

This gives us the spatial accuracy of PaddleOCR combined with the Vietnamese language quality of VietOCR.

---

### Idea 2: 3-Layer Prediction with Priority Cascade

Pure ML with ~5,000 training samples struggles with unseen brands. Pure rules miss edge cases.

**Our solution: three layers that fall back gracefully:**

```
Layer 1 (Highest Priority): Rule-Based
  → 101 handcrafted BRAND_RULES regex patterns
  → 80+ PRODUCT_LINE_PATTERNS
  → High precision, covers known brands perfectly

Layer 2 (Fallback): Machine Learning
  → BrandClassifier: TF-IDF char-ngram + Logistic Regression
  → ProductClassifier: 2-stage gate + classifier
  → BrandKNN: cosine similarity k=5 neighbors
  → Generalizes to unseen OCR text variations

Layer 3 (Supplement only): Layout Analysis
  → Canny edge detection → Red box localization
  → Prominence score (text height × uppercase × vertical position)
  → ONLY supplements when both Layer 1 & 2 return empty results
  → Never overrides ML predictions
```

---

### Idea 3: Anti-Hallucination Guard

ML classifiers can predict brand names that do not actually appear in the image.

**Our solution:** Before assigning any brand prediction, we verify that the predicted brand tokens **actually exist in the OCR text**. If not → discard the prediction.

---

### Idea 4: Fault-Tolerant Checkpoint System

Processing 1,200+ images on a Kaggle CPU session (~30–45 min) risks session timeouts.

**Our solution:** Save a checkpoint CSV every 50 images. On restart, automatically resume from the last checkpoint — no images are processed twice.

---

### Idea 5: Multi-Media Support

Social media images are not always static JPEGs.

**Our solution:** `open_media_for_ocr()` handles:
- **Animated GIFs** — samples 11 frames, picks the sharpest
- **Video clips (mp4/webm)** — uses `cv2.VideoCapture`, picks sharpest frame by Laplacian variance
- **Animated WebP** — uses Pillow `n_frames`

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              📸  INPUT MEDIA                                      │
│     JPG · PNG · GIF (animated) · MP4/WebM · WebP (animated)     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │      PREPROCESSING       │
            │  • Resize (max 1280px)   │
            │  • Contrast ×1.45        │
            │  • Sharpen filter        │
            │  • Frame selection       │
            │    (sharpest by          │
            │     Laplacian variance)  │
            └──────────────┬───────────┘
                           │
              ┌────────────▼────────────┐
              │     DUAL-ENGINE OCR     │
              │                         │
    ┌─────────┴────────┐   ┌────────────┴──────────┐
    │  PaddleOCR       │   │  VietOCR              │
    │  PP-OCRv4        │   │  vgg_seq2seq           │
    │  (detect only)   │   │  (recognize only)      │
    │  → bounding boxes│   │  → text per crop       │
    └─────────┬────────┘   └────────────┬──────────┘
              │  sort boxes (reading    │
              │  order: top→bottom,     │
              │  left→right)            │
              └────────────┬────────────┘
                           │
               ┌───────────▼──────────────┐
               │   BILINGUAL POST-PROC    │
               │  • Fix accented English  │
               │    (máscara → mascara)   │
               │  • Fix OCR digit/letter  │
               │    (m0del → model)       │
               │  • Dedup adjacent tokens │
               └───────────┬──────────────┘
                           │  ocr_text (clean)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  3-LAYER PREDICTION ENGINE                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 1 — Rule-Based                   [FIRST PRIORITY] │    │
│  │  • 101 BRAND_RULES regex patterns                        │    │
│  │  • 80+ PRODUCT_LINE_PATTERNS                             │    │
│  │  • OCR typo correction map                               │    │
│  │  • detect_brand_in_ocr() alias registry                 │    │
│  │  Output: brand_name, product_name (if matched)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                     ↓  (if still empty)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 2 — Machine Learning              [FALLBACK]      │    │
│  │  • BrandClassifier: TF-IDF char(2-4) + LogReg           │    │
│  │    threshold 0.62, trained on 4,892 samples             │    │
│  │  • ProductClassifier: gate(0.40) + classifier(0.45)     │    │
│  │  • BrandKNN: cosine_similarity, k=5 neighbors           │    │
│  │  • Anti-hallucination: verify tokens exist in OCR       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                     ↓  (supplement only)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 3 — Layout Analysis             [SUPPLEMENT]      │    │
│  │  • Canny edge + GaussianBlur + dilate + findContours    │    │
│  │  • find_red_box(): locate product bounding object       │    │
│  │  • Prominence score = height × uppercase × y-position   │    │
│  │  • Activates ONLY when: prominence ≥ 30 AND ML empty    │    │
│  │  • NEVER overwrites Layer 1 or Layer 2 results          │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     POST-PROCESSING     │
              │  • reconcile_brand_     │
              │    product()            │
              │  • Social noise filter  │
              │    (TikTok/hashtag)     │
              │  • Generic token filter │
              │  • Dedup product tokens │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        OUTPUT           │
              │  • submission.csv       │
              │  • Checkpoint / 50 imgs │
              │  • 7 validation checks  │
              │    (AC1 – AC7)          │
              └─────────────────────────┘
```

---

## 📓 Notebook Overview

**Main notebook:** `tanahi-demo-ver3.ipynb`

The notebook consists of **10 code cells** plus **12 markdown documentation cells**:

| Cell | Name | Description |
|---|---|---|
| **Cell 1** | 📦 Install | Installs `paddleocr`, `vietocr`, `underthesea`, `scikit-learn`, `Pillow==9.5.0` (pinned) |
| **Cell 2** | ⚙️ Config | Auto-discovers dataset paths (works for Phase 1 & Phase 2 without hardcoding) |
| **Cell 3** | 📏 Rules | 101 `BRAND_RULES` regex + 80+ `PRODUCT_LINE_PATTERNS` + typo correction + noise filters |
| **Cell 4** | 🖼️ Layout | `find_red_box()` (Canny/contour) + `extract_v5()` (prominence scoring) + bilingual post-processor |
| **Cell 5** | 🤖 Train ML | Trains `BrandClassifier`, `ProductClassifier`, `BrandKNN` from `train_labels.csv` + dynamic brand discovery |
| **Cell 6** | 🔮 Predict | `predict_labels()` (3-layer cascade) + `predict_image()` + `evaluate_pipeline()` |
| **Cell 7** | 🔍 OCR Engine | PaddleOCR detect → crop → VietOCR recognize → reading-order sort |
| **Cell 7b** | 🎬 Media | `open_media_for_ocr()` for GIF/video/WebP — selects sharpest frame |
| **Cell 8** | 🚀 Inference | Batch inference on all test images with checkpoint every 50 images |
| **Cell 9** | ✂️ Post-proc | `postprocess_submission_df()` — remove consecutive duplicate tokens |
| **Cell 10** | 📤 Export | Write `submission.csv` with 7 validation assertions (AC1–AC7) |

---

## 🚀 How to Run on Kaggle

### Prerequisites
- Kaggle account with access to **The 2nd URA Hackathon 2026** competition

### Step 1 — Create a new Kaggle Notebook
Go to the competition page → **Code** tab → **New Notebook**

### Step 2 — Add the dataset
In the Notebook editor → **+ Add Input** (right sidebar) → search **"The 2nd URA Hackathon 2026"** → **Add**

The dataset will appear at:
```
/kaggle/input/the-2nd-ura-hackathon/
  ├── phase2_dataset/
  │   └── phase2_dataset/
  │       ├── images/               ← test images (Phase 2)
  │       ├── private_test.csv
  │       └── sample_submission_private.csv
  ├── train_images/                 ← training images
  └── train_labels.csv
```

### Step 3 — Configure session settings
```
Accelerator:  None  (CPU only — GPU not needed)
Internet:     ON    (required to download VietOCR model ~100 MB on first run)
Persistence:  No persistence
```

### Step 4 — Upload the notebook
Click **File → Import Notebook** → upload `tanahi-demo-ver3.ipynb`

### Step 5 — Clear previous outputs
Click **Run → Restart & Clear All Outputs** before running to ensure a fresh start.

> ⚠️ **Important:** If you are resuming after a crash, do NOT clear outputs — the notebook will automatically resume from the last checkpoint.

### Step 6 — Run All
Click **Run All** and wait. Estimated time: **25–45 minutes** for 1,202 images.

You should see progress logs like:
```
CPU threads: 4 | pip workers: 4
Installing packages...
Pillow pinned to 9.5.0 ✓
PaddleOCR detector OK | conf>0.35 | 4 workers
VietOCR vgg_seq2seq loaded (CPU) ✓
Rules v4.0 | 101 rules | Smoke test: 13/13 ✓
BrandClassifier trained: 4892 samples ✓
ProductClassifier trained ✓
BrandKNN built ✓
[50/1202]  checkpoint saved...
[100/1202] checkpoint saved...
...
[1202/1202] done ✓
AC1 ✓  AC2 ✓  AC3 ✓  AC4 ✓  AC5 ✓  AC6 ✓  AC7 ✓
Submission saved → /kaggle/working/submission.csv
```

### Step 7 — Download the output
In the right sidebar → **Output** tab → download `submission.csv`

---

## 💻 How to Run Locally

### Requirements
```bash
Python >= 3.10
RAM    >= 8 GB
Disk   >= 5 GB (for OCR models)
```

### Installation
```bash
# Clone this repo
git clone https://github.com/johntai271-png/ura-hackathon-2026.git
cd ura-hackathon-2026

# Install dependencies (order matters!)
pip install paddlepaddle
pip install paddleocr>=2.7,<3
pip install vietocr underthesea
pip install scikit-learn tqdm
pip install Pillow==9.5.0   # Must be last — pins Pillow to avoid conflict
```

### Data setup
Download the competition data and organize it as:
```
data/
  ├── train_labels.csv
  ├── private_test.csv
  ├── sample_submission_private.csv
  ├── train_images/
  └── images/              ← test images
```

### Configure paths
In **Cell 2** of the notebook, set:
```python
RUNNING_ON_KAGGLE = False   # Change this to False for local run
```

The notebook will then use this path (already set in Cell 2):
```python
COMP_ROOT = Path(r'C:\path\to\your\data')   # Update this path
```

### Run the notebook
```bash
jupyter notebook tanahi-demo-ver3.ipynb
# Then run all cells
```

Or run headlessly:
```bash
jupyter nbconvert --to notebook --execute tanahi-demo-ver3.ipynb \
    --output tanahi-output.ipynb --ExecutePreprocessor.timeout=3600
```

---

## 📁 File Structure

```
ura-hackathon-2026/
│
├── tanahi-demo-ver3.ipynb   ← 🏆 MAIN SUBMISSION NOTEBOOK
│                               (12 markdown cells + 10 code cells)
│
├── have_train.ipynb         ← v6.0 pipeline (PaddleOCR + VietOCR merged)
├── pipeline.ipynb           ← R&D notebook (object detection experiments)
│
└── README.md                ← This file
```

---

## 🔩 Technical Details

### Libraries & Versions

| Library | Version | Purpose |
|---|---|---|
| `paddlepaddle` | latest CPU | PaddleOCR runtime |
| `paddleocr` | `>=2.7, <3` | Text detection (PP-OCRv4) |
| `vietocr` | latest | Vietnamese text recognition |
| `underthesea` | latest | Vietnamese NLP utilities |
| `scikit-learn` | latest | ML classifiers (LogReg, TF-IDF) |
| `opencv-python` | bundled | Canny edge, contour detection |
| `Pillow` | `==9.5.0` | Image preprocessing (**pinned!**) |
| `tqdm` | latest | Progress bars |

> **Why Pillow==9.5.0?**  
> Pillow 10.0+ removed `PIL._util.is_directory` and `PIL._util.is_path`, which PaddleOCR internally imports. VietOCR automatically upgrades Pillow to the latest version when installed, breaking PaddleOCR. We pin Pillow **as the very last install step** to prevent this.

### Model Details

| Model | Size | Task |
|---|---|---|
| PaddleOCR PP-OCRv4 det | ~4 MB | Text bounding box detection |
| VietOCR vgg_seq2seq | ~100 MB | Vietnamese text recognition |
| BrandClassifier | <1 MB | Brand name classification |
| ProductClassifier | <1 MB | Product name classification |
| BrandKNN | <5 MB | TF-IDF cosine similarity search |

### Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `OCR_CONF_THRESHOLD` | `0.35` | Minimum PaddleOCR detection confidence |
| `BRAND_CLF_THRESHOLD` | `0.62` | Minimum probability to accept BrandClassifier output |
| `PRODUCT_GATE_THRESHOLD` | `0.40` | Minimum probability to predict "has product" |
| `PRODUCT_CLF_THRESHOLD` | `0.45` | Minimum probability for ProductClassifier output |
| `PIPELINE_LAYOUT_MIN_PROMINENCE` | `30.0` | Minimum prominence score to activate Layout layer |
| `MIN_BRAND_SUPPORT` | `2` | Minimum occurrences in train to register a new brand |
| `KNN_K` | `5` | Number of neighbors for BrandKNN |
| `CHECKPOINT_INTERVAL` | `50` | Save checkpoint every N images |

---

## 📈 Results

| Metric | Score |
|---|---|
| F1 Brand | ~0.52 |
| 1 − CER (OCR accuracy) | ~0.75 |
| F1 Product | ~0.62 |
| **Overall Score** | **~0.62** |

*Estimates based on train-set evaluation. Private test results may vary.*

---

## 📚 References

- Baek, J. et al. (2019). [What Is Wrong With Scene Text Recognition Model Comparisons?](https://arxiv.org/abs/1904.01906)
- Du, Y. et al. (2021). [PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System](https://arxiv.org/abs/2109.03144)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR GitHub](https://github.com/pbcquoc/vietocr) — pbcquoc
- [The 2nd URA Hackathon 2026](https://kaggle.com/competitions/the-2nd-ura-hackathon) — HCMUT URA Research Group

---

<div align="center">

**TANAHI Team** · Ho Chi Minh City University of Technology  
URA Hackathon 2026

</div>
