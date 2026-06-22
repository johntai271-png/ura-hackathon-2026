# URA Hackathon 2026 — Vietnamese Product OCR Recognition Pipeline

A pipeline for recognizing Vietnamese consumer product names from real-world images, combining **PaddleOCR** (text extraction) with **Brand Rules + Machine Learning** (brand identification). Built for **The 2nd URA Hackathon 2026**.

---

## Pipeline Design & Idea

The core challenge: product photos taken in the real world are messy — tilted shots, blurry labels, social media overlays, and OCR misreads of brand names. A naive OCR + string matching approach would produce too much noise. Pipeline v4.0 tackles this in 5 stages:

```
Input Image
     │
     ▼
[Stage 1] Image Preprocessing
     Contrast boost 1.35x + Sharpen + Resize to max 1280px
     Goal: help PaddleOCR read text on low-quality photos
     │
     ▼
[Stage 2] PaddleOCR  (CPU, latin_PP-OCRv3)
     Read all text in image, filter boxes with confidence > 0.35
     Sort by Y-coordinate (top to bottom) to preserve natural reading order
     │
     ▼
[Stage 3] Social Media Noise Filter  (clean_social_ocr)
     Remove: @mentions, #hashtags, URLs, timestamps, long digit strings
     Fix common OCR typos: "vinamill" → "vinamilk", "canfuco" → "canfoco", ...
     Discard image entirely if OCR text is pure social noise (tiktok, capcut, fyp...)
     │
     ▼
[Stage 4] Brand Recognition  (Brand Rules + Anti-hallucination Filter)
     - ~30 hand-crafted regex rules for popular Vietnamese brands
     - Anti-hallucination: only keep brand tokens that actually appear
       in the OCR text — prevents the model from inventing brand names
     - If no rule matches → ProductPredictor (Logistic Regression) takes over
     │
     ▼
[Stage 5] Post-processing  (post_process_prediction)
     Upgrades generic labels to specific ones by re-scanning the OCR text:
       "Pate" + "Hải Phòng" + "cột đen"  →  "Pate Cột Đèn Hải Phòng"
       "Nestlé" + "nan optipro"           →  "Nestlé NAN"
       "Vinamilk" + "dielac"              →  "Vinamilk Dielac"
     │
     ▼
submission.csv
```

### Why this design?

**Problem 1 — OCR misreads brand names:** The `OCR_TYPO_MAP` dictionary and `_normalize_for_rules()` normalize common OCR errors before regex matching, so "HALONG CANFUCO" or "vinamill" are still correctly identified.

**Problem 2 — TikTok/Instagram overlays on product photos:** `clean_social_ocr()` and `_is_ocr_noise_only()` filter out images where OCR only reads social media text, preventing false positives.

**Problem 3 — Model hallucinating brand names:** `_assign_readable_brand_tokens()` cross-checks every token in the predicted brand name against the actual OCR output. If a token isn't present in the image text, it gets dropped.

**Problem 4 — Duplicate labels with different casing/diacritics:** `normalize_product_name()` + `_build_product_canon_map()` collapse all spelling variants into one canonical name, chosen by frequency in the training data.

---

## Repository Structure

```
ura-hackathon-2026/
├── ura_hackathon_v4.ipynb   # Main notebook (pipeline v4.0)
└── README.md
```

---

## Requirements

| Library | Version |
|---|---|
| Python | ≥ 3.8 |
| paddlepaddle | CPU build (latest) |
| paddleocr | ≥ 2.7, < 3 |
| scikit-learn | latest |
| pandas | latest |
| Pillow | latest |
| tqdm | latest |
| joblib | latest |

---

## How to Run

### Option 1: Kaggle (recommended)

Kaggle provides multi-core CPUs and the notebook installs all dependencies and locates the data directory automatically.

1. Go to [kaggle.com](https://www.kaggle.com) → **Create Notebook** → upload `ura_hackathon_v4.ipynb`
2. Open the **Data** tab on the right → Add dataset → search `the-2nd-ura-hackathon` → Add
3. Make sure `RUNNING_ON_KAGGLE = True` (this is the default)
4. Click **Run All**
5. `submission.csv` will appear in `/kaggle/working/` when done

The notebook checkpoints every 50 images to `checkpoint.csv`. If the session is interrupted, re-running will resume from where it left off — no need to re-OCR from scratch.

### Option 2: Local (Windows)

1. Download and extract the competition dataset to a local folder

2. Install dependencies:
```bash
pip install paddlepaddle tqdm scikit-learn pillow pandas joblib
pip install "paddleocr>=2.7,<3"
```

3. Open the notebook and update these two lines:

```python
# Change to False
RUNNING_ON_KAGGLE = False

# Point to your local data folder
COMP_ROOT = Path(r'C:\path\to\the-2nd-ura-hackathon')
```

4. Run all cells top to bottom. `submission.csv` will be saved in the same directory as the notebook.

---

## Supported Brands

The pipeline recognizes ~30 Vietnamese consumer brands, including sub-product lines:

| Category | Brands |
|---|---|
| Domestic milk | Vinamilk (Flex, Dielac, Ông Thọ, ColosBaby...), TH True Milk, Dutch Lady, Nutifood, Ba Vì, Lothamilk, Yomost, Fami, Đà Lạt Milk, Kun |
| Imported milk | Abbott Ensure, Abbott PediaSure, Abbott Similac, Nestlé NAN, Aptamil, HiPP, Friso, Meiji, Anlene, Anchor |
| Pâté & canned food | Ha Long Canfoco (Pate, Pate Cột Đèn), Đồ Hộp Hạ Long, Pate Cột Đèn Hải Phòng, Vissan, Hafi, Ba Huân, San Hà, CP, Long Biên, Nhân Hòa Foods |
| Other | Nestlé Milo, Highlands Coffee, The Coffee House, Acnes |

---

## Validation & Submission Check

The pipeline auto-evaluates on 20% of training data (up to 500 samples) and prints:

```
F1=0.XXXX  P=0.XXXX  R=0.XXXX  ExactAcc=0.XXXX  N=500
Est.Score (CER~0.25): 0.XXXX   ← estimated public leaderboard score
```

Before writing the output file, 7 acceptance criteria are verified:

```
[PASS] AC1 rows        — row count matches sample_submission.csv
[PASS] AC2 no extra    — no extra image_ids
[PASS] AC3 no miss     — no missing image_ids
[PASS] AC4 no dup      — no duplicate image_ids
[PASS] AC5 no null     — no null values
[PASS] AC6 no newline  — no newline characters in text fields
[PASS] AC7 col order   — correct column order: image_id, ocr_text, product_name
```

If any check fails, the file is not written and the error is printed for debugging.

---

## Smoke Test Results

| OCR input text | Predicted output |
|---|---|
| `HALONG CANFOCO Pate Cột Đèn` | `Ha Long Canfoco Pate Cột Đèn` |
| `ate Jate Cotcen 130 tan thit lon` | `Pate Cột Đèn` |
| `HIGHLANDS COFFEE tra sen vang` | `Highlands Coffee Trà Sen Vàng` |
| `hophalong thu hoi san pham` | `Đồ Hộp Hạ Long` |
| `HiPP Combiotic organic milk` | `HiPP Combiotic` |
| `tiktok viral capcut fyp news` | *(empty — filtered as noise)* |
