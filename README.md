# URA Hackathon 2026 – OCR & Product Recognition Pipeline v5.0

> **Competition:** [The 2nd URA Hackathon 2026](https://www.kaggle.com/competitions/the-2nd-ura-hackathon)  
> **Task:** Given product images (social-media screenshots, shelf photos, ads), extract OCR text and identify the **brand name** and **product name** separately.  
> **Metric (Phase 2 – new scoring formula):**
> ```
> Score = 0.40 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product
> ```

---

## 📓 Notebook

**[`have_train.ipynb`](have_train.ipynb)** – Final submission notebook (Pipeline v5.0)  
Executed: **2026-06-23** · Runtime: ~25 min on Kaggle CPU (4 threads, 1 202 private-test images)

---

## 🆕 What changed from v4 → v5

| | v4.0 | v5.0 |
|---|---|---|
| **Output columns** | `image_id, ocr_text, product_name` | `image_id, ocr_text, brand_name, product_name` |
| **Scoring** | `0.6×F1_prod + 0.4×(1-CER)` | `0.4×F1_brand + 0.35×(1-CER) + 0.25×F1_prod` |
| **Test set** | 2 006 images (Phase 1) | 1 202 images (`priv_h_*`, Phase 2) |
| **Anti-hallucination** | `_assign_readable_brand_tokens` (cut multi-word labels) | Fixed: `extract_by_rules` returns full matched label directly |
| **Brand/product split** | Merged into one string | Explicit `MERGED_SPLIT` lookup + `split_brand_product()` |
| **Schema auto-detect** | Hard-coded columns | Auto-discovers test/train CSV columns & submission format |
| **Path discovery** | Hard-coded paths | `_find_first()` + `_find_images_dir()` – robust across phases |
| **Brand ML** | Not present | `BrandClassifier` (TF-IDF + LogReg) trained separately |
| **Brand registry** | Rule-only | `BRAND_REGISTRY` seeded from `KNOWN_BRANDS` + train data |

---

## 🔑 Pipeline Overview

```
Image
  └─► PaddleOCR (CPU, latin_PP-OCRv3 + ch_PP-OCRv4 detector)
          └─► clean_social_ocr()           ← strip @, #, URLs, timestamps
              └─► _is_ocr_noise_only()     ← discard pure-noise text
                    │
                    ├─► extract_by_rules() ─► BRAND_RULES (46 regex, no token-cut bug)
                    │         └─► post_process_prediction()   ← Pate/Nestlé/Vinamilk upgrade
                    │
                    ├─► detect_brand_in_ocr()     ← BRAND_REGISTRY longest-alias match
                    │
                    ├─► BrandClassifier (TF-IDF char n-gram + LogReg)
                    │
                    └─► split_brand_product()     ← MERGED_SPLIT lookup → (brand, product)
                              └─► submission.csv  (image_id, ocr_text, brand_name, product_name)
```

---

## 🔑 Key Design Decisions

### 1. Phase 2 Schema Auto-detection
`_find_first()` recursively searches for `private_test.csv` → `test.csv` (in priority order).  
`_find_images_dir()` picks the directory with the most images, preferring folders with "test" in the path to avoid accidentally picking train images.  
Submission columns are read directly from `sample_submission_private.csv` so the code works regardless of column name changes between phases.

### 2. Bug Fix: Anti-hallucination Token Cut
v4's `_assign_readable_brand_tokens` would cut multi-token labels like `"Pate Cột Đèn"` → `"Cột"` because intermediate tokens weren't in the brand vocabulary.  
**v5 fix:** `extract_by_rules()` returns the full matched label directly (trust the regex, not the token filter).

### 3. Brand / Product Split Layer
New `MERGED_SPLIT` dictionary maps historical merged labels (from v4 training data) into `(brand_name, product_name)` pairs:
```python
'Ha Long Canfoco Pate Cột Đèn' → ('Ha Long Canfoco', 'Pate Cột Đèn')
'Đồ Hộp Hạ Long'               → ('Hạ Long', 'Đồ Hộp')
'Nestlé NAN'                   → ('Nestlé', 'NAN')
```
`split_brand_product()` falls back to longest-prefix match against `KNOWN_BRANDS`.

### 4. Brand Registry & Train-driven Discovery
`BRAND_REGISTRY` is seeded from 30 `KNOWN_BRANDS` entries.  
If `train_labels.csv` has a `brand_name` column, `build_brand_knowledge_from_train()` auto-registers new brands (min 2 samples, alias ≥ 4 chars).  
`detect_brand_in_ocr()` scans the OCR text for any registered alias (multi-word: substring match; single-word: whole-word match only → reduces false positives).

### 5. Generic Product Token Guard
`GENERIC_PRODUCT_TOKENS` = `{'cot', 'hop', 'milk', 'sua', 'gold', ...}`  
Any single-token prediction that is a generic word is suppressed to prevent noisy labels like `"Cột"` or `"Hộp"`.

### 6. OCR Engine – PaddleOCR (CPU, latin_PP-OCRv3)
- Detector: `ch_PP-OCRv4_det_infer`
- Recogniser: `latin_PP-OCRv3_rec_infer`
- Confidence threshold: **0.35** (boxes below this are discarded)
- Image preprocessing: resize to ≤ 1 280 px, contrast ×1.35, sharpen
- Parallel: 4 workers via `joblib` threading backend

### 7. Social-Noise Filter
Strips `@mentions`, `#hashtags`, URLs, timestamps, and long digit runs.  
`_is_ocr_noise_only()` discards text that contains only social-media keywords (TikTok, Reels, Capcut, viral …) with no meaningful product token.

### 8. 46 Brand Rules
Hierarchical regex patterns with OCR typo tolerance:

| Group | Patterns |
|---|---|
| Ha Long Canfoco + Pate Cột Đèn | `canfoco`, `canfuco`, `halong canf`, `cotcen`, `cotden` |
| Đồ Hộp Hạ Long | `đồ hộp hạ long`, `hophalong`, `hop ha long` |
| Pate Cột Đèn Hải Phòng | `cot den hai phong`, `cotd hai` |
| Dairy / Infant Formula | Vinamilk, TH True Milk, Dutch Lady, Nutifood, Abbott (Ensure/PediaSure/Similac), Nestlé NAN, Aptamil, HiPP, Friso, Meiji |
| Coffee & Beverage | Highlands Coffee (+ product lines), The Coffee House |
| Meat & Pate | Vissan, Hafi, CP, Ba Huân, San Hà |
| Other | Acnes, Ba Vì, Fami, Yomost, Anlene, Anchor … |

---

## 📊 Execution Summary (Phase 2)

| Stat | Value |
|---|---|
| Test images | 1 202 |
| OCR successful | ~1 200 |
| Product predicted | ~530 |
| Runtime | ~25 min (4 CPU threads) |
| Speed | ~0.65 s / image |

---

## 🗂️ Dataset Structure (Kaggle – Phase 2)

```
/kaggle/input/competitions/the-2nd-ura-hackathon/
    train_labels.csv                                  # 4 892 rows: image_id, ocr_text, product_name
    phase2_dataset/phase2_dataset/
        private_test.csv                              # 1 202 rows: image_id
        sample_submission_private.csv                 # format: image_id, ocr_text, brand_name, product_name
        images/images/                                # 1 202 JPG private-test images (priv_h_*.jpg)
```

> The notebook auto-discovers all paths. No manual path editing needed.

---

## 🚀 How to Run on Kaggle

1. Add the competition dataset to your Kaggle notebook (Competitions tab → Add data).
2. Set `RUNNING_ON_KAGGLE = True` (already the default).
3. **Delete any leftover checkpoint before re-running:**
   ```python
   import os
   os.remove('/kaggle/working/checkpoint.csv')
   ```
4. Run all cells top-to-bottom. Progress is checkpointed every 50 images.
5. Download `/kaggle/working/submission.csv` and submit.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `paddlepaddle` (CPU) | PaddleOCR backend |
| `paddleocr ≥ 2.7, < 3` | OCR pipeline |
| `scikit-learn` | TF-IDF + Logistic Regression |
| `joblib` | Parallel OCR workers |
| `tqdm` | Progress bars |
| `Pillow` | Image loading & preprocessing |
| `pandas`, `numpy` | Data handling |

All installed automatically in the first notebook cell.

---

## 🧩 Notebook Cell Guide

| Cell | Purpose |
|---|---|
| Install | Parallel pip: `paddlepaddle`, `scikit-learn`, `tqdm`, `paddleocr` |
| Config | Auto-discover paths; detect schema (brand_name / product_name columns) |
| Image index | Build `IMAGE_PATH_INDEX` for fast lookup |
| Rules + split layer | 46 brand rules; `BRAND_REGISTRY`; `split_brand_product()`; `MERGED_SPLIT`; smoke test 13/13 |
| OCR engine | PaddleOCR init; image preprocessing; smoke test on first image |
| Legacy predictor | `ProductPredictor` class (kept for reference, **disabled** in v5) |
| Dual predictor v5 | `BrandClassifier` + `ProductClassifier` (TF-IDF + LogReg, 249 classes) |
| Sklearn / KNN | `SklearnPredictor` (L4) + `KNNPredictor` (L5) for ML fallback |
| Pipeline | `predict_product()` orchestrator: rules → ML → split → guard |
| Cross-val | Local F1 evaluation on 20% hold-out |
| Main loop | Parallel OCR (joblib) → predict → checkpoint every 50 imgs |
| Export | 7 acceptance checks → save `submission.csv` aligned to `sample_submission_private.csv` |
