# URA Hackathon 2026 – OCR & Product Recognition Pipeline v5.0

> **Competition:** [The 2nd URA Hackathon 2026](https://www.kaggle.com/competitions/the-2nd-ura-hackathon)  
> **Task:** Given product images (social-media screenshots, shelf photos, ads), extract OCR text and identify the **brand name** and **product name** separately.  
> **Metric:**
> ```
> Score = 0.40 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product
> ```

---

## 📓 Notebook

**[`have_train.ipynb`](have_train.ipynb)** – Final submission notebook  
Executed: **2026-06-23** · Runtime: ~25 min on Kaggle CPU (4 threads, 1 202 private-test images)

---

## 🔑 Pipeline Overview

```
Image
  └─► PaddleOCR (CPU, latin_PP-OCRv3 + ch_PP-OCRv4 detector)
          └─► clean_social_ocr()           ← strip @, #, URLs, timestamps
              └─► _is_ocr_noise_only()     ← discard pure-noise text
                    │
                    ├─► extract_by_rules()     ← 46 brand regex rules
                    │         └─► post_process_prediction()
                    │
                    ├─► detect_brand_in_ocr()  ← BRAND_REGISTRY longest-alias match
                    │
                    ├─► BrandClassifier + ProductClassifier (TF-IDF + LogReg)
                    │
                    └─► split_brand_product()  ← MERGED_SPLIT → (brand_name, product_name)
                              └─► submission.csv
```

---

## 🔑 Key Design Decisions

### 1. Phase 2 Schema Auto-detection
`_find_first()` recursively searches for `private_test.csv` → `test.csv` in priority order.  
`_find_images_dir()` picks the directory with the most images, preferring folders with "test" in the path to avoid accidentally picking train images.  
Submission columns are read directly from `sample_submission_private.csv` so the code adapts automatically to any column name changes.

### 2. Brand / Product Split Layer
`MERGED_SPLIT` maps full predicted labels into separate `(brand_name, product_name)` pairs:

```python
'Ha Long Canfoco Pate Cột Đèn' → ('Ha Long Canfoco', 'Pate Cột Đèn')
'Đồ Hộp Hạ Long'               → ('Hạ Long', 'Đồ Hộp')
'Nestlé NAN'                   → ('Nestlé', 'NAN')
'Vinamilk Dielac'              → ('Vinamilk', 'Dielac')
```

`split_brand_product()` falls back to longest-prefix match against `KNOWN_BRANDS` for unseen labels.

### 3. Brand Registry & Train-driven Discovery
`BRAND_REGISTRY` is seeded from 30 `KNOWN_BRANDS` entries.  
If `train_labels.csv` has a `brand_name` column, `build_brand_knowledge_from_train()` auto-registers new brands (min 2 samples, alias ≥ 4 chars).  
`detect_brand_in_ocr()` scans OCR text for registered aliases:
- Multi-word aliases: substring match
- Single-word aliases: whole-word match only (reduces false positives)

### 4. Generic Product Token Guard
`GENERIC_PRODUCT_TOKENS` = `{'cot', 'hop', 'milk', 'sua', 'gold', ...}`  
Single-token predictions that are generic words are suppressed to prevent noisy labels like `"Cột"` or `"Hộp"` appearing alone.

### 5. 46 Brand Rules
Hierarchical regex patterns with OCR typo tolerance:

| Group | Covered brands / patterns |
|---|---|
| Ha Long Canfoco + Pate Cột Đèn | `canfoco`, `canfuco`, `halong canf`, `cotcen`, `cotden` |
| Đồ Hộp Hạ Long | `đồ hộp hạ long`, `hophalong` |
| Pate Cột Đèn Hải Phòng | `cot den hai phong`, `cotd hai` |
| Dairy / Infant Formula | Vinamilk, TH True Milk, Dutch Lady, Nutifood, Abbott, Nestlé NAN, Aptamil, HiPP, Friso, Meiji |
| Coffee & Beverage | Highlands Coffee, The Coffee House |
| Meat & Pate | Vissan, Hafi, CP, Ba Huân, San Hà |
| Other | Acnes, Ba Vì, Fami, Yomost, Anlene, Anchor … |

### 6. OCR Engine – PaddleOCR (CPU)
- Detector: `ch_PP-OCRv4_det_infer`
- Recogniser: `latin_PP-OCRv3_rec_infer`
- Confidence threshold: **0.35**
- Image preprocessing: resize ≤ 1 280 px, contrast ×1.35, sharpen
- 4 parallel workers via `joblib` threading backend

### 7. ML Predictors
| Model | Purpose |
|---|---|
| `BrandClassifier` (TF-IDF char n-gram 2–4 + LogReg) | Predict brand from OCR when rules don't match |
| `ProductClassifier` (same architecture, 249 classes) | Predict product line from OCR |
| `SklearnPredictor` (L4, max_feat=6000, C=1.5) | Higher-capacity LogReg fallback |
| `KNNPredictor` (k=5, cosine similarity, thr=0.60) | Nearest-neighbour fallback |

All models are trained on `train_labels.csv` at notebook startup.

---

## 📊 Execution Summary (Phase 2 – Private Test)

| Stat | Value |
|---|---|
| Test images | 1 202 |
| OCR successful | ~1 200 |
| Runtime | ~25 min (4 CPU threads) |
| Speed | ~0.65 s / image |

---

## 🗂️ Dataset Structure (Kaggle – Phase 2)

```
/kaggle/input/competitions/the-2nd-ura-hackathon/
    train_labels.csv                                   # 4 892 rows
    phase2_dataset/phase2_dataset/
        private_test.csv                               # 1 202 rows (image_id)
        sample_submission_private.csv                  # image_id, ocr_text, brand_name, product_name
        images/images/                                 # 1 202 JPG test images (priv_h_*.jpg)
```

> All paths are auto-discovered. No manual editing needed.

---

## 🚀 How to Run on Kaggle

1. Add the competition dataset (Competitions tab → Add data).
2. Ensure `RUNNING_ON_KAGGLE = True` (default).
3. Delete any leftover checkpoint before re-running:
   ```python
   import os; os.remove('/kaggle/working/checkpoint.csv')
   ```
4. **Run All** cells top-to-bottom.
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

All installed automatically in the first cell.

---

## 🧩 Notebook Cell Guide

| Cell | Purpose |
|---|---|
| Install | Parallel pip: `paddlepaddle`, `scikit-learn`, `tqdm`, `paddleocr` |
| Config | Auto-discover paths; detect schema columns; load train/test CSV |
| Image index | Build `IMAGE_PATH_INDEX` for fast path lookup |
| Rules + split layer | 46 brand rules; `BRAND_REGISTRY`; `MERGED_SPLIT`; `split_brand_product()`; smoke test 13/13 |
| OCR engine | PaddleOCR init; image preprocessing; smoke test on first image |
| Dual predictor v5 | `BrandClassifier` + `ProductClassifier` (TF-IDF + LogReg, 249 classes) |
| Sklearn / KNN | `SklearnPredictor` (L4) + `KNNPredictor` (L5) for ML fallback |
| Pipeline | `predict_product()` orchestrator: rules → ML → split → guard |
| Cross-val | Local F1 evaluation on 20% hold-out |
| Main loop | Parallel OCR (joblib) → predict → checkpoint every 50 imgs |
| Export | 7 acceptance checks → save `submission.csv` aligned to `sample_submission_private.csv` |
