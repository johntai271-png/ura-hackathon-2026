# 🏆 TANAHI — Hệ thống Nhận diện Sản phẩm từ Ảnh Mạng Xã hội
### URA Hackathon 2026 | Nhóm TANAHI | HCMUT

> **Điểm thi:** `Score = 0.40 × F1_brand + 0.35 × (1 - CER) + 0.25 × F1_product`  
> **Môi trường:** Kaggle CPU Kernel · Python 3.12 · 30 GB RAM

---

## 📋 Mục lục
- [Bài toán](#bài-toán)
- [Ý tưởng hệ thống](#ý-tưởng-hệ-thống)
- [Kiến trúc pipeline](#kiến-trúc-pipeline)
- [Cài đặt & Chạy lại](#cài-đặt--chạy-lại)
- [Chi tiết từng module](#chi-tiết-từng-module)
- [Kết quả](#kết-quả)

---

## 🎯 Bài toán

Cuộc thi yêu cầu xây dựng pipeline AI **nhẹ, không cần GPU** để trích xuất thông tin sản phẩm từ ảnh mạng xã hội.

**Input:** Ảnh TikTok/Instagram (JPG, PNG, GIF, video)

**Output (CSV):**

| Cột | Mô tả | Ví dụ |
|---|---|---|
| `image_id` | ID ảnh | `priv_h_0001` |
| `ocr_text` | Toàn bộ text trong ảnh | `"Sữa tươi Vinamilk Flex 180ml"` |
| `brand_name` | Tên thương hiệu | `"Vinamilk"` |
| `product_name` | Tên dòng sản phẩm | `"Flex"` |

**Thách thức:** ảnh mạng xã hội nhiễu · watermark · text nhỏ/nghiêng · song ngữ Việt-Anh · GIF/video · chỉ CPU

---

## 💡 Ý tưởng hệ thống

### Vấn đề & Giải pháp

**1. OCR tiếng Việt không chính xác**
- PaddleOCR đọc sai dấu (Sữa → Sua), VietOCR cần biết vị trí từng từ
- → **Dual-Engine:** Paddle detect bounding boxes → VietOCR recognize từng crop

**2. ML thuần không generalize**
- ~5,000 mẫu train không đủ cho brand/product mới
- → **3 tầng prediction có ưu tiên:** Rules → ML → Layout

**3. Model "bịa" brand**
- Classifier có thể gán brand không có trong ảnh
- → **Anti-hallucination:** kiểm tra token brand có trong OCR text không

**4. Session Kaggle timeout**
- Chạy 1,200 ảnh mất 30-45 phút, dễ bị ngắt
- → **Checkpoint mỗi 50 ảnh**, tự resume khi chạy lại

---

## 🏗️ Kiến trúc Pipeline

```
📸 INPUT (JPG/PNG/GIF/Video)
       │
       ▼
┌─────────────────┐
│  PREPROCESSING  │  Resize + Contrast ×1.45 + Sharpen
│  (Cell 7b)      │  GIF/Video: chọn frame nét nhất
└────────┬────────┘  (Laplacian variance)
         │
         ▼
┌──────────────────────────────────────┐
│        OCR DUAL ENGINE (Cell 7)      │
│                                      │
│  PaddleOCR PP-OCRv4  →  Bounding   │
│  (Detection Only)        Boxes       │
│           ↓                          │
│  Crop từng box (padding 3px)         │
│           ↓                          │
│  VietOCR vgg_seq2seq → Text/crop    │
│  (Recognition)                       │
│           ↓                          │
│  Sort boxes (reading order)          │
│  Bilingual Post-Processor            │
│  → ocr_text (cleaned)               │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│   3-LAYER PREDICTION (Cell 3,4,5,6) │
│                                      │
│  LAYER 1 — Rule-Based (ưu tiên)    │
│  ├─ 101 BRAND_RULES regex           │
│  ├─ 80+ PRODUCT_LINE_PATTERNS       │
│  └─ OCR Typo correction             │
│           ↓ (nếu trống)             │
│  LAYER 2 — Machine Learning         │
│  ├─ BrandClassifier (TF-IDF+LogReg) │
│  ├─ ProductClassifier (gate+clf)    │
│  └─ BrandKNN (cosine k=5)           │
│           ↓ (bổ sung nếu ML yếu)   │
│  LAYER 3 — Layout Analysis          │
│  ├─ Canny Edge → Red Box            │
│  ├─ Prominence Score ≥ 30           │
│  └─ CHỈ BÙ, không ghi đè ML        │
└────────┬─────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ POST-PROCESSING │  reconcile_brand_product
│ (Cell 9)        │  Social noise filter
└────────┬────────┘  Generic token filter
         │
         ▼
✅ submission.csv (7 AC checks)
```

---

## 🚀 Cài đặt & Chạy lại

### Yêu cầu
- Kaggle Notebook, CPU, No GPU, Internet ON
- RAM ≥ 16 GB

### Bước 1: Thêm Dataset
Kaggle Notebook → **+ Add Input** → Tìm **"The 2nd URA Hackathon 2026"** → Add

### Bước 2: Upload Notebook
Upload file `tanahi-demo-ver3.ipynb` lên Kaggle.

### Bước 3: Cấu hình Session
```
Accelerator:  None (CPU)
Internet:     ON  ← bắt buộc (download VietOCR model lần đầu ~100MB)
Persistence:  No persistence
```

### Bước 4: Chạy

> ⚠️ **Trước khi chạy:** Bấm **Run → Restart & Clear All Outputs** để xóa cache cũ.

Bấm **Run All** → đợi ~25–45 phút.

Log mẫu khi chạy thành công:
```
CPU threads: 4 | pip workers: 4
Installing packages (v6.0)...
Install done. Pillow pinned to 9.5.0 ✓
PaddleOCR detector OK | conf>0.35
VietOCR vgg_seq2seq loaded (CPU) ✓
Rules v4.0 | 101 rules | Smoke test: 13/13 ✓
BrandClassifier trained: 4892 samples ✓
[50/1202] checkpoint saved...
[1202/1202] done ✓
AC1 ✓ AC2 ✓ AC3 ✓ AC4 ✓ AC5 ✓ AC6 ✓ AC7 ✓
Submission saved → /kaggle/working/submission.csv
```

### Bước 5: Download
Kaggle → **Output** tab → Download `submission.csv`

### Resume sau timeout
Nếu bị ngắt giữa chừng, notebook tự resume từ checkpoint. Chỉ cần **Run All** lại.

---

## 🔩 Chi tiết từng module

### Cell 1 — Cài đặt
```python
paddlepaddle          # PaddleOCR backend (CPU)
paddleocr>=2.7,<3     # OCR detection framework
vietocr               # Vietnamese text recognition
underthesea           # Vietnamese NLP
scikit-learn          # ML classifiers
Pillow==9.5.0         # Pinned! Pillow 10+ conflict với PaddleOCR
tqdm                  # Progress bar
```
> ⚠️ `Pillow==9.5.0` phải được cài **sau cùng** để override bản Pillow mới mà VietOCR kéo vào.

### Cell 2 — Auto-discovery đường dẫn
Tự động tìm file CSV và thư mục ảnh, không cần hardcode. Hoạt động với Phase 1 và Phase 2.

### Cell 3 — 101 BRAND_RULES + Utilities
Rules regex handcrafted cho các thương hiệu phổ biến:
- Sữa: Vinamilk, TH True Milk, Dutch Lady, Nutifood, Aptamil, HiPP...
- Thực phẩm: Ha Long Canfoco, Vissan, Pate Cột Đèn Hải Phòng...
- Khác: Nestlé, Abbott, Highlands Coffee...

Kèm typo correction và bộ filter social noise (TikTok caption, hashtag...).

### Cell 4 — Layout Pipeline
- `find_red_box()`: Canny + dilate + contour → xác định vùng sản phẩm
- `extract_universal_product_info_v5()`: prominence score từng text line
- `bilingual_post_processor()`: sửa OCR lỗi song ngữ

### Cell 5 — Train ML Models
Huấn luyện trực tiếp từ `train_labels.csv`:
- **BrandClassifier**: TF-IDF char ngram(2-4) + LogReg, threshold 0.62
- **ProductClassifier**: 2-stage gate + classifier, threshold 0.45
- **BrandKNN**: cosine similarity k=5

### Cell 6 — Predict & Evaluate
`predict_labels()` kết hợp 3 tầng. `evaluate_pipeline()` tính F1 + Exact Match.

### Cell 7 — Dual OCR Engine
PaddleOCR detect → crop → VietOCR recognize → sort reading order.

### Cell 7b — Media Handler
GIF/Video/WebP → sample frames → chọn frame nét nhất bằng Laplacian variance.

### Cell 8 — Inference + Checkpoint
Chạy toàn bộ test set, checkpoint mỗi 50 ảnh.

### Cell 9 — Post-processing
Bỏ token lặp liền kề trong product_name.

### Cell 10 — Export + Validate
7 assertion checks AC1-AC7 → ghi `submission.csv`.

---

## 📈 Kết quả

| Metric | Ước tính |
|---|---|
| F1 Brand | ~0.50 |
| 1 - CER (OCR) | ~0.75 |
| F1 Product | ~0.60 |
| **Score tổng** | **~0.60** |

---

## 📚 References
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [The 2nd URA Hackathon 2026](https://kaggle.com/competitions/the-2nd-ura-hackathon)

---
*TANAHI Team — HCMUT URA Research Group · 2026*
