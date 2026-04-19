<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/ASVspoof-2019%20%7C%202021-1F3864?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-In%20Progress-F59E0B?style=for-the-badge"/>

# 🎙️ Voice Spoofing Detection

### Phát hiện giọng nói giả mạo bằng Deep Learning trên tập dữ liệu ASVspoof

*Bài tập lớn cuối kỳ — Môn Deep Learning*

</div>

---

## 📌 Giới thiệu

Hệ thống xác thực bằng giọng nói (Automatic Speaker Verification — ASV) ngày càng phổ biến trong các ứng dụng bảo mật: mở khoá điện thoại, xác thực ngân hàng, kiểm soát truy cập. Tuy nhiên, chúng đang bị đe dọa bởi các kỹ thuật **giả mạo giọng nói**:

| Loại tấn công | Mô tả | Ví dụ |
|---|---|---|
| **TTS** (Text-to-Speech) | AI tổng hợp giọng từ văn bản | Tacotron, FastSpeech |
| **Voice Conversion** | Biến giọng người này thành người khác | StarGAN-VC, AutoVC |
| **Replay Attack** | Phát lại bản ghi âm giọng thật | Thu âm & phát qua loa |

Dự án này xây dựng và so sánh các mô hình Deep Learning để **phân biệt giọng thật (bonafide) và giọng giả (spoof)** trên bộ dữ liệu chuẩn ASVspoof 2019 và đánh giá khả năng tổng quát hoá trên ASVspoof 2021.

---

## 👥 Nhóm thực hiện

| Thành viên | MSSV | Vai trò | Nhiệm vụ chính |
|---|---|---|---|
| Đặng Quốc Dũng | B22DCKH019 | **Feature Engineer** | Trích xuất đặc trưng MFCC / LFCC / Spectrogram, visualisation |
| Bùi Đức Đại | B22DCKH025 | **Model Developer** | Xây dựng & huấn luyện CNN / ResNet / LCNN |
| Nguyễn Thế Dương | B22DCKH023 | **Evaluator & Analyst** | Tính EER, generalization test, error analysis |

> 🎓 **Giảng viên hướng dẫn:** Ts.Nguyễn Xuân Đức — Môn: Deep Learning

---

## 🎯 Mục tiêu nghiên cứu

Dự án trả lời 3 câu hỏi nghiên cứu:

1. **Đặc trưng nào phù hợp nhất?** So sánh MFCC vs LFCC vs Log-mel Spectrogram trong bài toán spoof detection.
2. **Mô hình nào tốt nhất?** So sánh CNN (baseline) vs ResNet vs LCNN về accuracy, EER và mức độ overfit.
3. **Khả năng tổng quát hoá ra sao?** Đánh giá mô hình train trên ASVspoof 2019 khi test trên ASVspoof 2021 (unseen vocoders).

---

## 📂 Cấu trúc dự án

```
voice-spoofing-detection/
│
├── 📁 data/
│   ├── raw/                    # Dữ liệu gốc ASVspoof (không push lên Git)
│   │   ├── ASVspoof2019/
│   │   │   ├── LA/
│   │   │   │   ├── ASVspoof2019_LA_train/
│   │   │   │   ├── ASVspoof2019_LA_dev/
│   │   │   │   └── ASVspoof2019_LA_eval/
│   │   └── ASVspoof2021/
│   │       └── LA/
│   └── features/               # Đặc trưng đã trích xuất (.npy)
│       ├── mfcc/
│       ├── lfcc/
│       └── spectrogram/
│
├── 📁 src/
│   ├── features/
│   │   ├── extract_mfcc.py     # Trích xuất MFCC
│   │   ├── extract_lfcc.py     # Trích xuất LFCC
│   │   └── extract_spec.py     # Trích xuất Spectrogram
│   │
│   ├── models/
│   │   ├── cnn.py              # CNN baseline
│   │   ├── resnet.py           # ResNet
│   │   └── lcnn.py             # Light CNN (LCNN)
│   │
│   ├── utils/
│   │   ├── dataset.py          # ASVspoofDataset (PyTorch)
│   │   ├── metrics.py          # Tính EER, confusion matrix
│   │   └── visualize.py        # Vẽ spectrogram, learning curve
│   │
│   ├── train.py                # Script huấn luyện chính
│   ├── evaluate.py             # Đánh giá mô hình
│   └── predict.py              # Demo dự đoán 1 file .wav
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb            # Khám phá dữ liệu (EDA)
│   ├── 02_feature_analysis.ipynb  # Phân tích và so sánh đặc trưng
│   └── 03_results_analysis.ipynb  # Phân tích kết quả thí nghiệm
│
├── 📁 outputs/
│   ├── checkpoints/            # Model checkpoint (.pth)
│   ├── figures/                # Biểu đồ, spectrogram, confusion matrix
│   └── results/                # CSV kết quả thí nghiệm
│
├── 📁 docs/
│   ├── timeline.docx           # Timeline & phân công công việc
│   └── report.docx             # Báo cáo cuối kỳ
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Cài đặt môi trường

### Yêu cầu hệ thống
- Python 3.9+
- CUDA 11.8+ (khuyến nghị, có thể chạy CPU nhưng chậm hơn)
- RAM ≥ 16 GB (xử lý toàn bộ dataset)
- Dung lượng ≥ 50 GB (data + features)

### Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/<username>/voice-spoofing-detection.git
cd voice-spoofing-detection

# 2. Tạo môi trường ảo
conda create -n spoof python=3.9
conda activate spoof

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
librosa>=0.10.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
tqdm>=4.65.0
soundfile>=0.12.0
```

---

## 📥 Tải dữ liệu

Dữ liệu ASVspoof **không được lưu trên repo** do kích thước lớn. Tải tại:

| Dataset | Link | Kích thước |
|---|---|---|
| ASVspoof 2019 LA | [asvspoof.org](https://datashare.ed.ac.uk/handle/10283/3336) | ~9 GB |
| ASVspoof 2021 LA | [zenodo.org/record/4837263](https://zenodo.org/record/4837263) | ~19 GB |

Sau khi tải, giải nén vào thư mục `data/raw/` theo đúng cấu trúc ở trên.

---

## 🔬 Pipeline thực nghiệm

### Bước 1 — Trích xuất đặc trưng

```bash
# Trích xuất MFCC (40 coefficients, frame 25ms, hop 10ms)
python src/feature/extract_mfcc.py

# Trích xuất LFCC (40 coefficients, linear filterbank)
python src/feature/extract_lfcc.py

# Trích xuất Log-mel Spectrogram (64 mel bins, 128 time frames)
python src/feature/extract_spec.py
```

Feature `.npy` sau khi trích xuất được chuẩn hoá theo layout sau, và `--data_root` phải trỏ tới thư mục cha `data`:

```text
data/
  features/
    output_npy_2019/
      output_mfcc/
        train/
        dev/
        eval/
      output_lfcc/
      output_spectrogram/
    output_npy_2021/
      output_mfcc/
        eval_2021/
      output_lfcc/
      output_spectrogram/
      labels_eval_2021.csv
```

### Bước 2 — Huấn luyện mô hình

```bash
# CNN baseline với MFCC
python train.py --feature mfcc --model cnn --epochs 50 --batch_size 256 --num_workers 4 --amp --device cuda --data_root data

# ResNet với LFCC
python train.py --feature lfcc --model resnet --epochs 50 --batch_size 256 --num_workers 4 --amp --device cuda --data_root data

# LCNN với Spectrogram
python train.py --feature spectrogram --model lcnn --epochs 50 --batch_size 128 --num_workers 4 --amp --device cuda --data_root data

# Chạy toàn bộ 9 tổ hợp (3 feature × 3 model)
bash scripts/run_all_experiments.sh
```

### Bước 3 — Đánh giá

```bash
# Đánh giá trên ASVspoof 2019 eval set
python evaluate.py --checkpoint outputs/checkpoints/baseline_cnn_mfcc/best.ckpt --eval_2019 --data_root data

# Generalization test trên ASVspoof 2021
python evaluate.py --checkpoint outputs/checkpoints/baseline_cnn_mfcc/best.ckpt --eval_2021 --data_root data

# Đánh giá đồng thời trên 2019 và 2021 với labels 2021 explicit
python evaluate.py --checkpoint outputs/checkpoints/baseline_cnn_mfcc/best.ckpt --eval_2019 --eval_2021 --data_root data --eval_2021_labels data/features/output_npy_2021/labels_eval_2021.csv
```

### Bước 4 — Demo

```bash
# Dự đoán 1 file .wav bất kỳ
python predict.py --audio path/to/your/audio.wav --checkpoint outputs/checkpoints/best_model.pth
```

---

## 🧪 Các mô hình so sánh

### Đặc trưng đầu vào

| Đặc trưng | Chiều | Mô tả |
|---|---|---|
| **MFCC** | (40, T) | Mel-frequency cepstral coefficients — phổ biến, compact |
| **LFCC** | (40, T) | Linear-frequency cepstral coefficients — tốt hơn cho spoof detection |
| **Log-mel Spectrogram** | (64, 128) | Biểu diễn 2D giàu thông tin về tần số |

### Kiến trúc mô hình

| Mô hình | Mô tả | Tham số |
|---|---|---|
| **CNN** | 3 lớp Conv + BatchNorm + ReLU + Pool, Dropout 0.3, FC | ~500K |
| **ResNet** | ResNet-18 adapted cho 2D feature, residual connections | ~11M |
| **LCNN** | Light CNN với Max-Feature-Map activation | ~1.5M |

### Metric đánh giá

- **Accuracy** — tỷ lệ phân loại đúng (lưu ý: không đủ khi data imbalanced)
- **EER (Equal Error Rate)** — metric chuẩn của ASVspoof, càng thấp càng tốt
- **Confusion Matrix** — phân tích lỗi theo từng loại

---

## 📊 Kết quả thực nghiệm

> ⚠️ *Bảng kết quả sẽ được cập nhật sau khi chạy xong thí nghiệm*

### Bảng so sánh chính (EER% — càng thấp càng tốt)

| Đặc trưng | CNN | ResNet | LCNN |
|---|---|---|---|
| MFCC | — | — | — |
| LFCC | — | — | — |
| Spectrogram | — | — | — |
| **Tốt nhất** | | | |

### Generalization test (EER% trên ASVspoof 2021)

| Mô hình | EER 2019 | EER 2021 | Gap |
|---|---|---|---|
| Best CNN | — | — | — |
| Best ResNet | — | — | — |
| Best LCNN | — | — | — |

---

## 💡 Những phân tích chính (Key Insights)

> Phần này sẽ được điền sau khi có kết quả thực nghiệm.

- **Tại sao LFCC tốt hơn MFCC?** Giọng giả thường có artifacts ở vùng tần số cao — LFCC dùng linear filterbank nên giữ lại thông tin này, trong khi mel scale của MFCC nén vùng tần số cao lại.
- **Mô hình nào bị overfit?** Sẽ phân tích qua learning curve và khoảng cách train/val accuracy.
- **Generalization gap là bao nhiêu?** Mức độ EER tăng khi test trên 2021 nói lên điều gì về khả năng tổng quát hoá của mô hình.

---

## 📁 Bỏ qua (`.gitignore`)

```
# Dữ liệu lớn
data/raw/
data/features/

# Model checkpoint
outputs/checkpoints/

# Cache Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Môi trường
.env
wandb/

# IDE
.vscode/
.idea/
```

---

## 📚 Tài liệu tham khảo

1. Nautsch et al., *"ASVspoof 2019: A Large-scale Public Database..."*, Computer Speech & Language, 2021.
2. Liu et al., *"ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild"*, IEEE/ACM TASLP, 2023.
3. Cheng et al., *"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"*, ICASSP 2022.
4. Lavrentyeva et al., *"STC Antispoofing Systems for the ASVspoof2019 Challenge"*, Interspeech 2019.
5. Wu et al., *"Light CNN for Deep Face Representation with Noisy Labels"*, IEEE TIFS, 2018.
6. He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016.
7. Sahidullah et al., *"A Comparison of Features for Synthetic Speech Detection"*, Interspeech 2015.

---

<div align="center">

*Bài tập lớn cuối kỳ — Môn Deep Learning*

</div>
