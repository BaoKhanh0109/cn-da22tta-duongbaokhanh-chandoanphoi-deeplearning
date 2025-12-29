# Hệ Thống Hỗ Trợ Chẩn Đoán Bệnh Phổi Từ Ảnh X-Quang

## 📋 Giới thiệu

Dự án xây dựng hệ thống AI hỗ trợ chẩn đoán bệnh phổi từ ảnh X-quang ngực thẳng, sử dụng mô hình Deep Learning kết hợp **EfficientNet-B4** và **Graph Convolutional Network (GCN)**. Hệ thống được huấn luyện trên bộ dữ liệu **VinDr-CXR** và có khả năng phát hiện **14 loại bệnh lý phổi**.

### 🎯 Các bệnh lý được hỗ trợ chẩn đoán:
| STT | Tiếng Anh | Tiếng Việt |
|-----|-----------|------------|
| 1 | Aortic enlargement | Phình động mạch chủ |
| 2 | Atelectasis | Xẹp phổi |
| 3 | Calcification | Vôi hóa |
| 4 | Cardiomegaly | Bóng tim to |
| 5 | Consolidation | Đông đặc phổi |
| 6 | ILD | Bệnh phổi mô kẽ |
| 7 | Infiltration | Thâm nhiễm |
| 8 | Lung Opacity | Mờ phổi |
| 9 | Nodule/Mass | Nốt mờ / Khối u |
| 10 | Other lesion | Tổn thương khác |
| 11 | Pleural effusion | Tràn dịch màng phổi |
| 12 | Pleural thickening | Dày màng phổi |
| 13 | Pneumothorax | Tràn khí màng phổi |
| 14 | Pulmonary fibrosis | Xơ phổi |

## ✨ Tính năng

- 🔍 **Chẩn đoán đa nhãn**: Phát hiện đồng thời 14 loại bệnh lý phổi
- 🗺️ **Grad-CAM Heatmap**: Hiển thị bản đồ nhiệt vùng tổn thương trên ảnh X-quang
- 🌐 **Web Application**: Giao diện web thân thiện, dễ sử dụng
- ⚡ **Xử lý nhanh**: Trả kết quả chẩn đoán trong vài giây
- 🇻🇳 **Hỗ trợ tiếng Việt**: Hiển thị kết quả bằng tiếng Việt

## 🏗️ Cấu trúc dự án

```
lung-diagnosis/
├── api/                    # FastAPI Web Application
│   ├── main.py            # API endpoints
│   ├── static/            # CSS, JS files
│   └── templates/         # HTML templates
├── data/                   # Dữ liệu và CSV
├── models/                 # Mô hình đã huấn luyện
│   └── best_model_soup.pth
├── src/                    # Source code
│   ├── model.py           # Kiến trúc mô hình EfficientNet-GCN
│   ├── dataset.py         # Xử lý dữ liệu
│   ├── gradcam.py         # Grad-CAM visualization
│   ├── train.py           # Huấn luyện mô hình
│   ├── eval.py            # Đánh giá mô hình
│   └── utils.py           # Các hàm tiện ích
├── setup/
│   └── requirements.txt   # Thư viện cần thiết
└── README.md
```

## 🔧 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- PyTorch 1.9+
- CUDA (khuyến nghị cho GPU acceleration)

### Các bước cài đặt

1. **Clone dự án:**
```bash
git clone https://github.com/BaoKhanh0109/cn-da22tta-duongbaokhanh-chandoanphoi-deeplearning.git
cd lung-diagnosis
```

2. **Tạo môi trường ảo (khuyến nghị):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt thư viện:**
```bash
pip install -r setup/requirements.txt
```

4. **Tải mô hình đã huấn luyện:**
   - Đặt file `best_model_soup.pth` vào thư mục `models/`

5. Tải bộ dữ liệu VinDr-CXR:**
   - Tải từ: https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset/data
   - Giải nén và đặt thư mục `train` vào thư mục `data`

## 🚀 Chạy ứng dụng

### Khởi động Web Server:
```bash
uvicorn api.main:app --reload
```

### Truy cập ứng dụng:
Mở trình duyệt và truy cập: **http://127.0.0.1:8000**

### Sử dụng:
1. Upload ảnh X-quang ngực (định dạng JPG, PNG, DICOM)
2. Nhấn nút "Chẩn đoán"
3. Xem kết quả và heatmap vùng tổn thương

## 🧠 Kiến trúc mô hình

Mô hình sử dụng kiến trúc lai ghép:
- **Backbone**: EfficientNet-B4 (pre-trained trên ImageNet)
- **GCN**: 2 lớp Graph Convolutional Network để học mối quan hệ giữa các bệnh
- **Input size**: 512x512 pixels

## 📊 Kết quả

Mô hình đạt được kết quả tốt trên tập validation của VinDr-CXR với các chỉ số AUC cao cho các bệnh lý phổ biến.

## 👨‍💻 Tác giả

- **Dương Bảo Khánh**
- Đồ án chuyên ngành - DA22TTA

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.
