# Hệ Thống Hỗ Trợ Chẩn Đoán Bệnh Phổi 

## 1. Giới thiệu
Dự án này xây dựng một hệ thống AI có khả năng tự động phân tích ảnh X-quang ngực thẳng để phát hiện 14 loại tổn thương và bệnh lý. Hệ thống sử dụng mô hình Deep Learning lai ghép (EfficientNet-GCN) được huấn luyện trên bộ dữ liệu VinDr-CXR.

## 2. Tính năng
* Chẩn đoán 14 bệnh lý phổi.
* Hiển thị bản đồ nhiệt (Heatmap) vùng tổn thương.
* Web App tích hợp, trả kết quả nhanh chóng.

## 3. Cài đặt
Yêu cầu: Python 3.8+, PyTorch.

1. Clone dự án về máy.
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Tải bộ dữ liệu VinDr-CXR https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection và đặt vào thư mục `data/train/`.

## 4. Chạy chương trình
Chạy lệnh sau để mở Web Demo:
uvicorn api.main:app --reload

Truy cập: http://127.0.0.1:8000