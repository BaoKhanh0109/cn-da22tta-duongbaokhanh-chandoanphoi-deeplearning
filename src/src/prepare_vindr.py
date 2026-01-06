import pandas as pd
import numpy as np
import os

# Đường dẫn file gốc tải từ Kaggle/VinBigData
INPUT_CSV = "data/train.csv"  
OUTPUT_CSV = "data/train_fixed.csv"

# Danh sách 14 bệnh của VinDr (Theo đúng thứ tự 0-13)
VINDR_LABELS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis"
]

def prepare_data():
    print(">>> Đang đọc file CSV VinDr...")
    df = pd.read_csv(INPUT_CSV)
    
    print(">>> Đang xử lý dữ liệu (Group by)...")
    data = {}
    
    for idx, row in df.iterrows():
        img_id = row['image_id']
        class_id = int(row['class_id'])
        
        if img_id not in data:
            data[img_id] = np.zeros(len(VINDR_LABELS), dtype=int)
            
        # Nếu class_id nằm trong khoảng 0-13 thì đánh dấu là 1
        if 0 <= class_id <= 13:
            data[img_id][class_id] = 1

    # Tạo DataFrame mới
    print(">>> Đang tạo file CSV chuẩn...")
    new_rows = []
    for img_id, labels in data.items():
        # Thêm đuôi .jpg hoặc .png nếu cần khớp với folder ảnh
        # Dataset VinDr 512x512 thường là .jpg
        filename = f"{img_id}.png" 
        
        row_dict = {"Image Index": filename}
        # Tạo chuỗi Finding Labels giả lập (để khớp logic cũ nếu cần)
        finding_list = []
        for i, val in enumerate(labels):
            if val == 1:
                row_dict[VINDR_LABELS[i]] = 1
                finding_list.append(VINDR_LABELS[i])
            else:
                row_dict[VINDR_LABELS[i]] = 0
        
        if not finding_list:
            row_dict["Finding Labels"] = "No Finding"
        else:
            row_dict["Finding Labels"] = "|".join(finding_list)
            
        new_rows.append(row_dict)
        
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(OUTPUT_CSV, index=False)
    print(f">>> Xong! File đã lưu tại: {OUTPUT_CSV}")
    print(df_new.head())

if __name__ == "__main__":
    prepare_data()