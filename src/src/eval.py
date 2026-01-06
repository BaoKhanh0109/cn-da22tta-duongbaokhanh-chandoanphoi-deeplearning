import torch
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import VinDrCXRDataset, get_transforms, LABELS
from model import GCNResnet
from utils import compute_metrics, get_adj_matrix

def evaluate(csv_path, images_dir, model_path, batch_size=8, device='cuda'):
    print(f">>> Đang đánh giá model: {model_path}")
    
    # 1. Tái tạo lại Split (Logic phải khớp với train.py)
    df = pd.read_csv(csv_path)
    df = df[df['Image Index'].apply(lambda x: os.path.exists(os.path.join(images_dir, x)))].reset_index(drop=True)
    
    idxs = list(df.index)
    train_idx, val_idx = train_test_split(idxs, test_size=0.1, random_state=42)
    
    # Lấy tập Validation để test
    test_idx = val_idx
    print(f">>> Số lượng ảnh Test (Validation): {len(test_idx)}")

    # 2. Dataset & DataLoader
    test_ds = VinDrCXRDataset(csv_path, images_dir, indices=test_idx, transform=get_transforms(512, train=False))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Khởi tạo Model
    print(">>> Đang tính toán Ma trận kề...")
    if os.path.exists(csv_path):
        adj = get_adj_matrix(csv_path, LABELS).to(device)
    else:
        adj = torch.eye(len(LABELS)).to(device)
    
    model = GCNResnet(num_classes=len(LABELS), in_channel=1792, adj_matrix=adj)
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Lỗi load weights: {e}")
        return

    model.to(device)
    model.eval()

    # 4. Chạy dự đoán và LƯU KẾT QUẢ
    all_preds = []
    all_targets = []
    results_data = [] 
    
    print(">>> Đang chạy dự đoán và xuất file kết quả...")
    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader):
            imgs = imgs.to(device)
            
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.numpy()
            
            all_preds.append(probs)
            all_targets.append(targets)

            for i in range(len(names)):
                row = {"Image Index": names[i]}
                # Lưu xác suất của từng bệnh
                for j, label in enumerate(LABELS):
                    row[label] = probs[i][j]
                
                # Lưu nhãn thực tế (Ground Truth) để đối chiếu
                true_labels = [LABELS[k] for k, val in enumerate(targets[i]) if val == 1]
                row["Ground_Truth"] = "|".join(true_labels) if true_labels else "No Finding"
                
                results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("data/validation_results.csv", index=False)
    print(f"\n>>> Đã lưu danh sách ảnh test vào file: validation_results.csv")

    # 5. Tính Metric như cũ
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    
    print("\n" + "="*40)
    print(f"MEAN AUC: {metrics['mean_auc']:.4f}")
    print("="*40)
    
    for i, label in enumerate(LABELS):
        auc = metrics['per_label_auc'][i]
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"{label:<25} | {auc_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train_fixed.csv")
    parser.add_argument("--images", default="data/train")
    parser.add_argument("--model", default="models/best_model_soup.pth") 
    parser.add_argument("--batch", type=int, default=8)
    
    args = parser.parse_args()
    evaluate(args.csv, args.images, args.model, args.batch)