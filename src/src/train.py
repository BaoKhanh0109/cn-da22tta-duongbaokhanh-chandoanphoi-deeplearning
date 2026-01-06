import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse 
import heapq  
import copy   

from dataset import VinDrCXRDataset, get_transforms, LABELS
from loss import AsymmetricLoss
from utils import save_checkpoint, compute_metrics, get_adj_matrix, seed_everything
from model import GCNResnet

def set_bn_eval(m):
    """Khóa Batch Norm để tránh nhiễu do batch size nhỏ"""
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train_with_souping(csv_path, images_dir, epochs=25, batch_size=4, accum_steps=8, lr=1e-4, image_size=380, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training on {device} | Batch={batch_size} | Accum={accum_steps} | LR={lr}")
    print(f">>> Image Size: {image_size}")

    # 1. Load Data
    df = pd.read_csv(csv_path)
    # Lọc ảnh tồn tại
    df = df[df['Image Index'].apply(lambda x: os.path.exists(os.path.join(images_dir, x)))].reset_index(drop=True)
    
    # Chia tập train/val cố định random_state để đảm bảo không bị lẫn lộn khi resume
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f">>> Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_ds = VinDrCXRDataset(csv_path, images_dir, indices=train_df.index, transform=get_transforms(image_size, train=True))
    val_ds = VinDrCXRDataset(csv_path, images_dir, indices=val_df.index, transform=get_transforms(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Setup Model & Loss
    print(">>> Computing Adjacency Matrix...")
    # Nếu file csv VinDr (14 labels) thì ma trận kề sẽ khác file NIH
    adj_matrix = get_adj_matrix(csv_path, LABELS).to(device)
    model = GCNResnet(num_classes=len(LABELS), in_channel=1792, adj_matrix=adj_matrix).to(device)

    # Gamma_neg=6 giúp phạt nặng các ca dự đoán sai trên tập dữ liệu mất cân bằng
    criterion = AsymmetricLoss(gamma_neg=6, gamma_pos=0, clip=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    start_epoch = 1
    top_k_models = []       
    global_best_auc = 0.0   
    TOP_K = 3 

    # 3. Resume / Transfer Learning Logic
    if resume and os.path.exists(resume):
        print(f">>> Loading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint

        try:
            # Thử load toàn bộ (Dành cho trường hợp resume training VinDr)
            model.load_state_dict(state_dict, strict=True)
            print(">>> [INFO] Đã load trọn vẹn Model (Resume Training bình thường).")
            
            # Nếu load thành công full model, phục hồi cả optimizer và epoch
            if isinstance(checkpoint, dict):
                if 'optimizer_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'epoch' in checkpoint and isinstance(checkpoint['epoch'], int):
                    start_epoch = checkpoint['epoch'] + 1
                if 'best_auc' in checkpoint:
                    global_best_auc = checkpoint['best_auc']
            
            print(f">>> Tiếp tục từ Epoch {start_epoch} | Best AUC cũ: {global_best_auc:.4f}")
                    
        except RuntimeError as e:
            print(">>> [INFO] Phát hiện lệch kiến trúc (Khác dataset/Head). Chuyển sang chế độ Transfer Learning (Fine-tune)...")
            # Chỉ lấy các layer thuộc backbone (EfficientNet)
            new_state_dict = {}
            matched_layers = 0
            model_dict = model.state_dict()
            
            for k, v in state_dict.items():
                if k in model_dict and v.size() == model_dict[k].size():
                    # Chỉ lấy Backbone, BỎ QUA GCN và Label Embeddings để model học lại nhãn mới
                    if "backbone" in k: 
                        new_state_dict[k] = v
                        matched_layers += 1
            
            print(f">>> Đã load {matched_layers} layers từ Backbone cũ.")
            model.load_state_dict(new_state_dict, strict=False)
            
            # Reset lại epoch để train phần Head từ đầu
            start_epoch = 1 
            global_best_auc = 0.0

    model.apply(set_bn_eval)
    
    # 4. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.apply(set_bn_eval) # Luôn đóng băng BN

        train_loss = 0
        train_preds, train_targets = [], [] # Lưu kết quả để tính Train AUC
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss = loss / accum_steps 
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            
            # Lưu lại dự đoán train (detach để không tốn VRAM)
            probs = torch.sigmoid(logits).detach()
            train_preds.append(probs.cpu().numpy())
            train_targets.append(labels.detach().cpu().numpy())
            
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                optimizer.zero_grad()
                model.apply(set_bn_eval) 
            
            train_loss += loss.item() * accum_steps
            if i % 10 == 0:
                pbar.set_postfix({'loss': f"{loss.item()*accum_steps:.4f}"})
        
        # Tính Train Metrics
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_metrics = compute_metrics(train_targets, train_preds)
        mean_train_auc = train_metrics['mean_auc']
        avg_train_loss = train_loss / len(train_ds)

        # Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_metrics = compute_metrics(val_targets, val_preds)
        mean_val_auc = val_metrics['mean_auc']

        # --- IN KẾT QUẢ CHI TIẾT ---
        print("\n" + "="*60)
        print(f"EPOCH {epoch} SUMMARY:")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"Mean Train AUC: {mean_train_auc:.4f} | Mean Val AUC: {mean_val_auc:.4f}")
        print("-" * 60)
        print(f"{'Bệnh Lý':<25} | {'Train AUC':<10} | {'Val AUC':<10}")
        print("-" * 60)
        
        for idx, label in enumerate(LABELS):
            t_auc = train_metrics['per_label_auc'][idx]
            v_auc = val_metrics['per_label_auc'][idx]
            # Xử lý hiển thị NaN
            t_str = f"{t_auc:.4f}" if not np.isnan(t_auc) else "N/A"
            v_str = f"{v_auc:.4f}" if not np.isnan(v_auc) else "N/A"
            print(f"{label:<25} | {t_str:<10} | {v_str:<10}")
        print("="*60 + "\n")

        scheduler.step()

        # Save Checkpoint & Top K Logic
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'mean_auc': mean_val_auc,
            'top_k_models': top_k_models,
            'best_auc': global_best_auc
        }
        
        current_save_path = f"models/checkpoint_ep{epoch}.pth"
        save_checkpoint(ckpt_dict, current_save_path)
        
        if mean_val_auc > global_best_auc:
            global_best_auc = mean_val_auc
            save_checkpoint(ckpt_dict, "models/best_model.pth")
            print(f">>> NEW BEST MODEL SAVED! (AUC: {global_best_auc:.4f})")

        # Heap logic for Souping
        if len(top_k_models) < TOP_K:
            heapq.heappush(top_k_models, (mean_val_auc, epoch, current_save_path))
        else:
            min_auc, min_ep, min_path = top_k_models[0]
            if mean_val_auc > min_auc:
                heapq.heappop(top_k_models)
                if os.path.exists(min_path) and min_path != "models/best_model.pth":
                    try: os.remove(min_path)
                    except: pass
                heapq.heappush(top_k_models, (mean_val_auc, epoch, current_save_path))
            else:
                if os.path.exists(current_save_path): os.remove(current_save_path)

    # --- MODEL SOUPING ---
    print("\n>>> STARTING MODEL SOUPING...")
    if not top_k_models: return

    soup_state_dict = None
    num_models = len(top_k_models)

    for auc, ep, path in top_k_models:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            current_state = checkpoint['model_state']
            if soup_state_dict is None:
                soup_state_dict = copy.deepcopy(current_state)
            else:
                for key in soup_state_dict:
                    soup_state_dict[key] += current_state[key]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            num_models -= 1
    
    if num_models > 0:
        for key in soup_state_dict:
            soup_state_dict[key] = soup_state_dict[key] / num_models
            
        save_checkpoint({
            'epoch': 'soup',
            'model_state': soup_state_dict,
            'mean_auc': max([x[0] for x in top_k_models]),
        }, "models/best_model_soup.pth")
        print(">>> Souping Complete.")

if __name__ == "__main__":
    # Khóa seed ngay đầu chương trình
    seed_everything(42)
    os.makedirs("models", exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Data_Entry_2017.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=380)
    args = parser.parse_args()

    train_with_souping(args.csv, args.images, epochs=args.epochs, batch_size=args.batch, accum_steps=args.accum, lr=args.lr, image_size=args.size, resume=args.resume)