import base64
import io
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.model import GCNResnet  
from src.gradcam import GradCAM, overlay_heatmap_on_image
from src.dataset import crop_center_blind 
from src.utils import get_adj_matrix 

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# --- 1. DANH SÁCH NHÃN CHUẨN CỦA VINDR (Theo thứ tự 0-13) ---
LABELS = [
    "Aortic enlargement",   
    "Atelectasis",          
    "Calcification",        
    "Cardiomegaly",         
    "Consolidation",        
    "ILD",                  
    "Infiltration",         
    "Lung Opacity",         
    "Nodule/Mass",          
    "Other lesion",         
    "Pleural effusion",     
    "Pleural thickening",   
    "Pneumothorax",         
    "Pulmonary fibrosis"    
]

# --- 2. TỪ ĐIỂN DỊCH SANG TIẾNG VIỆT ---
VIETNAMESE_LABELS = {
    "Aortic enlargement": "Phình động mạch chủ",
    "Atelectasis": "Xẹp phổi",
    "Calcification": "Vôi hóa",
    "Cardiomegaly": "Bóng tim to",
    "Consolidation": "Đông đặc phổi",
    "ILD": "Bệnh phổi mô kẽ",
    "Infiltration": "Thâm nhiễm",
    "Lung Opacity": "Mờ phổi",
    "Nodule/Mass": "Nốt mờ / Khối u",
    "Other lesion": "Tổn thương khác",
    "Pleural effusion": "Tràn dịch màng phổi",
    "Pleural thickening": "Dày màng phổi",
    "Pneumothorax": "Tràn khí màng phổi",
    "Pulmonary fibrosis": "Xơ phổi",
    "No Finding": "Bình thường"
}

# --- CẤU HÌNH ---
if os.path.exists("models/best_model_soup.pth"):
    MODEL_PATH = "models/best_model_soup.pth"
else:
    MODEL_PATH = "models/best_model.pth"

# Trỏ vào file CSV mới của VinDr để lấy ma trận kề (Graph)
CSV_PATH = "data/train_fixed.csv" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

# --- 3. CẬP NHẬT NGƯỠNG (THRESHOLDS) ---

THRESHOLDS = {
    "Pneumothorax": 0.30,      
    "Atelectasis": 0.30,       
    "Consolidation": 0.25,   
    "ILD": 0.30,              
    "Nodule/Mass": 0.35,
    "Calcification": 0.35,    
    "Infiltration": 0.40,     
    "Cardiomegaly": 0.50,       
    "Aortic enlargement": 0.50, 
    "Pleural effusion": 0.50,   
    "Pleural thickening": 0.50, 
    "Pulmonary fibrosis": 0.45,
    "Lung Opacity": 0.45,               
    "Other lesion": 0.50,
    "default": 0.40
}

# --- LOAD MODEL ---
def load_model():
    print(f">>> Đang tải Model từ: {MODEL_PATH} trên thiết bị {DEVICE}")
    
    # Tính toán ma trận kề từ CSV
    if os.path.exists(CSV_PATH):
        adj = get_adj_matrix(CSV_PATH, LABELS).to(DEVICE)
    else:
        print("⚠ Không tìm thấy CSV, sử dụng ma trận đơn vị cho Graph.")
        adj = torch.eye(len(LABELS)).to(DEVICE)

    # Khởi tạo model với đúng 14 classes
    model = GCNResnet(num_classes=len(LABELS), in_channel=1792, adj_matrix=adj)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        
        try:
            model.load_state_dict(state_dict)
            print(">>> Load weights thành công!")
        except Exception as e:
            print(f"⚠ LỖI LOAD WEIGHTS: {e}")
            print("Gợi ý: Kiểm tra xem file model đã được train lại với 14 nhãn VinDr chưa?")
    else:
        print("⚠ CẢNH BÁO: Không tìm thấy file model! Đang dùng trọng số ngẫu nhiên.")

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- GRADCAM SETUP ---
try:
    target_layer = model.backbone.conv_head
except AttributeError:
    # Fallback nếu tên layer khác
    target_layer = list(model.backbone.children())[-1]

gradcam = GradCAM(model, target_layer)

# --- PREPROCESS ---
preprocess = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 1. Tiền xử lý: Cắt bỏ viền đen/chữ nhiễu trước (Blind Crop)
    image_np = np.array(image_pil) 
    image_cropped_np = crop_center_blind(image_np, crop_percent=0.08)
    image_cropped_pil = Image.fromarray(image_cropped_np)

    # 2. TTA - TẠO 5 PHIÊN BẢN CỦA ẢNH (5-View Strategy)
    tta_images = []  
    tta_images.append(preprocess(image_cropped_pil))  
    tta_images.append(preprocess(image_cropped_pil.transpose(Image.FLIP_LEFT_RIGHT))) 
    tta_images.append(preprocess(image_cropped_pil.rotate(5)))
    tta_images.append(preprocess(image_cropped_pil.rotate(-5)))
    # Giúp loại bỏ hoàn toàn nhiễu ở rìa và tập trung vào nhu mô phổi
    w, h = image_cropped_pil.size
    crop_size = int(min(w, h) * 0.9)
    img_zoom = TF.center_crop(image_cropped_pil, [crop_size, crop_size])
    tta_images.append(preprocess(img_zoom)) # preprocess sẽ tự resize lại về 380

    # Đóng gói thành 1 batch (5, 3, 380, 380) để đưa vào GPU xử lý 1 lần
    input_batch = torch.stack(tta_images).to(DEVICE)
    
    # 3. DỰ ĐOÁN
    with torch.no_grad():
        # Model trả về (5, 14) - 5 dòng kết quả cho 5 ảnh
        logits = model(input_batch)
        probs_all = torch.sigmoid(logits)
        
        probs_avg = torch.mean(probs_all, dim=0).cpu().numpy()
        
    # 4. XỬ LÝ KẾT QUẢ 
    results = []
    has_disease = False
    
    for i, label_en in enumerate(LABELS):
        score = float(probs_avg[i])
        thresh = THRESHOLDS.get(label_en, THRESHOLDS["default"])
        
        is_sick = score > thresh
        if is_sick: has_disease = True
        
        label_vi = VIETNAMESE_LABELS.get(label_en, label_en)
            
        results.append({
            "label": label_vi,          
            "label_en": label_en,       
            "prob": score, 
            "is_normal": not is_sick
        })
    
    # Xử lý trường hợp bình thường
    if has_disease:
        results.append({"label": "Bình thường", "label_en": "No Finding", "prob": 0.01, "is_normal": True})
    else:
        max_prob = float(np.max(probs_avg)) 
        results.append({
            "label": "Bình thường", 
            "label_en": "No Finding", 
            "prob": 1.0 - max_prob, 
            "is_normal": True
        })
        
    results.sort(key=lambda x: (not x["is_normal"], x["prob"]), reverse=True)
    top_results = results[:6]
    
    # 5. VẼ HEATMAP (
    target_idx = 0 
    draw_label_en = top_results[0]["label_en"]
    if draw_label_en == "No Finding" and len(top_results) > 1:
        draw_label_en = top_results[1]["label_en"]
    
    if draw_label_en in LABELS:
        target_idx = LABELS.index(draw_label_en)
        
    try:
        # Input tensor cho GradCAM là ảnh gốc (View 1)
        # Cần unsqueeze(0) để tạo batch dimension (1, 3, 380, 380)
        input_tensor_orig = tta_images[0].unsqueeze(0).to(DEVICE)
        heatmap = gradcam.generate(input_tensor_orig, class_idx=target_idx)
        overlay = overlay_heatmap_on_image(image_cropped_pil, heatmap)
        overlay_b64 = pil_to_base64(overlay)
    except Exception as e:
        print(f"GradCAM Error: {e}")
        overlay_b64 = pil_to_base64(image_cropped_pil)
        
    return JSONResponse({
        "top": top_results,
        "orig": pil_to_base64(image_pil),         
        "overlay": overlay_b64                    
    })