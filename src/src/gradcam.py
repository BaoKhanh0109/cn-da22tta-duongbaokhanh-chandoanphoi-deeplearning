import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None        # Bản sao detached để xử lý CPU
        self.activations_raw = None    # Tensor gốc để tính gradient
        
        # Chỉ đăng ký Forward Hook để bắt Feature Map
        def forward_hook(module, inp, out):
            self.activations_raw = out
            try:
                self.activations = out.detach().clone()
            except Exception:
                self.activations = out.detach()

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Tạo Heatmap giải thích vùng ảnh hưởng tới quyết định của model.
        """
        # 1. Forward Pass
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        
        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()
        score = logits[0, class_idx]
        
        # 2. Tính Gradient (Backpropagation)
        self.model.zero_grad()
        if self.activations_raw is None:
            raise RuntimeError("GradCAM: activations not captured")

        # Tính đạo hàm riêng của Score theo Feature Map: d(Score)/d(A_k)
        grads_tensor = torch.autograd.grad(score, self.activations_raw, retain_graph=True)[0]
        
        grads = grads_tensor[0].cpu().detach().numpy()
        acts = self.activations[0].cpu().numpy()
        
        # 3. Global Average Pooling (GAP) để lấy trọng số w_k
        weights = np.mean(grads, axis=(1,2))
        
        # 4. Tổ hợp tuyến tính: Heatmap = sum(w_k * A_k)
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        # 5. ReLU: Chỉ giữ lại ảnh hưởng dương tính
        cam = np.maximum(cam, 0)
        
        # 6. Chuẩn hóa về [0, 1] và resize về kích thước ảnh gốc
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return np.uint8(255 * cam)

def overlay_heatmap_on_image(img_pil, heatmap):
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
        
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return Image.fromarray(overlay)