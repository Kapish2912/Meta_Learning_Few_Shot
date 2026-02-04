import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image

from src.baseline_cnn import SimpleCNN


# -------------------------------
# Grad-CAM Class
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam[0].cpu().numpy()


# -------------------------------
# Utility functions
# -------------------------------
def overlay_cam(image, cam):
    h, w, _ = image.shape
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


# -------------------------------
# Main
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs/gradcam", exist_ok=True)

    # ✅ Load model
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(
        torch.load("outputs/model_weights.pth", map_location=device)
    )
    model.eval()

    # ✅ Target layer (LAST conv layer)
    target_layer = model.features[-3]

    cam_generator = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ✅ Pick some images manually
    sample_images = [
        ("data/images/ISIC_0024306.jpg", "benign"),
        ("data/images/ISIC_0034321.jpg", "benign"),
        ("data/images/ISIC_0029856.jpg", "malignant"),
        ("data/images/ISIC_0031595.jpg", "malignant"),
    ]

    for img_path, label_name in sample_images:
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Forward
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward
        model.zero_grad()
        output[0, pred_class].backward()

        cam = cam_generator.generate(pred_class)

        # Prepare visualization
        img_np = np.array(img_pil.resize((224, 224)))
        overlay = overlay_cam(img_np, cam)

        out_path = f"outputs/gradcam/{os.path.basename(img_path)}"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved Grad-CAM: {out_path}")


if __name__ == "__main__":
    main()