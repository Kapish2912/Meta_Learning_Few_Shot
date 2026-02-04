import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image


# --------------------------------------------------
# Generic CNN shell (only for Grad-CAM)
# --------------------------------------------------
class GradCAMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam[0].detach().cpu().numpy()


# --------------------------------------------------
# Utils
# --------------------------------------------------
def overlay_cam(image, cam):
    h, w, _ = image.shape
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/gradcam", exist_ok=True)

    model = GradCAMModel().to(device)

    # ✅ Load weights non-strictly
    state = torch.load("outputs/model_weights.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ✅ Find last conv layer
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break

    assert target_layer is not None, "No Conv2d layer found"

    cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ✅ Automatically pick existing images
    image_dir = "data/images"
    all_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ]

    NUM_IMAGES = 6
    count = 0

    for img_path in all_images:
        if count >= NUM_IMAGES:
            break

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue

        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, pred_class].backward()

        cam_map = cam.generate()

        img_np = np.array(img_pil.resize((224, 224)))
        overlay = overlay_cam(img_np, cam_map)

        out_path = f"outputs/gradcam/{os.path.basename(img_path)}"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved Grad-CAM: {out_path}")
        count += 1


if __name__ == "__main__":
    main()