import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.baseline_cnn import SimpleCNN

# ================================ #
#      CONFIGURATION SECTION       #
# ================================ #

# Map class indices back to readable labels
label_map = {0: "bkl", 1: "df", 2: "mel", 3: "nv"}

# Choose colormap and overlay strength
colormap = cv2.COLORMAP_JET    # try also: COLORMAP_INFERNO, COLORMAP_MAGMA
overlay_intensity = 0.45       # 0.3 = lighter overlay, 0.7 = strong heatmap

# Create folder if not present
save_dir = "outputs/gradcam"
os.makedirs(save_dir, exist_ok=True)

# ================================ #
#          LOAD MODEL              #
# ================================ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("outputs/model_weights.pth", map_location=device))
model.eval()

# ================================ #
#          LOAD IMAGE              #
# ================================ #
img_path = "data/images/ISIC_0025030.jpg"  # Replace with your own
original_image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(original_image).unsqueeze(0).to(device)

# ================================ #
#        SELECT TARGET LAYER       #
# ================================ #
target_layer = model.conv2
gradients, activations = [], []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)  # use full backward hook

# ================================ #
#          GENERATE GRAD-CAM       #
# ================================ #
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
pred_label = label_map.get(pred_class, str(pred_class))
print(f"Predicted class index: {pred_class}  →  {pred_label}")

model.zero_grad()
output[:, pred_class].backward()

grads = gradients[0].cpu().data.numpy()[0]
acts = activations[0].cpu().data.numpy()[0]

weights = np.mean(grads, axis=(1, 2))  # GAP over H, W
cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i, :, :]
cam = np.maximum(cam, 0)
cam /= np.max(cam) + 1e-8
cam = cv2.resize(cam, (224, 224))

# ================================ #
#        HEATMAP GENERATION        #
# ================================ #
img_cv = np.array(original_image)
img_cv = cv2.resize(img_cv, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

superimposed_img = np.uint8(heatmap * overlay_intensity + img_cv * (1 - overlay_intensity))

# ================================ #
#        DISPLAY AND SAVE          #
# ================================ #
plt.figure(figsize=(10, 4))
plt.suptitle(f"Predicted: {pred_label}", fontsize=14, fontweight='bold')

plt.subplot(1, 3, 1)
plt.imshow(img_cv)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(save_dir, f"gradcam_{pred_label}.jpg")
plt.savefig(save_path)
plt.show()

print(f"Grad-CAM saved successfully at: {save_path}")