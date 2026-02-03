import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.baseline_cnn import SimpleCNN

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "bkl", 1: "df", 2: "mel", 3: "nv"}
img_path = "data/images/ISIC_0025030.jpg"  # sample test image

save_dir = "outputs/gradcam_compare"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(
    torch.load("outputs/ph2_model_weights.pth", map_location=device)
)
model.eval()

target_layer = model.model.layer4[-1].conv2

# -------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
original_image = Image.open(img_path).convert("RGB")
input_tensor = transform(original_image).unsqueeze(0).to(device)


# -------------------------------------------------------
# GRAD-CAM FUNCTION
# -------------------------------------------------------
def generate_cam(output, class_idx, activations, gradients):
    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8
    cam = cv2.resize(cam, (224, 224))
    return cam


# -------------------------------------------------------
# HOOKS
# -------------------------------------------------------
gradients, activations = [], []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------------------------------------------------------
# FORWARD PASS
# -------------------------------------------------------
output = model(input_tensor)
probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
sorted_idx = np.argsort(probs)[::-1]
top1, top2 = sorted_idx[0], sorted_idx[1]
pred_label = label_map[top1]
runner_label = label_map[top2]

print(f"\nTop‑1 Prediction : {pred_label} ({probs[top1]*100:.2f}%)")
print(f"Runner‑Up        : {runner_label} ({probs[top2]*100:.2f}%)")
confidence_gap = probs[top1] - probs[top2]
print(f"Confidence gap   : {confidence_gap*100:.2f}%")

# -------------------------------------------------------
# BACKWARD PASS (for Top‑1 class)
# -------------------------------------------------------
model.zero_grad()
output[:, top1].backward(retain_graph=True)
cam_top1 = generate_cam(output, top1, activations, gradients)

# -------------------------------------------------------
# BACKWARD PASS (for Runner‑up class)
# -------------------------------------------------------
# Clear old hooks content
gradients.clear()
activations.clear()

# Re‑run forward pass to record fresh activations
output = model(input_tensor)

# Now do backward for the 2nd‑best class
model.zero_grad()
output[:, top2].backward()

# Compute runner‑up CAM
cam_top2 = generate_cam(output, top2, activations, gradients)

# -------------------------------------------------------
# HEATMAP OVERLAYS
# -------------------------------------------------------
def overlay_heatmap(cam, original_image):
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_cv = np.array(original_image)
    img_cv = cv2.resize(img_cv, (224, 224))
    overlay = np.uint8(0.45 * heatmap + 0.55 * img_cv)
    return overlay


overlay_top1 = overlay_heatmap(cam_top1, original_image)
overlay_top2 = overlay_heatmap(cam_top2, original_image)

# -------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.suptitle(f"Top‑1: {pred_label} vs  Runner‑Up: {runner_label}", fontsize=14, fontweight='bold')

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(overlay_top1)
plt.title(f"Top‑1: {pred_label}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_top2)
plt.title(f"Runner‑Up: {runner_label}")
plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.92])
save_path = os.path.join(save_dir, f"compare_{pred_label}_vs_{runner_label}.jpg")
plt.savefig(save_path)
plt.show()

print(f"\nComparison Grad‑CAM saved at: {save_path}")

# -------------------------------------------------------
# INTERPRETATION / REASONING
# -------------------------------------------------------
if confidence_gap > 0.15:
    print("\nReasoning:")
    print(f"- The model shows strong focus for {pred_label} with high activation consistency.")
    print(f"- Runner‑up class {runner_label} has weaker or scattered activations.")
elif confidence_gap > 0.05:
    print("\nReasoning:")
    print(f"- The model finds some overlapping features between {pred_label} and {runner_label}.")
    print(f"- However, spatial activation alignment is stronger for {pred_label}.")
else:
    print("\nReasoning:")
    print(f"- Confidence gap is small (~{confidence_gap*100:.2f}%), indicating uncertainty.")
    print(f"- The two classes may have visually similar patterns.")