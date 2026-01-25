import os, random, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.baseline_cnn import SimpleCNN

# ==============================
# CONFIGURATION
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {0: "bkl", 1: "df", 2: "mel", 3: "nv"}
colormap = cv2.COLORMAP_INFERNO
overlay_intensity = 0.45
input_size = (224, 224)

img_dir = "data/images"
meta_path = "data/HAM10000_metadata.csv"

# Output directory
save_root = "outputs/gradcam/batch"
os.makedirs(save_root, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("outputs/model_weights.pth", map_location=device))
model.eval()

target_layer = model.conv2

# ==============================
# IMAGE TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

# ==============================
# GRAD-CAM FUNCTION
# ==============================
def generate_gradcam(img_path):
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # Load and preprocess image
    original_image = Image.open(img_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Forward + backward for predicted class
    output = model(input_tensor)
    pred = output.argmax(dim=1).item()
    model.zero_grad()
    output[:, pred].backward()

    # Compute Grad-CAM
    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8
    cam = cv2.resize(cam, input_size)

    # Prepare overlay
    img_cv = np.array(original_image)
    img_cv = cv2.resize(img_cv, input_size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap * overlay_intensity + img_cv * (1 - overlay_intensity))

    handle_f.remove()
    handle_b.remove()

    return superimposed, pred

# ==============================
# LOOP THROUGH MULTIPLE IMAGES
# ==============================
all_imgs = [os.path.join(img_dir, f) 
            for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
sample_imgs = random.sample(all_imgs, 8)  # choose any number you want

for img_path in sample_imgs:
    gradcam_img, pred_idx = generate_gradcam(img_path)
    pred_label = label_map.get(pred_idx, str(pred_idx))
    fname = os.path.basename(img_path).split(".")[0]
    save_path = os.path.join(save_root, f"{fname}_{pred_label}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM for {fname} as {save_path}")

# ==============================
# OPTIONAL: SUMMARY GRID (FIXED)
# ==============================
import glob

gradcam_files = sorted(glob.glob(os.path.join(save_root, "*.jpg")))
print(f"Creating gallery from {len(gradcam_files)} Grad-CAM images...")

plt.figure(figsize=(15, 10))

for i, img_path in enumerate(gradcam_files[:9]):  # show 9 examples
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    fname = os.path.basename(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(fname.split("_")[-1].replace(".jpg", ""), fontsize=8)
    plt.axis("off")

plt.tight_layout()
gallery_path = os.path.join(save_root, "gradcam_gallery.jpg")
plt.savefig(gallery_path)
plt.show()

print(f"Gallery saved at: {gallery_path}")