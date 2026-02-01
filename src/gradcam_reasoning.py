import torch, os, cv2
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.baseline_cnn import SimpleCNN

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "bkl", 1: "df", 2: "mel", 3: "nv"}

def generate_cam(model, input_tensor, class_idx, target_layer):
    gradients, activations = [], []

    def f_hook(m, i, o):  activations.append(o)
    def b_hook(m, gi, go): gradients.append(go[0])

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)

    # forward + backward
    output = model(input_tensor)
    model.zero_grad()
    output[:, class_idx].backward()

    # create cam
    grads = gradients[0].detach().cpu().numpy()[0]
    acts  = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1,2))
    cam = np.maximum(np.sum(weights[:,None,None]*acts, axis=0),0)
    cam = cv2.resize(cam, (224,224))
    cam /= cam.max() + 1e-8

    h1.remove(); h2.remove()
    return cam

def reasoning_for_image(img_path):
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load("outputs/model_weights.pth", map_location=device))
    model.eval()
    target_layer = model.conv2

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # predictions
    out = model(x)
    probs = F.softmax(out, dim=1)[0].detach().cpu().numpy()
    sort_idx = np.argsort(probs)[::-1]
    top1, top2 = sort_idx[0], sort_idx[1]
    conf_gap = probs[top1] - probs[top2]

    cam1 = generate_cam(model, x, top1, target_layer)
    cam2 = generate_cam(model, x, top2, target_layer)

    # crude attention statistics
    mean1, mean2 = cam1.mean(), cam2.mean()
    focus1 = np.sum(cam1 > 0.7)
    focus2 = np.sum(cam2 > 0.7)

    # -------- text explanation --------
    if conf_gap > 0.15:
        reason = (f"Predicted **{label_map[top1]}** ({probs[top1]*100:.1f}%). "
                  f"High activation concentration ({focus1} px >0.7) "
                  f"shows strong confidence.\n"
                  f"Runner‑up **{label_map[top2]}** "
                  f"has weaker scattered activations ({focus2} px >0.7).")
    elif conf_gap > 0.05:
        reason = (f"Model slightly prefers **{label_map[top1]}** "
                  f"({probs[top1]*100:.1f}% vs {probs[top2]*100:.1f}%). "
                  f"Both share similar patterns, but {label_map[top1]} "
                  f"shows tighter focus in lesion core.")
    else:
        reason = (f"Model is uncertain between **{label_map[top1]}** "
                  f"and **{label_map[top2]}**.")
    
    # -------- show 2 heatmaps --------
    overlay = lambda cam: cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    img_cv = cv2.resize(np.array(img), (224,224))
    ov1 = np.uint8(0.45*overlay(cam1) + 0.55*img_cv)
    ov2 = np.uint8(0.45*overlay(cam2) + 0.55*img_cv)

    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1); plt.imshow(img_cv); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB))
    plt.title(f"Top‑1: {label_map[top1]}")
    plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB))
    plt.title(f"Runner‑up: {label_map[top2]}")
    plt.tight_layout()
    plt.show()

    print("\nReasoning:\n", reason)
    return reason

# ------------ Example ------------
if __name__ == "__main__":
    reasoning_for_image("data/images/ISIC_0025030.jpg")