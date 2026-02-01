import streamlit as st
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import numpy as np, cv2
from src.baseline_cnn import SimpleCNN
from src.gradcam_reasoning import generate_cam   # reuse the function you wrote earlier

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "bkl", 1: "df", 2: "mel", 3: "nv"}

# load model once
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load("outputs/model_weights.pth", map_location=device))
    model.eval()
    return model

model = load_model()
target_layer = model.conv2

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------------------------------------
# UPLOAD SECTION
# ----------------------------------------------------------
st.title("Rare Disease Diagnosis with Explainable AI")
file = st.file_uploader("Upload a skin lesion image (.jpg/.png)")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    x = transform(img).unsqueeze(0).to(device)

    # Predict
    out = model(x)
    probs = F.softmax(out, dim=1)[0].detach().cpu().numpy()
    sort_idx = np.argsort(probs)[::-1]
    top1, top2 = sort_idx[0], sort_idx[1]

    st.write(f"### Prediction: **{label_map[top1]}** ({probs[top1]*100:.2f}%)")
    st.write(f"#### Runner‑up: {label_map[top2]} ({probs[top2]*100:.2f}%)")

    # Grad‑CAM overlays
    cam = generate_cam(model, x, top1, target_layer)
    cam_img = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    cam_img = np.uint8(0.4*cam_img + 0.6*np.array(img.resize((224,224))))

    st.image(cam_img, caption=f"Grad‑CAM map for {label_map[top1]}", use_column_width=True)

    # ------------------------------------------------------
    # SIMPLE FEATURE EXPECTATION EXPLANATION
    # ------------------------------------------------------
    predictions_text = {
        "nv": "Smooth, round lesion with uniform brown color. If it resembled melanoma, we’d expect irregular borders or varied pigmentation.",
        "mel": "Irregular dark areas with uneven edges. For a benign nevus, the focus would shift toward consistent, symmetric coloring.",
        "bkl": "Rough texture or scaly pattern. For melanoma, activations would highlight multi‑tone pigmentation rather than uniform surface.",
        "df": "Small, firm nodule with limited pigment. A melanoma focus would show dispersed, asymmetric highlights."
    }

    st.write("### Reasoning & Feature Expectation:")
    st.write(predictions_text[label_map[top1]])