import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

from src.baseline_cnn import SimpleCNN
from src.gradcam_reasoning import generate_cam

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    0: "bkl",
    1: "df",
    2: "mel",
    3: "nv",
}

# ----------------------------------------------------------
# LOAD MODEL (CACHED)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(
        torch.load(
            "outputs/model_weights.pth",
            map_location=device,
        )
    )
    model.eval()
    return model


model = load_model()
target_layer = model.conv2

# ----------------------------------------------------------
# TRANSFORMS
# ----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------
st.sidebar.header("Explanation Controls")

show_cam = st.sidebar.checkbox(
    "Show Grad‑CAM",
    value=True,
)

compare_runnerup = st.sidebar.checkbox(
    "Compare with Runner‑up",
    value=True,
)

cam_alpha = st.sidebar.slider(
    "Heatmap Strength",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
)

colormap_name = st.sidebar.selectbox(
    "Colormap",
    ["JET", "VIRIDIS", "PLASMA", "INFERNO"],
)

cmap_dict = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
}

# ----------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------
st.title("Rare Disease Diagnosis with Explainable AI")

file = st.file_uploader(
    "Upload a skin lesion image (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
)

if file is not None:
    img = Image.open(file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        width="stretch",
    )

    x = transform(img).unsqueeze(0).to(device)

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------
    with st.spinner("Running model inference..."):
        out = model(x)
        probs = F.softmax(out, dim=1)[0].detach().cpu().numpy()

    sort_idx = np.argsort(probs)[::-1]
    top1, top2 = sort_idx[0], sort_idx[1]

    st.subheader(
        f"Prediction: {label_map[top1]} "
        f"({probs[top1] * 100:.2f}%)"
    )

    st.write(
        f"Runner‑up: {label_map[top2]} "
        f"({probs[top2] * 100:.2f}%)"
    )

    # ------------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # ------------------------------------------------------
    st.subheader("Prediction Confidence Distribution")

    st.bar_chart({
        label_map[i]: float(probs[i])
        for i in range(len(probs))
    })

    # ------------------------------------------------------
    # GRAD‑CAM VISUALIZATION
    # ------------------------------------------------------
    if show_cam:
        col1, col2 = st.columns(2)

        # ---------- TOP‑1 ----------
        cam1 = generate_cam(
            model,
            x,
            top1,
            target_layer,
        )

        cam1 = cv2.applyColorMap(
            np.uint8(cam1 * 255),
            cmap_dict[colormap_name],
        )

        cam1 = cv2.cvtColor(
            cam1,
            cv2.COLOR_BGR2RGB,
        )

        cam1 = np.uint8(
            cam_alpha * cam1
            + (1 - cam_alpha)
            * np.array(img.resize((224, 224)))
        )

        with col1:
            st.subheader("Top‑1 Grad‑CAM")
            st.image(
                cam1,
                caption=(
                    f"{label_map[top1]} "
                    f"({probs[top1] * 100:.2f}%)"
                ),
                width="stretch",
            )

        # ---------- RUNNER‑UP ----------
        if compare_runnerup:
            cam2 = generate_cam(
                model,
                x,
                top2,
                target_layer,
            )

            cam2 = cv2.applyColorMap(
                np.uint8(cam2 * 255),
                cmap_dict[colormap_name],
            )

            cam2 = cv2.cvtColor(
                cam2,
                cv2.COLOR_BGR2RGB,
            )

            cam2 = np.uint8(
                cam_alpha * cam2
                + (1 - cam_alpha)
                * np.array(img.resize((224, 224)))
            )

            with col2:
                st.subheader("Runner‑up Grad‑CAM")
                st.image(
                    cam2,
                    caption=(
                        f"{label_map[top2]} "
                        f"({probs[top2] * 100:.2f}%)"
                    ),
                    width="stretch",
                )

        st.download_button(
            label="Download Top‑1 Grad‑CAM",
            data=cam1.tobytes(),
            file_name="gradcam_top1.png",
            mime="image/png",
        )

    # ------------------------------------------------------
    # REASONING & FEATURE EXPECTATION
    # ------------------------------------------------------
    predictions_text = {
        "nv": (
            "Smooth, round lesion with uniform brown color. "
            "If it resembled melanoma, irregular borders or "
            "varied pigmentation would be highlighted."
        ),
        "mel": (
            "Irregular dark areas with uneven edges. "
            "A benign nevus would show more symmetric and "
            "consistent coloring."
        ),
        "bkl": (
            "Rough texture or scaly appearance. "
            "Melanoma would typically activate multi‑tone "
            "pigmentation regions."
        ),
        "df": (
            "Small, firm nodule with limited pigmentation. "
            "Melanoma would show dispersed and asymmetric "
            "highlighted regions."
        ),
    }

    with st.expander("Model Reasoning & Feature Expectation"):
        st.write(predictions_text[label_map[top1]])

        if compare_runnerup:
            st.markdown(
                f"**Why not {label_map[top2]}?**  \n"
                "The runner‑up class shows partial feature overlap "
                "but lacks the dominant discriminative regions "
                "required for confident diagnosis."
            )

    # ------------------------------------------------------
    # LOW CONFIDENCE WARNING
    # ------------------------------------------------------
    if probs[top1] < 0.5:
        st.warning(
            "Low confidence prediction. Visual features overlap "
            "between multiple disease classes."
        )