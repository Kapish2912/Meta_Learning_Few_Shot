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
    0.1,
    1.0,
    0.4,
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.3,
    0.9,
    0.6,
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
    # CONFIDENCE SAFETY CHECK
    # ------------------------------------------------------
    if probs[top1] < confidence_threshold:
        st.error(
            "⚠️ Prediction confidence below threshold. "
            "Expert review is recommended."
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

        cam1_color = cv2.applyColorMap(
            np.uint8(cam1 * 255),
            cmap_dict[colormap_name],
        )

        cam1_color = cv2.cvtColor(
            cam1_color,
            cv2.COLOR_BGR2RGB,
        )

        cam1_overlay = np.uint8(
            cam_alpha * cam1_color
            + (1 - cam_alpha)
            * np.array(img.resize((224, 224)))
        )

        with col1:
            st.subheader("Top‑1 Grad‑CAM")
            st.image(
                cam1_overlay,
                caption=label_map[top1],
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

            cam2_color = cv2.applyColorMap(
                np.uint8(cam2 * 255),
                cmap_dict[colormap_name],
            )

            cam2_color = cv2.cvtColor(
                cam2_color,
                cv2.COLOR_BGR2RGB,
            )

            cam2_overlay = np.uint8(
                cam_alpha * cam2_color
                + (1 - cam_alpha)
                * np.array(img.resize((224, 224)))
            )

            with col2:
                st.subheader("Runner‑up Grad‑CAM")
                st.image(
                    cam2_overlay,
                    caption=label_map[top2],
                    width="stretch",
                )

            # --------------------------------------------------
            # CAM OVERLAP METRIC (NEW)
            # --------------------------------------------------
            cam1_bin = cam1 > 0.5
            cam2_bin = cam2 > 0.5

            overlap_score = (
                np.sum(cam1_bin & cam2_bin)
                / (np.sum(cam1_bin) + 1e-8)
            )

            st.metric(
                "Grad‑CAM Overlap Score",
                f"{overlap_score:.2f}",
            )

            if overlap_score > 0.5:
                st.warning(
                    "High overlap between Top‑1 and Runner‑up "
                    "indicates class ambiguity."
                )

        st.download_button(
            "Download Top‑1 Grad‑CAM",
            data=cam1_overlay.tobytes(),
            file_name="gradcam_top1.png",
            mime="image/png",
        )

    # ------------------------------------------------------
    # REASONING & FEATURE EXPECTATION
    # ------------------------------------------------------
    predictions_text = {
        "nv": (
            "Smooth, symmetric lesion with uniform pigmentation. "
            "Melanoma would show irregular borders and color variation."
        ),
        "mel": (
            "Irregular pigmentation and asymmetric regions. "
            "Benign lesions typically lack such heterogeneity."
        ),
        "bkl": (
            "Keratinized or scaly texture. "
            "Melanoma would emphasize color irregularity instead."
        ),
        "df": (
            "Firm nodule with limited pigment spread. "
            "Malignant lesions show broader activation patterns."
        ),
    }

    with st.expander("Model Reasoning & Feature Expectation"):
        st.write(predictions_text[label_map[top1]])

        if compare_runnerup:
            st.markdown(
                f"**Why not {label_map[top2]}?**  \n"
                "Although visually similar, the runner‑up lacks "
                "dominant discriminative regions required for "
                "confident diagnosis."
            )