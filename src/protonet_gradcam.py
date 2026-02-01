import torch, cv2, numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.proto_network import ProtoNet
from src.fewshot_sampler import FewShotDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load ProtoNet model
# ---------------------------
model = ProtoNet().to(device)
model.load_state_dict(torch.load("outputs/protonet_fewshot.pth", map_location=device))
model.eval()

target_layer = model.encoder.conv2   # GradCAM on encoder’s last conv

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ---------------------------
# Select Random Query Image
# ---------------------------
img_path = "data/images/ISIC_0025030.jpg"    # any lesion image
original_image = Image.open(img_path).convert("RGB")
input_tensor = transform(original_image).unsqueeze(0).to(device)

# ---------------------------
# Grad‑CAM Computation
# ---------------------------
gradients, activations = [], []

def forward_hook(m, i, o): activations.append(o)
def backward_hook(m, gi, go): gradients.append(go[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

output = model(input_tensor)
# create a dummy class index (ProtoNet doesn't have softmax classes)
score = output.norm()    # maximize feature norm for visualization
score.backward()

grads = gradients[0].detach().cpu().numpy()[0]
acts = activations[0].detach().cpu().numpy()[0]
weights = np.mean(grads, axis=(1,2))
cam = np.maximum(np.sum(weights[:,None,None]*acts, axis=0),0)
cam = cv2.resize(cam, (224,224))
cam /= cam.max()+1e-8
heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
img_cv = cv2.resize(np.array(original_image), (224,224))
overlay = np.uint8(0.4*heatmap + 0.6*img_cv)

plt.imshow(overlay)
plt.title("Grad‑CAM on ProtoNet Encoder")
plt.axis("off")
plt.savefig("outputs/proto_encoder_gradcam.jpg")
plt.show()
print("[INFO] ProtoNet encoder Grad‑CAM saved to outputs/proto_encoder_gradcam.jpg")