import os, glob
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------
# CONFIG PATHS  (rename folders first to remove spaces)
# ----------------------------------------------------------
cnn_report_path  = "outputs/week_b_analysis/cnn_metrics_report.txt"
proto_report_path = "outputs/fewshot_results/metrics_report_protonet_few-shot.txt"

# Auto-find any confusion-matrix images if exact name missing
cnn_cm_candidates = glob.glob("outputs/week_b_analysis/confusion_matrix*.png")
proto_cm_candidates = glob.glob("outputs/fewshot_results/confusion_matrix*.png")
cnn_cm_path = cnn_cm_candidates[0] if cnn_cm_candidates else None
proto_cm_path = proto_cm_candidates[0] if proto_cm_candidates else None

save_dir = "outputs/comparison_dashboard"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------------------------------------
# STRONG PARSER FOR ACCURACY & MACRO F1
# ----------------------------------------------------------
def parse_report(file_path):
    acc, macro_f1 = 0.0, 0.0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                low = line.lower()

                # ---- read "Overall Accuracy: 78.00%" ----
                if "overall accuracy" in low:
                    # find any token that ends with %
                    parts = [p for p in line.split() if "%" in p or p.replace('.', '', 1).isdigit()]
                    if parts:
                        num = parts[-1].replace('%', '').strip()
                        try:
                            val = float(num)
                            acc = val / 100 if val > 1 else val
                        except:
                            pass

                # ---- read plain accuracy line in the table ----
                elif low.strip().startswith("accuracy") and acc == 0.0:
                    vals = [p for p in line.split() if p.replace('.', '', 1).isdigit()]
                    if vals:
                        acc = float(vals[0])

                # ---- read macro avg ----
                if "macro avg" in low:
                    nums = [w for w in line.split() if w.replace(".", "", 1).isdigit()]
                    if len(nums) >= 3:
                        try:
                            macro_f1 = float(nums[2])
                        except:
                            pass

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"[WARN] Could not parse {file_path}: {e}")

    print(f"[DEBUG] Parsed from {file_path} -> Accuracy: {acc}, Macro F1: {macro_f1}")
    return acc, macro_f1

# ----------------------------------------------------------
# EXTRACT VALUES
# ----------------------------------------------------------
cnn_acc, cnn_f1 = parse_report(cnn_report_path)
proto_acc, proto_f1 = parse_report(proto_report_path)

# fallback check
if cnn_acc == 0.0:
    print("[HINT] Could not read CNN accuracy; check cnn_metrics_report.txt formatting.")

# ----------------------------------------------------------
# BAR PLOTS
# ----------------------------------------------------------
models = ["Baseline CNN", "ProtoNet Few‑Shot"]
acc_values = [cnn_acc, proto_acc]
f1_values = [cnn_f1, proto_f1]

print(f"\nBaseline CNN  => Acc: {cnn_acc:.2f}, F1: {cnn_f1:.2f}")
print(f"ProtoNet Few‑Shot => Acc: {proto_acc:.2f}, F1: {proto_f1:.2f}")

plt.figure(figsize=(6,4))
plt.bar(models, acc_values, color=["steelblue", "darkorange"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
for i, v in enumerate(acc_values):
    plt.text(i, v+0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
plt.show()

plt.figure(figsize=(6,4))
plt.bar(models, f1_values, color=["skyblue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Macro F1‑Score")
plt.title("Macro F1‑Score Comparison")
for i, v in enumerate(f1_values):
    plt.text(i, v+0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "f1_comparison.png"))
plt.show()

# ----------------------------------------------------------
# SAFELY SHOW CONFUSION MATRICES
# ----------------------------------------------------------
def safe_open(path):
    try:
        return Image.open(path)
    except Exception as e:
        print(f"[WARN] Could not open {path}: {e}")
        from PIL import ImageDraw
        img = Image.new("RGB", (256,256), color="white")
        d = ImageDraw.Draw(img)
        d.text((20,120), "Missing Image", fill="black")
        return img

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(safe_open(cnn_cm_path))
axes[0].set_title("Baseline CNN")
axes[0].axis("off")

axes[1].imshow(safe_open(proto_cm_path))
axes[1].set_title("ProtoNet Few‑Shot")
axes[1].axis("off")

plt.suptitle("Confusion Matrices Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix_comparison.png"))
plt.show()

print(f"\n[INFO] Comparison visuals saved to {save_dir}")