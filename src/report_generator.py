from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_metrics_report(y_true, y_pred, labels, model_name="Model", save_dir="outputs"):
    """
    Generate and save classification metrics, confusion matrix and accuracy summary.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Use string labels directly
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{model_name.replace(' ','_').lower()}.png")
    plt.tight_layout(); plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # 2️⃣  Classification Report
    report = classification_report(y_true, y_pred, target_names=labels, digits=2, output_dict=False)
    text_report = classification_report(y_true, y_pred, target_names=labels, digits=2)
    txt_path = os.path.join(save_dir, f"metrics_report_{model_name.replace(' ','_').lower()}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Classification Report ({model_name}) ===\n")
        f.write(text_report)
        f.write(f"\nOverall Accuracy: {acc*100:.2f}%\n")
    print(f"[INFO] Report text saved → {txt_path}")

    # 3️⃣  Also print summary table on console
    print("\n", "="*40)
    print(f"=== Evaluation Summary ({model_name}) ===")
    print(text_report)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("="*40)

    return cm, acc