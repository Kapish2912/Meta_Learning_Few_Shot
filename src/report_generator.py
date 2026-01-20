with open("outputs/week2_analysis/metrics_summary.txt", "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=le.classes_))