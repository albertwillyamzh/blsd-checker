import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

OUTPUT_DIR = "static/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_heatmap(matrix):
    plt.figure(figsize=(4, 4), dpi=120)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm")
    plt.title("Heatmap")
    path = os.path.join(OUTPUT_DIR, "heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.tight_layout(pad=0.3)
    plt.close()
    return path

def generate_confusion_matrix(cm):
    plt.figure(figsize=(4, 4), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.tight_layout(pad=0.3)
    plt.close()
    return path

def generate_roc(fpr, tpr):
    plt.figure(figsize=(5, 3), dpi=120)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    path = os.path.join(OUTPUT_DIR, "roc.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.tight_layout(pad=0.3)
    plt.close()
    return path
