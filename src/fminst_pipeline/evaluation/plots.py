from fminst_pipeline.entity.artifact import Artifact
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], artifact: Artifact):
    try:
        save_path = artifact.get_plot_path(filename="confusion_matrix.png")
        plt.figure(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logger.error(f"Plotting confusion matrix failed - {str(e)}")
        raise e