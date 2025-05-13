import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compare_models_heatmap(autoencoder_results, siamese_results):
    data = pd.DataFrame([autoencoder_results, siamese_results], index=["Autoencoder", "Siamese Network"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={"label": "Metric Value"})
    plt.title("Clustering Metric Comparison")
    plt.tight_layout()
    plt.show()
