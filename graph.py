import matplotlib.pyplot as plt
import numpy as np

def compare_models(autoencoder_results, siamese_results):
    metrics = list(autoencoder_results.keys())
    ae_values = list(autoencoder_results.values())
    si_values = list(siamese_results.values())

    x = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))
    plt.plot(x, ae_values, marker='o', linestyle='-', label='Autoencoder', color='blue')
    plt.plot(x, si_values, marker='s', linestyle='--', label='Siamese Network', color='green')

    plt.xticks(x, metrics)
    plt.ylabel("Metric Value")
    plt.title("Clustering Metrics: Autoencoder vs Siamese Network")
    plt.legend()
    plt.grid(True)

    for i, (ae, si) in enumerate(zip(ae_values, si_values)):
        plt.annotate(f"{ae:.2f}", (x[i], ae), textcoords="offset points", xytext=(0, 5), ha='center', color='blue')
        plt.annotate(f"{si:.2f}", (x[i], si), textcoords="offset points", xytext=(0, -15), ha='center', color='green')

    plt.tight_layout()
    plt.show()
