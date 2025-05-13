import matplotlib.pyplot as plt
import numpy as np

def compare_models_bar(autoencoder_results, siamese_results):
    metrics = list(autoencoder_results.keys())
    ae_values = list(autoencoder_results.values())
    si_values = list(siamese_results.values())

    epsilon = 1e-8
    ae_log = [np.log10(val + epsilon) for val in ae_values]
    si_log = [np.log10(val + epsilon) for val in si_values]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, ae_log, width, label='Autoencoder')
    bars2 = ax.bar(x + width/2, si_log, width, label='Siamese Network')

    ax.set_ylabel('Log₁₀(Score)')
    ax.set_title('Clustering Performance (Log Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def annotate(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    annotate(bars1, ae_values)
    annotate(bars2, si_values)

    fig.tight_layout()
    plt.show()
