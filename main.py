from autoencoder_clustering import run_autoencoder_clustering
from siamese_clustering import run_siamese_clustering
from graph import compare_models
from heatmap import compare_models_heatmap
from bar_chart import compare_models_bar

print("Autoencoder: ")
ae_results = run_autoencoder_clustering()
print("Siamese: ")
siamese_results = run_siamese_clustering()

print("Autoencoder Results:")
for key, value in ae_results.items():
    print(f"  {key}: {value:.4f}")

print("Siamese Results:")
for key, value in siamese_results.items():
    print(f"  {key}: {value:.4f}")

compare_models(ae_results, siamese_results)
compare_models_heatmap(ae_results, siamese_results)
compare_models_bar(ae_results, siamese_results)