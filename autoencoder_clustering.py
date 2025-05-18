import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

def run_autoencoder_clustering():
    hf_dataset = load_dataset("fashion_mnist")
    from torch.utils.data import Dataset
    from PIL import Image

    class FashionMNISTDataset(Dataset):
        def __init__(self, dataset_split, transform=None):
            self.dataset = dataset_split
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]["image"]
            label = self.dataset[idx]["label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNISTDataset(hf_dataset["train"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    class ConvAutoencoder(nn.Module):
        def __init__(self, embedding_dim=32):
            super(ConvAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),  
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, embedding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 7 * 7 * 64),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return z, out

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Autoencoder
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            encoded, decoded = model(data)
            loss = criterion(decoded, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Extract Latent Representations
    def extract_latents(model, dataloader):
        model.eval()
        all_latents, all_labels = [], []
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                encoded, _ = model(data)
                all_latents.append(encoded.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.concatenate(all_latents), np.concatenate(all_labels)

    train_latent, train_labels = extract_latents(model, train_loader)

    # Normalize embeddings
    train_latent = StandardScaler().fit_transform(train_latent)

    # K-Means Clustering
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(train_latent)

    # Evaluation Metrics
    silhouette = silhouette_score(train_latent, cluster_labels)
    davies_bouldin = davies_bouldin_score(train_latent, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(train_latent, cluster_labels)

    # print(f"Silhouette Score: {silhouette:.4f}")
    # print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    # print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

    # t-SNE 
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(train_latent[:2000])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels[:2000], cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

    # Confusion Matrix
    confusion_matrix = np.zeros((10, 10))
    for true_label, cluster_label in zip(train_labels[:2000], cluster_labels[:2000]):
        confusion_matrix[true_label, cluster_label] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
    plt.xlabel('Cluster Label')
    plt.ylabel('True Label')
    plt.title('Cluster Assignment vs True Label')
    plt.show()
    return {
        "Silhouette": silhouette,
        "DB Index": davies_bouldin,
        "CH Index": calinski_harabasz
    }
