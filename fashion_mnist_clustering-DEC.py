# 1. Install and Import Required Libraries
# Uncomment the next line if running in a notebook or new environment
# !pip install torch torchvision datasets transformers scikit-learn matplotlib seaborn

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

# 2. Load Fashion-MNIST using Hugging Face Datasets
hf_dataset = load_dataset("fashion_mnist")

# 3. Custom Dataset Wrapper
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

# 4. Define Convolutional Autoencoder with BatchNorm and Dropout
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

# 5. Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train Autoencoder
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

# 7. Extract Latent Representations
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

# 8. Define DEC Clustering Layer and KL Divergence Loss
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, embedding_dim):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embedding_dim))

    def forward(self, z):
        # Compute Student's t-distribution (q_ij)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2))
        q = q ** ((1 + 1) / 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

# 9. Initialize Clustering Layer with Pretrained Encoder Embeddings
train_latent, train_labels = extract_latents(model, train_loader)
train_latent = StandardScaler().fit_transform(train_latent)

# Run KMeans to initialize DEC cluster centers
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
cluster_ids = kmeans.fit_predict(train_latent)

# Attach DEC layer
dec_layer = ClusteringLayer(n_clusters=n_clusters, embedding_dim=32).to(device)
dec_layer.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

# 10. DEC Training (Fine-tune Encoder + Cluster Assignments)
dec_optimizer = optim.Adam(list(model.encoder.parameters()) + list(dec_layer.parameters()), lr=0.001)
dec_epochs = 50
update_interval = 10

print("\nFine-tuning with DEC...")
for epoch in range(dec_epochs):
    model.eval()
    all_q, all_labels = [], []

    # Step 1: Compute soft assignments (q_ij) for entire dataset
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            z, _ = model(data)
            q = dec_layer(z)
            all_q.append(q.cpu())
            all_labels.append(labels)
    all_q = torch.cat(all_q)
    all_labels = torch.cat(all_labels)
    p = target_distribution(all_q)

    # Step 2: Train with KL divergence loss
    model.train()
    idx = 0
    for data, _ in train_loader:
        data = data.to(device)
        batch_size = data.size(0)

        z, _ = model(data)
        q_batch = dec_layer(z)
        p_batch = p[idx:idx + batch_size].to(device)
        idx += batch_size

        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(q_batch), p_batch)

        dec_optimizer.zero_grad()
        loss.backward()
        dec_optimizer.step()

    print(f"DEC Epoch [{epoch + 1}/{dec_epochs}], KL Loss: {loss.item():.4f}")

# 11. Final Inference: Assign Clusters
model.eval()
dec_layer.eval()
final_latents, final_labels = extract_latents(model, train_loader)
final_latents = torch.tensor(StandardScaler().fit_transform(final_latents)).to(device)

with torch.no_grad():
    z = final_latents
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - dec_layer.cluster_centers.data.cpu())**2, dim=2))
    q = q ** ((1 + 1) / 2)
    q = (q.t() / torch.sum(q, dim=1)).t()
    final_cluster_assignments = torch.argmax(q, dim=1).cpu().numpy()

# 12. Evaluation
from sklearn.metrics import accuracy_score
from scipy.stats import mode

def cluster_purity(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            continue
        labels[mask] = mode(y_true[mask], keepdims=False).mode
    return accuracy_score(y_true, labels)

silhouette = silhouette_score(final_latents.cpu(), final_cluster_assignments)
davies_bouldin = davies_bouldin_score(final_latents.cpu(), final_cluster_assignments)
calinski_harabasz = calinski_harabasz_score(final_latents.cpu(), final_cluster_assignments)
purity = cluster_purity(train_labels, final_cluster_assignments)

print(f"\nDEC Results:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Cluster Purity: {purity:.4f}")

# 13. Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(final_latents.cpu().numpy()[:2000])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=final_cluster_assignments[:2000], cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of DEC Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Confusion matrix
confusion_matrix = np.zeros((10, 10))
for true_label, cluster_label in zip(train_labels[:2000], final_cluster_assignments[:2000]):
    confusion_matrix[true_label, cluster_label] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
plt.xlabel('Cluster Label')
plt.ylabel('True Label')
plt.title('Cluster Assignment vs True Label (DEC)')
plt.show()
