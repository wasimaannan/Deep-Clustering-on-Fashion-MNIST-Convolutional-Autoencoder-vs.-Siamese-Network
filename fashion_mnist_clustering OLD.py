# 1. Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# 2. Load Fashion-MNIST from Hugging Face
dataset = load_dataset("fashion_mnist")

# 3. Custom Torch Dataset
class FashionMNISTDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
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

train_data = FashionMNISTDataset(dataset["train"], transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# 4. Define Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 → 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 7 * 7 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 → 28x28
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# 5. Train Autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(embedding_dim=32).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training autoencoder...")
model.train()
for epoch in range(100):
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 6. Extract Embeddings
print("Extracting embeddings...")
model.eval()
embeddings = []
labels = []

with torch.no_grad():
    for images, lbls in train_loader:
        images = images.to(device)
        z = model.encoder(images)
        embeddings.append(z.cpu())
        labels.append(lbls)

# Concatenate tensors and convert to NumPy
embeddings = torch.cat(embeddings).numpy()
labels = torch.cat(labels).numpy()
embeddings = StandardScaler().fit_transform(embeddings)

# 7. Clustering with K-Means
print("Clustering...")
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 8. Evaluate with Silhouette Score
print("Evaluating clustering quality...")
sil_score = silhouette_score(embeddings, clusters)
db_index = davies_bouldin_score(embeddings, clusters)
ch_index = calinski_harabasz_score(embeddings, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies Bouldin Score: {db_index:.4f}")
print(f"Calinski Harabasz Score: {ch_index:.4f}")

# Map each predicted cluster to the most common true label in it
label_map = {}
for i in range(10):
    mask = (clusters == i)
    if np.sum(mask) == 0:
        continue
    majority_label = mode(labels[mask], keepdims=False).mode
    label_map[i] = majority_label

# Generate predicted labels based on the mapping
predicted_labels = np.array([label_map[c] for c in clusters])

# Calculate cluster purity (approximated as classification accuracy)
purity = accuracy_score(labels, predicted_labels)
print(f"Cluster Purity (Label Alignment Accuracy): {purity:.4f}")

# 9. Visualize with t-SNE
print("Visualizing cluster separation with t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='tab10', s=10)
plt.title("t-SNE Visualization of Clusters")
plt.colorbar(scatter)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()