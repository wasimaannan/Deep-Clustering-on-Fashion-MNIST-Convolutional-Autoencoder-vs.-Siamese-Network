import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def run_siamese_clustering():
    dataset = load_dataset("fashion_mnist")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    class SiameseFashionDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.data = hf_dataset
            self.transform = transform
            self.label_to_indices = self._build_label_index()

        def _build_label_index(self):
            label_to_indices = {}
            for i, item in enumerate(self.data):
                label = item["label"]
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(i)
            return label_to_indices

        def __getitem__(self, index):
            img1, label1 = self.data[index]["image"], self.data[index]["label"]
            if random.random() < 0.5:
                label2 = label1
                index2 = random.choice(self.label_to_indices[label2])
            else:
                label2 = random.choice([l for l in self.label_to_indices if l != label1])
                index2 = random.choice(self.label_to_indices[label2])

            img2 = self.data[index2]["image"]

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            label = torch.tensor(int(label1 == label2), dtype=torch.float32)
            return img1, img2, label

        def __len__(self):
            return len(self.data)

    train_dataset = SiameseFashionDataset(dataset["train"], transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    class EmbeddingNet(nn.Module):
        def __init__(self):
            super(EmbeddingNet, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

        def forward(self, x):
            return self.net(x)

    class SiameseNetwork(nn.Module):
        def __init__(self, embedding_net):
            super(SiameseNetwork, self).__init__()
            self.embedding_net = embedding_net

        def forward(self, x1, x2):
            z1 = self.embedding_net(x1)
            z2 = self.embedding_net(x2)
            return z1, z2

    # Contrastive Loss 
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, z1, z2, label):
            d = F.pairwise_distance(z1, z2)
            loss = label * d.pow(2) + (1 - label) * F.relu(self.margin - d).pow(2)
            return loss.mean()

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_net = EmbeddingNet().to(device)
    model = SiameseNetwork(embedding_net).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            z1, z2 = model(img1, img2)
            loss = criterion(z1, z2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

    # Embeddings for Clustering 
    class FashionEmbeddingDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.data = hf_dataset
            self.transform = transform

        def __getitem__(self, index):
            img = self.data[index]["image"]
            label = self.data[index]["label"]
            if self.transform:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.data)

    eval_dataset = FashionEmbeddingDataset(dataset["train"], transform)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    all_embeddings, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(device)
            embeddings = embedding_net(imgs).cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    # Clustering Evaluation 
    kmeans = KMeans(n_clusters=10, random_state=42).fit(all_embeddings)
    sil_score = silhouette_score(all_embeddings, kmeans.labels_)
    db_index = davies_bouldin_score(all_embeddings, kmeans.labels_)
    ch_index = calinski_harabasz_score(all_embeddings, kmeans.labels_)

    # print(f"Silhouette Score: {sil_score:.4f}")
    # print(f"Davies-Bouldin Index: {db_index:.4f}")
    # print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    # t-SNE 
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_embeddings[:2000])  

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                        c=kmeans.labels_[:2000], cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Clusters (Siamese Embeddings)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    # Confusion Matrix 
    from sklearn.metrics import confusion_matrix
    from scipy.stats import mode

    def cluster_label_map(true_labels, cluster_labels, n_clusters=10):
        mapping = {}
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) == 0:
                continue
            majority_label = mode(true_labels[mask], keepdims=False).mode
            mapping[i] = majority_label
        return np.array([mapping[c] for c in cluster_labels])

    mapped_labels = cluster_label_map(all_labels[:2000], kmeans.labels_[:2000])
    conf_mat = confusion_matrix(all_labels[:2000], mapped_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Cluster Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: Cluster vs True Class (Siamese)")
    plt.tight_layout()
    plt.show()
    return {
        "Silhouette": sil_score,
        "DB Index": db_index,
        "CH Index": ch_index
    }
