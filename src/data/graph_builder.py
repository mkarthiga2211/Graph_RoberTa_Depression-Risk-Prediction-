"""
Phase 1: Data & Graph Construction
Responsibility: Graph Construction using PyTorch Geometric (PyG)

This module handles:
- Constructing a Semantic Similarity Graph (k-NN Graph).
- Reason: The primary dataset lacks User IDs/Interactions, so we build edges based on 
  content similarity (TF-IDF Cosine Similarity). Nodes = Posts.
- Converting processed data into PyG Data objects.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InteractionGraphBuilder:
    """
    Builds a Semantic k-NN Graph where nodes are posts and edges represent content similarity.
    """
    def __init__(self, k_neighbors=5, similarity_threshold=0.3):
        self.k = k_neighbors
        self.threshold = similarity_threshold

    def build_graph(self, texts: list, labels: list = None) -> Data:
        """
        Constructs the graph where nodes are Posts and edges represent Semantic Similarity.
        Returns a torch_geometric.data.Data object.
        
        Args:
            texts: List of strings (social media posts).
            labels: Optional list of labels for the nodes.
        """
        num_nodes = len(texts)
        print(f"[-] Building Semantic Graph for {num_nodes} nodes...")
        
        # 1. Compute TF-IDF Features
        # Limit features to keep memory usage low for large graphs
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 2. Compute Cosine Similarity Matrix
        # Note: robust implementation for large data should use Sparse/Annoy, 
        # but standard sklearn is fine for <10k nodes or batches.
        # For simplicity in this skeleton, we assume a reasonable batch size.
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # 3. Build Edges (k-NN)
        sources = []
        targets = []
        weights = []
        
        # Remove self-loops by settings diag to 0
        np.fill_diagonal(sim_matrix, 0)
        
        for i in range(num_nodes):
            # Get indices of top k most similar posts
            # argsort returns low-to-high, so we take last k
            row = sim_matrix[i]
            top_k_indices = row.argsort()[-self.k:]
            
            for j in top_k_indices:
                score = row[j]
                if score > self.threshold:
                    sources.append(i)
                    targets.append(j)
                    weights.append(score)
                    
        # 4. Create PyG Tensors
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float)
        
        # Node features (Identity or placeholder, RoBERTa will provide real ones later)
        x = torch.zeros((num_nodes, 1)) # Placeholder
        
        # Labels
        y = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        print(f"    [+] Semantic Graph Built: {data.num_nodes} nodes, {data.num_edges} edges.")
        return data

    def save_graph(self, graph: Data, path: str):
        """
        Saves the PyG Data object to disk.
        """
        print(f"[-] Saving graph to {path}...")
        torch.save(graph, path)
        print("[+] Graph saved successfully.")

    def load_graph(self, path: str) -> Data:
        """
        Loads a PyG Data object from disk.
        """
        print(f"[-] Loading graph from {path}...")
        data = torch.load(path)
        print(f"[+] Graph loaded: {data}")
        return data

    def get_node_features(self, graph: Data) -> torch.Tensor:
        """
        Extracts node features from the graph.
        """
        return graph.x
