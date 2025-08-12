import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft
from tqdm import tqdm


def graph_cluster_sims(decoder, top_k, sim_cutoff=0.5, prune_clusters=False):
    """
    Create a graph from similarity scores, keeping only the top_k neighbors
    """

    all_sims = decoder @ decoder.T
    all_sims.fill_diagonal_(0)

    near_neighbors = torch.topk(all_sims, k=top_k, dim=1)

    graph = [[] for _ in range(all_sims.shape[0])]

    for i in range(all_sims.shape[0]):
        top_indices = near_neighbors.indices[i]
        top_sims = near_neighbors.values[i]
        top_indices = top_indices[top_sims > sim_cutoff]
        graph[i] = top_indices.tolist()

    for i in tqdm(range(all_sims.shape[0])):
        for j in graph[i]:
            if i not in graph[j]:
                graph[j].append(i)

    visited = [False] * all_sims.shape[0]
    components = []
    for i in range(all_sims.shape[0]):
        if visited[i]:
            continue
        component = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            stack.extend(graph[node])
        components.append(component)

    if prune_clusters:
        threshold = 3000
        components = [c for c in components if len(c) < threshold and len(c) > 1]

    print(f"Found {len(components)} clusters with size < {threshold} and > 1")

    with open(f"clusters_{top_k}_sim_cutoff_{sim_cutoff}.pkl", "wb") as f:
        pickle.dump(components, f)
    print(f"Saved clusters to clusters_{top_k}_sim_cutoff_{sim_cutoff}.pkl")


def plot_pca_fourier_components(output_pca, max_freq=56, plot=True):
    """
    Analyze Fourier components of PCA output
    output_pca: shape (113, 7) - your PCA components
    """
    P, n_components = output_pca.shape  # (113, 7)

    # Apply FFT along the first axis (the 113 samples)
    pca_fft = fft(output_pca, axis=0)  # Shape: (113, 7)

    # Extract cosine and sine components for each PCA component
    cos_components = np.real(pca_fft)  # Real part = cosine components
    sin_components = np.imag(pca_fft)  # Imaginary part = sine components

    # Calculate norms across the 7 PCA dimensions for each frequency
    cos_norms = np.linalg.norm(cos_components, axis=1)  # Shape: (113,)
    sin_norms = np.linalg.norm(sin_components, axis=1)  # Shape: (113,)

    freqs = np.arange(max_freq + 1)

    if plot:
        # Plot overall frequency spectrum
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.bar(freqs, cos_norms[: max_freq + 1], color="b", label="cos", alpha=0.7)
        plt.bar(freqs, sin_norms[: max_freq + 1], color="r", label="sin", alpha=0.7)
        plt.xlabel("Frequency k")
        plt.ylabel("Norm across all PCA components")
        plt.title("Overall Fourier Components (PCA)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max_freq)

        # Plot individual PCA component spectra
        for i in range(min(3, n_components)):  # Show first 3 components
            plt.subplot(2, 2, i + 2)

            cos_single = np.abs(cos_components[: max_freq + 1, i])
            sin_single = np.abs(sin_components[: max_freq + 1, i])

            plt.bar(freqs, cos_single, color="b", label="cos", alpha=0.7)
            plt.bar(freqs, sin_single, color="r", label="sin", alpha=0.7)
            plt.xlabel("Frequency k")
            plt.ylabel("Magnitude")
            plt.title(f"PCA Component {i + 1}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max_freq)

        plt.tight_layout()
        plt.show()

    # Get top 3 frequencies for each PCA component (cos and sin combined)
    top_frequencies = {}
    for i in range(n_components):
        cos_single = np.abs(cos_components[:57, i])
        sin_single = np.abs(sin_components[:57, i])

        # Combine cos and sin frequencies for this component
        component_frequencies = []
        for freq in range(57):
            component_frequencies.append(
                {"type": "cos", "frequency": freq, "magnitude": cos_single[freq]}
            )
            component_frequencies.append(
                {"type": "sin", "frequency": freq, "magnitude": sin_single[freq]}
            )

        # Sort by magnitude and get top 3
        component_frequencies.sort(key=lambda x: x["magnitude"], reverse=True)
        top_frequencies[i] = component_frequencies[:3]

    return top_frequencies
