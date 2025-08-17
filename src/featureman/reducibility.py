import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.special import expit
import os
from datetime import datetime
import time

def estimate_mutual_information(a, b, n_bins=40, clip_range=6):
    """
    Estimate mutual information between a and b using binning
    """
    # Clip to [-clip_range, clip_range] and bin
    a_clipped = np.clip(a, -clip_range, clip_range)
    b_clipped = np.clip(b, -clip_range, clip_range)
    
    # Create 2D histogram
    hist_2d, _, _ = np.histogram2d(a_clipped, b_clipped, bins=n_bins, 
                                  range=[[-clip_range, clip_range], 
                                         [-clip_range, clip_range]])
    
    # Add small epsilon to avoid log(0)
    hist_2d = hist_2d + 1e-10
    
    # Normalize to get probabilities
    p_ab = hist_2d / np.sum(hist_2d)
    p_a = np.sum(p_ab, axis=1)
    p_b = np.sum(p_ab, axis=0)
    
    # Compute mutual information
    mi = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_ab[i, j] > 1e-10:
                mi += p_ab[i, j] * np.log2(p_ab[i, j] / (p_a[i] * p_b[j]))
    
    return mi

def compute_separability_index_silent(points, n_angles=1000):
    """Silent version - no progress prints"""
    # Normalize points
    points_centered = points - np.mean(points, axis=0)
    rms_norm = np.sqrt(np.mean(np.sum(points_centered**2, axis=1)))
    points_normalized = points_centered / rms_norm * np.sqrt(2)
    
    angles = np.linspace(0, 2*np.pi, n_angles)
    mutual_infos = []
    
    for angle in angles:
        # Rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        
        # Rotate points
        rotated = points_normalized @ R.T
        a, b = rotated[:, 0], rotated[:, 1]
        
        # Estimate mutual information using binning
        mi = estimate_mutual_information(a, b)
        mutual_infos.append(mi)
    
    min_mi = np.min(mutual_infos)
    best_angle = angles[np.argmin(mutual_infos)]
    
    return min_mi, best_angle, np.array(mutual_infos)

def compute_mixture_index_silent(points, epsilon=0.1, n_steps=200, lr=0.1):
    """Silent version - no progress prints, reduced steps for speed"""
    n_samples, n_dims = points.shape
    
    # Initialize random direction and offset
    v = np.random.randn(n_dims)
    v = v / np.linalg.norm(v)
    c = np.random.randn()
    
    best_fraction = 0
    best_v = v.copy()
    best_c = c
    
    # Temperature schedule
    temperatures = np.linspace(1.0, 0.0, n_steps)
    
    for step in range(n_steps):
        T = temperatures[step]
        
        # Compute projections
        projections = points @ v + c
        rms = np.sqrt(np.mean(projections**2))
        normalized_projections = projections / (rms + 1e-8)
        
        # Soft version of the indicator function using sigmoid
        indicators = expit((epsilon - np.abs(normalized_projections)) / (T + 1e-8))
        current_fraction = np.mean(indicators)
        
        if current_fraction > best_fraction:
            best_fraction = current_fraction
            best_v = v.copy()
            best_c = c
        
        # Compute gradients
        grad_v = np.zeros_like(v)
        grad_c = 0
        
        for i in range(n_samples):
            proj = projections[i]
            norm_proj = normalized_projections[i]
            indicator = indicators[i]
            
            sigmoid_grad = indicator * (1 - indicator)
            sign_proj = np.sign(norm_proj)
            
            grad_v += sigmoid_grad * (-sign_proj / (T + 1e-8)) * (points[i] / (rms + 1e-8))
            grad_c += sigmoid_grad * (-sign_proj / (T + 1e-8)) * (1 / (rms + 1e-8))
        
        grad_v /= n_samples
        grad_c /= n_samples
        
        # Update parameters
        v += lr * grad_v
        v = v / (np.linalg.norm(v) + 1e-8)
        c += lr * grad_c
    
    return best_fraction, best_v, best_c

def test_irreducibility_on_projections_silent(cluster_reconstructions, pc_pairs=None, epsilon=0.1):
    """
    Silent version of the projection testing - minimal output
    """
    # Fit PCA on full data
    pca = PCA()
    pca_transformed = pca.fit_transform(cluster_reconstructions)
    
    # Determine which PC pairs to test
    if pc_pairs is None:
        pc_pairs = [(1, 2), (0, 1), (2, 3), (0, 2)]
        pc_pairs = [(i, j) for i, j in pc_pairs 
                   if i < len(pca.explained_variance_ratio_) and 
                      j < len(pca.explained_variance_ratio_) and
                      pca.explained_variance_ratio_[i] > 0.01 and 
                      pca.explained_variance_ratio_[j] > 0.01]
    
    results = {}
    
    for pc_i, pc_j in pc_pairs:
        # Extract 2D projection
        points_2d = pca_transformed[:, [pc_i, pc_j]]
        
        # Test this specific 2D projection (silent versions)
        sep_index, best_angle, mi_curve = compute_separability_index_silent(points_2d)
        mix_index, best_v, best_c = compute_mixture_index_silent(points_2d, epsilon=epsilon)
        
        results[(pc_i, pc_j)] = {
            'separability_index': sep_index,
            'mixture_index': mix_index,
            'variance_explained': (pca.explained_variance_ratio_[pc_i], 
                                 pca.explained_variance_ratio_[pc_j]),
            'points_2d': points_2d,
            'best_angle': best_angle,
            'mi_curve': mi_curve,
            'mixture_direction': best_v,
            'mixture_offset': best_c
        }
    
    return pca, results

def get_best_projection(pca, results):
    """
    Find the projection with the best irreducibility scores
    """
    best_score = -1
    best_pair = None
    
    for pc_pair, result in results.items():
        # Combine the metrics: High separability + Low mixture = good irreducibility
        score = result['separability_index'] * (1 - result['mixture_index'])
        
        if score > best_score:
            best_score = score
            best_pair = pc_pair
    
    return best_pair, results[best_pair]

def compute_aggregate_scores(results):
    """
    Compute mean scores across all projections
    """
    sep_scores = [r['separability_index'] for r in results.values()]
    mix_scores = [r['mixture_index'] for r in results.values()]
    
    mean_sep = np.mean(sep_scores)
    mean_mix = np.mean(mix_scores)
    
    return mean_sep, mean_mix

def plot_and_save_projection(pc_pair, result, cluster_idx, save_dir="irreducibility_results"):
    """Save the projection plots instead of displaying them"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    points = result['points_2d']
    sep_idx = result['separability_index']
    mix_idx = result['mixture_index']
    
    # Scatter plot
    axes[0].scatter(points[:, 0], points[:, 1], alpha=0.6, s=20)
    axes[0].set_xlabel(f'PC{pc_pair[0]}')
    axes[0].set_ylabel(f'PC{pc_pair[1]}')
    axes[0].set_title(f'PC{pc_pair[0]}-PC{pc_pair[1]} Projection')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Mixture test histogram
    v, c = result['mixture_direction'], result['mixture_offset']
    projections = points @ v + c
    rms = np.sqrt(np.mean(projections**2))
    normalized_proj = projections / rms
    
    axes[1].hist(normalized_proj, bins=40, alpha=0.7, density=True)
    axes[1].axvline(-0.1, color='red', linestyle='--', label='ε band')
    axes[1].axvline(0.1, color='red', linestyle='--')
    axes[1].set_title(f'M_ε(f) = {mix_idx:.3f}')
    axes[1].set_xlabel('Normalized projection')
    axes[1].legend()
    
    # Separability test
    angles = np.linspace(0, 2*np.pi, len(result['mi_curve']))
    axes[2].plot(angles, result['mi_curve'])
    axes[2].axhline(sep_idx, color='red', linestyle='--')
    axes[2].set_title(f'S(f) = {sep_idx:.3f}')
    axes[2].set_xlabel('Rotation angle')
    axes[2].set_ylabel('Mutual Info')
    axes[2].grid(True, alpha=0.3)
    
    # Overall title with key metrics
    fig.suptitle(f'Cluster {cluster_idx} - PC{pc_pair[0]}-PC{pc_pair[1]} | S(f)={sep_idx:.3f}, M_ε(f)={mix_idx:.3f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"cluster_{cluster_idx:03d}_PC{pc_pair[0]}-PC{pc_pair[1]}_S{sep_idx:.3f}_M{mix_idx:.3f}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()  # Important: close to free memory
    
    return filepath

def analyze_cluster_irreducibility_silent(cluster_reconstructions, cluster_idx, save_plots=True, save_dir="irreducibility_results"):
    """
    Silent version that saves plots and returns summary data
    """
    # Test multiple 2D projections (no progress prints)
    pca, results = test_irreducibility_on_projections_silent(cluster_reconstructions)
    
    # Get aggregate scores
    mean_sep, mean_mix = compute_aggregate_scores(results)
    
    # Find best projection
    best_pair, best_result = get_best_projection(pca, results)
    
    # Save plots for all PC pairs
    saved_plots = []
    if save_plots:
        for pc_pair, result in results.items():
            filepath = plot_and_save_projection(pc_pair, result, cluster_idx, save_dir)
            saved_plots.append(filepath)
    
    # Return summary data
    summary = {
        'cluster_idx': cluster_idx,
        'mean_separability': mean_sep,
        'mean_mixture': mean_mix,
        'best_projection': best_pair,
        'best_separability': best_result['separability_index'],
        'best_mixture': best_result['mixture_index'],
        'pca_variance': pca.explained_variance_ratio_[:4].tolist(),
        'all_results': results,
        'saved_plots': saved_plots,
        'is_irreducible': mean_sep > 0.4 and mean_mix < 0.4
    }
    
    return summary

def save_analysis_summary(all_summaries, save_dir="irreducibility_results"):
    """Save a CSV summary of all cluster analyses"""
    import pandas as pd
    
    # Create summary dataframe
    summary_data = []
    for summary in all_summaries:
        row = {
            'cluster_idx': summary['cluster_idx'],
            'mean_separability': summary['mean_separability'],
            'mean_mixture': summary['mean_mixture'],
            'best_separability': summary['best_separability'],
            'best_mixture': summary['best_mixture'],
            'best_projection': f"PC{summary['best_projection'][0]}-PC{summary['best_projection'][1]}",
            'pc1_variance': summary['pca_variance'][0] if len(summary['pca_variance']) > 0 else 0,
            'pc2_variance': summary['pca_variance'][1] if len(summary['pca_variance']) > 1 else 0,
            'pc3_variance': summary['pca_variance'][2] if len(summary['pca_variance']) > 2 else 0,
            'is_irreducible': summary['is_irreducible'],
            'irreducibility_score': summary['mean_separability'] * (1 - summary['mean_mixture'])
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Sort by irreducibility score
    df = df.sort_values('irreducibility_score', ascending=False)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"irreducibility_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Summary saved to: {csv_path}")
    print(f"🏆 Top 5 irreducible clusters:")
    print(df.head()[['cluster_idx', 'irreducibility_score', 'mean_separability', 'mean_mixture', 'is_irreducible']])
    
    return df, csv_path