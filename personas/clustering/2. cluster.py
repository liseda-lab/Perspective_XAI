#!/usr/bin/env python3
"""
User Preference Clustering Analysis Script

This script performs comprehensive clustering analysis on user preference embeddings:
1. K-means clustering with visualization
2. Hierarchical clustering with automatic threshold selection
3. HDBSCAN density-based clustering
4. Comparison and persona assignment
5. Results saving with timestamped outputs
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# Clustering and analysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import hdbscan


# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UserPreferenceClustering:
    def __init__(self, json_file_path):
        """Initialize with the path to JSON file containing user embeddings."""
        self.json_file_path = json_file_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"clustering_results_{self.timestamp}"
        
        # Data storage
        self.user_data = None
        self.embeddings = None
        self.user_ids = None
        self.embeddings_normalized = None
        
        # Dimensionality reduction
        self.pca_2d = None
        self.tsne_2d = None
        
        # Clustering results
        self.kmeans_labels = None
        self.hierarchical_labels = None
        self.hdbscan_labels = None
        
        # Final persona assignments
        self.persona_mapping = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        
    def load_and_preprocess_data(self):
        """Load JSON data and preprocess embeddings."""
        print("Loading user preference embeddings...")
        
        with open(self.json_file_path, 'r') as f:
            self.user_data = json.load(f)
        
        # Extract user IDs and embeddings
        self.user_ids = [record['id'] for record in self.user_data]
        self.embeddings = np.array([record['embedding'] for record in self.user_data])
        
        print(f"Loaded {len(self.user_ids)} users with {self.embeddings.shape[1]}-dimensional embeddings")
        
        # L2 normalization
        self.embeddings_normalized = normalize(self.embeddings, norm='l2')
        print("Applied L2 normalization to embeddings")
        
        # Dimensionality reduction for visualization
        print("Computing PCA and t-SNE for visualization...")
        pca = PCA(n_components=2, random_state=42)
        self.pca_2d = pca.fit_transform(self.embeddings_normalized)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.user_ids)-1))
        self.tsne_2d = tsne.fit_transform(self.embeddings_normalized)
    

    def find_optimal_k(self, k_min=2, k_max=5):
        """Scan k from k_min to k_max and pick the one maximizing silhouette."""
        silhouettes = []
        inertias = []
        Ks = list(range(k_min, k_max + 1))
        
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.embeddings_normalized)
            sil = silhouette_score(self.embeddings_normalized, labels)
            silhouettes.append(sil)
            inertias.append(km.inertia_)
            print(f"k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.4f}")
        
        best_idx = int(np.argmax(silhouettes))
        best_k = Ks[best_idx]
        print(f"→ Optimal k by silhouette: {best_k}")
        
        # (Optional) plot Silhouette & Elbow curves for inspection
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(Ks, silhouettes, marker='o')
        plt.title("Silhouette vs. k")
        plt.xlabel("k")
        plt.ylabel("Silhouette Score")
        
        plt.subplot(1,2,2)
        plt.plot(Ks, inertias, marker='o')
        plt.title("Inertia (Elbow) vs. k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_k

    def perform_kmeans_clustering(self):
        """Perform K-means using the best k from silhouette analysis."""
        print("\nFinding optimal number of clusters for K-means…")
        n_samples = self.embeddings_normalized.shape[0]
        k_max = min(10, n_samples - 1)  # Limit k to avoid overfitting
        k_opt = self.find_optimal_k(k_min=2, k_max=k_max)
        
        print(f"\nRunning K-means with k={k_opt}")
        kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
        self.kmeans_labels = kmeans.fit_predict(self.embeddings_normalized)
        
        # metrics
        sil = silhouette_score(self.embeddings_normalized, self.kmeans_labels)
        print(f"Silhouette (k={k_opt}): {sil:.4f}")
        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        for c, cnt in zip(unique, counts):
            print(f"Cluster {c}: {cnt} users")
        
        return k_opt, sil

    def run_complete_analysis(self):
        """Updated driver to accept dynamic k for personas."""
        self.load_and_preprocess_data()
        k_opt, sil = self.perform_kmeans_clustering()
        # … hierarchical and HDBSCAN unchanged …
        linkage_matrix, threshold, hier_outliers = self.perform_hierarchical_clustering()
        hdbscan_outliers = self.perform_hdbscan_clustering()
        comparison_df = self.compare_and_assign_personas(hier_outliers, hdbscan_outliers)
        self.create_visualizations(linkage_matrix)
        self.save_results(comparison_df)

    
    def perform_kmeans_clustering(self):
        """Perform K-means clustering with best k ."""
        print("\n" + "="*50)
        print("PERFORMING K-MEANS CLUSTERING (k=best)")
        print("="*50)
        n_samples = self.embeddings_normalized.shape[0]
        k_max = min(10, n_samples - 1)
        k_opt = self.find_optimal_k(k_min=2, k_max=k_max)
        #kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
        self.kmeans_labels = kmeans.fit_predict(self.embeddings_normalized)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(self.embeddings_normalized, self.kmeans_labels)
        
        print(f"K-means Inertia: {inertia:.4f}")
        print(f"K-means Silhouette Score: {silhouette_avg:.4f}")
        
        # Print cluster sizes
        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"Cluster {cluster}: {count} users")
            
        return inertia, silhouette_avg
    
    def perform_hierarchical_clustering(self):
        """Perform hierarchical clustering with automatic threshold selection."""
        print("\n" + "="*50)
        print("PERFORMING HIERARCHICAL CLUSTERING")
        print("="*50)
        
        # Calculate linkage matrix using Ward method
        linkage_matrix = linkage(self.embeddings_normalized, method='ward')
        
        # Automatically select threshold to get 3 main clusters
        # We'll try different thresholds and find one that gives us 3 main clusters
        distances = linkage_matrix[:, 2]
        threshold_candidates = np.linspace(distances.min(), distances.max(), 100)
        
        best_threshold = None
        for threshold in reversed(threshold_candidates):  # Start from high threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            n_clusters = len(np.unique(clusters))
            if n_clusters >= 3:
                best_threshold = threshold
                break
        
        if best_threshold is None:
            best_threshold = np.percentile(distances, 70)  # Fallback
            
        self.hierarchical_labels = fcluster(linkage_matrix, best_threshold, criterion='distance')
        
        print(f"Selected threshold: {best_threshold:.4f}")
        
        # Analyze cluster sizes and identify outliers
        unique, counts = np.unique(self.hierarchical_labels, return_counts=True)
        cluster_stats = list(zip(unique, counts))
        cluster_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by size
        
        print("Hierarchical clustering results:")
        main_clusters = []
        outlier_users = []
        
        for cluster_id, count in cluster_stats:
            user_indices = np.where(self.hierarchical_labels == cluster_id)[0]
            user_ids_in_cluster = [self.user_ids[i] for i in user_indices]
            
            if count == 1:  # Singleton
                print(f"Singleton cluster {cluster_id}: 1 user - {user_ids_in_cluster[0]}")
                outlier_users.extend(user_ids_in_cluster)
            elif len(main_clusters) < 3:  # Main cluster
                print(f"Main cluster {cluster_id}: {count} users")
                main_clusters.append(cluster_id)
            else:  # Small cluster treated as outliers
                print(f"Small cluster {cluster_id}: {count} users - treating as outliers")
                outlier_users.extend(user_ids_in_cluster)
        
        print(f"\nUsers outside the three main branches: {outlier_users}")
        
        return linkage_matrix, best_threshold, outlier_users
    
    def perform_hdbscan_clustering(self):
        """Perform HDBSCAN clustering to detect dense clusters and outliers."""
        print("\n" + "="*50)
        print("PERFORMING HDBSCAN CLUSTERING")
        print("="*50)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        self.hdbscan_labels = clusterer.fit_predict(self.embeddings_normalized)
        
        # Analyze results
        unique, counts = np.unique(self.hdbscan_labels, return_counts=True)
        n_clusters = len(unique) - (1 if -1 in unique else 0)  # -1 indicates noise/outliers
        n_outliers = counts[unique == -1][0] if -1 in unique else 0
        
        print(f"HDBSCAN found {n_clusters} dense clusters")
        print(f"HDBSCAN identified {n_outliers} outliers")
        
        # Print cluster information
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                outlier_indices = np.where(self.hdbscan_labels == -1)[0]
                outlier_user_ids = [self.user_ids[i] for i in outlier_indices]
                print(f"Outliers (cluster -1): {count} users - {outlier_user_ids}")
            else:
                print(f"Dense cluster {cluster_id}: {count} users")
        
        # Get outlier user IDs
        outlier_indices = np.where(self.hdbscan_labels == -1)[0]
        hdbscan_outliers = [self.user_ids[i] for i in outlier_indices]
        
        return hdbscan_outliers
    
    def compare_and_assign_personas(self, hierarchical_outliers, hdbscan_outliers):
        """Compare clustering methods and assign final personas."""
        print("\n" + "="*50)
        print("COMPARING CLUSTERING METHODS & ASSIGNING PERSONAS")
        print("="*50)
        
        # Find users flagged as outliers by both methods
        consensus_outliers = list(set(hierarchical_outliers) & set(hdbscan_outliers))
        print(f"Users flagged as outliers by both hierarchical and HDBSCAN: {consensus_outliers}")
        
        # Assign personas based on K-means clusters (our base method)
        self.persona_mapping = {}
        
        for i, user_id in enumerate(self.user_ids):
            kmeans_cluster = self.kmeans_labels[i]
            
            if user_id in consensus_outliers:
                self.persona_mapping[user_id] = "Persona 4"  # Outlier persona
            else:
                self.persona_mapping[user_id] = f"Persona {kmeans_cluster + 1}"  # Personas 1, 2, 3
        
        # Print persona assignments summary
        persona_counts = {}
        for persona in self.persona_mapping.values():
            persona_counts[persona] = persona_counts.get(persona, 0) + 1
        
        print("\nFinal Persona Assignments:")
        for persona in sorted(persona_counts.keys()):
            print(f"{persona}: {persona_counts[persona]} users")
        
        # Print detailed comparison
        print("\nDetailed clustering comparison:")
        comparison_df = pd.DataFrame({
            'User_ID': self.user_ids,
            'K-means': [f"Cluster {label}" for label in self.kmeans_labels],
            'Hierarchical': [f"Cluster {label}" for label in self.hierarchical_labels],
            'HDBSCAN': [f"Cluster {label}" if label != -1 else "Outlier" for label in self.hdbscan_labels],
            'Final_Persona': list(self.persona_mapping.values())
        })
        
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_visualizations(self, linkage_matrix):
        """Create all visualization plots."""
        print("\nCreating visualizations...")
        
        # Set up the plotting
        fig = plt.figure(figsize=(20, 15))
        
        # 1. K-means PCA plot
        plt.subplot(3, 3, 1)
        scatter = plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                            c=self.kmeans_labels, cmap='viridis', s=50, alpha=0.7)
        plt.title('K-means Clustering (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # 2. K-means t-SNE plot
        plt.subplot(3, 3, 2)
        scatter = plt.scatter(self.tsne_2d[:, 0], self.tsne_2d[:, 1], 
                            c=self.kmeans_labels, cmap='viridis', s=50, alpha=0.7)
        plt.title('K-means Clustering (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # 3. Hierarchical clustering dendrogram
        plt.subplot(3, 3, 3)
        dendrogram(linkage_matrix, truncate_mode='level', p=10, 
                  leaf_rotation=90, leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        # 4. Hierarchical PCA plot
        plt.subplot(3, 3, 4)
        scatter = plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                            c=self.hierarchical_labels, cmap='tab10', s=50, alpha=0.7)
        plt.title('Hierarchical Clustering (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # 5. Hierarchical t-SNE plot
        plt.subplot(3, 3, 5)
        scatter = plt.scatter(self.tsne_2d[:, 0], self.tsne_2d[:, 1], 
                            c=self.hierarchical_labels, cmap='tab10', s=50, alpha=0.7)
        plt.title('Hierarchical Clustering (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # 6. HDBSCAN PCA plot
        plt.subplot(3, 3, 6)
        scatter = plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                            c=self.hdbscan_labels, cmap='viridis', s=50, alpha=0.7)
        plt.title('HDBSCAN Clustering (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # 7. HDBSCAN t-SNE plot
        plt.subplot(3, 3, 7)
        scatter = plt.scatter(self.tsne_2d[:, 0], self.tsne_2d[:, 1], 
                            c=self.hdbscan_labels, cmap='viridis', s=50, alpha=0.7)
        plt.title('HDBSCAN Clustering (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # 8. Final personas PCA plot
        plt.subplot(3, 3, 8)
        persona_numeric = [int(p.split()[-1]) for p in self.persona_mapping.values()]
        scatter = plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                            c=persona_numeric, cmap='Set1', s=50, alpha=0.7)
        plt.title('Final Persona Assignment (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Persona')
        
        # 9. Final personas t-SNE plot
        plt.subplot(3, 3, 9)
        scatter = plt.scatter(self.tsne_2d[:, 0], self.tsne_2d[:, 1], 
                            c=persona_numeric, cmap='Set1', s=50, alpha=0.7)
        plt.title('Final Persona Assignment (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter, label='Persona')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self, comparison_df):
        """Save all results to files."""
        print(f"\nSaving results to {self.output_dir}/...")
        
        # Save persona mapping as JSON
        with open(f'{self.output_dir}/persona_mapping.json', 'w') as f:
            json.dump(self.persona_mapping, f, indent=2)
        
        # Save detailed comparison as CSV
        comparison_df.to_csv(f'{self.output_dir}/clustering_comparison.csv', index=False)
        
        # Save summary statistics
        summary = {
            'timestamp': self.timestamp,
            'total_users': len(self.user_ids),
            'embedding_dimension': self.embeddings.shape[1],
            'persona_distribution': {
                persona: list(self.persona_mapping.values()).count(persona) 
                for persona in set(self.persona_mapping.values())
            }
        }
        
        with open(f'{self.output_dir}/analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Results saved successfully!")
        print(f"- Persona mapping: {self.output_dir}/persona_mapping.json")
        print(f"- Clustering comparison: {self.output_dir}/clustering_comparison.csv")
        print(f"- Visualizations: {self.output_dir}/clustering_analysis.png")
        print(f"- Summary: {self.output_dir}/analysis_summary.json")
    
    def run_complete_analysis(self):
        """Run the complete clustering analysis pipeline."""
        print("Starting User Preference Clustering Analysis")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: K-means clustering
        self.perform_kmeans_clustering()
        
        # Step 3: Hierarchical clustering
        linkage_matrix, threshold, hierarchical_outliers = self.perform_hierarchical_clustering()
        
        # Step 4: HDBSCAN clustering
        hdbscan_outliers = self.perform_hdbscan_clustering()
        
        # Step 5: Compare and assign personas
        comparison_df = self.compare_and_assign_personas(hierarchical_outliers, hdbscan_outliers)
        
        # Step 6: Create visualizations
        self.create_visualizations(linkage_matrix)
        
        # Step 7: Save results
        self.save_results(comparison_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

def main():
    """Main function to run the clustering analysis."""
    # Replace with your JSON file path
    json_file_path = "user_preference_embeddings.json"

    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found!")
        print("Please ensure your JSON file exists and has the following format:")
        print('[{"id": "user1", "embedding": [0.1, 0.2, ...]}, ...]')
        return
    
    # Run analysis
    analyzer = UserPreferenceClustering(json_file_path)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()