#This code adds colour to the dendrogram and lists genotypes that are in the same cluster
# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.patches as mpatches

# ===============================
# 2. Load Data
# ===============================
file_path = "31_Full Data Table_PAU 96 Hexaploid_combined ABC - Copy.csv"
df = pd.read_csv(file_path)

# Remove metadata columns
metadata_cols = ["Index", "Name", "Address", "Chr", "Position"]
geno_df = df.drop(columns=metadata_cols)

print("Original markers:", geno_df.shape[0])
print("Genotypes:", geno_df.shape[1])

# ===============================
# 3. SNP Filtering
# ===============================
missing_rate = geno_df.isna().mean(axis=1)

def calculate_maf(row):
    counts = row.value_counts(dropna=True)
    if len(counts) < 2:
        return 0
    freqs = counts / counts.sum()
    return freqs.min()

maf = geno_df.apply(calculate_maf, axis=1)

maf_threshold = 0.05
missing_threshold = 0.1

filtered_df = geno_df[(maf >= maf_threshold) & (missing_rate <= missing_threshold)]
print("Markers after filtering:", filtered_df.shape[0])

# ===============================
# 4. Encode Alleles Numerically
# ===============================
encoded_df = filtered_df.apply(lambda col: pd.factorize(col)[0])
encoded_df = encoded_df.T  # rows = genotypes
encoded_df = encoded_df.fillna(encoded_df.mean())

# ===============================
# 5. PCA Analysis
# ===============================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Genotype": encoded_df.index
})

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

# ===============================
# 6. K-means Clustering (Clusters start from 1)
# ===============================
k = 3  # adjust based on dataset
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data) + 1  # add 1 so clusters = 1,2,3
pca_df["Cluster"] = clusters

# Export genotype-cluster assignments
cluster_df = pca_df[["Genotype", "Cluster"]]
cluster_df.to_csv("genotypes_with_clusters.csv", index=False)
print("Cluster assignments saved to 'genotypes_with_clusters.csv'")

# ===============================
# 7. PCA Plot (colored by cluster only)
# ===============================
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    s=80,
    legend='full'
)

plt.title("PCA of Genotypes (Clusters 1,2,3)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
plt.tight_layout()
plt.savefig("PCA_plot_clusters.png", dpi=300)
plt.show()

# ===============================
# 8. Genetic Distance Matrix
# ===============================
distance_matrix = pairwise_distances(scaled_data, metric="euclidean")
distance_df = pd.DataFrame(distance_matrix, index=encoded_df.index, columns=encoded_df.index)
distance_df.to_csv("genetic_distance_matrix.csv")

# ===============================
# 9. Clustered Heatmap with Dendrogram (with color-coded groups)
# ===============================
# Perform hierarchical clustering on distance matrix
Z = linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)], method='average')

# Cut dendrogram to create clusters (adjust percentile threshold as needed)
distance_threshold = np.percentile(Z[:, 2], 70)  # Cut at 70th percentile
dendro_clusters = fcluster(Z, distance_threshold, criterion='distance')

# Create color mapping for clusters
unique_clusters = np.unique(dendro_clusters)
n_clusters = len(unique_clusters)

# Use HSV colormap for many distinct colors
colors = plt.cm.hsv(np.linspace(0, 0.95, n_clusters))
cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

# Create color labels for rows and columns based on dendrogram groups
row_colors = [cluster_colors[dendro_clusters[list(encoded_df.index).index(genotype)]]
              for genotype in distance_df.index]
col_colors = row_colors  # Same coloring for columns

# Create clustermap with color-coded dendrograms
g = sns.clustermap(
    distance_df,
    cmap="viridis",
    figsize=(14,14),
    method="average",  # hierarchical clustering
    row_cluster=True,
    col_cluster=True,
    xticklabels=True,
    yticklabels=True,
    row_colors=row_colors,
    col_colors=col_colors,
    cbar_kws={"label": "Genetic Distance"}
)

plt.suptitle(f"Genetic Distance Heatmap with Dendrogram ({n_clusters} Groups)",
             fontsize=16, y=0.995)

# Add legend for cluster colors
handles = [mpatches.Patch(facecolor=cluster_colors[cluster],
                         label=f'Group {cluster}')
          for cluster in sorted(unique_clusters)]
g.ax_heatmap.legend(handles=handles, bbox_to_anchor=(1.25, 1),
                   loc='upper left', ncol=2, fontsize=7, title='Dendrogram Groups')

plt.savefig("genetic_distance_heatmap_dendrogram.png", dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# 10. Extract Dendrogram Groupings and Export
# ===============================
# Dendrogram groups already calculated in section 9, now export them

# Create dataframe with dendrogram cluster assignments
dendro_df = pd.DataFrame({
    "Genotype": encoded_df.index,
    "Dendrogram_Group": dendro_clusters
})

# Group genotypes by dendrogram clusters
grouped_by_dendro = dendro_df.groupby("Dendrogram_Group")["Genotype"].apply(list).to_dict()

# Export in wide format: each group as a column
max_genotypes = max(len(genotypes) for genotypes in grouped_by_dendro.values())
cluster_dict = {}

for group_num in sorted(grouped_by_dendro.keys()):
    genotypes = grouped_by_dendro[group_num]
    # Pad with empty strings to match max length
    padded = genotypes + [""] * (max_genotypes - len(genotypes))
    cluster_dict[f"Group_{group_num}"] = padded

dendro_grouped_df = pd.DataFrame(cluster_dict)
dendro_grouped_df.to_csv("genotypes_grouped_by_dendrogram.csv", index=False)
print("Genotypes grouped by dendrogram saved to 'genotypes_grouped_by_dendrogram.csv'")

# Print summary
print("\n--- Dendrogram Groupings (from Genetic Distance Heatmap) ---")
for group_num in sorted(grouped_by_dendro.keys()):
    genotypes = grouped_by_dendro[group_num]
    print(f"Group {group_num}: {len(genotypes)} genotypes")
    print(f"  {', '.join(genotypes)}\n")

print("\nAnalysis complete.")
print("Outputs generated:")
print("- PCA_plot_clusters.png")
print("- genetic_distance_matrix.csv")
print("- genetic_distance_heatmap_dendrogram.png")
print("- genotypes_with_clusters.csv")
print("- genotypes_grouped_by_dendrogram.csv")
