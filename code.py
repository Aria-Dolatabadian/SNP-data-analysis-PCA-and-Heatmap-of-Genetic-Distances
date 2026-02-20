# # ===============================
# # 1. Import Libraries
# # ===============================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import StandardScaler
# import scipy.cluster.hierarchy as sch
#
# # ===============================
# # 2. Load Data
# # ===============================
# file_path = "61_Full Data Table_Top alleles_Harsh 4_default.csv"
# df = pd.read_csv(file_path)
#
# # Remove metadata columns
# metadata_cols = ["Index", "Name", "Address", "Chr", "Position", "GenTrain Score"]
# geno_df = df.drop(columns=metadata_cols)
#
# print("Original markers:", geno_df.shape[0])
# print("Genotypes:", geno_df.shape[1])
#
# # ===============================
# # 3. SNP Filtering
# # ===============================
# missing_rate = geno_df.isna().mean(axis=1)
#
# def calculate_maf(row):
#     counts = row.value_counts(dropna=True)
#     if len(counts) < 2:
#         return 0
#     freqs = counts / counts.sum()
#     return freqs.min()
#
# maf = geno_df.apply(calculate_maf, axis=1)
#
# maf_threshold = 0.05
# missing_threshold = 0.1
#
# filtered_df = geno_df[(maf >= maf_threshold) & (missing_rate <= missing_threshold)]
# print("Markers after filtering:", filtered_df.shape[0])
#
# # ===============================
# # 4. Encode Alleles Numerically
# # ===============================
# encoded_df = filtered_df.apply(lambda col: pd.factorize(col)[0])
# encoded_df = encoded_df.T  # rows = genotypes
# encoded_df = encoded_df.fillna(encoded_df.mean())
#
# # ===============================
# # 5. PCA Analysis
# # ===============================
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(encoded_df)
#
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(scaled_data)
#
# pca_df = pd.DataFrame({
#     "PC1": pca_result[:, 0],
#     "PC2": pca_result[:, 1],
#     "Genotype": encoded_df.index
# })
#
# print("Explained variance ratio:")
# print(pca.explained_variance_ratio_)
#
# # ===============================
# # 6. K-means Clustering (Clusters start from 1)
# # ===============================
# k = 3  # adjust based on dataset
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(scaled_data) + 1  # add 1 so clusters = 1,2,3
# pca_df["Cluster"] = clusters
#
# # Export genotype-cluster assignments
# cluster_df = pca_df[["Genotype", "Cluster"]]
# cluster_df.to_csv("genotypes_with_clusters.csv", index=False)
# print("Cluster assignments saved to 'genotypes_with_clusters.csv'")
#
# # ===============================
# # 7. PCA Plot with Cluster Labels
# # ===============================
# plt.figure(figsize=(8,6))
# sns.scatterplot(
#     data=pca_df,
#     x="PC1",
#     y="PC2",
#     hue="Cluster",
#     palette="Set2",
#     s=80,
#     legend='full'
# )
#
# # Annotate cluster numbers
# for i, row in pca_df.iterrows():
#     plt.text(row["PC1"], row["PC2"], str(row["Cluster"]),
#              fontsize=9, alpha=0.8, horizontalalignment='center')
#
# plt.title("PCA of Genotypes (Clusters 1,2,3)")
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
# plt.tight_layout()
# plt.savefig("PCA_plot_clusters.png", dpi=300)
# plt.show()
#
# # ===============================
# # 8. Genetic Distance Matrix
# # ===============================
# distance_matrix = pairwise_distances(scaled_data, metric="euclidean")
# distance_df = pd.DataFrame(distance_matrix, index=encoded_df.index, columns=encoded_df.index)
# distance_df.to_csv("genetic_distance_matrix.csv")
#
# # ===============================
# # 9. Clustered Heatmap with Dendrogram
# # ===============================
# sns.clustermap(
#     distance_df,
#     cmap="viridis",
#     figsize=(12,12),
#     method="average",  # hierarchical clustering
#     row_cluster=True,
#     col_cluster=True,
#     xticklabels=True,
#     yticklabels=True
# )
# plt.title("Genetic Distance Heatmap with Dendrogram", pad=100)
# plt.savefig("genetic_distance_heatmap_dendrogram.png", dpi=300)
# plt.show()
#
# print("\nAnalysis complete.")
# print("Outputs generated:")
# print("- PCA_plot_clusters.png")
# print("- genetic_distance_matrix.csv")
# print("- genetic_distance_heatmap_dendrogram.png")
# print("- genotypes_with_clusters.csv")


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

# ===============================
# 2. Load Data
# ===============================
file_path = "61_Full Data Table_Top alleles_Harsh 4_default.csv"
df = pd.read_csv(file_path)

# Remove metadata columns
metadata_cols = ["Index", "Name", "Address", "Chr", "Position", "GenTrain Score"]
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
# 9. Clustered Heatmap with Dendrogram
# ===============================
sns.clustermap(
    distance_df,
    cmap="viridis",
    figsize=(12,12),
    method="average",  # hierarchical clustering
    row_cluster=True,
    col_cluster=True,
    xticklabels=True,
    yticklabels=True
)
plt.title("Genetic Distance Heatmap with Dendrogram", pad=100)
plt.savefig("genetic_distance_heatmap_dendrogram.png", dpi=300)
plt.show()

print("\nAnalysis complete.")
print("Outputs generated:")
print("- PCA_plot_clusters.png")
print("- genetic_distance_matrix.csv")
print("- genetic_distance_heatmap_dendrogram.png")
print("- genotypes_with_clusters.csv")
