import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("credit_card.csv")

print("Dataset loaded successfully!")
print("Number of customers:", len(df))
print("Number of features:", len(df.columns))


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

# Remove customer ID because it is not useful for clustering
X = df.drop("CUST_ID", axis=1)

# Handle missing values
X = X.fillna(X.median())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData preprocessing completed!")


# =========================================================
# 3. ELBOW METHOD
# =========================================================

inertia = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# =========================================================
# 4. CREATE FINAL K-MEANS MODEL
# =========================================================

# Based on the Elbow Method
optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Assign cluster to every customer
df["Cluster"] = kmeans.fit_predict(X_scaled)


# =========================================================
# 5. DISPLAY NUMBER OF CUSTOMERS IN EACH CLUSTER
# =========================================================

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCustomers in each cluster:")
print(cluster_counts)


# =========================================================
# 6. CLUSTER ANALYSIS
# =========================================================

important_features = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT"
]

cluster_summary = df.groupby("Cluster")[important_features].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))


# =========================================================
# 7. GIVE MEANINGFUL NAMES TO CLUSTERS
# =========================================================

cluster_names = {
    0: "Low-Activity Customers",
    1: "High-Value Customers",
    2: "High-Risk Cash-Advance Customers",
    3: "Regular Responsible Customers"
}

df["Customer_Segment"] = df["Cluster"].map(cluster_names)


# =========================================================
# 8. DISPLAY CUSTOMER SEGMENT INFORMATION
# =========================================================

print("\nCustomer Segment Summary:")

for cluster in sorted(df["Cluster"].unique()):

    count = len(df[df["Cluster"] == cluster])

    print(
        f"Cluster {cluster}: "
        f"{cluster_names[cluster]} - "
        f"{count} customers"
    )


# =========================================================
# 9. SAVE CLUSTERED DATASET
# =========================================================

df.to_csv("customer_segments.csv", index=False)

print("\nClustered dataset saved as: customer_segments.csv")


# =========================================================
# 10. VISUALIZATION 1 - ELBOW METHOD
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid(True)


# =========================================================
# 11. VISUALIZATION 2 - BALANCE VS PURCHASES
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["BALANCE"],
    df["PURCHASES"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.5
)

plt.xlabel("Balance")
plt.ylabel("Purchases")
plt.title("Customer Segments: Balance vs Purchases")

plt.colorbar(label="Cluster")

plt.grid(True)


# =========================================================
# 12. VISUALIZATION 3 - AVERAGE PURCHASES
# =========================================================

avg_purchases = df.groupby("Cluster")["PURCHASES"].mean()

plt.figure(figsize=(8, 5))

avg_purchases.plot(
    kind="bar"
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Purchases")
plt.title("Average Purchases by Customer Cluster")

plt.xticks(rotation=0)

plt.grid(axis="y")


# =========================================================
# 13. VISUALIZATION 4 - CLUSTER COMPARISON
# =========================================================

cluster_means = df.groupby("Cluster")[important_features].mean()

cluster_means.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Value")
plt.title("Customer Cluster Comparison")

plt.xticks(rotation=0)
plt.legend(title="Features")

plt.tight_layout()


# =========================================================
# 14. SHOW ALL GRAPHS
# =========================================================

plt.show()


# =========================================================
# 15. FINAL MESSAGE
# =========================================================

print("\n======================================")
print(" CREDIT CARD CUSTOMER SEGMENTATION")
print("======================================")

print("\nOptimal number of clusters:", optimal_k)

print("\nCustomer Segments:")

for cluster, name in cluster_names.items():

    count = len(df[df["Cluster"] == cluster])

    print(
        f"Cluster {cluster} -> {name} "
        f"({count} customers)"
    )

print("\nProject completed successfully!")
print("Output file: customer_segments.csv")