import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Libraries imported successfully!")


# Step 3: Load Dataset
file_path = "dataset/Sample - Superstore.csv"

df = pd.read_csv(file_path, encoding="latin1")

print("\nDataset loaded successfully!")

# Step 4: Display First 5 Records

print("\nFirst 5 Rows:")
print(df.head())

# Step 5: Dataset Shape

print("\nDataset Shape:")
print(df.shape)

 # Display Column Names
 
print("\nColumn Names:")
print(df.columns.tolist())

 # Dataset Information

print("\nDataset Information:")
df.info()

#  Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Step 9: Statistical Summary
 
print("\nStatistical Summary:")
print(df.describe())

print("\nSTEP 1 COMPLETED SUCCESSFULLY!")

# DATA CLEANING

# 1. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# 5. Display updated information
print("\nUpdated Dataset Information:")
df.info()

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")

#EDA

# STEP 3: CUSTOMER PURCHASE ANALYSIS

# 1. Total number of customers
print("\nTotal Customers:")
print(df["Customer ID"].nunique())

# 2. Total sales
print("\nTotal Sales:")
print(df["Sales"].sum())

# 3. Total profit
print("\nTotal Profit:")
print(df["Profit"].sum())

# 4. Top 10 customers by sales
customer_sales = df.groupby("Customer ID")["Sales"].sum()

print("\nTop 10 Customers:")
print(customer_sales.sort_values(ascending=False).head(10))

# 5. Sales by customer segment
print("\nSales by Segment:")
print(df.groupby("Segment")["Sales"].sum())

print("\nCUSTOMER ANALYSIS COMPLETED!")


# STEP 4: DATA VISUALIZATION

# 1. Sales by Customer Segment

segment_sales = df.groupby("Segment")["Sales"].sum()

plt.figure(figsize=(10, 6))
segment_sales.plot(kind="bar", color="skyblue")

plt.title("Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.tight_layout()


plt.show()

# 2. Top 10 Customers by Sales

top_customers = (
    df.groupby("Customer ID")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
top_customers.plot(kind="bar", color="orange")

plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer ID")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

print("VISUALIZATION COMPLETED!")

# STEP 5: RFM ANALYSIS

# Set reference date (one day after the last order)
reference_date = df["Order Date"].max() + pd.Timedelta(days=1)

# Calculate RFM values for each customer
rfm = df.groupby("Customer ID").agg({

    "Order Date": lambda x: (reference_date - x.max()).days,

    "Order ID": "nunique",

    "Sales": "sum"

})

# Rename columns
rfm.rename(columns={

    "Order Date": "Recency",
    "Order ID": "Frequency",
    "Sales": "Monetary"

}, inplace=True)

# Display RFM results
print("\nRFM ANALYSIS RESULTS:")
print(rfm.head(10))

# Display summary
print("\nRFM Summary:")
print(rfm.describe())

# Save RFM results
rfm.to_csv("outputs/customer_rfm.csv")
print("\nRFM ANALYSIS COMPLETED SUCCESSFULLY!")


# STEP 6: K-MEANS CLUSTERING

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select RFM features
rfm_features = rfm[["Recency", "Frequency", "Monetary"]]

# Scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Apply K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Assign cluster labels
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Display results
print("\nCustomer Cluster Results:")
print(rfm.head(10))

# Cluster summary
cluster_summary = rfm.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean().round(2)

print("\nCluster Summary:")
print(cluster_summary)

# Save results
rfm.to_csv("outputs/customer_segments.csv")

print("\nK-MEANS CLUSTERING COMPLETED!")

# STEP 7: CUSTOMER CLUSTER VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns

# Plot customers by Recency and Monetary value
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segments: Recency vs Monetary")
plt.xlabel("Recency (Days)")
plt.ylabel("Total Sales")
plt.legend(title="Cluster")
plt.show()

# Plot average sales for each cluster
cluster_summary["Monetary"].plot(
    kind="bar",
    figsize=(8, 5),
    title="Average Monetary Value by Cluster"
)

plt.xlabel("Cluster")
plt.ylabel("Average Sales")
plt.show()

# STEP 8: INTERPRET CUSTOMER SEGMENTS

print("\nCUSTOMER SEGMENT INTERPRETATION:")

for cluster in sorted(rfm["Cluster"].unique()):
    data = rfm[rfm["Cluster"] == cluster]

    print(f"\nCluster {cluster}")
    print("Number of Customers:", len(data))
    print("Average Recency:", round(data["Recency"].mean(), 2))
    print("Average Frequency:", round(data["Frequency"].mean(), 2))
    print("Average Monetary:", round(data["Monetary"].mean(), 2))

    # STEP 9: SAVE FINAL REPORT


# CLUSTER SUMMARY

cluster_summary = rfm.groupby("Cluster").agg(

    Customer_Count=("Recency", "size"),

    Avg_Recency=("Recency", "mean"),

    Avg_Frequency=("Frequency", "mean"),

    Avg_Monetary=("Monetary", "mean")

).round(2)

print("\nCUSTOMER CLUSTER SUMMARY:")
print(cluster_summary)

# Save cluster summary
cluster_summary.to_csv("outputs/cluster_summary.csv")

print("\nCLUSTER SUMMARY COMPLETED SUCCESSFULLY!")
