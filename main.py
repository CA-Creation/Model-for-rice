# -------------------------------
# Rice Crop Disease Risk Clustering
# -------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load CSV
# -------------------------------
data = pd.read_csv('rice_data.csv', encoding='latin1')

# Clean column names (remove non-breaking spaces)
data.columns = [col.replace('\xa0', ' ').strip() for col in data.columns]

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Display basic info
print(data.head())
print(data.info())

# -------------------------------
# Step 2: Define features
# -------------------------------
feature_columns = [
    'Average Temperature', 'Maximum temperature', 'Minimum temperature',
    'Average relative humidity', 'Total rainfall', 'Average wind speed', 
    'Average visibility', 'Average pH', 'Maximam pH', 'Minimum pH',
    'Average Sanility', 'Maximam Sanility', 'Minimum Sanility'
]

X = data[feature_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: K-Means clustering
# -------------------------------
# We'll create 3 clusters: Low, Medium, High risk
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data['risk_cluster'] = clusters

# Optional: Map clusters to risk labels based on cluster center sum
cluster_centers = kmeans.cluster_centers_.sum(axis=1)
cluster_order = cluster_centers.argsort()
risk_map = {cluster_order[0]: 'Low', cluster_order[1]: 'Medium', cluster_order[2]: 'High'}
data['risk_label'] = data['risk_cluster'].map(risk_map)

print(data[['Date','risk_cluster','risk_label']].head(10))

# -------------------------------
# Step 4: Feature importance using Random Forest
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, data['risk_cluster'])

importances = rf.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='viridis')
plt.title("Feature Importance for Rice Disease Risk Clusters")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 5: Visualize clusters
# -------------------------------
plt.figure(figsize=(10,6))
plt.scatter(data['Average Temperature'], data['Total rainfall'], c=data['risk_cluster'], cmap='viridis')
plt.xlabel('Average Temperature')
plt.ylabel('Total Rainfall')
plt.title('Rice Disease Risk Clusters')
plt.colorbar(label='Cluster')
plt.show()
