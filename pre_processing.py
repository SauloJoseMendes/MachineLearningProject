import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from Classes.DataReader import DataReader
# Load the dataset
dataset = DataReader()

# Apply PCA
pca = PCA()
pca.fit(dataset.X)

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Get the feature importance (principal components)
loadings = pca.components_

# Create a DataFrame showing feature contributions to the PCs
feature_importance = pd.DataFrame(loadings.T, columns=[f"PC{i+1}" for i in range(loadings.shape[0])], index=dataset.X_not_normalized.columns)

# Display results
print("Explained variance by each principal component:")
print(explained_variance)

print("\nFeature importance (loadings):")
print(feature_importance)

# Optional: Identify features with the least contribution to the first few PCs
least_important_features = feature_importance.abs().sum(axis=1).sort_values()
print("\nLeast important features:")
print(least_important_features)

feature = "HEART RATE"
correlation_matrix = dataset.numeric_data.corr()
print("\n\nCORRELATIONS FOR FEATURE " + str(feature))
print(correlation_matrix[feature].sort_values(ascending=False))

gender_vs_target = pd.crosstab(dataset.numeric_data[feature], dataset.numeric_data['TARGET'])
chi2, p, dof, expected = chi2_contingency(gender_vs_target)
print("\n\np-value:", p)