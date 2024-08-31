import random

random.seed(14872673)

random_integer = random.randint(1, 100)
print(random_integer)

import pandas as pd
from scipy.stats import skew, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
import numpy as np

#Question 1
# Load the dataset
file_path = '/Users/default/Desktop/spotify52kData.csv'
data = pd.read_csv(file_path)

# List of features to check
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Remove rows with missing values
data_cleaned = data.dropna()

# Apply transformations to appropriate features
data_transformed = data_cleaned.copy()
data_transformed['duration'] = np.log1p(data_transformed['duration'])
data_transformed['danceability'] = data_transformed['danceability']  # No transformation
data_transformed['energy'] = np.sqrt(data_transformed['energy'].max() + 1 - data_transformed['energy'])  # Square root transformation for negative skewness
data_transformed['loudness'] = np.sqrt(-data_transformed['loudness'] + data_transformed['loudness'].max() + 1)
data_transformed['speechiness'] = np.log1p(data_transformed['speechiness'])
data_transformed['acousticness'] = np.log1p(data_transformed['acousticness'])
data_transformed['instrumentalness'] = np.log1p(data_transformed['instrumentalness'])
data_transformed['liveness'] = np.log1p(data_transformed['liveness'])
data_transformed['valence'] = data_transformed['valence']  # No transformation
data_transformed['tempo'] = np.sqrt(data_transformed['tempo'])

# Check skewness after transformation
transformed_skewness = data_transformed[features].apply(skew)
print("Skewness after transformation:")
print(transformed_skewness)

# Plot histograms of the transformed features
plt.figure(figsize=(20, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 5, i + 1)
    sns.histplot(data_transformed[feature], kde=True, bins=30)
    plt.title(f'{feature}\nSkewness: {transformed_skewness[i]:.2f}')
plt.tight_layout()
plt.show()

#Question 2 Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(data['duration'], data['popularity'], alpha=0.5)
plt.title('Relationship Between Song Duration and Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity (Score from 0 to 100)')
plt.grid(True)
plt.show()

# Create a scatter plot with a logarithmic scale on the duration axis
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='duration', y='popularity', alpha=0.5, edgecolor=None)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.title('Relationship between Song Duration (log scale) and Popularity')
plt.xlabel('Duration (log scale, milliseconds)')
plt.ylabel('Popularity')
plt.show()

# Calculate and print the Pearson correlation coefficient
correlation = data['duration'].corr(data['popularity'])
print("Pearson correlation coefficient:", correlation)

#Question 3
data_cleaned = data.dropna(subset=['explicit', 'popularity'])

# Check the distribution of 'popularity' to decide on transformation
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['popularity'], kde=True, bins=30)
plt.title('Distribution of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.show()

# Transform 'popularity' if needed (here, it's reasonably normal, so no transformation)
data_cleaned['popularity_transformed'] = data_cleaned['popularity']

# Identify and remove outliers using the IQR method
Q1 = data_cleaned[['popularity_transformed']].quantile(0.25)
Q3 = data_cleaned[['popularity_transformed']].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data_cleaned[['popularity_transformed']] < (Q1 - 1.5 * IQR)) | 
                     (data_cleaned[['popularity_transformed']] > (Q3 + 1.5 * IQR)))
data_no_outliers = data_cleaned[~outlier_condition.any(axis=1)]

# Separate the data into explicit and non-explicit groups
explicit_songs = data_no_outliers[data_no_outliers['explicit'] == 1]['popularity_transformed']
non_explicit_songs = data_no_outliers[data_no_outliers['explicit'] == 0]['popularity_transformed']

# Perform an independent t-test to compare the means of the two groups
t_stat, p_value = ttest_ind(explicit_songs, non_explicit_songs)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Visualize the data using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='explicit', y='popularity_transformed', data=data_no_outliers)
plt.xticks([0, 1], ['Non-Explicit', 'Explicit'])
plt.xlabel('Song Explicitness')
plt.ylabel('Transformed Popularity')
plt.title('Popularity of Explicit vs. Non-Explicit Songs')
plt.show()

#Question 4
# Remove rows with missing values in 'mode' and 'popularity'
data_cleaned = data.dropna(subset=['mode', 'popularity'])

# Check the distribution of 'popularity' to decide on transformation
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['popularity'], kde=True, bins=30)
plt.title('Distribution of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.show()

# Transform 'popularity' if needed (here, it's reasonably normal, so no transformation)
data_cleaned['popularity_transformed'] = data_cleaned['popularity']

# Identify and remove outliers using the IQR method
Q1 = data_cleaned[['popularity_transformed']].quantile(0.25)
Q3 = data_cleaned[['popularity_transformed']].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data_cleaned[['popularity_transformed']] < (Q1 - 1.5 * IQR)) | 
                     (data_cleaned[['popularity_transformed']] > (Q3 + 1.5 * IQR)))
data_no_outliers = data_cleaned[~outlier_condition.any(axis=1)]

# Separate the data into major and minor key groups
major_key_songs = data_no_outliers[data_no_outliers['mode'] == 1]['popularity_transformed']
minor_key_songs = data_no_outliers[data_no_outliers['mode'] == 0]['popularity_transformed']

# Perform an independent t-test to compare the means of the two groups
t_stat, p_value = ttest_ind(major_key_songs, minor_key_songs)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Visualize the data using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='mode', y='popularity_transformed', data=data_no_outliers)
plt.xticks([0, 1], ['Minor Key', 'Major Key'])
plt.xlabel('Song Key')
plt.ylabel('Transformed Popularity')
plt.title('Popularity of Songs in Major vs. Minor Key')
plt.show()

#Question 5
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loudness', y='energy', data=data)
plt.title('Scatter Plot of Energy vs. Loudness')
plt.xlabel('Loudness (dB)')
plt.ylabel('Energy')
plt.show()

# Calculate Pearson correlation
correlation, p_value = pearsonr(data['loudness'], data['energy'])
print(f"Pearson Correlation Coefficient: {correlation:.3f}, P-value: {p_value:.4f}")

# Regression analysis
X = sm.add_constant(data['loudness'])  # adding a constant
model = sm.OLS(data['energy'], X).fit()
print(model.summary())

import statsmodels.api as sm
X = sm.add_constant(data['loudness'])  # adding a constant
Y = data['energy']

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Get the confidence intervals of the coefficients
confidence_intervals = model.conf_int(alpha=0.05)  # 95% CI
print("Confidence Intervals for Regression Coefficients:")
print(confidence_intervals)


#Question 6
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Define the features to be analyzed
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Initialize a dictionary to store R-squared values for each feature
results = {}

# Perform linear regression for each feature
for feature in features:
    X = data[feature]
    y = data['popularity']
    X = sm.add_constant(X)  # Add constant term for intercept
    model = sm.OLS(y, X).fit()
    results[feature] = model.rsquared

# Find the feature with the highest R-squared value
best_feature = max(results, key=results.get)
best_r_squared = results[best_feature]

print(f"Best predictor of popularity: {best_feature}")
print(f"R-squared value for the best predictor: {best_r_squared}")

# Fit the model for the best predictor
X_best = sm.add_constant(data[best_feature])
model_best = sm.OLS(data['popularity'], X_best).fit()
summary_best = model_best.summary()

# Print the summary of the best model
print(summary_best)

#Question 7
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load and preprocess the data
file_path = '/Users/default/Desktop/spotify52kData.csv'
data = pd.read_csv(file_path)

# Handle missing data
data.dropna(inplace=True)

# Features and target variable
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']
X = data[features]
y = data['popularity']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Build and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'R²: {r2}')
print(f'RMSE: {rmse}')

# Step 4: Compare with the best model from question 6
best_model_r2 = 0.021104664943677243
best_model_rmse = 21.45660764283447

improvement_r2 = r2 - best_model_r2
improvement_rmse = best_model_rmse - rmse

print(f'Improvement in R²: {improvement_r2}')
print(f'Improvement in RMSE: {improvement_rmse}')

#Question 8
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# List of features to check
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Remove rows with missing values
data_cleaned = data.dropna(subset=features)

# Apply transformations to appropriate features (same as question 1)
data_transformed = data_cleaned.copy()
data_transformed['duration'] = np.log1p(data_transformed['duration'])
data_transformed['danceability'] = data_transformed['danceability']  # No transformation
data_transformed['energy'] = np.log1p(data_transformed['energy'])  # Log transformation
data_transformed['loudness'] = np.sqrt(-data_transformed['loudness'] + data_transformed['loudness'].max() + 1)
data_transformed['speechiness'] = np.log1p(data_transformed['speechiness'])
data_transformed['acousticness'] = np.log1p(data_transformed['acousticness'])
data_transformed['instrumentalness'] = np.log1p(data_transformed['instrumentalness'])
data_transformed['liveness'] = np.log1p(data_transformed['liveness'])
data_transformed['valence'] = data_transformed['valence']  # No transformation
data_transformed['tempo'] = np.sqrt(data_transformed['tempo'])

# Remove outliers using the IQR method
Q1 = data_transformed[features].quantile(0.25)
Q3 = data_transformed[features].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data_transformed[features] < (Q1 - 1.5 * IQR)) | 
                     (data_transformed[features] > (Q3 + 1.5 * IQR)))
data_no_outliers = data_transformed[~outlier_condition.any(axis=1)]

# Perform PCA
pca = PCA()
pca.fit(data_no_outliers[features])

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Determine the number of meaningful principal components
num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1  # Components that explain at least 95% of the variance

print(f"Number of meaningful principal components: {num_components}")
print(f"Proportion of variance explained by these components: {cumulative_explained_variance[num_components-1]}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

#Question 9
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# List of features to check
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Remove rows with missing values
data_cleaned = data.dropna(subset=features + ['mode'])

# Remove outliers using the IQR method
Q1 = data_cleaned[features].quantile(0.25)
Q3 = data_cleaned[features].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data_cleaned[features] < (Q1 - 1.5 * IQR)) | 
                     (data_cleaned[features] > (Q3 + 1.5 * IQR)))
data_no_outliers = data_cleaned[~outlier_condition.any(axis=1)]

# Prepare data for classification
X = data_no_outliers[features]
y = data_no_outliers['mode']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform Random Forest classification
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Predict the mode on the test set
y_pred = rf_clf.predict(X_test)
y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC Score: {roc_auc}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Random Forest (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

#Question 10
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# List of features to check
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Remove rows with missing values
data_cleaned = data.dropna(subset=features + ['track_genre'])

# Convert genre label to binary (classical or not)
data_cleaned['is_classical'] = data_cleaned['track_genre'].apply(lambda x: 1 if 'classical' in x.lower() else 0)

# Apply transformations to appropriate features (same as question 1)
data_transformed = data_cleaned.copy()
data_transformed['duration'] = np.log1p(data_transformed['duration'])
data_transformed['danceability'] = data_transformed['danceability']  # No transformation
data_transformed['energy'] = np.log1p(data_transformed['energy'])  # Log transformation
data_transformed['loudness'] = np.sqrt(-data_transformed['loudness'] + data_transformed['loudness'].max() + 1)
data_transformed['speechiness'] = np.log1p(data_transformed['speechiness'])
data_transformed['acousticness'] = np.log1p(data_transformed['acousticness'])
data_transformed['instrumentalness'] = np.log1p(data_transformed['instrumentalness'])
data_transformed['liveness'] = np.log1p(data_transformed['liveness'])
data_transformed['valence'] = data_transformed['valence']  # No transformation
data_transformed['tempo'] = np.sqrt(data_transformed['tempo'])

# Remove outliers using the IQR method
Q1 = data_transformed[features].quantile(0.25)
Q3 = data_transformed[features].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data_transformed[features] < (Q1 - 1.5 * IQR)) | 
                     (data_transformed[features] > (Q3 + 1.5 * IQR)))
data_no_outliers = data_transformed[~outlier_condition.any(axis=1)]

# Separate the data into features and target variable
X_duration = data_no_outliers[['duration']]
y = data_no_outliers['is_classical']

# Perform PCA to extract principal components
pca = PCA(n_components=4)
X_pca = pca.fit_transform(data_no_outliers[features])

# Split the data into training and testing sets for both models
X_train_duration, X_test_duration, y_train, y_test = train_test_split(X_duration, y, test_size=0.3, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Logistic Regression with Duration
log_reg_duration = LogisticRegression()
log_reg_duration.fit(X_train_duration, y_train)
y_pred_duration = log_reg_duration.predict(X_test_duration)
y_pred_prob_duration = log_reg_duration.predict_proba(X_test_duration)[:, 1]

# Logistic Regression with Principal Components
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_pca, y_train)
y_pred_pca = log_reg_pca.predict(X_test_pca)
y_pred_prob_pca = log_reg_pca.predict_proba(X_test_pca)[:, 1]

# Evaluate the models
accuracy_duration = accuracy_score(y_test, y_pred_duration)
roc_auc_duration = roc_auc_score(y_test, y_pred_prob_duration)

accuracy_pca = accuracy_score(y_test, y_pred_pca)
roc_auc_pca = roc_auc_score(y_test, y_pred_prob_pca)

print(f"Duration Model - Accuracy: {accuracy_duration}, ROC AUC: {roc_auc_duration}")
print(f"PCA Model - Accuracy: {accuracy_pca}, ROC AUC: {roc_auc_pca}")

# Plot ROC curves
fpr_duration, tpr_duration, _ = roc_curve(y_test, y_pred_prob_duration)
fpr_pca, tpr_pca, _ = roc_curve(y_test, y_pred_prob_pca)

plt.figure(figsize=(10, 6))
plt.plot(fpr_duration, tpr_duration, label=f'Duration (area = {roc_auc_duration:.2f})')
plt.plot(fpr_pca, tpr_pca, label=f'PCA (area = {roc_auc_pca:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()