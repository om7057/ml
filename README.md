# ml


## 1

import numpy as np
from scipy.stats import skewnorm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

f_heights = skewnorm.rvs(a=3, loc=150, scale=5, size=10000)
m_heights = np.random.normal(loc=166, scale=5.5, size=10000)

f_labels = np.zeros(10000)
m_labels = np.ones(10000)

heights = np.concatenate([f_heights, m_heights])
labels = np.concatenate([f_labels, m_labels])

X_train_all, X_test, y_train_all, y_test = train_test_split(
    heights, labels, test_size=0.2, stratify=labels, random_state=42
)

shuffle_idx = np.random.permutation(len(X_train_all))
X_train_all = X_train_all[shuffle_idx]
y_train_all = y_train_all[shuffle_idx]

X_train = X_train_all[:6000]
y_train = y_train_all[:6000]
X_val = X_train_all[-2000:]
y_val = y_train_all[-2000:]

def trains(X, y, step):
    edges = np.arange(X.min(), X.max() + step, step)
    ids = np.digitize(X, edges)
    bin_map = {}
    for i in np.unique(ids):
        group = y[ids == i]
        if len(group) > 0:
            bin_map[i] = int(np.mean(group) >= 0.5)
    return edges, bin_map

def predicts(X, edges, bin_map):
    ids = np.digitize(X, edges)
    return np.array([bin_map.get(i, 1) for i in ids])

steps = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6]
scores = []

for s in steps:
    bins, labels_map = trains(X_train, y_train, s)
    preds = predicts(X_val, bins, labels_map)
    acc = np.mean(preds == y_val)
    scores.append(acc)
    print(f"Bin size: {s:.6f}, Val Acc: {acc:.4f}")

best_idx = np.argmax(scores)
best_step = steps[best_idx]
print(f"\nBest bin size: {best_step}, Val Acc: {scores[best_idx]:.4f}")

bins, labels_map = trains(X_train_all, y_train_all, best_step)
test_preds = predicts(X_test, bins, labels_map)
test_acc = np.mean(test_preds == y_test)
print(f"\nTest Acc with bin size {best_step}: {test_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(steps, scores, marker='o')
plt.xscale('log')
plt.xlabel("Bin Size")
plt.ylabel("Val Accuracy")
plt.title("Accuracy vs Bin Size")
plt.grid(True)
plt.tight_layout()
plt.show()



## 2


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

B = np.random.normal(loc=5, scale=2, size=10000)
I = stats.powerlaw.rvs(a=0.3, size=10000)
H = np.random.geometric(p=0.005, size=10000)

B_zscore = (B - np.mean(B)) / np.std(B)
I_zscore = (I - np.mean(I)) / np.std(I)
H_zscore = (H - np.mean(H)) / np.std(H)

def robust_normalization(x):
    median = np.median(x)
    iqr = stats.iqr(x)
    return (x - median) / iqr

B_robust = robust_normalization(B)
I_robust = robust_normalization(I)
H_robust = robust_normalization(H)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
sns.boxplot(data=[B, I, H], ax=axes[0])
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['B', 'I', 'H'])
axes[0].set_title("Original Variables")

sns.boxplot(data=[B_zscore, I_zscore, H_zscore], ax=axes[1])
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['B', 'I', 'H'])
axes[1].set_title("Z-score Normalized")

sns.boxplot(data=[B_robust, I_robust, H_robust], ax=axes[2])
axes[2].set_xticks([0, 1, 2])
axes[2].set_xticklabels(['B', 'I', 'H'])
axes[2].set_title("Robust Normalized")

fig.suptitle("Box Plot Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

def plot_histogram_comparison(variable_name, original, zscore, robust):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    axes[0].hist(original, bins=50, color='blue', alpha=0.7)
    axes[0].set_title(f"{variable_name} - Original")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(zscore, bins=50, color='orange', alpha=0.7)
    axes[1].set_title(f"{variable_name} - Z-score")
    axes[1].set_xlabel("Values")
    
    axes[2].hist(robust, bins=50, color='green', alpha=0.7)
    axes[2].set_title(f"{variable_name} - Robust")
    axes[2].set_xlabel("Values")
    
    fig.suptitle(f"Histogram Comparison for {variable_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_histogram_comparison("B (Gaussian)", B, B_zscore, B_robust)
plot_histogram_comparison("I (Power-law)", I, I_zscore, I_robust)
plot_histogram_comparison("H (Geometric)", H, H_zscore, H_robust)



## 3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, chi2
from scipy.spatial.distance import mahalanobis

np.random.seed(42)

def generate_group(height_mean, height_std, bmi_loc, bmi_skew, bmi_scale, n=1000):
    heights = np.random.normal(height_mean, height_std, n)
    bmi = skewnorm.rvs(bmi_skew, bmi_loc, bmi_scale, n)
    weights = (heights / 100.0) ** 2 * bmi
    return pd.DataFrame({'height': heights, 'weight': weights})

females = generate_group(152, 2.5, 21.4, 8, 0.4)
males   = generate_group(166, 3.0, 22.0, 8, 0.5)

def outliers(df, method='euclidean', quantile=0.95):
    data = df.values
    center = np.mean(data, axis=0)

    if method == 'euclidean':
        dists = np.linalg.norm(data - center, axis=1)
        cutoff = np.percentile(dists, quantile * 100)
    elif method == 'mahalanobis':
        cov = np.cov(data, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        dists = np.array([mahalanobis(x, center, inv_cov) for x in data])
        cutoff = np.sqrt(chi2.ppf(quantile, df.shape[1]))
    else:
        raise ValueError("Only 'euclidean' and 'mahalanobis' supported.")

    return dists > cutoff, dists

fem_e_out, _ = outliers(females, 'euclidean')
fem_m_out, _ = outliers(females, 'mahalanobis')
male_e_out, _ = outliers(males, 'euclidean')
male_m_out, _ = outliers(males, 'mahalanobis')

def visualize_outliers(df, eucl_out, maha_out, label):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(df['height'], df['weight'], c='lightgray')
    plt.scatter(df.loc[eucl_out, 'height'], df.loc[eucl_out, 'weight'], c='red', label='Euclidean Outliers')
    plt.title(f'{label} - Euclidean')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(df['height'], df['weight'], c='lightgray')
    plt.scatter(df.loc[maha_out, 'height'], df.loc[maha_out, 'weight'], c='green', label='Mahalanobis Outliers')
    plt.title(f'{label} - Mahalanobis')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_outliers(females, fem_e_out, fem_m_out, 'Females')
visualize_outliers(males, male_e_out, male_m_out, 'Males')

print("Female Group:")
print(f"  Euclidean outliers: {fem_e_out.sum()}")
print(f"  Mahalanobis outliers: {fem_m_out.sum()}")
print(f"  Shared outliers: {(fem_e_out & fem_m_out).sum()}")

print("\nMale Group:")
print(f"  Euclidean outliers: {male_e_out.sum()}")
print(f"  Mahalanobis outliers: {male_m_out.sum()}")
print(f"  Shared outliers: {(male_e_out & male_m_out).sum()}")

print("\nCorrelations:")
print(f"  Female height-weight correlation: {females.corr().loc['height', 'weight']:.3f}")
print(f"  Male height-weight correlation:   {males.corr().loc['height', 'weight']:.3f}")


## 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

np.random.seed(42)

male_heights = np.random.normal(166, 5.5, 1000)
female_heights = np.random.normal(152, 4.5, 1000)

male_train, male_test = train_test_split(male_heights, test_size=0.2, random_state=42)
female_train, female_test = train_test_split(female_heights, test_size=0.2, random_state=42)

def classify_height(height, male_mean, male_std, female_mean, female_std):
    male_prob = (1 / (male_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((height - male_mean) / male_std) ** 2)
    female_prob = (1 / (female_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((height - female_mean) / female_std) ** 2)
    return "Male" if male_prob > female_prob else "Female"

def calculate_accuracy(true_labels, predicted_labels):
    correct = np.sum(true_labels == predicted_labels)
    return correct / len(true_labels)

def calculate_classification_error(true_labels, predicted_labels):
    return 1 - calculate_accuracy(true_labels, predicted_labels)

male_mean, male_std = np.mean(male_train), np.std(male_train)
female_mean, female_std = np.mean(female_train), np.std(female_train)

train_labels = np.array(["Male"] * 800 + ["Female"] * 800)
train_predictions = [classify_height(h, male_mean, male_std, female_mean, female_std) for h in np.concatenate((male_train, female_train))]
train_accuracy = calculate_accuracy(train_labels, train_predictions)
train_error = calculate_classification_error(train_labels, train_predictions)

test_labels = np.array(["Male"] * 200 + ["Female"] * 200)
test_predictions = [classify_height(h, male_mean, male_std, female_mean, female_std) for h in np.concatenate((male_test, female_test))]
test_accuracy = calculate_accuracy(test_labels, test_predictions)
test_error = calculate_classification_error(test_labels, test_predictions)

print(f"Original Train Accuracy: {train_accuracy:.4f}, Error: {train_error:.4f}")
print(f"Original Test Accuracy: {test_accuracy:.4f}, Error: {test_error:.4f}")

top_50_female_indices = np.argsort(female_train)[-50:]
female_train_outliers = female_train.copy()
female_train_outliers[top_50_female_indices] += 10

new_female_mean, new_female_std = np.mean(female_train_outliers), np.std(female_train_outliers)

plt.figure(figsize=(8, 5))
plt.bar(["Original", "With Outliers"], [female_std, new_female_std], color=["blue", "red"], alpha=0.7)
plt.ylabel("Standard Deviation")
plt.title("Impact of Outliers on Standard Deviation")
plt.grid(True)
plt.show()

train_predictions_outliers = [classify_height(h, male_mean, male_std, new_female_mean, new_female_std) for h in np.concatenate((male_train, female_train_outliers))]
train_accuracy_outliers = calculate_accuracy(train_labels, train_predictions_outliers)
train_error_outliers = calculate_classification_error(train_labels, train_predictions_outliers)

test_predictions_outliers = [classify_height(h, male_mean, male_std, new_female_mean, new_female_std) for h in np.concatenate((male_test, female_test))]
test_accuracy_outliers = calculate_accuracy(test_labels, test_predictions_outliers)
test_error_outliers = calculate_classification_error(test_labels, test_predictions_outliers)

print(f"Train Accuracy with Outliers: {train_accuracy_outliers:.4f}, Error: {train_error_outliers:.4f}")
print(f"Test Accuracy with Outliers: {test_accuracy_outliers:.4f}, Error: {test_error_outliers:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(["Original", "With Outliers"], [train_error, train_error_outliers], color=["blue", "red"], alpha=0.7)
plt.ylabel("Classification Error")
plt.title("Impact of Outliers on Classification Error")
plt.grid(True)
plt.show()

female_train_zscores = zscore(female_train_outliers)
female_train_cleaned = female_train_outliers[np.abs(female_train_zscores) < 3]

new_female_mean_cleaned, new_female_std_cleaned = np.mean(female_train_cleaned), np.std(female_train_cleaned)

plt.figure(figsize=(8, 5))
plt.bar(["With Outliers", "After Removal"], [new_female_std, new_female_std_cleaned], color=["red", "green"], alpha=0.7)
plt.ylabel("Standard Deviation")
plt.title("Effect of Outlier Removal on Standard Deviation")
plt.grid(True)
plt.show()

train_accuracies_trimmed, test_accuracies_trimmed = [], []
train_errors_trimmed = []
test_errors_trimmed = []
std_devs_trimmed = []
k_values = list(range(1, 26))

for k in k_values:
    lower_bound = np.percentile(female_train_outliers, k)
    upper_bound = np.percentile(female_train_outliers, 100 - k)
    female_train_trimmed = female_train_outliers[(female_train_outliers >= lower_bound) & (female_train_outliers <= upper_bound)]
    
    female_mean_trimmed, female_std_trimmed = np.mean(female_train_trimmed), np.std(female_train_trimmed)
    std_devs_trimmed.append(female_std_trimmed)
    
    train_predictions_trimmed = [classify_height(h, male_mean, male_std, female_mean_trimmed, female_std_trimmed) for h in np.concatenate((male_train, female_train_trimmed))]
    train_accuracy_trimmed = calculate_accuracy(train_labels[:len(male_train) + len(female_train_trimmed)], train_predictions_trimmed)
    train_error_trimmed = calculate_classification_error(train_labels[:len(male_train) + len(female_train_trimmed)], train_predictions_trimmed)

    test_predictions_trimmed = [classify_height(h, male_mean, male_std, female_mean_trimmed, female_std_trimmed) for h in np.concatenate((male_test, female_test))]
    test_accuracy_trimmed = calculate_accuracy(test_labels, test_predictions_trimmed)
    test_error_trimmed = calculate_classification_error(test_labels, test_predictions_trimmed)

    train_accuracies_trimmed.append(train_accuracy_trimmed)
    test_accuracies_trimmed.append(test_accuracy_trimmed)
    train_errors_trimmed.append(train_error_trimmed)
    test_errors_trimmed.append(test_error_trimmed)

plt.figure(figsize=(10, 5))
plt.plot(k_values, std_devs_trimmed, marker='o', color='purple')
plt.xlabel('Trimming Percentage (k%)')
plt.ylabel('Standard Deviation')
plt.title('Effect of Trimming on Standard Deviation')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(k_values, train_errors_trimmed, marker='o', label='Train Error', color='blue')
plt.plot(k_values, test_errors_trimmed, marker='o', label='Test Error', color='red')
plt.xlabel('Trimming Percentage (k%)')
plt.ylabel('Classification Error')
plt.title('Impact of Trimming on Classification Error')
plt.legend()
plt.grid(True)
plt.show()



## 6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("linear_regression_3.csv")

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Original rows:", len(df))
print("Rows after outlier removal:", len(df_clean))

X = df_clean.drop(columns=["y"])
y = df_clean["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
score = 10 * r2

print("\nModel Evaluation:")
print("R^2 Score:", r2)
print("R^2(10 * R^2):", score)



## 7 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

target_column = "y"
random_state = 42

df = pd.read_csv("linear_regression_3.csv")
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

vif_dropped = []
pval_dropped = []

def remove_cooks_outliers(X, y, threshold=None):
    if threshold is None:
        threshold = 4 / len(X)
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    mask = cooks_d < threshold
    return X[mask], y[mask], sum(~mask)

def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_features(X_train, X_test, y_train, threshold=10):
    X_temp = X_train.copy()
    while X_temp.shape[1] > 0:
        vif = compute_vif(X_temp)
        max_vif = vif["VIF"].max()
        print(f"max_vif : {max_vif}")
        print(f"VIF Value: {vif}")
        if max_vif > threshold:
            feature_to_drop = vif.loc[vif["VIF"].idxmax(), "Feature"]
            print(f"feature to drop: {feature_to_drop}")
            X_temp = X_temp.drop(columns=[feature_to_drop])
            X_test = X_test.drop(columns=[feature_to_drop])
            vif_dropped.append(feature_to_drop)
        else:
            break
    return X_temp, X_test

def remove_insignificant_features(X_train, X_test, y_train, p_value_threshold=0.05):
    while True:
        X_train_const = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_const).fit()
        pvalues = model.pvalues.drop("const")
        max_pval = pvalues.max()
        if max_pval > p_value_threshold:
            feature_to_drop = pvalues.idxmax()
            X_train = X_train.drop(columns=[feature_to_drop])
            X_test = X_test.drop(columns=[feature_to_drop])
            pval_dropped.append(feature_to_drop)
        else:
            return X_train, X_test, model

def score_after_outlier_removal(X, y, preds):
    residuals = y - preds
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (residuals >= lower_bound) & (residuals <= upper_bound)
    return r2_score(y[mask], preds[mask]), sum(~mask)

def plot_residuals(y_test, y_test_pred):
    residuals = y_test - y_test_pred
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=1, label="Outlier Boundaries")
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=1)
    plt.title('Histogram of Test Set Residuals with Outlier Boundaries (IQR)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()


X_train_const_raw = sm.add_constant(X_train)
raw_model = sm.OLS(y_train, X_train_const_raw).fit()
y_train_pred_raw = raw_model.predict(X_train_const_raw)
r2_raw_train = r2_score(y_train, y_train_pred_raw)
print(f"R² Score on Training Set (before cleaning): {r2_raw_train:.4f}")

X_train_cleaned, X_test_cleaned = remove_high_vif_features(X_train.copy(), X_test.copy(), y_train)
X_train_cleaned, y_train_cleaned, train_outliers = remove_cooks_outliers(X_train_cleaned, y_train)

X_train_cleaned, X_test_cleaned, _ = remove_insignificant_features(X_train_cleaned, X_test_cleaned, y_train_cleaned)

X_train_const = sm.add_constant(X_train_cleaned)
X_test_const = sm.add_constant(X_test_cleaned)
final_model = sm.OLS(y_train_cleaned, X_train_const).fit()

y_train_pred = final_model.predict(X_train_const)
y_test_pred = final_model.predict(X_test_const)

r2_train = r2_score(y_train_cleaned, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_test_cleaned, cleaned_test_outliers = score_after_outlier_removal(X_test_cleaned, y_test, y_test_pred)

print("Dropped due to VIF:", vif_dropped)
print("Dropped due to p-value:", pval_dropped)
print("Final features used:", X_train_cleaned.columns.tolist())
print(f"R² Score on Training Set (after cleaning): {r2_train:.4f}")
print(f"R² Score on Test Set (all points): {r2_test:.4f}")
print(f"R² Score on Test Set (outliers removed using IQR): {r2_test_cleaned:.4f}")
print(f"Number of Outliers Removed from Training (Cook's Distance): {train_outliers}")
print(f"Number of Outliers Removed from Test (IQR): {cleaned_test_outliers}")

plot_residuals(y_test, y_test_pred)


print(f"Final Model Summary:\n{final_model.summary()}")


## 9

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Cancer_Data.csv")

# 1. Drop the 'Id' and 'Diagnosis' columns
df_clean = df.drop(columns=['id', 'diagnosis'])
df = df.drop(columns=['Unnamed: 32'])

# 2. Standardizing the dataset
scaler = StandardScaler()
df_clean = scaler.fit_transform(df_clean)

# 2. K-Means Clustering
# a. Elbow Method (Inertia)
inertia = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_clean)
    inertia.append(kmeans.inertia_)

plt.plot(cluster_range, inertia)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# b. Silhouette Analysis
sil_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_clean)
    score = silhouette_score(df_clean, labels)
    sil_scores.append(score)

plt.plot(cluster_range, sil_scores)
plt.title("Silhouette Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Determine Optimal k based on Elbow and Silhouette score
optimal_k_elbow = 5  # Chosen based on the Elbow Method (Inertia)
optimal_k_silhouette = 2  # Chosen based on the highest Silhouette Score

# Homogeneity Score for K-Means
df['diagnosis_encoded'] = np.where(df['diagnosis'] == 'M', 1, 0)  # Encoding diagnosis: 'M' = 1, 'B' = 0
kmeans_best = KMeans(n_clusters=5, random_state=42)  # Using optimal k=5
kmeans_best.fit(df_clean)
kmeans_labels = kmeans_best.predict(df_clean)
homogeneity_kmeans = homogeneity_score(df['diagnosis_encoded'], kmeans_labels)
print(f"Homogeneity Score for K-Means: {homogeneity_kmeans:.4f}")

# 3. DBSCAN Clustering
# a. Determine appropriate eps and min_samples parameters
# k-distance plot to find eps value
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(df_clean)
distances, indices = neighbors.kneighbors(df_clean)

# Sorting the distances to the 5th nearest neighbor
k_distance = sorted(distances[:, 4])
plt.plot(k_distance)
plt.title("K-Distance Plot")
plt.xlabel("Data points sorted by distance")
plt.ylabel("Distance to 5th nearest neighbor")
plt.show()

# b. Find the optimal min_samples and eps
min_samples_range = range(3, 11)
sil_scores = []

for min_samples in min_samples_range:
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # eps=0.5 (from k-distance plot)
    labels = dbscan.fit_predict(df_clean)
    
    # Calculate Silhouette Score
    if len(set(labels)) > 1:
        score = silhouette_score(df_clean, labels)
        sil_scores.append(score)
    else:
        sil_scores.append(-1)

    print(f"min_samples= {min_samples}, No. of clusters = {len(set(labels)) - (1 if -1 in labels else 0)}, Silhouette_score = {score:.4f}")

# Homogeneity Score for DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)  # Chosen based on k-distance plot and silhouette score
dbscan_labels = dbscan.fit_predict(df_clean)
homogeneity_dbscan = homogeneity_score(df['diagnosis_encoded'], dbscan_labels)
print(f"Homogeneity Score for DBSCAN: {homogeneity_dbscan:.4f}")




## 10 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    if 'diagnosis' not in df.columns:
        raise ValueError(f"Column 'diagnosis' not found. Available columns: {df.columns.tolist()}")

    df.rename(columns={'diagnosis': 'Diagnosis'}, inplace=True)

    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    df['Diagnosis'] = LabelEncoder().fit_transform(df['Diagnosis'])  # M=1, B=0

    print("\n Data loaded and preprocessed.")
    print(f" Total samples: {len(df)}, Features: {df.shape[1] - 1}")
    print(df['Diagnosis'].value_counts().rename({1: 'Malignant', 0: 'Benign'}))

    return df

def skewed_train_test_split(df, test_size=0.2, seed=42):
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    train_df = X_train.copy()
    train_df['Diagnosis'] = y_train.values
    test_df = X_test.copy()
    test_df['Diagnosis'] = y_test.values

    malignant_rows = train_df[train_df['Diagnosis'] == 1]
    move_to_test = malignant_rows.sample(n=120, random_state=seed)
    train_df = train_df.drop(move_to_test.index)
    test_df = pd.concat([test_df, move_to_test], axis=0)

    print(f"\n After skewing: Training set = {len(train_df)}, Test set = {len(test_df)}")
    print(f"Train Diagnosis counts:\n{train_df['Diagnosis'].value_counts()}")
    print(f"Test Diagnosis counts:\n{test_df['Diagnosis'].value_counts()}")

    return train_df, test_df

def train_base_trees(train_df, n_trees=10, max_features='sqrt', seed=42):
    trees = []
    feature_names = train_df.columns.drop('Diagnosis')
    print("\n Training base decision trees...")
    for i in range(n_trees):
        sampled_df = resample(train_df, replace=True, n_samples=len(train_df), random_state=seed+i)
        X = sampled_df[feature_names]
        y = sampled_df['Diagnosis']
        tree = DecisionTreeClassifier(max_features=max_features, class_weight='balanced', random_state=seed+i)
        tree.fit(X, y)
        acc = accuracy_score(y, tree.predict(X))
        recall = recall_score(y, tree.predict(X))
        print(f"  Tree {i+1} → Train Accuracy: {acc:.4f}, Recall: {recall:.4f}")
        trees.append(tree)
    return trees

def get_aggregated_feature_importance(trees, feature_names, method='average', train_df=None):
    print(f"\n Aggregating feature importances using method: {method}")
    importances = []
    weights = []

    for tree in trees:
        if method == 'average':
            importances.append(tree.feature_importances_)
        elif method == 'weighted':
            X = train_df[feature_names]
            y = train_df['Diagnosis']
            acc = accuracy_score(y, tree.predict(X))
            importances.append(tree.feature_importances_)
            weights.append(acc)

    importances = np.array(importances)

    if method == 'average':
        return np.mean(importances, axis=0)
    elif method == 'weighted':
        weights = np.array(weights)
        return np.average(importances, axis=0, weights=weights)

def select_top_features(importance_scores, feature_names, top_k=None, threshold=0.95):
    ranked = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)
    print("\n Top feature importances:")
    for name, score in ranked[:10]:
        print(f"  {name}: {score:.4f}")
    
    if top_k:
        selected = [name for name, _ in ranked[:top_k]]
    else:
        total = sum(score for _, score in ranked)
        selected, acc = [], 0
        for name, score in ranked:
            selected.append(name)
            acc += score
            if acc >= total * threshold:
                break
    print(f"\n Selected top {len(selected)} features")
    return selected

def get_tree_outputs(trees, df, feature_cols):
    tree_preds = []
    for tree in trees:
        preds = tree.predict(df[feature_cols])
        tree_preds.append(preds)
    return np.array(tree_preds).T 

def train_meta_models(train_df, selected_features, tree_preds):
    X_train = train_df[selected_features]
    meta_input = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(tree_preds, columns=[f'Tree_{i}' for i in range(tree_preds.shape[1])])], axis=1)
    y_train = train_df['Diagnosis']

    log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_model.fit(meta_input, y_train)

    master_tree = DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=42)
    master_tree.fit(meta_input, y_train)

    print("\n Meta-models trained: Logistic Regression and Master Tree")
    return log_model, master_tree

def evaluate_models(test_df, selected_features, trees, log_model, master_tree):
    X_test = test_df[selected_features]
    y_test = test_df['Diagnosis']

    tree_preds = get_tree_outputs(trees, test_df, selected_features)
    meta_input = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(tree_preds, columns=[f'Tree_{i}' for i in range(tree_preds.shape[1])])], axis=1)

    log_preds = log_model.predict(meta_input)
    master_preds = master_tree.predict(meta_input)

    def evaluate(name, preds):
        acc = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        print(f"\n{name} Evaluation:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall (Malignant): {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        return acc, recall

    evaluate("Logistic Regression", log_preds)
    evaluate("Master Decision Tree", master_preds)

    print("\n Individual Tree Accuracies (on test set):")
    for i, tree in enumerate(trees):
        acc = accuracy_score(y_test, tree.predict(X_test))
        recall = recall_score(y_test, tree.predict(X_test))
        print(f"  Tree {i+1}: Accuracy = {acc:.4f}, Recall = {recall:.4f}")


if __name__ == "__main__":
    path = "Cancer_Data.csv"
    df = load_and_preprocess(path)

    train_df, test_df = skewed_train_test_split(df)

    base_trees = train_base_trees(train_df)

    feature_names = train_df.columns.drop('Diagnosis')
    importances = get_aggregated_feature_importance(base_trees, feature_names, method='weighted', train_df=train_df)
    selected_features = select_top_features(importances, feature_names, top_k=10)

    reduced_train_df = train_df[selected_features + ['Diagnosis']]
    final_trees = train_base_trees(reduced_train_df)

    tree_preds = get_tree_outputs(final_trees, reduced_train_df, selected_features)
    log_model, master_tree = train_meta_models(reduced_train_df, selected_features, tree_preds)

    reduced_test_df = test_df[selected_features + ['Diagnosis']]
    evaluate_models(reduced_test_df, selected_features, final_trees, log_model, master_tree)




## Q1
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data for 3 teachers
def generate_marks():
    np.random.seed(42)
    marks_A = np.random.normal(loc=8, scale=1, size=100)  # generous
    marks_B = np.random.normal(loc=5, scale=1.5, size=100)  # average
    marks_C = np.random.normal(loc=3, scale=1.2, size=100)  # strict

    # Clip marks between 0 and 10
    marks_A = np.clip(marks_A, 0, 10)
    marks_B = np.clip(marks_B, 0, 10)
    marks_C = np.clip(marks_C, 0, 10)

    return marks_A, marks_B, marks_C

# Step 2: Normalize and scale all marks
def normalize_and_transform(marks_A, marks_B, marks_C):
    all_marks = np.concatenate([marks_A, marks_B, marks_C])
    
    # Step 3: Z-score normalization
    mean = np.mean(all_marks)
    std = np.std(all_marks)
    normalized = (all_marks - mean) / std

    # Step 4: Scale to 0-10 using min-max from middle 96% (exclude outliers)
    q1 = np.percentile(normalized, 2)
    q2 = np.percentile(normalized, 98)
    scaled = 10 * (normalized - q1) / (q2 - q1)
    scaled = np.clip(scaled, 0, 10)

    return scaled

# Step 5: Run and visualize
def run():
    marks_A, marks_B, marks_C = generate_marks()
    scaled_marks = normalize_and_transform(marks_A, marks_B, marks_C)

    # Check constraint satisfaction
    percent_10 = np.sum(scaled_marks >= 10.0) / len(scaled_marks) * 100
    percent_0 = np.sum(scaled_marks <= 1.0) / len(scaled_marks) * 100

    print(f"Percent scoring 10: {percent_10:.2f}%")
    print(f"Percent scoring ≤1: {percent_0:.2f}%")

    # Optional: Plot
    plt.hist(scaled_marks, bins=20, color='skyblue', edgecolor='black')
    plt.title("Normalized & Scaled Marks Distribution")
    plt.xlabel("Scaled Marks")
    plt.ylabel("Number of Students")
    plt.show()

run()



## Q1
import numpy as np
from scipy.stats import ttest_ind

# ----- Step 1: Simulated population sampling -----

def get_sample_from_population1(n):
    # Simulated population 1: mean = 50, std = 10
    return np.random.normal(loc=50, scale=10, size=n)

def get_sample_from_population2(n):
    # Simulated population 2: mean = 55, std = 10
    return np.random.normal(loc=55, scale=10, size=n)

# ----- Step 2: Get random samples from both populations -----

def get_samples(sample_size=100):
    sample1 = get_sample_from_population1(sample_size)
    sample2 = get_sample_from_population2(sample_size)
    return sample1, sample2

# ----- Step 3: Compare the means using two-sample t-test -----

def compare_means(sample1, sample2, alpha=0.05):
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=True)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    return "Different" if p_value < alpha else "Same"

# ----- Step 4: Run everything -----

def test_population_means():
    sample1, sample2 = get_samples(sample_size=100)
    result = compare_means(sample1, sample2)
    print("Conclusion:", result)

# Run the test
test_population_means()



## Q4



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(0)

# Function to generate data with outliers
def generate_data_with_outliers():
    normal_data = np.random.normal(loc=50, scale=5, size=100)
    outliers = np.array([150, 160, 170])  # extreme values
    combined = np.concatenate([normal_data, outliers])
    return combined

# Function to generate skewed data using geometric distribution
def generate_skewed_data():
    skewed_data = np.random.geometric(p=0.1, size=100)
    return skewed_data

# Function to apply Z-score and Robust scaling
def apply_scalers(data):
    data = data.reshape(-1, 1)
    z_scaled = StandardScaler().fit_transform(data).flatten()
    r_scaled = RobustScaler().fit_transform(data).flatten()
    return z_scaled, r_scaled

# Function to plot comparison
def plot_comparison(original, z_scaled, r_scaled, title):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(original, kde=True, color='skyblue')
    plt.title(f"{title}: Original")

    plt.subplot(1, 3, 2)
    sns.histplot(z_scaled, kde=True, color='salmon')
    plt.title(f"{title}: Z-Score Scaled")

    plt.subplot(1, 3, 3)
    sns.histplot(r_scaled, kde=True, color='lightgreen')
    plt.title(f"{title}: Robust Scaled")

    plt.tight_layout()
    plt.show()

# ----------- Run for Outlier Data -----------
data_outliers = generate_data_with_outliers()
z_scaled_outliers, r_scaled_outliers = apply_scalers(data_outliers)
plot_comparison(data_outliers, z_scaled_outliers, r_scaled_outliers, "Outliers")

# ----------- Run for Skewed Data -----------
data_skewed = generate_skewed_data()
z_scaled_skewed, r_scaled_skewed = apply_scalers(data_skewed)
plot_comparison(data_skewed, z_scaled_skewed, r_scaled_skewed, "Skewed")






