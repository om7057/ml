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


## 8 

