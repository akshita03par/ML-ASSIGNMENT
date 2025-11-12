# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------
# Load the dataset
# ------------------------------------------------------

# Read dataset
db = pd.read_csv("dataset.csv")

print("\n✅ Dataset Loaded Successfully!\n")

# Display first 10 rows
pd.set_option('display.max_columns', None)
print("🔹 First 10 Rows:")
print(db.head(10))

# Shape (rows, columns)
print("\n📏 Shape of Dataset:", db.shape)

# ------------------------------------------------------
# Basic Info and Summary Statistics
# ------------------------------------------------------
print("\n📄 Dataset Information:")
print(db.info())

print("\n📊 Summary Statistics:")
print(db.describe())


# ------------------------------------------------------
# Initial Data Cleanup
# ------------------------------------------------------
# Drop the ID column as it's not a feature
if 'ID' in db.columns:
    db = db.drop('ID', axis=1)
    print("✅ 'ID' column dropped.")
else:
    print("⚠️ 'ID' column not found, assuming already dropped.")

# ------------------------------------------------------
# Check for missing values
# ------------------------------------------------------
print("\n🔍 Checking for Missing Values:")
print(db.isnull().sum())
# Note: Your dataset info shows no null values, so this should all be 0.
# We are not replacing 0s as they are valid values in your columns 
# (e.g., Num_children=0, Unemployed=0).

# ------------------------------------------------------
# Separate Column Types
# ------------------------------------------------------
# Identify numerical, categorical, and binary/flag columns
binary_flag_cols = ['Gender', 'Own_car', 'Own_property', 'Work_phone', 'Phone', 'Email', 'Unemployed']
continuous_numerical_cols = ['Num_children', 'Num_family', 'Account_length', 'Total_income', 'Age', 'Years_employed']
categorical_cols = ['Income_type', 'Education_type', 'Family_status', 'Housing_type', 'Occupation_type']

# Get all numerical columns (for correlation heatmap and some plots)
all_numerical_cols = binary_flag_cols + continuous_numerical_cols + ['Target']

# ------------------------------------------------------
# Univariate Analysis – Histograms (Numerical)
# ------------------------------------------------------
print("\n📊 Plotting Numerical Feature Distributions...")
db[all_numerical_cols].hist(figsize=(14, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Numerical Feature Distributions", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ------------------------------------------------------
# Univariate Analysis – Count Plots (Categorical)
# ------------------------------------------------------
print("\n📊 Plotting Categorical Feature Distributions...")
plt.figure(figsize=(14, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 2, i + 1)
    sns.countplot(y=db[col], order=db[col].value_counts().index, palette='Blues_r')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('Count')
    plt.ylabel('')
plt.suptitle("Categorical Feature Distributions", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------
print("\n🔥 Correlation Heatmap:")
plt.figure(figsize=(12, 10))
# db.corr() will automatically select only numerical columns
sns.heatmap(db.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()

# ------------------------------------------------------
# Boxplots to detect outliers (Continuous Numerical Features)
# ------------------------------------------------------
print("\n📦 Visualizing Outliers in Continuous Features:")
plt.figure(figsize=(12, 8))
db[continuous_numerical_cols].boxplot()
plt.title("Boxplot of Continuous Numerical Features", fontsize=14)
plt.xticks(rotation=45)
plt.show()

# ------------------------------------------------------
# Key Feature Insights (Template)
# ------------------------------------------------------
print("\n💡 Key Insights (Update these after reviewing your plots!):")
print("""
1️⃣ Check the 'Correlation Heatmap' for features highly correlated with 'Target'.
   (e.g., 'Account_length', 'Age', 'Years_employed').
2️⃣ Look at the 'Target' distribution in the final summary below to check for class imbalance.
   (There are likely far more 0s than 1s).
3️⃣ 'Total_income' and 'Years_employed' distributions might be skewed.
   Check their histograms and boxplots for outliers.
4️⃣ All categorical features (like 'Occupation_type', 'Income_type') will need to be 
   encoded (e.g., using One-Hot Encoding) before you can feed them to a model.
5️⃣ Scaling has been applied to normalize the continuous numerical features.
""")

# ------------------------------------------------------
# Feature Scaling (Standardization)
# ------------------------------------------------------
X = db.drop('Target', axis=1)
y = db['Target']

# We only scale the continuous numerical columns
# Binary/flag columns (0/1) don't need scaling.
# Categorical columns (text) can't be scaled directly.
print(f"\n📏 Scaling the following columns: {continuous_numerical_cols}")

scaler = StandardScaler()

# Create a copy of X to hold the scaled data
X_scaled = X.copy()

# Fit and transform ONLY the continuous numerical columns
X_scaled[continuous_numerical_cols] = scaler.fit_transform(X[continuous_numerical_cols])

# Display the head of the scaled data (showing numerical cols are now scaled)
print("\n📏 Scaled Data (first 5 rows):")
print(X_scaled.head())

# ------------------------------------------------------
# Final Dataset Summary
# ------------------------------------------------------
print("\n✅ Final Dataset Ready for Model Building!")
print(f"Total Records: {db.shape[0]}")
print(f"Features (before encoding): {db.shape[1] - 1}")

# Check for imbalance in the Target variable
target_counts = db['Target'].value_counts()
print(f"Applicants Rejected (Target=0): {target_counts.get(0, 0)}")
print(f"Applicants Approved (Target=1): {target_counts.get(1, 0)}")


# ------------------------------------------------------
# 📘 STEP 3: PRE-MODELING (ENCODING)
# ------------------------------------------------------
#
# ⚠️ CRITICAL FIX: You must convert text columns to numbers before training.
# We use One-Hot Encoding (pd.get_dummies) to convert categorical columns
# (like 'Income_type', 'Occupation_type') into numerical (0/1) columns.
#
print("\n🔄 Converting categorical text columns to numbers using One-Hot Encoding...")
X_encoded = pd.get_dummies(X_scaled)

print(f"Data shape before encoding: {X_scaled.shape}")
print(f"Data shape after encoding:  {X_encoded.shape}")
print("✅ Encoding complete! All data is now numerical.")

# ------------------------------------------------------
# 📘 STEP 4: MODEL BUILDING & COMPARISON
# ------------------------------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data into training and testing sets (80/20)
# *** We use the new X_encoded DataFrame ***
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("\n✅ Data split completed!")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ------------------------------------------------------
# Model 1: Logistic Regression
# ------------------------------------------------------
# Set class_weight='balanced' to help with the imbalanced 'Target' column
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print("\n📊 Logistic Regression Accuracy:", round(acc_lr * 100, 2), "%")
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# ------------------------------------------------------
# Model 2: Random Forest Classifier
# ------------------------------------------------------
# Set class_weight='balanced' here as well
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Accuracy:", round(acc_rf * 100, 2), "%")
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# ------------------------------------------------------
# Cross-Validation (5-fold)
# ------------------------------------------------------
print("\n🔁 Performing 5-Fold Cross Validation:")

# ⚠️ FIX: Use X_encoded (fully numerical) not X_scaled (has text)
cv_lr = cross_val_score(log_reg, X_encoded, y, cv=5, scoring='accuracy')
cv_rf = cross_val_score(rf, X_encoded, y, cv=5, scoring='accuracy')

print("Average CV Accuracy - Logistic Regression:", round(cv_lr.mean() * 100, 2), "%")
print("Average CV Accuracy - Random Forest:", round(cv_rf.mean() * 100, 2), "%")

# ------------------------------------------------------
# Hyperparameter Tuning (GridSearchCV for Random Forest)
# ------------------------------------------------------
# (This part was already correct, assuming X_train was made from X_encoded)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                   param_grid,
                   cv=3,
                   scoring='accuracy',
                   n_jobs=-1) # n_jobs=-1 uses all available CPU cores

print("\n⏳ Starting GridSearchCV (this may take a minute)...")
grid.fit(X_train, y_train)
print("✅ GridSearchCV complete!")

print("\n🔍 Best Parameters from GridSearchCV:")
print(grid.best_params_)

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)

print("\n✅ Tuned Random Forest Accuracy:", round(acc_best * 100, 2), "%")
print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best))

# ------------------------------------------------------
# Final Model Performance Summary
# ------------------------------------------------------
print("\n📈 Final Model Performance Comparison:")
print(f"Logistic Regression: {round(acc_lr * 100, 2)}%")
print(f"Random Forest (Base): {round(acc_rf * 100, 2)}%")
print(f"Random Forest (Tuned): {round(acc_best * 100, 2)}%")

# ------------------------------------------------------
# 📘 STEP 5: MODEL EVALUATION (METRICS & VISUALS)
# ------------------------------------------------------
from sklearn.metrics import (
    f1_score, roc_curve, roc_auc_score
)

# Define our labels
plot_labels = ['Rejected (0)', 'Approved (1)']

# --- Logistic Regression Evaluation ---
print("\n📊 Logistic Regression Evaluation:")
f1_lr = f1_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=plot_labels, yticklabels=plot_labels)
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ROC Curve for Logistic Regression
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance') # Dashed line for random chance
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.grid(True) # Add a grid for easier reading
plt.show()

# --- Tuned Random Forest Evaluation ---
print("\n🌲 Tuned Random Forest Evaluation:")
f1_rf = f1_score(y_test, y_pred_best)
cm_rf = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=plot_labels, yticklabels=plot_labels)
plt.title("Confusion Matrix – Tuned Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ROC Curve for Tuned Random Forest
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

plt.figure(figsize=(7, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Tuned Random Forest (AUC = {auc_rf:.3f})", color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Tuned Random Forest")
plt.legend()
plt.grid(True)
plt.show()

# --- Model Comparison ---
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Tuned Random Forest'],
    'Accuracy': [round(acc_lr * 100, 2), round(acc_best * 100, 2)],
    'F1-Score': [round(f1_lr, 3), round(f1_rf, 3)],
    'AUC': [round(auc_lr, 3), round(auc_rf, 3)]
})
print("\n📈 Model Comparison Summary:")
print(comparison.to_string()) # .to_string() ensures it prints nicely

# Bar Chart Comparison
plt.figure(figsize=(8, 6))
# Create the bar plot
ax = sns.barplot(x='Model', y='Accuracy', data=comparison, palette=['#4c72b0', '#55a868'])
plt.title("Model Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")

# Add the percentage labels on top of the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=12)
plt.ylim(0, 100) # Set y-axis to go from 0 to 100
plt.show()