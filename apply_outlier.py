import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('ignore_dir/transposed_GDS3257_add_target.csv', index_col=0)
df.dropna(axis=1, inplace=True)
X = df.drop(columns=['target'])
y = df['target'].map({'tumor': 1, 'normal': 0})

# 2. Split Data
X_train_full, X_test_full, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Row Outlier Removal
X_train_full = X_train_full.loc[:, X_train_full.var() > X_train_full.var().quantile(0.25)]
iso = IsolationForest(contamination=0.05, random_state=42) 
outlier_preds = iso.fit_predict(X_train_full)
X_train_cleaned = X_train_full[outlier_preds == 1]
y_train_cleaned = y_train_full[outlier_preds == 1]

# 4. Feature Quality Filtering (CV Quantile)
cv_scores = X_train_cleaned.std() / (X_train_cleaned.mean())
cv_threshold = cv_scores.quantile(0.95)
reliable_features = cv_scores[cv_scores < cv_threshold].index
X_train_reliable = X_train_cleaned[reliable_features]

# 5. Feature Selection (Top 400)
train_variances = X_train_reliable.var().sort_values(ascending=False)
top_400_features = train_variances.head(12).index

X_train_final = X_train_reliable[top_400_features]
X_test_final = X_test_full[top_400_features]

# 6. Train & Compare Linear Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVC (Linear Kernel)": SVC(kernel='linear', random_state=42),
    "Perceptron (Single Node)": Perceptron(max_iter=1000, random_state=42)
}

print(f"Final features selected: {X_train_final.shape[1]}")
print("-" * 30)

for name, model in models.items():
    model.fit(X_train_final, y_train_cleaned)
    y_pred = model.predict(X_test_final)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))