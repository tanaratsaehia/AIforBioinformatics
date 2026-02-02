import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    # ทุกโมเดลในกลุ่มนี้ใช้ข้อมูล Scaled
    model.fit(X_train_final, y_train_cleaned)
    y_pred = model.predict(X_test_final)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


# 1. เตรียมข้อมูล Importance (อิงจากโมเดลที่ถูก Train แล้วในขั้นตอนก่อนหน้า)
# ตรวจสอบว่าได้ทำ Scaling ข้อมูลก่อน Train หรือไม่ เพื่อให้ค่า Importance เปรียบเทียบกันได้ภายในโมเดล
models_to_plot = {
    "Logistic Regression": models["Logistic Regression"],
    "SVC (Linear Kernel)": models["SVC (Linear Kernel)"],
    "Perceptron (Single Node)": models["Perceptron (Single Node)"]
}

# 2. สร้าง Subplots 1 แถว 3 คอลัมน์
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
colors = ['#4C72B0', '#55A868', '#C44E52'] # สีแยกแต่ละโมเดล

for i, (name, model) in enumerate(models_to_plot.items()):
    # ดึงค่า Coefficients
    coef_values = abs(model.coef_[0])
    features = top_400_features # รายชื่อยีน 12 ตัว
    
    # สร้าง DataFrame เฉพาะของโมเดลนั้นๆ และเรียงลำดับจากมากไปน้อย
    df_temp = pd.DataFrame({'Gene': features, 'Importance': coef_values})
    df_temp = df_temp.sort_values(by='Importance', ascending=False) # เรียงให้บวกอยู่บน
    
    # พล็อตกราฟแท่งแนวนอน
    sns.barplot(
        x='Importance', 
        y='Gene', 
        data=df_temp, 
        ax=axes[i], 
        color=colors[i]
    )
    
    # ตกแต่งกราฟ
    axes[i].set_title(f'Importance: {name}', fontsize=15, fontweight='bold')
    axes[i].axvline(x=0, color='black', linestyle='-', linewidth=1) # เส้นกลางที่ 0
    axes[i].set_xlabel('Coefficient Value', fontsize=12)
    axes[i].set_ylabel('') # ลบชื่อแกน Y ออกในรูปที่ 2 และ 3 เพื่อความสวยงาม
    # if i > 0:
    #     axes[i].set_yticklabels([]) # ซ่อนชื่อยีนในกราฟที่ 2 และ 3 เพราะเหมือนกัน

plt.tight_layout()
plt.show()