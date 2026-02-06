import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# データの読み込み
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
submission_sample = pd.read_csv("data/sample_submission.csv")

# 前処理
print("Preprocessing...")

# ターゲット変数の変換 (Presence -> 1, Absence -> 0)
target_mapping = {"Presence": 1, "Absence": 0}
train_df["Heart Disease"] = train_df["Heart Disease"].map(target_mapping)

# 特徴量とターゲットの分離
# 'id' は予測に使わないので削除
X = train_df.drop(["id", "Heart Disease"], axis=1)
y = train_df["Heart Disease"]

# テストデータも同様に 'id' を削除 (予測用に特徴量だけにする)
X_test_sub = test_df.drop(["id"], axis=1)

# 学習用と検証用に分割 (8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築 (ランダムフォレスト)
print("Training model (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 検証データでの評価
print("Evaluating model...")
y_pred_val = model.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))

# テストデータに対する予測
print("Predicting on test set...")
predictions = model.predict(X_test_sub)

# 提出用ファイルの作成
submission = pd.DataFrame({"id": test_df["id"], "Heart Disease": predictions})

# 保存
output_path = "submission_baseline.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file saved to {output_path}")
