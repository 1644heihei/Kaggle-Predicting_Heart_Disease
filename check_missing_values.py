import pandas as pd
import os

# データのパス
train_path = "data/train.csv"
test_path = "data/test.csv"


def check_missing_values(file_path, data_name):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} が見つかりません。")
        return

    print(f"Loading {data_name} data from {file_path}...")
    df = pd.read_csv(file_path)

    # 欠損値の確認
    missing_values = df.isnull().sum()
    missing_count = missing_values[missing_values > 0]

    print(f"--- {data_name} Data Missing Values ---")
    if missing_count.empty:
        print("欠損値はありません (No missing values found).")
    else:
        print(f"欠損値があるカラム数: {len(missing_count)}")
        print("\n欠損値の数:")
        print(missing_count)

        print("\n欠損値の割合 (%):")
        print((missing_count / len(df)) * 100)
    print("-" * 30 + "\n")


if __name__ == "__main__":
    check_missing_values(train_path, "Train")
    check_missing_values(test_path, "Test")
