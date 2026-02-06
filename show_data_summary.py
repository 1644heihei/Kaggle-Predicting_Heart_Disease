import pandas as pd

# 表示設定: 全ての列を表示するようにする
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def show_summary(path, name):
    try:
        df = pd.read_csv(path)
        print(f"\n{'='*20} {name} Data Summary {'='*20}")
        print(f"File: {path}")
        print(f"Shape: {df.shape} (Rows, Columns)")

        print(f"\n--- Columns List ---")
        print(df.columns.tolist())

        print(f"\n--- First 5 Rows ---")
        print(df.head())

        print(f"\n--- Basic Info ---")
        df.info()

        print(f"\n--- Statistical Summary ---")
        print(df.describe())

    except FileNotFoundError:
        print(f"Error: {path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    show_summary("data/train.csv", "Train")
    show_summary("data/test.csv", "Test")
    show_summary("data/sample_submission.csv", "Sample Submission")
