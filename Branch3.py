import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

class DataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_split(self, file_path, target_column, preprocessor_path='models/preprocessor.pkl'):
        try:
            print("🔄 Loading dataset...")
            df = pd.read_csv(file_path)

            print("🔄 Loading preprocessor...")
            preprocessor = joblib.load(preprocessor_path)

            print("🔄 Preparing data...")
            X = preprocessor.transform(df.drop(columns=[target_column]))
            y = df[target_column]

            print("🔄 Splitting data...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            print(f"✅ Training set: {self.X_train.shape}")
            print(f"✅ Test set: {self.X_test.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            print(f"❌ Error: {e}")

    def save_splits(self, prefix='data_splits'):
        try:
            print("💾 Saving data splits...")

            os.makedirs("models", exist_ok=True)

            joblib.dump(self.X_train, f'models/{prefix}_X_train.pkl')
            joblib.dump(self.X_test, f'models/{prefix}_X_test.pkl')
            joblib.dump(self.y_train, f'models/{prefix}_y_train.pkl')
            joblib.dump(self.y_test, f'models/{prefix}_y_test.pkl')

            print("✅ Data saved successfully!")

        except Exception as e:
            print(f"❌ Error saving data: {e}")


# ✅ Main execution block
if __name__ == "__main__":
    print("🚀 Program Started")

    loader = DataLoader()

    # 🔥 IMPORTANT: Change these values according to your project
    file_path = "data.csv"          # 👈 your dataset file
    target_column = "charges"        # 👈 your target column name

    # Run pipeline
    result = loader.load_and_split(file_path, target_column)

    if result:
        loader.save_splits()

    print("🏁 Program Finished")