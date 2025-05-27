import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/rajathkmp/ML-Disease-Prediction/main/Training.csv"
df = pd.read_csv("Training.csv")



# Prepare data
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model + features
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ Model trained and saved as model.pkl")

