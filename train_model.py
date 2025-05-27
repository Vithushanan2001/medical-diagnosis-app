import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("Training.csv")  # Make sure this file exists locally

# Prepare features and labels
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model + feature names (as a tuple)
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ model.pkl saved successfully")
