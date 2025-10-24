import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib

data = pd.read_csv("features_train.csv")
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f" Model accuracy: {accuracy * 100:.2f}%")
print(" Confusion matrix:")
print(cm)
print(f"  F1 (F1-score): {f1:.2f}")

joblib.dump(model, "hand_gesture_model.joblib")
print("Model saved: hand_gesture_model.joblib")
