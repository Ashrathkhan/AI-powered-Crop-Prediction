# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle   # ✅ added

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("crop.csv")

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
data.fillna(data.mean(numeric_only=True), inplace=True)
data.drop_duplicates(inplace=True)

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

scaler = StandardScaler()
features = ["N","P","K","temperature","humidity","ph","rainfall"]
data[features] = scaler.fit_transform(data[features])

# -----------------------------
# 3. Split Data
# -----------------------------
X = data[features]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ -----------------------------
# 5. Save Model (IMPORTANT)
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model, scaler, and encoder saved successfully!")

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -----------------------------
# 7. Get Input from User
# -----------------------------
print("\nEnter the following details:")

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH value: "))
rainfall = float(input("Rainfall (mm): "))

new_data = [[N, P, K, temperature, humidity, ph, rainfall]]

# Apply scaling
new_data = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data)
crop_name = le.inverse_transform(prediction)

print("\n🌱 Recommended Crop:", crop_name[0])