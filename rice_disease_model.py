# ==========================
# Rice Disease Prediction + Future Factor Prediction
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Rice_Diseases_Factors_550.csv")

# -----------------------------
# 2. Handle missing values & types
# -----------------------------
# Fill missing environmental factors with median
feature_cols = [
    "Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Soil Moisture (%)",
    "pH Level", "Nitrogen Content (mg/kg)", "Potassium Content (mg/kg)", "Wind Speed (m/s)"
]
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing Disease Type with 'Unknown' and ensure string
df['Disease Type'] = df['Disease Type'].fillna('Unknown').astype(str)

# -----------------------------
# 3. Targets
# -----------------------------
# Binary target: Disease Presence
y_presence = df["Disease Presence"].map({"No": 0, "Yes": 1})

# Multiclass target: Disease Type
le_type = LabelEncoder()
y_type_enc = le_type.fit_transform(df['Disease Type'])

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X = df[feature_cols]
X_train, X_test, yP_train, yP_test, yT_train, yT_test = train_test_split(
    X, y_presence, y_type_enc, test_size=0.2, random_state=42, stratify=y_type_enc
)

# -----------------------------
# 5. Random Forest Models
# -----------------------------
# Binary model
rf_bin = RandomForestClassifier(n_estimators=200, random_state=42)
rf_bin.fit(X_train, yP_train)
yP_pred = rf_bin.predict(X_test)
yP_proba = rf_bin.predict_proba(X_test)[:,1]

# Multiclass model
rf_multi = RandomForestClassifier(n_estimators=300, random_state=42)
rf_multi.fit(X_train, yT_train)
yT_pred = rf_multi.predict(X_test)

# Metrics
print("=== Binary Classification: Disease Presence ===")
print(f"Accuracy: {accuracy_score(yP_test, yP_pred):.4f}")
print(f"Precision: {precision_score(yP_test, yP_pred):.4f}")
print(f"Recall: {recall_score(yP_test, yP_pred):.4f}")
print(f"F1-score: {f1_score(yP_test, yP_pred):.4f}")

print("\n=== Multiclass Classification: Disease Type ===")
print(f"Accuracy: {accuracy_score(yT_test, yT_pred):.4f}")
print(classification_report(yT_test, yT_pred, target_names=le_type.classes_))

# -----------------------------
# 6. Feature Importance
# -----------------------------
def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_features = np.array(feature_names)[idx]
    top_importances = importances[idx]
    plt.figure()
    plt.bar(range(len(top_features)), top_importances)
    plt.xticks(range(len(top_features)), top_features, rotation=45)
    plt.title(title)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
    return pd.DataFrame({"Feature": top_features, "Importance": top_importances})

fi_bin = plot_feature_importance(rf_bin, feature_cols, "Feature Importance - Binary RF")
fi_multi = plot_feature_importance(rf_multi, feature_cols, "Feature Importance - Multiclass RF")

# -----------------------------
# 7. LSTM for future environmental factors
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

n_steps = 5
def create_sequences(data, n_steps):
    X_seq, y_seq = [], []
    for i in range(len(data)-n_steps):
        X_seq.append(data[i:i+n_steps])
        y_seq.append(data[i+n_steps])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, n_steps)
split_idx = int(len(X_seq)*0.8)
X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

# LSTM model
lstm = Sequential()
lstm.add(LSTM(64, activation='relu', input_shape=(n_steps, X_scaled.shape[1])))
lstm.add(Dense(X_scaled.shape[1]))
lstm.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq),
         epochs=100, batch_size=16, callbacks=[es], verbose=0)

# Predict future factors (next day)
last_seq = X_scaled[-n_steps:]
pred_future_scaled = lstm.predict(last_seq.reshape(1, n_steps, X_scaled.shape[1]))
pred_future = scaler.inverse_transform(pred_future_scaled)
pred_future_df = pd.DataFrame(pred_future, columns=feature_cols)
print("\nPredicted Future Environmental Factors:")
print(pred_future_df)

# -----------------------------
# 8. Predict Disease from predicted factors
# -----------------------------
pred_bin_future = rf_bin.predict(pred_future_df)
pred_bin_future_proba = rf_bin.predict_proba(pred_future_df)[:,1]
pred_multi_future = rf_multi.predict(pred_future_df)
pred_multi_future_label = le_type.inverse_transform(pred_multi_future)

print("\nPredicted Disease Presence (future):", "Yes" if pred_bin_future[0]==1 else "No", 
      f"(Probability={pred_bin_future_proba[0]:.2f})")
print("Predicted Disease Type (future):", pred_multi_future_label[0])

# -----------------------------
# 9. Save models and scaler
# -----------------------------
joblib.dump(rf_bin, "rice_binary_rf_model.pkl")
joblib.dump(rf_multi, "rice_multiclass_rf_model.pkl")
joblib.dump(le_type, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(lstm, "lstm_future_model.pkl")

print("\nAll models and scaler saved successfully.")
