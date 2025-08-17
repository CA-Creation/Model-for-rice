import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


df2022 = pd.read_excel("Climate.xlsx", sheet_name="2022")
df2023 = pd.read_excel("Climate.xlsx", sheet_name="2023")
df2024 = pd.read_excel("Climate.xlsx", sheet_name="2024")

data = pd.concat([df2022, df2023, df2024])


X = data[["Rainfall", "Temperature", "pH", "Salinity"]]
y_yield = data["Yield"]             
y_disease = data["Disease"]          


X_train, X_test, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.3, random_state=42)
_, _, y_train_d, y_test_d = train_test_split(X, y_disease, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train_y)
y_pred_y = reg_model.predict(X_test)
print("Yield Prediction RMSE:", mean_squared_error(y_test_y, y_pred_y, squared=False))
print("Yield Prediction R2:", r2_score(y_test_y, y_pred_y))


clf_model = RandomForestClassifier()
clf_model.fit(X_train, y_train_d)
y_pred_d = clf_model.predict(X_test)
print("Disease Prediction Accuracy:", accuracy_score(y_test_d, y_pred_d))
print(classification_report(y_test_d, y_pred_d))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------- Load Dataset -----------------
df2022 = pd.read_excel("climate.xlsx", sheet_name="2022")
df2023 = pd.read_excel("climate.xlsx", sheet_name="2023")
df2024 = pd.read_excel("climate.xlsx", sheet_name="2024")

data = pd.concat([df2022, df2023, df2024])

# Ensure we have a Date or Month column
# If you have "Year" and "Month" columns, combine them:
if "Year" in data.columns and "Month" in data.columns:
    data["Date"] = pd.to_datetime(data[["Year", "Month"]].assign(DAY=1))
elif "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"])
else:
    raise ValueError("Dataset must contain either (Year, Month) or Date column!")

data = data.sort_values("Date")

# ----------------- Prepare Features -----------------
features = ["Rainfall", "Temperature", "pH", "Salinity"]
target = "Yield"

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features + [target]])

# Convert back into DataFrame for clarity
scaled_df = pd.DataFrame(scaled, columns=features + [target], index=data["Date"])

# ----------------- Create Sequences -----------------
def create_sequences(dataset, seq_length=3):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length, :-1])  # features
        y.append(dataset[i+seq_length, -1])     # target (Yield)
    return np.array(X), np.array(y)

seq_length = 3  # use last 3 months to predict next month
X, y = create_sequences(scaled, seq_length)

# Reshape for LSTM (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# ----------------- Split Train/Test -----------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------- Build LSTM Model -----------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
model.summary()

# ----------------- Train -----------------
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# ----------------- Predictions -----------------
y_pred = model.predict(X_test)

# Inverse transform predictions to original scale
dummy_test = np.zeros((len(y_pred), len(features)+1))
dummy_test[:, -1] = y_pred[:, 0]
y_pred_rescaled = scaler.inverse_transform(dummy_test)[:, -1]

dummy_actual = np.zeros((len(y_test), len(features)+1))
dummy_actual[:, -1] = y_test
y_test_rescaled = scaler.inverse_transform(dummy_actual)[:, -1]

# ----------------- Plot Actual vs Predicted -----------------
plt.figure(figsize=(10,5))
plt.plot(data["Date"].iloc[-len(y_test):], y_test_rescaled, label="Actual Yield", marker="o")
plt.plot(data["Date"].iloc[-len(y_test):], y_pred_rescaled, label="Predicted Yield", marker="x")
plt.title("Rice Yield Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Yield")
plt.legend()
plt.show()

# ----------------- Forecast Future -----------------
last_sequence = scaled[-seq_length:, :-1]  # last N months
last_sequence = np.expand_dims(last_sequence, axis=0)

future_preds = []
n_future = 6  # predict next 6 months

for _ in range(n_future):
    pred = model.predict(last_sequence)
    future_preds.append(pred[0,0])
    
    # update sequence with new prediction
    new_entry = np.append(last_sequence[:, -1, :-1], pred, axis=0).reshape(seq_length, len(features))
    last_sequence = np.expand_dims(new_entry, axis=0)

# Rescale future predictions
dummy_future = np.zeros((len(future_preds), len(features)+1))
dummy_future[:, -1] = future_preds
future_preds_rescaled = scaler.inverse_transform(dummy_future)[:, -1]

print("\n📌 Future Predicted Yield (next 6 months):")
print(future_preds_rescaled)
