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
