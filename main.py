
# main.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# -------------------------
# 1. Load Dataset
# -------------------------
data = pd.read_csv("F:\\New folder\\price predictor\\data\\sales.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# -------------------------
# 2. Plot Sales Over Time
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Sales'], marker='o', label="Actual Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# -------------------------
# 3. Create Features for ML
# -------------------------
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

X = data[['Day', 'Month']]
y = data['Sales']

# -------------------------
# 4. Split Data & Train Model
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# -------------------------
# 5. Evaluate Model
# -------------------------
y_pred = model.predict(X_test)
error = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {error:.2f}")

# -------------------------
# 6. Predict Next 7 Days
# -------------------------
last_date = data['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

future_data = pd.DataFrame({
    'Day': [d.day for d in future_dates],
    'Month': [d.month for d in future_dates]
})

future_predictions = model.predict(future_data)

# Print predictions
print("\nPredicted Sales for Next 7 Days:")
for date, pred in zip(future_dates, future_predictions):
    print(f"{date.date()}: {int(pred)}")

# -------------------------
# 7. Plot Predictions
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Sales'], marker='o', label="Actual Sales")
plt.plot(future_dates, future_predictions, marker='x', linestyle='--', color='red', label="Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Prediction for Next 7 Days")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()