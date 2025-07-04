import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib

# Load dataset
df = pd.read_excel("Cleand_data.xlsx")

# Encode 'Flood' column
df['Flood'] = df['Flood'].map({'YES': 1, 'NO': 0})
df.dropna(inplace=True)

# Features and labels
X = df.loc[:, 'JAN':'DEC'].values
y = df['Flood'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

# Save model
model.save("flood_prediction_model.h5")
