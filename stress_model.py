import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fix numpy compatibility
np.trapz = np.trapezoid

features = []
labels = []

# Generate synthetic ECG signals for demo
for i in range(40):

    # Normal condition
    ecg = nk.ecg_simulate(duration=10, heart_rate=70, sampling_rate=1000)
    signals, info = nk.ecg_process(ecg, sampling_rate=1000)

    hrv = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=1000)
    features.append(hrv.values[0])
    labels.append(0)  # No stress

    # Stress condition
    ecg_stress = nk.ecg_simulate(duration=10, heart_rate=100, sampling_rate=1000)
    signals, info = nk.ecg_process(ecg_stress, sampling_rate=1000)

    hrv = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=1000)
    features.append(hrv.values[0])
    labels.append(1)  # Stress

# Convert to dataframe
X = pd.DataFrame(features)
y = np.array(labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, pred))

