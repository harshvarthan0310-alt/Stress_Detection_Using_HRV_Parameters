import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.trapz = np.trapezoid

# Generate ECG signal
ecg = nk.ecg_simulate(duration=10, heart_rate=75, sampling_rate=1000)

# Process ECG
signals, info = nk.ecg_process(ecg, sampling_rate=1000)

# HRV Features
hrv_time = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=1000)

print("HRV Features")
print(hrv_time)

# Create dataset for ML
features = []
labels = []

for i in range(40):

    ecg_normal = nk.ecg_simulate(duration=10, heart_rate=70)
    signals, info = nk.ecg_process(ecg_normal)

    hrv = nk.hrv_time(info["ECG_R_Peaks"])
    features.append(hrv.values[0])
    labels.append(0)

    ecg_stress = nk.ecg_simulate(duration=10, heart_rate=100)
    signals, info = nk.ecg_process(ecg_stress)

    hrv = nk.hrv_time(info["ECG_R_Peaks"])
    features.append(hrv.values[0])
    labels.append(1)

X = pd.DataFrame(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, pred))

# Plot ECG
nk.ecg_plot(signals)
plt.show()
