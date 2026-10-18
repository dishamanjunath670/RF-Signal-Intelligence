
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ==========================================
# STEP 1: CREATE SIMULATED SIGNAL DATA
# ==========================================

np.random.seed(42)

sampling_rate = 1000
frequency = 50
samples_per_window = 100
number_of_windows = 200

def create_signal_window(anomaly=False):
    time = np.arange(samples_per_window) / sampling_rate

    # Generate a 50 Hz signal with small random noise
    signal = np.sin(2 * np.pi * frequency * time)
    signal += np.random.normal(0, 0.1, samples_per_window)

    # Artificial anomaly: unusually large amplitude
    if anomaly:
        signal *= 4

    return signal


# ==========================================
# STEP 2: EXTRACT FEATURES FROM EACH WINDOW
# ==========================================

def extract_features(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    power = np.mean(signal ** 2)
    peak = np.max(np.abs(signal))

    fft_values = np.fft.rfft(signal)
    frequency_axis = np.fft.rfftfreq(
        len(signal), d=1 / sampling_rate
    )

    magnitude = np.abs(fft_values)
    peak_index = np.argmax(magnitude[1:]) + 1
    dominant_frequency = frequency_axis[peak_index]

    return [rms, power, peak, dominant_frequency]


# ==========================================
# STEP 3: CREATE TRAINING DATA
# ==========================================

training_features = []

for i in range(number_of_windows):
    signal = create_signal_window(anomaly=False)
    features = extract_features(signal)
    training_features.append(features)

X_train = np.array(training_features)

print("Training samples:", len(X_train))
print("Features per sample:", X_train.shape[1])


# ==========================================
# STEP 4: TRAIN THE ISOLATION FOREST MODEL
# ==========================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X_train)

print("\nMachine-learning model trained successfully.")


# ==========================================
# STEP 5: TEST NORMAL AND ANOMALOUS WINDOWS
# ==========================================

test_signals = []
test_features = []
true_labels = []

# First, create normal test windows
for i in range(10):
    signal = create_signal_window(anomaly=False)
    test_signals.append(signal)
    test_features.append(extract_features(signal))
    true_labels.append("Normal")

# Then, create artificial anomalous windows
for i in range(10):
    signal = create_signal_window(anomaly=True)
    test_signals.append(signal)
    test_features.append(extract_features(signal))
    true_labels.append("Artificial anomaly")

X_test = np.array(test_features)

# Isolation Forest returns 1 for normal and -1 for anomaly
predictions = model.predict(X_test)

print("\nMACHINE-LEARNING RESULTS")
print("=" * 40)

for i, prediction in enumerate(predictions):
    if prediction == -1:
        result = "ANOMALY DETECTED"
    else:
        result = "Normal"

    print(
        f"Sample {i + 1:02d} | "
        f"Expected: {true_labels[i]:18s} | "
        f"Predicted: {result}"
    )

# ==========================================
# STEP 6: SUMMARIZE THE RESULTS
# ==========================================

normal_correct = sum(
    predictions[i] == 1
    for i in range(10)
)

anomaly_detected = sum(
    predictions[i] == -1
    for i in range(10, 20)
)

print("\nSUMMARY")
print("-" * 40)
print("Normal samples correctly accepted:",
      normal_correct, "/ 10")
print("Artificial anomalies detected:",
      anomaly_detected, "/ 10")

print("\nNote: Results can vary with different data.")
print("This is a simulated learning experiment.")


# ==========================================
# STEP 7: VISUALIZE THE FEATURES
# ==========================================

plt.figure(figsize=(10, 6))

normal_rms = X_test[:10, 0]
anomaly_rms = X_test[10:, 0]

plt.scatter(
    range(1, 11),
    normal_rms,
    label="Normal test windows"
)

plt.scatter(
    range(11, 21),
    anomaly_rms,
    label="Artificial anomaly windows"
)

plt.axhline(
    np.mean(normal_rms),
    linestyle="--",
    label="Mean normal RMS"
)

plt.title("RF Signal Feature Analysis Using Machine Learning")
plt.xlabel("Test window")
plt.ylabel("RMS amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()