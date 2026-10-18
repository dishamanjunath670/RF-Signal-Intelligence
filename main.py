
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: GENERATE SIMULATED RF SIGNALS
# ==========================================

sampling_rate = 1000
duration = 1
frequency = 50

time = np.arange(0, duration, 1 / sampling_rate)

# Generate normal signal with noise
normal_signal = np.sin(2 * np.pi * frequency * time)
noise = np.random.normal(0, 0.1, len(time))
normal_signal = normal_signal + noise

# Create an artificial amplitude anomaly
anomaly_signal = normal_signal.copy()
anomaly_signal[400:500] *= 4

# ==========================================
# STEP 2: DETECT AMPLITUDE ANOMALIES
# ==========================================

threshold = 1.5

anomaly_mask = np.abs(anomaly_signal) > threshold
detected_time = time[anomaly_mask]
detected_amplitude = anomaly_signal[anomaly_mask]

number_detected = np.sum(anomaly_mask)
total_samples = len(anomaly_signal)

anomaly_percentage = (number_detected / total_samples) * 100

# ==========================================
# STEP 3: EXTRACT SIGNAL FEATURES
# ==========================================

def extract_features(signal, sampling_rate):

    # RMS: effective signal amplitude
    rms = np.sqrt(np.mean(signal ** 2))

    # Average signal power (assuming unit resistance)
    power = np.mean(signal ** 2)

    # Largest absolute signal value
    peak_amplitude = np.max(np.abs(signal))

    # Calculate FFT
    N = len(signal)
    fft_values = np.fft.rfft(signal)
    magnitude = np.abs(fft_values)

    # Create frequency values
    frequencies = np.fft.rfftfreq(
        N, d=1 / sampling_rate
    )

    # Find strongest non-zero frequency
    peak_index = np.argmax(magnitude[1:]) + 1
    dominant_frequency = frequencies[peak_index]

    return rms, power, peak_amplitude, dominant_frequency


# Extract features from both signals
normal_features = extract_features(
    normal_signal, sampling_rate
)

anomaly_features = extract_features(
    anomaly_signal, sampling_rate
)

feature_names = [
    "RMS amplitude",
    "Average squared amplitude",
    "Peak amplitude",
    "Dominant frequency (Hz)"
]

# ==========================================
# STEP 4: DISPLAY RESULTS
# ==========================================

print("\nRF SIGNAL INTELLIGENCE SYSTEM")
print("=" * 40)

print("\nNORMAL SIGNAL FEATURES")
print("-" * 40)

for name, value in zip(feature_names, normal_features):
    print(f"{name}: {value:.4f}")

print("\nANOMALOUS SIGNAL FEATURES")
print("-" * 40)

for name, value in zip(feature_names, anomaly_features):
    print(f"{name}: {value:.4f}")

print("\nANOMALY DETECTION RESULTS")
print("-" * 40)

print("Total samples:", total_samples)
print("Anomalous samples:", number_detected)
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

if number_detected > 0:
    print("Result: Unusual amplitude detected")
else:
    print("Result: No amplitude anomaly detected")

# ==========================================
# STEP 5: PLOT SIGNAL AND FFT
# ==========================================

plt.figure(figsize=(12, 10))

# Plot 1: Anomaly detection
plt.subplot(3, 1, 1)

plt.plot(time, anomaly_signal,
         label="Simulated RF Signal")

plt.scatter(detected_time, detected_amplitude,
            color="red", s=12,
            label="Detected Anomalies")

plt.axhline(threshold, color="orange",
            linestyle="--", label="Threshold")

plt.axhline(-threshold, color="orange",
            linestyle="--")

plt.title("RF Signal Anomaly Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Plot 2: Normal signal frequency spectrum
plt.subplot(3, 1, 2)

normal_fft = np.fft.rfft(normal_signal)
frequencies = np.fft.rfftfreq(
    len(normal_signal), d=1 / sampling_rate
)

normal_magnitude = np.abs(normal_fft) / len(normal_signal)

plt.plot(frequencies, normal_magnitude)
plt.title("Normal Signal - FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 200)
plt.grid(True)

# Plot 3: Anomalous signal frequency spectrum
plt.subplot(3, 1, 3)

anomaly_fft = np.fft.rfft(anomaly_signal)
anomaly_magnitude = np.abs(anomaly_fft) / len(anomaly_signal)

plt.plot(frequencies, anomaly_magnitude)
plt.title("Anomalous Signal - FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 200)
plt.grid(True)

plt.tight_layout()
plt.show()
