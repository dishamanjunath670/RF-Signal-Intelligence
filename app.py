
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="RF Signal Intelligence",
    page_icon="📡",
    layout="wide"
)

st.title("📡 RF Signal Intelligence Dashboard")
st.write(
    "Simulated RF signal analysis, FFT frequency analysis, "
    "and machine-learning anomaly detection."
)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Signal Settings")

frequency = st.sidebar.slider(
    "Signal frequency (Hz)",
    min_value=20,
    max_value=100,
    value=50
)

anomaly_start = st.sidebar.slider(
    "Anomaly start time (seconds)",
    min_value=0.1,
    max_value=0.8,
    value=0.4,
    step=0.1
)

anomaly_strength = st.sidebar.slider(
    "Anomaly amplitude multiplier",
    min_value=1.5,
    max_value=6.0,
    value=4.0,
    step=0.5
)

# ==========================================
# GENERATE SIGNAL
# ==========================================

sampling_rate = 1000
duration = 1.0
N = int(sampling_rate * duration)

time = np.arange(N) / sampling_rate

rng = np.random.default_rng(42)

signal = np.sin(2 * np.pi * frequency * time)
signal += rng.normal(0, 0.1, N)

# Add an artificial anomaly
anomaly_end = min(anomaly_start + 0.1, duration)

anomaly_mask = (
    (time >= anomaly_start) &
    (time < anomaly_end)
)

signal[anomaly_mask] *= anomaly_strength

# ==========================================
# FEATURE EXTRACTION
# ==========================================

window_size = 100

def extract_features(data):
    rms = np.sqrt(np.mean(data ** 2))
    power = np.mean(data ** 2)
    peak = np.max(np.abs(data))

    spectrum = np.abs(np.fft.rfft(data))
    frequencies = np.fft.rfftfreq(
        len(data),
        d=1 / sampling_rate
    )

    if len(spectrum) > 1:
        peak_index = np.argmax(spectrum[1:]) + 1
        dominant_frequency = frequencies[peak_index]
    else:
        dominant_frequency = 0

    return [
        rms,
        power,
        peak,
        dominant_frequency
    ]

# ==========================================
# TRAIN ISOLATION FOREST ON NORMAL WINDOWS
# ==========================================

training_features = []

for _ in range(200):
    training_time = np.arange(window_size) / sampling_rate

    normal_window = np.sin(
        2 * np.pi * frequency * training_time
    )

    normal_window += rng.normal(0, 0.1, window_size)

    training_features.append(
        extract_features(normal_window)
    )

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(np.array(training_features))

# ==========================================
# ANALYZE THE SIGNAL IN WINDOWS
# ==========================================

windows = []
window_times = []
window_features = []

for start in range(0, N, window_size):
    end = start + window_size

    if end > N:
        break

    window = signal[start:end]

    windows.append(window)
    window_times.append(time[start])
    window_features.append(extract_features(window))

X_test = np.array(window_features)
predictions = model.predict(X_test)

detected_windows = predictions == -1
detected_count = int(np.sum(detected_windows))

# ==========================================
# DISPLAY SUMMARY
# ==========================================

rms, power, peak, dominant_frequency = extract_features(signal)

col1, col2, col3, col4 = st.columns(4)

col1.metric("RMS Amplitude", f"{rms:.3f}")
col2.metric("Average Squared Amplitude", f"{power:.3f}")
col3.metric("Peak Amplitude", f"{peak:.3f}")
col4.metric("Dominant Frequency", f"{dominant_frequency:.1f} Hz")

st.subheader("Machine-Learning Detection")

if detected_count > 0:
    st.warning(
        f"Isolation Forest flagged {detected_count} "
        f"out of {len(windows)} signal windows."
    )
else:
    st.success("No anomalous windows were flagged.")

st.caption(
    "These are simulated signals. A flagged window is an unusual "
    "pattern, not proof of malicious radio activity."
)

# ==========================================
# GRAPH 1: SIGNAL AND DETECTED WINDOWS
# ==========================================

st.subheader("1. RF Signal and Anomaly Detection")

fig1, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(time, signal, linewidth=1, label="RF Signal")

for i, flagged in enumerate(detected_windows):
    if flagged:
        ax1.axvspan(
            window_times[i],
            window_times[i] + window_size / sampling_rate,
            color="red",
            alpha=0.2
        )

ax1.axvline(
    anomaly_start,
    color="orange",
    linestyle="--",
    label="Artificial anomaly start"
)

ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Red shading = windows flagged by Isolation Forest")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)
plt.close(fig1)

# ==========================================
# GRAPH 2: FFT SPECTRUM
# ==========================================

st.subheader("2. FFT Frequency Spectrum")

fft_values = np.fft.rfft(signal)
frequency_axis = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

magnitude = np.abs(fft_values) / N

fig2, ax2 = plt.subplots(figsize=(12, 4))

ax2.plot(frequency_axis, magnitude)

ax2.set_xlim(0, 200)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("Frequency Components of the Simulated RF Signal")
ax2.grid(True)

st.pyplot(fig2)
plt.close(fig2)

# ==========================================
# GRAPH 3: FEATURE COMPARISON
# ==========================================

st.subheader("3. Signal Features by Window")

fig3, ax3 = plt.subplots(figsize=(12, 4))

rms_values = X_test[:, 0]

colors = [
    "red" if flagged else "blue"
    for flagged in detected_windows
]

ax3.bar(
    np.arange(len(rms_values)),
    rms_values,
    color=colors
)

ax3.set_xlabel("Signal Window")
ax3.set_ylabel("RMS Amplitude")
ax3.set_title("Blue = Not flagged | Red = Flagged by the model")
ax3.grid(axis="y")

st.pyplot(fig3)
plt.close(fig3)

# ==========================================
# RESULTS TABLE
# ==========================================

st.subheader("4. Window-by-Window Results")

results = []

for i, features in enumerate(X_test):
    results.append({
        "Window": i + 1,
        "Start Time (s)": round(window_times[i], 2),
        "RMS Amplitude": round(features[0], 3),
        "Power": round(features[1], 3),
        "Peak Amplitude": round(features[2], 3),
        "Dominant Frequency (Hz)": round(features[3], 1),
        "ML Result": (
            "Anomaly" if predictions[i] == -1 else "Normal"
        )
    })

st.dataframe(results, use_container_width=True)

st.info(
    "The Isolation Forest model is retrained on simulated normal "
    "windows whenever the dashboard reruns. Validate it with varied "
    "signals before drawing conclusions about real RF environments."
)
