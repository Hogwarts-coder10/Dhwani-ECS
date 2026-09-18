# Dhwani-ECS 🎧🌾

## Acoustic Insect Detection for Stored Grains

**Dhwani-ECS** is an acoustic machine-learning system designed to detect insect activity in stored grains by analyzing ultrasonic/acoustic recordings.

The system extracts meaningful audio features from grain-storage recordings and uses a **Random Forest classifier** to distinguish between:

- 🟢 **Clean** — no target insect activity detected
- 🔴 **Insect** — acoustic patterns associated with target insects detected

The project is designed with future **Raspberry Pi edge deployment** in mind, allowing recordings to be processed locally without requiring a powerful computer for inference.

---

## 📌 Problem Statement

Insects inside stored grains can cause significant quality and quantity losses before they become visible.

Traditional inspection methods may require:

- Opening or disturbing stored grain
- Manual inspection
- Periodic sampling
- Specialized equipment
- Human expertise

Dhwani-ECS explores an alternative approach:

> **Can insect activity be detected from the acoustic signals produced inside stored grain?**

The project uses audio signal processing and machine learning to identify acoustic patterns associated with insect activity.

---

## 🎯 Objectives

The main objectives of Dhwani-ECS are:

1. Capture acoustic signals from stored grain environments.
2. Preprocess recordings into usable audio segments.
3. Extract meaningful acoustic features.
4. Train a machine-learning classifier.
5. Detect insect-containing audio windows.
6. Aggregate window-level predictions into an overall detection result.
7. Prepare the model for lightweight edge deployment such as Raspberry Pi.

---

# 🧠 System Overview

```text
             Acoustic Recording
                    │
                    ▼
          ┌─────────────────────┐
          │   Audio Preprocessing│
          │      24 kHz          │
          └──────────┬──────────┘
                     │
                     ▼
             1-second Windows
                     │
                     ▼
          ┌─────────────────────┐
          │  Feature Extraction │
          │                     │
          │  • MFCC             │
          │  • Spectral Centroid│
          │  • Spectral Bandwidth│
          │  • Zero Crossing    │
          │  • RMS Energy       │
          │  • Peakiness        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Random Forest Model │
          └──────────┬──────────┘
                     │
                     ▼
             Insect Probability
                     │
                     ▼
          ┌─────────────────────┐
          │ Detection Decision  │
          └─────────────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
          🟢 CLEAN      🔴 INSECTS

```

---

# 🔬 Machine Learning Pipeline

## 1. Audio Sampling

The current pipeline uses:

```text
Sampling rate: 24,000 Hz
Channel: Mono
Window size: 1 second

```

Recordings are resampled to the target 24 kHz sampling rate before feature extraction.

---

## 2. Windowing

Each recording is divided into **1-second non-overlapping windows**.

For a 24 kHz signal:

```text
1 second = 24,000 samples

```

Each window is independently converted into a feature vector.

---

# 🎵 Feature Extraction

Dhwani-ECS extracts multiple acoustic characteristics from each 1-second window.

## MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** are used to represent the spectral characteristics of the audio.

The system calculates:

- 13 MFCC coefficients
- Mean of each coefficient
- Standard deviation of each coefficient

This produces:

```text
13 mean values
+
13 standard deviation values
=
26 features

```

---

## Spectral Centroid

The spectral centroid represents the approximate center of mass of the frequency spectrum.

It provides information about the distribution of frequency content in the recording.

---

## Spectral Bandwidth

Spectral bandwidth measures how widely distributed the frequencies are around the spectral centroid.

---

## Zero-Crossing Rate

Zero-crossing rate measures how frequently the audio waveform crosses zero amplitude.

It can help distinguish different types of acoustic signals.

---

## RMS Energy

The system calculates RMS energy:

```text
RMS = sqrt(mean(y²))

```

This represents the overall energy level of the audio window.

---

## Peakiness

The project also calculates a peakiness measure:

```text
peakiness = maximum absolute amplitude / RMS energy

```

This is useful for identifying sharp acoustic impulses.

The feature is particularly motivated by the observation that insect activity may produce sharp acoustic events.

---

# 🌲 Machine Learning Model

The current classifier is:

**Random Forest Classifier**

Configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

```

### Why Random Forest?

Random Forest is suitable for this stage of the project because:

- It works well with tabular feature vectors.
- It can model nonlinear relationships.
- It requires relatively little feature scaling.
- It is comparatively lightweight for inference.
- It provides probability estimates.
- It can be deployed more easily on resource-constrained hardware than many deep-learning models.

---

# 📊 Dataset

The data preparation pipeline uses the:

**Stored Product Insect Database / A-SPIDS dataset**

The dataset is accessed through Kaggle.

The project separates recordings into:

```text
data/
├── clean/
└── insect/

```

The data preparation script maps dataset recordings to these two classes based on the associated experimental metadata.

Target insects currently considered include:

- *Tenebrio molitor*
- *Tenebrio molitor* larvae
- *Tribolium confusum*
- *Callosobruchus maculatus*

The current data extraction configuration focuses insect samples on selected grain materials such as:

- Wheat Groats
- Rice

The extraction script also applies filtering rules to avoid ambiguous recordings and problematic sessions.

---

# 🧪 Training and Evaluation

The training process attempts to avoid overly optimistic results caused by having recordings from the same session in both training and testing data.

When sufficient wheat data is available, the project uses:

```text
Test = Wheat recordings
Train = Other grain recordings

```

Otherwise, it uses a session-based split:

```text
75% sessions → Training
25% sessions → Testing

```

This is important because randomly splitting individual audio windows can result in highly similar windows from the same recording appearing in both training and testing data.

---

# 📈 Evaluation Metrics

The training script reports:

- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

The evaluation is performed on the held-out test set before the final model is retrained on all available labeled data.

---

# 💾 Model Output

After training, the final Random Forest model is saved as:

```text
model.pkl

```

The model is serialized using **Joblib**.

---

# 🔍 Prediction Pipeline

The prediction script loads:

```text
model.pkl

```

and processes a supplied WAV file.

Example:

```bash
python predict.py recording.wav

```

The audio is divided into 1-second windows.

For each window, the model produces an insect probability.

A window is considered insect-positive when:

```text
insect probability > 0.5

```

The system then counts the number of positive windows.

The overall recording is classified as:

```text
INSECTS DETECTED

```

when more than 20% of the available windows are flagged.

Otherwise:

```text
clean

```

---

# 📁 Project Structure

```text
Dhwani-ECS/
│
├── data/
│   ├── clean/
│   └── insect/
│
├── cache/
│
├── train.py
├── predict.py
├── get_data.py
├── check_ch.py
├── requirements.txt
├── model.pkl
└── README.md

```

---

# 🐍 File Description

## `get_data.py`

Downloads and prepares acoustic data from the Kaggle dataset.

Responsibilities include:

- Kaggle dataset access
- File discovery
- Session identification
- Audio extraction
- Metadata-based labeling
- Clean/insect separation
- WAV generation

Run:

```bash
python get_data.py

```

---

## `check_ch.py`

Used to inspect the available audio channels and compare signal levels.

It helps determine which recording channel should be used for the ML pipeline.

---

## `train.py`

Main machine-learning training script.

Responsibilities:

```text
Load WAV files
      ↓
Create 1-second windows
      ↓
Extract features
      ↓
Create training/test split
      ↓
Train Random Forest
      ↓
Evaluate model
      ↓
Train final model
      ↓
Save model.pkl

```

Run:

```bash
python train.py

```

---

## `predict.py`

Runs inference on a WAV recording.

Example:

```bash
python predict.py recording.wav

```

Output includes:

- Timestamp of detected windows
- Insect probability
- Number of flagged windows
- Overall detection result

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Hogwarts-coder10/Dhwani-ECS.git
cd Dhwani-ECS

```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate

```

### Linux / Raspberry Pi

```bash
python3 -m venv .venv
source .venv/bin/activate

```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt

```

The current requirements include packages such as:

- NumPy
- Librosa
- SoundFile
- Scikit-learn
- Joblib
- SciPy
- Kaggle API

See `requirements.txt` for the exact pinned versions used by the project.

---

# 🔑 Kaggle API Setup

`get_data.py` uses the Kaggle API to access the dataset.

Configure your Kaggle credentials before running the data preparation script.

Typical Kaggle configuration uses:

```text
~/.kaggle/kaggle.json

```

Do **not** commit Kaggle credentials to GitHub.

---

# 🚀 Running the Complete Pipeline

## Step 1 — Prepare Dataset

```bash
python get_data.py

```

This creates:

```text
data/
├── clean/
└── insect/

```

---

## Step 2 — Train Model

```bash
python train.py

```

The trained model is saved as:

```text
model.pkl

```

---

## Step 3 — Run Prediction

```bash
python predict.py path/to/recording.wav

```

Example:

```bash
python predict.py test.wav

```

---

# 🍓 Raspberry Pi Deployment

Dhwani-ECS is intended to eventually run as an edge ML system.

A possible deployment architecture is:

```text
        Microphone / Sensor
                │
                ▼
          Raspberry Pi
                │
                ▼
        Audio Acquisition
                │
                ▼
        24 kHz Resampling
                │
                ▼
       1-second Audio Window
                │
                ▼
        Feature Extraction
                │
                ▼
       Random Forest Model
                │
                ▼
       Insect Probability
                │
                ▼
       Detection Decision
                │
        ┌───────┴────────┐
        ▼                ▼
   No Insects       Insects Detected

```

Because the current model uses handcrafted audio features and a Random Forest rather than a large neural network, the inference stage is intended to be relatively lightweight.

---

# 🧩 Hardware Concept

A future hardware implementation may consist of:

```text
┌──────────────────┐
│ Acoustic Sensor  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Raspberry Pi     │
│                  │
│ Audio Capture    │
│ Feature Extract. │
│ ML Inference     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Detection Result │
└──────────────────┘

```

The exact sensor, microphone interface, sampling configuration, and enclosure design can be finalized based on the hardware prototype.

---

# 🔐 Data Leakage Consideration

A key design consideration is avoiding leakage between training and testing data.

Audio recordings contain highly correlated windows. If windows from the same recording are randomly divided between training and testing, the evaluation may appear better than the model's ability to generalize to genuinely new recordings.

Therefore, Dhwani-ECS uses session-level grouping where appropriate.

The project also supports holding out wheat recordings when both classes are available in the wheat subset.

---

# ⚠️ Current Limitations

The current version is a research/prototype implementation.

Important limitations include:

- Dataset size and diversity may limit generalization.
- Performance on real-world storage environments still needs validation.
- Environmental noise can affect acoustic features.
- Microphone/sensor characteristics can change the recorded signal.
- The current decision thresholds are fixed.
- The model currently uses handcrafted features rather than end-to-end deep learning.
- Raspberry Pi deployment still requires hardware and performance validation.
- The current classifier provides detection rather than precise insect counting or species identification.

Therefore, model performance should be evaluated on representative real-world recordings before operational use.

---

# 🔮 Future Improvements

Potential future development includes:

### 1. Real-time detection

Process incoming microphone audio continuously rather than using pre-recorded WAV files.

### 2. Raspberry Pi integration

Deploy the trained model directly on Raspberry Pi.

### 3. Sensor integration

Connect the acoustic sensing hardware directly to the edge device.

### 4. Noise reduction

Add preprocessing techniques for:

- Background noise
- Machinery noise
- Human speech
- Environmental interference

### 5. Improved temporal modeling

Instead of treating every 1-second window independently, investigate temporal aggregation or sequence models.

### 6. Deep-learning comparison

Compare Random Forest against models such as:

- CNN
- 1D CNN
- 2D CNN on spectrograms
- Lightweight audio classification models

### 7. Confidence-based alerts

Introduce configurable confidence thresholds and multiple-window confirmation.

### 8. Dashboard

Create a dashboard showing:

```text
Storage Unit
     │
     ├── Current Status
     ├── Insect Probability
     ├── Detection History
     ├── Audio Events
     └── Alerts

```

### 9. Long-term monitoring

Store detection events and monitor changes in insect activity over time.

---

# 🛠️ Technology Stack

| ComponentTechnology  |                          |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Audio Processing     | Librosa                  |
| Numerical Computing  | NumPy                    |
| ML                   | Scikit-learn             |
| Model                | Random Forest            |
| Model Serialization  | Joblib                   |
| Audio I/O            | SoundFile                |
| Dataset API          | Kaggle API               |
| Target Edge Device   | Raspberry Pi             |
| Input                | Acoustic recordings      |
| Output               | Clean / Insect detection |

---

# 📦 Main Dependencies

The project currently relies on:

```text
numpy
librosa
scikit-learn
joblib
soundfile
kaggle
scipy

```

Exact versions are maintained in:

```text
requirements.txt

```

---

# 🧠 End-to-End Workflow

```text
                    DATA COLLECTION
                          │
                          ▼
                 A-SPIDS DATASET
                          │
                          ▼
                  DATA EXTRACTION
                          │
                          ▼
                CLEAN / INSECT DATA
                          │
                          ▼
                 AUDIO PREPROCESSING
                     24 kHz Mono
                          │
                          ▼
                    1-sec Windows
                          │
                          ▼
                  FEATURE EXTRACTION
                          │
              ┌───────────┼───────────┐
              │           │           │
            MFCC      Spectral     Energy /
                       Features     Peakiness
              │           │           │
              └───────────┼───────────┘
                          │
                          ▼
                  FEATURE VECTORS
                          │
                          ▼
                 RANDOM FOREST MODEL
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Evaluation          Final Model
                                   │
                                   ▼
                              model.pkl
                                   │
                                   ▼
                              PREDICTION
                                   │
                                   ▼
                         INSECT PROBABILITY
                                   │
                                   ▼
                         WINDOW AGGREGATION
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                       CLEAN          INSECTS DETECTED

```

---

# 📊 Example Prediction Output

A prediction may produce output similar to:

```text
2s  insect  (0.87)
5s  insect  (0.72)
8s  insect  (0.91)

3/10 windows flagged
INSECTS DETECTED

```

The probabilities shown above are only examples; actual values depend on the recording and trained model.

---

# 🎓 Project Context

**Dhwani-ECS** is developed as an experimental edge-computing / machine-learning project exploring acoustic insect detection in stored grains.

The project combines:

- Digital signal processing
- Audio feature engineering
- Machine learning
- Dataset preparation
- Edge inference
- Raspberry Pi deployment

---

# 📜 License

Add the project's intended license here before distributing the repository.

Example:

```text
MIT License

```

if the project team decides to release the code under MIT.

---

# 👥 Contributors

Add project team members here:

```text
- Name — Role
- Name — Role
- Name — Role

```

---

# ⭐ Project Status

**Current stage:** Machine-learning prototype

### Completed

-  Dataset extraction pipeline
-  Clean/insect labeling
-  Audio preprocessing
-  Feature extraction
-  Random Forest training
-  Model evaluation
-  Model serialization
-  WAV prediction pipeline

### In Progress / Future

-  Raspberry Pi integration
-  Real-time audio capture
-  Hardware sensor integration
-  Field testing
-  Long-term monitoring
-  Real-time alerts
-  Deployment optimization

---

## 🚀 Dhwani-ECS

**Listen → Analyze → Detect**

An edge-oriented machine-learning approach to detecting hidden insect activity through acoustic signals.