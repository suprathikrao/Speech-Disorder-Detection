# Speech Disorder Detection Using Machine Learning
**B.Tech Major Project (IV B.Tech I Sem)**  
*Department of Information Technology*

---

> ### ⚠️ Clinical Disclaimer
> **PRELIMINARY SCREENING TOOL ONLY:** This software is designed as a computer-aided preliminary screening and educational research tool. It is **NOT** a certified medical diagnostic device and is **NOT** a replacement for formal clinical diagnosis and evaluation by a licensed Speech-Language Pathologist (SLP), neurologist, or ENT physician.

---

## 1. Project Overview
Speech disorders such as **Dysarthria**, **Dysphonia**, and **Stuttering** severely impact verbal communication, social interaction, and quality of life. Traditional diagnosis requires specialized clinical acoustic evaluation and perceptual speech assessments. 

This project implements an automated, machine-learning-based acoustic speech screening system. It analyzes `.wav` speech audio, extracts 40 clinical acoustic and spectral features (Mel-Frequency Cepstral Coefficients, fundamental frequency/pitch F0, energy/RMS, zero-crossing rate, spectral centroid, rolloff, and bandwidth), and evaluates multiple Scikit-Learn classification algorithms (**Support Vector Machine with RBF kernel**, **Random Forest**, and **Logistic Regression**).

The system includes a **FastAPI backend** connected to a **MySQL database** for persistent patient records, screening history, and model benchmark tracking, alongside a modern **React (Vite) web application** featuring live microphone recording with real-time waveform visualization, file upload, confidence gauges, and acoustic breakdown cards.

---

## 2. Project Architecture & Directory Structure

```
d:\SDD\
├── data\                                # Categorized audio recordings (.wav)
│   ├── normal\                          # Non-pathological fluent speech
│   ├── dysarthria\                      # Articulatory & motor impairment speech
│   ├── dysphonia\                       # Vocal cord hoarseness/perturbation
│   ├── stuttering\                      # Repetitions, blocks, and disfluency
│   └── features.csv                     # Extracted 40-feature acoustic matrix
├── models\                              # Persisted models and transformers
│   ├── best_model.joblib                # Highest F1-score trained classifier
│   ├── scaler.joblib                    # Fitted StandardScaler
│   ├── label_encoder.joblib             # Fitted class label encoder
│   └── model_metadata.json              # Benchmark evaluation metrics & matrices
├── notebooks\                           # Academic exploratory analysis
│   └── exploratory_analysis.ipynb       # Spectrograms, waveforms, feature distributions
├── src\                                 # Core standalone ML pipeline
│   ├── preprocessing.py                 # 16kHz resampling, silence trimming, normalization
│   ├── feature_extraction.py            # MFCCs, F0 pitch, RMS, ZCR, Spectral features
│   ├── generate_mock_data.py            # Multi-class synthetic acoustic wave generator
│   ├── train.py                         # Stratified 80/20 train/test, SVM, RF, LogReg
│   └── predict.py                       # Single audio file CLI & programmatic inference
├── BACKEND\                             # FastAPI REST API & Database Layer
│   ├── app\
│   │   ├── config.py                    # Database credentials & directory paths
│   │   ├── database.py                  # SQLAlchemy engine & session factory
│   │   ├── models.py                    # MySQL tables (patients, screening_records, benchmarks)
│   │   ├── schemas.py                   # Pydantic data schemas
│   │   ├── services.py                  # ML bridge & persistence services
│   │   └── main.py                      # FastAPI routes & endpoints
│   ├── uploads\                         # Uploaded speech audio recordings
│   └── requirements.txt
├── FRONTEND\
│   └── Speech Disorder Detection\       # Modern React 19 + Vite Dashboard
│       ├── src\
│       │   ├── App.jsx                  # Main screening studio, benchmarks, history UI
│       │   ├── App.css                  # Custom styling, animations, progress meters
│       │   ├── index.css                # Global medical dark mode & glassmorphism tokens
│       │   └── main.jsx
│       ├── package.json
│       └── vite.config.js
├── requirements.txt                     # Global Python requirements
└── README.md                            # Complete documentation & user guide
```

---

## 3. Acoustic Feature Engineering Pipeline

For every audio input sample, the system extracts **40 quantitative acoustic features**:

| Feature Family | Count | Clinical & Acoustic Relevance |
| :--- | :---: | :--- |
| **MFCCs (1 to 13)** | 26 (Mean & Std) | Captures vocal tract shape, formant envelope dynamics, and articulatory precision. |
| **Fundamental Frequency (F0)** | 4 (Mean, Std, Min, Max) | Pitch stability, phonatory tremors, monotone contours in dysarthria, and voice breaks. |
| **RMS Energy** | 2 (Mean, Std) | Speech power, amplitude stability, loudness decay, and syllable stress. |
| **Zero-Crossing Rate (ZCR)** | 2 (Mean, Std) | Unvoiced phoneme ratio, breathiness, turbulence noise in dysphonic speech. |
| **Spectral Centroid** | 2 (Mean, Std) | Center of spectral mass ("brightness"), highlighting vocal friction and noise. |
| **Spectral Rolloff** | 2 (Mean, Std) | High-frequency energy roll-off threshold (85% energy boundary). |
| **Spectral Bandwidth** | 2 (Mean, Std) | Width of spectral spread around the centroid; indicator of voice roughness. |

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 to 3.14
- Node.js (v18+) and npm
- MySQL Server (default credentials configured: `Root115:suprathik123@127.0.0.1:3306`)

### Step 1: Clone or Navigate to Directory
```powershell
cd d:\SDD
```

### Step 2: Set Up Python Virtual Environment
```powershell
# Using the existing venv or creating a new one:
cd BACKEND
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python requirements
pip install -r requirements.txt
cd ..
```

### Step 3: Set Up MySQL Database
The backend automatically executes `CREATE DATABASE IF NOT EXISTS speech_disorder_db` and builds the database tables (`patients`, `screening_records`, `model_benchmarks`) when the server initializes.
If you need to customize database credentials, update `BACKEND/app/config.py` or set environment variables:
```powershell
$env:DB_USER="Root115"
$env:DB_PASSWORD="suprathik123"
$env:DB_HOST="127.0.0.1"
$env:DB_PORT="3306"
$env:DB_NAME="speech_disorder_db"
```

---

## 5. Running the Machine Learning Pipeline (CLI)

### 1. Generate Synthetic Audio Dataset
If you do not yet have local recordings, generate 80 realistic acoustic samples:
```powershell
d:\SDD\BACKEND\venv\Scripts\python.exe src/generate_mock_data.py
```

### 2. Extract Acoustic Features to CSV
```powershell
d:\SDD\BACKEND\venv\Scripts\python.exe src/feature_extraction.py --build data
```
This processes all audio in `data/` and saves `data/features.csv`.

### 3. Train and Benchmark Classifiers
```powershell
d:\SDD\BACKEND\venv\Scripts\python.exe src/train.py
```
This trains:
1. **Support Vector Machine (SVM)** with RBF kernel and probability calibration
2. **Random Forest Classifier** (100 ensemble estimators)
3. **Logistic Regression** (L2 regularized)

Outputs a comparative table (Accuracy, Precision, Recall, F1-Score) and saves:
- `models/best_model.joblib`
- `models/scaler.joblib`
- `models/model_metadata.json`

### 4. Run Inference on a Single Audio File
```powershell
d:\SDD\BACKEND\venv\Scripts\python.exe src/predict.py --file data/dysarthria/sample_dysarthria_01.wav
```
Outputs formatted terminal report:
```
=================================================================
  SPEECH DISORDER SCREENING REPORT
=================================================================
  File Analyzed        : sample_dysarthria_01.wav
  Predicted Condition  : DYSARTHRIA
  Confidence Score     : 92.7%
  Classifier Model     : Support Vector Machine (SVM)
-----------------------------------------------------------------
  Probability Distribution:
    - dysarthria     : 92.73% | #######################
    - dysphonia      :  1.99% | 
    - normal         :  2.43% | 
    - stuttering     :  2.85% | 
-----------------------------------------------------------------
  Key Acoustic Indicators:
    - mean_pitch_f0_hz         : 221.1
    - pitch_variability_std    : 102.1
    - energy_rms               : 0.2005
    - zero_crossing_rate       : 0.2364
    - spectral_centroid_hz     : 2389.1
=================================================================
```

---

## 6. Running the Web Application (Backend + Frontend)

### Start the FastAPI Backend
```powershell
cd d:\SDD\BACKEND
.\venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

### Start the React Frontend
Open a second terminal:
```powershell
cd "d:\SDD\FRONTEND\Speech Disorder Detection"
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 7. Web Application Features

1. **Screening Studio**:
   - **Live Microphone Recording**: Real-time canvas audio frequency visualizer with Web Audio API.
   - **File Upload**: Drag-and-drop `.wav` files with instant audio player playback.
   - **One-Click Presets**: Test samples for Normal, Dysarthria, Dysphonia, and Stuttering.
   - **Patient Metadata**: Capture patient name, age, gender, and clinical notes, automatically stored in MySQL.
   - **Screening Report**: Visual confidence gauge, 4-class probability distribution, extracted acoustic parameters, and clinical notes.
2. **Model Benchmarks**:
   - Live performance comparison table (SVM vs. Random Forest vs. Logistic Regression).
   - Interactive Confusion Matrix heatmap.
   - One-click "Re-Train All Models" trigger.
3. **Screening History**:
   - Searchable table of past patient screenings loaded directly from MySQL (`speech_disorder_db`).
   - In-line audio playback of recorded or uploaded samples.
4. **Dataset Manager**:
   - Monitor audio sample counts per class in `data/`.
   - One-click dataset re-generation.

---

## 8. Plugging in Real Clinical Datasets

When ready to evaluate clinical recordings, place `.wav` files into the corresponding class subdirectories:
- **TORGO Database** or **UASpeech**: Copy dysarthric patient recordings into `data/dysarthria/`.
- **Saarbruecken Voice Database (SVD)**: Copy sustained vowels with pathologies (e.g. dysphonia, cord paresis) into `data/dysphonia/`.
- **UCLASS (University College London Archive of Stuttered Speech)**: Copy recordings into `data/stuttering/`.
- **Healthy Controls**: Copy control speaker recordings into `data/normal/`.

After copying files, click **"Re-Train All Models"** on the web dashboard or run `python src/train.py` to update the model weights.

---

## 9. Academic Project Information
- **Course**: IV B.Tech I Semester (Major Project)
- **Department**: Department of Information Technology
- **Project Domain**: Speech Signal Processing, Acoustic Feature Extraction, and Supervised Machine Learning
