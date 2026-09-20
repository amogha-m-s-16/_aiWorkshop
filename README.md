# 🎧 AI Machine Sound Detector

An AI-powered full-stack application that detects and classifies machine sounds (e.g., normal vs. faulty/abnormal operation) using machine learning, with an interactive **Streamlit** web interface.

## 📌 Overview

AI Machine Sound Detector analyzes audio recordings from industrial or mechanical equipment and predicts whether the machine is operating normally or exhibiting signs of a fault. This can help with predictive maintenance, early anomaly detection, and reducing downtime.

## ✨ Features

- 🎙️ Upload or record machine audio for analysis
- 🧠 ML/DL model for sound classification (normal vs. abnormal)
- 📊 Real-time visualization of audio waveform and spectrogram
- ⚡ Fast, interactive Streamlit UI
- 📁 Support for common audio formats (WAV, MP3)
- 📈 Prediction confidence scores

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/ML:** Python, NumPy, Pandas
- **Audio Processing:** Librosa
- **Model:** Scikit-learn / TensorFlow / PyTorch *(update based on what you used)*
- **Visualization:** Matplotlib / Plotly

## 📂 Project Structure

```
AI-Machine-Sound-Detector/
│
├── app.py                  # Main Streamlit application
├── model/
│   └── sound_model.pkl     # Trained ML model
├── utils/
│   ├── preprocessing.py    # Audio preprocessing functions
│   └── feature_extraction.py
├── data/
│   └── samples/            # Sample audio files
├── requirements.txt        # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/AI-Machine-Sound-Detector.git
   cd AI-Machine-Sound-Detector
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🎯 How It Works

1. **Upload Audio** — User uploads a `.wav`/`.mp3` file of the machine sound.
2. **Preprocessing** — Audio is cleaned and converted into features (e.g., MFCCs, spectrograms).
3. **Prediction** — The trained model classifies the sound as normal or abnormal.
4. **Results** — Streamlit displays the prediction, confidence score, and visualizations.

## 📊 Model

- **Input:** Audio features (MFCC / Mel-spectrogram)
- **Output:** Classification label (Normal / Abnormal) with confidence score
- **Training Data:** *(mention your dataset, e.g., MIMII dataset, custom recordings, etc.)*

## 🧪 Example Usage

```python
from utils.feature_extraction import extract_features
from model import load_model

features = extract_features("sample.wav")
model = load_model("model/sound_model.pkl")
prediction = model.predict(features)
```

## 📸 Screenshots

*(Add screenshots or a demo GIF of your Streamlit app here)*

## 🗺️ Roadmap

- [ ] Add support for real-time microphone input
- [ ] Multi-class fault classification
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add model retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or feedback, feel free to reach out or open an issue on GitHub.
