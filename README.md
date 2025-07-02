# 🎵 Music Assistant AI

An intelligent music genre classifier that analyzes audio files and provides educational insights about music characteristics. Built with machine learning and explainable AI.

## ✨ Features

### 🎯 AI Genre Classification
- **81.6% accuracy** on 4-genre classification (Classical, Jazz, Rock, Disco)
- Real-time audio analysis of uploaded files
- Probability scores for each genre prediction

### 🔬 Audio Feature Analysis
- **34 audio features** extracted using librosa
- **6 educational categories**: Rhythm & Timing, Voice & Instrument Texture, Musical Style, Energy & Dynamics, Musical Notes & Harmony, Sound Color & Texture
- User-friendly explanations of technical audio measurements

### 🧠 AI Explainability
- **SHAP analysis** shows which features influenced the prediction
- Visual feature importance rankings
- Educational insights into how AI "listens" to music

### 📚 Educational Value
- Plain English explanations of audio processing concepts
- Interactive tooltips with analogies and listening tips
- Bridge between technical measurements and musical understanding

## 🛠️ Tech Stack

### Backend
- **Python Flask** - Web framework
- **librosa** - Audio analysis and feature extraction
- **XGBoost** - Machine learning classifier
- **SHAP** - Model explainability
- **scikit-learn** - Data preprocessing and scaling
- **NumPy** - Numerical computations

### Frontend
- **HTML/CSS/JavaScript** - User interface
- **Responsive design** - Works on desktop and mobile
- **Interactive visualizations** - Progress bars, tooltips, feature cards

### Machine Learning Pipeline
- **Dataset**: GTZAN (400 samples across 4 genres)
- **Features**: 34 audio characteristics (MFCCs, chroma, spectral contrast, tonnetz, tempo)
- **Model**: XGBoost classifier with 81.2% test accuracy
- **Cross-validation**: 79.6% ± 2.6% accuracy

## 📊 Model Performance

```
              precision    recall  f1-score   support
   classical       0.76      0.95      0.84        20
       disco       0.80      0.80      0.80        20
        jazz       0.84      0.80      0.82        20
        rock       0.88      0.70      0.78        20

    accuracy                           0.816        80
```

## 📁 Project Structure

```
MusicAssistant/
├── backend.py              # Flask backend with API endpoints
├── requirements.txt        # Python dependencies
├── music_classifier.pkl    # Trained XGBoost model
├── feature_scaler.pkl      # StandardScaler for features
├── shap_explainer.pkl      # SHAP TreeExplainer
├── label_encoder.pkl       # Genre label encoder
├── index.html             # Frontend HTML
└── frontend/              # CSS and JavaScript files
```

## 🚀 Deployment

### Backend (Railway)
1. Deploy Flask backend to Railway
2. Install dependencies from requirements.txt
3. Serves API endpoints for audio analysis

### Frontend (Netlify)
1. Deploy static files to Netlify
2. Update API endpoints to point to Railway backend
3. Provides user interface for file uploads

## 🎵 Supported Audio Formats
- MP3
- WAV
- M4A
- Maximum file size: 50MB

## 🔬 Audio Features Analyzed

### Rhythm & Timing (1 feature)
- **Tempo**: Beats per minute detection

### Voice & Instrument Texture (5 features)
- **MFCCs 0-4**: Fundamental audio characteristics

### Musical Style Characteristics (4 features)
- **MFCCs 5-8**: Genre-specific audio patterns

### Energy & Dynamics (4 features)
- **MFCCs 9-12**: Intensity and dynamic measurements

### Musical Notes & Harmony (13 features)
- **Chroma 0-11**: Individual musical note presence
- **Tonnetz**: Harmonic stability measurement

### Sound Color & Texture (7 features)
- **Spectral Contrast 0-6**: Frequency distribution analysis

## 🎓 Educational Impact

This project demonstrates:
- **Music Information Retrieval** techniques
- **Explainable AI** in real-world applications
- **Audio signal processing** concepts
- **Machine learning** for creative domains
- **User experience design** for technical concepts

## 📈 Future Improvements

- [ ] Expand to more genres
- [ ] Add temporal analysis (verse/chorus detection)
- [ ] Implement deep learning models
- [ ] Real-time audio streaming analysis
- [ ] Mobile app development

## 📝 License

Educational project - feel free to learn from and build upon this work.

---

**Built with ❤️ for music education and AI transparency**
