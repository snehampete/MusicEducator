🎵 The ultimate music technicality educator!!

Presenting an intelligent music genre classifier that analyzes audio files and provides educational insights about music characteristics! Built with machine learning and explainable AI(for a change- you don’t just see the output numerical values of audio analysis, but a simple intuitive way to understand WHY exactly AI made those decisions)

🚀 Quick Start (5 Steps)
Step 1: Clone the Repository
git clone https://github.com/snehampet/MusicEducator.git
cd MusicEducator

Step 2: Install Python Dependencies
pip install -r requirements.txt

Step 3: Start the Backend Server
python backend.py

Step 4: Open the Website
•	Open your web browser
•	Go to: http://localhost:5000

Step 5: Test with Audio
•	Click "Choose File" or drag and drop an audio file
•	Wait for analysis (30-60 seconds)
•	View AI results and educational insights
📋 Prerequisites
Before starting, ensure you have:
•	Python 3.8 or higher installed
•	Git installed
•	At least 2GB free disk space
•	Audio files to test (MP3, WAV, M4A formats)


🛠️ Detailed Installation Guide
For Windows Users
1.	Install Python:
o	Download from python.org
o	✅ Check "Add Python to PATH" during installation
3.	Install Git:
o	Download from git-scm.com
4.	Open Command Prompt:
o	Press Win + R, type cmd, press Enter
5.	Run the commands:
6.	git clone https://github.com/snehampet/MusicEducator.git
7.	cd MusicEducator
8.	pip install -r requirements.txt
9.	python backend.py

   
For Mac Users
1.	Install Homebrew (if not installed):
2.	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
3.	Install Python and Git:
4.	brew install python git
5.	Clone and run:
6.	git clone https://github.com/snehampet/MusicEducator.git
7.	cd MusicEducator
8.	pip3 install -r requirements.txt
9.	python3 backend.py
For Linux Users
1.	Install dependencies:
2.	sudo apt update
3.	sudo apt install python3 python3-pip git
4.	Clone and run:
5.	git clone https://github.com/snehampet/MusicEducator.git
6.	cd MusicEducator
7.	pip3 install -r requirements.txt
8.	python3 backend.py
🔧 Troubleshooting Common Issues
"Command not found" errors
•	Windows: Reinstall Python with "Add to PATH" checked
•	Mac/Linux: Use python3 and pip3 instead of python and pip
"No module named 'librosa'" error
pip install librosa numpy scipy
"Port already in use" error
•	Close other applications using port 5000
•	Or change port in backend.py: app.run(port=5001)
Large file download issues
•	Model files are 10MB+ total
•	Ensure stable internet connection
•	Use git clone rather than downloading ZIP
Audio processing takes too long
•	Expected: 30-60 seconds for analysis
•	Ensure sufficient RAM (4GB+ recommended)
•	Try with smaller audio files first


📊 Testing the Installation
Test Files to Try
1.	Classical music - Should predict "Classical"
2.	Rock song - Should predict "Rock"
3.	Jazz track - Should predict "Jazz"
4.	Electronic/Dance - Should predict "Disco"

Expected Results
1.	Genre prediction with confidence percentages(beware, confidence and accuracy are different metrics! Accuracy tries to convey how good the model generally is, while confidence is on how sure the model is to make a prediction.)
2.	6 feature categories filled with data
3.	SHAP analysis showing feature importance 
4.	Educational explanations for each measurement
   
✨ Features
🎯 AI Genre Classification
1.	82% accuracy on 4-genre classification (Classical, Jazz, Rock, Disco)
2. Real-time audio analysis of uploaded files
3. Confidence scores for each genre prediction

🔬 Audio Feature Analysis
1.	34 audio features extracted using librosa
2.	6 educational categories: Rhythm & Timing, Voice & Instrument Texture, Musical Style, Energy & Dynamics, Musical Notes & Harmony, Sound Color & Texture
3.	User-friendly explanations of technical audio measurements
   
🧠 AI Explainability
1.	SHAP analysis shows which features influenced the prediction: imagine this as a sort of “scorecard” for each decision on a certain audio feature made by the trained AI. 
2.	Visual feature importance rankings
3.	Educational insights into how AI "listens" to music
   
📚 Educational Value
1.	Simplified explanations of audio processing concepts,
2.	Interactive tooltips with analogies and listening tips,
3.	Bridge between technical measurements and musical understanding
   
🛠️ Tech Stack
Backend
•	Python Flask - Web framework
•	librosa - Audio analysis and feature extraction
•	XGBoost - Machine learning classifier
•	SHAP - Model explainability
•	scikit-learn - Data preprocessing and scaling
•	NumPy - Numerical computations

Frontend
•	HTML/CSS/JavaScript - User interface
•	Responsive design
•	Interactive visualizations - Progress bars, tooltips, feature cards

Machine Learning Pipeline
•	Dataset: GTZAN (400 samples across 4 genres)(the dataset originally contains 1000 samples across 10 genres but due to unimpressive credibility of the dataset, model had to be trained among 4 genres only)
•	Features: 34 audio characteristics (MFCCs, chroma, spectral contrast, tonnetz, tempo)
•	Model: XGBoost classifier with 82% test accuracy
•	Cross-validation: 79.6% ± 2.6% accuracy
📊 Model Performance
              precision    recall  f1-score   support
   classical       0.76      0.95      0.84        20
       disco       0.80      0.80      0.80        20
        jazz       0.84      0.80      0.82        20
        rock       0.88      0.70      0.78        20

    accuracy                           0.82         80
  	
💻 System Requirements
Minimum
•	OS: Windows 10, macOS 10.14, Ubuntu 18.04
•	RAM: 4GB
•	Storage: 2GB free space
•	Python: 3.8+

Recommended
•	RAM: 8GB (faster processing)
•	Storage: 5GB (for music files)
•	Python: 3.9 or 3.10

📁 Project Structure
MusicEducator/
├── backend.py              # Flask backend with API endpoints
├── requirements.txt        # Python dependencies
├── music_classifier.pkl    # Trained XGBoost model
├── feature_scaler.pkl      # StandardScaler for features
├── shap_explainer.pkl      # SHAP TreeExplainer
├── label_encoder.pkl       # Genre label encoder
└── index.html             # Frontend HTML with embedded CSS/JS

🎵 Supported Audio Formats
•	MP3
•	WAV
•	M4A
•	Maximum file size: 50MB

🔬 Audio Features Analyzed
Rhythm & Timing (1 feature)
•	Tempo: Beats per minute detection
Voice & Instrument Texture (5 features)
•	MFCCs 0-4: Fundamental audio characteristics
Musical Style Characteristics (4 features)
•	MFCCs 5-8: Genre-specific audio patterns
Energy & Dynamics (4 features)
•	MFCCs 9-12: Intensity and dynamic measurements
Musical Notes & Harmony (13 features)
•	Chroma 0-11: Individual musical note presence
•	Tonnetz: Harmonic stability measurement
Sound Color & Texture (7 features)
•	Spectral Contrast 0-6: Frequency distribution analysis

This project demonstrates:
•	Music Information Retrieval techniques
•	Explainable AI in real-world applications
•	Audio signal processing concepts
•	Machine learning for creative domains
•	User experience design for technical concepts

🏆 Success Confirmation
You'll know it's working when:
•	✅ Browser shows the Music Assistant interface
•	✅ File upload area is visible and functional
•	✅ Audio analysis completes without errors
•	✅ Results show genre prediction and feature analysis
•	✅ SHAP analysis displays properly

THE AI WORKFLOW(just for a big picture view):
1.	librosa extracts 34 features from audio
2.	StandardScaler normalizes the features
3.	🤖 XGBoost model (THE AI) takes features → predicts genre
4.	SHAP explains what the XGBoost model was "thinking"
SO WHEN WE SAY:
•	"AI predicts Rock with 73% confidence" = XGBoost model output
•	"AI has 82% accuracy" = XGBoost model performance
•	"AI looks at spectral contrast" = XGBoost learned these features matter

📈 Future Improvements
•	[ ] Expand to more genres(can be implemented with a more trustworthy dataset-would help ensure accuracy doesn't fall)
•	[ ] Add temporal analysis (verse/chorus detection)
•	[ ] Implement deep learning models(by working on a way better dataset next time, so that spectrogram analysis has a good quality and quantity of audio data to learn from!)
•	[ ] Real-time audio streaming analysis
•	[ ] Mobile app development
(Educational project - feel free to learn from and build upon this work. Built with ❤️ for music education😍)
________________________________________








