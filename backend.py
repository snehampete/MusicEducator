from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import shap
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)


# Load your saved models
print("Loading models...")
try:
    model = joblib.load('music_classifier.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    le = label_encoder  # Create alias for shorter name
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# FIXES FOR AUDIO PROCESSING AND SHAP

def extract_music_features(audio_path):
    """Extract 34 music features matching training data exactly - FIXED TEMPO WARNING"""
    try:
        print(f"🎵 Starting feature extraction for: {audio_path}")
        
        # Load audio (30 second clips like your training data)
        y, sr = librosa.load(audio_path, duration=30.0)
        print(f"✅ Audio loaded: {len(y)} samples at {sr} Hz")
        
        # Initialize feature dictionary in exact order
        features = {}
        
        # 1. TEMPO (1 feature) - FIXED THE NUMPY WARNING
        print("⏱️ Extracting tempo...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # FIX: Proper tempo handling to avoid numpy deprecation warning
        if isinstance(tempo, np.ndarray):
            features['tempo'] = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            features['tempo'] = float(tempo)
        
        # 2. MFCC FEATURES (13 features: mfcc_0 to mfcc_12)
        print("🎤 Extracting MFCCs...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
        
        # 3. CHROMA FEATURES (12 features: chroma_0 to chroma_11)
        print("🎹 Extracting chroma...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}'] = float(np.mean(chroma[i]))
        
        # 4. SPECTRAL CONTRAST (7 features: spectral_contrast_0 to spectral_contrast_6)
        print("🌈 Extracting spectral contrast...")
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(7):
            features[f'spectral_contrast_{i}'] = float(np.mean(spectral_contrast[i]))
        
        # 5. TONNETZ (1 feature: tonnetz_0 only)
        print("🎼 Extracting tonnetz...")
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_0'] = float(np.mean(tonnetz[0]))
        
        print(f"🎉 Feature extraction complete! Total features: {len(features)}")
        
        # Verify feature count
        if len(features) != 34:
            print(f"⚠️ Warning: Expected 34 features, got {len(features)}")
        
        return features
        
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

class FeatureTranslator:
    def __init__(self):
        self.feature_ranges = {
            'tempo': (60, 180),
            'spectral_centroid': (500, 4000),
            'spectral_bandwidth': (500, 3000),
            'rms_energy': (0.001, 0.1)
        }
    
    def translate_feature(self, feature_name, value):
        """Convert raw feature to educational format - NOW HANDLES ALL 34 FEATURES"""
        
        # Get visual level (0-100%)
        if feature_name in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_name]
            visual_level = ((value - min_val) / (max_val - min_val)) * 100
            visual_level = max(0, min(100, visual_level))
        else:
            # For MFCCs, chroma, spectral_contrast, tonnetz - normalize around 50%
            visual_level = min(100, max(0, (abs(value) / 20) * 100))  # Reasonable scaling
        
        return {
            'name': feature_name,
            'display_name': self.get_display_name(feature_name),
            'value': value,
            'visual_level': visual_level,
            'interpretation': self.get_interpretation(feature_name, value),
            'explanation': self.get_explanation(feature_name, value),
            'analogy': self.get_analogy(feature_name),
            'listen_tip': self.get_listen_tip(feature_name),
            'confidence': self.get_confidence(feature_name),
            'category': self.get_category(feature_name),
            'unit': self.get_unit(feature_name)
        }
    
    def get_display_name(self, feature_name):
        """COMPLETE DISPLAY NAMES FOR ALL 34 FEATURES"""
        display_names = {
            # Tempo
            'tempo': 'Tempo (Speed)',
            
            # MFCCs - Voice & Instrument Character (13 features)
            'mfcc_0': 'Overall Sound Character',
            'mfcc_1': 'Vocal Warmth',
            'mfcc_2': 'Sound Clarity', 
            'mfcc_3': 'Instrument Texture',
            'mfcc_4': 'Voice Roughness',
            'mfcc_5': 'Sound Depth',
            'mfcc_6': 'Tonal Quality',
            'mfcc_7': 'Voice/Instrument Blend',
            'mfcc_8': 'Sound Smoothness',
            'mfcc_9': 'Harmonic Richness',
            'mfcc_10': 'Audio Brightness',
            'mfcc_11': 'Sound Fullness',
            'mfcc_12': 'Voice/Instrument Detail',
            
            # Chroma - Musical Notes (12 features)
            'chroma_0': 'C Note Presence',
            'chroma_1': 'C# Note Presence', 
            'chroma_2': 'D Note Presence',
            'chroma_3': 'D# Note Presence',
            'chroma_4': 'E Note Presence',
            'chroma_5': 'F Note Presence',
            'chroma_6': 'F# Note Presence',
            'chroma_7': 'G Note Presence',
            'chroma_8': 'G# Note Presence',
            'chroma_9': 'A Note Presence',
            'chroma_10': 'A# Note Presence',
            'chroma_11': 'B Note Presence',
            
            # Spectral Contrast - Sound Color (7 features) - THESE WERE MISSING!
            'spectral_contrast_0': 'Bass vs Treble Balance',
            'spectral_contrast_1': 'Low-Mid Frequency Color',
            'spectral_contrast_2': 'Mid Frequency Richness',
            'spectral_contrast_3': 'Upper-Mid Brightness',
            'spectral_contrast_4': 'High Frequency Sparkle',
            'spectral_contrast_5': 'Extreme High Presence',
            'spectral_contrast_6': 'Overall Frequency Contrast',
            
            # Tonnetz - THESE WERE MISSING TOO!
            'tonnetz_0': 'Harmonic Stability'
        }
        
        return display_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def get_interpretation(self, feature_name, value):
        """COMPLETE INTERPRETATIONS FOR ALL FEATURES"""
        if feature_name == 'tempo':
            if value < 70: return "Very Slow - Ballad pace"
            elif value < 90: return "Slow - Relaxed walking pace" 
            elif value < 110: return "Moderate - Comfortable groove"
            elif value < 140: return "Fast - Energetic and driving"
            else: return "Very Fast - High energy dance pace"
        
        elif 'mfcc' in feature_name:
            if abs(value) < 5: return "Subtle - Gentle characteristic"
            elif abs(value) < 15: return "Moderate - Noticeable quality"
            else: return "Strong - Prominent feature"
        
        elif 'chroma' in feature_name:
            if value < 0.3: return "Weak - Note barely present"
            elif value < 0.6: return "Moderate - Note somewhat present"
            else: return "Strong - Note clearly present"
        
        elif 'spectral_contrast' in feature_name:
            if value < 10: return "Low Contrast - Smooth sound"
            elif value < 20: return "Moderate Contrast - Balanced"
            else: return "High Contrast - Dynamic sound"
        
        elif feature_name == 'tonnetz_0':
            if abs(value) < 0.3: return "Stable - Consonant harmony"
            elif abs(value) < 0.6: return "Moderate - Some tension"
            else: return "Unstable - Dissonant harmony"
        
        else:
            return f"Value: {value:.3f}"
    
    def get_explanation(self, feature_name, value):
        """COMPLETE EXPLANATIONS - FIXED THE MISSING ONES"""
        explanations = {
            # Tempo
            'tempo': f"The beat moves at {value:.1f} BPM - this sets the energy and feel",
            
            # MFCCs - Complete set
            'mfcc_0': "The most important characteristic of how voices and instruments sound",
            'mfcc_1': "How warm and inviting the vocals or lead instruments sound",
            'mfcc_2': "How clear and distinct the sounds are in the mix",
            'mfcc_3': "The unique texture that makes instruments recognizable",
            'mfcc_4': "The roughness or smoothness of vocal and instrument tones",
            'mfcc_5': "How deep and full the overall sound feels",
            'mfcc_6': "The tonal quality that gives music its character",
            'mfcc_7': "How well voices and instruments blend together",
            'mfcc_8': "The smoothness of the audio texture",
            'mfcc_9': "How rich and complex the harmonies sound",
            'mfcc_10': "The brightness and sparkle in the high frequencies",
            'mfcc_11': "How full and complete the sound feels",
            'mfcc_12': "Fine details in voice and instrument characteristics",
            
            # Chroma - Complete set
            'chroma_0': "How much the note C contributes to the overall harmony",
            'chroma_1': "How much the note C# contributes to the overall harmony",
            'chroma_2': "How much the note D contributes to the overall harmony",
            'chroma_3': "How much the note D# contributes to the overall harmony",
            'chroma_4': "How much the note E contributes to the overall harmony",
            'chroma_5': "How much the note F contributes to the overall harmony",
            'chroma_6': "How much the note F# contributes to the overall harmony",
            'chroma_7': "How much the note G contributes to the overall harmony",
            'chroma_8': "How much the note G# contributes to the overall harmony",
            'chroma_9': "How much the note A contributes to the overall harmony",
            'chroma_10': "How much the note A# contributes to the overall harmony", 
            'chroma_11': "How much the note B contributes to the overall harmony",
            
            # Spectral Contrast - THESE WERE COMPLETELY MISSING!
            'spectral_contrast_0': "The difference between bass and treble - creates sonic depth",
            'spectral_contrast_1': "How much variation exists in the low-mid frequencies",
            'spectral_contrast_2': "The richness and complexity in the middle frequencies",
            'spectral_contrast_3': "The brightness and clarity in upper-mid frequencies",
            'spectral_contrast_4': "The sparkle and presence in high frequencies",
            'spectral_contrast_5': "The very high frequency content that adds air and space",
            'spectral_contrast_6': "Overall contrast across all frequency ranges",
            
            # Tonnetz - THIS WAS MISSING TOO!
            'tonnetz_0': "How stable or tense the harmony feels - affects emotional impact"
        }
        
        return explanations.get(feature_name, f"Audio characteristic: {value:.3f}")
    
    def get_analogy(self, feature_name):
        """COMPLETE ANALOGIES - FIXED THE MISSING ONES"""
        analogies = {
            # Tempo
            'tempo': 'Like walking speed - slow stroll vs fast jog',
            
            # MFCCs - Complete set
            'mfcc_0': 'Like the main color in a painting',
            'mfcc_1': 'Like the warmth of someone\'s voice when they speak',
            'mfcc_2': 'Like how clearly you hear someone through a phone',
            'mfcc_3': 'Like the difference between velvet and sandpaper',
            'mfcc_4': 'Like the difference between smooth silk and rough burlap',
            'mfcc_5': 'Like the depth of a swimming pool vs a puddle',
            'mfcc_6': 'Like the difference between a wooden flute and metal trumpet',
            'mfcc_7': 'Like how well ingredients blend in a recipe',
            'mfcc_8': 'Like the difference between smooth peanut butter and chunky',
            'mfcc_9': 'Like a simple melody vs a complex symphony',
            'mfcc_10': 'Like the difference between candlelight and bright sunlight',
            'mfcc_11': 'Like a thin vs thick blanket',
            'mfcc_12': 'Like fine details in a photograph',
            
            # Chroma - Complete set
            'chroma_0': 'Like checking how much red paint is in the color mix',
            'chroma_1': 'Like checking how much orange paint is in the color mix',
            'chroma_2': 'Like checking how much yellow paint is in the color mix',
            'chroma_3': 'Like checking how much green paint is in the color mix',
            'chroma_4': 'Like checking how much blue paint is in the color mix',
            'chroma_5': 'Like checking how much purple paint is in the color mix',
            'chroma_6': 'Like checking how much pink paint is in the color mix',
            'chroma_7': 'Like checking how much brown paint is in the color mix',
            'chroma_8': 'Like checking how much gray paint is in the color mix',
            'chroma_9': 'Like checking how much black paint is in the color mix',
            'chroma_10': 'Like checking how much white paint is in the color mix',
            'chroma_11': 'Like checking how much silver paint is in the color mix',
            
            # Spectral Contrast - THESE WERE COMPLETELY MISSING!
            'spectral_contrast_0': 'Like the difference between thunder (bass) and bird song (treble)',
            'spectral_contrast_1': 'Like comparing a cello to a violin',
            'spectral_contrast_2': 'Like the difference between a piano and a flute',
            'spectral_contrast_3': 'Like comparing a guitar to a piccolo',
            'spectral_contrast_4': 'Like the difference between cymbals and drums',
            'spectral_contrast_5': 'Like the sparkle on top of a Christmas tree',
            'spectral_contrast_6': 'Like the full range from deep ocean to mountain peaks',
            
            # Tonnetz - THIS WAS MISSING!
            'tonnetz_0': 'Like how peaceful vs tense a room feels'
        }
        
        return analogies.get(feature_name, 'Audio measurement that affects how music sounds')
    
    def get_listen_tip(self, feature_name):
        """COMPLETE LISTEN TIPS - FIXED THE MISSING ONES"""
        tips = {
            # Tempo
            'tempo': 'Tap your foot to the beat - notice the speed',
            
            # MFCCs - Complete set
            'mfcc_0': 'Focus on the overall character of voices and instruments',
            'mfcc_1': 'Listen for how warm or cold the vocals sound',
            'mfcc_2': 'Notice how clear or muddy the sound is',
            'mfcc_3': 'Focus on what makes each instrument unique',
            'mfcc_4': 'Listen for rough vs smooth vocal textures',
            'mfcc_5': 'Feel how deep and full the sound is',
            'mfcc_6': 'Notice the unique tonal character',
            'mfcc_7': 'Listen for how well everything blends together',
            'mfcc_8': 'Focus on the smoothness of the overall texture',
            'mfcc_9': 'Listen for the richness and complexity',
            'mfcc_10': 'Notice the brightness in the high notes',
            'mfcc_11': 'Feel how full and complete the sound is',
            'mfcc_12': 'Listen for fine details in the instruments',
            
            # Chroma - Complete set
            'chroma_0': 'Try humming a C note while listening',
            'chroma_1': 'Try humming a C# note while listening',
            'chroma_2': 'Try humming a D note while listening',
            'chroma_3': 'Try humming a D# note while listening',
            'chroma_4': 'Try humming an E note while listening',
            'chroma_5': 'Try humming an F note while listening',
            'chroma_6': 'Try humming an F# note while listening',
            'chroma_7': 'Try humming a G note while listening',
            'chroma_8': 'Try humming a G# note while listening',
            'chroma_9': 'Try humming an A note while listening',
            'chroma_10': 'Try humming an A# note while listening',
            'chroma_11': 'Try humming a B note while listening',
            
            # Spectral Contrast - THESE WERE COMPLETELY MISSING!
            'spectral_contrast_0': 'Notice the balance between deep bass and bright highs',
            'spectral_contrast_1': 'Focus on the richness in the lower-middle frequencies',
            'spectral_contrast_2': 'Listen to the fullness in the middle range',
            'spectral_contrast_3': 'Notice the clarity in the upper-middle frequencies',
            'spectral_contrast_4': 'Listen for the sparkle in the high frequencies',
            'spectral_contrast_5': 'Focus on the very highest sounds and "air"',
            'spectral_contrast_6': 'Listen to the full frequency spectrum from low to high',
            
            # Tonnetz - THIS WAS MISSING!
            'tonnetz_0': 'Feel whether the harmony is peaceful or creates tension'
        }
        
        return tips.get(feature_name, 'Listen carefully to this musical element')
    
    def get_confidence(self, feature_name):
        """Updated confidence levels"""
        high_confidence = ['tempo']
        medium_confidence = ['mfcc_0', 'mfcc_1', 'mfcc_2', 'spectral_contrast_0', 'spectral_contrast_1']
        
        if feature_name in high_confidence:
            return {'level': 'High', 'description': 'Very reliable measurement'}
        elif feature_name in medium_confidence:
            return {'level': 'Medium', 'description': 'Good estimate'}
        else:
            return {'level': 'Moderate', 'description': 'Reasonable estimate'}
    
    def get_category(self, feature_name):
        """BALANCED DISTRIBUTION ACROSS ALL 6 CATEGORIES"""
    
        if feature_name == 'tempo':
            return 'rhythm'   # Rhythm & Timing
    
        elif feature_name in ['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4']:
            return 'texture'  # Voice & Instrument Texture (5 features)
    
        elif feature_name in ['mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8']:
            return 'style'    # Musical Style Characteristics (4 features)
    
        elif feature_name in ['mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12']:
            return 'energy'   # Energy & Dynamics (4 features)
    
        elif 'chroma' in feature_name:
            return 'harmony'  # Musical Notes & Harmony (12 features)
    
        elif 'spectral_contrast' in feature_name:
            return 'timbre'   # Sound Color & Texture (7 features)
    
        elif feature_name == 'tonnetz_0':
            return 'harmony'  # Add to harmony (1 feature)
    
        else:
            return 'energy'
    
    def get_unit(self, feature_name):
        """Units for features"""
        units = {
            'tempo': 'BPM'
        }
        return units.get(feature_name, 'coefficient')

# Initialize translator
translator = FeatureTranslator()

@app.route('/')
def index():
        """Serve the HTML file"""
        try:
            with open('index.html', 'r', encoding='utf-8') as f:
                        return f.read()
        except Exception as e:
            return f"Error loading index.html: {str(e)}<br>Current directory: {os.getcwd()}<br>Files: {os.listdir('.')}"


            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            print(f"💾 Saving as: {filename}")
            filepath = os.path.join('temp_audio.wav')
            file.save(filepath)
            print(f"✅ File saved to: {filepath}")

            # Extract features
            print("🔧 Extracting features...")
            features = extract_music_features(filepath)
            print(f"✅ Features extracted: {type(features)}")
            
            if features is None:
                print("❌ Feature extraction failed")
                return jsonify({'error': 'Could not extract features from audio file'}), 500

            print(f"🎯 Feature count: {len(features) if hasattr(features, '__len__') else 'Unknown'}")
            
            # Continue with rest of your code...
            # (Add more print statements as needed)
                    
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join('temp_audio.wav')
            file.save(filepath)
            
            # Extract features
            print("Extracting features...")
            features = extract_music_features(filepath)
            
            if features is None:
                return jsonify({'error': 'Could not extract features from audio file'}), 500
            
            # Prepare features for model (same order as training)
            feature_names = list(features.keys())
            feature_values = np.array([features[name] for name in feature_names]).reshape(1, -1)
            
            # Scale features
            print("Scaling features...")
            feature_values_scaled = scaler.transform(feature_values)
            
            # Get predictions
            print("Making predictions...")
            prediction_proba = model.predict_proba(feature_values_scaled)[0]
            prediction = model.predict(feature_values_scaled)[0]
            
            # Get genre names
            genre_names = label_encoder.classes_
            
            # Get SHAP values
            print("Computing SHAP values...")
            shap_values = explainer.shap_values(feature_values_scaled)
            
            # COMPLETE TRANSLATION FUNCTION - FIXED!
            # Translate features to educational format
            print("🔄 Translating features...")
            translated_features = {}
            failed_translations = []

            for feature_name, value in features.items():
                try:
                        print(f"🔧 Translating: {feature_name}")
                        translated_features[feature_name] = translator.translate_feature(feature_name, value)
                        print(f"✅ Success: {feature_name}")
                except Exception as e:
                        print(f"❌ Failed to translate {feature_name}: {e}")
                        failed_translations.append(feature_name)
                        continue

            print(f"🔍 Successfully translated {len(translated_features)} out of {len(features)} features")

            # Optional: Show which ones failed (for debugging)
            if failed_translations:
                    print(f"❌ Failed translations: {failed_translations}")
            else:
                    print("🎉 All features translated successfully!")

            # Group by category (only with successfully translated features)
            categories = {}
            for feature_name, feature_data in translated_features.items():
                category = feature_data['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(feature_data)

            print("🔍 DEBUG - Final categories:")
            for category, features_list in categories.items():
                print(f"  {category}: {len(features_list)} features")
            
            
                # Clean up temp file
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Return results
            results = {
                        'success': True,
                        'genre_prediction': {
                            'primary_genre': genre_names[prediction],
                            'probabilities': {genre_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}
                        },
                        'features_by_category': categories,
                        'shap_analysis': {
                            'feature_importance': {feature_names[i]: float(shap_values[0][i]) for i in range(len(feature_names))}
                        }
                    }
                        
            print("✅ Analysis complete!")
            return jsonify(results)
                
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500



# FIXED ANALYZE FUNCTION - PROPER SHAP HANDLING
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file - FIXED ALL ISSUES"""
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join('temp_audio.wav')
        file.save(filepath)
        
        # Extract features
        print("🔧 Extracting features...")
        features = extract_music_features(filepath)

        # DEBUG: Print all extracted features
        print("🔍 DEBUG - All extracted features:")
        for i, (name, value) in enumerate(features.items()):
            print(f"  {i+1:2d}. {name}: {value}")

        print(f"🔍 Total features extracted: {len(features)}")
        
        if features is None:
            return jsonify({'error': 'Could not extract features from audio file'}), 500
        
        # Prepare features for model (same order as training)
        feature_names = list(features.keys())
        feature_values = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features
        print("📏 Scaling features...")
        feature_values_scaled = scaler.transform(feature_values)
        
        # Get predictions
        print("🎯 Making predictions...")
        prediction_proba = model.predict_proba(feature_values_scaled)[0]
        prediction = model.predict(feature_values_scaled)[0]
        
        # Get genre names
        genre_names = label_encoder.classes_
        
        # FIXED: Translate features to educational format - NOW WITH ERROR HANDLING
        print("🔄 Translating features...")
        translated_features = {}
        failed_translations = []
        
        for feature_name, value in features.items():
            try:
                print(f"🔧 Translating: {feature_name}")
                translated_features[feature_name] = translator.translate_feature(feature_name, value)
                print(f"✅ Success: {feature_name}")
            except Exception as e:
                print(f"❌ Failed to translate {feature_name}: {e}")
                failed_translations.append(feature_name)
                continue

        print(f"🔍 Successfully translated {len(translated_features)} out of {len(features)} features")
        if failed_translations:
            print(f"❌ Failed translations: {failed_translations}")

        # FIXED: Group by category (only with successfully translated features)
        categories = {}
        for feature_name, feature_data in translated_features.items():
            category = feature_data['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(feature_data)

        print("🔍 DEBUG - Final categories:")
        for category, features_list in categories.items():
            print(f"  {category}: {len(features_list)} features")

        # FIXED: Get SHAP values with proper array handling
        print("🧠 Computing SHAP values...")
        shap_values = explainer.shap_values(feature_values_scaled)
        
        # Create feature names list in correct order (same as extraction)
        feature_names_list = list(features.keys())
        
        # FIXED: Handle SHAP array properly
        if isinstance(shap_values, list):
            # Multi-class case: shap_values is list of arrays for each class
            shap_vals = shap_values[prediction]  # Get values for predicted class
            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]  # Take first sample if 2D
        else:
            # Single array case
            if shap_values.ndim == 3:
                # Shape (samples, features, classes)
                shap_vals = shap_values[0, :, prediction]
            elif shap_values.ndim == 2:
                # Shape (samples, features)
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

        print(f"🔍 Using SHAP values for class {prediction} ({genre_names[prediction]})")
        print(f"🔍 SHAP values shape: {shap_vals.shape}")

        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)

        # FIXED: Return results with working SHAP and all translated features
        results = {
            'success': True,
            'genre_prediction': {
                'primary_genre': genre_names[prediction],
                'probabilities': {genre_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}
            },
            'features_by_category': categories,
            'shap_analysis': {
                'feature_importance': {feature_names_list[i]: float(shap_vals[i]) for i in range(len(feature_names_list))}
            },
            'debug_info': {
                'total_features_extracted': len(features),
                'total_features_translated': len(translated_features),
                'failed_translations': failed_translations,
                'categories_found': list(categories.keys())
            }
        }
        
        print("✅ Analysis complete!")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🎵 Music Feature Explorer Backend Starting...")
    print("Make sure your index.html is in the same folder!")
    app.run(debug=True, port=5000, use_reloader=False)