import os
import time
import pandas as pd
import base64
import io
import google.generativeai as genai
from flask import Flask, render_template_string, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# ==========================================
# ⚙️ CONFIGURATION & AI SETUP
# ==========================================

# Replace with your actual API Key or use Environment Variables
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)

def get_best_available_model():
    """Selects the most stable Gemini model available in the current environment."""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priority_list = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for model in priority_list:
            if model in available_models:
                return model
        return available_models[0]
    except Exception:
        return "gemini-1.5-flash"

SELECTED_MODEL_NAME = get_best_available_model()
ai_model = genai.GenerativeModel(SELECTED_MODEL_NAME)

app = Flask(__name__)

# ==========================================
# 🧠 MACHINE LEARNING LOGIC (Predictor)
# ==========================================

def initialize_ml_engine():
    """Initializes and trains a simple RandomForest model for grade prediction."""
    training_data = {
        'Attendance': [95, 80, 70, 60, 90, 85, 65],
        'Prev_Score': [88, 70, 60, 50, 85, 75, 55],
        'Study_Hours': [3.0, 1.5, 0.5, 0.2, 2.5, 2.0, 0.4],
        'Test_Att': [100, 70, 60, 50, 90, 80, 55],
        'Chap_Comp': [95, 75, 50, 40, 90, 80, 50],
        'Final_Grade': [94, 77, 65, 52, 90, 82, 58] 
    }
    df = pd.DataFrame(training_data)
    X = df.drop('Final_Grade', axis=1)
    y = df['Final_Grade']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

ml_predictor = initialize_ml_engine()

def safe_ai_generate(content_parts):
    """Wrapper for AI generation with basic Rate Limit (429) handling."""
    try:
        response = ai_model.generate_content(content_parts)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ AI is currently busy (Rate Limit reached). Please try again in a few moments."
        return f"Error: {str(e)}"

# ==========================================
# 🎨 UI TEMPLATE (Tailwind CSS)
# ==========================================

UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StudyBuddy AI | Intelligent Learning Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/all.min.css">
    <style>
        body { background: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(255, 255, 255, 0.02); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .sidebar-btn.active { background: #4f46e5; color: white; box-shadow: 0 0 20px rgba(79, 70, 229, 0.4); }
    </style>
</head>
<body class="h-screen flex p-4 gap-4 overflow-hidden">

    <div class="w-64 glass rounded-3xl p-6 flex flex-col gap-3">
        <h1 class="text-xl font-bold text-center text-indigo-500 mb-8 tracking-widest">STUDYBUDDY</h1>
        <button onclick="switchTab('chat')" id="btn-chat" class="sidebar-btn active w-full p-4 rounded-xl flex items-center gap-3 transition-all">
            <i class="fas fa-robot"></i> AI Mentor
        </button>
        <button onclick="switchTab('ml')" id="btn-ml" class="sidebar-btn w-full p-4 rounded-xl flex items-center gap-3 transition-all">
            <i class="fas fa-chart-line"></i> Performance
        </button>
        <button onclick="switchTab('hand')" id="btn-hand" class="sidebar-btn w-full p-4 rounded-xl flex items-center gap-3 transition-all">
            <i class="fas fa-file-signature"></i> Writing Lab
        </button>
        <div class="mt-auto p-4 bg-white/5 rounded-xl text-[10px] text-center text-gray-500">
            Running on: """ + SELECTED_MODEL_NAME + """
        </div>
    </div>

    <div class="flex-1 glass rounded-3xl overflow-hidden flex flex-col">
        
        <div id="tab-chat" class="tab-content flex flex-col h-full p-8">
            <div id="chat-box" class="flex-1 overflow-y-auto space-y-4 text-sm px-2"></div>
            <div class="mt-4 flex gap-2 p-2 bg-white/5 rounded-xl border border-white/10">
                <input type="text" id="chat-input" placeholder="Ask your mentor anything..." class="flex-1 bg-transparent px-4 outline-none">
                <button onclick="sendChat()" class="bg-indigo-600 hover:bg-indigo-500 px-6 py-2 rounded-lg font-bold transition-colors">SEND</button>
            </div>
        </div>

        <div id="tab-ml" class="tab-content hidden h-full p-10">
            <h2 class="text-2xl font-bold mb-2 text-indigo-400">Success Predictor</h2>
            <p class="text-gray-400 mb-8">Enter your data to predict final grades using Random Forest ML.</p>
            <div class="grid grid-cols-2 gap-8">
                <div class="space-y-4">
                    <input type="number" id="f1" placeholder="Attendance %" class="w-full bg-white/5 p-4 rounded-xl border border-white/10 focus:border-indigo-500 outline-none">
                    <input type="number" id="f2" placeholder="Previous Score" class="w-full bg-white/5 p-4 rounded-xl border border-white/10 focus:border-indigo-500 outline-none">
                    <input type="number" id="f3" placeholder="Daily Study Hours" class="w-full bg-white/5 p-4 rounded-xl border border-white/10 focus:border-indigo-500 outline-none">
                    <button onclick="runML()" class="w-full bg-indigo-600 py-4 rounded-xl font-bold hover:bg-indigo-500 transition-all">CALCULATE SCORE</button>
                </div>
                <div id="ml-result" class="glass rounded-2xl flex flex-col items-center justify-center p-6 text-center border-dashed border-2 border-white/10 text-gray-500">
                    <i class="fas fa-calculator mb-4 text-3xl"></i>
                    <p class="uppercase text-xs tracking-widest">Awaiting Input Data</p>
                </div>
            </div>
        </div>

        <div id="tab-hand" class="tab-content hidden h-full p-10 overflow-y-auto">
            <div class="max-w-2xl mx-auto text-center">
                <h2 class="text-3xl font-black text-indigo-500 mb-2 italic">Writing Expert AI</h2>
                <p class="text-gray-400 mb-8">Upload a photo of your handwriting for structural analysis.</p>
                <div class="p-10 border-2 border-dashed border-indigo-500/30 rounded-[2.5rem] bg-indigo-600/5 flex flex-col items-center gap-6">
                    <input type="file" id="handwriting-file" accept="image/*" class="text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500">
                    <button onclick="analyzeHandwriting()" id="hand-btn" class="bg-indigo-600 px-10 py-4 rounded-2xl font-bold hover:shadow-lg hover:shadow-indigo-500/20 transition-all">ANALYZE IMAGE</button>
                </div>
                <div id="hand-output" class="mt-6 hidden p-6 bg-white/5 rounded-2xl border border-white/10 text-left text-sm whitespace-pre-wrap leading-relaxed"></div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.sidebar-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.remove('hidden');
            document.getElementById('btn-' + tabName).classList.add('active');
        }

        async function sendChat() {
            const input = document.getElementById('chat-input');
            const box = document.getElementById('chat-box');
            if(!input.value) return;
            
            box.innerHTML += `<div class="bg-indigo-600/30 p-4 rounded-2xl ml-auto max-w-[80%] self-end border border-indigo-500/20">${input.value}</div>`;
            const userMsg = input.value; 
            input.value = '';
            
            const response = await fetch('/api/chat', {
                method:'POST', 
                headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({message: userMsg})
            });
            const data = await response.json();
            box.innerHTML += `<div class="bg-white/5 p-4 rounded-2xl max-w-[90%] border border-white/10"><b>StudyBuddy:</b><br>${data.reply}</div>`;
            box.scrollTop = box.scrollHeight;
        }

        async function runML() {
            const res = await fetch('/api/predict', {
                method:'POST', 
                headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({
                    f1: document.getElementById('f1').value, 
                    f2: document.getElementById('f2').value, 
                    f3: document.getElementById('f3').value
                })
            });
            const data = await res.json();
            document.getElementById('ml-result').innerHTML = `
                <h3 class="text-6xl font-black text-white">${data.score}%</h3>
                <p class="font-bold text-indigo-400 mt-2">ESTIMATED GRADE: ${data.grade}</p>
            `;
        }

        async function analyzeHandwriting() {
            const file = document.getElementById('handwriting-file').files[0];
            if(!file) return alert("Please select an image first.");
            
            const btn = document.getElementById('hand-btn');
            const output = document.getElementById('hand-output');
            
            btn.innerHTML = "<i class='fas fa-spinner fa-spin mr-2'></i> Analyzing..."; 
            btn.disabled = true;
            output.classList.add('hidden');

            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = async () => {
                const res = await fetch('/api/analyze-handwriting', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({ image: reader.result.split(',')[1] })
                });
                const data = await res.json();
                output.classList.remove('hidden');
                output.innerHTML = data.feedback;
                btn.innerHTML = "ANALYZE IMAGE"; 
                btn.disabled = false;
            };
        }
    </script>
</body>
</html>
"""

# ==========================================
# 🔌 BACKEND API ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template_string(UI_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    prompt = f"Context: You are StudyBuddy, a helpful AI academic mentor. User says: {user_message}"
    response_text = safe_ai_generate(prompt)
    return jsonify({"reply": response_text})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Mocking fixed values for Test Attendance and Chapter Completion for simplicity
        features = [[float(data['f1']), float(data['f2']), float(data['f3']), 85, 85]]
        prediction = ml_predictor.predict(features)[0]
        grade = "A" if prediction >= 80 else "B" if prediction >= 60 else "C"
        return jsonify({"score": round(prediction, 2), "grade": grade})
    except Exception:
        return jsonify({"score": 0, "grade": "N/A"})

@app.route('/api/analyze-handwriting', methods=['POST'])
def analyze_handwriting():
    image_b64 = request.json.get('image')
    try:
        image_part = {"mime_type": "image/jpeg", "data": image_b64}
        prompt = "Analyze this handwriting. Check for legibility, character spacing, and alignment. Provide constructive feedback."
        response_text = safe_ai_generate([prompt, image_part])
        return jsonify({"feedback": response_text})
    except Exception:
        return jsonify({"feedback": "Analysis failed. Ensure the image is clear and try again."})

if __name__ == '__main__':
    app.run(debug=True)
