# filename: app.py
from flask import Flask, request, jsonify
import os
import requests
from flask_cors import CORS
import re
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# --- Student Data Storage ---
STUDENT_DATA_FILE = "student_data.json"
INTERACTIONS_FILE = "student_interactions.json"

class StudentDataModel:
    """Stores and manages student learning data"""
    
    def __init__(self):
        self.student_profile = self.load_profile()
        self.interactions = self.load_interactions()
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.interaction_vectors = []
        self.rebuild_vectors()
    
    def load_profile(self):
        """Load student profile from file"""
        if os.path.exists(STUDENT_DATA_FILE):
            with open(STUDENT_DATA_FILE, 'r') as f:
                return json.load(f)
        return {
            "topics_discussed": defaultdict(int),
            "difficulty_preferences": {},
            "question_types": defaultdict(int),
            "learning_patterns": [],
            "total_interactions": 0,
            "first_interaction": None,
            "last_interaction": None
        }
    
    def load_interactions(self):
        """Load interaction history"""
        if os.path.exists(INTERACTIONS_FILE):
            with open(INTERACTIONS_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def save_profile(self):
        """Save student profile to file"""
        with open(STUDENT_DATA_FILE, 'w') as f:
            json.dump(self.student_profile, f, indent=2)
    
    def save_interactions(self):
        """Save interactions to file"""
        with open(INTERACTIONS_FILE, 'w') as f:
            json.dump(self.interactions, f, indent=2)
    
    def rebuild_vectors(self):
        """Rebuild TF-IDF vectors from interactions"""
        if len(self.interactions) > 0:
            texts = [f"{i['question']} {i['answer']}" for i in self.interactions]
            try:
                self.interaction_vectors = self.vectorizer.fit_transform(texts)
            except:
                self.interaction_vectors = []
    
    def add_interaction(self, question, answer, topics=None):
        """Store new interaction and update profile"""
        timestamp = datetime.now().isoformat()
        
        # Store interaction
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "topics": topics or []
        }
        self.interactions.append(interaction)
        
        # Update profile
        if self.student_profile["first_interaction"] is None:
            self.student_profile["first_interaction"] = timestamp
        self.student_profile["last_interaction"] = timestamp
        self.student_profile["total_interactions"] += 1
        
        # Extract topics (simple keyword extraction)
        keywords = self.extract_keywords(question)
        for keyword in keywords:
            self.student_profile["topics_discussed"][keyword] = \
                self.student_profile["topics_discussed"].get(keyword, 0) + 1
        
        # Determine question type
        question_type = self.classify_question(question)
        self.student_profile["question_types"][question_type] = \
            self.student_profile["question_types"].get(question_type, 0) + 1
        
        # Save to files
        self.save_profile()
        self.save_interactions()
        
        # Rebuild vectors
        self.rebuild_vectors()
        
        print(f"✅ Stored interaction. Total: {len(self.interactions)}")
    
    def extract_keywords(self, text):
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why', 
                      'does', 'did', 'can', 'could', 'would', 'should', 'this', 
                      'that', 'there', 'their', 'they', 'have', 'has', 'been'}
        return [w for w in words if w not in stop_words][:5]
    
    def classify_question(self, question):
        """Classify question type"""
        question_lower = question.lower()
        if any(word in question_lower for word in ['explain', 'what is', 'define']):
            return "concept_explanation"
        elif any(word in question_lower for word in ['how to', 'how do', 'how can']):
            return "procedural"
        elif any(word in question_lower for word in ['why', 'reason']):
            return "reasoning"
        elif '?' in question:
            return "general_question"
        else:
            return "statement"
    
    def find_similar_interactions(self, question, top_k=3):
        """Find similar past interactions using RAG"""
        if len(self.interactions) == 0:
            return []
        
        try:
            # Vectorize current question
            question_vector = self.vectorizer.transform([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.interaction_vectors)[0]
            
            # Get top_k most similar
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Similarity threshold
                    similar.append({
                        "question": self.interactions[idx]["question"],
                        "answer": self.interactions[idx]["answer"],
                        "similarity": float(similarities[idx])
                    })
            
            return similar
        except:
            return []
    
    def get_student_context(self):
        """Generate context about student for the AI"""
        if self.student_profile["total_interactions"] == 0:
            return "This is a new student with no interaction history."
        
        # Get top topics
        topics = dict(self.student_profile["topics_discussed"])
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_str = ", ".join([f"{t[0]} ({t[1]}x)" for t in top_topics])
        
        # Get question preferences
        q_types = dict(self.student_profile["question_types"])
        main_type = max(q_types.items(), key=lambda x: x[1])[0] if q_types else "varied"
        
        context = f"""Student Context:
- Total interactions: {self.student_profile["total_interactions"]}
- Main topics: {topic_str}
- Preferred question style: {main_type}
- Learning since: {self.student_profile["first_interaction"][:10]}

Based on this history, adjust your teaching style accordingly."""
        
        return context

# Initialize student data model
student_model = StudentDataModel()

# --- Utils ---
def clean_text(text):
    """Remove non-ASCII characters from text"""
    return re.sub(r'[^\x00-\x7F]+', '', text)

# --- OpenRouter setup ---
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY", 
    "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"
)
MODEL = "anthropic/claude-3.5-sonnet"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

base_chatbot_profile = """You are Bob, a friendly and supportive AI tutor.
Explain clearly in short paragraphs and a warm, encouraging tone."""

def ask_openrouter(messages):
    """Send request to OpenRouter API with RAG context"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "AI Tutor"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise

# --- Routes ---
@app.route('/ask', methods=['POST'])
def ask():
    """Handle chatbot questions with RAG"""
    data = request.json
    user_message = data.get("question", "").strip()
    
    if not user_message:
        return jsonify({"error": "No question provided"}), 400
    
    print(f"\n📝 Question: {user_message}")
    
    try:
        # Get student context
        student_context = student_model.get_student_context()
        
        # Find similar past interactions (RAG)
        similar_interactions = student_model.find_similar_interactions(user_message)
        
        # Build RAG context
        rag_context = ""
        if similar_interactions:
            rag_context = "\n\nPrevious related discussions:\n"
            for i, sim in enumerate(similar_interactions, 1):
                rag_context += f"{i}. Q: {sim['question']}\n   A: {sim['answer'][:100]}...\n"
        
        # Build enhanced prompt with student context and RAG
        enhanced_profile = f"""{base_chatbot_profile}

{student_context}
{rag_context}

Remember previous discussions and build upon them naturally."""
        
        messages = [
            {"role": "system", "content": enhanced_profile},
            {"role": "user", "content": user_message}
        ]
        
        # Get AI response
        answer = ask_openrouter(messages)
        cleaned_answer = clean_text(answer)
        
        # Store interaction
        keywords = student_model.extract_keywords(user_message)
        student_model.add_interaction(user_message, cleaned_answer, keywords)
        
        print(f"✅ Response generated and stored")
        print(f"📚 Found {len(similar_interactions)} similar interactions")
        if similar_interactions:
            for sim in similar_interactions:
                print(f"   - '{sim['question'][:50]}...' (similarity: {sim['similarity']:.2f})")
        
        return jsonify({
            "answer": cleaned_answer,
            "similar_count": len(similar_interactions),
            "total_interactions": student_model.student_profile["total_interactions"],
            "rag_details": similar_interactions  # Include for debugging
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/student-profile', methods=['GET'])
def get_student_profile():
    """Get student learning profile"""
    profile = dict(student_model.student_profile)
    # Convert defaultdicts to regular dicts for JSON serialization
    profile['topics_discussed'] = dict(profile.get('topics_discussed', {}))
    profile['question_types'] = dict(profile.get('question_types', {}))
    return jsonify(profile)

@app.route('/interactions', methods=['GET'])
def get_interactions():
    """Get all interactions"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "total": len(student_model.interactions),
        "recent": student_model.interactions[-limit:]
    })

@app.route('/reset-student', methods=['POST'])
def reset_student():
    """Reset student data (for testing)"""
    if os.path.exists(STUDENT_DATA_FILE):
        os.remove(STUDENT_DATA_FILE)
    if os.path.exists(INTERACTIONS_FILE):
        os.remove(INTERACTIONS_FILE)
    global student_model
    student_model = StudentDataModel()
    return jsonify({"message": "Student data reset successfully"})

@app.route('/')
def home():
    """Health check endpoint"""
    return f"""AI Tutor Backend Running ✅
    <br>Total Interactions: {student_model.student_profile['total_interactions']}
    <br><a href='/student-profile'>View Profile</a>
    <br><a href='/interactions'>View Interactions</a>"""

if __name__ == '__main__':
    print("🚀 Starting AI Tutor with RAG...")
    print(f"📊 Loaded {len(student_model.interactions)} previous interactions")
    print(f"📁 Data stored in: {os.path.abspath(STUDENT_DATA_FILE)}")
    app.run(host='127.0.0.1', port=5000, debug=True)