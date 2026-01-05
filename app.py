from flask import Flask, request, jsonify
import os
import requests
from flask_cors import CORS
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)

# --- Dataset RAG Setup ---
DATASET_FILE = "train.csv"

class DatasetRetriever:
    """Retrieves relevant rows from a CSV dataset for RAG"""
    
    def __init__(self, csv_path):
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.text_corpus = self.build_text_corpus()
            self.vectorizer = TfidfVectorizer(max_features=500)
            self.embeddings = self.vectorizer.fit_transform(self.text_corpus)
            print(f"📘 Loaded dataset with {len(self.df)} rows.")
        else:
            print("⚠️ Dataset not found, skipping dataset RAG.")
            self.df = None
    
    def build_text_corpus(self):
        """Combine all column text per row into one searchable string"""
        corpus = []
        for _, row in self.df.iterrows():
            row_text = " ".join(str(v) for v in row.values)
            corpus.append(row_text)
        return corpus
    
    def search(self, query, top_k=3):
        """Return top_k most similar rows to the query"""
        if self.df is None:
            return []
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    row_data = self.df.iloc[idx].to_dict()
                    results.append({
                        "similarity": float(similarities[idx]),
                        "data": row_data
                    })
            return results
        except Exception as e:
            print(f"Dataset search error: {e}")
            return []

dataset_retriever = DatasetRetriever(DATASET_FILE)


# --- NEW: Advanced Learning Analytics Models ---

class LearningStyleAnalyzer:
    """Analyzes and tracks student's learning style preferences"""
    
    LEARNING_STYLES = {
        "visual": ["show", "see", "picture", "diagram", "graph", "chart", "image", "visualize"],
        "auditory": ["explain", "tell", "describe", "sound", "hear", "listen", "say"],
        "kinesthetic": ["practice", "do", "hands-on", "try", "example", "apply", "exercise"],
        "reading_writing": ["read", "write", "note", "text", "document", "article", "list"]
    }
    
    def __init__(self):
        self.style_scores = defaultdict(float)
        self.recent_indicators = deque(maxlen=20)
    
    def analyze_question(self, question):
        """Detect learning style indicators in question"""
        question_lower = question.lower()
        detected_styles = []
        
        for style, keywords in self.LEARNING_STYLES.items():
            for keyword in keywords:
                if keyword in question_lower:
                    self.style_scores[style] += 1
                    detected_styles.append(style)
                    break
        
        if detected_styles:
            self.recent_indicators.append({
                "timestamp": datetime.now().isoformat(),
                "styles": detected_styles
            })
        
        return detected_styles
    
    def get_dominant_style(self):
        """Get the student's dominant learning style"""
        if not self.style_scores:
            return "balanced"
        
        return max(self.style_scores.items(), key=lambda x: x[1])[0]
    
    def get_style_distribution(self):
        """Get distribution of learning styles"""
        total = sum(self.style_scores.values())
        if total == 0:
            return {}
        
        return {
            style: round((count / total) * 100, 1)
            for style, count in self.style_scores.items()
        }


class CognitiveLoadTracker:
    """Tracks cognitive load and question complexity patterns"""
    
    def __init__(self):
        self.question_complexities = deque(maxlen=50)
        self.response_times = deque(maxlen=50)
        self.confusion_indicators = 0
        self.mastery_indicators = 0
    
    def analyze_question_complexity(self, question):
        """Estimate question complexity"""
        complexity_score = 0
        
        # Length factor
        word_count = len(question.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Complexity indicators
        complex_words = ["complex", "advanced", "detailed", "comprehensive", "intricate"]
        confusion_words = ["confused", "don't understand", "unclear", "lost", "stuck"]
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in complex_words):
            complexity_score += 2
        
        if any(word in question_lower for word in confusion_words):
            self.confusion_indicators += 1
            complexity_score += 1
        
        # Multi-part questions
        if question.count('?') > 1 or any(word in question_lower for word in ['and', 'also', 'additionally']):
            complexity_score += 1
        
        self.question_complexities.append({
            "timestamp": datetime.now().isoformat(),
            "score": complexity_score
        })
        
        return complexity_score
    
    def get_average_complexity(self):
        """Get average question complexity"""
        if not self.question_complexities:
            return 0
        
        return sum(q["score"] for q in self.question_complexities) / len(self.question_complexities)
    
    def detect_struggle_pattern(self):
        """Detect if student is struggling"""
        if len(self.question_complexities) < 5:
            return False
        
        recent_confusion = self.confusion_indicators > 3
        high_complexity = self.get_average_complexity() > 2
        
        return recent_confusion or high_complexity


class EngagementAnalyzer:
    """Analyzes student engagement patterns"""
    
    def __init__(self):
        self.interaction_timestamps = deque(maxlen=100)
        self.session_durations = []
        self.question_depth_scores = deque(maxlen=30)
        self.follow_up_rate = 0
        self.total_questions = 0
        self.follow_up_questions = 0
    
    def log_interaction(self, question):
        """Log interaction timestamp and analyze engagement"""
        now = datetime.now()
        self.interaction_timestamps.append(now)
        self.total_questions += 1
        
        # Analyze question depth
        depth_score = self._analyze_question_depth(question)
        self.question_depth_scores.append(depth_score)
        
        # Detect follow-up questions
        if self._is_follow_up(question):
            self.follow_up_questions += 1
        
        self.follow_up_rate = self.follow_up_questions / self.total_questions if self.total_questions > 0 else 0
    
    def _analyze_question_depth(self, question):
        """Analyze how deep/thoughtful the question is"""
        depth_score = 0
        question_lower = question.lower()
        
        # Deep thinking indicators
        deep_words = ["why", "how does", "what if", "explain", "relationship", "compare", "analyze"]
        shallow_words = ["what is", "define", "list"]
        
        if any(word in question_lower for word in deep_words):
            depth_score += 2
        elif any(word in question_lower for word in shallow_words):
            depth_score += 1
        
        # Length as depth indicator
        if len(question.split()) > 15:
            depth_score += 1
        
        return depth_score
    
    def _is_follow_up(self, question):
        """Detect if question is a follow-up"""
        follow_up_indicators = [
            "also", "additionally", "what about", "how about", 
            "can you explain more", "tell me more", "further", "elaborate"
        ]
        
        return any(indicator in question.lower() for indicator in follow_up_indicators)
    
    def get_engagement_level(self):
        """Calculate overall engagement level"""
        if self.total_questions < 3:
            return "warming_up"
        
        avg_depth = sum(self.question_depth_scores) / len(self.question_depth_scores) if self.question_depth_scores else 0
        
        if avg_depth > 2 and self.follow_up_rate > 0.3:
            return "highly_engaged"
        elif avg_depth > 1 or self.follow_up_rate > 0.2:
            return "engaged"
        else:
            return "passive"
    
    def get_interaction_frequency(self):
        """Calculate questions per session"""
        if len(self.interaction_timestamps) < 2:
            return "new_student"
        
        recent_interactions = len([
            ts for ts in self.interaction_timestamps 
            if datetime.now() - ts < timedelta(hours=1)
        ])
        
        if recent_interactions > 10:
            return "intensive"
        elif recent_interactions > 5:
            return "active"
        else:
            return "casual"


class KnowledgeGraphTracker:
    """Tracks student's knowledge progression through topics"""
    
    def __init__(self):
        self.topic_mastery = defaultdict(lambda: {
            "exposure_count": 0,
            "question_types": [],
            "complexity_progression": [],
            "first_seen": None,
            "last_seen": None,
            "mastery_level": 0
        })
        self.topic_connections = defaultdict(set)
    
    def update_topic_knowledge(self, topics, complexity, question_type):
        """Update knowledge graph with new interaction"""
        now = datetime.now().isoformat()
        
        for topic in topics:
            topic_data = self.topic_mastery[topic]
            topic_data["exposure_count"] += 1
            topic_data["question_types"].append(question_type)
            topic_data["complexity_progression"].append(complexity)
            topic_data["last_seen"] = now
            
            if topic_data["first_seen"] is None:
                topic_data["first_seen"] = now
            
            # Calculate mastery level
            topic_data["mastery_level"] = self._calculate_mastery(topic_data)
        
        # Track topic connections
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                self.topic_connections[topic1].add(topic2)
                self.topic_connections[topic2].add(topic1)
    
    def _calculate_mastery(self, topic_data):
        """Calculate mastery level (0-10)"""
        exposure = min(topic_data["exposure_count"] / 5, 1) * 4
        
        # Complexity progression bonus
        if len(topic_data["complexity_progression"]) >= 3:
            is_progressing = topic_data["complexity_progression"][-1] > topic_data["complexity_progression"][0]
            progression_bonus = 3 if is_progressing else 0
        else:
            progression_bonus = 0
        
        # Variety bonus
        unique_types = len(set(topic_data["question_types"]))
        variety_bonus = min(unique_types / 3, 1) * 3
        
        return min(exposure + progression_bonus + variety_bonus, 10)
    
    def get_weak_topics(self):
        """Get topics that need reinforcement"""
        return [
            topic for topic, data in self.topic_mastery.items()
            if data["mastery_level"] < 4
        ]
    
    def get_strong_topics(self):
        """Get topics student has mastered"""
        return [
            topic for topic, data in self.topic_mastery.items()
            if data["mastery_level"] >= 7
        ]
    
    def suggest_next_topics(self):
        """Suggest related topics to explore"""
        strong = self.get_strong_topics()
        suggestions = set()
        
        for topic in strong:
            suggestions.update(self.topic_connections.get(topic, set()))
        
        # Remove already strong topics
        suggestions = suggestions - set(strong)
        
        return list(suggestions)[:3]


class PersonalityAdapter:
    """Adapts teaching style based on student personality indicators"""
    
    def __init__(self):
        self.traits = {
            "formality_preference": 5,  # 0-10 scale
            "humor_receptiveness": 5,
            "detail_orientation": 5,
            "encouragement_need": 5,
            "patience_level": 5
        }
        self.question_patterns = deque(maxlen=30)
    
    def analyze_communication_style(self, question):
        """Analyze student's communication style"""
        question_lower = question.lower()
        
        # Formality detection
        formal_indicators = ["please", "could you", "would you", "thank you"]
        casual_indicators = ["hey", "gonna", "wanna", "yeah", "cool"]
        
        if any(ind in question_lower for ind in formal_indicators):
            self.traits["formality_preference"] = min(self.traits["formality_preference"] + 0.5, 10)
        elif any(ind in question_lower for ind in casual_indicators):
            self.traits["formality_preference"] = max(self.traits["formality_preference"] - 0.5, 0)
        
        # Detail orientation
        if len(question.split()) > 20 or "specifically" in question_lower or "exactly" in question_lower:
            self.traits["detail_orientation"] = min(self.traits["detail_orientation"] + 0.3, 10)
        
        # Encouragement need
        if any(word in question_lower for word in ["struggling", "difficult", "hard", "confused"]):
            self.traits["encouragement_need"] = min(self.traits["encouragement_need"] + 0.5, 10)
        
        self.question_patterns.append({
            "timestamp": datetime.now().isoformat(),
            "traits_snapshot": dict(self.traits)
        })
    
    def get_teaching_style_recommendations(self):
        """Get recommendations for teaching approach"""
        recommendations = {
            "tone": "formal" if self.traits["formality_preference"] > 6 else "casual",
            "detail_level": "high" if self.traits["detail_orientation"] > 6 else "moderate",
            "encouragement": "high" if self.traits["encouragement_need"] > 6 else "standard",
            "pace": "patient" if self.traits["patience_level"] > 6 else "efficient"
        }
        
        return recommendations


# --- Enhanced Student Data Model ---
class EnhancedStudentModel:
    """Comprehensive student learning analytics and adaptation"""
    
    def __init__(self):
        self.student_profile = self.load_profile()
        self.interactions = self.load_interactions()
        
        # Initialize all analytics modules
        self.learning_style_analyzer = LearningStyleAnalyzer()
        self.cognitive_load_tracker = CognitiveLoadTracker()
        self.engagement_analyzer = EngagementAnalyzer()
        self.knowledge_graph = KnowledgeGraphTracker()
        self.personality_adapter = PersonalityAdapter()
        
        # Traditional RAG components
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.interaction_vectors = []
        
        # Restore analytics state from saved interactions
        self._restore_analytics_state()
        self.rebuild_vectors()
    
    def load_profile(self):
        """Load student profile from file"""
        if os.path.exists("student_data.json"):
            with open("student_data.json", 'r') as f:
                return json.load(f)
        return {
            "topics_discussed": defaultdict(int),
            "difficulty_preferences": {},
            "question_types": defaultdict(int),
            "learning_patterns": [],
            "total_interactions": 0,
            "first_interaction": None,
            "last_interaction": None,
            # New analytics fields
            "learning_style_scores": {},
            "engagement_metrics": {},
            "personality_traits": {},
            "knowledge_mastery": {}
        }
    
    def load_interactions(self):
        """Load interaction history"""
        if os.path.exists("student_interactions.json"):
            with open("student_interactions.json", 'r') as f:
                return json.load(f)
        return []
    
    def _restore_analytics_state(self):
        """Restore analytics from saved profile"""
        profile = self.student_profile
        
        # Restore learning style scores
        if "learning_style_scores" in profile:
            self.learning_style_analyzer.style_scores = defaultdict(
                float, profile["learning_style_scores"]
            )
        
        # Restore personality traits
        if "personality_traits" in profile:
            self.personality_adapter.traits = profile["personality_traits"]
        
        # Restore knowledge graph
        if "knowledge_mastery" in profile:
            self.knowledge_graph.topic_mastery = defaultdict(
                lambda: {
                    "exposure_count": 0,
                    "question_types": [],
                    "complexity_progression": [],
                    "first_seen": None,
                    "last_seen": None,
                    "mastery_level": 0
                },
                profile["knowledge_mastery"]
            )
    
    def save_profile(self):
        """Save comprehensive profile"""
        # Update profile with current analytics
        self.student_profile["learning_style_scores"] = dict(self.learning_style_analyzer.style_scores)
        self.student_profile["personality_traits"] = self.personality_adapter.traits
        self.student_profile["knowledge_mastery"] = dict(self.knowledge_graph.topic_mastery)
        self.student_profile["engagement_metrics"] = {
            "engagement_level": self.engagement_analyzer.get_engagement_level(),
            "interaction_frequency": self.engagement_analyzer.get_interaction_frequency(),
            "follow_up_rate": self.engagement_analyzer.follow_up_rate
        }
        
        with open("student_data.json", 'w') as f:
            json.dump(self.student_profile, f, indent=2, default=str)
    
    def save_interactions(self):
        """Save interactions to file"""
        with open("student_interactions.json", 'w') as f:
            json.dump(self.interactions, f, indent=2)
    
    def rebuild_vectors(self):
        """Rebuild TF-IDF vectors from interactions"""
        if len(self.interactions) > 0:
            texts = [f"{i['question']} {i['answer']}" for i in self.interactions]
            try:
                self.interaction_vectors = self.vectorizer.fit_transform(texts)
            except:
                self.interaction_vectors = []
    
    def analyze_question(self, question):
        """Comprehensive real-time analysis of question"""
        # Run all analyzers
        learning_styles = self.learning_style_analyzer.analyze_question(question)
        complexity = self.cognitive_load_tracker.analyze_question_complexity(question)
        self.engagement_analyzer.log_interaction(question)
        self.personality_adapter.analyze_communication_style(question)
        
        # Extract topics
        topics = self.extract_keywords(question)
        question_type = self.classify_question(question)
        
        # Update knowledge graph
        self.knowledge_graph.update_topic_knowledge(topics, complexity, question_type)
        
        return {
            "learning_styles": learning_styles,
            "complexity": complexity,
            "topics": topics,
            "question_type": question_type
        }
    
    def add_interaction(self, question, answer, analysis_results):
        """Store interaction with comprehensive analytics"""
        timestamp = datetime.now().isoformat()
        
        # Store interaction
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "topics": analysis_results["topics"],
            "complexity": analysis_results["complexity"],
            "learning_styles": analysis_results["learning_styles"],
            "question_type": analysis_results["question_type"]
        }
        self.interactions.append(interaction)
        
        # Update profile
        if self.student_profile["first_interaction"] is None:
            self.student_profile["first_interaction"] = timestamp
        self.student_profile["last_interaction"] = timestamp
        self.student_profile["total_interactions"] += 1
        
        # Update topic counts
        for keyword in analysis_results["topics"]:
            self.student_profile["topics_discussed"][keyword] = \
                self.student_profile["topics_discussed"].get(keyword, 0) + 1
        
        # Update question types
        self.student_profile["question_types"][analysis_results["question_type"]] = \
            self.student_profile["question_types"].get(analysis_results["question_type"], 0) + 1
        
        # Save everything
        self.save_profile()
        self.save_interactions()
        self.rebuild_vectors()
        
        print(f"✅ Stored interaction with full analytics. Total: {len(self.interactions)}")
    
    def extract_keywords(self, text):
        """Extract keywords from text"""
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
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.interaction_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    similar.append({
                        "question": self.interactions[idx]["question"],
                        "answer": self.interactions[idx]["answer"],
                        "similarity": float(similarities[idx])
                    })
            
            return similar
        except:
            return []
    
    def generate_adaptive_context(self):
        """Generate comprehensive adaptive context for AI"""
        if self.student_profile["total_interactions"] == 0:
            return "This is a new student. Be warm and welcoming. Start with clear, accessible explanations."
        
        # Learning style
        dominant_style = self.learning_style_analyzer.get_dominant_style()
        style_dist = self.learning_style_analyzer.get_style_distribution()
        
        # Engagement
        engagement = self.engagement_analyzer.get_engagement_level()
        frequency = self.engagement_analyzer.get_interaction_frequency()
        
        # Cognitive load
        is_struggling = self.cognitive_load_tracker.detect_struggle_pattern()
        avg_complexity = self.cognitive_load_tracker.get_average_complexity()
        
        # Knowledge
        weak_topics = self.knowledge_graph.get_weak_topics()
        strong_topics = self.knowledge_graph.get_strong_topics()
        next_topics = self.knowledge_graph.suggest_next_topics()
        
        # Personality
        teaching_style = self.personality_adapter.get_teaching_style_recommendations()
        
        # Build adaptive context
        context = f"""📊 STUDENT LEARNING PROFILE (Real-time Adaptive Context)

🎯 Learning Style: {dominant_style.upper()}
{f"   Distribution: {', '.join(f'{k}:{v}%' for k,v in style_dist.items())}" if style_dist else ""}
   → Adapt explanations to be {dominant_style}-friendly

💡 Engagement Level: {engagement.upper()} ({frequency} learner)
   → {"Maintain momentum with deeper challenges" if engagement == "highly_engaged" else "Build engagement gradually"}

🧠 Cognitive State:
   - Average complexity handled: {avg_complexity:.1f}/5
   - {"⚠️ SHOWING STRUGGLE PATTERNS - Simplify and encourage!" if is_struggling else "✅ Comfortable learning pace"}

📚 Knowledge Mastery:
   - Strong topics: {', '.join(strong_topics[:3]) if strong_topics else 'Building foundation'}
   - Needs reinforcement: {', '.join(weak_topics[:3]) if weak_topics else 'None identified'}
   {f"- Suggested next: {', '.join(next_topics)}" if next_topics else ""}

👤 Communication Style:
   - Tone preference: {teaching_style['tone']}
   - Detail level: {teaching_style['detail_level']}
   - Encouragement need: {teaching_style['encouragement']}
   - Pacing: {teaching_style['pace']}

📈 Session Stats:
   - Total interactions: {self.student_profile['total_interactions']}
   - Follow-up rate: {self.engagement_analyzer.follow_up_rate:.1%}
   - Learning since: {self.student_profile['first_interaction'][:10] if self.student_profile['first_interaction'] else 'today'}

🎓 TEACHING INSTRUCTIONS:
{self._generate_teaching_instructions(dominant_style, engagement, is_struggling, teaching_style)}
"""
        
        return context
    
    def _generate_teaching_instructions(self, style, engagement, struggling, teaching_style):
        """Generate specific teaching instructions"""
        instructions = []
        
        # Style-based
        if style == "visual":
            instructions.append("• Use analogies, diagrams descriptions, and visual metaphors")
        elif style == "auditory":
            instructions.append("• Explain step-by-step verbally, use clear descriptions")
        elif style == "kinesthetic":
            instructions.append("• Provide practical examples and hands-on exercises")
        elif style == "reading_writing":
            instructions.append("• Give structured text, lists, and written summaries")
        
        # Engagement-based
        if engagement == "highly_engaged":
            instructions.append("• Challenge with deeper questions and advanced concepts")
        elif engagement == "passive":
            instructions.append("• Ask engaging questions to boost interaction")
        
        # Struggle-based
        if struggling:
            instructions.append("• 🆘 BREAK DOWN concepts into simpler parts")
            instructions.append("• 🌟 Provide extra encouragement and positive reinforcement")
            instructions.append("• ✅ Check understanding frequently")
        
        # Tone-based
        if teaching_style["tone"] == "formal":
            instructions.append("• Maintain professional, respectful tone")
        else:
            instructions.append("• Use friendly, conversational tone")
        
        return "\n".join(instructions)
    
    def get_analytics_summary(self):
        """Get summary of all analytics for API response"""
        return {
            "learning_style": {
                "dominant": self.learning_style_analyzer.get_dominant_style(),
                "distribution": self.learning_style_analyzer.get_style_distribution()
            },
            "engagement": {
                "level": self.engagement_analyzer.get_engagement_level(),
                "frequency": self.engagement_analyzer.get_interaction_frequency(),
                "follow_up_rate": round(self.engagement_analyzer.follow_up_rate * 100, 1)
            },
            "cognitive_load": {
                "average_complexity": round(self.cognitive_load_tracker.get_average_complexity(), 2),
                "is_struggling": self.cognitive_load_tracker.detect_struggle_pattern()
            },
            "knowledge": {
                "strong_topics": self.knowledge_graph.get_strong_topics()[:3],
                "weak_topics": self.knowledge_graph.get_weak_topics()[:3],
                "suggested_next": self.knowledge_graph.suggest_next_topics()
            },
            "personality": self.personality_adapter.get_teaching_style_recommendations()
        }


# Initialize enhanced student model
student_model = EnhancedStudentModel()

# --- Utils ---
def clean_text(text):
    """Remove non-ASCII characters from text"""
    return re.sub(r'[^\x00-\x7F]+', '', text)

# --- OpenRouter setup ---
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY", 
    "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"
)
MODEL = "allenai/olmo-3.1-32b-think:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

base_chatbot_profile = """You are Bob, an adaptive AI tutor with real-time learning analytics.
You adjust your teaching style dynamically based on each student's learning patterns, engagement, and needs.
Explain clearly with warmth and encouragement."""

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


def build_learning_roadmap(goal, time_per_day, learning_style, analytics):
    """
    Generate adaptive learning roadmap
    """
    time_per_day = float(time_per_day)

    # Duration adapts to time commitment
    weeks = 2 if time_per_day >= 3 else 3 if time_per_day >= 1.5 else 4

    weak_topics = analytics["knowledge"]["weak_topics"]
    strong_topics = analytics["knowledge"]["strong_topics"]

    weekly_plan = []

    for i in range(weeks):
        if i == 0:
            weekly_plan.append(
                f"Week 1: Fundamentals of {goal} "
                f"(focus on basics and weak areas: {', '.join(weak_topics) if weak_topics else 'core concepts'})"
            )
        elif i == weeks - 1:
            weekly_plan.append(
                f"Week {i+1}: Mini project + revision + self-assessment"
            )
        else:
            weekly_plan.append(
                f"Week {i+1}: Intermediate concepts of {goal} "
                f"(build on strengths: {', '.join(strong_topics) if strong_topics else 'general practice'})"
            )

    # Study pattern adapts to learning style
    if learning_style == "visual":
        study_pattern = "Watch videos → draw diagrams → summarize visually"
    elif learning_style == "kinesthetic":
        study_pattern = "Learn concept → practice → build small exercises"
    elif learning_style == "auditory":
        study_pattern = "Listen → explain aloud → revise"
    else:
        study_pattern = "Read → take notes → revise next day"

    resources = [
        {
            "title": f"{goal} Full Course",
            "link": f"https://www.youtube.com/results?search_query={goal}+full+course"
        },
        {
            "title": f"{goal} Practice",
            "link": f"https://www.google.com/search?q={goal}+practice+problems"
        },
        {
            "title": f"{goal} Documentation",
            "link": f"https://www.google.com/search?q={goal}+documentation"
        }
    ]

    return {
        "goal": goal,
        "weeks": weeks,
        "time_per_day": time_per_day,
        "learning_style": learning_style,
        "weekly_plan": weekly_plan,
        "study_pattern": study_pattern,
        "resources": resources,
        "generated_at": datetime.now().isoformat()
    }



# --- Routes ---
@app.route('/ask', methods=['POST'])
def ask():
    """Handle chatbot questions with real-time adaptive learning"""
    data = request.get_json(force=True, silent=True)
    user_message = data.get("question", "").strip()
    
    if not user_message:
        return jsonify({"error": "No question provided"}), 400
    
    print(f"\n📝 Question: {user_message}")
    
    try:
        # STEP 1: Analyze question in real-time
        analysis = student_model.analyze_question(user_message)
        print(f"🔍 Analysis: {analysis}")
        
        # STEP 2: Get adaptive context
        adaptive_context = student_model.generate_adaptive_context()
        
        # STEP 3: Find similar past interactions (RAG)
        similar_interactions = student_model.find_similar_interactions(user_message)
        
        # STEP 4: Search dataset
        dataset_results = dataset_retriever.search(user_message)
        dataset_context = ""
        if dataset_results:
            dataset_context = "\n\n📚 Relevant dataset information:\n"
            for res in dataset_results:
                summary = ", ".join(f"{k}: {v}" for k, v in list(res["data"].items())[:3])
                dataset_context += f"- {summary}\n"
        
        # STEP 5: Build RAG context
        rag_context = ""
        if similar_interactions:
            rag_context = "\n\n💭 Previous related discussions:\n"
            for i, sim in enumerate(similar_interactions, 1):
                rag_context += f"{i}. Q: {sim['question']}\n   A: {sim['answer'][:100]}...\n"
        
        # STEP 6: Build enhanced prompt with adaptive context
        enhanced_profile = f"""{base_chatbot_profile}

{adaptive_context}
{rag_context}
{dataset_context}

Remember: Adapt your response based on the student profile above. Be natural and conversational."""
        
        messages = [
            {"role": "system", "content": enhanced_profile},
            {"role": "user", "content": user_message}
        ]
        
        # STEP 7: Get AI response
        answer = ask_openrouter(messages)
        cleaned_answer = clean_text(answer)
        
        # STEP 8: Store interaction with analytics
        student_model.add_interaction(user_message, cleaned_answer, analysis)
        
        # STEP 9: Get analytics summary
        analytics_summary = student_model.get_analytics_summary()
        
        print(f"✅ Response generated with adaptive learning")
        print(f"📊 Learning Style: {analytics_summary['learning_style']['dominant']}")
        print(f"💡 Engagement: {analytics_summary['engagement']['level']}")
        print(f"🧠 Cognitive Load: {analytics_summary['cognitive_load']['average_complexity']:.2f}")
        
        return jsonify({
            "answer": cleaned_answer,
            "analytics": analytics_summary,
            "similar_count": len(similar_interactions),
            "total_interactions": student_model.student_profile["total_interactions"],
            "analysis": analysis
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/student-profile', methods=['GET'])
def get_student_profile():
    """Get comprehensive student learning profile"""
    profile = dict(student_model.student_profile)
    # Convert defaultdicts to regular dicts
    profile['topics_discussed'] = dict(profile.get('topics_discussed', {}))
    profile['question_types'] = dict(profile.get('question_types', {}))
    
    # Add real-time analytics
    profile['real_time_analytics'] = student_model.get_analytics_summary()
    
    return jsonify(profile)

@app.route('/learning-analytics', methods=['GET'])
def get_learning_analytics():
    """Get detailed learning analytics"""
    analytics = student_model.get_analytics_summary()
    
    # Add detailed breakdowns
    analytics['detailed'] = {
        "learning_styles": {
            "scores": dict(student_model.learning_style_analyzer.style_scores),
            "recent_indicators": list(student_model.learning_style_analyzer.recent_indicators)
        },
        "cognitive_load": {
            "confusion_count": student_model.cognitive_load_tracker.confusion_indicators,
            "mastery_indicators": student_model.cognitive_load_tracker.mastery_indicators,
            "recent_complexities": list(student_model.cognitive_load_tracker.question_complexities)
        },
        "engagement": {
            "total_questions": student_model.engagement_analyzer.total_questions,
            "follow_up_questions": student_model.engagement_analyzer.follow_up_questions,
            "recent_depths": list(student_model.engagement_analyzer.question_depth_scores)
        },
        "personality": {
            "traits": student_model.personality_adapter.traits,
            "recommendations": student_model.personality_adapter.get_teaching_style_recommendations()
        }
    }
    
    return jsonify(analytics)
@app.route('/learning-roadmap', methods=['POST'])
def learning_roadmap():
    """
    Generate personalized learning roadmap
    """
    data = request.get_json(force=True)

    goal = data.get("goal", "").strip()
    time_per_day = data.get("time_per_day", 1)
    learning_style = data.get(
        "learning_style",
        student_model.learning_style_analyzer.get_dominant_style()
    )

    if not goal:
        return jsonify({"error": "Learning goal is required"}), 400

    analytics = student_model.get_analytics_summary()

    roadmap = build_learning_roadmap(
        goal=goal,
        time_per_day=time_per_day,
        learning_style=learning_style,
        analytics=analytics
    )

    return jsonify(roadmap)

@app.route('/knowledge-graph', methods=['GET'])
def get_knowledge_graph():
    """Get student's knowledge graph"""
    knowledge_data = {
        "topic_mastery": dict(student_model.knowledge_graph.topic_mastery),
        "topic_connections": {
            topic: list(connections) 
            for topic, connections in student_model.knowledge_graph.topic_connections.items()
        },
        "weak_topics": student_model.knowledge_graph.get_weak_topics(),
        "strong_topics": student_model.knowledge_graph.get_strong_topics(),
        "suggested_topics": student_model.knowledge_graph.suggest_next_topics()
    }
    
    return jsonify(knowledge_data)

@app.route('/interactions', methods=['GET'])
def get_interactions():
    """Get all interactions"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "total": len(student_model.interactions),
        "recent": student_model.interactions[-limit:]
    })

@app.route('/adaptive-context', methods=['GET'])
def get_adaptive_context():
    """Get the current adaptive context that will be sent to AI"""
    context = student_model.generate_adaptive_context()
    return jsonify({
        "context": context,
        "analytics": student_model.get_analytics_summary()
    })

@app.route('/reset-student', methods=['POST'])
def reset_student():
    """Reset student data (for testing)"""
    if os.path.exists("student_data.json"):
        os.remove("student_data.json")
    if os.path.exists("student_interactions.json"):
        os.remove("student_interactions.json")
    global student_model
    student_model = EnhancedStudentModel()
    return jsonify({"message": "Student data reset successfully"})

@app.route('/learning-insights', methods=['GET'])
def get_learning_insights():
    """Get actionable insights about student's learning"""
    analytics = student_model.get_analytics_summary()
    
    insights = []
    
    # Learning style insights
    style = analytics['learning_style']['dominant']
    if style != 'balanced':
        insights.append({
            "type": "learning_style",
            "message": f"Student prefers {style} learning. Adapt explanations accordingly.",
            "action": f"Use more {style}-friendly teaching methods"
        })
    
    # Engagement insights
    engagement = analytics['engagement']['level']
    if engagement == 'highly_engaged':
        insights.append({
            "type": "engagement",
            "message": "Student is highly engaged! Ready for advanced topics.",
            "action": "Introduce challenging concepts and deeper questions"
        })
    elif engagement == 'passive':
        insights.append({
            "type": "engagement",
            "message": "Student engagement is low. Need to boost interaction.",
            "action": "Ask more questions and provide interactive examples"
        })
    
    # Cognitive load insights
    if analytics['cognitive_load']['is_struggling']:
        insights.append({
            "type": "cognitive_load",
            "message": "⚠️ Student showing signs of struggle.",
            "action": "Simplify explanations, break down concepts, provide encouragement"
        })
    
    # Knowledge insights
    weak = analytics['knowledge']['weak_topics']
    if weak:
        insights.append({
            "type": "knowledge_gaps",
            "message": f"Topics needing reinforcement: {', '.join(weak)}",
            "action": "Revisit these topics with different approaches"
        })
    
    strong = analytics['knowledge']['strong_topics']
    if strong:
        insights.append({
            "type": "knowledge_strengths",
            "message": f"Strong grasp of: {', '.join(strong)}",
            "action": "Build on these strengths to teach related concepts"
        })
    
    suggested = analytics['knowledge']['suggested_next']
    if suggested:
        insights.append({
            "type": "next_topics",
            "message": f"Ready for: {', '.join(suggested)}",
            "action": "Introduce these related topics naturally"
        })
    
    return jsonify({
        "insights": insights,
        "analytics_summary": analytics,
        "generated_at": datetime.now().isoformat()
    })

@app.route('/teaching-strategy', methods=['GET'])
def get_teaching_strategy():
    """Get current teaching strategy recommendations"""
    strategy = {
        "current_context": student_model.generate_adaptive_context(),
        "recommendations": student_model._generate_teaching_instructions(
            student_model.learning_style_analyzer.get_dominant_style(),
            student_model.engagement_analyzer.get_engagement_level(),
            student_model.cognitive_load_tracker.detect_struggle_pattern(),
            student_model.personality_adapter.get_teaching_style_recommendations()
        ),
        "personality_profile": student_model.personality_adapter.get_teaching_style_recommendations(),
        "analytics": student_model.get_analytics_summary()
    }
    
    return jsonify(strategy)

@app.route('/')
def home():
    """Health check endpoint with enhanced info"""
    analytics = student_model.get_analytics_summary() if student_model.student_profile['total_interactions'] > 0 else None
    
    html = f"""
    <h1>🎓 AI Tutor with Real-Time Adaptive Learning ✅</h1>
    <h3>Session Stats:</h3>
    <ul>
        <li>Total Interactions: {student_model.student_profile['total_interactions']}</li>
        <li>Learning Style: {analytics['learning_style']['dominant'] if analytics else 'Not determined'}</li>
        <li>Engagement Level: {analytics['engagement']['level'] if analytics else 'New student'}</li>
        <li>Is Struggling: {'⚠️ Yes' if analytics and analytics['cognitive_load']['is_struggling'] else '✅ No'}</li>
    </ul>
    
    <h3>API Endpoints:</h3>
    <ul>
        <li><a href='/student-profile'>📊 Student Profile</a> - Comprehensive learning profile</li>
        <li><a href='/learning-analytics'>🧠 Learning Analytics</a> - Detailed analytics breakdown</li>
        <li><a href='/knowledge-graph'>📚 Knowledge Graph</a> - Topic mastery map</li>
        <li><a href='/learning-insights'>💡 Learning Insights</a> - Actionable recommendations</li>
        <li><a href='/teaching-strategy'>🎯 Teaching Strategy</a> - Current adaptive strategy</li>
        <li><a href='/adaptive-context'>🔄 Adaptive Context</a> - Real-time AI context</li>
        <li><a href='/interactions?limit=5'>💬 Recent Interactions</a> - Conversation history</li>
    </ul>
    
    <h3>Features:</h3>
    <ul>
        <li>✅ Real-time learning style detection (Visual/Auditory/Kinesthetic/Reading-Writing)</li>
        <li>✅ Cognitive load tracking and struggle detection</li>
        <li>✅ Engagement pattern analysis</li>
        <li>✅ Knowledge graph with topic mastery</li>
        <li>✅ Personality adaptation (formality, detail level, encouragement)</li>
        <li>✅ Dynamic teaching strategy adjustment</li>
        <li>✅ RAG with interaction history</li>
    </ul>
    """
    
    return html

if __name__ == '__main__':
    print("🚀 Starting Enhanced AI Tutor with Real-Time Adaptive Learning...")
    print(f"📊 Loaded {len(student_model.interactions)} previous interactions")
    print(f"📁 Data stored in: {os.path.abspath('student_data.json')}")
    print("\n🎯 NEW FEATURES:")
    print("   ✓ Learning style detection")
    print("   ✓ Cognitive load tracking")
    print("   ✓ Engagement analysis")
    print("   ✓ Knowledge graph")
    print("   ✓ Personality adaptation")
    print("   ✓ Real-time teaching strategy adjustment")
    print("\n📡 API Endpoints:")
    print("   • POST /ask - Ask questions (with adaptive learning)")
    print("   • GET /learning-analytics - Detailed analytics")
    print("   • GET /knowledge-graph - Topic mastery")
    print("   • GET /learning-insights - Actionable insights")
    print("   • GET /teaching-strategy - Current strategy")
    print("\n")
    app.run(host='127.0.0.1', port=5000, debug=True)