import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
import os
import time
import html
import markdown
from datetime import datetime
from difflib import get_close_matches
import requests
from dotenv import load_dotenv
import json
import random
from PIL import Image

# Optional ML
USE_SKLEARN = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except:
    USE_SKLEARN = False

# Load .env
load_dotenv()

# ---------------- CONFIG ----------------
APP_NAME = "AgriChat Pro"
CSV_FILENAME = "agriculture_data.csv"
IMAGES_DIR = os.path.abspath(os.getenv("IMAGES_DIR", "images"))
DB_PATH = "chatbot_data.db"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
SIM_THRESHOLD = 0.2
DIFFLIB_CUTOFF = 0.45

os.makedirs(IMAGES_DIR, exist_ok=True)
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🌱")

# ---------------- HELPERS ----------------
def now_iso():
    return datetime.now().isoformat()

def parse_time_iso(ts):
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return ""

def render_markdown(text):
    return markdown.markdown(text)

def show_typing_indicator():
    typing_placeholder = st.empty()
    typing_placeholder.markdown("⚪ Bot is thinking ⚪")
    time.sleep(1)
    typing_placeholder.empty()

# ---------------- AGRICULTURAL EXPERT SYSTEM ----------------
class AgriculturalExpert:
    def __init__(self):
        self.conversation_history = []

    def add_to_conversation(self, user_input, bot_response):
        """Remember conversation context"""
        self.conversation_history.append({"user": user_input, "bot": bot_response})
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def get_conversation_context(self):
        """Get recent conversation context"""
        return self.conversation_history[-3:] if self.conversation_history else []

agricultural_expert = AgriculturalExpert()

# ---------------- FIXED DATABASE ----------------
class ChatDB:
    def __init__(self, path=DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self._init()
        except Exception as e:
            st.error(f"Database connection error: {e}")
            self.conn = None

    def _init(self):
        if not self.conn:
            return
        try:
            c = self.conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    role TEXT,
                    message TEXT,
                    image_path TEXT,
                    timestamp TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")

    def register(self, username, password):
        if not self.conn:
            return False
        try:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            c = self.conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?,?)",
                     (username, hashed))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Registration error: {e}")
            return False

    def verify(self, username, password):
        if not self.conn:
            return False, {}
        try:
            c = self.conn.cursor()
            c.execute("SELECT password FROM users WHERE username=?", (username,))
            row = c.fetchone()
            if not row:
                time.sleep(1)
                return False, {}
            if bcrypt.checkpw(password.encode(), row[0].encode()):
                return True, {}
            return False, {}
        except Exception as e:
            st.error(f"Login verification error: {e}")
            return False, {}

    def save_chat(self, username, role, message, image_path=None):
        if not self.conn:
            return
        try:
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO conversations
                (username, role, message, image_path, timestamp)
                VALUES (?,?,?,?,?)
            """, (username, role, message, image_path, now_iso()))
            self.conn.commit()
        except Exception as e:
            st.error(f"Chat save error: {e}")

    def load_chats(self, username, limit=200):
        if not self.conn:
            return []
        try:
            c = self.conn.cursor()
            c.execute("""
                SELECT role, message, image_path, timestamp
                FROM conversations
                WHERE username=?
                ORDER BY id
            """, (username,))
            rows = c.fetchall()
            return [
                {"role": r[0], "text": r[1], "image": r[2], "time": r[3]}
                for r in rows
            ][-limit:]
        except Exception as e:
            st.error(f"Chat load error: {e}")
            return []

chat_db = ChatDB()

# ---------------- LOAD CSV ----------------
@st.cache_data(ttl=3600)
def load_csv(path):
    try:
        df = pd.read_csv(path, encoding='utf-8')
        df.columns = df.columns.str.strip().str.lower()
        return df.fillna("")
    except Exception as e:
        st.error(f"CSV load error: {e}")
        return pd.DataFrame({
            'topic': ['crops'], 
            'question': ['sample question'], 
            'problem': ['sample problem'], 
            'solution': ['sample solution']
        })

# ---------------- KNOWLEDGE BASE ----------------
def build_knowledge_base(df):
    knowledge_base = {}
    for _, row in df.iterrows():
        topic = row.get("topic", "").strip().lower()
        question = row.get("question", "").strip().lower()
        solution = row.get("solution", "").strip()
        problem = row.get("problem", "").strip()
        
        if topic not in knowledge_base:
            knowledge_base[topic] = {}
        
        knowledge_base[topic][question] = {
            "solution": solution,
            "problem": problem
        }
    return knowledge_base

def get_planting_calendar(knowledge_base):
    planting_calendar = {}
    for topic in knowledge_base:
        for question, data in knowledge_base[topic].items():
            if "plant" in question and "when" in question:
                crop_keywords = ["maize", "wheat", "beans", "tomato", "carrot", "cabbage", "watermelon", "sorghum"]
                for crop in crop_keywords:
                    if crop in question:
                        planting_calendar[crop] = data
                        break
    return planting_calendar

# Load data
df = load_csv(CSV_FILENAME)
knowledge_base = build_knowledge_base(df)
planting_calendar = get_planting_calendar(knowledge_base)

# ---------------- TF-IDF ----------------
@st.cache_resource
def get_vectorizer(corpus):
    if not USE_SKLEARN or not corpus:
        return None
    try:
        return TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(corpus)
    except Exception as e:
        st.error(f"Vectorizer error: {e}")
        return None

# Prepare corpus for ML
corpus_texts = []
for topic in knowledge_base:
    for question in knowledge_base[topic].keys():
        corpus_texts.append(question)

vectorizer = get_vectorizer(corpus_texts)
lowered_questions = [q.lower() for q in corpus_texts]

# ---------------- ENHANCED WEATHER WITH FALLBACKS ----------------
def get_weather(city="Maseru", days=3):
    try:
        if not OPENWEATHER_API_KEY:
            return "🌤️ Weather service is currently unavailable. For accurate forecasts, check local weather reports."
        
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&cnt={days*8}&appid={OPENWEATHER_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        forecast = []
        for entry in data.get("list", [])[::8]:
            forecast.append(f"{entry['dt_txt'].split()[0]}: {entry['weather'][0]['description']}, {entry['main']['temp']}°C")
        return "\n".join(forecast) if forecast else "No forecast available."
    except Exception as e:
        return "🌤️ Weather service is temporarily unavailable. For Lesotho, typical seasons are:\n• Rainy: October-March\n• Dry: April-September\nPlan your planting accordingly."

def get_weather_advisory(city="Maseru"):
    try:
        if not OPENWEATHER_API_KEY:
            return ["📡 Weather advisory service unavailable. Monitor local conditions and protect crops during extreme weather."]
        
        return [
            "⚠️ Regular farm advisory:",
            "• Check soil moisture before watering",
            "• Monitor crops for pest signs weekly", 
            "• Prepare for seasonal changes",
            "• Rotate crops to maintain soil health"
        ]
    except Exception as e:
        return ["📡 Advisory service unavailable. Practice good farm management: regular monitoring and soil care."]

# ---------------- ENHANCED HUMAN-LIKE CONVERSATION WITH FALLBACKS ----------------
def get_conversational_response(user_input, context, chat_history):
    """Handle human-like conversation flow with fallbacks"""
    user_input_lower = user_input.lower()
    
    recent_context = agricultural_expert.get_conversation_context()
    
    # Handle greetings
    if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey", "greetings"]):
        responses = [
            "Hello! 👋 I'm your agricultural assistant. How can I help with your crops today?",
            "Hi there! 🌱 Ready to discuss your farming questions?",
            "Hey! 👨‍🌾 What can I help you with in your farm today?"
        ]
        return random.choice(responses)
    
    # Handle thanks
    if any(thanks in user_input_lower for thanks in ["thank", "thanks", "thank you"]):
        responses = [
            "You're welcome! 😊 Let me know if you need any more help with your crops.",
            "Happy to help! 🌾 Don't hesitate to ask if you have more questions.",
            "Glad I could assist! 🌱 Feel free to ask anything else about farming."
        ]
        return random.choice(responses)
    
    # Handle follow-ups
    if any(phrase in user_input_lower for phrase in ["i already did that", "i tried that", "it didn't work", "not working"]):
        if recent_context:
            last_advice = recent_context[-1]["bot"] if recent_context else ""
            if "fertilizer" in last_advice.lower():
                return "I see the fertilizer didn't work. 🤔 Try checking soil pH first, as nutrients may not be available to plants. Test soil or add compost to improve conditions."
            elif "water" in last_advice.lower():
                return "If watering adjustments didn't help, check root health and soil drainage. 🕵️‍♂️ Poor drainage can cause root rot even with proper watering."
        
        return "I understand the previous suggestion didn't work. 🔄 Let's try a different approach. Could you describe what happened after trying my advice?"
    
    # Handle "what can I do about..." with fallback
    if "what can i do" in user_input_lower or "what should i do" in user_input_lower:
        if "yellow" in user_input_lower:
            return "For yellow leaves:\n\n1. **Check soil moisture** - don't overwater\n2. **Test nutrients** - may need nitrogen\n3. **Look for pests** - check under leaves\n4. **Improve drainage** - add organic matter\n5. **Consider sunlight** - ensure adequate exposure"
    
    # Handle uncertainty
    if any(phrase in user_input_lower for phrase in ["i don't know", "not sure", "what do you think"]):
        return "That's okay! 🌟 Describe what you're seeing - leaf color, plant growth, soil condition - and I'll help diagnose the issue."
    
    # Handle positive feedback
    if any(phrase in user_input_lower for phrase in ["it worked", "thanks it helped", "that worked"]):
        return "That's great to hear! 🎉 I'm glad the advice helped. Is there anything else you'd like to know about farming?"
    
    return None

def update_context(user_input, context):
    user_input = user_input.lower()
    
    crop_keywords = {
        'maize': ['maize', 'corn'],
        'wheat': ['wheat', 'bread grain'],
        'beans': ['bean', 'beans', 'legume'],
        'tomato': ['tomato', 'tomatoes'],
        'potato': ['potato', 'potatoes'],
        'watermelon': ['watermelon', 'melon'],
        'cabbage': ['cabbage', 'cole'],
        'carrot': ['carrot'],
        'sorghum': ['sorghum']
    }
    
    for crop, keywords in crop_keywords.items():
        if any(keyword in user_input for keyword in keywords):
            context["crop_type"] = crop
            break
    
    return context

# ---------------- ENHANCED ANSWER SYSTEM WITH COMPREHENSIVE FALLBACKS ----------------
def find_best_answer(user_text, context):
    try:
        # First check for conversational responses
        conversational_response = get_conversational_response(user_text, context, st.session_state.get("chat", []))
        if conversational_response:
            agricultural_expert.add_to_conversation(user_text, conversational_response)
            return conversational_response

        user_text = user_text.strip().lower()
        if not user_text:
            return "Please type a question about crops, weather, or farming in Lesotho."

        # Enhanced weather with fallback
        if "weather" in user_text or "forecast" in user_text:
            city = st.session_state.get("weather_city", "Maseru")
            forecast = get_weather(city=city, days=3)
            return f"**Weather for {city}:**\n{forecast}"

        # Enhanced advisory with fallback
        if "advisory" in user_text or "alert" in user_text:
            advisory = get_weather_advisory()
            return "**Farm Advisory:**\n" + "\n".join([f"• {advice}" for advice in advisory])

        # Enhanced planting questions with fallback
        if "plant" in user_text or "grow" in user_text or "sow" in user_text:
            for crop in planting_calendar.keys():
                if crop in user_text:
                    context["crop_type"] = crop
                    guide = show_planting_calendar(crop, planting_calendar)
                    agricultural_expert.add_to_conversation(user_text, guide)
                    return guide

        # ML-based matching with fallback
        if vectorizer and corpus_texts:
            query_vec = vectorizer.transform([user_text])
            similarities = cosine_similarity(query_vec, vectorizer.transform(corpus_texts))[0]
            best_idx = similarities.argmax()
            if similarities[best_idx] > SIM_THRESHOLD:
                for topic in knowledge_base:
                    if corpus_texts[best_idx] in knowledge_base[topic]:
                        response = knowledge_base[topic][corpus_texts[best_idx]]["solution"]
                        agricultural_expert.add_to_conversation(user_text, response)
                        return response

        # Enhanced difflib matching with fallback
        matches = get_close_matches(user_text, lowered_questions, n=1, cutoff=DIFFLIB_CUTOFF)
        if matches:
            for topic in knowledge_base:
                if matches[0] in knowledge_base[topic]:
                    response = knowledge_base[topic][matches[0]]["solution"]
                    agricultural_expert.add_to_conversation(user_text, response)
                    return response

        # COMPREHENSIVE FALLBACK SYSTEM
        fallback_keywords = {
            'planting': "For planting advice, tell me which crop you're interested in (maize, wheat, beans, etc.) and I'll provide specific guidance.",
            'fertilizer': "For fertilizer questions, specify the crop and I'll recommend the right nutrients and application methods.",
            'pest': "For pest issues, describe the damage you're seeing and which crop is affected for specific solutions.",
            'disease': "For disease problems, tell me the symptoms and crop type for treatment recommendations.",
            'soil': "For soil questions, I can help with soil preparation, pH adjustment, and fertility improvement.",
            'water': "For irrigation advice, I can provide watering schedules and efficient water management tips.",
            'harvest': "For harvesting guidance, specify the crop and I'll tell you when and how to harvest for best results."
        }
        
        for keyword, response in fallback_keywords.items():
            if keyword in user_text:
                agricultural_expert.add_to_conversation(user_text, response)
                return response

        # Final intelligent fallback
        final_fallbacks = [
            "I specialize in agricultural advice for Lesotho. Could you ask about crop planting, pest control, soil management, or farming techniques?",
            "I'm here to help with farming questions. Try asking about specific crops, weather planning, or common farming challenges in Lesotho.",
            "As your farming assistant, I can help with crop care, planting schedules, and agricultural best practices. What specific farming topic can I help with?"
        ]
        response = random.choice(final_fallbacks)
        agricultural_expert.add_to_conversation(user_text, response)
        return response
        
    except Exception as e:
        # Ultimate fallback for any error
        error_fallbacks = [
            "I'm having trouble processing that right now. Could you try rephrasing your farming question?",
            "Let me try that again. Could you ask about crops, weather, or farming practices in a different way?",
            "I want to make sure I understand correctly. Could you rephrase your agricultural question?"
        ]
        return random.choice(error_fallbacks)

def show_planting_calendar(crop, planting_calendar):
    if crop in planting_calendar:
        data = planting_calendar[crop]
        return f"""
**{crop.capitalize()} Planting Guide** 🌱

**Planting Advice:** {data['solution']}

**Common Considerations:** {data.get('problem', 'Ensure proper soil preparation and drainage')}

**Pro Tip:** Test your soil before planting and add organic matter for best results!
"""
    return f"For {crop}, general planting advice: Prepare soil with compost, ensure good drainage, and plant at the start of the rainy season for most crops in Lesotho."

# ---------------- ENHANCED IMAGE PROCESSING WITH FALLBACKS ----------------
def process_camera_image(picture):
    """Process camera image with comprehensive fallbacks"""
    try:
        if picture is not None:
            image = Image.open(picture)
            img_path = os.path.join(IMAGES_DIR, f"camera_capture_{int(time.time())}.jpg")
            image.save(img_path, "JPEG")
            
            analysis = f"""
**📸 Image Received**

I've captured your crop image! For detailed analysis:

**Please describe:**
• What crop is this?
• What specific issues do you see?
• How are the leaves/soil/fruits looking?

**I can then provide specific advice on:**
• Disease identification
• Pest problems  
• Nutrient issues
• Growth recommendations

The more details you provide, the better I can help! 🌱
"""
            return analysis, img_path
        return "No image was captured. Please try again.", None
    except Exception as e:
        return "I had trouble processing the image. Please try again or describe what you're seeing in words.", None

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "weather_city" not in st.session_state:
    st.session_state["weather_city"] = "Maseru"
if "context" not in st.session_state:
    st.session_state.context = {"crop_type": None}
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "camera_capture" not in st.session_state:
    st.session_state.camera_capture = None
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ---------------- FIXED UI STYLING ----------------
def apply_custom_styling():
    dark_mode = st.session_state.dark_mode
    
    bg_color = '#0e1117' if dark_mode else '#f8f9fa'
    card_bg = '#1e2130' if dark_mode else 'white'
    text_color = '#fafafa' if dark_mode else '#262730'
    border_color = '#2d3349' if dark_mode else '#e6e6e6'
    chat_bg = '#1a1d2a' if dark_mode else 'white'
    bot_msg_bg = '#2d3349' if dark_mode else '#f1f3f6'
    header_gradient = 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)' if dark_mode else 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    button_gradient = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' if dark_mode else 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    
    custom_css = f"""
    <style>
        .stApp {{
            background: {bg_color};
            color: {text_color};
        }}
        
        .main-header {{
            background: {header_gradient};
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            color: white;
            text-align: center;
        }}
        
        .custom-card {{
            background: {card_bg};
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid {border_color};
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }}
        
        .chat-container {{
            background: {chat_bg};
            border-radius: 16px;
            padding: 1.5rem;
            height: 60vh;
            overflow-y: auto;
            border: 1px solid {border_color};
        }}
        
        .msg-user {{
            text-align: right;
            margin-left: auto;
            margin-bottom: 1rem;
        }}
        
        .msg-bot {{
            text-align: left;
            margin-right: auto;
            margin-bottom: 1rem;
        }}
        
        .msg-bubble {{
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            display: inline-block;
        }}
        
        .msg-user .msg-bubble {{
            background: {button_gradient};
            color: white;
        }}
        
        .msg-bot .msg-bubble {{
            background: {bot_msg_bg};
            color: {text_color};
            border: 1px solid {border_color};
        }}
        
        .input-container {{
            background: {card_bg};
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid {border_color};
            margin-top: 1rem;
        }}
        
        .stButton>button {{
            background: {button_gradient};
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
        }}
        
        .chat-image {{
            max-width: 200px;
            border-radius: 12px;
            margin: 8px 0;
        }}
        
        .stTextInput>div>div>input {{
            background: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_styling()

# ---------------- HEADER ----------------
st.markdown(
    f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 1rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/194/194938.png" width="60" style="border-radius: 50%; background: white; padding: 8px;">
            <div>
                <h1>🌱 {APP_NAME}</h1>
                <p>Your intelligent farming assistant for Lesotho</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1, 2])

with col1:
    # Auth panel
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🔐 Authentication")
    
    try:
        if not st.session_state["logged_in"]:
            auth_mode = st.radio("Auth Mode", ["Guest", "Login", "Register"], index=0)
            if auth_mode == "Register":
                new_user = st.text_input("New Username")
                new_pwd = st.text_input("New Password", type="password")
                if st.button("Register") and new_user and new_pwd:
                    if chat_db.register(new_user, new_pwd):
                        st.success("✅ Registered successfully! Please log in.")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = new_user
                        st.session_state["chat"] = []
                        st.rerun()
                    else:
                        st.error("❌ Registration failed. Username may already exist.")
            elif auth_mode == "Login":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.button("Login") and username and password:
                    success, prefs = chat_db.verify(username, password)
                    if success:
                        st.success(f"✅ Welcome back, {username}!")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["chat"] = chat_db.load_chats(username)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            else:
                if st.button("Continue as Guest"):
                    st.session_state["logged_in"] = False
                    st.session_state["username"] = "Guest"
                    st.session_state["chat"] = []
                    st.success("🎉 Welcome! You're browsing as a guest.")
                    st.rerun()
        else:
            st.success(f"✅ Logged in as: **{st.session_state['username']}**")
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                st.session_state["chat"] = []
                st.rerun()
    except Exception as e:
        st.warning(f"Authentication error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Weather
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🌦️ Weather")
    city = st.text_input("City", value=st.session_state["weather_city"])
    if st.button("Get Forecast"):
        st.session_state["weather_city"] = city
        forecast = get_weather(city=city)
        st.info(forecast)
    st.markdown('</div>', unsafe_allow_html=True)

    # Weather Advisory
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("⚠️ Weather Advisory")
    if st.button("Get Advisory"):
        advisory = get_weather_advisory()
        st.warning("\n".join(advisory))
    st.markdown('</div>', unsafe_allow_html=True)

    # Planting Calendar
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📅 Planting Calendar")
    crop_options = list(planting_calendar.keys())
    if crop_options:
        selected_crop = st.selectbox("Select a crop:", crop_options, key="crop_select")
        if st.button("Show Guide"):
            guide = show_planting_calendar(selected_crop, planting_calendar)
            st.info(guide)
    else:
        st.info("No planting data available")
    st.markdown('</div>', unsafe_allow_html=True)

    # Theme toggle
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🎨 Theme")
    if st.checkbox("Dark Mode", value=st.session_state.dark_mode):
        st.session_state.dark_mode = True
        apply_custom_styling()
    else:
        st.session_state.dark_mode = False
        apply_custom_styling()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Chat display
    st.markdown('<div class="chat-container" id="chat_container">', unsafe_allow_html=True)
    for msg in st.session_state["chat"][-50:]:
        cls = "msg-user" if msg["role"] == "user" else "msg-bot"
        content = render_markdown(msg["text"]) if msg["role"] == "assistant" else html.escape(msg["text"])
        if msg.get("image"):
            content = f'<img src="file://{msg["image"]}" class="chat-image"><br>{content}'
        st.markdown(
            f"""
        <div class="{cls}">
            <div class="msg-bubble">{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            document.getElementById('chat_container').scrollTop = document.getElementById('chat_container').scrollHeight;
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ENHANCED CHAT INPUT WITH BOTH CAMERA AND UPLOAD BUTTONS
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    with st.form("chat_input", clear_on_submit=True):
        # Single line for input and buttons - BOTH CAMERA AND UPLOAD
        input_col1, input_col2, input_col3, input_col4 = st.columns([5, 1, 1, 1])
        
        with input_col1:
            user_input = st.text_input("Type your message...", label_visibility="collapsed", placeholder="Ask about crops, weather, or farming...")
        
        with input_col2:
            # Upload button
            uploaded_file = st.file_uploader("📁", type=["png", "jpg", "jpeg"], 
                                           label_visibility="collapsed", 
                                           help="Upload an image",
                                           key="file_uploader")
        
        with input_col3:
            # Camera button - single word
            camera_clicked = st.form_submit_button("📷", help="Take a photo")
            if camera_clicked:
                st.session_state.show_camera = True
        
        with input_col4:
            submit = st.form_submit_button("Send")

    # Handle uploaded file
    if uploaded_file and not st.session_state.show_camera:
        try:
            img_path = os.path.join(IMAGES_DIR, f"upload_{int(time.time())}.png")
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state["chat"].append({
                "role": "user", 
                "text": "[Image uploaded for analysis]", 
                "time": now_iso(), 
                "image": img_path
            })
            st.session_state["chat"].append({
                "role": "assistant", 
                "text": "**📸 Image Uploaded Successfully!**\n\nPlease describe what you're seeing in the image so I can provide specific agricultural advice.", 
                "time": now_iso()
            })
            
            if st.session_state.get("logged_in"):
                chat_db.save_chat(st.session_state["username"], "user", "[Image uploaded]", img_path)
                chat_db.save_chat(st.session_state["username"], "assistant", "Image uploaded successfully")
            
            st.rerun()
        except Exception as e:
            st.error(f"❌ Upload error: {e}")

    # Camera appears only when requested and CLOSES AFTER CAPTURE
    if st.session_state.show_camera:
        st.write("**Camera - Take a picture of your crops**")
        picture = st.camera_input("", key="camera_input", label_visibility="collapsed")
        
        if picture:
            # Process the image immediately
            try:
                analysis, img_path = process_camera_image(picture)
                if img_path:
                    st.session_state["chat"].append({
                        "role": "user", 
                        "text": "[Camera image captured]", 
                        "time": now_iso(), 
                        "image": img_path
                    })
                    st.session_state["chat"].append({
                        "role": "assistant", 
                        "text": analysis, 
                        "time": now_iso()
                    })
                    if st.session_state.get("logged_in"):
                        chat_db.save_chat(st.session_state["username"], "user", "[Camera image captured]", img_path)
                        chat_db.save_chat(st.session_state["username"], "assistant", analysis)
                    
                    # ✅ FIXED: CLOSE CAMERA AFTER CAPTURE
                    st.session_state.show_camera = False
                    st.session_state.camera_capture = None
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Camera processing error: {e}")

        # Close camera button
        if st.button("Close Camera"):
            st.session_state.show_camera = False
            st.rerun()

    # Process text message if submitted
    if user_input and submit:
        st.session_state["chat"].append({
            "role": "user", 
            "text": user_input, 
            "time": now_iso()
        })
        if st.session_state.get("logged_in"):
            chat_db.save_chat(st.session_state["username"], "user", user_input)

        st.session_state.context = update_context(user_input, st.session_state.context)

        show_typing_indicator()
        answer = find_best_answer(user_input, st.session_state.context)

        st.session_state["chat"].append({
            "role": "assistant", 
            "text": answer, 
            "time": now_iso()
        })
        if st.session_state.get("logged_in"):
            chat_db.save_chat(st.session_state["username"], "assistant", answer)

        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.9rem;">
        <hr style="margin-bottom: 1rem;">
        <p>🌱 <strong>AgriChat Pro</strong> - Your AI Agricultural Assistant for Lesotho</p>
        <p>Ready for deployment | Complete image support | Robust error handling</p>
    </div>
    """,
    unsafe_allow_html=True
)