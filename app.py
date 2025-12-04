import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from openai import OpenAI
from streamlit_lottie import st_lottie
import requests
import time
import random

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================

st.set_page_config(page_title="VibeCheck Player", page_icon="🎧", layout="wide")

# Custom CSS for a modern look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #1DB954, #191414);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & ASSETS
# ==========================================

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Lottie Animation URLs
LOTTIE_URLS = {
    "Happy": "https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json",  # Dancing/Happy
    "Sad": "https://assets4.lottiefiles.com/packages/lf20_kht4ll.json",      # Rain/Sad
    "Angry": "https://assets3.lottiefiles.com/packages/lf20_9xR7G8.json",    # Fire/Angry
    "Fear": "https://assets1.lottiefiles.com/packages/lf20_h55dy8.json",     # Ghost/Scared
    "Disgust": "https://assets9.lottiefiles.com/packages/lf20_nm1z8p.json",  # Yuck face
    "Surprise": "https://assets8.lottiefiles.com/packages/lf20_6ofl4l.json", # Shocked
    "Neutral": "https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json" # Chill music waves
}

# Fallback tracks
FALLBACK_TRACKS = {
    "Happy": ["spotify:track:6CSLNGruNhqpb5zhfs5n3i", "spotify:track:5mCPDVBb16L4XQwDdbRUpz"], # Celebration, Happy
    "Sad": ["spotify:track:4k7x3QKrc3h3U0Viqk0uop", "spotify:track:3Qm86XLflmIXVm1wcwkgDK"], # Skinny Love, All I Want
    "Angry": ["spotify:track:5cZqsjVs6MevCnAkasbEOX", "spotify:track:2M9ro2krNb7nr7HSprkEgo"], # Break Stuff, Chop Suey
    "Fear": ["spotify:track:4NzMOnvSJVNKF7nw5NkXIP", "spotify:track:27GmP9AWRs744SzKcpJsTZ"], # Thriller
    "Disgust": ["spotify:track:2dLLR6qlu5UJ5gk0dKz0h3", "spotify:track:0c4IEciLCDdXEhhKxj4ThA"], # Creep
    "Surprise": ["spotify:track:6wOYnPq1hPihRepcDClPUw", "spotify:track:5mCPDVBb16L4XQwDdbRUpz"], # Firework
    "Neutral": ["spotify:track:2fTIn1AqeTGR6tSSfiS8zk", "spotify:track:407ltk0BvEXlPztuv40Fjj"], # Weightless, Lofi
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_resources(model_path):
    try:
        model = load_model(model_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, face_cascade
    except Exception as e:
        return None, None

def init_spotify(client_id, client_secret):
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri='http://127.0.0.1:8888/callback',
            scope='user-read-playback-state user-modify-playback-state'
        ))
        return sp
    except:
        return None

def get_gpt_recommendation(client, emotion):
    try:
        prompt = f"Suggest a song that perfectly matches the emotion '{emotion}'. Return ONLY the string in this format: 'Artist - Song Title'"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        return None

def play_track(sp, query_or_uri, is_uri=False):
    """Plays track and returns track info + album art."""
    try:
        devices = sp.devices()
        active_device = next((d['id'] for d in devices['devices'] if d['is_active']), None)
        
        if not active_device:
            return None, None, "No Active Device"

        uri = query_or_uri
        track_info = {}

        if not is_uri:
            results = sp.search(q=query_or_uri, type='track', limit=1)
            if results['tracks']['items']:
                item = results['tracks']['items'][0]
                uri = item['uri']
                track_info = {
                    "name": item['name'],
                    "artist": item['artists'][0]['name'],
                    "image": item['album']['images'][0]['url']
                }
            else:
                return None, None, "Song Not Found"
        else:
            # If URI, fetch metadata separately to show art
            track_meta = sp.track(uri)
            track_info = {
                "name": track_meta['name'],
                "artist": track_meta['artists'][0]['name'],
                "image": track_meta['album']['images'][0]['url']
            }

        sp.start_playback(device_id=active_device, uris=[uri])
        return True, track_info, None

    except Exception as e:
        return False, None, str(e)

# ==========================================
# 4. MAIN APP
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Settings")
        with st.expander("🔑 API Keys", expanded=True):
            model_file = st.text_input("Model Path", "EmotionRecognition.h5")
            spotify_id = st.text_input("Spotify Client ID", type="password")
            spotify_secret = st.text_input("Spotify Client Secret", type="password")
            openai_key = st.text_input("OpenAI API Key", type="password")
        
        st.divider()
        run_app = st.toggle("Start Experience", value=False)
        st.info("Ensure Spotify is OPEN and PLAYING on your device first.")

    # --- Header ---
    st.title("🎧 VibeCheck: Emotion-Based DJ")
    st.caption("AI that sees how you feel and sets the mood.")

    # --- Session State ---
    if 'last_play_time' not in st.session_state: st.session_state.last_play_time = 0
    if 'current_track' not in st.session_state: st.session_state.current_track = None
    if 'current_emotion' not in st.session_state: st.session_state.current_emotion = "Neutral"

    # --- Layout ---
    col_cam, col_info = st.columns([2, 1.2])

    # Placeholders
    with col_cam:
        cam_placeholder = st.empty()
    
    with col_info:
        # Container for dynamic content
        with st.container(border=True):
            st.markdown("### 🎭 Current Vibe")
            col_stat1, col_stat2 = st.columns(2)
            emoji_placeholder = col_stat1.empty()
            conf_placeholder = col_stat2.empty()
            lottie_placeholder = st.empty()
        
        with st.container(border=True):
            st.markdown("### 🎵 Now Playing")
            art_placeholder = st.empty()
            song_info_placeholder = st.empty()

    # --- Main Loop ---
    if run_app:
        # Load Resources
        model, face_cascade = load_resources(model_file)
        if not model:
            st.error("Model not found! Check path.")
            return

        # Initialize Clients
        sp = init_spotify(spotify_id, spotify_secret) if spotify_id else None
        gpt = OpenAI(api_key=openai_key) if openai_key else None

        cap = cv2.VideoCapture(0)

        while run_app:
            ret, frame = cap.read()
            if not ret: break

            # Image Processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(48,48))

            emotion_label = "Neutral"
            confidence = 0.0

            if len(faces) > 0:
                # Get largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                # Draw stylish box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
                
                # Predict
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = roi.astype("float") / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))
                
                prediction = model.predict(roi, verbose=0)
                idx = np.argmax(prediction)
                emotion_label = EMOTION_LABELS[idx]
                confidence = np.max(prediction)

                # Overlay Text
                text = f"{emotion_label} ({int(confidence*100)}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)

            # Update Camera Feed
            cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Update Stats & Animations
            emoji_placeholder.metric("Emotion", emotion_label)
            conf_placeholder.metric("Confidence", f"{int(confidence*100)}%")
            
            # Update Lottie if emotion changed
            if st.session_state.current_emotion != emotion_label:
                lottie_json = load_lottieurl(LOTTIE_URLS.get(emotion_label, LOTTIE_URLS["Neutral"]))
                if lottie_json:
                    with lottie_placeholder:
                        st_lottie(lottie_json, height=150, key=f"anim_{time.time()}")
                st.session_state.current_emotion = emotion_label

            # Music Logic (15s cooldown)
            if sp and (time.time() - st.session_state.last_play_time > 15) and emotion_label != "Neutral":
                st.session_state.last_play_time = time.time()
                
                # 1. GPT Suggestion
                song_query = None
                if gpt:
                    song_query = get_gpt_recommendation(gpt, emotion_label)
                
                # 2. Play (GPT or Fallback)
                success = False
                track_data = None
                
                if song_query:
                    success, track_data, _ = play_track(sp, song_query)
                
                if not success:
                    # Fallback
                    uri = random.choice(FALLBACK_TRACKS.get(emotion_label, FALLBACK_TRACKS["Neutral"]))
                    success, track_data, _ = play_track(sp, uri, is_uri=True)

                # 3. Update Display
                if success and track_data:
                    st.session_state.current_track = track_data

            # Persistent Song Display
            if st.session_state.current_track:
                t = st.session_state.current_track
                art_placeholder.image(t['image'], width=150)
                song_info_placeholder.markdown(f"**{t['name']}**\n\n*{t['artist']}*")
            else:
                song_info_placeholder.info("Waiting for vibes...")

            time.sleep(0.1)

        cap.release()

if __name__ == "__main__":
    main()