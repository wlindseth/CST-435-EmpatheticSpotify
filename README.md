# Emotion-Based Music Player

This Streamlit application uses your webcam to detect your facial emotions in real-time and automatically plays matching music on Spotify. It utilizes a deep learning model for emotion recognition and OpenAI's GPT to generate song suggestions.

## Prerequisites

1.  **Python 3.8+**
2.  **Spotify Premium Account** (Required to control playback via API).
3.  **Spotify Developer App**:
    - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
    - Create a new App.
    - Get your `Client ID` and `Client Secret`.
    - **Crucial:** Click "Edit Settings" and add `http://127.0.0.1:8888/callback` to the "Redirect URIs" section. Save the changes.
4.  **OpenAI API Key** (Optional, but recommended for dynamic suggestions).

## Installation

1.  Clone this project or download the files into a folder.
2.  Install the required dependencies by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model File:** Ensure you have your trained model file (e.g., `EmotionRecognition.h5`) inside this directory. If you haven't trained one yet, you must run your Jupyter Notebook training script first to generate this file.

## Running the App

1.  Open your terminal in the project directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  The app will automatically open in your default web browser (usually at `http://localhost:8501`).

## How to Use

1.  **Setup Configuration (Sidebar):**
    - **Model Path:** Enter the filename of your `.h5` model (default is `EmotionRecognition.h5`).
    - **Spotify Keys:** Paste your **Spotify Client ID** and **Client Secret**.
    - **OpenAI Key:** Paste your **OpenAI API Key**.
2.  **Prepare Spotify:** Open the Spotify desktop or mobile app on your device and start playing _any_ song. This registers the device as "Active" so the Python script can take control.
3.  **Start Detection:** Check the **"Start Camera"** box in the sidebar.
4.  **Experience:** The app will detect your emotion and change the song to match your mood every 15 seconds (if the emotion is stable).

## Troubleshooting

- **"No active Spotify device found":** You must have Spotify open and playing music (or paused) on the device you are running the code on. The API cannot wake up a closed Spotify application.
- **"Model not found":** Make sure the `.h5` file name in the sidebar input matches the actual file name in your folder.
- **"INVALID_CLIENT: Invalid redirect URI":** Ensure `http://127.0.0.1:8888/callback` is exactly what is listed in your Spotify Developer Dashboard settings.
# CST-435-EmpatheticSpotify
