# config.py

# --- Brand and Jingle Details ---
BRAND_NAME = "Kochi Fresh"
LYRICS = "Welcome to Kochi Fresh... [laughs] where every bite is a taste of the coast."
MUSIC_PROMPT = "A cheerful, upbeat, coastal brand jingle with ukulele, light drums, and a hint of flute melody. Sunny and inviting."

# --- File and Path Settings ---
# Note: The '.wav' extension will be added automatically
MUSIC_FILENAME = "background_music"
VOICE_FILENAME = "voice_track"
FINAL_JINGLE_FILENAME = f"jingle_{BRAND_NAME.lower().replace(' ', '_')}"

# --- Audio Mixing Settings ---
# Volume reduction for the background music in decibels (dB).
# A larger number means the music will be quieter. 6-10 is a good range.
MUSIC_VOLUME_REDUCTION = 8