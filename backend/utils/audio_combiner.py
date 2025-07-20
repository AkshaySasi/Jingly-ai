# utils/audio_combiner.py
from pydub import AudioSegment

def combine_audio(music_file: str, voice_file: str, output_file: str, music_reduction_db: int) -> str:
    """
    Overlays a voice track onto a music track.
    """
    print("🎛️ Combining audio tracks...")
    music = AudioSegment.from_wav(music_file)
    voice = AudioSegment.from_wav(voice_file)

    # Reduce music volume to make voice audible
    quieter_music = music - music_reduction_db
    
    # Overlay voice onto music
    final_audio = quieter_music.overlay(voice)
    
    final_audio.export(output_file, format="wav")
    print(f"✅ Final jingle saved to: {output_file}")
    return output_file