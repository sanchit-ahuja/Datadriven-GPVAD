import os
import numpy as np
import soundfile as sf
"""
    Take an input as a audio_path and chunk it into smaller chunks of size chunk_size

"""

def chunk_audio(audio_path, chunk_size, save=False):
    """
        Get 500 ms chunks from audio file
    """
    audio_data, sample_rate = sf.read(audio_path)

    """Chunk audio into chunks of size chu and save them in a folder named chunks"""
    file = audio_path
    audio, fs = sf.read(file)
    chunk_size = int(chunk_size * fs)
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        chunks.append(chunk)

    # Save chunks
    if save:
        folder_path = "chunks"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Save audio
        for i, chunk in enumerate(chunks):
            sf.write(os.path.join(folder_path, str(i)+".wav"), chunk, fs)
    return chunks

if __name__ == "__main__":
    audio_path = '/home/sanchit/res_vad/data/chola_audio_clean/hermes-cca%2F2021-11-05%2F8abc664d-467b-4c48-96d4-4e97d63a4763%2F28744e27-041d-47e6-be93-6a377ed57ff5.wav'
    chunked_audio_path = chunk_audio(audio_path, 0.5)