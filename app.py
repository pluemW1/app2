import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import boto3
import os
import soundfile as sf
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from PIL import Image

# AWS and model settings...

# Your model loading code remains the same...

def preprocess_audio_file(file_path, max_pad_len=174):
    # Your audio preprocessing code remains the same...
    pass

class AudioProcessor(AudioProcessorBase):
    # Your audio processing class remains the same...
    pass

# Load your watermelon images
unripe_image = Image.open("image/watermelon_unripe.jpg")
semiripe_image = Image.open("image/watermelon_semiripe.jpg")
ripe_image = Image.open("image/watermelon_ripe.jpg")

# Your WebRTC streaming code remains the same...

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียงหรือวิดีโอ", type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format='audio/wav')

    try:
        if file_path.endswith(('.mp4', '.mov', '.avi')):
            video = VideoFileClip(file_path)
            audio = video.audio
            audio.write_audiofile("temp_audio.wav")
            file_path = "temp_audio.wav"

        processed_data = preprocess_audio_file(file_path)
        prediction = model.predict(np.expand_dims(processed_data, axis=0))
        predicted_class = np.argmax(prediction)

        # Map the prediction to display the corresponding image and label
        if predicted_class == 0:
            result = 'แตงโมไม่สุก (แตงโมที่มีเนื้อเป็นขาวอมชมพู)'
            st.image(unripe_image, caption=result)
        elif predicted_class == 1:
            result = 'แตงโมกึ่งสุก (แตงโมที่มีเนื้อเป็นสีแดงอ่อน)'
            st.image(semiripe_image, caption=result)
        else:
            result = 'แตงโมสุก (แตงโมที่มีเนื้อเป็นสีแดงเข้ม)'
            st.image(ripe_image, caption=result)

        # Display confidence score
        confidence = np.max(prediction)
        st.write(f"ความมั่นใจของการทำนาย: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
