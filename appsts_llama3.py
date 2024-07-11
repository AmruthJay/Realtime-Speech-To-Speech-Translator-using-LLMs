import gradio as gr
import requests
import whisper
import json
from gtts import gTTS
import os
import ollama

# Load the Whisper model
modelname = "tiny"
model = whisper.load_model(modelname)

# Define function to transcribe audio, generate translation, and convert to speech
def transcribe_translate_tts(file_path):
    # Transcribe audio
    result = model.transcribe(file_path, fp16=False, language="en")
    transcription_text = result['text']
    
    # Prepare prompt for translation
    prompt = f"translate this English text into french precisely.you MUST translate the whole text.give just the translated sentence. no nonsense explanation. dont change the meaning of translated sentence. translate as it is , be crystal clear and precise, stick to the original meaning provided and get the emotions right. never get 1st, 2nd, 3rd (he, she, it, they) persons and genders (only male and female [he and she]) wrong. get the places, locations and the directions right. translation is must:"
    try:
        response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': prompt+" "+ transcription_text}        ],
    )
        translation_text= response['message']['content']  


        # Convert translated text to speech
        tts = gTTS(text=translation_text, lang='en')
        tts.save("translated_audio.mp3")
        
        return transcription_text, translation_text, "translated_audio.mp3"

    except requests.exceptions.RequestException as e:
        return transcription_text, f"Error during API request: {e}", None

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe_translate_tts,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Translation"), gr.Audio(label="Translated Audio")],
    title="Accent Handling and Translation",
    description="Upload an audio file to transcribe it to English text, translate the transcription, and convert the translation to speech."
)

# Launch the interface with public link sharing
iface.launch(share=True)
