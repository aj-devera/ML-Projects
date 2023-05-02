import os
import uuid
import shutil
import whisper
import uvicorn
from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
stt_model = whisper.load_model('base')
sentiment_model = pipeline("sentiment-analysis")

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/transcribe_and_predict")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded audio file to disk
    temp_file = f"{str(uuid.uuid4())}.wav"
    with open(temp_file, "wb") as audio:
        shutil.copyfileobj(file.file, audio)
    
    # Load the audio file and transcribe it using SpeechRecognition
    transcript = stt_model.transcribe(audio=temp_file)['text']
    sentiment = sentiment_model(transcript)[0]

    # Clean up the temporary audio file
    os.remove(temp_file)
    
    # Return the transcript as a response
    return {"transcript": transcript, "sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)