from flask import Flask, request
from flask_socketio import SocketIO, emit
import logging
import base64
import numpy as np
from deepspeech import Model
import os
import sys
import io
from gtts import gTTS
import time
import ffmpeg
from openai import OpenAI
from collections import defaultdict
import argparse

# Set up the Flask application
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load DeepSpeech model
model_file_path = 'deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 500
model = Model(model_file_path)
model.enableExternalScorer(scorer_file_path)
model.setScorerAlphaBeta(alpha=0.75, beta=1.85)
model.setBeamWidth(beam_width)

SAMPLE_RATE = 16000

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the application with OpenAI API")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    return parser.parse_args()

args = parse_arguments()

# OpenAI API setup
if args.api_key:
    client = OpenAI(api_key=args.api_key)
else:
    raise ValueError("Please provide the OpenAI API key using the --api_key flag")

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a single log handler
file_handler = logging.FileHandler('logs/server.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Dictionary to store conversation history for each user
conversation_histories = defaultdict(list)

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid} from {request.remote_addr}')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')
    if request.sid in conversation_histories:
        del conversation_histories[request.sid]

def validate_webm(webm_data):
    try:
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='null')
            .run(input=webm_data, capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"Invalid WebM file: {e.stderr.decode()}")
        return False

def convert_webm_to_wav(webm_data):
    try:
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run(input=webm_data, capture_stdout=True, capture_stderr=True)
        )
        return process[0]
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        logger.info(f"Received audio chunk. Size: {len(data['audio'])} bytes")

        audio_data = base64.b64decode(data['audio'])
        logger.info(f"Decoded audio data size: {len(audio_data)} bytes")

        if not audio_data:
            raise ValueError("Received empty audio data")

        if not validate_webm(audio_data):
            raise ValueError("Invalid WebM audio data received")

        wav_data = convert_webm_to_wav(audio_data)

        audio_np = np.frombuffer(wav_data, dtype=np.int16)

        transcription_start = time.time()
        text = model.stt(audio_np)
        transcription_time = time.time() - transcription_start
        logger.info(f"ASR result: {text} (processing time: {transcription_time:.2f}s)")

        if text.strip():
            llm_start = time.time()
            llm_response = llm_process(text, request.sid)
            llm_time = time.time() - llm_start
            logger.info(f"LLM response: {llm_response} (processing time: {llm_time:.2f}s)")

            tts_start = time.time()
            tts_audio = tts_process(llm_response)
            tts_time = time.time() - tts_start
            logger.info(f"TTS audio size: {len(tts_audio)} bytes (processing time: {tts_time:.2f}s)")

            emit('response', {
                'text': text,
                'llm_response': llm_response,
                'audio': base64.b64encode(tts_audio).decode('utf-8')
            })
            logger.info("Emitted response")
        else:
            logger.info("Empty transcription, no response sent")

        emit('ready_for_input')
        logger.info("Emitted ready_for_input")

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        emit('error', {'message': f"Error processing audio: {str(e)}"})

def llm_process(text, user_id):
    conversation_history = conversation_histories[user_id]
    conversation_history.append({"role": "user", "content": text})

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + conversation_history

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    llm_response = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": llm_response})

    return llm_response

def tts_process(text):
    try:
        tts = gTTS(text, lang='en')
        wav_io = io.BytesIO()
        tts.write_to_fp(wav_io)
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        logger.error(f"Error in TTS processing: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
