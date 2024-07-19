# Voice AI Chat Application

This is a voice AI chat application that utilises Flask, SocketIO, DeepSpeech for ASR (Automatic Speech Recognition), OpenAI for language processing, and gTTS for TTS (Text-to-Speech).

## Features

- Real-time voice chat using WebSockets
- Automatic Speech Recognition (ASR) using DeepSpeech
- Natural language processing using OpenAI's GPT-3.5-turbo
- Text-to-Speech (TTS) using Google Text-to-Speech (gTTS)
- Comprehensive logging and error handling
- Handles multiple users simultaneously

## Requirements

Requirements include:
- Python 3.9 or higher
- Flask
- Flask-SocketIO
- DeepSpeech
- gTTS
- ffmpeg
- OpenAI
- numpy

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/Sharday/voice-ai-chat.git
cd voice-ai-chat
```

### Dependencies

Please install the required packages in your environment using:

```bash
pip install -r requirements.txt
```

#### Install DeepSpeech
Download and install the DeepSpeech model files:

```bash
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
#### Install FFmpeg
Ensure FFmpeg is installed on your system. You can download it from [FFmpeg's official site](https://www.ffmpeg.org/download.html).

#### Set Up OpenAI API Key
Obtain an API key from OpenAI and set it up in the application. You can pass the API key as a command-line argument when running the application.





### Run the Application
Start the Flask application:

```bash
python app.py --api_key=your_openai_api_key_here
```

### Access the Application
Open your web browser and navigate to the path of index.html. For example, if you are running the application locally, open file:///path/to/voice-ai-chat/index.html in your browser.

### Usage
#### Starting and Stopping Conversation
 - Click the "Start" button to begin the conversation.
 - Speak into your microphone.
 - The system will transcribe your speech, generate a response using OpenAI, convert the response to speech, and play it back to you.
 - Click the "Stop" button to end the conversation.

#### Troubleshooting
 - Ensure your microphone is properly set up and has the necessary permissions.
 - Check the logs/server.log file for any error messages if something goes wrong.
 - Verify that the OpenAI API key is correct and has sufficient quota.
