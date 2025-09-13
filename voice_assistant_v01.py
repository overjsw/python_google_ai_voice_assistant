# -*- coding: utf-8 -*-
import os
import time  # 디바운싱을 위해 time 모듈 추가
import numpy as np
import RPi.GPIO as GPIO
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai

# --- 설정 (Configuration) ---

# GPIO 핀 설정
BUTTON_PIN = 17

# 오디오 설정
SAMPLE_RATE = 16000  # 음성 인식에 권장되는 샘플 속도
CHANNELS = 1
FILENAME_INPUT = "user_audio.wav"
FILENAME_OUTPUT = "assistant_audio.mp3"

# 묵음 감지 설정
SILENCE_THRESHOLD = 0.01  # 묵음으로 간주할 오디오 볼륨 임계값
SILENCE_SECONDS = 5       # 이 시간(초) 동안 묵음이 지속되면 녹음 중지

# Google Cloud 설정
# GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되어 있어야 합니다.
GCP_PROJECT_ID = "august-craft-159401"

# Gemini API 설정
# TODO: 여기에 자신의 Gemini API 키를 입력하세요.
GEMINI_API_KEY = "AIzaSyBKY51wu4g1ilISsPZMy4ZznBuSb87sD98"

# AI 페르소나 설정 (초등학교 영어 교사)
SYSTEM_PROMPT = """
You are 'Teacher Emily', a friendly and patient English teacher for elementary school students.
Your personality is cheerful, encouraging, and kind.
Your primary goal is to help young children practice speaking English in a fun and comfortable way.
Follow these rules strictly:
1.  Always speak in simple, easy-to-understand English, suitable for a 7-year-old child.
2.  Use short sentences.
3.  Ask simple, open-ended questions to encourage the child to speak. (e.g., "What's your favorite color?", "What did you do today?")
4.  Be very patient. If the user's response is unclear or grammatically incorrect, gently guide them without direct correction.
5.  Keep your responses concise, typically 1-2 sentences.
6.  Start the first conversation by introducing yourself and asking for the user's name.
"""

# --- 초기화 (Initialization) ---

# Gemini API 클라이언트 설정
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    print(f"Gemini API 설정 중 오류 발생: {e}")
    print("Gemini API 키가 올바르게 설정되었는지 확인하세요.")
    exit()

# Google Cloud 클라이언트 설정
try:
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"Google Cloud 클라이언트 설정 중 오류 발생: {e}")
    print("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되었는지 확인하세요.")
    exit()

# 대화 기록 초기화
conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': [""]}]

# GPIO 설정
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# 내부 풀업 저항을 활성화하여 플로팅 상태를 방지합니다.
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# --- 핵심 기능 함수 ---

def record_audio_with_silence_detection():
    """사용자의 음성을 묵음이 감지될 때까지 녹음합니다."""
    print("음성 녹음 시작... 말씀하세요.")
    audio_data = []
    silent_chunks = 0
    chunk_size = int(SAMPLE_RATE / 10) # 0.1초 단위로 처리

    def callback(indata, frames, time, status):
        nonlocal silent_chunks
        volume_norm = np.linalg.norm(indata) * 10
        audio_data.append(indata.copy())

        if volume_norm < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        # 0.1초짜리 청크가 50개 모이면 5초
        if silent_chunks > (SILENCE_SECONDS * 10):
            raise sd.CallbackStop

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=chunk_size):
            while True: # CallbackStop 예외가 발생할 때까지 계속 실행
                sd.sleep(100)
    except sd.CallbackStop:
        print("묵음이 감지되어 녹음이 중지되었습니다.")
    except Exception as e:
        print(f"녹음 중 오류 발생: {e}")
        return False

    sf.write(FILENAME_INPUT, np.concatenate(audio_data), SAMPLE_RATE)
    return True

def google_stt(file_path):
    """Google Speech-to-Text API를 사용하여 오디오 파일을 텍스트로 변환합니다."""
    print("음성을 텍스트로 변환 중...")
    try:
        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
        )
        response = speech_client.recognize(config=config, audio=audio)

        if not response.results:
            print("음성을 인식하지 못했습니다.")
            return None
        return response.results[0].alternatives[0].transcript
    except Exception as e:
        print(f"STT API 오류: {e}")
        return None

def get_gemini_response(text):
    """Gemini AI에게 텍스트를 보내 응답을 받습니다."""
    global conversation_history
    print("AI가 응답을 생성 중입니다...")
    try:
        # 대화 기록에 사용자 메시지 추가
        conversation_history.append({'role': 'user', 'parts': [text]})
        
        # 모델로부터 응답 생성
        chat = gemini_model.start_chat(history=conversation_history)
        response = chat.send_message(text, stream=False)
        
        ai_response = response.text
        
        # 대화 기록에 모델 응답 추가
        conversation_history.append({'role': 'model', 'parts': [ai_response]})
        print(f"Emily 선생님: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        return "Sorry, I'm having a little trouble thinking right now."

def google_tts(text):
    """Google Text-to-Speech API를 사용하여 텍스트를 음성으로 변환합니다."""
    print("텍스트를 음성으로 변환 중...")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F", # 친근한 여성 목소리
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(FILENAME_OUTPUT, "wb") as out:
            out.write(response.audio_content)
        return True
    except Exception as e:
        print(f"TTS API 오류: {e}")
        return False

def play_audio(file_path):
    """오디오 파일을 재생합니다."""
    print("음성 응답을 재생합니다.")
    os.system(f"mpg321 {file_path}")

# --- 메인 실행 로직 ---

def main():
    """메인 프로그램 루프"""
    # 첫 인사말 생성 및 재생
    initial_prompt = "Hello! I'm Teacher Emily. What's your name?"
    if google_tts(initial_prompt):
        play_audio(FILENAME_OUTPUT)
    
    # conversation_history에 첫 인사말 기록 추가
    global conversation_history
    conversation_history[1]['parts'] = [initial_prompt]

    try:
        while True:
            print("\n프로그램이 시작되었습니다. 버튼을 눌러 대화를 시작하세요.")
            # FALLING 엣지를 기다림 (버튼이 눌리는 순간)
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
            
            # 디바운싱: 버튼의 물리적 떨림으로 인한 중복 입력을 방지
            time.sleep(0.2)
            
            print("버튼이 감지되었습니다!")

            if record_audio_with_silence_detection():
                user_text = google_stt(FILENAME_INPUT)
                if user_text:
                    print(f"나: {user_text}")
                    ai_text = get_gemini_response(user_text)
                    if ai_text and google_tts(ai_text):
                        play_audio(FILENAME_OUTPUT)
                else:
                    # 음성 인식을 못했을 때 안내 음성
                    if google_tts("Sorry, I couldn't hear you. Could you please say that again?"):
                        play_audio(FILENAME_OUTPUT)
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")
    finally:
        # 프로그램 종료 시 GPIO 핀 정리
        GPIO.cleanup()
        # 임시 오디오 파일 삭제
        if os.path.exists(FILENAME_INPUT):
            os.remove(FILENAME_INPUT)
        if os.path.exists(FILENAME_OUTPUT):
            os.remove(FILENAME_OUTPUT)
        print("GPIO 및 임시 파일 정리 완료.")

if __name__ == "__main__":
    main()


