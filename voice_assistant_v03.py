# -*- coding: utf-8 -*-
import os
import subprocess
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai

# --- 설정 (Configuration) ---

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 2 # Seeed 2-mic hat은 스테레오(2채널)만 지원
FILENAME_INPUT = "user_audio.wav" # arecord의 녹음 파일
FILENAME_OUTPUT = "assistant_audio.mp3"
DEVICE_NAME = "hw:1,0" # 'arecord -l'로 확인된 장치 이름

# 묵음 감지 설정 (arecord는 묵음 감지 기능이 없으므로 고정 시간 녹음)
RECORD_SECONDS = 7 # 사용자가 말할 시간을 7초로 설정 (조절 가능)

# Gemini API 설정
# TODO: 여기에 자신의 Gemini API 키를 입력하세요.
GEMINI_API_KEY = "AIzaSyBKY51wu4g1ilISsPZMy4ZznBuSb87sD98"

# AI 페르소나 설정
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


# --- 핵심 기능 함수 ---

def record_audio(duration):
    """arecord를 사용하여 오디오를 녹음합니다."""
    print(f"{duration}초 동안 음성을 녹음합니다...")
    
    # arecord로 스테레오 녹음
    arecord_command = [
        'arecord', '-D', DEVICE_NAME, '-f', 'S16_LE', '-r', str(SAMPLE_RATE),
        '-c', str(CHANNELS), '-d', str(duration), FILENAME_INPUT
    ]
    try:
        # 기존 파일이 있다면 덮어쓰기 위해 삭제
        if os.path.exists(FILENAME_INPUT):
            os.remove(FILENAME_INPUT)
        
        subprocess.run(arecord_command, check=True, stderr=subprocess.PIPE)
        print("녹음 완료.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"'arecord' 녹음 실패: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"알 수 없는 녹음 오류: {e}")
        return False


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
            audio_channel_count=CHANNELS,
            enable_separate_recognition_per_channel=False
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
        conversation_history.append({'role': 'user', 'parts': [text]})
        chat = gemini_model.start_chat(history=conversation_history)
        response = chat.send_message(text, stream=False)
        ai_response = response.text
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
            language_code="en-US", name="en-US-Wavenet-F"
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
    os.system(f"mpg321 -q {file_path}")

# --- 메인 실행 로직 ---

def main():
    """메인 프로그램 루프"""
    initial_prompt = "Hello! I'm Teacher Emily. Let's talk!"
    print(f"Emily 선생님: {initial_prompt}")
    if google_tts(initial_prompt):
        play_audio(FILENAME_OUTPUT)
    
    global conversation_history
    conversation_history[1]['parts'] = [initial_prompt]

    try:
        while True:
            if record_audio(RECORD_SECONDS):
                user_text = google_stt(FILENAME_INPUT)
                if user_text:
                    print(f"나: {user_text}")
                    ai_text = get_gemini_response(user_text)
                    if ai_text and google_tts(ai_text):
                        play_audio(FILENAME_OUTPUT)
                else:
                    # 7초 동안 말을 안하면 녹음은 되지만 STT 결과가 없음
                    continue # 다시 녹음 시작
            else:
                print("녹음에 실패하여 프로그램을 종료합니다.")
                break
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        # 임시 파일 모두 정리
        for f in [FILENAME_INPUT, FILENAME_OUTPUT]:
            if os.path.exists(f):
                os.remove(f)
        print("임시 파일 정리 완료.")

if __name__ == "__main__":
    main()


