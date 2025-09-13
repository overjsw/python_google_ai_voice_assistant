# -*- coding: utf-8 -*-
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai

# --- 설정 (Configuration) ---

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


# --- 핵심 기능 함수 ---

def get_microphone_device_id():
    """시스템에서 사용 가능한 마이크 장치 ID를 찾습니다. (개선된 로직)"""
    try:
        devices = sd.query_devices()
        
        # 1. 키워드 기반으로 마이크 검색
        for index, device in enumerate(devices):
            device_name = device['name'].lower()
            if any(keyword in device_name for keyword in ['mic', 'usb', 'seeed', 'voicecard']):
                if device['max_input_channels'] > 0:
                    print(f"키워드 검색으로 마이크를 찾았습니다: ID {index} - {device['name']}")
                    return index
        
        # 2. 키워드 검색 실패 시, 시스템의 기본 입력 장치 확인
        try:
            default_device_index = sd.default.device[0] # [0] for input
            device_info = sd.query_devices(default_device_index, 'input')
            if device_info['max_input_channels'] > 0:
                print(f"시스템 기본 입력 장치를 마이크로 사용합니다: ID {default_device_index} - {device_info['name']}")
                return default_device_index
        except (ValueError, IndexError, TypeError):
             pass

    except Exception as e:
        print(f"오디오 장치 목록을 가져오는 중 오류 발생: {e}")

    # 3. 모든 방법 실패 시, 전체 장치 목록을 진단 정보로 출력 후 None 반환
    print("\n[진단 정보] 아래는 Sounddevice가 인식하는 전체 오디오 장치 목록입니다.")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"장치 목록을 출력하는 중에도 오류가 발생했습니다: {e}")
    return None

def record_audio_with_silence_detection(device_id):
    """사용자의 음성을 묵음이 감지될 때까지 녹음합니다. (안정적인 블로킹 방식으로 변경)"""
    print("음성 녹음 시작... 말씀하세요.")
    
    chunks_per_second = 10
    chunk_size = int(SAMPLE_RATE / chunks_per_second)
    silent_chunks_needed = SILENCE_SECONDS * chunks_per_second
    
    audio_data = []
    silent_chunks_count = 0

    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=chunk_size) as stream:
            print("녹음 스트림이 열렸습니다. 묵음 감지를 시작합니다.")
            while True:
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("경고: 녹음 중 오디오 버퍼 오버플로우 발생")
                
                audio_data.append(chunk)
                volume_norm = np.linalg.norm(chunk) * 10

                if volume_norm < SILENCE_THRESHOLD:
                    silent_chunks_count += 1
                else:
                    silent_chunks_count = 0
                
                if silent_chunks_count > silent_chunks_needed:
                    print("묵음이 감지되어 녹음이 중지되었습니다.")
                    break
    except Exception as e:
        print(f"녹음 중 오류 발생: {e}")
        return False

    if not audio_data:
        print("녹음된 오디오 데이터가 없습니다.")
        return False
        
    recording = np.concatenate(audio_data, axis=0)
    sf.write(FILENAME_INPUT, recording, SAMPLE_RATE)
    print(f"오디오 파일 '{FILENAME_INPUT}' 저장 완료.")
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
    mic_device_id = get_microphone_device_id()
    if mic_device_id is None:
        print("\n[오류] 사용 가능한 마이크를 찾을 수 없습니다.")
        print("마이크가 올바르게 연결되었는지, 시스템에서 인식하는지 확인해주세요.")
        return

    initial_prompt = "Hello! I'm Teacher Emily. Let's talk!"
    print(f"Emily 선생님: {initial_prompt}")
    if google_tts(initial_prompt):
        play_audio(FILENAME_OUTPUT)
    
    global conversation_history
    conversation_history[1]['parts'] = [initial_prompt]

    try:
        while True:
            if record_audio_with_silence_detection(mic_device_id):
                user_text = google_stt(FILENAME_INPUT)
                if user_text:
                    print(f"나: {user_text}")
                    ai_text = get_gemini_response(user_text)
                    if ai_text and google_tts(ai_text):
                        play_audio(FILENAME_OUTPUT)
                else:
                    goodbye_message = "It looks like you're busy. Let's talk again next time. Goodbye!"
                    print(f"Emily 선생님: {goodbye_message}")
                    if google_tts(goodbye_message):
                        play_audio(FILENAME_OUTPUT)
                    print("\n5초 동안 음성 입력이 없어 프로그램을 종료합니다.")
                    break
            else:
                print("녹음에 실패하여 프로그램을 종료합니다.")
                break
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")
    finally:
        if os.path.exists(FILENAME_INPUT):
            os.remove(FILENAME_INPUT)
        if os.path.exists(FILENAME_OUTPUT):
            os.remove(FILENAME_OUTPUT)
        print("임시 파일 정리 완료.")

if __name__ == "__main__":
    main()


