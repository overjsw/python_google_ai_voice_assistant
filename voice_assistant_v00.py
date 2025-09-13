import os
import time
import queue
import threading
import numpy as np
import RPi.GPIO as GPIO
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai

# --- 설정 ---
# GPIO 설정
BUTTON_PIN = 17

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE_INDEX = None # None으로 두면 기본 마이크 사용
AUDIO_FILENAME = "user_audio.wav"
RESPONSE_FILENAME = "assistant_response.wav"

# 침묵 감지 설정
SILENCE_THRESHOLD = 0.01  # 침묵으로 간주할 오디오 볼륨 임계값
SILENCE_SECONDS = 5       # 침묵 지속 시간 (초)

# Google Cloud 설정
# TODO: 여기에 자신의 GCP 프로젝트 ID를 입력하세요.
GCP_PROJECT_ID = "august-craft-159401"

# Gemini API 설정
# TODO: 여기에 자신의 Gemini API 키를 입력하세요.
GEMINI_API_KEY = "AIzaSyBKY51wu4g1ilISsPZMy4ZznBuSb87sD98"

# AI 페르소나 설정 (초등학교 영어 교사)
SYSTEM_PROMPT = """
You are "Teacher Emily," a friendly and patient English teacher for elementary school students.
Your goal is to have a simple, encouraging, and educational conversation in English.
- Use simple vocabulary and short, easy-to-understand sentences.
- Always be cheerful and encouraging.
- Ask simple questions to keep the conversation going.
- Start the first conversation with a warm greeting like "Hello! I'm Teacher Emily. What's your name?"
"""

# --- 전역 변수 ---
q = queue.Queue()
recording = False
conversation_history = []

# --- Google Cloud 클라이언트 초기화 ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Gemini 모델 설정
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-latest',
    system_instruction=SYSTEM_PROMPT
)


def audio_callback(indata, frames, time, status):
    """오디오 스트림에서 데이터를 받아 큐에 넣는 콜백 함수"""
    if status:
        print(status)
    if recording:
        q.put(indata.copy())

def record_audio_with_silence_detection():
    """침묵을 감지할 때까지 오디오를 녹음하고 파일로 저장"""
    global recording
    recording = True
    
    with sf.SoundFile(AUDIO_FILENAME, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS) as file:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, device=DEVICE_INDEX):
            print("듣고 있어요...")
            silent_frames = 0
            total_frames = 0
            
            while True:
                try:
                    audio_chunk = q.get(timeout=1)
                    file.write(audio_chunk)
                    
                    # RMS를 계산하여 볼륨 측정
                    volume_norm = np.linalg.norm(audio_chunk) * 10
                    
                    if volume_norm < SILENCE_THRESHOLD:
                        silent_frames += len(audio_chunk)
                    else:
                        silent_frames = 0 # 말이 감지되면 초기화
                    
                    total_frames += len(audio_chunk)
                    
                    # 5초 이상 침묵이 지속되면 녹음 종료
                    if silent_frames > SILENCE_SECONDS * SAMPLE_RATE:
                        print("5초 동안 침묵이 감지되어 녹음을 종료합니다.")
                        break
                    # 사용자가 너무 길게 말하는 경우를 대비 (예: 60초)
                    if total_frames > 60 * SAMPLE_RATE:
                        print("최대 녹음 시간을 초과했습니다.")
                        break
                except queue.Empty:
                    # 큐가 비어있을 때(1초 타임아웃), 침묵으로 간주
                    silent_frames += SAMPLE_RATE
                    if silent_frames > SILENCE_SECONDS * SAMPLE_RATE:
                        print("5초 동안 침묵이 감지되어 녹음을 종료합니다.")
                        break
    
    recording = False
    # 녹음된 내용이 거의 없는 경우 (버튼만 누르고 말을 안 한 경우)
    if total_frames < SAMPLE_RATE * 0.5: # 0.5초 미만 녹음은 무시
        return False
    return True

def google_stt(file_path):
    """오디오 파일을 텍스트로 변환"""
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
            return ""
        
        transcript = response.results[0].alternatives[0].transcript
        print(f"나: {transcript}")
        return transcript
    except Exception as e:
        print(f"STT API 오류: {e}")
        return ""

def get_gemini_response(prompt):
    """Gemini API를 통해 AI 응답 생성"""
    global conversation_history
    print("AI가 응답을 생성 중입니다...")
    try:
        # 대화 기록에 사용자 메시지 추가
        conversation_history.append({'role': 'user', 'parts': [{'text': prompt}]})
        
        # API 호출 시 전체 대화 기록을 사용
        chat_session = gemini_model.start_chat(history=conversation_history[:-1]) # 마지막 사용자 입력 제외
        response = chat_session.send_message(prompt)
        
        ai_response = response.text
        
        # 대화 기록에 AI 응답 추가
        conversation_history.append({'role': 'model', 'parts': [{'text': ai_response}]})

        print(f"Emily 선생님: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        return "Sorry, I'm having a little trouble right now. Let's try again."

def google_tts(text):
    """텍스트를 음성으로 변환하고 파일로 저장"""
    print("텍스트를 음성으로 변환 중...")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",  # 친근한 여성 목소리
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(RESPONSE_FILENAME, "wb") as out:
            out.write(response.audio_content)
        return True
    except Exception as e:
        print(f"TTS API 오류: {e}")
        return False

def play_audio(file_path):
    """오디오 파일 재생"""
    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"오디오 재생 오류: {e}")

def cleanup():
    """종료 시 GPIO 정리 및 임시 파일 삭제"""
    GPIO.cleanup()
    if os.path.exists(AUDIO_FILENAME):
        os.remove(AUDIO_FILENAME)
    if os.path.exists(RESPONSE_FILENAME):
        os.remove(RESPONSE_FILENAME)
    print("\n프로그램을 종료합니다.")

def main():
    """메인 로직"""
    global conversation_history
    
    # GPIO 설정
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    print("프로그램이 시작되었습니다. 버튼을 눌러 대화를 시작하세요.")

    try:
        while True:
            # 버튼이 눌릴 때까지 대기 (Falling edge 감지)
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
            
            # 디바운싱
            time.sleep(0.2)
            
            print("\n------ 새 대화 시작 ------")
            play_audio("start_beep.wav") # 시작 알림음 (필요 시 파일 생성)
            
            if not record_audio_with_silence_detection():
                print("녹음된 내용이 없어 대화를 건너뜁니다.")
                continue

            user_text = google_stt(AUDIO_FILENAME)

            if user_text:
                ai_response_text = get_gemini_response(user_text)
                if google_tts(ai_response_text):
                    play_audio(RESPONSE_FILENAME)
            else:
                print("음성을 인식하지 못했습니다.")
                play_audio("error_beep.wav") # 에러 알림음 (필요 시 파일 생성)
            
            print("다음 질문을 하시려면 버튼을 다시 눌러주세요.")


    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    # 간단한 시작/에러 알림음 생성 (없을 경우)
    if not os.path.exists("start_beep.wav"):
        t = np.linspace(0., 0.2, int(0.2 * 44100), False)
        sine = np.sin(t * 2 * np.pi * 660) * 0.5
        sf.write("start_beep.wav", sine, 44100)
    
    if not os.path.exists("error_beep.wav"):
        t = np.linspace(0., 0.3, int(0.3 * 44100), False)
        sine = np.sin(t * 2 * np.pi * 220) * 0.5
        sf.write("error_beep.wav", sine, 44100)
        
    # 대화 기록 초기화
    conversation_history.append({'role': 'user', 'parts': [{'text': 'Start conversation'}]})
    ai_first_greeting = get_gemini_response("Start conversation")
    google_tts(ai_first_greeting)
    
    # 첫 인사말 재생
    play_audio(RESPONSE_FILENAME)
    
    main()
