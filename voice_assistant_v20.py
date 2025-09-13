# -*- coding: utf-8 -*-
import os
import numpy as np
import sounddevice as sd
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai
import time
import json
import atexit
import re
import subprocess
try:
    from pixel_ring import pixel_ring
except ImportError:
    print("pixel_ring 라이브러리를 찾을 수 없습니다.")
    print("'pip install pixel_ring seeed-voicecard' 명령어로 설치해주세요.")
    pixel_ring = None
try:
    # [수정] RPi.GPIO 대신 gpiozero를 import 합니다.
    from gpiozero import Button
except (ImportError, RuntimeError):
    print("gpiozero 라이브러리를 찾을 수 없거나, 이 시스템에서 지원되지 않습니다.")
    print("'pip install gpiozero'로 설치해주세요.")
    Button = None

# --- 설정 파일 이름 (Configuration Filenames) ---
USER_CONFIG_FILE = "user.json"
API_KEY_FILE = "gemini_api_key.json"
HISTORY_FILE = "conversation_history.json"
SUMMARY_FILE = "conversation_summary.json"
PERSONA_FILE = "persona.json"

# --- 하드웨어 설정 ---
BUTTON_PIN = 17

# --- 설정 로드 함수 ---
def load_json_file(filename, purpose):
    """지정된 JSON 파일을 로드하고 데이터를 반환합니다."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[오류] {purpose} 파일('{filename}')을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"[오류] {purpose} 파일('{filename}')의 형식이 잘못되었습니다.")
        return None
    except Exception as e:
        print(f"'{filename}' 파일을 읽는 중 오류 발생: {e}")
        return None

# --- 설정 및 초기화 ---

# 설정 파일 로드
user_config = load_json_file(USER_CONFIG_FILE, "사용자 정보")
api_key_config = load_json_file(API_KEY_FILE, "Gemini API 키")
persona_config = load_json_file(PERSONA_FILE, "AI 페르소나")

if not user_config or not api_key_config or not persona_config:
    exit()

USER_NAME = user_config.get("name", "friend")
USER_INTRODUCTION = user_config.get("introduction", "a student")
GEMINI_API_KEY = api_key_config.get("api_key")

if not GEMINI_API_KEY or "your-gemini-api-key" in GEMINI_API_KEY:
    print(f"[오류] '{API_KEY_FILE}'에 유효한 Gemini API 키를 입력해주세요.")
    exit()

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 2
FILENAME_OUTPUT = "assistant_audio.mp3"

# 묵음 감지 설정
SILENCE_THRESHOLD = 3000

# AI 페르소나 설정
persona_rules = "\n".join([f"{i+1}.  {rule}" for i, rule in enumerate(persona_config.get("rules", []))])
SYSTEM_PROMPT = f"""
{persona_config.get("description", "")}
{persona_config.get("user_context", "").format(USER_NAME=USER_NAME, USER_INTRODUCTION=USER_INTRODUCTION)}
{persona_config.get("language_rule", "")}
Follow these rules strictly:
{persona_rules}
{persona_config.get("greeting", "").format(USER_NAME=USER_NAME)}
"""

# LED 초기화
if pixel_ring:
    pixel_ring.set_brightness(10)
    atexit.register(pixel_ring.off)

# Gemini API 클라이언트 설정
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Gemini API 설정 중 오류 발생: {e}")
    exit()

# Google Cloud 클라이언트 설정
try:
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"Google Cloud 클라이언트 설정 중 오류 발생: {e}")
    exit()

# 대화 기록 변수
conversation_history = []


# --- 핵심 기능 함수 ---

def save_history():
    global conversation_history
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"대화 기록 저장 중 오류 발생: {e}")

def get_microphone_device_id():
    try:
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if 'seeed-2mic-voicecard' in device['name'] and device['max_input_channels'] > 0:
                print(f"지정된 마이크를 찾았습니다: ID {index} - {device['name']}")
                return index
    except Exception as e:
        print(f"오디오 장치 목록을 가져오는 중 오류 발생: {e}")
    print("[오류] 적합한 마이크를 찾지 못했습니다.")
    return None

def calibrate_silence_threshold(device_id):
    global SILENCE_THRESHOLD
    print("\n묵음 기준 자동 보정을 시작합니다. 5초간 조용한 상태를 유지해주세요...")
    calibration_duration = 5
    volume_levels = []
    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                chunk, _ = stream.read(1024)
                volume_levels.append(np.linalg.norm(chunk))
        if volume_levels:
            average_noise = np.mean(volume_levels)
            SILENCE_THRESHOLD = int((average_noise * 1.8) + 500)
            print(f"묵음 기준 보정 완료! 새로운 기준: {SILENCE_THRESHOLD}")
        else:
            print("소음 측정에 실패하여 기본값을 사용합니다.")
    except Exception as e:
        print(f"묵음 기준 보정 중 오류 발생: {e}. 기본값을 사용합니다.")
    print("-" * 20)


def record_and_stream_stt(device_id):
    print("음성 녹음 및 실시간 STT 시작... 말씀하세요.")
    if pixel_ring: pixel_ring.set_color(g=255)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        alternative_language_codes=["ko-KR"],
        audio_channel_count=CHANNELS,
        enable_separate_recognition_per_channel=False
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)

    chunks_per_second = 20
    chunk_size = int(SAMPLE_RATE / chunks_per_second)
    stream_silence_seconds = 2
    silent_chunks_needed = stream_silence_seconds * chunks_per_second
    start_timeout_seconds = 10
    
    silent_chunks_count = 0
    speech_started = False
    transcript = ""

    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=chunk_size) as stream:
            def audio_generator():
                nonlocal speech_started, silent_chunks_count
                start_time = time.time()
                while True:
                    chunk, overflowed = stream.read(chunk_size)
                    if overflowed: print("경고: 녹음 중 오디오 버퍼 오버플로우 발생")
                    
                    yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
                    
                    volume_norm = np.linalg.norm(chunk)
                    
                    if not speech_started:
                        if volume_norm > SILENCE_THRESHOLD:
                            speech_started = True
                        elif time.time() - start_time > start_timeout_seconds:
                            break
                    else:
                        if volume_norm < SILENCE_THRESHOLD:
                            silent_chunks_count += 1
                        else:
                            silent_chunks_count = 0
                        if silent_chunks_count > silent_chunks_needed:
                            break
            
            requests = audio_generator()
            responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)

            for response in responses:
                if result := response.results and response.results[0]:
                    if result.is_final:
                        transcript = result.alternatives[0].transcript
                        break
    except Exception as e:
        print(f"STT 스트리밍 중 오류 발생: {e}")
        return None

    return transcript

def get_gemini_response(text, add_to_history=True):
    global conversation_history
    print("AI가 응답을 생성 중입니다...")
    try:
        if add_to_history:
            conversation_history.append({'role': 'user', 'parts': [text]})
        
        chat = gemini_model.start_chat(history=conversation_history)
        response = chat.send_message(text, stream=False)
        ai_response = response.text
        
        if add_to_history:
            conversation_history.append({'role': 'model', 'parts': [ai_response]})
            save_history()
        
        print(f"Emily 선생님: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        return "Sorry, I'm having a little trouble thinking right now."

def google_tts(text):
    print("텍스트를 음성으로 변환 중...")
    try:
        is_korean = bool(re.search('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]', text))
        lang_code, voice_name = ("ko-KR", "ko-KR-Studio-B") if is_korean else ("en-US", "en-US-Studio-O")

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        with open(FILENAME_OUTPUT, "wb") as out:
            out.write(response.audio_content)
        return True
    except Exception as e:
        print(f"TTS API 오류: {e}")
        return False

def summarize_conversation():
    global conversation_history
    print("\n대화 내용을 요약 중입니다...")
    actual_conversation = conversation_history[1:]
    
    if len(actual_conversation) < 2:
        print("요약할 대화 내용이 충분하지 않습니다.")
        if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
        return

    try:
        conversation_text = "\n".join([f"{msg['role']}: {msg['parts'][0]}" for msg in actual_conversation])
        summary_prompt = f"Please summarize the following conversation between 'Teacher Emily' and '{USER_NAME}' in Korean:\n\n{conversation_text}"
        response = gemini_model.generate_content(summary_prompt)
        summary = response.text
        summary_data = {"summary": summary}
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"대화 요약이 '{SUMMARY_FILE}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"대화 요약 중 오류 발생: {e}")

def play_audio(file_path):
    print("음성 응답을 재생합니다.")
    os.system(f"mpg321 -q {file_path}")

# [수정] RPi.GPIO 대신 gpiozero를 사용합니다.
def wait_for_button_press(button):
    """버튼이 눌릴 때까지 대기합니다."""
    print("버튼을 눌러 다시 시작하세요.")
    if pixel_ring: pixel_ring.off()
    try:
        button.wait_for_press()
        print("버튼이 눌렸습니다. 대화를 재개합니다.")
    except Exception as e:
        print(f"\nGPIO 대기 중 오류 발생: {e}. 프로그램을 종료합니다.")
        raise

# --- 메인 실행 로직 ---

def main():
    global conversation_history
    
    # [수정] gpiozero Button 객체를 생성합니다.
    button = None
    if Button:
        try:
            button = Button(BUTTON_PIN, pull_up=True)
        except Exception as e:
            print(f"GPIO 버튼 초기화 실패: {e}")
            print("GPIO 기능을 사용하지 않고 계속합니다.")

    mic_device_id = get_microphone_device_id()
    if mic_device_id is None: return
        
    calibrate_silence_threshold(mic_device_id)
    
    conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}]
    initial_prompt = ""

    if os.path.exists(SUMMARY_FILE):
        print(f"'{SUMMARY_FILE}'에서 이전 대화 요약을 불러옵니다.")
        summary_data = load_json_file(SUMMARY_FILE, "대화 요약")
        if summary_data and "summary" in summary_data:
            summary_text = summary_data["summary"]
            context_prompt = f"Let's continue our conversation. Here's a summary of our last talk. Please start with a friendly greeting to {USER_NAME} and continue the conversation based on this summary:\n\n{summary_text}"
            initial_prompt = get_gemini_response(context_prompt, add_to_history=False)
        else:
            initial_prompt = f"Hello {USER_NAME}! I'm Teacher Emily. Let's talk!"
    else:
        print("저장된 대화 요약이 없습니다. 새로운 대화를 시작합니다.")
        initial_prompt = f"Hello {USER_NAME}! I'm Teacher Emily. Let's talk!"
    
    if pixel_ring: pixel_ring.think()
    if google_tts(initial_prompt):
        if pixel_ring: pixel_ring.speak()
        play_audio(FILENAME_OUTPUT)
    
    conversation_history.append({'role': 'model', 'parts': [initial_prompt]})
    save_history()

    try:
        while True:
            if pixel_ring: pixel_ring.wakeup()
            
            # [수정] TTS 재생 후 발생할 수 있는 에코나 잔향이 마이크에 입력되는 것을 방지하기 위해 짧은 딜레이를 추가합니다.
            time.sleep(0.5)
            
            user_text = record_and_stream_stt(mic_device_id)

            if user_text:
                if pixel_ring: pixel_ring.think()
                print(f"나: {user_text}")
                ai_text = get_gemini_response(user_text)
                
                if "END_CONVERSATION" in ai_text:
                    print("사용자가 대화 종료를 요청했습니다.")
                    summarize_conversation()
                    if button: wait_for_button_press(button)
                    else: input("Enter를 눌러 재개합니다...")
                    if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
                    conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}]
                    continue
                
                if ai_text and google_tts(ai_text):
                    if pixel_ring: pixel_ring.speak()
                    play_audio(FILENAME_OUTPUT)
            else:
                print("10초 동안 음성 입력이 없어 대화를 중단합니다.")
                summarize_conversation()
                if button: wait_for_button_press(button)
                else: input("Enter를 눌러 재개합니다...")
                if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
                conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}]
                continue
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n프로그램을 종료합니다. ({e})")
        summarize_conversation()
    finally:
        if os.path.exists(FILENAME_OUTPUT): os.remove(FILENAME_OUTPUT)
        print("임시 파일 정리 완료.")
        if pixel_ring: pixel_ring.off()
        # [수정] gpiozero는 자동 정리되므로 cleanup 코드가 필요 없습니다.

if __name__ == "__main__":
    main()


