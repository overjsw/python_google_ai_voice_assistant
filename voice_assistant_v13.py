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
try:
    from pixel_ring import pixel_ring
except ImportError:
    print("pixel_ring 라이브러리를 찾을 수 없습니다.")
    print("'pip install pixel_ring seeed-voicecard' 명령어로 설치해주세요.")
    pixel_ring = None

# --- 설정 파일 이름 (Configuration Filenames) ---
USER_CONFIG_FILE = "user.json"
API_KEY_FILE = "gemini_api_key.json"
HISTORY_FILE = "conversation_history.json"

# --- 설정 로드 함수 ---
def load_json_file(filename, purpose):
    """지정된 JSON 파일을 로드하고 데이터를 반환합니다."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[오류] {purpose} 파일('{filename}')을 찾을 수 없습니다.")
        print("프로그램을 계속하려면 해당 파일을 생성해주세요.")
        return None
    except json.JSONDecodeError:
        print(f"[오류] {purpose} 파일('{filename}')의 형식이 잘못되었습니다.")
        return None
    except Exception as e:
        print(f"'{filename}' 파일을 읽는 중 오류 발생: {e}")
        return None

# --- 설정 및 초기화 ---

# 사용자 정보 및 API 키 로드
user_config = load_json_file(USER_CONFIG_FILE, "사용자 정보")
api_key_config = load_json_file(API_KEY_FILE, "Gemini API 키")

if not user_config or not api_key_config:
    exit() # 설정 파일이 없으면 프로그램 종료

USER_NAME = user_config.get("name", "friend")
USER_INTRODUCTION = user_config.get("introduction", "a student")
GEMINI_API_KEY = api_key_config.get("api_key")

if not GEMINI_API_KEY or "your-gemini-api-key" in GEMINI_API_KEY:
    print(f"[오류] '{API_KEY_FILE}'에 유효한 Gemini API 키를 입력해주세요.")
    exit()

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 2 # Seeed 2-mic hat은 스테레오(2채널)만 지원
FILENAME_OUTPUT = "assistant_audio.mp3"

# 묵음 감지 설정
SILENCE_THRESHOLD = 3000

# AI 페르소나 설정
SYSTEM_PROMPT = f"""
You are 'Teacher Emily', a friendly and patient English teacher.
The user's name is {USER_NAME}, who is {USER_INTRODUCTION}. Always be friendly and sometimes use their name.
If the user speaks in Korean, you can also answer in Korean if necessary to help them.
Follow these rules strictly:
1.  Always speak in simple, easy-to-understand English.
2.  Use short sentences.
3.  Ask simple, open-ended questions to encourage the child to speak.
4.  Be very patient. If the user's response is unclear, gently guide them.
5.  Keep your responses concise, typically 1-2 sentences.
6.  Start the first conversation by greeting the user by their name. For example, "Hello, {USER_NAME}! Let's talk!".
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
    print("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되었는지 확인하세요.")
    exit()

# 대화 기록 변수
conversation_history = []


# --- 핵심 기능 함수 ---

def save_history():
    """대화 기록을 JSON 파일에 저장합니다."""
    global conversation_history
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"대화 기록 저장 중 오류 발생: {e}")

def load_history():
    """JSON 파일에서 대화 기록을 불러옵니다."""
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                conversation_history = json.load(f)
            print(f"'{HISTORY_FILE}'에서 이전 대화 기록을 불러왔습니다.")
        except Exception as e:
            print(f"대화 기록을 불러오는 중 오류 발생: {e}. 새로운 대화를 시작합니다.")
            conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': [""]}]
    else:
        print("저장된 대화 기록이 없습니다. 새로운 대화를 시작합니다.")
        conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': [""]}]

def get_microphone_device_id():
    """시스템에서 Seeed 2-mic voicecard 마이크 장치 ID를 찾습니다."""
    try:
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if 'seeed-2mic-voicecard' in device['name']:
                 if device['max_input_channels'] > 0:
                    print(f"지정된 마이크를 찾았습니다: ID {index} - {device['name']}")
                    return index
        for index, device in enumerate(devices):
            device_name = device['name'].lower()
            if any(keyword in device_name for keyword in ['mic', 'usb', 'seeed']):
                if device['max_input_channels'] > 0:
                    print(f"키워드 검색으로 마이크를 찾았습니다: ID {index} - {device['name']}")
                    return index
    except Exception as e:
        print(f"오디오 장치 목록을 가져오는 중 오류 발생: {e}")
    print("[오류] 적합한 마이크를 찾지 못했습니다.")
    return None

def calibrate_silence_threshold(device_id):
    """프로그램 시작 시 주변 소음을 5초간 측정하여 묵음 기준을 자동으로 설정합니다."""
    global SILENCE_THRESHOLD
    print("\n묵음 기준 자동 보정을 시작합니다. 5초간 조용한 상태를 유지해주세요...")
    calibration_duration = 5
    volume_levels = []
    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                chunk, overflowed = stream.read(1024)
                volume_norm = np.linalg.norm(chunk)
                volume_levels.append(volume_norm)

        if volume_levels:
            average_noise = np.mean(volume_levels)
            calibrated_threshold = int((average_noise * 1.8) + 500)
            SILENCE_THRESHOLD = calibrated_threshold
            print(f"묵음 기준 보정 완료! 새로운 기준: {SILENCE_THRESHOLD}")
        else:
            print("소음 측정에 실패하여 기본값을 사용합니다.")
    except Exception as e:
        print(f"묵음 기준 보정 중 오류 발생: {e}. 기본값을 사용합니다.")
    print("-" * 20)


def record_and_stream_stt(device_id):
    """마이크 입력을 실시간으로 STT API에 스트리밍하여 텍스트로 변환합니다."""
    print("음성 녹음 및 실시간 STT 시작... 말씀하세요.")
    if pixel_ring:
        pixel_ring.set_color(g=255)

    # Streaming STT 설정
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        alternative_language_codes=["ko-KR"],
        audio_channel_count=CHANNELS,
        enable_separate_recognition_per_channel=False
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)

    # 상수 설정
    chunks_per_second = 10
    chunk_size = int(SAMPLE_RATE / chunks_per_second)
    stream_silence_seconds = 2  # 응답성을 위해 묵음 감지 시간 단축
    silent_chunks_needed = stream_silence_seconds * chunks_per_second
    start_timeout_seconds = 10
    start_timeout_chunks = start_timeout_seconds * chunks_per_second
    
    # 상태 변수
    silent_chunks_count = 0
    speech_started = False
    total_chunks_read = 0
    transcript = ""

    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=chunk_size) as stream:
            # print(f"녹음 스트림이 열렸습니다. (묵음 기준: {SILENCE_THRESHOLD})")

            def audio_generator():
                nonlocal speech_started, silent_chunks_count, total_chunks_read
                while True:
                    chunk, overflowed = stream.read(chunk_size)
                    if overflowed:
                        print("경고: 녹음 중 오디오 버퍼 오버플로우 발생")
                    
                    yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
                    
                    total_chunks_read += 1
                    volume_norm = np.linalg.norm(chunk)
                    
                    if not speech_started:
                        if volume_norm > SILENCE_THRESHOLD:
                            # print("음성 입력을 감지했습니다. 묵음 감지를 시작합니다.")
                            speech_started = True
                        elif total_chunks_read > start_timeout_chunks:
                            print(f"{start_timeout_seconds}초 동안 음성 입력이 없어 녹음을 중단합니다.")
                            break
                    else:
                        if volume_norm < SILENCE_THRESHOLD:
                            silent_chunks_count += 1
                        else:
                            silent_chunks_count = 0

                        if silent_chunks_count > silent_chunks_needed:
                            # print("묵음이 감지되어 스트리밍을 중지합니다.")
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
        save_history()
        print(f"Emily 선생님: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        return "Sorry, I'm having a little trouble thinking right now."

def google_tts(text):
    """Google Text-to-Speech API를 사용하여 텍스트를 음성으로 변환합니다."""
    print("텍스트를 음성으로 변환 중...")
    try:
        # 텍스트에 한글이 포함되어 있는지 확인
        is_korean = bool(re.search('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]', text))
        
        if is_korean:
            lang_code = "ko-KR"
            voice_name = "ko-KR-Wavenet-A" # 한국어 여성 목소리
        else:
            lang_code = "en-US"
            voice_name = "en-US-Wavenet-F" # 영어 여성 목소리

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code, name=voice_name
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

def summarize_conversation():
    """대화 기록을 요약하고 파일에 저장합니다."""
    global conversation_history
    print("\n대화 내용을 요약 중입니다...")

    # 시스템 프롬프트를 제외한 실제 대화 내용만 추출
    actual_conversation = conversation_history[1:]
    
    if len(actual_conversation) < 2: # 실제 대화가 거의 없는 경우
        print("요약할 대화 내용이 충분하지 않습니다.")
        return

    try:
        # 대화 내용을 하나의 문자열로 만듭니다.
        conversation_text = "\n".join([f"{msg['role']}: {msg['parts'][0]}" for msg in actual_conversation])
        
        # Gemini에 요약 요청
        summary_prompt = f"Please summarize the following conversation between 'Teacher Emily' and '{USER_NAME}' in Korean:\n\n{conversation_text}"
        response = gemini_model.generate_content(summary_prompt)
        summary = response.text

        # 요약된 내용을 파일에 저장
        summary_filename = "conversation_summary.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("--- 대화 요약 ---\n")
            f.write(summary)
        print(f"대화 요약이 '{summary_filename}' 파일에 저장되었습니다.")

    except Exception as e:
        print(f"대화 요약 중 오류 발생: {e}")

def play_audio(file_path):
    """오디오 파일을 재생합니다."""
    print("음성 응답을 재생합니다.")
    os.system(f"mpg321 -q {file_path}")

# --- 메인 실행 로직 ---

def main():
    """메인 프로그램 루프"""
    mic_device_id = get_microphone_device_id()
    if mic_device_id is None:
        print("프로그램을 시작할 수 없습니다. 마이크 연결을 확인해주세요.")
        return
        
    calibrate_silence_threshold(mic_device_id)
    load_history()

    is_new_conversation = len(conversation_history) <= 2
    if is_new_conversation:
        print("새로운 대화를 시작합니다.")
        initial_prompt = f"Hello {USER_NAME}! I'm Teacher Emily. Let's talk!"
        print(f"Emily 선생님: {initial_prompt}")
        if pixel_ring:
            pixel_ring.think()
        if google_tts(initial_prompt):
            if pixel_ring:
                pixel_ring.speak()
            play_audio(FILENAME_OUTPUT)
        conversation_history[1]['parts'] = [initial_prompt]
        save_history()
    else:
        print("이전 대화에 이어서 시작합니다.")
        last_response = conversation_history[-1]['parts'][0]
        if last_response:
            if pixel_ring:
                pixel_ring.think()
            if google_tts(last_response):
                if pixel_ring:
                    pixel_ring.speak()
                play_audio(FILENAME_OUTPUT)

    try:
        while True:
            if pixel_ring:
                pixel_ring.wakeup()
            
            stt_start_time = time.time()
            user_text = record_and_stream_stt(mic_device_id)
            stt_end_time = time.time()

            if user_text:
                if pixel_ring:
                    pixel_ring.think()
                
                print(f"  > STT 전체 과정 소요 시간: {stt_end_time - stt_start_time:.2f}초")
                print(f"나: {user_text}")

                llm_start_time = time.time()
                ai_text = get_gemini_response(user_text)
                llm_end_time = time.time()
                
                if ai_text:
                    print(f"  > LLM 응답 시간: {llm_end_time - llm_start_time:.2f}초")
                    
                    tts_start_time = time.time()
                    if google_tts(ai_text):
                        tts_end_time = time.time()
                        print(f"  > TTS 변환 시간: {tts_end_time - tts_start_time:.2f}초")
                        if pixel_ring:
                            pixel_ring.speak()
                        play_audio(FILENAME_OUTPUT)
            else:
                # print("대기 상태로 돌아갑니다. 다시 말씀해주세요.")
                continue
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        summarize_conversation()
    finally:
        if os.path.exists(FILENAME_OUTPUT):
            os.remove(FILENAME_OUTPUT)
        print("임시 파일 정리 완료.")
        if pixel_ring:
            pixel_ring.off()

if __name__ == "__main__":
    main()


