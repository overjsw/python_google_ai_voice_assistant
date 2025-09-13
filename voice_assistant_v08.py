# -*- coding: utf-8 -*-
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai
import time

# --- 설정 (Configuration) ---

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 2 # Seeed 2-mic hat은 스테레오(2채널)만 지원
FILENAME_INPUT = "user_audio.wav" # 녹음 파일
FILENAME_OUTPUT = "assistant_audio.mp3"

# 묵음 감지 설정
SILENCE_SECONDS = 5 # 5초간의 묵음이 감지되면 녹음 중지
# 이 값은 프로그램 시작 시 자동으로 보정됩니다.
SILENCE_THRESHOLD = 3000 # 기본값

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
    # 404 오류 해결을 위해 모델 이름을 최신 모델('gemini-1.5-flash')로 변경합니다.
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

# 대화 기록 초기화
conversation_history = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': [""]}]


# --- 핵심 기능 함수 ---

def get_microphone_device_id():
    """시스템에서 Seeed 2-mic voicecard 마이크 장치 ID를 찾습니다."""
    try:
        devices = sd.query_devices()
        # 'arecord -l'에서 확인한 'seeed-2mic-voicecard'를 우선적으로 찾습니다.
        for index, device in enumerate(devices):
            if 'seeed-2mic-voicecard' in device['name']:
                 if device['max_input_channels'] > 0:
                    print(f"지정된 마이크를 찾았습니다: ID {index} - {device['name']}")
                    return index

        # 찾지 못했을 경우, 키워드 기반으로 다시 검색
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
    
    calibration_duration = 5 # 5초간 측정
    
    volume_levels = []
    try:
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                chunk, overflowed = stream.read(1024)
                volume_norm = np.linalg.norm(chunk)
                volume_levels.append(volume_norm)

        if volume_levels:
            # 측정된 소음의 평균값에 1.8을 곱하고, 최소 보장값으로 500을 더하여 기준 설정
            average_noise = np.mean(volume_levels)
            calibrated_threshold = int((average_noise * 1.8) + 500)
            SILENCE_THRESHOLD = calibrated_threshold
            print(f"묵음 기준 보정 완료! 새로운 기준: {SILENCE_THRESHOLD}")
        else:
            print("소음 측정에 실패하여 기본값을 사용합니다.")
    except Exception as e:
        print(f"묵음 기준 보정 중 오류 발생: {e}. 기본값을 사용합니다.")
    print("-" * 20)


def record_audio_with_silence_detection(device_id):
    """sounddevice를 사용하여 묵음이 감지될 때까지 오디오를 녹음합니다."""
    print("음성 녹음 시작... 말씀하세요.")

    chunks_per_second = 10
    chunk_size = int(SAMPLE_RATE / chunks_per_second)
    silent_chunks_needed = SILENCE_SECONDS * chunks_per_second

    audio_data = []
    silent_chunks_count = 0

    try:
        # 모든 디버깅을 통해 확인된 최종 설정값(장치 ID, 샘플링 속도, 2채널, int16 타입)으로 스트림을 엽니다.
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=chunk_size) as stream:
            print(f"녹음 스트림이 열렸습니다. (묵음 기준: {SILENCE_THRESHOLD})")
            while True:
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("경고: 녹음 중 오디오 버퍼 오버플로우 발생")

                audio_data.append(chunk)
                
                # L2 norm을 사용하여 현재 청크의 볼륨을 계산합니다.
                volume_norm = np.linalg.norm(chunk)

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
            # 한국어를 대체 언어로 추가하여 다중 언어 인식을 지원합니다.
            alternative_language_codes=["ko-KR"],
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
    # 프로그램 시작 시 마이크 장치를 찾습니다.
    mic_device_id = get_microphone_device_id()
    if mic_device_id is None:
        print("프로그램을 시작할 수 없습니다. 마이크 연결을 확인해주세요.")
        return
        
    # 주변 소음을 측정하여 묵음 기준을 자동으로 설정합니다.
    calibrate_silence_threshold(mic_device_id)

    initial_prompt = "Hello! I'm Teacher Emily. Let's talk!"
    print(f"Emily 선생님: {initial_prompt}")
    if google_tts(initial_prompt):
        play_audio(FILENAME_OUTPUT)
    
    global conversation_history
    conversation_history[1]['parts'] = [initial_prompt]

    try:
        while True:
            # 묵음 감지 녹음 함수를 호출합니다.
            if record_audio_with_silence_detection(mic_device_id):
                
                # --- STT 시간 측정 ---
                stt_start_time = time.time()
                user_text = google_stt(FILENAME_INPUT)
                stt_end_time = time.time()

                if user_text:
                    print(f"  > STT 소요 시간: {stt_end_time - stt_start_time:.2f}초")
                    print(f"나: {user_text}")

                    # --- LLM 시간 측정 ---
                    llm_start_time = time.time()
                    ai_text = get_gemini_response(user_text)
                    llm_end_time = time.time()
                    
                    if ai_text:
                        print(f"  > LLM 응답 시간: {llm_end_time - llm_start_time:.2f}초")
                        
                        # --- TTS 시간 측정 ---
                        tts_start_time = time.time()
                        if google_tts(ai_text):
                            tts_end_time = time.time()
                            print(f"  > TTS 변환 시간: {tts_end_time - tts_start_time:.2f}초")
                            play_audio(FILENAME_OUTPUT)
                else:
                    # 사용자가 말을 하지 않아 녹음된 내용이 없는 경우
                    print("음성 입력이 없어 대화를 이어가지 않습니다. 다시 말씀해주세요.")
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


