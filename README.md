라즈베리파이 AI 영어회화 튜터
이 프로젝트는 라즈베리파이 제로 2W와 ReSpeaker 2-Mic HAT을 사용하여, 아이들을 위한 대화형 AI 영어 선생님을 만드는 것을 목표로 합니다. Google Cloud의 STT, TTS, Gemini API를 활용하여 실시간으로 자연스러운 영어 대화가 가능하도록 구현되었습니다.

주요 기능
실시간 음성 대화: 사용자의 음성을 실시간으로 텍스트로 변환하고, AI의 텍스트 응답을 자연스러운 음성으로 출력합니다.

다국어 지원: 영어와 한국어 음성 입력을 모두 인식하며, AI의 답변에 따라 영어 또는 한국어 음성으로 출력합니다.

지능형 묵음 감지: 사용자의 발화가 시작된 후에만 묵음을 감지하여, 자연스러운 대화 시작이 가능합니다.

대화 기억 및 요약: 대화 내용을 파일로 기록하고, 프로그램 종료 시 대화를 요약하여 다음 실행 시 이전 대화에 이어서 시작할 수 있습니다.

시각적 피드백: ReSpeaker HAT의 RGB LED를 통해 음성 비서의 상태(대기, 녹음, 생각, 말하기)를 시각적으로 표시합니다.

자동 실행: 라즈베리파이 부팅 시 자동으로 음성 비서 프로그램이 실행됩니다.

하드웨어 준비물
라즈베리파이 제로 2W (또는 그 이상)

ReSpeaker 2-Mic Pi HAT

마이크로 SD 카드 (16GB 이상 권장)

전원 어댑터

스피커 (3.5mm 오디오 잭 또는 블루투스)

(선택 사항) 대화 재시작을 위한 푸시 버튼

설치 및 설정 가이드
1단계: Google Cloud Platform (GCP) 및 Gemini API 설정
이 프로젝트는 Google의 API를 사용하므로, 가장 먼저 API 사용 설정을 완료해야 합니다.

GCP 프로젝트 생성: Google Cloud Console에서 새 프로젝트를 생성합니다.

API 활성화: 생성한 프로젝트에서 다음 3개의 API를 검색하여 '사용 설정'합니다.

Cloud Speech-to-Text API

Cloud Text-to-Speech API

Vertex AI API (Gemini API 사용을 위해 필요)

결제 계정 연결: API의 무료 사용량 한도를 초과하여 사용하려면 프로젝트에 결제 계정을 연결해야 합니다. (신규 사용자에게는 $300의 무료 크레딧이 제공됩니다.)

서비스 계정 키 생성:

IAM 및 관리자 > 서비스 계정 메뉴로 이동하여 새 서비스 계정을 만듭니다.

역할은 편집자(Editor)를 부여하여 모든 API에 접근할 수 있도록 합니다.

생성된 서비스 계정의 키 탭에서 키 추가 > 새 키 만들기를 선택하고, JSON 형식으로 키를 다운로드합니다. 이 파일은 잠시 후 라즈베리파이로 옮겨야 합니다.

Gemini API 키 발급:

Google AI Studio에 접속합니다.

Get API key > Create API key in new project를 클릭하여 새로운 API 키를 발급받습니다. 이 키 문자열을 복사해 둡니다.

2단계: 라즈베리파이 초기 설정
OS 설치 및 기본 설정: 최신 Raspberry Pi OS (64-bit 권장)를 설치하고, Wi-Fi 연결, SSH 활성화 등 기본 설정을 완료합니다.

시스템 패키지 설치: 터미널에서 아래 명령어를 실행하여 필요한 도구들을 설치합니다.

sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip alsa-utils mpg321

오디오 드라이버 설치: ReSpeaker 2-Mic HAT의 공식 가이드에 따라 오디오 드라이버를 설치합니다.

git clone [https://github.com/respeaker/seeed-voicecard.git](https://github.com/respeaker/seeed-voicecard.git)
cd seeed-voicecard
sudo ./install.sh
# 설치 후 재부팅이 필요합니다.
sudo reboot

GPIO 권한 설정 (Sudo 없이 실행하기 위한 가장 중요한 단계):

현재 사용자를 gpio 그룹에 추가합니다. (pi가 아닌 다른 사용자 이름을 사용한다면 pi를 해당 이름으로 변경하세요.)

sudo usermod -aG gpio pi

udev 규칙을 추가하여 부팅 시 GPIO 장치 파일의 권한을 gpio 그룹에 부여합니다.

echo 'SUBSYSTEM=="gpio*", PROGRAM="/bin/sh -c '\''chown -R root:gpio /sys/class/gpio && chmod -R 770 /sys/class/gpio; chown -R root:gpio /sys/devices/platform/soc/*.gpio/gpio && chmod -R 770 /sys/devices/platform/soc/*.gpio/gpio'\''"' | sudo tee /etc/udev/rules.d/99-gpio.rules

변경 사항을 적용하기 위해 반드시 재부팅합니다.

sudo reboot

3단계: 프로젝트 소스 코드 다운로드 및 가상 환경 설정
소스 코드 복제:

cd ~
git clone [Your-GitHub-Repository-URL] voice_assistant
cd voice_assistant

파이썬 가상 환경 생성 및 활성화:

python3 -m venv venv
source venv/bin/activate

필요한 라이브러리 설치:

pip install --upgrade pip
pip install -r requirements.txt

4단계: 설정 파일 구성
voice_assistant 폴더 안에 아래 4개의 설정 파일을 생성하고 내용을 채워야 합니다.

서비스 계정 키 파일:

1단계에서 다운로드한 GCP 서비스 계정 JSON 키 파일을 이 폴더로 복사합니다. 파일 이름은 자유롭게 지정할 수 있습니다. (예: gcp-key.json)

gemini_api_key.json:

{
  "api_key": "1단계에서-발급받은-Gemini-API-키를-여기에-붙여넣기"
}

user.json:

{
  "name": "David",
  "introduction": "an elementary school student who loves dinosaurs"
}

persona.json: (AI의 역할과 규칙을 자유롭게 수정할 수 있습니다.)

{
  "description": "You are 'Teacher Emily', a friendly and patient English teacher.",
  "user_context": "The user's name is {USER_NAME}, who is {USER_INTRODUCTION}. Always be friendly and sometimes use their name.",
  "language_rule": "If the user speaks in Korean, you can also answer in Korean if necessary to help them.",
  "rules": [
    "Always speak in simple, easy-to-understand English.",
    "Use short sentences.",
    "Ask simple, open-ended questions to encourage the child to speak.",
    "Be very patient. If the user's response is unclear, gently guide them.",
    "Keep your responses concise, typically 1-2 sentences.",
    "If the user clearly expresses a desire to stop talking (e.g., 'goodbye', 'stop'), your only response must be the exact single word: END_CONVERSATION"
  ],
  "greeting": "Start the first conversation by greeting the user by their name. For example, \"Hello, {USER_NAME}! Let's talk!\"."
}

5단계: 수동 실행 및 테스트
모든 설정이 완료되었습니다. 가상 환경이 활성화된 상태에서 아래 명령어로 음성 비서를 실행하여 테스트합니다.

# (venv)가 프롬프트 앞에 표시되어 있어야 합니다.
python3 voice_assistant_final.py

6단계: 부팅 시 자동 실행 (Systemd 서비스 등록)
안정적인 동작이 확인되면, 아래 단계에 따라 부팅 시 음성 비서가 자동으로 시작되도록 설정합니다.

서비스 파일 생성:

sudo nano /etc/systemd/system/voice_assistant.service

서비스 내용 붙여넣기: 열린 편집기에 아래 내용을 그대로 붙여넣습니다. 두 군데를 반드시 수정해야 합니다.

User=pi: pi를 실제 사용자 이름으로 변경

Environment=...: gcp-key.json을 실제 서비스 계정 키 파일 이름과 경로로 변경

[Unit]
Description=Voice Assistant Service
After=network-online.target sound.target

[Service]
# [수정] 'pi'를 실제 라즈베리파이 사용자 이름으로 변경하세요.
User=pi
# [수정] GCP 서비스 계정 키 파일의 절대 경로를 지정하세요.
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/pi/voice_assistant/gcp-key.json"

WorkingDirectory=/home/pi/voice_assistant
ExecStart=/home/pi/voice_assistant/venv/bin/python3 /home/pi/voice_assistant/voice_assistant_final.py
Restart=always

[Install]
WantedBy=multi-user.target

서비스 활성화 및 시작:

sudo systemctl daemon-reload
sudo systemctl enable voice_assistant.service
sudo systemctl start voice_assistant.service

상태 확인:

# 서비스가 잘 동작하는지 확인
sudo systemctl status voice_assistant.service

# 실시간 로그 확인
journalctl -u voice_assistant.service -f
