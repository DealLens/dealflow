# 💼 Dealflow AI Agent

LangChain + Azure OpenAI + Streamlit 기반 투자/M&A 전문 AI 어시스턴트

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4-orange)

## 🚀 주요 기능

- **투자 정보 검색**: M&A, 투자 거래 정보 조회
- **시장 분석**: 시장 동향 및 트렌드 분석
- **회사 정보 조회**: 기업 정보 및 재무 데이터
- **실시간 채팅**: Streamlit 기반 웹 인터페이스
- **도구 기반 AI**: LangChain 에이전트로 정확한 정보 제공

## ⚙️ 환경 설정

### 1. 저장소 클론
```bash
git clone <your-repo-url>
cd dealflow/langchain-azure-agent
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`env_example.txt` 파일을 참고하여 `.env` 파일을 생성하세요:

```bash
# .env 파일 생성
cp env_example.txt .env
```

`.env` 파일에 Azure OpenAI 설정을 입력하세요:
```env
AOAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AOAI_API_KEY=your-api-key-here
AOAI_DEPLOY_GPT4O=your-gpt-4o-deployment-name
AOAI_DEPLOY_GPT4O_MINI=your-gpt-4o-mini-deployment-name
```

### 5. 애플리케이션 실행
```bash
streamlit run main.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

## 🐳 Docker 배포

### Docker로 실행
```bash
# 이미지 빌드
docker build -t dealflow-agent .

# 컨테이너 실행
docker run -p 8501:8501 --env-file .env dealflow-agent
```

### Docker Compose로 실행
```bash
docker-compose up -d
```

## ☁️ 클라우드 배포

### Streamlit Cloud 배포
1. GitHub에 코드 푸시
2. [Streamlit Cloud](https://share.streamlit.io/) 접속
3. "New app" 클릭
4. GitHub 저장소 선택
5. 환경 변수 설정 (Secrets)
6. 배포 완료!

### Heroku 배포
```bash
# Heroku CLI 설치 후
heroku create your-app-name
heroku config:set AOAI_ENDPOINT=your-endpoint
heroku config:set AOAI_API_KEY=your-key
heroku config:set AOAI_DEPLOY_GPT4O=your-deployment
git push heroku main
```

### Azure Container Instances 배포
```bash
# Azure CLI로 로그인
az login

# 리소스 그룹 생성
az group create --name dealflow-rg --location eastus

# 컨테이너 인스턴스 배포
az container create \
  --resource-group dealflow-rg \
  --name dealflow-agent \
  --image your-registry/dealflow-agent:latest \
  --dns-name-label dealflow-agent \
  --ports 8501 \
  --environment-variables \
    AOAI_ENDPOINT=your-endpoint \
    AOAI_API_KEY=your-key \
    AOAI_DEPLOY_GPT4O=your-deployment
```

## 🛠️ 개발 가이드

### 프로젝트 구조
```
langchain-azure-agent/
├── main.py                 # 메인 애플리케이션
├── streamlit_app.py        # Streamlit Cloud 엔트리 포인트
├── requirements.txt        # Python 의존성
├── Dockerfile             # Docker 설정
├── docker-compose.yml     # Docker Compose 설정
├── .streamlit/
│   └── config.toml        # Streamlit 설정
├── env_example.txt        # 환경 변수 템플릿
└── README.md              # 프로젝트 문서
```

### 도구 추가하기
`main.py`의 `_setup_tools()` 메서드에서 새로운 도구를 추가할 수 있습니다:

```python
def new_tool(query: str) -> str:
    """새로운 도구 설명"""
    return f"'{query}'에 대한 결과"

Tool(
    name="new_tool",
    description="새로운 도구 설명",
    func=new_tool
)
```

## 🔧 문제 해결

### 일반적인 오류
1. **Azure OpenAI 연결 오류**: `.env` 파일의 설정을 확인하세요
2. **포트 충돌**: 다른 포트를 사용하거나 기존 프로세스를 종료하세요
3. **의존성 오류**: `pip install -r requirements.txt`를 다시 실행하세요

### 로그 확인
```bash
# Streamlit 로그
streamlit run main.py --logger.level debug

# Docker 로그
docker logs <container-id>
```

## 📝 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
