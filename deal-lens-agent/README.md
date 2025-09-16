# 🎯 DealLens: 수주·실주 인사이트 AI 에이전트

LangChain + Azure OpenAI + Streamlit 기반 RFP 분석 및 수주 전략 AI 에이전트

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4-orange)

## 🚀 프로젝트 개요

**DealLens**는 과거 프로젝트·경쟁사·고객 맥락 데이터를 수집·정규화하여 비교 SWOT, 보완책, 난이도 예측을 한 번에 제공하는 AI 에이전트입니다.

### 📋 주요 기능

- **📄 RFP 분석 & 이해**: PDF(+OCR) 인식, 요구사항 구조화, 평가기준 추출, 리스크 플래그
- **🏠 내부지식 매칭**: 과거 유사 프로젝트, 성과, 고객 피드백, 모듈/솔루션, 인력 스킬 매칭
- **🏢 경쟁사 분석**: 경쟁사 포트폴리오, 수주 이력, 가격 포지셔닝, 기술 스택, 파트너십 수집
- **🎯 난이도 스코어링**: 평가 가중치 × (당사 적합도 − 경쟁사 우위) 기반 분포화
- **💡 전략 합성**: 보완책, 당사 SWOT, 경쟁사 대비 차별화 메시지 통합
- **💬 Q&A / 리포팅**: 사용자의 자연어 질문에 맞춰 결과 재구성/시각화

## 🏗️ 아키텍처

### 에이전트별 역할

```
A. RFP 분석 & 이해 에이전트 (Document Understanding)
   ↓
B. 내부지식 매칭 에이전트 (Internal RAG)
   ↓
C. 경쟁사 분석 에이전트 (Competitor Intelligence)
   ↓
D. 난이도 스코어링 에이전트 (Win Probability)
   ↓
E. 전략 합성 에이전트 (Strategy Synthesizer)
   ↓
F. Q&A / 리포팅 에이전트 (User-facing)
```

### 프로젝트 구조

```
deal-lens-agent/
├── src/
│   ├── agents/                    # AI 에이전트들
│   │   ├── rfp_understanding_agent.py
│   │   ├── internal_rag_agent.py
│   │   ├── competitor_intelligence_agent.py
│   │   ├── win_probability_agent.py
│   │   ├── strategy_synthesizer_agent.py
│   │   ├── reporting_agent.py
│   │   └── deal_lens_orchestrator.py
│   ├── analyzers/                 # 분석 모듈들
│   ├── tools/                     # 도구들
│   ├── utils/                     # 유틸리티
│   └── ui/                        # UI 컴포넌트
├── data/                          # 데이터 저장소
├── configs/                       # 설정 파일들
├── tests/                         # 테스트
├── docs/                          # 문서
├── main.py                        # 메인 애플리케이션
├── requirements.txt               # Python 의존성
├── Dockerfile                     # Docker 설정
├── docker-compose.yml             # Docker Compose 설정
└── README.md                      # 프로젝트 문서
```

## ⚙️ 환경 설정

### 1. 저장소 클론
```bash
git clone <your-repo-url>
cd deal-lens-agent
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
AOAI_DEPLOY_EMBEDDINGS=text-embedding-ada-002
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
docker build -t deal-lens-agent .

# 컨테이너 실행
docker run -p 8501:8501 --env-file .env deal-lens-agent
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

## 🎯 사용 방법

### 1. RFP 분석
- RFP 문서를 업로드하거나 직접 입력
- 경쟁사 목록 선택
- 추가 정보 입력 (최근 개선사항, 사업 방향, 예산, 인력 등)

### 2. 분석 결과 확인
- 수주 난이도 및 확률
- 요구사항 분석 결과
- 내부 매칭 결과
- 경쟁사 분석 결과
- 전략 합성 결과

### 3. Q&A 기능
- 자연어로 분석 결과에 대해 질문
- AI가 맞춤형 답변 제공

### 4. 리포트 다운로드
- 전체 분석 리포트 (JSON)
- 요약 리포트 (JSON)

## 🔧 개발 가이드

### 에이전트 추가하기
새로운 에이전트를 추가하려면:

1. `src/agents/` 폴더에 새 에이전트 클래스 생성
2. `deal_lens_orchestrator.py`에 에이전트 통합
3. 필요한 경우 UI에 새 기능 추가

### 데이터 소스 확장
- `src/analyzers/` 폴더에 새로운 분석 모듈 추가
- 데이터베이스 연결 또는 API 연동
- 벡터 스토어 업데이트

## 📊 성능 최적화

### 벡터 검색 최적화
- FAISS 인덱스 튜닝
- 임베딩 모델 최적화
- 검색 결과 캐싱

### LLM 응답 최적화
- 프롬프트 엔지니어링
- 응답 캐싱
- 배치 처리

## 🔧 문제 해결

### 일반적인 오류
1. **Azure OpenAI 연결 오류**: `.env` 파일의 설정을 확인하세요
2. **포트 충돌**: 다른 포트를 사용하거나 기존 프로세스를 종료하세요
3. **의존성 오류**: `pip install -r requirements.txt`를 다시 실행하세요
4. **메모리 부족**: Docker 컨테이너 메모리 제한 증가

### 로그 확인
```bash
# Streamlit 로그
streamlit run main.py --logger.level debug

# Docker 로그
docker logs <container-id>
```

## 🚀 향후 계획

- [ ] PDF/DOCX 파일 업로드 지원
- [ ] 실시간 경쟁사 데이터 수집
- [ ] 고급 시각화 차트
- [ ] 팀 협업 기능
- [ ] API 엔드포인트 제공
- [ ] 모바일 앱 지원

## 📝 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 있거나 제안사항이 있으시면 이슈를 생성해주세요.

---

**DealLens**로 더 스마트한 수주 전략을 수립하세요! 🎯
