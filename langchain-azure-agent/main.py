from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI")

class DealflowAgent:
    def __init__(self):
        """Dealflow 에이전트 초기화"""
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
    
    def _setup_llm(self):
        """Azure OpenAI LLM 설정"""
        return AzureChatOpenAI(
            azure_deployment=AOAI_DEPLOY_GPT4O,
            azure_endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            api_version="2024-02-15-preview",
            temperature=0.7
        )
    
    def _setup_tools(self):
        """에이전트가 사용할 도구들 설정"""
        def search_deals(query: str) -> str:
            """거래 정보를 검색합니다."""
            # 실제로는 데이터베이스나 API를 호출
            return f"'{query}'에 대한 거래 정보를 검색했습니다. (실제 구현 필요)"
        
        def analyze_market(query: str) -> str:
            """시장 분석을 수행합니다."""
            return f"'{query}'에 대한 시장 분석을 완료했습니다. (실제 구현 필요)"
        
        def get_company_info(company: str) -> str:
            """회사 정보를 조회합니다."""
            return f"'{company}'의 회사 정보를 조회했습니다. (실제 구현 필요)"
        
        return [
            Tool(
                name="search_deals",
                description="거래 정보를 검색합니다. 거래, 투자, M&A 관련 질문에 사용하세요.",
                func=search_deals
            ),
            Tool(
                name="analyze_market",
                description="시장 분석을 수행합니다. 시장 동향, 트렌드 분석에 사용하세요.",
                func=analyze_market
            ),
            Tool(
                name="get_company_info",
                description="회사 정보를 조회합니다. 특정 회사에 대한 정보가 필요할 때 사용하세요.",
                func=get_company_info
            )
        ]
    
    def _setup_agent(self):
        """LangChain 에이전트 설정"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 Dealflow 전문 AI 어시스턴트입니다. 
            투자, M&A, 스타트업, 벤처캐피털 관련 질문에 전문적으로 답변합니다.
            사용 가능한 도구들을 적절히 활용하여 정확하고 유용한 정보를 제공하세요.
            한국어로 친근하고 전문적으로 답변해주세요."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def chat(self, message: str) -> str:
        """사용자 메시지에 대한 응답 생성"""
        try:
            response = self.agent.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

# Streamlit 앱
def main():
    st.set_page_config(
        page_title="💼 Dealflow Agent",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("💼 Dealflow AI Agent")
    st.markdown("투자, M&A, 스타트업 관련 질문에 답변하는 AI 어시스턴트입니다.")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("에이전트를 초기화하는 중..."):
            st.session_state.agent = DealflowAgent()
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                response = st.session_state.agent.chat(prompt)
            st.markdown(response)
        
        # AI 응답 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        st.info("Azure OpenAI 설정이 완료되어야 합니다.")
        
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("📊 사용 가능한 도구")
        st.markdown("""
        - 🔍 **거래 검색**: 투자, M&A 거래 정보
        - 📈 **시장 분석**: 시장 동향 및 트렌드
        - 🏢 **회사 정보**: 기업 정보 조회
        """)

if __name__ == "__main__":
    main()