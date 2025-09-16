"""
DealLens: 수주·실주 인사이트 AI 에이전트
Streamlit 메인 애플리케이션
"""

import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# 에이전트 임포트
from src.agents.deal_lens_orchestrator import DealLensOrchestrator

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_EMBEDDINGS = os.getenv("AOAI_DEPLOY_EMBEDDINGS", "text-embedding-ada-002")

# 기본 경쟁사 목록
DEFAULT_COMPETITORS = [
    "삼성 SDS", "LG CNS", "포스코DX", "KT", 
    "현대 오토에버", "카카오", "CJ 올리브네트웍스"
]


def initialize_agents():
    """에이전트 초기화"""
    try:
        # Azure OpenAI LLM 설정
        llm = AzureChatOpenAI(
            azure_deployment=AOAI_DEPLOY_GPT4O,
            azure_endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            api_version="2024-02-15-preview",
            temperature=0.7
        )
        
        # Azure OpenAI Embeddings 설정
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AOAI_DEPLOY_EMBEDDINGS,
            azure_endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            api_version="2024-02-15-preview"
        )
        
        # 오케스트레이터 초기화
        orchestrator = DealLensOrchestrator(llm, embeddings)
        
        return orchestrator
        
    except Exception as e:
        st.error(f"에이전트 초기화 실패: {str(e)}")
        return None


def main():
    """메인 애플리케이션"""
    st.set_page_config(
        page_title="DealLens: 수주·실주 인사이트 AI 에이전트",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 헤더
    st.title("🎯 DealLens: 수주·실주 인사이트 AI 에이전트")
    st.markdown("""
    **과거 프로젝트·경쟁사·고객 맥락 데이터를 수집·정규화하여 비교 SWOT, 보완책, 난이도 예측을 한 번에 제공합니다.**
    """)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Azure OpenAI 설정 확인
        if not all([AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O]):
            st.error("⚠️ Azure OpenAI 설정이 필요합니다.")
            st.info("환경 변수를 설정해주세요:\n- AOAI_ENDPOINT\n- AOAI_API_KEY\n- AOAI_DEPLOY_GPT4O")
            return
        
        st.success("✅ Azure OpenAI 설정 완료")
        
        # 경쟁사 선택
        st.header("🏢 경쟁사 선택")
        selected_competitors = st.multiselect(
            "분석할 경쟁사를 선택하세요:",
            options=DEFAULT_COMPETITORS,
            default=DEFAULT_COMPETITORS[:5]
        )
        
        if not selected_competitors:
            st.warning("최소 1개 이상의 경쟁사를 선택해주세요.")
            return
    
    # 메인 컨텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["📄 RFP 분석", "📊 분석 결과", "💬 Q&A", "📋 리포트"])
    
    with tab1:
        st.header("📄 RFP 문서 분석")
        
        # RFP 업로드 또는 입력
        upload_method = st.radio(
            "RFP 입력 방법을 선택하세요:",
            ["파일 업로드", "직접 입력"]
        )
        
        rfp_content = ""
        
        if upload_method == "파일 업로드":
            uploaded_file = st.file_uploader(
                "RFP 파일을 업로드하세요 (PDF, TXT, DOCX)",
                type=['pdf', 'txt', 'docx']
            )
            
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    rfp_content = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("현재는 텍스트 파일만 지원됩니다. PDF/DOCX 지원은 개발 중입니다.")
        
        else:
            rfp_content = st.text_area(
                "RFP 내용을 입력하세요:",
                height=300,
                placeholder="RFP 문서의 주요 내용을 입력해주세요..."
            )
        
        # 추가 정보 입력
        st.subheader("📝 추가 정보")
        col1, col2 = st.columns(2)
        
        with col1:
            recent_improvements = st.text_area(
                "최근 개선한 과거 약점:",
                placeholder="예: 클라우드 전문가 영입, AI 팀 강화 등"
            )
            
            business_direction = st.text_area(
                "추구하는 사업 방향:",
                placeholder="예: AI/클라우드 중심, 중소기업 시장 진출 등"
            )
        
        with col2:
            budget_info = st.text_input(
                "예산 정보:",
                placeholder="예: 50억원, 100억원 등"
            )
            
            manpower_info = st.text_input(
                "인력 정보:",
                placeholder="예: 개발자 10명, PM 2명 등"
            )
        
        # 분석 실행 버튼
        if st.button("🚀 분석 시작", type="primary", disabled=not rfp_content.strip()):
            if not rfp_content.strip():
                st.error("RFP 내용을 입력해주세요.")
                return
            
            # 에이전트 초기화
            orchestrator = initialize_agents()
            if not orchestrator:
                return
            
            # 입력 검증
            validation = orchestrator.validate_inputs(rfp_content, selected_competitors)
            if not validation["is_valid"]:
                for error in validation["errors"]:
                    st.error(error)
                return
            
            for warning in validation["warnings"]:
                st.warning(warning)
            
            # 세션 상태에 저장
            st.session_state.rfp_content = rfp_content
            st.session_state.selected_competitors = selected_competitors
            st.session_state.additional_info = {
                "recent_improvements": recent_improvements,
                "business_direction": business_direction,
                "budget_info": budget_info,
                "manpower_info": manpower_info
            }
            
            # 분석 실행
            with st.spinner("분석을 진행하고 있습니다..."):
                analysis_result = orchestrator.run_full_analysis(
                    rfp_content, selected_competitors
                )
            
            # 결과 저장
            st.session_state.analysis_result = analysis_result
            st.session_state.orchestrator = orchestrator
            
            st.success("✅ 분석이 완료되었습니다! '분석 결과' 탭을 확인해주세요.")
    
    with tab2:
        st.header("📊 분석 결과")
        
        if "analysis_result" not in st.session_state:
            st.info("먼저 RFP 분석을 실행해주세요.")
            return
        
        analysis_result = st.session_state.analysis_result
        
        # 요약 정보
        st.subheader("📋 분석 요약")
        summary = st.session_state.orchestrator.get_analysis_summary(analysis_result)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("수주 난이도", summary["수주_난이도"])
        with col2:
            st.metric("수주 확률", summary["수주_확률"])
        with col3:
            st.metric("요구사항 수", summary["요구사항_수"])
        with col4:
            st.metric("경쟁사 수", summary["경쟁사_수"])
        
        # 상세 결과
        st.subheader("🔍 상세 분석 결과")
        
        # RFP 분석 결과
        with st.expander("📄 RFP 분석 결과"):
            rfp_analysis = analysis_result.rfp_analysis
            st.write(f"**프로젝트 제목:** {rfp_analysis.title if hasattr(rfp_analysis, 'title') else 'RFP 분석'}")
            st.write(f"**요구사항 수:** {len(rfp_analysis.requirements)}")
            st.write(f"**평가 기준 수:** {len(rfp_analysis.evaluation_criteria)}")
            st.write(f"**리스크 플래그 수:** {len(rfp_analysis.risk_flags)}")
            
            if rfp_analysis.requirements:
                st.write("**주요 요구사항:**")
                for i, req in enumerate(rfp_analysis.requirements[:5], 1):
                    st.write(f"{i}. {req.description}")
        
        # 내부 매칭 결과
        with st.expander("🏠 내부 매칭 결과"):
            internal_match = analysis_result.internal_match
            st.write(f"**전체 준비도:** {internal_match.overall_readiness}")
            st.write(f"**신뢰도 점수:** {internal_match.confidence_score:.2f}")
            st.write(f"**유사 프로젝트 수:** {len(internal_match.project_matches)}")
            st.write(f"**스킬 갭 수:** {len(internal_match.skill_gaps)}")
            
            if internal_match.skill_gaps:
                st.write("**스킬 갭:**")
                for gap in internal_match.skill_gaps:
                    st.write(f"- {gap.skill_area}: {gap.gap_size}")
        
        # 경쟁사 분석 결과
        with st.expander("🏢 경쟁사 분석 결과"):
            competitor_analysis = analysis_result.competitor_analysis
            st.write(f"**분석 경쟁사 수:** {len(competitor_analysis.competitor_profiles)}")
            st.write(f"**우리 강점 수:** {len(competitor_analysis.our_advantages)}")
            st.write(f"**우리 약점 수:** {len(competitor_analysis.our_disadvantages)}")
            
            if competitor_analysis.our_advantages:
                st.write("**우리 강점:**")
                for advantage in competitor_analysis.our_advantages:
                    st.write(f"- {advantage}")
        
        # 수주 난이도 결과
        with st.expander("🎯 수주 난이도 분석"):
            win_probability = analysis_result.win_probability
            st.write(f"**전체 난이도:** {win_probability.overall_difficulty}")
            st.write(f"**수주 확률:** {win_probability.win_probability:.1%}")
            st.write(f"**신뢰도:** {win_probability.confidence_level:.2f}")
            
            if win_probability.key_drivers:
                st.write("**핵심 드라이버:**")
                for driver in win_probability.key_drivers:
                    st.write(f"- {driver}")
        
        # 전략 합성 결과
        with st.expander("💡 전략 합성 결과"):
            strategy_synthesis = analysis_result.strategy_synthesis
            st.write(f"**보완책 액션 수:** {len(strategy_synthesis.improvement_actions)}")
            st.write(f"**전략 권고사항 수:** {len(strategy_synthesis.strategic_recommendations)}")
            st.write(f"**성공 요인 수:** {len(strategy_synthesis.success_factors)}")
            
            if strategy_synthesis.strategic_recommendations:
                st.write("**전략 권고사항:**")
                for rec in strategy_synthesis.strategic_recommendations:
                    st.write(f"- {rec}")
    
    with tab3:
        st.header("💬 Q&A")
        
        if "analysis_result" not in st.session_state:
            st.info("먼저 RFP 분석을 실행해주세요.")
            return
        
        st.write("분석 결과에 대해 궁금한 점을 질문해보세요!")
        
        # 질문 입력
        user_question = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 수주 확률이 낮은 이유는 무엇인가요? 경쟁사 대비 우리의 강점은?"
        )
        
        if st.button("질문하기", disabled=not user_question.strip()):
            if not user_question.strip():
                st.error("질문을 입력해주세요.")
                return
            
            # 응답 생성
            with st.spinner("답변을 생성하고 있습니다..."):
                response = st.session_state.orchestrator.generate_user_response(
                    user_question, st.session_state.analysis_result
                )
            
            # 응답 표시
            st.subheader("🤖 AI 답변")
            st.write(response.get("answer", "답변을 생성할 수 없습니다."))
            
            # 지원 데이터 표시
            if "supporting_data" in response:
                st.subheader("📊 지원 데이터")
                supporting_data = response["supporting_data"]
                for key, value in supporting_data.items():
                    st.write(f"**{key}:** {value}")
    
    with tab4:
        st.header("📋 리포트")
        
        if "analysis_result" not in st.session_state:
            st.info("먼저 RFP 분석을 실행해주세요.")
            return
        
        # 리포트 다운로드
        st.subheader("📥 리포트 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 전체 분석 리포트 (JSON)"):
                report_data = st.session_state.orchestrator.export_analysis_report(
                    st.session_state.analysis_result, "json"
                )
                st.download_button(
                    label="다운로드",
                    data=report_data,
                    file_name="deal_lens_analysis_report.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 요약 리포트 (JSON)"):
                summary_data = st.session_state.orchestrator.export_analysis_report(
                    st.session_state.analysis_result, "summary"
                )
                st.download_button(
                    label="다운로드",
                    data=summary_data,
                    file_name="deal_lens_summary.json",
                    mime="application/json"
                )
        
        # 리포트 미리보기
        st.subheader("👀 리포트 미리보기")
        
        if st.button("미리보기 생성"):
            # Deal Brief 생성
            response = st.session_state.orchestrator.generate_user_response(
                "전체 분석 결과를 요약해주세요", st.session_state.analysis_result
            )
            
            st.write("**Deal Brief:**")
            st.write(response.get("answer", "미리보기를 생성할 수 없습니다."))


if __name__ == "__main__":
    main()
