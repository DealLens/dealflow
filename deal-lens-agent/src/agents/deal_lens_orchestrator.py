"""
DealLens 메인 오케스트레이터
모든 에이전트를 조율하여 전체 분석 파이프라인을 실행합니다.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st

# 에이전트 임포트
from .rfp_understanding_agent import RFPUnderstandingAgent, RFPAnalysisResult
from .internal_rag_agent import InternalRAGAgent, InternalMatchResult
from .competitor_intelligence_agent import CompetitorIntelligenceAgent, CompetitiveAnalysis
from .win_probability_agent import WinProbabilityAgent, WinProbabilityResult
from .strategy_synthesizer_agent import StrategySynthesizerAgent, StrategySynthesis
from .reporting_agent import ReportingAgent


@dataclass
class AnalysisPipelineResult:
    """전체 분석 파이프라인 결과"""
    rfp_analysis: RFPAnalysisResult
    internal_match: InternalMatchResult
    competitor_analysis: CompetitiveAnalysis
    win_probability: WinProbabilityResult
    strategy_synthesis: StrategySynthesis
    execution_summary: Dict[str, Any]


class DealLensOrchestrator:
    """DealLens 메인 오케스트레이터"""
    
    def __init__(self, llm: AzureChatOpenAI, embeddings: AzureOpenAIEmbeddings):
        self.llm = llm
        self.embeddings = embeddings
        
        # 에이전트 초기화
        self.rfp_agent = RFPUnderstandingAgent(llm)
        self.internal_rag_agent = InternalRAGAgent(llm, embeddings)
        self.competitor_agent = CompetitorIntelligenceAgent(llm)
        self.win_probability_agent = WinProbabilityAgent(llm)
        self.strategy_agent = StrategySynthesizerAgent(llm)
        self.reporting_agent = ReportingAgent(llm)
        
        # 기본 경쟁사 목록
        self.default_competitors = [
            "삼성 SDS", "LG CNS", "포스코DX", "KT", 
            "현대 오토에버", "카카오", "CJ 올리브네트웍스"
        ]
    
    def run_full_analysis(self, 
                         rfp_content: str,
                         competitors: Optional[List[str]] = None,
                         additional_info: Optional[Dict] = None) -> AnalysisPipelineResult:
        """전체 분석 파이프라인 실행"""
        
        if competitors is None:
            competitors = self.default_competitors
        
        try:
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. RFP 분석 & 이해
            status_text.text("1/6: RFP 분석 중...")
            progress_bar.progress(1/6)
            rfp_analysis = self.rfp_agent.analyze_rfp(rfp_content)
            
            # 2. 내부지식 매칭
            status_text.text("2/6: 내부 지식 매칭 중...")
            progress_bar.progress(2/6)
            requirements = [req.description for req in rfp_analysis.requirements]
            internal_match = self.internal_rag_agent.match_requirements(requirements)
            
            # 3. 경쟁사 분석
            status_text.text("3/6: 경쟁사 분석 중...")
            progress_bar.progress(3/6)
            competitor_analysis = self.competitor_agent.analyze_competitors(
                competitors, requirements
            )
            
            # 4. 난이도 스코어링
            status_text.text("4/6: 수주 난이도 계산 중...")
            progress_bar.progress(4/6)
            win_probability = self.win_probability_agent.calculate_win_probability(
                [asdict(criteria) for criteria in rfp_analysis.evaluation_criteria],
                asdict(internal_match),
                asdict(competitor_analysis),
                [asdict(risk) for risk in rfp_analysis.risk_flags]
            )
            
            # 5. 전략 합성
            status_text.text("5/6: 전략 합성 중...")
            progress_bar.progress(5/6)
            strategy_synthesis = self.strategy_agent.synthesize_strategy(
                asdict(rfp_analysis),
                asdict(internal_match),
                asdict(competitor_analysis),
                asdict(win_probability)
            )
            
            # 6. 완료
            status_text.text("6/6: 분석 완료!")
            progress_bar.progress(1.0)
            
            # 실행 요약 생성
            execution_summary = self._generate_execution_summary(
                rfp_analysis, internal_match, competitor_analysis, 
                win_probability, strategy_synthesis
            )
            
            return AnalysisPipelineResult(
                rfp_analysis=rfp_analysis,
                internal_match=internal_match,
                competitor_analysis=competitor_analysis,
                win_probability=win_probability,
                strategy_synthesis=strategy_synthesis,
                execution_summary=execution_summary
            )
            
        except Exception as e:
            st.error(f"분석 파이프라인 실행 중 오류가 발생했습니다: {str(e)}")
            return self._create_error_result()
    
    def generate_user_response(self, 
                             user_question: str,
                             analysis_result: AnalysisPipelineResult) -> Dict[str, Any]:
        """사용자 질문에 대한 응답 생성"""
        try:
            # 분석 결과를 딕셔너리로 변환
            analysis_data = {
                'rfp_analysis': asdict(analysis_result.rfp_analysis),
                'internal_match': asdict(analysis_result.internal_match),
                'competitor_analysis': asdict(analysis_result.competitor_analysis),
                'win_probability': asdict(analysis_result.win_probability),
                'strategy_synthesis': asdict(analysis_result.strategy_synthesis)
            }
            
            # 리포팅 에이전트로 응답 생성
            response = self.reporting_agent.generate_response(user_question, analysis_data)
            
            return response
            
        except Exception as e:
            st.error(f"사용자 응답 생성 중 오류가 발생했습니다: {str(e)}")
            return {"error": str(e)}
    
    def get_analysis_summary(self, result: AnalysisPipelineResult) -> Dict[str, Any]:
        """분석 결과 요약"""
        return {
            "프로젝트_제목": result.rfp_analysis.title if hasattr(result.rfp_analysis, 'title') else "RFP 분석",
            "수주_난이도": result.win_probability.overall_difficulty,
            "수주_확률": f"{result.win_probability.win_probability:.1%}",
            "요구사항_수": len(result.rfp_analysis.requirements),
            "경쟁사_수": len(result.competitor_analysis.competitor_profiles),
            "리스크_수": len(result.rfp_analysis.risk_flags),
            "보완책_수": len(result.strategy_synthesis.improvement_actions),
            "전체_준비도": result.internal_match.overall_readiness
        }
    
    def export_analysis_report(self, result: AnalysisPipelineResult, format: str = "json") -> str:
        """분석 결과 리포트 내보내기"""
        try:
            if format == "json":
                return json.dumps(asdict(result), ensure_ascii=False, indent=2)
            elif format == "summary":
                summary = self.get_analysis_summary(result)
                return json.dumps(summary, ensure_ascii=False, indent=2)
            else:
                return "지원하지 않는 형식입니다."
                
        except Exception as e:
            return f"내보내기 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_execution_summary(self, 
                                  rfp_analysis: RFPAnalysisResult,
                                  internal_match: InternalMatchResult,
                                  competitor_analysis: CompetitiveAnalysis,
                                  win_probability: WinProbabilityResult,
                                  strategy_synthesis: StrategySynthesis) -> Dict[str, Any]:
        """실행 요약 생성"""
        return {
            "execution_time": "분석 완료",
            "total_requirements": len(rfp_analysis.requirements),
            "total_competitors": len(competitor_analysis.competitor_profiles),
            "total_risks": len(rfp_analysis.risk_flags),
            "overall_difficulty": win_probability.overall_difficulty,
            "win_probability": win_probability.win_probability,
            "confidence_level": win_probability.confidence_level,
            "key_recommendations": len(strategy_synthesis.strategic_recommendations)
        }
    
    def _create_error_result(self) -> AnalysisPipelineResult:
        """오류 결과 생성"""
        from .rfp_understanding_agent import RFPAnalysisResult
        from .internal_rag_agent import InternalMatchResult
        from .competitor_intelligence_agent import CompetitiveAnalysis
        from .win_probability_agent import WinProbabilityResult
        from .strategy_synthesizer_agent import StrategySynthesis
        
        return AnalysisPipelineResult(
            rfp_analysis=RFPAnalysisResult(
                requirements=[], evaluation_criteria=[], risk_flags=[],
                timeline=None, budget_range=None, submission_format=None
            ),
            internal_match=InternalMatchResult(
                project_matches=[], skill_gaps=[], references=[],
                overall_readiness="하", confidence_score=0.0
            ),
            competitor_analysis=CompetitiveAnalysis(
                competitor_profiles=[], our_advantages=[], our_disadvantages=[],
                differentiation_points=[], market_positioning={}
            ),
            win_probability=WinProbabilityResult(
                overall_difficulty="중", win_probability=0.5,
                evaluation_scores=[], risk_factors=[], key_drivers=[],
                confidence_level=0.0
            ),
            strategy_synthesis=StrategySynthesis(
                improvement_actions=[], company_swot=None,
                differentiation_messages=[], strategic_recommendations=[],
                success_factors=[]
            ),
            execution_summary={"error": "분석 실행 실패"}
        )
    
    def validate_inputs(self, rfp_content: str, competitors: List[str]) -> Dict[str, Any]:
        """입력 검증"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # RFP 내용 검증
        if not rfp_content or len(rfp_content.strip()) < 100:
            validation_result["errors"].append("RFP 내용이 너무 짧습니다. 최소 100자 이상 필요합니다.")
            validation_result["is_valid"] = False
        
        # 경쟁사 목록 검증
        if not competitors or len(competitors) == 0:
            validation_result["warnings"].append("경쟁사 목록이 비어있습니다. 기본 목록을 사용합니다.")
        
        if len(competitors) > 10:
            validation_result["warnings"].append("경쟁사가 너무 많습니다. 상위 10개만 분석합니다.")
        
        return validation_result
