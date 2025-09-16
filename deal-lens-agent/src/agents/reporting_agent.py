"""
F. Q&A / 리포팅 에이전트 (User-facing)
사용자의 자연어 질문에 맞춰 결과 재구성/시각화
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import pandas as pd


@dataclass
class DealBrief:
    """Deal Brief 데이터"""
    project_title: str
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    risk_assessment: str
    win_probability: str
    next_steps: List[str]


@dataclass
class ReportSection:
    """리포트 섹션"""
    title: str
    content: str
    charts_data: Optional[Dict[str, Any]]
    tables_data: Optional[List[Dict[str, Any]]]


@dataclass
class UserQuery:
    """사용자 질문"""
    question: str
    query_type: str  # summary/detail/comparison/risk
    focus_area: Optional[str]


class ReportingAgent:
    """Q&A / 리포팅 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_prompts()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.reporting_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 DealLens 리포팅 전문가입니다. 
            주어진 분석 결과를 바탕으로 사용자 질문에 맞는 답변을 생성해주세요:
            
            1. Deal Brief (1-2페이지 요약)
            2. 상세 섹션별 분석
            3. 시각화 데이터 (차트/표)
            4. 실행 가능한 권고사항
            
            사용자 친화적이고 실행 가능한 인사이트를 제공해주세요."""),
            ("user", """사용자 질문: {user_question}
            
            분석 결과:
            RFP 분석: {rfp_analysis}
            내부 매칭: {internal_match}
            경쟁사 분석: {competitor_analysis}
            수주 난이도: {win_probability}
            전략 합성: {strategy_synthesis}""")
        ])
    
    def generate_response(self, 
                        user_question: str,
                        analysis_results: Dict) -> Dict[str, Any]:
        """사용자 질문에 대한 응답 생성"""
        try:
            # 질문 타입 분석
            query_type = self._analyze_query_type(user_question)
            
            # 질문 타입별 응답 생성
            if query_type == "summary":
                return self._generate_summary_response(analysis_results)
            elif query_type == "detail":
                return self._generate_detail_response(user_question, analysis_results)
            elif query_type == "comparison":
                return self._generate_comparison_response(analysis_results)
            elif query_type == "risk":
                return self._generate_risk_response(analysis_results)
            else:
                return self._generate_general_response(user_question, analysis_results)
                
        except Exception as e:
            st.error(f"리포팅 중 오류가 발생했습니다: {str(e)}")
            return self._create_error_response()
    
    def _analyze_query_type(self, question: str) -> str:
        """질문 타입 분석"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["요약", "summary", "개요", "전체"]):
            return "summary"
        elif any(word in question_lower for word in ["상세", "detail", "자세히", "구체적"]):
            return "detail"
        elif any(word in question_lower for word in ["비교", "comparison", "경쟁사", "차이"]):
            return "comparison"
        elif any(word in question_lower for word in ["리스크", "risk", "위험", "문제"]):
            return "risk"
        else:
            return "general"
    
    def _generate_summary_response(self, analysis_results: Dict) -> Dict[str, Any]:
        """요약 응답 생성"""
        # Deal Brief 생성
        deal_brief = self._create_deal_brief(analysis_results)
        
        # 핵심 인사이트 추출
        key_insights = self._extract_key_insights(analysis_results)
        
        # 시각화 데이터 생성
        charts_data = self._create_summary_charts(analysis_results)
        
        return {
            "response_type": "summary",
            "deal_brief": asdict(deal_brief),
            "key_insights": key_insights,
            "charts_data": charts_data,
            "recommendations": deal_brief.recommendations
        }
    
    def _generate_detail_response(self, question: str, analysis_results: Dict) -> Dict[str, Any]:
        """상세 응답 생성"""
        # 질문에 맞는 섹션 생성
        sections = self._create_detail_sections(question, analysis_results)
        
        # 관련 차트/표 데이터
        visualization_data = self._create_detail_visualizations(analysis_results)
        
        return {
            "response_type": "detail",
            "sections": [asdict(section) for section in sections],
            "visualization_data": visualization_data
        }
    
    def _generate_comparison_response(self, analysis_results: Dict) -> Dict[str, Any]:
        """비교 응답 생성"""
        # 경쟁사 비교 데이터
        comparison_data = self._create_comparison_data(analysis_results)
        
        # 비교 차트
        comparison_charts = self._create_comparison_charts(analysis_results)
        
        return {
            "response_type": "comparison",
            "comparison_data": comparison_data,
            "comparison_charts": comparison_charts
        }
    
    def _generate_risk_response(self, analysis_results: Dict) -> Dict[str, Any]:
        """리스크 응답 생성"""
        # 리스크 분석
        risk_analysis = self._analyze_risks(analysis_results)
        
        # 리스크 매트릭스
        risk_matrix = self._create_risk_matrix(analysis_results)
        
        # 리스크 완화 방안
        mitigation_plans = self._create_mitigation_plans(analysis_results)
        
        return {
            "response_type": "risk",
            "risk_analysis": risk_analysis,
            "risk_matrix": risk_matrix,
            "mitigation_plans": mitigation_plans
        }
    
    def _generate_general_response(self, question: str, analysis_results: Dict) -> Dict[str, Any]:
        """일반 응답 생성"""
        # LLM을 사용한 맞춤형 응답
        chain = self.reporting_prompt | self.llm
        response = chain.invoke({
            "user_question": question,
            "rfp_analysis": json.dumps(analysis_results.get('rfp_analysis', {})),
            "internal_match": json.dumps(analysis_results.get('internal_match', {})),
            "competitor_analysis": json.dumps(analysis_results.get('competitor_analysis', {})),
            "win_probability": json.dumps(analysis_results.get('win_probability', {})),
            "strategy_synthesis": json.dumps(analysis_results.get('strategy_synthesis', {}))
        })
        
        return {
            "response_type": "general",
            "answer": response.content,
            "supporting_data": self._extract_supporting_data(analysis_results)
        }
    
    def _create_deal_brief(self, analysis_results: Dict) -> DealBrief:
        """Deal Brief 생성"""
        rfp_analysis = analysis_results.get('rfp_analysis', {})
        win_probability = analysis_results.get('win_probability', {})
        strategy_synthesis = analysis_results.get('strategy_synthesis', {})
        
        return DealBrief(
            project_title=rfp_analysis.get('title', 'RFP 프로젝트'),
            executive_summary=self._generate_executive_summary(analysis_results),
            key_findings=self._extract_key_findings(analysis_results),
            recommendations=strategy_synthesis.get('strategic_recommendations', []),
            risk_assessment=self._assess_risks(analysis_results),
            win_probability=win_probability.get('overall_difficulty', '중'),
            next_steps=self._generate_next_steps(analysis_results)
        )
    
    def _generate_executive_summary(self, analysis_results: Dict) -> str:
        """실행 요약 생성"""
        win_prob = analysis_results.get('win_probability', {}).get('win_probability', 0.5)
        difficulty = analysis_results.get('win_probability', {}).get('overall_difficulty', '중')
        
        return f"""
        이 프로젝트는 {difficulty} 난이도로 평가되며, 수주 확률은 {win_prob:.1%}입니다. 
        주요 경쟁사들과 비교하여 우리의 강점을 최대한 활용하고, 
        식별된 약점에 대한 보완책을 통해 경쟁력을 강화할 수 있습니다.
        """
    
    def _extract_key_findings(self, analysis_results: Dict) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        # RFP 분석 결과
        rfp_analysis = analysis_results.get('rfp_analysis', {})
        if rfp_analysis.get('risk_flags'):
            findings.append(f"RFP에서 {len(rfp_analysis['risk_flags'])}개의 리스크 플래그 발견")
        
        # 내부 매칭 결과
        internal_match = analysis_results.get('internal_match', {})
        readiness = internal_match.get('overall_readiness', '하')
        findings.append(f"내부 준비도: {readiness}")
        
        # 경쟁사 분석 결과
        competitor_analysis = analysis_results.get('competitor_analysis', {})
        if competitor_analysis.get('our_advantages'):
            findings.append(f"경쟁사 대비 {len(competitor_analysis['our_advantages'])}개의 강점 보유")
        
        return findings
    
    def _assess_risks(self, analysis_results: Dict) -> str:
        """리스크 평가"""
        rfp_analysis = analysis_results.get('rfp_analysis', {})
        risk_flags = rfp_analysis.get('risk_flags', [])
        
        if not risk_flags:
            return "특별한 리스크 요인은 발견되지 않았습니다."
        
        high_risks = [r for r in risk_flags if r.get('severity') == '상']
        if high_risks:
            return f"높은 심각도의 리스크 {len(high_risks)}개 발견. 즉시 대응 필요."
        else:
            return f"중간 수준의 리스크 {len(risk_flags)}개 발견. 모니터링 필요."
    
    def _generate_next_steps(self, analysis_results: Dict) -> List[str]:
        """다음 단계 생성"""
        strategy_synthesis = analysis_results.get('strategy_synthesis', {})
        improvement_actions = strategy_synthesis.get('improvement_actions', [])
        
        next_steps = []
        
        # 우선순위 높은 액션
        high_priority_actions = [a for a in improvement_actions if a.get('priority') == '상']
        for action in high_priority_actions[:3]:
            next_steps.append(f"{action.get('description')} ({action.get('timeline')})")
        
        # 기본 다음 단계
        next_steps.extend([
            "제안서 초안 작성",
            "고객 미팅 일정 조율",
            "팀 구성 및 역할 분담"
        ])
        
        return next_steps[:5]
    
    def _create_summary_charts(self, analysis_results: Dict) -> Dict[str, Any]:
        """요약 차트 데이터 생성"""
        charts = {}
        
        # 수주 확률 차트
        win_prob = analysis_results.get('win_probability', {})
        charts['win_probability'] = {
            'type': 'gauge',
            'value': win_prob.get('win_probability', 0.5),
            'max': 1.0,
            'title': '수주 확률'
        }
        
        # 경쟁사 비교 차트
        competitor_analysis = analysis_results.get('competitor_analysis', {})
        competitor_profiles = competitor_analysis.get('competitor_profiles', [])
        charts['competitor_comparison'] = {
            'type': 'bar',
            'data': [
                {'name': profile.get('company_name', 'Unknown'), 
                 'value': len(profile.get('strengths', []))}
                for profile in competitor_profiles[:5]
            ],
            'title': '경쟁사 강점 비교'
        }
        
        return charts
    
    def _create_detail_sections(self, question: str, analysis_results: Dict) -> List[ReportSection]:
        """상세 섹션 생성"""
        sections = []
        
        # RFP 분석 섹션
        rfp_analysis = analysis_results.get('rfp_analysis', {})
        sections.append(ReportSection(
            title="RFP 분석 결과",
            content=f"총 {len(rfp_analysis.get('requirements', []))}개의 요구사항이 식별되었습니다.",
            charts_data=None,
            tables_data=None
        ))
        
        # 경쟁사 분석 섹션
        competitor_analysis = analysis_results.get('competitor_analysis', {})
        sections.append(ReportSection(
            title="경쟁사 분석",
            content=f"{len(competitor_analysis.get('competitor_profiles', []))}개 경쟁사 분석 완료",
            charts_data=None,
            tables_data=None
        ))
        
        return sections
    
    def _create_comparison_data(self, analysis_results: Dict) -> Dict[str, Any]:
        """비교 데이터 생성"""
        competitor_analysis = analysis_results.get('competitor_analysis', {})
        competitor_profiles = competitor_analysis.get('competitor_profiles', [])
        
        comparison_data = {
            'competitors': [],
            'our_position': {
                'strengths': competitor_analysis.get('our_advantages', []),
                'weaknesses': competitor_analysis.get('our_disadvantages', [])
            }
        }
        
        for profile in competitor_profiles:
            comparison_data['competitors'].append({
                'name': profile.get('company_name', 'Unknown'),
                'strengths': profile.get('strengths', []),
                'weaknesses': profile.get('weaknesses', []),
                'price_positioning': profile.get('price_positioning', 'Unknown')
            })
        
        return comparison_data
    
    def _create_risk_matrix(self, analysis_results: Dict) -> Dict[str, Any]:
        """리스크 매트릭스 생성"""
        rfp_analysis = analysis_results.get('rfp_analysis', {})
        risk_flags = rfp_analysis.get('risk_flags', [])
        
        risk_matrix = {
            'high_impact_high_probability': [],
            'high_impact_low_probability': [],
            'low_impact_high_probability': [],
            'low_impact_low_probability': []
        }
        
        for risk in risk_flags:
            severity = risk.get('severity', '중')
            risk_type = risk.get('risk_type', 'Unknown')
            
            if severity == '상':
                risk_matrix['high_impact_high_probability'].append(risk_type)
            else:
                risk_matrix['low_impact_high_probability'].append(risk_type)
        
        return risk_matrix
    
    def _create_error_response(self) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            "response_type": "error",
            "message": "분석 결과를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.",
            "error_code": "PROCESSING_ERROR"
        }
    
    def _extract_key_insights(self, analysis_results: Dict) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        
        # 수주 확률 기반 인사이트
        win_prob = analysis_results.get('win_probability', {}).get('win_probability', 0.5)
        if win_prob > 0.7:
            insights.append("높은 수주 확률 - 적극적인 접근 권장")
        elif win_prob < 0.3:
            insights.append("낮은 수주 확률 - 신중한 접근 필요")
        
        # 경쟁사 분석 기반 인사이트
        competitor_analysis = analysis_results.get('competitor_analysis', {})
        our_advantages = competitor_analysis.get('our_advantages', [])
        if len(our_advantages) > 3:
            insights.append("경쟁사 대비 다수의 강점 보유")
        
        return insights
    
    def _extract_supporting_data(self, analysis_results: Dict) -> Dict[str, Any]:
        """지원 데이터 추출"""
        return {
            'rfp_requirements_count': len(analysis_results.get('rfp_analysis', {}).get('requirements', [])),
            'competitor_count': len(analysis_results.get('competitor_analysis', {}).get('competitor_profiles', [])),
            'win_probability': analysis_results.get('win_probability', {}).get('win_probability', 0.5),
            'risk_count': len(analysis_results.get('rfp_analysis', {}).get('risk_flags', []))
        }
