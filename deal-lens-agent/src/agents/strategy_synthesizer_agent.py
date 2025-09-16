"""
E. 전략 합성 에이전트 (Strategy Synthesizer)
보완책, 당사 SWOT, 경쟁사 대비 차별화 메시지 통합
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


@dataclass
class ImprovementAction:
    """보완책 액션"""
    action_type: str  # 내부보강/외부파트너/PoC제안/스펙질의
    description: str
    priority: str  # 상/중/하
    timeline: str
    cost_estimate: Optional[str]
    success_probability: float


@dataclass
class CompanySWOT:
    """당사 SWOT 분석"""
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    strategic_focus: List[str]


@dataclass
class DifferentiationMessage:
    """차별화 메시지"""
    key_point: str
    supporting_evidence: List[str]
    evaluation_criteria: str
    competitive_advantage: str


@dataclass
class StrategySynthesis:
    """전략 합성 결과"""
    improvement_actions: List[ImprovementAction]
    company_swot: CompanySWOT
    differentiation_messages: List[DifferentiationMessage]
    strategic_recommendations: List[str]
    success_factors: List[str]


class StrategySynthesizerAgent:
    """전략 합성 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_prompts()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 전략 합성 전문가입니다. 
            주어진 모든 분석 결과를 종합하여 다음을 생성해주세요:
            
            1. 보완책 액션 플랜 (우선순위별)
            2. 당사 SWOT 분석 (통합)
            3. 경쟁사 대비 차별화 메시지
            4. 전략적 권고사항
            5. 성공 요인
            
            JSON 형태로 구조화된 응답을 제공해주세요."""),
            ("user", """RFP 분석: {rfp_analysis}
            내부 매칭: {internal_match}
            경쟁사 분석: {competitor_analysis}
            수주 난이도: {win_probability}""")
        ])
    
    def synthesize_strategy(self, 
                          rfp_analysis: Dict,
                          internal_match: Dict,
                          competitor_analysis: Dict,
                          win_probability: Dict) -> StrategySynthesis:
        """전략 합성 수행"""
        try:
            # 보완책 액션 생성
            improvement_actions = self._generate_improvement_actions(
                rfp_analysis, internal_match, win_probability
            )
            
            # 당사 SWOT 통합
            company_swot = self._synthesize_company_swot(
                internal_match, competitor_analysis, rfp_analysis
            )
            
            # 차별화 메시지 생성
            differentiation_messages = self._create_differentiation_messages(
                competitor_analysis, rfp_analysis
            )
            
            # 전략적 권고사항 생성
            strategic_recommendations = self._generate_strategic_recommendations(
                improvement_actions, company_swot, win_probability
            )
            
            # 성공 요인 식별
            success_factors = self._identify_success_factors(
                internal_match, competitor_analysis, win_probability
            )
            
            return StrategySynthesis(
                improvement_actions=improvement_actions,
                company_swot=company_swot,
                differentiation_messages=differentiation_messages,
                strategic_recommendations=strategic_recommendations,
                success_factors=success_factors
            )
            
        except Exception as e:
            st.error(f"전략 합성 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_synthesis()
    
    def _generate_improvement_actions(self, 
                                    rfp_analysis: Dict,
                                    internal_match: Dict,
                                    win_probability: Dict) -> List[ImprovementAction]:
        """보완책 액션 생성"""
        actions = []
        
        # 스킬 갭 기반 액션
        skill_gaps = internal_match.get('skill_gaps', [])
        for gap in skill_gaps:
            if gap.gap_size == "심각":
                actions.append(ImprovementAction(
                    action_type="내부보강",
                    description=f"{gap.skill_area} 전문가 영입",
                    priority="상",
                    timeline="1-2개월",
                    cost_estimate="높음",
                    success_probability=0.8
                ))
            elif gap.gap_size == "부족":
                actions.append(ImprovementAction(
                    action_type="외부파트너",
                    description=f"{gap.skill_area} 파트너십 구축",
                    priority="중",
                    timeline="2-3개월",
                    cost_estimate="중간",
                    success_probability=0.7
                ))
        
        # 리스크 기반 액션
        risk_flags = rfp_analysis.get('risk_flags', [])
        for risk in risk_flags:
            if risk.severity == "상":
                actions.append(ImprovementAction(
                    action_type="스펙질의",
                    description=f"{risk.risk_type} 관련 요구사항 명확화",
                    priority="상",
                    timeline="1주",
                    cost_estimate="낮음",
                    success_probability=0.9
                ))
        
        # 일반적인 액션
        actions.extend([
            ImprovementAction(
                action_type="PoC제안",
                description="핵심 기술 PoC 제안",
                priority="중",
                timeline="2-4주",
                cost_estimate="중간",
                success_probability=0.6
            ),
            ImprovementAction(
                action_type="내부보강",
                description="프로젝트 관리 프로세스 개선",
                priority="하",
                timeline="1개월",
                cost_estimate="낮음",
                success_probability=0.8
            )
        ])
        
        return actions[:8]  # 상위 8개만 반환
    
    def _synthesize_company_swot(self, 
                               internal_match: Dict,
                               competitor_analysis: Dict,
                               rfp_analysis: Dict) -> CompanySWOT:
        """당사 SWOT 통합"""
        # 강점 (내부 매칭 + 경쟁사 분석에서 우리 우위)
        strengths = []
        strengths.extend(internal_match.get('our_advantages', []))
        strengths.extend([
            "빠른 의사결정 구조",
            "유연한 프로젝트 관리",
            "혁신적인 기술 접근"
        ])
        
        # 약점 (내부 매칭 + 경쟁사 분석에서 우리 열위)
        weaknesses = []
        weaknesses.extend(internal_match.get('our_disadvantages', []))
        skill_gaps = internal_match.get('skill_gaps', [])
        for gap in skill_gaps:
            if gap.gap_size in ["부족", "심각"]:
                weaknesses.append(f"{gap.skill_area} 역량 부족")
        
        # 기회 (RFP 요구사항 + 시장 트렌드)
        opportunities = [
            "디지털 전환 시장 확대",
            "AI/클라우드 수요 증가",
            "중소기업 시장 진출 기회"
        ]
        
        # 위협 (경쟁사 + 시장 환경)
        threats = [
            "대기업 경쟁 심화",
            "기술 변화 속도",
            "가격 경쟁 압박"
        ]
        
        # 전략적 포커스
        strategic_focus = [
            "고객 맞춤형 솔루션",
            "빠른 프로토타이핑",
            "경쟁력 있는 가격"
        ]
        
        return CompanySWOT(
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            strategic_focus=strategic_focus
        )
    
    def _create_differentiation_messages(self, 
                                       competitor_analysis: Dict,
                                       rfp_analysis: Dict) -> List[DifferentiationMessage]:
        """차별화 메시지 생성"""
        messages = []
        
        # 평가 기준별 차별화 메시지
        evaluation_criteria = rfp_analysis.get('evaluation_criteria', [])
        differentiation_points = competitor_analysis.get('differentiation_points', [])
        
        for i, criteria in enumerate(evaluation_criteria[:3]):
            if i < len(differentiation_points):
                messages.append(DifferentiationMessage(
                    key_point=differentiation_points[i],
                    supporting_evidence=[
                        "과거 프로젝트 성공 사례",
                        "고객 만족도 95% 이상",
                        "빠른 프로젝트 완료율"
                    ],
                    evaluation_criteria=criteria.get('item', 'Unknown'),
                    competitive_advantage="경쟁사 대비 30% 빠른 완료"
                ))
        
        # 기본 차별화 메시지
        if not messages:
            messages.append(DifferentiationMessage(
                key_point="고객 맞춤형 솔루션",
                supporting_evidence=[
                    "유연한 개발 프로세스",
                    "투명한 커뮤니케이션",
                    "지속적인 고객 지원"
                ],
                evaluation_criteria="전체 평가",
                competitive_advantage="고객 중심의 접근 방식"
            ))
        
        return messages
    
    def _generate_strategic_recommendations(self, 
                                          improvement_actions: List[ImprovementAction],
                                          company_swot: CompanySWOT,
                                          win_probability: Dict) -> List[str]:
        """전략적 권고사항 생성"""
        recommendations = []
        
        # 수주 난이도 기반 권고사항
        difficulty = win_probability.get('overall_difficulty', '중')
        if difficulty == "상":
            recommendations.extend([
                "높은 난이도 프로젝트 - 신중한 접근 필요",
                "리스크 관리 강화 및 대안 계획 수립",
                "경쟁사 대비 차별화 포인트 강화"
            ])
        elif difficulty == "중":
            recommendations.extend([
                "균형잡힌 접근으로 경쟁력 확보",
                "핵심 요구사항 충족에 집중",
                "가격 경쟁력 유지"
            ])
        else:
            recommendations.extend([
                "우리 강점을 최대한 활용",
                "빠른 의사결정으로 선점 효과",
                "고객 관계 강화"
            ])
        
        # SWOT 기반 권고사항
        if len(company_swot.weaknesses) > len(company_swot.strengths):
            recommendations.append("약점 보완을 위한 전략적 투자 필요")
        
        # 액션 기반 권고사항
        high_priority_actions = [a for a in improvement_actions if a.priority == "상"]
        if high_priority_actions:
            recommendations.append("우선순위 높은 보완책 즉시 실행")
        
        return recommendations[:5]  # 상위 5개만 반환
    
    def _identify_success_factors(self, 
                                internal_match: Dict,
                                competitor_analysis: Dict,
                                win_probability: Dict) -> List[str]:
        """성공 요인 식별"""
        success_factors = []
        
        # 내부 매칭 기반 성공 요인
        readiness = internal_match.get('overall_readiness', '하')
        if readiness == "상":
            success_factors.append("높은 내부 준비도")
        
        # 경쟁사 분석 기반 성공 요인
        our_advantages = competitor_analysis.get('our_advantages', [])
        if our_advantages:
            success_factors.append("경쟁사 대비 차별화된 강점")
        
        # 수주 확률 기반 성공 요인
        win_prob = win_probability.get('win_probability', 0.5)
        if win_prob > 0.6:
            success_factors.append("높은 수주 확률")
        
        # 일반적인 성공 요인
        success_factors.extend([
            "명확한 요구사항 이해",
            "효과적인 프로젝트 관리",
            "고객과의 긴밀한 협력",
            "기술적 우수성",
            "경쟁력 있는 제안서"
        ])
        
        return success_factors[:7]  # 상위 7개만 반환
    
    def _create_default_synthesis(self) -> StrategySynthesis:
        """기본 합성 결과 생성"""
        return StrategySynthesis(
            improvement_actions=[],
            company_swot=CompanySWOT(
                strengths=[],
                weaknesses=[],
                opportunities=[],
                threats=[],
                strategic_focus=[]
            ),
            differentiation_messages=[],
            strategic_recommendations=[],
            success_factors=[]
        )
    
    def get_synthesis_summary(self, synthesis: StrategySynthesis) -> Dict:
        """합성 결과 요약"""
        return {
            "보완책_액션_수": len(synthesis.improvement_actions),
            "강점_수": len(synthesis.company_swot.strengths),
            "약점_수": len(synthesis.company_swot.weaknesses),
            "차별화_메시지_수": len(synthesis.differentiation_messages),
            "전략_권고사항_수": len(synthesis.strategic_recommendations),
            "성공_요인_수": len(synthesis.success_factors)
        }
