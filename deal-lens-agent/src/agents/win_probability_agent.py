"""
D. 난이도 스코어링 에이전트 (Win Probability)
평가 가중치 × (당사 적합도 − 경쟁사 우위) 기반 분포화
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


@dataclass
class EvaluationScore:
    """평가 점수"""
    criteria: str
    weight: float
    our_score: float
    competitor_avg_score: float
    score_difference: float
    weighted_score: float


@dataclass
class RiskFactor:
    """리스크 요인"""
    risk_type: str
    severity: str  # 상/중/하
    impact: float  # 0-1
    probability: float  # 0-1
    mitigation: Optional[str]


@dataclass
class WinProbabilityResult:
    """수주 난이도 예측 결과"""
    overall_difficulty: str  # 상/중/하
    win_probability: float  # 0-1
    evaluation_scores: List[EvaluationScore]
    risk_factors: List[RiskFactor]
    key_drivers: List[str]
    confidence_level: float


class WinProbabilityAgent:
    """난이도 스코어링 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_prompts()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 수주 난이도 예측 전문가입니다. 
            주어진 정보를 바탕으로 다음을 분석해주세요:
            
            1. 평가 기준별 점수 계산
            2. 리스크 요인 분석
            3. 수주 난이도 예측 (상/중/하)
            4. 핵심 드라이버 식별
            
            JSON 형태로 구조화된 응답을 제공해주세요."""),
            ("user", """평가 기준: {evaluation_criteria}
            내부 매칭 결과: {internal_match}
            경쟁사 분석: {competitor_analysis}
            리스크 플래그: {risk_flags}""")
        ])
    
    def calculate_win_probability(self, 
                                evaluation_criteria: List[Dict],
                                internal_match: Dict,
                                competitor_analysis: Dict,
                                risk_flags: List[Dict]) -> WinProbabilityResult:
        """수주 난이도 및 확률 계산"""
        try:
            # 평가 기준별 점수 계산
            evaluation_scores = self._calculate_evaluation_scores(
                evaluation_criteria, internal_match, competitor_analysis
            )
            
            # 리스크 요인 분석
            risk_factors = self._analyze_risk_factors(risk_flags)
            
            # 전체 점수 계산
            total_score = sum(score.weighted_score for score in evaluation_scores)
            
            # 리스크 조정
            risk_adjustment = self._calculate_risk_adjustment(risk_factors)
            adjusted_score = total_score * (1 - risk_adjustment)
            
            # 수주 난이도 및 확률 결정
            difficulty, win_probability = self._determine_difficulty_and_probability(
                adjusted_score, risk_factors
            )
            
            # 핵심 드라이버 식별
            key_drivers = self._identify_key_drivers(evaluation_scores, risk_factors)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(evaluation_scores, risk_factors)
            
            return WinProbabilityResult(
                overall_difficulty=difficulty,
                win_probability=win_probability,
                evaluation_scores=evaluation_scores,
                risk_factors=risk_factors,
                key_drivers=key_drivers,
                confidence_level=confidence
            )
            
        except Exception as e:
            st.error(f"수주 난이도 계산 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_result()
    
    def _calculate_evaluation_scores(self, 
                                   evaluation_criteria: List[Dict],
                                   internal_match: Dict,
                                   competitor_analysis: Dict) -> List[EvaluationScore]:
        """평가 기준별 점수 계산"""
        scores = []
        
        for criteria in evaluation_criteria:
            # 우리 점수 계산 (내부 매칭 결과 기반)
            our_score = self._calculate_our_score(criteria, internal_match)
            
            # 경쟁사 평균 점수 계산
            competitor_avg_score = self._calculate_competitor_score(
                criteria, competitor_analysis
            )
            
            # 점수 차이 계산
            score_difference = our_score - competitor_avg_score
            
            # 가중 점수 계산
            weight = criteria.get('weight', 0.2)
            weighted_score = score_difference * weight
            
            scores.append(EvaluationScore(
                criteria=criteria.get('item', 'Unknown'),
                weight=weight,
                our_score=our_score,
                competitor_avg_score=competitor_avg_score,
                score_difference=score_difference,
                weighted_score=weighted_score
            ))
        
        return scores
    
    def _calculate_our_score(self, criteria: Dict, internal_match: Dict) -> float:
        """우리 점수 계산"""
        # 내부 매칭 결과를 기반으로 점수 계산
        readiness = internal_match.get('overall_readiness', '하')
        confidence = internal_match.get('confidence_score', 0.0)
        
        # 준비도에 따른 기본 점수
        readiness_scores = {'상': 0.9, '중': 0.6, '하': 0.3}
        base_score = readiness_scores.get(readiness, 0.3)
        
        # 신뢰도로 조정
        adjusted_score = base_score * (0.5 + confidence * 0.5)
        
        return min(1.0, max(0.0, adjusted_score))
    
    def _calculate_competitor_score(self, criteria: Dict, competitor_analysis: Dict) -> float:
        """경쟁사 평균 점수 계산"""
        # 경쟁사 분석 결과를 기반으로 점수 계산
        profiles = competitor_analysis.get('competitor_profiles', [])
        
        if not profiles:
            return 0.5  # 기본 경쟁사 점수
        
        # 경쟁사별 점수 계산 (간단한 예시)
        total_score = 0
        for profile in profiles:
            # 강점과 약점을 기반으로 점수 계산
            strength_score = len(profile.strengths) * 0.1
            weakness_penalty = len(profile.weaknesses) * 0.05
            score = min(1.0, max(0.0, 0.5 + strength_score - weakness_penalty))
            total_score += score
        
        return total_score / len(profiles)
    
    def _analyze_risk_factors(self, risk_flags: List[Dict]) -> List[RiskFactor]:
        """리스크 요인 분석"""
        risk_factors = []
        
        for risk in risk_flags:
            risk_type = risk.get('risk_type', 'Unknown')
            severity = risk.get('severity', '중')
            
            # 심각도에 따른 영향도 계산
            impact_scores = {'상': 0.8, '중': 0.5, '하': 0.2}
            impact = impact_scores.get(severity, 0.5)
            
            # 발생 확률 (리스크 타입에 따라)
            probability_scores = {
                '모순': 0.3,
                '불명확': 0.6,
                '법적': 0.2,
                '보안': 0.4,
                '라이선스': 0.3
            }
            probability = probability_scores.get(risk_type, 0.4)
            
            risk_factors.append(RiskFactor(
                risk_type=risk_type,
                severity=severity,
                impact=impact,
                probability=probability,
                mitigation=risk.get('mitigation')
            ))
        
        return risk_factors
    
    def _calculate_risk_adjustment(self, risk_factors: List[RiskFactor]) -> float:
        """리스크 조정 계수 계산"""
        if not risk_factors:
            return 0.0
        
        total_risk = 0
        for risk in risk_factors:
            # 영향도 × 발생확률
            risk_score = risk.impact * risk.probability
            total_risk += risk_score
        
        # 평균 리스크 점수
        avg_risk = total_risk / len(risk_factors)
        
        # 조정 계수 (최대 0.3까지)
        return min(0.3, avg_risk)
    
    def _determine_difficulty_and_probability(self, 
                                            adjusted_score: float,
                                            risk_factors: List[RiskFactor]) -> Tuple[str, float]:
        """수주 난이도 및 확률 결정"""
        # 점수를 확률로 변환 (0-1 범위)
        base_probability = (adjusted_score + 1) / 2  # -1~1을 0~1로 변환
        
        # 리스크 요인으로 조정
        high_risk_count = len([r for r in risk_factors if r.severity == '상'])
        risk_penalty = high_risk_count * 0.1
        
        win_probability = max(0.0, min(1.0, base_probability - risk_penalty))
        
        # 난이도 결정
        if win_probability >= 0.7:
            difficulty = "하"
        elif win_probability >= 0.4:
            difficulty = "중"
        else:
            difficulty = "상"
        
        return difficulty, win_probability
    
    def _identify_key_drivers(self, 
                            evaluation_scores: List[EvaluationScore],
                            risk_factors: List[RiskFactor]) -> List[str]:
        """핵심 드라이버 식별"""
        drivers = []
        
        # 가장 높은 가중 점수를 가진 평가 기준
        top_score = max(evaluation_scores, key=lambda x: abs(x.weighted_score))
        if top_score.weighted_score > 0:
            drivers.append(f"{top_score.criteria}에서 경쟁 우위")
        else:
            drivers.append(f"{top_score.criteria}에서 경쟁 열위")
        
        # 가장 심각한 리스크
        high_risks = [r for r in risk_factors if r.severity == '상']
        if high_risks:
            drivers.append(f"{high_risks[0].risk_type} 리스크 관리 필요")
        
        # 일반적인 드라이버
        drivers.extend([
            "기술적 역량",
            "프로젝트 관리 능력",
            "가격 경쟁력"
        ])
        
        return drivers[:5]  # 상위 5개만 반환
    
    def _calculate_confidence(self, 
                            evaluation_scores: List[EvaluationScore],
                            risk_factors: List[RiskFactor]) -> float:
        """신뢰도 계산"""
        # 평가 기준 수에 따른 신뢰도
        criteria_confidence = min(1.0, len(evaluation_scores) / 5)
        
        # 리스크 요인 수에 따른 신뢰도 (리스크가 적을수록 신뢰도 높음)
        risk_confidence = max(0.3, 1.0 - len(risk_factors) * 0.1)
        
        # 전체 신뢰도
        overall_confidence = (criteria_confidence + risk_confidence) / 2
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _create_default_result(self) -> WinProbabilityResult:
        """기본 결과 생성"""
        return WinProbabilityResult(
            overall_difficulty="중",
            win_probability=0.5,
            evaluation_scores=[],
            risk_factors=[],
            key_drivers=[],
            confidence_level=0.0
        )
    
    def get_scoring_summary(self, result: WinProbabilityResult) -> Dict:
        """점수 결과 요약"""
        return {
            "전체_난이도": result.overall_difficulty,
            "수주_확률": result.win_probability,
            "평가_기준_수": len(result.evaluation_scores),
            "리스크_요인_수": len(result.risk_factors),
            "핵심_드라이버_수": len(result.key_drivers),
            "신뢰도": result.confidence_level
        }
