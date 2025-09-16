"""
A. RFP 분석 & 이해 에이전트 (Document Understanding)
PDF(+OCR) 인식, 요구사항 구조화, 평가기준 추출, 리스크 플래그
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


@dataclass
class Requirement:
    """요구사항 데이터 클래스"""
    category: str  # 기술/운영/보안/가격
    description: str
    priority: str  # 필수/선택/권장
    complexity: str  # 상/중/하


@dataclass
class EvaluationCriteria:
    """평가기준 데이터 클래스"""
    item: str
    weight: float  # 가중치 (0-1)
    min_score: Optional[float]  # 최저점 컷오프
    evaluation_type: str  # 정량/정성
    description: str


@dataclass
class RiskFlag:
    """리스크 플래그 데이터 클래스"""
    risk_type: str  # 모순/불명확/법적/보안/라이선스
    description: str
    severity: str  # 상/중/하
    mitigation: Optional[str]


@dataclass
class RFPAnalysisResult:
    """RFP 분석 결과"""
    requirements: List[Requirement]
    evaluation_criteria: List[EvaluationCriteria]
    risk_flags: List[RiskFlag]
    timeline: Optional[str]
    budget_range: Optional[str]
    submission_format: Optional[str]


class RFPUnderstandingAgent:
    """RFP 분석 & 이해 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_prompts()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.requirement_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 RFP 문서 분석 전문가입니다. 
            주어진 RFP 문서에서 다음을 추출해주세요:
            
            1. 요구사항 구조화 (기술/운영/보안/가격별 분류)
            2. 평가기준 및 가중치
            3. 리스크 플래그 (모순, 불명확, 법적/보안/라이선스 이슈)
            4. 납기/SLA 정보
            5. 제출 양식 요구사항
            
            JSON 형태로 구조화된 응답을 제공해주세요."""),
            ("user", "다음 RFP 문서를 분석해주세요:\n\n{rfp_content}")
        ])
    
    def analyze_rfp(self, rfp_content: str) -> RFPAnalysisResult:
        """RFP 문서를 분석하여 구조화된 정보를 추출합니다."""
        try:
            # LLM을 사용한 분석
            chain = self.requirement_extraction_prompt | self.llm
            response = chain.invoke({"rfp_content": rfp_content})
            
            # 응답 파싱
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            st.error(f"RFP 분석 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_result()
    
    def _parse_analysis_response(self, response: str) -> RFPAnalysisResult:
        """LLM 응답을 파싱하여 구조화된 결과를 생성합니다."""
        try:
            # JSON 응답 파싱 시도
            if response.strip().startswith('{'):
                data = json.loads(response)
                return self._create_result_from_json(data)
            else:
                # 텍스트 응답 파싱
                return self._parse_text_response(response)
        except json.JSONDecodeError:
            return self._parse_text_response(response)
    
    def _create_result_from_json(self, data: Dict) -> RFPAnalysisResult:
        """JSON 데이터에서 결과 객체 생성"""
        requirements = [
            Requirement(**req) for req in data.get('requirements', [])
        ]
        evaluation_criteria = [
            EvaluationCriteria(**crit) for crit in data.get('evaluation_criteria', [])
        ]
        risk_flags = [
            RiskFlag(**risk) for risk in data.get('risk_flags', [])
        ]
        
        return RFPAnalysisResult(
            requirements=requirements,
            evaluation_criteria=evaluation_criteria,
            risk_flags=risk_flags,
            timeline=data.get('timeline'),
            budget_range=data.get('budget_range'),
            submission_format=data.get('submission_format')
        )
    
    def _parse_text_response(self, text: str) -> RFPAnalysisResult:
        """텍스트 응답에서 정보 추출"""
        requirements = self._extract_requirements(text)
        evaluation_criteria = self._extract_evaluation_criteria(text)
        risk_flags = self._extract_risk_flags(text)
        
        return RFPAnalysisResult(
            requirements=requirements,
            evaluation_criteria=evaluation_criteria,
            risk_flags=risk_flags,
            timeline=self._extract_timeline(text),
            budget_range=self._extract_budget(text),
            submission_format=self._extract_submission_format(text)
        )
    
    def _extract_requirements(self, text: str) -> List[Requirement]:
        """요구사항 추출"""
        requirements = []
        
        # 기술 요구사항
        tech_pattern = r'기술[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[가-힣]+[:\s]|$)'
        tech_matches = re.findall(tech_pattern, text, re.MULTILINE)
        for match in tech_matches:
            requirements.append(Requirement(
                category="기술",
                description=match.strip(),
                priority="필수",
                complexity="중"
            ))
        
        # 운영 요구사항
        ops_pattern = r'운영[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[가-힣]+[:\s]|$)'
        ops_matches = re.findall(ops_pattern, text, re.MULTILINE)
        for match in ops_matches:
            requirements.append(Requirement(
                category="운영",
                description=match.strip(),
                priority="필수",
                complexity="중"
            ))
        
        return requirements[:10]  # 상위 10개만 반환
    
    def _extract_evaluation_criteria(self, text: str) -> List[EvaluationCriteria]:
        """평가기준 추출"""
        criteria = []
        
        # 평가 기준 패턴
        eval_pattern = r'평가[:\s]*([^\n]+)'
        eval_matches = re.findall(eval_pattern, text)
        
        for i, match in enumerate(eval_matches[:5]):
            criteria.append(EvaluationCriteria(
                item=match.strip(),
                weight=0.2,  # 기본 가중치
                min_score=None,
                evaluation_type="정성",
                description=match.strip()
            ))
        
        return criteria
    
    def _extract_risk_flags(self, text: str) -> List[RiskFlag]:
        """리스크 플래그 추출"""
        risk_flags = []
        
        # 리스크 키워드 검색
        risk_keywords = {
            '모순': ['모순', '상충', '충돌'],
            '불명확': ['불명확', '모호', '애매'],
            '법적': ['법적', '규정', '컴플라이언스'],
            '보안': ['보안', '암호화', '인증'],
            '라이선스': ['라이선스', '저작권', '특허']
        }
        
        for risk_type, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    risk_flags.append(RiskFlag(
                        risk_type=risk_type,
                        description=f"{keyword} 관련 이슈 발견",
                        severity="중",
                        mitigation="추가 검토 필요"
                    ))
                    break
        
        return risk_flags
    
    def _extract_timeline(self, text: str) -> Optional[str]:
        """타임라인 추출"""
        timeline_patterns = [
            r'납기[:\s]*([^\n]+)',
            r'일정[:\s]*([^\n]+)',
            r'기간[:\s]*([^\n]+)'
        ]
        
        for pattern in timeline_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_budget(self, text: str) -> Optional[str]:
        """예산 추출"""
        budget_patterns = [
            r'예산[:\s]*([0-9,]+원)',
            r'예산[:\s]*([0-9,]+억원)',
            r'예산[:\s]*([0-9,]+천만원)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_submission_format(self, text: str) -> Optional[str]:
        """제출 양식 추출"""
        format_patterns = [
            r'제출[:\s]*([^\n]+)',
            r'양식[:\s]*([^\n]+)',
            r'서식[:\s]*([^\n]+)'
        ]
        
        for pattern in format_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    def _create_default_result(self) -> RFPAnalysisResult:
        """기본 결과 생성"""
        return RFPAnalysisResult(
            requirements=[],
            evaluation_criteria=[],
            risk_flags=[],
            timeline=None,
            budget_range=None,
            submission_format=None
        )
    
    def get_analysis_summary(self, result: RFPAnalysisResult) -> Dict:
        """분석 결과 요약"""
        return {
            "총_요구사항_수": len(result.requirements),
            "평가기준_수": len(result.evaluation_criteria),
            "리스크_플래그_수": len(result.risk_flags),
            "타임라인": result.timeline,
            "예산_범위": result.budget_range,
            "제출_양식": result.submission_format
        }
