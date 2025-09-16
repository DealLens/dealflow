"""
RFP 분석 모듈
RFP 문서를 분석하여 주요 내용, 예산, 기술 요구사항 등을 추출합니다.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


@dataclass
class RFPInfo:
    """RFP 정보를 담는 데이터 클래스"""
    title: str
    summary: str
    budget_range: Optional[str]
    technical_requirements: List[str]
    timeline: Optional[str]
    key_requirements: List[str]
    evaluation_criteria: List[str]


class RFPAnalyzer:
    """RFP 문서 분석기"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 RFP(제안요청서) 분석 전문가입니다. 
            주어진 RFP 문서를 분석하여 다음 정보를 추출해주세요:
            
            1. 프로젝트 제목
            2. 프로젝트 요약 (3-4문장)
            3. 예산 범위 (있는 경우)
            4. 기술적 요구사항
            5. 프로젝트 일정/타임라인
            6. 핵심 요구사항
            7. 평가 기준
            
            각 항목을 명확하게 구분하여 JSON 형태로 응답해주세요."""),
            ("user", "다음 RFP 문서를 분석해주세요:\n\n{rfp_content}")
        ])
    
    def analyze_rfp(self, rfp_content: str) -> RFPInfo:
        """RFP 문서를 분석하여 구조화된 정보를 추출합니다."""
        try:
            # LLM을 사용한 분석
            chain = self.analysis_prompt | self.llm
            response = chain.invoke({"rfp_content": rfp_content})
            
            # 응답에서 정보 추출 (실제로는 JSON 파싱 필요)
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            st.error(f"RFP 분석 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_rfp_info()
    
    def _parse_analysis_response(self, response: str) -> RFPInfo:
        """LLM 응답을 파싱하여 RFPInfo 객체를 생성합니다."""
        # 실제 구현에서는 JSON 파싱이나 정규표현식 사용
        # 여기서는 간단한 예시로 구현
        
        return RFPInfo(
            title=self._extract_title(response),
            summary=self._extract_summary(response),
            budget_range=self._extract_budget(response),
            technical_requirements=self._extract_technical_requirements(response),
            timeline=self._extract_timeline(response),
            key_requirements=self._extract_key_requirements(response),
            evaluation_criteria=self._extract_evaluation_criteria(response)
        )
    
    def _extract_title(self, text: str) -> str:
        """제목 추출"""
        # 간단한 정규표현식으로 제목 추출
        title_match = re.search(r'제목[:\s]*([^\n]+)', text)
        return title_match.group(1).strip() if title_match else "RFP 분석 결과"
    
    def _extract_summary(self, text: str) -> str:
        """요약 추출"""
        summary_match = re.search(r'요약[:\s]*([^\n]+(?:\n[^\n]+)*)', text)
        return summary_match.group(1).strip() if summary_match else "RFP 내용 분석 중..."
    
    def _extract_budget(self, text: str) -> Optional[str]:
        """예산 추출"""
        budget_patterns = [
            r'예산[:\s]*([0-9,]+원)',
            r'예산[:\s]*([0-9,]+억원)',
            r'예산[:\s]*([0-9,]+천만원)',
            r'예산[:\s]*([0-9,]+만원)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_technical_requirements(self, text: str) -> List[str]:
        """기술 요구사항 추출"""
        # 기술 관련 키워드 추출
        tech_keywords = re.findall(r'[A-Za-z]+(?:\.js|\.py|\.net|\.java|API|DB|AI|ML)', text)
        return list(set(tech_keywords))[:10]  # 상위 10개만 반환
    
    def _extract_timeline(self, text: str) -> Optional[str]:
        """타임라인 추출"""
        timeline_match = re.search(r'일정[:\s]*([^\n]+)', text)
        return timeline_match.group(1).strip() if timeline_match else None
    
    def _extract_key_requirements(self, text: str) -> List[str]:
        """핵심 요구사항 추출"""
        # 요구사항 관련 문장들 추출
        requirements = re.findall(r'요구사항[:\s]*([^\n]+)', text)
        return requirements[:5]  # 상위 5개만 반환
    
    def _extract_evaluation_criteria(self, text: str) -> List[str]:
        """평가 기준 추출"""
        criteria = re.findall(r'평가[:\s]*([^\n]+)', text)
        return criteria[:5]  # 상위 5개만 반환
    
    def _create_default_rfp_info(self) -> RFPInfo:
        """기본 RFP 정보 생성"""
        return RFPInfo(
            title="RFP 분석 결과",
            summary="RFP 문서 분석을 완료했습니다.",
            budget_range=None,
            technical_requirements=[],
            timeline=None,
            key_requirements=[],
            evaluation_criteria=[]
        )
