"""
C. 경쟁사 분석 에이전트 (Competitor Intelligence)
경쟁사 포트폴리오, 수주 이력, 가격 포지셔닝, 기술 스택, 파트너십 수집
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
import streamlit as st


@dataclass
class CompetitorProfile:
    """경쟁사 프로필"""
    company_name: str
    portfolio: List[str]
    win_history: List[Dict[str, str]]
    price_positioning: str  # 프리미엄/표준/저가
    tech_stack: List[str]
    partnerships: List[str]
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]


@dataclass
class CompetitiveAnalysis:
    """경쟁 분석 결과"""
    competitor_profiles: List[CompetitorProfile]
    our_advantages: List[str]
    our_disadvantages: List[str]
    differentiation_points: List[str]
    market_positioning: Dict[str, str]


class CompetitorIntelligenceAgent:
    """경쟁사 분석 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_prompts()
        self._initialize_competitor_database()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 경쟁사 분석 전문가입니다. 
            주어진 경쟁사 목록에 대해 다음을 분석해주세요:
            
            1. 각 경쟁사의 포트폴리오 및 강점
            2. 수주 이력 및 성과
            3. 가격 포지셔닝
            4. 기술 스택 및 파트너십
            5. SWOT 분석
            6. 우리 회사 대비 차별화 포인트
            
            JSON 형태로 구조화된 응답을 제공해주세요."""),
            ("user", "경쟁사 목록: {competitors}\n\n프로젝트 요구사항: {requirements}")
        ])
    
    def _initialize_competitor_database(self):
        """경쟁사 데이터베이스 초기화"""
        self.competitor_data = {
            "삼성 SDS": {
                "portfolio": ["SI", "클라우드", "AI", "보안", "디지털 트랜스포메이션"],
                "price_positioning": "프리미엄",
                "tech_stack": ["Samsung Cloud", "AI Platform", "Security Solutions"],
                "partnerships": ["Microsoft", "AWS", "Google Cloud"],
                "strengths": ["대기업 고객 기반", "종합 IT 서비스", "R&D 투자"],
                "weaknesses": ["높은 가격", "복잡한 의사결정 구조"],
                "win_rate": 0.75
            },
            "LG CNS": {
                "portfolio": ["SI", "클라우드", "AI", "IoT", "스마트팩토리"],
                "price_positioning": "표준",
                "tech_stack": ["LG Cloud", "AI Platform", "IoT Solutions"],
                "partnerships": ["LG Electronics", "Microsoft", "SAP"],
                "strengths": ["제조업 전문성", "IoT 솔루션", "글로벌 네트워크"],
                "weaknesses": ["제한된 스타트업 경험", "높은 의존도"],
                "win_rate": 0.68
            },
            "포스코DX": {
                "portfolio": ["제조업 SI", "스마트팩토리", "AI", "데이터 분석"],
                "price_positioning": "표준",
                "tech_stack": ["POSCO AI", "Smart Factory Platform", "Data Analytics"],
                "partnerships": ["POSCO", "Microsoft", "Siemens"],
                "strengths": ["제조업 전문성", "스마트팩토리", "데이터 분석"],
                "weaknesses": ["제조업 한정", "신규 기술 부족"],
                "win_rate": 0.72
            },
            "KT": {
                "portfolio": ["통신", "클라우드", "AI", "IoT", "5G"],
                "price_positioning": "표준",
                "tech_stack": ["KT Cloud", "AI Platform", "5G Network"],
                "partnerships": ["KT&G", "Samsung", "LG"],
                "strengths": ["통신 인프라", "5G 기술", "클라우드 서비스"],
                "weaknesses": ["통신업 중심", "SI 경험 부족"],
                "win_rate": 0.65
            },
            "현대 오토에버": {
                "portfolio": ["자동차 SI", "AI", "자율주행", "IoT"],
                "price_positioning": "프리미엄",
                "tech_stack": ["Hyundai AI", "Autonomous Driving", "IoT Platform"],
                "partnerships": ["Hyundai Motor", "Kia", "Aptiv"],
                "strengths": ["자동차 전문성", "자율주행 기술", "글로벌 네트워크"],
                "weaknesses": ["자동차업 한정", "높은 가격"],
                "win_rate": 0.78
            },
            "카카오": {
                "portfolio": ["플랫폼", "AI", "클라우드", "모바일", "콘텐츠"],
                "price_positioning": "표준",
                "tech_stack": ["Kakao Cloud", "AI Platform", "KakaoTalk API"],
                "partnerships": ["KakaoBank", "KakaoPay", "KakaoMobility"],
                "strengths": ["플랫폼 경험", "AI 기술", "사용자 친화적"],
                "weaknesses": ["B2B 경험 부족", "대기업 고객 부족"],
                "win_rate": 0.62
            },
            "CJ 올리브네트웍스": {
                "portfolio": ["SI", "클라우드", "AI", "데이터 분석", "마케팅"],
                "price_positioning": "표준",
                "tech_stack": ["CJ Cloud", "AI Platform", "Marketing Analytics"],
                "partnerships": ["CJ Group", "Microsoft", "Google"],
                "strengths": ["마케팅 전문성", "데이터 분석", "CJ 그룹 지원"],
                "weaknesses": ["제한된 기술력", "높은 의존도"],
                "win_rate": 0.58
            }
        }
    
    def analyze_competitors(self, competitors: List[str], requirements: List[str]) -> CompetitiveAnalysis:
        """경쟁사 분석 수행"""
        try:
            # 경쟁사 프로필 생성
            competitor_profiles = []
            for competitor in competitors:
                if competitor in self.competitor_data:
                    profile = self._create_competitor_profile(competitor, requirements)
                    competitor_profiles.append(profile)
            
            # 경쟁 분석 수행
            our_advantages, our_disadvantages = self._analyze_competitive_position(
                competitor_profiles, requirements
            )
            
            differentiation_points = self._identify_differentiation_points(
                competitor_profiles, requirements
            )
            
            market_positioning = self._analyze_market_positioning(competitor_profiles)
            
            return CompetitiveAnalysis(
                competitor_profiles=competitor_profiles,
                our_advantages=our_advantages,
                our_disadvantages=our_disadvantages,
                differentiation_points=differentiation_points,
                market_positioning=market_positioning
            )
            
        except Exception as e:
            st.error(f"경쟁사 분석 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_analysis()
    
    def _create_competitor_profile(self, competitor: str, requirements: List[str]) -> CompetitorProfile:
        """경쟁사 프로필 생성"""
        data = self.competitor_data[competitor]
        
        # 요구사항과 매칭되는 포트폴리오 필터링
        relevant_portfolio = [
            item for item in data["portfolio"] 
            if any(req.lower() in item.lower() for req in requirements)
        ]
        
        # SWOT 분석
        swot = self._perform_swot_analysis(competitor, data, requirements)
        
        return CompetitorProfile(
            company_name=competitor,
            portfolio=relevant_portfolio,
            win_history=self._get_win_history(competitor),
            price_positioning=data["price_positioning"],
            tech_stack=data["tech_stack"],
            partnerships=data["partnerships"],
            strengths=swot["strengths"],
            weaknesses=swot["weaknesses"],
            opportunities=swot["opportunities"],
            threats=swot["threats"]
        )
    
    def _perform_swot_analysis(self, competitor: str, data: Dict, requirements: List[str]) -> Dict:
        """SWOT 분석 수행"""
        strengths = data["strengths"]
        weaknesses = data["weaknesses"]
        
        # 기회와 위협은 요구사항에 따라 동적으로 생성
        opportunities = [
            f"{req} 관련 시장 확대" for req in requirements[:3]
        ]
        
        threats = [
            "신규 경쟁사 진입",
            "기술 변화 속도",
            "가격 경쟁 심화"
        ]
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats
        }
    
    def _get_win_history(self, competitor: str) -> List[Dict[str, str]]:
        """수주 이력 조회"""
        # 실제로는 데이터베이스에서 조회
        sample_history = [
            {"project": "A사 클라우드 구축", "year": "2023", "value": "50억원"},
            {"project": "B사 AI 플랫폼", "year": "2023", "value": "30억원"},
            {"project": "C사 디지털 전환", "year": "2022", "value": "80억원"}
        ]
        return sample_history
    
    def _analyze_competitive_position(self, profiles: List[CompetitorProfile], 
                                    requirements: List[str]) -> Tuple[List[str], List[str]]:
        """경쟁적 위치 분석"""
        our_advantages = [
            "빠른 의사결정",
            "유연한 프로젝트 관리",
            "혁신적인 기술 접근",
            "경쟁력 있는 가격"
        ]
        
        our_disadvantages = [
            "대기업 고객 기반 부족",
            "글로벌 네트워크 제한",
            "R&D 투자 규모",
            "브랜드 인지도"
        ]
        
        return our_advantages, our_disadvantages
    
    def _identify_differentiation_points(self, profiles: List[CompetitorProfile], 
                                       requirements: List[str]) -> List[str]:
        """차별화 포인트 식별"""
        differentiation_points = [
            "고객 맞춤형 솔루션",
            "빠른 프로토타이핑",
            "애자일 개발 방법론",
            "투명한 커뮤니케이션",
            "경쟁력 있는 가격"
        ]
        
        return differentiation_points
    
    def _analyze_market_positioning(self, profiles: List[CompetitorProfile]) -> Dict[str, str]:
        """시장 포지셔닝 분석"""
        positioning = {}
        for profile in profiles:
            positioning[profile.company_name] = profile.price_positioning
        
        return positioning
    
    def _create_default_analysis(self) -> CompetitiveAnalysis:
        """기본 분석 결과 생성"""
        return CompetitiveAnalysis(
            competitor_profiles=[],
            our_advantages=[],
            our_disadvantages=[],
            differentiation_points=[],
            market_positioning={}
        )
    
    def get_analysis_summary(self, analysis: CompetitiveAnalysis) -> Dict:
        """분석 결과 요약"""
        return {
            "분석_경쟁사_수": len(analysis.competitor_profiles),
            "우리_강점_수": len(analysis.our_advantages),
            "우리_약점_수": len(analysis.our_disadvantages),
            "차별화_포인트_수": len(analysis.differentiation_points),
            "시장_포지셔닝": analysis.market_positioning
        }
