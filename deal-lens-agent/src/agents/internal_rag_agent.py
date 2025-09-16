"""
B. 내부지식 매칭 에이전트 (Internal RAG)
과거 유사 프로젝트, 성과, 고객 피드백, 모듈/솔루션, 인력 스킬 매칭
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
import streamlit as st


@dataclass
class ProjectMatch:
    """프로젝트 매칭 결과"""
    project_id: str
    project_name: str
    similarity_score: float
    matching_requirements: List[str]
    success_factors: List[str]
    lessons_learned: List[str]


@dataclass
class SkillGap:
    """스킬 갭 분석"""
    skill_area: str
    required_level: str
    current_level: str
    gap_size: str  # 충분/부족/심각
    improvement_suggestions: List[str]


@dataclass
class Reference:
    """레퍼런스 정보"""
    project_name: str
    client: str
    success_metrics: Dict[str, str]
    sla_performance: Dict[str, str]
    customer_feedback: str
    applicable_requirements: List[str]


@dataclass
class InternalMatchResult:
    """내부 매칭 결과"""
    project_matches: List[ProjectMatch]
    skill_gaps: List[SkillGap]
    references: List[Reference]
    overall_readiness: str  # 준비도 (상/중/하)
    confidence_score: float


class InternalRAGAgent:
    """내부지식 매칭 에이전트"""
    
    def __init__(self, llm: AzureChatOpenAI, embeddings: AzureOpenAIEmbeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = None
        self._setup_prompts()
        self._initialize_knowledge_base()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.matching_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 내부 지식 매칭 전문가입니다. 
            주어진 요구사항에 대해 다음을 분석해주세요:
            
            1. 과거 유사 프로젝트 매칭
            2. 현재 역량 대비 갭 분석
            3. 인용 가능한 레퍼런스 추천
            4. 전체 준비도 평가
            
            JSON 형태로 구조화된 응답을 제공해주세요."""),
            ("user", "요구사항: {requirements}\n\n과거 프로젝트 데이터: {project_data}")
        ])
    
    def _initialize_knowledge_base(self):
        """지식 베이스 초기화 (실제로는 DB나 파일에서 로드)"""
        # 샘플 프로젝트 데이터
        sample_projects = [
            Document(
                page_content="클라우드 마이그레이션 프로젝트 - AWS 기반 인프라 구축, 6개월 완료, 고객 만족도 95%",
                metadata={"project_id": "P001", "category": "클라우드", "success_rate": 0.95}
            ),
            Document(
                page_content="AI 플랫폼 구축 - 머신러닝 모델 개발, 데이터 파이프라인 구축, 8개월 완료",
                metadata={"project_id": "P002", "category": "AI/ML", "success_rate": 0.88}
            ),
            Document(
                page_content="보안 솔루션 도입 - SIEM 시스템 구축, 보안 정책 수립, 4개월 완료",
                metadata={"project_id": "P003", "category": "보안", "success_rate": 0.92}
            )
        ]
        
        try:
            self.vectorstore = FAISS.from_documents(sample_projects, self.embeddings)
        except Exception as e:
            st.warning(f"벡터 스토어 초기화 실패: {str(e)}")
            self.vectorstore = None
    
    def match_requirements(self, requirements: List[str]) -> InternalMatchResult:
        """요구사항에 대한 내부 매칭 수행"""
        try:
            # 벡터 검색으로 유사 프로젝트 찾기
            project_matches = self._find_similar_projects(requirements)
            
            # 스킬 갭 분석
            skill_gaps = self._analyze_skill_gaps(requirements)
            
            # 레퍼런스 추천
            references = self._recommend_references(requirements)
            
            # 전체 준비도 평가
            overall_readiness, confidence = self._evaluate_readiness(
                project_matches, skill_gaps, references
            )
            
            return InternalMatchResult(
                project_matches=project_matches,
                skill_gaps=skill_gaps,
                references=references,
                overall_readiness=overall_readiness,
                confidence_score=confidence
            )
            
        except Exception as e:
            st.error(f"내부 매칭 중 오류가 발생했습니다: {str(e)}")
            return self._create_default_result()
    
    def _find_similar_projects(self, requirements: List[str]) -> List[ProjectMatch]:
        """유사 프로젝트 검색"""
        if not self.vectorstore:
            return []
        
        try:
            # 요구사항을 하나의 쿼리로 결합
            query = " ".join(requirements)
            
            # 벡터 검색 수행
            docs = self.vectorstore.similarity_search_with_score(query, k=3)
            
            matches = []
            for doc, score in docs:
                matches.append(ProjectMatch(
                    project_id=doc.metadata.get("project_id", "Unknown"),
                    project_name=doc.page_content.split(" - ")[0],
                    similarity_score=1 - score,  # 거리를 유사도로 변환
                    matching_requirements=self._extract_matching_requirements(
                        doc.page_content, requirements
                    ),
                    success_factors=["기술적 우수성", "프로젝트 관리"],
                    lessons_learned=["초기 요구사항 분석의 중요성"]
                ))
            
            return matches
            
        except Exception as e:
            st.warning(f"유사 프로젝트 검색 실패: {str(e)}")
            return []
    
    def _extract_matching_requirements(self, project_content: str, requirements: List[str]) -> List[str]:
        """매칭되는 요구사항 추출"""
        matching = []
        for req in requirements:
            # 간단한 키워드 매칭
            if any(keyword in project_content.lower() for keyword in req.lower().split()):
                matching.append(req)
        return matching
    
    def _analyze_skill_gaps(self, requirements: List[str]) -> List[SkillGap]:
        """스킬 갭 분석"""
        # 실제로는 내부 인력 스킬 DB와 비교
        skill_areas = ["클라우드", "AI/ML", "보안", "데이터베이스", "프론트엔드"]
        
        gaps = []
        for area in skill_areas:
            if any(area.lower() in req.lower() for req in requirements):
                gaps.append(SkillGap(
                    skill_area=area,
                    required_level="상급",
                    current_level="중급",
                    gap_size="부족",
                    improvement_suggestions=[
                        f"{area} 전문가 영입",
                        f"{area} 교육 프로그램 참여",
                        "외부 파트너십 고려"
                    ]
                ))
        
        return gaps
    
    def _recommend_references(self, requirements: List[str]) -> List[Reference]:
        """레퍼런스 추천"""
        references = []
        
        # 샘플 레퍼런스 데이터
        sample_refs = [
            {
                "project_name": "A사 클라우드 마이그레이션",
                "client": "A사",
                "success_metrics": {"완료율": "100%", "고객만족도": "95%"},
                "sla_performance": {"가용성": "99.9%", "응답시간": "2초 이내"},
                "customer_feedback": "프로젝트 관리가 우수하고 기술적 품질이 높음",
                "applicable_requirements": ["클라우드", "마이그레이션"]
            },
            {
                "project_name": "B사 AI 플랫폼 구축",
                "client": "B사",
                "success_metrics": {"정확도": "92%", "처리속도": "10배 향상"},
                "sla_performance": {"모델 정확도": "95%", "응답시간": "1초 이내"},
                "customer_feedback": "혁신적인 솔루션으로 비즈니스 가치 창출",
                "applicable_requirements": ["AI", "머신러닝"]
            }
        ]
        
        for ref_data in sample_refs:
            if any(req.lower() in str(ref_data["applicable_requirements"]).lower() 
                   for req in requirements):
                references.append(Reference(**ref_data))
        
        return references
    
    def _evaluate_readiness(self, project_matches: List[ProjectMatch], 
                          skill_gaps: List[SkillGap], 
                          references: List[Reference]) -> Tuple[str, float]:
        """전체 준비도 평가"""
        # 간단한 점수 계산
        match_score = len(project_matches) * 0.3
        gap_penalty = len([g for g in skill_gaps if g.gap_size == "심각"]) * 0.2
        ref_score = len(references) * 0.2
        
        total_score = max(0, match_score + ref_score - gap_penalty)
        
        if total_score >= 0.8:
            return "상", total_score
        elif total_score >= 0.5:
            return "중", total_score
        else:
            return "하", total_score
    
    def _create_default_result(self) -> InternalMatchResult:
        """기본 결과 생성"""
        return InternalMatchResult(
            project_matches=[],
            skill_gaps=[],
            references=[],
            overall_readiness="하",
            confidence_score=0.0
        )
    
    def get_matching_summary(self, result: InternalMatchResult) -> Dict:
        """매칭 결과 요약"""
        return {
            "유사_프로젝트_수": len(result.project_matches),
            "스킬_갭_수": len(result.skill_gaps),
            "레퍼런스_수": len(result.references),
            "전체_준비도": result.overall_readiness,
            "신뢰도_점수": result.confidence_score
        }
