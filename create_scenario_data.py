import mysql.connector
import json
from datetime import datetime

# DB 설정
db_config = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1111',
    'database': 'codequery'
}

def create_connection():
    return mysql.connector.connect(**db_config)

def generate_scenario_data():
    conn = create_connection()
    cursor = conn.cursor()
    
    target_user_id = 2
    
    # 1. Clear existing resumes for user 2
    print(f"Clearing resumes for user_id={target_user_id}...")
    cursor.execute("DELETE FROM resume WHERE user_id = %s", (target_user_id,))
    
    # 2. Define the 6 scenarios
    # S: AI/LLM, A: Frontend, B: Fullstack, B: UI/UX, C: PM, F: Backend
    
    scenarios = [
        {
            "role": "AI/LLM Engineer",
            "grade": "S",
            "title": "LLM 아키텍처 및 최적화 전문가 박상원입니다",
            "name": "박상원",
            "is_main": True,
            "skills": "Python, PyTorch, TensorFlow, Transformers, LLM, LangChain, CUDA, MLOps, AWS, RAG, HuggingFace, vLLM, Vector DB (Milvus), Fine-tuning (LoRA/QLoRA), Prompt Engineering, Kubernetes, Docker, FastAPI",
            "careers": [  # Changed from experiences to careers
                {
                    "company": "NeuroTech Solutions (Series B)", # SME Experience
                    "position": "Lead AI Research Engineer",
                    "role": "AI 모델 연구 및 개발 리딩\n\n[주요 업무]\n- 기업 특화 LLM (sLLM) 구축 및 서비스 상용화 (H100 클러스터 운영)\n- RAG 파이프라인 최적화로 검색 정확도 95% 달성\n- Llama3 기반 한국어 파인튜닝 (LoRA) 및 모델 경량화 (Quantization)\n- Inference 서버 최적화 (vLLM 도입, 동시 접속자 10,000명 처리)",
                    "period": "2020.03 ~ 현재 (4년)"
                }
            ],
            "experiences": [
                {
                    "title": "[사이드 프로젝트] Legal Tech 판례 검색 RAG 서비스 개발 (LangChain, Pinecone)",
                    "period": "2023.01 - 2023.06"
                },
                {
                    "title": "[해커톤 대상] On-device AI 음성 비서 프로토타입 구현 (Whisper.cpp, Android)",
                    "period": "2022.08 - 2022.09"
                },
                {
                    "title": "[오픈소스 기여] HuggingFace Transformers 라이브러리 PR (New Scheduler Added)",
                    "period": "2021.11 - 2021.12"
                }
            ],
            "educations": [
                {
                    "school": "서울대학교",
                    "major": "컴퓨터공학부 인공지능 전공 (박사)",
                    "degree": "Doctor",
                    "period": "2012.03 ~ 2016.02"
                }
            ]
        },
        {
            "role": "Frontend",
            "grade": "A",
            "title": "사용자 경험을 최우선으로 하는 5년차 프론트엔드 개발자",
            "name": "김철수",
            "is_main": False,
            "skills": "JavaScript, TypeScript, React, Next.js, Redux, TailwindCSS, Webpack, Jest",
            "careers": [
                {
                    "company": "Toss Payments",
                    "position": "Senior Frontend Developer",
                    "role": "결제 시스템 프론트엔드 개발\n\n[주요 업무]\n- 공통 컴포넌트 라이브러리 구축 및 사내 배포\n- 결제 SDK 성능 최적화 (로딩 속도 50% 단축)\n- 마이크로 프론트엔드 아키텍처 도입",
                    "period": "2020.05 ~ 현재"
                },
                {
                    "company": "Woowa Bros",
                    "position": "Frontend Developer",
                    "role": "배달의민족 웹 서비스 개발\n\n[주요 업무]\n- 사장님 광장 페이지 개편\n- Legacy 코드 리팩토링 및 React 마이그레이션",
                    "period": "2018.06 ~ 2020.04"
                }
            ],
            "experiences": [],
            "educations": [
                {
                    "school": "연세대학교",
                    "major": "소프트웨어학과",
                    "degree": "Bachelor",
                    "period": "2014.03 ~ 2018.02"
                }
            ]
        },
        {
            "role": "Fullstack",
            "grade": "B",
            "title": "넓은 스펙트럼을 가진 풀스택 개발자 이영희",
            "name": "이영희",
            "is_main": False,
            "skills": "Java, Spring Boot, JPA, JavaScript, Vue.js, MySQL, Docker",
            "careers": [
                {
                    "company": "Startup A",
                    "position": "Fullstack Developer",
                    "role": "사내 어드민 및 서비스 개발\n\n- REST API 설계 및 구현\n- Vue.js 기반 프론트엔드 개발\n- AWS EC2 배포 및 운영",
                    "period": "2021.01 ~ 현재"
                }
            ],
            "experiences": [],
            "educations": [
                {
                    "school": "고려대학교",
                    "major": "컴퓨터학과",
                    "degree": "Bachelor",
                    "period": "2016.03 ~ 2021.02"
                }
            ]
        },
        {
            "role": "UI/UX Designer",
            "grade": "B",
            "title": "데이터 기반의 UI/UX 디자이너 정민수",
            "name": "정민수",
            "is_main": False,
            "skills": "Figma, Sketch, Adobe XD, Photoshop, Illustrator, Zeplin, User Research",
            "careers": [
                {
                    "company": "Design Agency B",
                    "position": "UI/UX Designer",
                    "role": "모바일/웹 디자인\n\n- 금융 앱 UI/UX 리뉴얼\n- 사용자 리서치 및 페르소나 정의\n- 프로토타이핑 및 디자인 시스템 구축",
                    "period": "2022.03 ~ 현재"
                }
            ],
            "experiences": [],
            "educations": [
                {
                    "school": "홍익대학교",
                    "major": "시각디자인과",
                    "degree": "Bachelor",
                    "period": "2017.03 ~ 2022.02"
                }
            ]
        },
        {
            "role": "PM",
            "grade": "C",
            "title": "소통을 중요시하는 주니어 PM 최지우",
            "name": "최지우",
            "is_main": False,
            "skills": "Jira, Confluence, Slack, Notion, Communication, Documentation",
            "careers": [
                {
                    "company": "IT Service C",
                    "position": "Junior PM",
                    "role": "서비스 기획 및 관리\n\n- 회의록 작성 및 일정 관리\n- 스토리보드(SB) 현행화\n- QA 진행 및 버그 리포팅",
                    "period": "2023.01 ~ 현재 (6개월)"
                }
            ],
            "experiences": [],
            "educations": [
                {
                    "school": "서강대학교",
                    "major": "경영학과",
                    "degree": "Bachelor",
                    "period": "2018.03 ~ 2023.02"
                }
            ]
        },
        {
            "role": "Backend",
            "grade": "F",
            "title": "열정만은 가득한 신입 백엔드 지망생",
            "name": "한지훈",
            "is_main": False,
            "skills": "HTML, CSS, Basic Python",
            "careers": [],
            "experiences": [],
            "educations": [
                {
                    "school": "비전공 (학점은행제)",
                    "major": "경영학",
                    "degree": "Bachelor",
                    "period": "2020.03 ~ 2024.02"
                }
            ]
        },
        {
            "role": "Fullstack",
            "grade": "Junior",
            "title": "성장을 갈망하는 2년차 주니어 풀스택 개발자",
            "name": "강진수",
            "is_main": False,
            "skills": "Java, Spring, Python, React, JavaScript, TypeScript, MySQL, Tailwind, Zustand",
            "careers": [
                {
                    "company": "(주)스타트업나우",
                    "position": "사원",
                    "role": "풀스택 개발 및 유지보수\n- 사내 시스템 관리 도구 개발\n- 기존 레거시 코드 수정 및 버그 대응",
                    "period": "2022.01 ~ 2023.12 (2년)"
                }
            ],
            "experiences": [
                {
                    "title": "[개인 프로젝트] 간단한 의류 쇼핑몰 (기능: 장바구니, 게시판)",
                    "period": "2021.06 - 2021.08"
                },
                {
                    "title": "[토이 프로젝트] 넷플릭스 UI 클론 코딩 (상세 페이지 구현)",
                    "period": "2021.03 - 2021.04"
                }
            ],
            "educations": [
                {
                    "school": "대한과학기술대학교",
                    "major": "정보통신공학과",
                    "degree": "Bachelor",
                    "period": "2016.03 ~ 2022.02"
                }
            ]
        }
    ]
    
    # experiences, careers, certificates 각각 별도 컬럼 매핑 필요
    sql_insert = """
    INSERT INTO resume (
        user_id, title, resume_name, resume_email, resume_phone, 
        job_category, skills, experiences, careers, educations, 
        is_main, visibility, view_count, status, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PUBLIC', 0, 'COMPLETED', NOW(), NOW())
    """
    
    for sc in scenarios:
        print(f"Creating {sc['grade']} grade resume for {sc['role']}...")
        
        experiences_json = json.dumps(sc.get('experiences', []), ensure_ascii=False)
        careers_json = json.dumps(sc.get('careers', []), ensure_ascii=False) # New
        educations_json = json.dumps(sc.get('educations', []), ensure_ascii=False)
        
        cursor.execute(sql_insert, (
            target_user_id,
            sc['title'],
            sc['name'],
            f"test_{sc['grade'].lower()}@example.com",
            f"010-{sc['grade']}-0000",
            sc['role'],
            sc['skills'],
            experiences_json,
            careers_json, # Insert into careers column
            educations_json,
            sc['is_main']
        ))
        
    conn.commit()
    conn.close()
    print("Scenario generation done!")

if __name__ == "__main__":
    generate_scenario_data()
