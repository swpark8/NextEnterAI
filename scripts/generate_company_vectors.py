"""
company_50_pool.json을 기반으로 company_jd_vectors.pkl 파일을 생성하는 스크립트
jhgan/ko-sroberta-multitask 모델로 768차원 벡터를 생성합니다.
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence_transformers 라이브러리가 설치되지 않았습니다.")
    print("pip install sentence-transformers 로 설치해주세요.")
    exit(1)


def create_text_from_company(company):
    """
    기업 정보를 임베딩용 텍스트로 변환
    
    Args:
        company: company_50_pool.json의 기업 객체
    
    Returns:
        str: 임베딩용 텍스트
    """
    parts = []
    
    # 기업명
    if company.get('name'):
        parts.append(company['name'])
    
    # 산업
    if company.get('industry'):
        parts.append(company['industry'])
    
    # 기술 스택
    if company.get('tech_stack'):
        parts.append(" ".join(company['tech_stack']))
    
    # 직무
    if company.get('target_roles'):
        parts.append(" ".join(company['target_roles']))
    
    # 위치 (선택적)
    if company.get('location'):
        parts.append(company['location'])
    
    return " ".join(parts)


def generate_company_vectors():
    """
    company_50_pool.json을 읽어서 company_jd_vectors.pkl 파일을 생성
    """
    # 경로 설정
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "app" / "data"
    input_path = data_dir / "company_50_pool.json"
    output_path = data_dir / "company_jd_vectors.pkl"
    
    # 입력 파일 확인
    if not input_path.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        return False
    
    # 기존 파일 백업
    if output_path.exists():
        print(f"📦 기존 벡터 파일 백업 중: {backup_path}")
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"✅ 백업 완료: {backup_path}")
    
    # JSON 파일 로드
    print(f"📂 기업 데이터 로드 중: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        companies = json.load(f)
    
    print(f"✅ 기업 데이터 로드 완료: {len(companies)}개 기업")
    
    # 모델 로드
    model_name = "jhgan/ko-sroberta-multitask"
    print(f"\n🧠 모델 로드 중: {model_name}")
    print("   (첫 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다...)")
    model = SentenceTransformer(model_name)
    print(f"✅ 모델 로드 완료 (출력 차원: {model.get_sentence_embedding_dimension()}차원)")
    
    # 텍스트 생성 및 임베딩
    print(f"\n🔄 벡터 생성 중... ({len(companies)}개 기업)")
    texts = []
    for i, company in enumerate(companies):
        text = create_text_from_company(company)
        texts.append(text)
        
        if (i + 1) % 10 == 0:
            print(f"   진행 중: {i + 1}/{len(companies)}")
    
    # 배치 임베딩 생성
    print(f"\n🚀 임베딩 생성 중...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # 결과 확인
    print(f"\n✅ 임베딩 생성 완료")
    print(f"   벡터 shape: {vectors.shape}")
    print(f"   예상 차원: 768차원")
    
    # pickle 파일로 저장
    print(f"\n💾 벡터 파일 저장 중: {output_path}")
    output_data = {
        'companies': companies,
        'vectors': vectors
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"✅ 저장 완료!")
    
    # 검증
    print(f"\n🔍 검증 중...")
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"   저장된 기업 수: {len(loaded_data['companies'])}")
    print(f"   저장된 벡터 shape: {loaded_data['vectors'].shape}")
    print(f"   데이터 타입: {loaded_data['vectors'].dtype}")
    print(f"   메모리 사용량: {loaded_data['vectors'].nbytes / 1024 / 1024:.2f} MB")
    
    # 구조 검증
    if not isinstance(loaded_data, dict):
        print(f"❌ 오류: 저장된 데이터가 dict 형식이 아닙니다.")
        return False
    
    if 'companies' not in loaded_data or 'vectors' not in loaded_data:
        print(f"❌ 오류: 저장된 데이터에 'companies' 또는 'vectors' 키가 없습니다.")
        return False
    
    if len(loaded_data['companies']) != loaded_data['vectors'].shape[0]:
        print(f"❌ 오류: 기업 수({len(loaded_data['companies'])})와 벡터 수({loaded_data['vectors'].shape[0]})가 일치하지 않습니다.")
        return False
    
    if loaded_data['vectors'].shape[1] != 768:
        print(f"⚠️  경고: 예상과 다른 차원입니다. ({loaded_data['vectors'].shape[1]}차원, 예상: 768차원)")
        return False
    
    print(f"✅ 768차원 벡터가 정상적으로 생성되었습니다!")
    print(f"✅ 기업 수: {len(loaded_data['companies'])}개")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("company_50_pool.json → company_jd_vectors.pkl 생성 스크립트")
    print("=" * 60)
    print()
    
    success = generate_company_vectors()
    
    print()
    print("=" * 60)
    if success:
        print("✅ 벡터 파일 생성이 완료되었습니다!")
        print("✅ AI 서버를 재시작하면 새로운 벡터 파일이 적용됩니다.")
    else:
        print("❌ 벡터 파일 생성 중 오류가 발생했습니다.")
    print("=" * 60)
