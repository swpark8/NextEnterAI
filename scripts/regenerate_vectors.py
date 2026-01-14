"""
벡터 데이터 재생성 스크립트
final_metadata.json을 사용하여 jhgan/ko-sroberta-multitask 모델로 768차원 벡터를 재생성합니다.
"""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence_transformers 라이브러리가 설치되지 않았습니다.")
    print("pip install sentence-transformers 로 설치해주세요.")
    exit(1)

def create_text_from_metadata(metadata_item):
    """
    메타데이터 항목을 임베딩용 텍스트로 변환
    """
    job_title = metadata_item.get('job_title', '')
    tech_stack = metadata_item.get('tech_stack', [])
    req_skills = metadata_item.get('req_skills', [])
    industry = metadata_item.get('industry', '')
    
    # 텍스트 조합: job_title + tech_stack + req_skills
    parts = [job_title]
    
    if tech_stack:
        parts.append(" ".join(tech_stack))
    
    if req_skills:
        parts.append(" ".join(req_skills))
    
    if industry:
        parts.append(industry)
    
    return " ".join(parts)

def regenerate_vectors():
    """
    final_metadata.json을 사용하여 벡터를 재생성
    """
    # 경로 설정
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    metadata_path = data_dir / "final_metadata.json"
    output_path = data_dir / "final_embedded_dataset.npy"
    backup_path = data_dir / f"final_embedded_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    
    # 메타데이터 파일 확인
    if not metadata_path.exists():
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        return False
    
    # 기존 벡터 파일 백업
    if output_path.exists():
        print(f"📦 기존 벡터 파일 백업 중: {backup_path}")
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"✅ 백업 완료: {backup_path}")
    
    # 메타데이터 로드
    print(f"📂 메타데이터 로드 중: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✅ 메타데이터 로드 완료: {len(metadata)}개 항목")
    
    # 모델 로드
    model_name = "jhgan/ko-sroberta-multitask"
    print(f"🧠 모델 로드 중: {model_name}")
    print("   (첫 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다...)")
    model = SentenceTransformer(model_name)
    print(f"✅ 모델 로드 완료 (출력 차원: {model.get_sentence_embedding_dimension()}차원)")
    
    # 텍스트 생성 및 임베딩
    print(f"\n🔄 벡터 생성 중... ({len(metadata)}개 항목)")
    texts = []
    for i, item in enumerate(metadata):
        text = create_text_from_metadata(item)
        texts.append(text)
        
        if (i + 1) % 100 == 0:
            print(f"   진행 중: {i + 1}/{len(metadata)}")
    
    # 배치 임베딩 생성
    print(f"\n🚀 임베딩 생성 중...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # 결과 확인
    print(f"\n✅ 임베딩 생성 완료")
    print(f"   벡터 shape: {vectors.shape}")
    print(f"   예상 차원: 768차원")
    
    # 저장
    print(f"\n💾 벡터 저장 중: {output_path}")
    np.save(output_path, vectors)
    print(f"✅ 저장 완료!")
    
    # 검증
    loaded = np.load(output_path)
    print(f"\n🔍 검증:")
    print(f"   저장된 벡터 shape: {loaded.shape}")
    print(f"   데이터 타입: {loaded.dtype}")
    print(f"   메모리 사용량: {loaded.nbytes / 1024 / 1024:.2f} MB")
    
    if loaded.shape[1] == 768:
        print(f"✅ 768차원 벡터가 정상적으로 생성되었습니다!")
        return True
    else:
        print(f"⚠️  경고: 예상과 다른 차원입니다. ({loaded.shape[1]}차원)")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("벡터 데이터 재생성 스크립트")
    print("=" * 60)
    print()
    
    success = regenerate_vectors()
    
    print()
    print("=" * 60)
    if success:
        print("✅ 벡터 재생성이 완료되었습니다!")
    else:
        print("❌ 벡터 재생성 중 오류가 발생했습니다.")
    print("=" * 60)
