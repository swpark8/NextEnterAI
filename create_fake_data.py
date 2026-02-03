import mysql.connector
from faker import Faker
import random
from datetime import datetime, timedelta
import json

# DB 설정
db_config = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1111',
    'database': 'codequery'
}

fake = Faker('ko_KR')

def create_connection():
    return mysql.connector.connect(**db_config)

def generate_users(cursor, count=10):
    user_ids = []
    print(f"Generating {count} users...")
    
    sql = """
    INSERT INTO users (email, password, name, phone, age, gender, address, detail_address, is_active, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    for _ in range(count):
        email = fake.unique.email()
        password = "password1234" # 암호화 안 된 상태 (테스트용)
        name = fake.name()
        phone = fake.phone_number()
        age = random.randint(20, 40)
        gender = random.choice(['MALE', 'FEMALE'])
        address = fake.address()
        detail_address = fake.building_number()
        now = datetime.now()
        
        cursor.execute(sql, (email, password, name, phone, age, gender, address, detail_address, True, now, now))
        user_ids.append(cursor.lastrowid)
        
    print("Users generated.")
    return user_ids

def generate_resumes(cursor, user_ids):
    resume_ids = []
    print("Generating resumes...")
    
    sql = """
    INSERT INTO resume (
        user_id, title, resume_name, resume_email, resume_phone, 
        job_category, skills, is_main, visibility, view_count, status, 
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    job_categories = ['Backend', 'Frontend', 'Fullstack', 'PM', 'UI/UX Designer', 'AI/LLM Engineer']
    skills_pool = ['Java', 'Python', 'Spring Boot', 'React', 'MySQL', 'AWS', 'Docker', 'Kubernetes', 'Figma', 'TensorFlow', 'PyTorch']
    
    for user_id in user_ids:
        # 각 유저당 1~2개 이력서 생성
        for i in range(random.randint(1, 2)):
            title = f"{fake.name()}의 {random.choice(['열정적인', '성실한', '준비된'])} {random.choice(job_categories)} 이력서"
            resume_name = fake.name()
            resume_email = fake.email()
            resume_phone = fake.phone_number()
            job_category = random.choice(job_categories)
            skills = ", ".join(random.sample(skills_pool, k=random.randint(3, 5)))
            is_main = (i == 0) # 첫 번째 이력서를 메인으로
            now = datetime.now()
            
            cursor.execute(sql, (
                user_id, title, resume_name, resume_email, resume_phone,
                job_category, skills, is_main, 'PUBLIC', 0, 'COMPLETED', now, now
            ))
            resume_ids.append(cursor.lastrowid)
            
    print(f"Generated {len(resume_ids)} resumes.")
    return resume_ids

def generate_interviews(cursor, user_ids, resume_ids):
    print("Generating interviews...")
    
    # 사용할 직무 목록
    job_categories = ['Backend', 'Frontend', 'Fullstack', 'PM', 'UI/UX Designer', 'AI/LLM Engineer']

    sql_interview = """
    INSERT INTO interview (
        user_id, resume_id, job_category, difficulty, total_turns, current_turn, 
        status, final_score, final_feedback, created_at, completed_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    for user_id in user_ids:
        for _ in range(random.randint(1, 3)):
            # 랜덤 이력서 매핑 (임시로 None 처리)
            resume_id = None 
            
            job_category = random.choice(job_categories)
            difficulty = random.choice(['JUNIOR', 'SENIOR'])
            status = random.choice(['IN_PROGRESS', 'COMPLETED', 'CANCELLED'])
            
            created_at = datetime.now() - timedelta(days=random.randint(0, 30))
            completed_at = None
            final_score = None
            final_feedback = None
            
            if status == 'COMPLETED':
                completed_at = created_at + timedelta(minutes=30)
                final_score = random.randint(60, 100)
                final_feedback = fake.text()
                current_turn = 5
            elif status == 'IN_PROGRESS':
                current_turn = random.randint(0, 4)
            else: # CANCELLED
                completed_at = created_at + timedelta(minutes=10)
                current_turn = random.randint(0, 2)
                
            cursor.execute(sql_interview, (
                user_id, resume_id, job_category, difficulty, 5, current_turn,
                status, final_score, final_feedback, created_at, completed_at
            ))
            
    print("Interviews generated.")

def fetch_existing_users(cursor):
    cursor.execute("SELECT user_id FROM users")
    users = cursor.fetchall()
    return [u[0] for u in users]

def main():
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # 1. 기존 유저 조회
        user_ids = fetch_existing_users(cursor)
        
        if not user_ids:
            print("No users found in the database. Please login via frontend first.")
            return

        print(f"Found {len(user_ids)} existing users: {user_ids}")

        # 2. 이력서 및 인터뷰 생성 (기존 유저 대상)
        resume_ids = generate_resumes(cursor, user_ids)
        generate_interviews(cursor, user_ids, resume_ids)
        
        conn.commit()
        print("Data generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
