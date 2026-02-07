import pymysql

# Database Config
DB_HOST = 'localhost'
DB_USER = 'admin'
DB_PASSWORD = '1111'
DB_NAME = 'codequery'

def verify_db():
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        charset='utf8mb4',
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"Found {len(tables)} tables:")
            for table in tables:
                print(f"- {list(table.values())[0]}")
                
            # Check user count
            try:
                cursor.execute("SELECT COUNT(*) as count FROM user")
                user_count = cursor.fetchone()['count']
                print(f"User count: {user_count}")
            except:
                print("Could not query user table")

            # Check company count
            try:
                cursor.execute("SELECT COUNT(*) as count FROM company")
                company_count = cursor.fetchone()['count']
                print(f"Company count: {company_count}")
            except:
                print("Could not query company table")
                
    finally:
        conn.close()

if __name__ == '__main__':
    verify_db()
