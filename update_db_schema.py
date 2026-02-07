import pymysql
import re
import os

# Database Config
DB_HOST = 'localhost'
DB_USER = 'admin'
DB_PASSWORD = '1111'
DB_NAME = 'codequery'

SCHEMA_FILE = '../NextEnterBack/DATABASE_SCHEMA.md'

def parse_markdown_sql(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find sql code blocks
    # Looking for ```sql ... ```
    sql_blocks = re.findall(r'```sql\s+(.*?)```', content, re.DOTALL)
    
    return sql_blocks

def execute_sql_blocks(sql_blocks):
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with conn.cursor() as cursor:
            # 1. Drop Database if exists
            print("Dropping database if exists...")
            cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
            
            # 2. Create Database logic is usually in the first block, but let's just specificially do it or rely on the script
            # The first block in the file is: CREATE DATABASE ... USE codequery;
            # We can just execute the blocks in order.
            
            for i, block in enumerate(sql_blocks):
                # Remove comments and empty lines for cleaner execution if needed, 
                # but pymysql might handle some. 
                # However, pymysql execute() usually expects one statement.
                # We need to split by semicolon.
                
                # A simple split by ; might fail if ; is in strings.
                # But looking at the schema, it seems safe enough for these specific INSERTs and CREATEs 
                # provided we are careful.
                
                # Let's try to execute the whole block if possible using a method that supports multiple statements 
                # or split carefully. 
                # Actually, pymysql doesn't support multiple statements in one execute call by default 
                # unless client_flag=CLIENT.MULTI_STATEMENTS is set.
                
                # Let's clean the block first
                statements = []
                # Naive split by ; at end of line or ; followed by whitespace
                # The schema file is well formatted. 
                # CREATE TABLE ends with ; 
                # INSERT ends with ;
                
                # We can split by `;\n` or just `;`
                raw_statements = block.split(';')
                
                for raw_stmt in raw_statements:
                    stmt = raw_stmt.strip()
                    if stmt:
                        statements.append(stmt)
                        
                for stmt in statements:
                    # Skip empty statements
                    if not stmt:
                        continue
                        
                    # Handle USE statement specially if needed, but we connect without DB initially 
                    # so USE codequery is fine.
                    try:
                        # print(f"Executing: {stmt[:50]}...")
                        cursor.execute(stmt)
                    except Exception as e:
                        print(f"Error executing statement in block {i+1}: {e}")
                        print(f"Statement: {stmt[:100]}...")
                        raise e
            
            conn.commit()
            print("Successfully updated database schema.")
            
    finally:
        conn.close()

if __name__ == '__main__':
    print(f"Reading schema from {SCHEMA_FILE}...")
    if not os.path.exists(SCHEMA_FILE):
        print(f"Error: File {SCHEMA_FILE} not found.")
        exit(1)
        
    blocks = parse_markdown_sql(SCHEMA_FILE)
    print(f"Found {len(blocks)} SQL blocks.")
    
    execute_sql_blocks(blocks)
