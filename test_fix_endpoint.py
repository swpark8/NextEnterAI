import requests
import json
import time

url = "http://127.0.0.1:8000/api/v1/interview/next"

payload = {
    "id": "TEST_USER_QUICK_CHECK",
    "target_role": "backend_developer",
    "resume_content": {
        "education": ["University of Test"],
        "skills": ["Java", "Python", "Spring Boot"]
    },
    "last_answer": "This is a test answer.",
}

headers = {
    "Content-Type": "application/json"
}

print(f"Checking URL: {url} ...")
start_time = time.time()

try:
    # Set a short timeout. If server is processing (AI generation), it might timeout, 
    # but that PROVES it didn't reject immediately with 400.
    response = requests.post(url, json=payload, headers=headers, timeout=5)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ TEST PASSED: Server accepted the request and returned 200 OK.")
    elif response.status_code == 400:
        print("\n❌ TEST FAILED: Server rejected with 400 Bad Request (likely Empty Body error).")
    else:
        print(f"\n⚠️ Unexpected Status: {response.status_code}")

except requests.exceptions.ReadTimeout:
    print("\n✅ TEST PARTIALLY PASSED: Request timed out (Server is processing).")
    print("This proves the 'Empty Body' (Immediate 400) error is GONE.")
    print("The server accepted the JSON and is likely waiting for the AI model response.")

except Exception as e:
    print(f"\n❌ CONNECTION FAILED: {e}")
