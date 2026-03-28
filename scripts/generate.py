import os
import asyncio
import json
import time
import path
import base64
from datetime import datetime
from typing import List, Dict, Any

import httpx
import cloudinary
import cloudinary.uploader
import smtplib
from email.mime.text import MIMEText
from ollama import AsyncClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================= CONFIG =================
EXTERNAL_API_BASE = os.getenv('EXTERNAL_API_BASE', 'https://cloud-text-manager-server.vercel.app')
EXTERNAL_API_URL = f"{EXTERNAL_API_BASE}/api/all-files"
GENERATED_DIR = os.path.join(os.getcwd(), 'generated-content')

BATCH_SIZE = 15
CONCURRENCY_LIMIT = 3 # Increased slightly for performance
REQUEST_DELAY = 1.0 # Seconds
MAX_RETRIES = 5

os.makedirs(GENERATED_DIR, exist_ok=True)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# ================= OLLAMA CLIENT =================
def get_ollama_client():
    if not OLLAMA_API_KEY:
        raise ValueError("OLLAMA_API_KEY is not defined.")
    return AsyncClient(host="https://ollama.com", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})

def select_model(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ['code', 'python', 'javascript', 'typescript', 'rust']):
        return "qwen3-coder:480b-cloud"
    if any(k in prompt_lower for k in ['reason', 'think', 'solve']):
        return "kimi-k2-thinking"
    return "gpt-oss:120b-cloud"

# ================= EMAIL =================
def send_email(subject: str, text: str):
    email_host = os.getenv('EMAIL_HOST')
    if not email_host:
        return

    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = os.getenv('EMAIL_FROM') or os.getenv('EMAIL_USER')
    msg['To'] = os.getenv('EMAIL_TO') or os.getenv('EMAIL_USER')

    try:
        with smtplib.SMTP(email_host, int(os.getenv('EMAIL_PORT', 587))) as server:
            if os.getenv('EMAIL_SECURE') == 'true':
                server.starttls()
            server.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASS'))
            server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def get_folder_stats(dir_path: str):
    if not os.path.exists(dir_path):
        return {"totalFiles": 0, "mdxFiles": 0}
    files = os.listdir(dir_path)
    mdx_files = [f for f in files if f.endswith('.mdx')]
    return {"totalFiles": len(files), "mdxFiles": len(mdx_files)}

async def send_batch_email(success: List[Dict], failed: List[Dict]):
    # Use BDT timezone or just local
    date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stats = get_folder_stats(GENERATED_DIR)
    
    subject = f"Ollama Cloud Generation Report - {date_time_str} - {len(success)} Success, {len(failed)} Failed"
    
    body = f"""
Ollama Cloud Content Generation Report
--------------------------------------
Date & Time: {date_time_str}
Total Processed: {len(success) + len(failed)}
Success: {len(success)}
Failed: {len(failed)}

Successful Files:
{chr(10).join(f"- {f['name']}" for f in success)}

Failed Files:
{chr(10).join(f"- {f['name']} ({f.get('error', 'Unknown error')})" for f in failed)}

Folder Stats:
- Total files in folder: {stats['totalFiles']}
- Total MDX files: {stats['mdxFiles']}

Generated content has been saved to the repository.
"""
    send_email(subject, body)

# ================= UTILS =================
def generate_safe_filename(original_filename: str) -> str:
    import re
    base_name = os.path.splitext(original_filename)[0]
    safe = base_name.lower()
    safe = re.sub(r'[^\w\s-]', '', safe).strip()
    safe = re.sub(r'[\s]+', '-', safe)
    safe = re.sub(r'-+', '-', safe)
    return f"{safe}.mdx"

# ================= FILE PROCESS =================
async def process_file(file: Dict[str, Any], client: AsyncClient):
    filename = file.get('originalFilename', 'unknown')
    file_id = file.get('_id') or file.get('id')
    print(f"Processing: {filename}")

    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(file['secureUrl'], timeout=30.0)
            resp.raise_for_status()
            prompt_text = resp.text

        model = select_model(prompt_text)
        
        # Generation with retry logic
        attempt = 0
        generated_content = ""
        while attempt < MAX_RETRIES:
            try:
                response = await client.generate(
                    model=model,
                    prompt=prompt_text,
                    options={"num_predict": 4096}
                )
                generated_content = response.get('response', '')
                if generated_content:
                    break
                raise Exception("Empty response")
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    raise e
                await asyncio.sleep(2 ** attempt)

        # Save locally
        safe_name = generate_safe_filename(filename)
        file_path = os.path.join(GENERATED_DIR, safe_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(generated_content)

        # Update external API
        async with httpx.AsyncClient() as http_client:
            await http_client.put(f"{EXTERNAL_API_URL}/{file_id}", json={
                "status": "AlreadyCopy",
                "completedTimestamp": int(time.time() * 1000)
            })

        print(f"✅ Done: {safe_name}")
        return {"success": True, "name": safe_name}

    except Exception as e:
        print(f"❌ Failed: {filename} - {str(e)}")
        return {"success": False, "name": filename, "error": str(e)}

# ================= MAIN =================
async def run():
    print('🚀 Starting Ollama Cloud Batch Processing...')
    
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(f"{EXTERNAL_API_BASE}/api/all-files?section=General")
            files = resp.json()
            pending_files = [f for f in files if f.get('status') == 'Pending'][:BATCH_SIZE]

        if not pending_files:
            print('No pending files.')
            return

        print(f"Found {len(pending_files)} files.")
        
        client = get_ollama_client()
        
        # Simple semaphore for concurrency control
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        async def sem_worker(file):
            async with sem:
                return await process_file(file, client)
        
        tasks = [sem_worker(f) for f in pending_files]
        results = await asyncio.gather(*tasks)

        success = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        await send_batch_email(success, failed)
        
        print(f"🎉 Batch Completed: {len(success)} Succeeded, {len(failed)} Failed")

    except Exception as e:
        print(f'Fatal error: {str(e)}')

if __name__ == "__main__":
    asyncio.run(run())
