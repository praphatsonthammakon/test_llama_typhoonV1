from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
import json
import os
import redis.asyncio as redis
import io
import zipfile
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import time

app = FastAPI()

# ใส่ไฟล์ JSON ของ service account credentials
SERVICE_ACCOUNT_FILE = 'bankdriveapi-b3a529b8f632.json'

# ระบุ scope ที่ต้องการใช้
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# File ID ของไฟล์โมเดลใน Google Drive (.zip)
FILE_ID = '1c7e3Wgra4XoGWlKeR5xquVWzk_QG1R3_'  # แทนที่ด้วย File ID ที่ถูกต้อง

# ระบุโฟลเดอร์และชื่อไฟล์ปลายทางที่ต้องการบันทึก
model_dir = r"models/llamatyphoon/llamatyphoon/llamatyphoon"
zip_file_name = "llamatyphoon.zip"  # ไฟล์ .zip ที่ดาวน์โหลด
destination_file = os.path.join(model_dir, zip_file_name)

# สร้าง Google Drive API service
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# ฟังก์ชันสำหรับดาวน์โหลดไฟล์จาก Google Drive แบบ async และแบ่ง chunk
async def download_model_from_gdrive():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ตรวจสอบว่ามีไฟล์อยู่แล้วหรือไม่
    if not os.path.exists(destination_file):
        print("Downloading model .zip from Google Drive...")

        try:
            request = service.files().get_media(fileId=FILE_ID)
            fh = io.FileIO(destination_file, 'wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}% complete.")
            fh.close()
            print(f"File downloaded to {destination_file}")
        except Exception as e:
            print(f"Error downloading file: {e}")
    else:
        print("Model .zip already exists.")

# ฟังก์ชันสำหรับแตกไฟล์ .zip
def extract_zip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # ตรวจสอบว่ามีการแตกไฟล์แล้วหรือไม่
    if not os.path.exists(os.path.join(extract_to, "config.json")):
        print("Extracting model from .zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
        except Exception as e:
            print(f"Error extracting file: {e}")
    else:
        print("Model already extracted.")

# ฟังก์ชันแฮชเพื่อสร้างคีย์ cache สำหรับ Redis
def generate_redis_key(prompt: str, max_length: int):
    key = f"{prompt}-{max_length}"
    return hashlib.md5(key.encode()).hexdigest()

# ฟังก์ชันสำหรับประมวลผลข้อความในพื้นหลัง
async def generate_text_background(prompt: str, max_length: int, redis_key: str):
    if os.path.exists(os.path.join(model_dir, "config.json")):
        print("Config file found, loading model from local directory...")
    else:
        print("Config file not found, please check the path.")
        return {"error": "Config file not found"}

    try:
        # โหลด tokenizer และโมเดลจาก local path
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

        # ตรวจสอบว่าโมเดลรองรับ float16 หรือไม่ ถ้าไม่ให้ใช้ float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)
            print("Model loaded with float16")
        except Exception as e:
            print(f"Failed to load in float16, falling back to float32. Error: {e}")
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
            print("Model loaded with float32")

        # # ตรวจสอบว่า GPU พร้อมใช้งานหรือไม่
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)

        # ตรวจสอบว่า GPU พร้อมใช้งานหรือไม่
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        model.to(device)


        # สร้าง pipeline สำหรับการสร้างข้อความ
        llama_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # เริ่มประมวลผลข้อความ
        print("Generating text...")
        start_time = time.time()
        outputs = llama_pipeline(prompt, max_new_tokens=max_length, num_return_sequences=1)
        end_time = time.time()

        generated_text = outputs[0]["generated_text"]
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        #torch.cuda.empty_cache()

        # แคชผลลัพธ์ใน Redis
        await app.state.redis.set(redis_key, json.dumps({"generated_text": generated_text}), ex=3600)
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return None

# โหลด Redis client
async def get_redis():
    return redis.Redis(host='192.168.127.133', port=6379, db=0)

@app.on_event("startup")
async def startup_event():
    app.state.redis = await get_redis()

    # ตรวจสอบว่าไฟล์โมเดลมีอยู่แล้วหรือไม่
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print("Model not found, starting download and extraction process...")
        await download_model_from_gdrive()
        extract_zip_file(destination_file, model_dir)
    else:
        print("Model files already exist, skipping download.")

    pong = await app.state.redis.ping()
    print(f"Redis ping result: {pong}")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()

# สร้าง class สำหรับรับ body
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50

# Endpoint สำหรับการสร้างข้อความ LLM AI
@app.post("/generate/")
async def generate_text(body: GenerateRequest, background_tasks: BackgroundTasks = None):
    redis_key = generate_redis_key(body.prompt, body.max_length)
    
    cached_result = await app.state.redis.get(redis_key)
    if cached_result:
        return json.loads(cached_result)
    
    background_tasks.add_task(generate_text_background, body.prompt, body.max_length, redis_key)
    return {"message": "Your request is being processed in the background, please check back later."}

# Endpoint ทั่วไปสำหรับ set และ get ค่าใน Redis
@app.get("/set/")
async def set_key(key: str, value: str):
    await app.state.redis.set(key, value)
    return {"message": f"Key '{key}' set to '{value}'"}

@app.get("/get/")
async def get_key(key: str):
    value = await app.state.redis.get(key)
    if value is None:
        return {"error": "Key not found"}
    return {"key": key, "value": value.decode("utf-8")}



