# ใช้ base image pytorch/pytorch:latest ที่มี PyTorch ติดตั้งแล้ว
FROM pytorch/pytorch:latest

# ตั้งค่า working directory เป็น /app
WORKDIR /app

# คัดลอกไฟล์ requirements.txt ไปยัง container
COPY requirements.txt .

# ติดตั้ง dependencies ที่ระบุใน requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ service account credentials สำหรับ Google Drive API
COPY bankdriveapi-b3a529b8f632.json /app/

# คัดลอกโค้ดทั้งหมดไปยัง container
COPY . .

# สร้างโฟลเดอร์สำหรับเก็บโมเดล
RUN mkdir -p /app/models

# ระบุ command ที่จะรัน FastAPI โดยใช้ Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
