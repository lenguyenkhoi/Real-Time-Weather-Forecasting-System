import requests
import csv
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# 1. Load biến môi trường (Cần tạo file .env cùng thư mục chứa: API_KEY=xxxx)
load_dotenv()
API_KEY = "490bb486298e6406592bb0ee4ab9d23d"
CITY = "Ho Chi Minh"
# Lưu ý: Nếu không load được key, hãy kiểm tra lại file .env hoặc điền trực tiếp để test (nhưng không khuyến khích)
if not API_KEY:
    print("Cảnh báo: Chưa tìm thấy API_KEY trong file .env")

URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
CSV_FILE = "dataset/weather_dataset.csv"

# Định nghĩa các cột dữ liệu (Feature) bạn muốn dùng cho Naive Bayes
# Tên cột này sẽ được dùng chung cho cả dữ liệu lịch sử và real-time
FIELDNAMES = ["timestamp", "temp", "humidity", "pressure", "rain", "weather_label"]

print(f"Bắt đầu thu thập dữ liệu thời tiết cho {CITY}...")

while True:
    try:
        response = requests.get(URL)
        data = response.json()

        if response.status_code != 200:
            print(f"Lỗi API ({response.status_code}): {data.get('message')}")
            time.sleep(60)
            continue

        # 2. Trích xuất dữ liệu (Feature Engineering)
        # Xử lý phần mưa: OpenWeatherMap trả về rain['1h'] nếu có mưa, nếu không thì key 'rain' không tồn tại
        rain_1h = 0.0
        if "rain" in data:
            rain_1h = data["rain"].get("1h", 0.0)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Thêm thời gian thực
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "rain": rain_1h,
            "weather_label": data["weather"][0]["main"] # Đây sẽ là Target Label (Nhãn) cho bài toán phân loại
        }

        # 3. Lưu vào CSV
        file_exists = os.path.isfile(CSV_FILE)
        
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader() # Chỉ viết header nếu file chưa tồn tại
            writer.writerow(record)

        print(f"[{record['timestamp']}] Đã lưu: Temp={record['temp']}, Rain={record['rain']}, Label={record['weather_label']}")

        # Chờ 10 phút (600s)
        time.sleep(600)

    except Exception as e:
        print(f"Lỗi Runtime: {e}")
        time.sleep(60)