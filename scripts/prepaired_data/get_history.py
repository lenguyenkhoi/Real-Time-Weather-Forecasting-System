import requests
import pandas as pd

# Tọa độ TP.HCM
LAT = 10.8231
LON = 106.6297

# Lấy dữ liệu 1 năm qua (2023 đến nay)
start_date = "2024-01-01"
end_date = "2025-11-30"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,rain,weather_code"
}

print("Đang tải dữ liệu lịch sử từ Open-Meteo...")
response = requests.get(url, params=params)
data = response.json()

hourly = data['hourly']

# Tạo DataFrame
df = pd.DataFrame({
    'timestamp': hourly['time'],
    'temp': hourly['temperature_2m'],          # Đổi tên cho khớp với code Real-time
    'humidity': hourly['relative_humidity_2m'],# Đổi tên cho khớp
    'pressure': hourly['surface_pressure'],    # Đổi tên cho khớp
    'rain': hourly['rain'],                    # Đổi tên cho khớp
    'weather_code': hourly['weather_code']     # Cần chuyển đổi code này sang label (text)
})

# Hàm chuyển đổi WMO Code của Open-Meteo sang Label (Text) tương tự OpenWeatherMap
# Tham khảo: https://open-meteo.com/en/docs
def map_weather_code(code):
    if code == 0: return "Clear"
    if 1 <= code <= 3: return "Clouds"
    if 51 <= code <= 67: return "Rain"
    if 80 <= code <= 82: return "Rain"
    if 95 <= code <= 99: return "Thunderstorm"
    return "Others"

df['weather_label'] = df['weather_code'].apply(map_weather_code)

# Bỏ cột weather_code cũ, chỉ giữ lại các cột giống hệt file Real-time
df = df[["timestamp", "temp", "humidity", "pressure", "rain", "weather_label"]]

# Lưu ra file CSV để train
df.to_csv("weather_history_train.csv", index=False)
print("Đã tạo file 'weather_history_train.csv' thành công. Bạn có thể dùng file này để train Naive Bayes ngay!")

import os

# Lấy đường dẫn thư mục hiện tại
current_folder = os.getcwd()
file_path = os.path.join(current_folder, "dataset/weather_dataset.csv")

print("File của bạn đang nằm ở đây:")
print(file_path)

# Kiểm tra xem file có thực sự ở đó không
if os.path.exists(file_path):
    print("✅ Đã tìm thấy file!")
else:
    print("❌ Chưa thấy file (Có thể bạn chưa chạy code tạo file hoặc bị lỗi khi lưu).")