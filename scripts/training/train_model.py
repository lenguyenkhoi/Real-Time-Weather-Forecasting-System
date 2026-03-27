import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dữ liệu lịch sử
csv_file = "weather_dataset.csv"

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Đã load thành công {len(df)} dòng dữ liệu.")
except FileNotFoundError:
    print("❌ Không tìm thấy file csv! Hãy chạy file 'get_history.py' trước.")
    exit()

# 2. Chọn Features (Đầu vào) và Target (Nhãn dự đoán)
# Lưu ý: Thứ tự các cột trong 'features' này RẤT QUAN TRỌNG.
# Khi chạy Real-time, bạn phải đưa dữ liệu vào đúng thứ tự này.
feature_cols = ['temp', 'humidity', 'pressure', 'rain']
target_col = 'weather_label'

X = df[feature_cols]  # Dữ liệu đầu vào
y = df[target_col]    # Nhãn (VD: Rain, Clouds, Clear...)

# 3. Chia tập Train (80%) và Test (20%)
# Việc này để kiểm tra xem model có học vẹt không
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nĐang train mô hình với {len(X_train)} mẫu dữ liệu...")

# 4. Khởi tạo và Huấn luyện mô hình Gaussian Naive Bayes
# Tại sao dùng GaussianNB? Vì nhiệt độ, độ ẩm là số thực liên tục (Continuous),
# tuân theo phân phối chuẩn (Gaussian distribution).
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"🎯 Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred))

# 6. Lưu mô hình để dùng sau này
model_filename = "weather_naive_bayes.pkl"
joblib.dump(model, model_filename)

print("-" * 30)
print(f"💾 Đã lưu model vào file: {model_filename}")
print("Bây giờ bạn có thể dùng file này cho code Real-time!")

# --- DEMO THỬ NGHIỆM ---
print("\n--- TEST THỬ DỰ ĐOÁN ---")
# Giả sử: Temp=32 độ, Humidity=80%, Pressure=1005, Rain=0.0 (Không mưa)
test_input = [[32, 80, 1005, 0.0]] 
prediction = model.predict(test_input)
print(f"Input: Temp=32, Hum=80 -> Dự đoán: {prediction[0]}")

# Giả sử: Temp=25 độ, Humidity=95%, Pressure=998, Rain=5.0 (Mưa to)
test_input_rain = [[25, 95, 998, 5.0]]
prediction_rain = model.predict(test_input_rain)
print(f"Input: Temp=25, Rain=5.0 -> Dự đoán: {prediction_rain[0]}")