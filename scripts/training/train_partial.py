import pandas as pd
import joblib
import time
import os
from sklearn.naive_bayes import GaussianNB

file = "dataset/weather_dataset.csv"
model_file = "model/weather_nb_realtime.pkl"

features = ['temp', 'humidity', 'pressure', 'rain']
label = 'weather_label'

# 1.LOAD / KHỞI TẠO MODEL
if os.path.exists(model_file):
    model = joblib.load(model_file)
    print("Đã load model cũ")
    is_initialized = True
else:
    model = GaussianNB()
    is_initialized = False
    print("Khởi tạo model mới")

# 2.LOOP REAL-TIME
while True:
    try:
        # Đọc dữ liệu realtime mới nhất
        df = pd.read_csv(file)

        # Lấy dòng cuối cùng (mới nhất)
        latest = df.iloc[-1:]

        X_new = latest[features]
        y_new = latest[label]

        # 3️. INIT MODEL (LẦN ĐẦU)
        if not is_initialized:
            classes = df[label].unique()
            model.partial_fit(X_new, y_new, classes=classes)
            is_initialized = True
            print("Model đã được khởi tạo")
        else:
            model.partial_fit(X_new, y_new)
            print("Model học thêm dữ liệu mới")

        # 4️. DỰ ĐOÁN NGAY SAU KHI HỌC
        prediction = model.predict(X_new)[0]
        print(f"Dự đoán realtime: {prediction}")

        # 5️. LƯU MODEL
        joblib.dump(model, model_file)

        # Chờ 10 phút
        time.sleep(600)

    except Exception as e:
        print("❌ Lỗi:", e)
        time.sleep(600)
