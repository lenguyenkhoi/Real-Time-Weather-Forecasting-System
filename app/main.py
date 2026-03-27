import pandas as pd
import streamlit as st 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title("Weather Real Time Naive Bayes")

st.subheader("Dữ liệu real-time chưa xử lý")
data = pd.read_csv("dataset/weather_dataset.csv")

data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce') #Chuẩn hóa timestamp

st.dataframe(data)

#Chuẩn hóa 

le = LabelEncoder()
data["weather_label"] = le.fit_transform(data["weather_label"])

print(f'Classes: {le.classes_}')
print(f'Encoded labels: {data["weather_label"]}')

#Classes: ['Clear' 'Clouds' 'Mist' 'Rain']
# Encoded labels: 0    1      2       3
st.subheader("Dữ liệu đã qua xử lý Encoder")

st.dataframe(data)

# Chọn Features (Đầu vào) và Target (Nhãn dự đoán)
X = data[["temp", "humidity", "pressure", "rain"]]
y = data["weather_label"]

# 3. Chia tập Train (80%) và Test (20%)
# Việc này để kiểm tra xem model có học vẹt không
st.subheader("Chia tập dữ liệu")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cot_x, cot_y = st.columns(2)

with cot_x:
    st.subheader("Kiểm tra dữ liệu tập X_train")
    if st.button("Kiểm tra X_train"):
        st.dataframe(X_train)
        st.write(f"Shape X_train: {X_train.shape} ")
    st.subheader("Kiểm tra dữ liệu X_test")
    if st.button("Kiểm tra X_test"):
        st.dataframe(X_test)
        st.write(f"Shape X_test {X_test.shape}")

with cot_y:
    st.subheader("Kiểm tra dữ liệu tập y_train")
    if st.button("Kiểm tra y_train"):
        st.dataframe(y_train)
        st.write(f"Shape y_train: {y_train.shape}")
    st.subheader("Kiểm tra dữ liệu y_test")
    if st.button("Kiểm tra y_test"):
        st.dataframe(y_test)
        st.write(f"Shape y_test: {y_test.shape}")

st.markdown("---")
st.subheader("Train model")

model = GaussianNB()
model.fit(X_train, y_train)
# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"🎯 Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred))

model_filename = "model/weather_naive_bayes.pkl"
joblib.dump(model, model_filename)

if st.button("TRAIN"):
    st.success("Mô hình đã được huấn luyện")
    st.subheader(f"Độ chính xác của mô hình {accuracy * 100:.2f}% ")
    if accuracy > 0.5:
        st.write("Mô hình có độ chính xác cao")
    else:
        st.write("Mô hình có độ chính xác thấp")


st.markdown("---")

# Giả sử: Temp=32 độ, Humidity=80%, Pressure=1005, Rain=0.0 (Không mưa)
with st.sidebar:
    st.subheader("Nhập dữ liệu dự đoán")

    temp = st.number_input("Nhập nhiệt độ")
    humidity = st.number_input("Nhập độ ẩm")
    pressure = st.number_input("Nhập áp suất khí quyển")
    rain = st.number_input("Nhập rain")

    test_input = [[temp, humidity, pressure, rain]]

    prediction = model.predict(test_input)
    prediction_cls = le.inverse_transform(prediction)[0]
    if st.button("Dự đoán"):
        st.write(f"Temp= {temp}")
        st.write(f"humidity= {humidity}")
        st.write(f"pressure= {pressure}")
        st.write(f"rain= {rain}")
        st.success(f"Dự đoán: {prediction_cls}")



