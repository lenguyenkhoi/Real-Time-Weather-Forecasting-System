# 🌦️ Real-Time Weather Forecasting System

## 📌 Project Overview

This project builds a **Real-Time Weather Forecasting System** by combining **historical data crawling** and **real-time data streaming** for Ho Chi Minh City. The system applies a **Gaussian Naive Bayes (GaussianNB)** model to predict weather conditions in real time, enabling continuous and automated forecasting.

---

## 🎯 Objectives

* Collect and process **historical weather data**
* Continuously **crawl real-time weather data**
* Build a **machine learning model (GaussianNB)** for prediction
* Provide **real-time weather forecasting**

---

## 📂 Data Collection

### Historical Data

* Crawled historical weather data for **Ho Chi Minh City** from online sources

### Real-Time Data

* Continuously fetched live weather data for **Ho Chi Minh City**

### Data Integration

* Integrating historical data with real-time data
* Used for training the prediction model
* Used as input for real-time prediction
---

## 🧹 Data Processing

* Cleaned and handled missing values
* Standardized numerical features (temperature, humidity, etc.)
* Encoded categorical variables (weather conditions)

---

## 🤖 Machine Learning Model

### Model: Gaussian Naive Bayes

* Suitable for probabilistic classification
* Fast and efficient for real-time prediction

### Workflow:

* Train model on historical and real-time data integration
* Evaluate model performance
* Deploy model for real-time inference

---

## ⚙️ System Pipeline

1. Crawl historical data → store dataset
2. Crawl real-time data continuously → store dataset
3. Integration of historical and real-time data → store dataset
4. Train GaussianNB model
5. Input real-time data into trained model
6. Output predicted weather condition

---

## 📊 Output

* Real-time weather predictions (e.g., Rain, Sunny, Cloudy)
* Continuous updates based on incoming data

---

## 🗂️ Project Structure

```id="wx01"
real-time-weather-forecasting/
│
├── dataset/
│   └── weather_dataset.csv
│
│
├── scripts/
│   ├── get_history.py
│   ├── collect_realtime.py
│   ├── train_model.py
│   └── train_partial.py
│
├── models/
│   ├── weather_naive_bayes.pkl
│   └── weather_nb_realtime.pkl
│
├── app/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 🛠 Tools & Technologies

* Python (pandas, numpy, scikit-learn)
* Web Crawling (BeautifulSoup / API)
* Framework (Streamlit)
* Machine Learning (GaussianNB)

---

## 📌 Conclusion

This project demonstrates how to integrate **data engineering (crawling)** with **machine learning** to build a **real-time predictive system**. It highlights the ability to handle streaming data and deploy lightweight models for instant predictions.

---

## 📎 Author

* LeNguyenKhoi
