# ==============================
# Streamlit Makine Öğrenmesi Projesi
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Sayfa Başlığı
# ------------------------------
st.title("🏃‍♂️ Egzersiz Verilerine Göre Kalori Seviyesi Tahmini")
st.write("Bu uygulama Streamlit kullanılarak geliştirilmiştir.")

# ------------------------------
# Veri Setini Yükleme
# ------------------------------
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    data = pd.concat([exercise, calories["Calories"]], axis=1)
    return data

data = load_data()

st.subheader("📊 Veri Seti (İlk 5 Satır)")
st.dataframe(data.head())

# ------------------------------
# Ön İşleme
# ------------------------------
# Gender sütununu sayısala çevirme
data.replace({"Gender": {"male": 0, "female": 1}}, inplace=True)

# Kalori seviyesini sınıflara ayırma
def calorie_level(cal):
    if cal < 50:
        return 0   # Düşük
    elif cal < 120:
        return 1   # Orta
    else:
        return 2   # Yüksek

data["Calories_Level"] = data["Calories"].apply(calorie_level)

# Girdi ve çıktı
X = data.drop(columns=["User_ID", "Calories", "Calories_Level"])
y = data["Calories_Level"]

# Eğitim / Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Model Seçimi
# ------------------------------
st.subheader("🧠 Makine Öğrenmesi Algoritması Seç")

model_choice = st.selectbox(
    "Bir model seçiniz:",
    (
        "Karar Ağacı",
        "KNN",
        "SVM",
        "Naive Bayes",
        "Lojistik Regresyonu"
    )
)

# ------------------------------
# Modeli Çalıştırma
# ------------------------------
if st.button("🚀 Modeli Eğit ve Test Et"):

    if model_choice == "Karar Ağacı":
        model = DecisionTreeClassifier(random_state=42)

    elif model_choice == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_choice == "SVM":
        model = SVC(kernel="rbf")

    elif model_choice == "Naive Bayes":
        model = GaussianNB()

    elif model_choice == "Lojistik Regresyonu":
        model = LogisticRegression(max_iter=1000)

    # Model eğitimi
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Accuracy: {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📌 Karışıklık Matrisi")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(model_choice)

    st.pyplot(fig)

# ------------------------------
# Açıklama
# ------------------------------
st.markdown("""
### ℹ️ Açıklama
- **0:** Düşük Kalori  
- **1:** Orta Kalori  
- **2:** Yüksek Kalori  

Bu uygulamada 5 farklı makine öğrenmesi algoritması denenmiştir.
""")
