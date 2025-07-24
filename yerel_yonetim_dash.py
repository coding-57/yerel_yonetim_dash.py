
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Başlık
st.title("Yerel Yönetim Harcama Analitiği Dashboard’u")
st.write("Sayıştay ve TÜİK verilerine dayalı SHAP destekli açıklanabilirlik modeli")

# Verileri yükleme
@st.cache_data
def load_data():
    df_sayistay = pd.read_csv("sayistay_harcamalar.csv")
    df_tuik = pd.read_csv("tuik_nufus_gelir.csv")
    df = pd.merge(df_sayistay, df_tuik, on="İl")
    return df

df = load_data()

# İl seçimi
selected_il = st.selectbox("İl seçiniz", df["İl"].unique())
selected_data = df[df["İl"] == selected_il]
st.dataframe(selected_data)

# Harcama görselleştirmesi
st.subheader("Harcama Kalemleri Dağılımı")
kalemler = ["Personel", "Sermaye", "MalHizmet", "Transfer"]
fig, ax = plt.subplots()
selected_data[kalemler].T.plot(kind="bar", legend=False, ax=ax)
plt.ylabel("Harcama (TL)")
st.pyplot(fig)

# Makine öğrenmesi modeli
X = df[["Nüfus", "Gelir", "İşsizlik", "Eğitim"]]
y = df["Sermaye"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
predictions = model.predict(X_test[:5])
st.subheader("Tahmini Sermaye Harcamaları (İlk 5 test verisi):")
st.write(predictions)

# SHAP analizi
st.subheader("SHAP Özellik Önem Analizi")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
fig_shap = shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig_shap)
