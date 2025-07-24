import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

st.set_page_config(page_title="Yerel Yönetim Harcamaları Analizi", layout="wide")

# Başlık
st.title("📊 Yerel Yönetim Harcamaları Analitiği")
st.markdown("**Açıklanabilir Yapay Zekâ ile Büyükşehir Harcamalarının Analizi – SHAP Destekli Yaklaşım**")

# Verileri yükleme
@st.cache_data
def load_data():
    df_sayistay = pd.read_csv("sayistay_harcamalar.csv")
    df_tuik = pd.read_csv("tuik_nufus_gelir.csv")
    df = pd.merge(df_sayistay, df_tuik, on="il")
    return df

df = load_data()

# Filtreleme
st.sidebar.header("🔎 Filtre Seçenekleri")
selected_il = st.sidebar.selectbox("İl Seçiniz", df["il"].unique())
selected_yil = st.sidebar.selectbox("Yıl Seçiniz", sorted(df["Yil"].unique()))
filtered = df[(df["il"] == selected_il) & (df["Yil"] == selected_yil)]

st.subheader(f"{selected_il} ({selected_yil}) Harcama Kalemleri")
st.dataframe(filtered)

# Harcama görselleştirmesi
kalemler = ["Personel", "Sermaye", "MalHizmet", "Transfer"]
fig, ax = plt.subplots()
filtered[kalemler].T.plot(kind="bar", legend=False, ax=ax)
plt.ylabel("Harcama Tutarı (TL)")
plt.title("Harcama Kalemleri Dağılımı")
st.pyplot(fig)

# Model eğitimi
st.subheader("🔍 Makine Öğrenmesi ile Tahmin")
X = df[["Nufus", "Gelir", "issizlik", "Egitim"]]
y = df["Sermaye"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
predictions = model.predict(X_test[:5])

st.markdown("**Test Verileri Üzerinde Tahmin Edilen Sermaye Harcamaları:**")
st.write(predictions)

# SHAP ile açıklanabilirlik
st.subheader("🧠 SHAP Açıklanabilirlik Analizi")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP beeswarm plot
import matplotlib.pyplot as plt
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(plt.gcf())

import streamlit as st
import streamlit.components.v1 as components

# Örnek gözlem seçimi
index = st.slider("🧬 Force plot için örnek seçin", 0, len(X_test) - 1, 0)
instance = X_test.iloc[index]

# SHAP force plot üretimi
force_plot_html = shap.plots.force(
    explainer.expected_value,           # modelin genel tahmini
    shap_values[index].values,          # o örneğin katkı değerleri
    instance,                           # özellik değerleri
    matplotlib=False                    # HTML çıktısı
)

# Streamlit'e gömme
st.subheader("🔎 SHAP Force Plot – Tahminin Açıklaması")
components.html(force_plot_html.html(), height=300, scrolling=True)

