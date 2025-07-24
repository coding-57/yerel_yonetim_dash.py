# Yerel Yönetim Harcama Analitiği Dashboard’u

Bu proje, Sayıştay ve TÜİK verilerini kullanarak yerel yönetimlerin harcama kalıplarını açıklanabilir yapay zekâ (XAI) yöntemleriyle analiz eden interaktif bir Streamlit uygulamasıdır.

## 🚀 Özellikler

- Harcama kalemleri (personel, sermaye vb.) görselleştirmesi
- Sosyoekonomik verilerle regresyon modeli
- SHAP ile açıklanabilirlik grafikleri
- İl düzeyinde veri seçimi ve yorumlama

## 📁 Dosya İçeriği

- `yerel_yonetim_dash.py` → Streamlit uygulaması
- `sayistay_harcamalar.csv` → Harcama verileri
- `tuik_nufus_gelir.csv` → Sosyoekonomik göstergeler

## ⚙️ Kurulum

```bash
pip install streamlit shap scikit-learn matplotlib pandas
streamlit run yerel_yonetim_dash.py
