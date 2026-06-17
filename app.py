import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import timedelta, datetime
import warnings
import requests
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RiceCast Alert | Early Warning System",
    page_icon="🌾",
    layout="wide"
)

# styling untuk halaman about
st.markdown("""
            <style>
            .about-card {
                background-color: #ffffff;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.05);
                border-left: 8px solid #558d57;
                margin-bottom: 20px;
            }
            .about-image img {
                border-radius: 20px;
                object-fit: cover;
                height: 100%;
            }
            .rancangan-text {
                font-size: 14px;
                color: #558d57;
                font-weight: 1000;
                letter-spacing: 4px;
                margin-bottom: 0px;
            }
            .judul-skripsi {
                font-size: 14px;
                font-weight: 800;
                letter-spacing: 0px;
                margin-bottom: 0px;
            }
            </style>
        """, unsafe_allow_html=True)

@st.cache_resource
def load_xgb_model(kota):
    try:
        model = xgb.XGBRegressor()
        model.load_model(f"Model_XGBoost_{kota}.json")
        return model
    except FileNotFoundError:
        return None

# deteksi bulan panen dan bulan besar otomatis
def get_auto_calendar(target_date):
    bulan = target_date.month
    is_harvest = 1 if bulan in [3, 4, 7, 8] else 0
    is_big = 1 if bulan in [1, 2, 3, 4, 12] else 0
    return is_harvest, is_big

# ========================================================
# FITUR BARU: MASTER DICTIONARY & FUNGSI TARIK API
# ========================================================
master_wilayah = {
    "Jakarta":    {"hulu": "Kab. Karawang",     "lat": -6.3227, "lon": 107.3376, "prov_id": 13, "reg_id": 34},
    "Yogyakarta": {"hulu": "Kab. Bantul",       "lat": -7.8860, "lon": 110.3318, "prov_id": 15, "reg_id": 41},
    "Surabaya":   {"hulu": "Kab. Lamongan",     "lat": -7.1182, "lon": 112.3155, "prov_id": 16, "reg_id": 42},
    "Malang":     {"hulu": "Kab. Malang",       "lat": -8.1333, "lon": 112.5667, "prov_id": 16, "reg_id": 43},
    "Medan":      {"hulu": "Kab. Deli Serdang", "lat": 3.4285,  "lon": 98.8302,  "prov_id": 2,  "reg_id": 4},
    "Makassar":   {"hulu": "Kab. Sidrap",       "lat": -3.9397, "lon": 119.8138, "prov_id": 26, "reg_id": 67},
    "Pontianak":  {"hulu": "Kab. Sambas",       "lat": 1.3639,  "lon": 109.3134, "prov_id": 20, "reg_id": 57}
}

def tarik_data_live(kota_pilihan):
    hari_ini = datetime.now()
    start_date = (hari_ini - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = hari_ini.strftime("%Y-%m-%d")
    
    lat, lon = master_wilayah[kota_pilihan]["lat"], master_wilayah[kota_pilihan]["lon"]
    prov_id, reg_id = master_wilayah[kota_pilihan]["prov_id"], master_wilayah[kota_pilihan]["reg_id"]
    hulu = master_wilayah[kota_pilihan]["hulu"]
    
    # A. TARIK CUACA OPEN-METEO
    url_meteo = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=Asia%2FJakarta&forecast_days=8"
    try:
        resp_cuaca = requests.get(url_meteo, timeout=10).json()
        hasil_cuaca = [f"{val} mm" for val in resp_cuaca['daily']['precipitation_sum'][1:8]]
    except:
        hasil_cuaca = ["Error"] * 7
        
    # B. TARIK HARGA PIHPS
    url_harga = f"https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah?price_type_id=1&comcat_id=com_3&province_id={prov_id}&regency_id={reg_id}&market_id=&tipe_laporan=1&start_date={start_date}&end_date={end_date}"
    headers = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
    
    try:
        resp_harga = requests.get(url_harga, headers=headers, timeout=15).json()
        target_data = next((item for item in resp_harga.get('data', []) if item.get('name') == 'Beras Kualitas Medium I'), None)
        if target_data:
            list_tgl = [k for k in target_data.keys() if '/' in k][-4:] # Ambil 4 tanggal kerja terakhir
            hasil_harga = [f"Rp {target_data[tgl]}" for tgl in list_tgl]
        else:
            list_tgl, hasil_harga = ["H-3", "H-2", "H-1", "H"], ["Data Kosong"] * 4
    except:
        list_tgl, hasil_harga = ["H-3", "H-2", "H-1", "H"], ["Error Server"] * 4
        
    return hasil_harga, hasil_cuaca, hulu, list_tgl
# ========================================================

# branding sidebar
st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px; border: 1px solid #eee;">
        <h1 style='margin: 0; font-size: 24px; color: #1f77b4;'>🌾 RiceCast Alert</h1>
        <p style='margin: 0; font-size: 11px; color: #666; font-weight: bold;'>EARLY WARNING SYSTEM</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.write("---")

st.sidebar.markdown("### 🧭 Navigasi")

menu = st.sidebar.radio(
    label="Pilih Halaman:",
    options=["🏠 Home", "ℹ️ About", "🚨 Early Warning System"],
    label_visibility="collapsed"
)

st.sidebar.write("---")

st.sidebar.markdown(
    """
    <div style="margin-top: 100px; font-size: 10px; color: #aaa; text-align: center;">
        RiceCast Alert<br>
        Timoty Wahyudi Pakpahan<br>
        535220043
    </div>
    """, 
    unsafe_allow_html=True
)

daftar_kota = ["Jakarta", "Surabaya", "Malang", "Medan", "Yogyakarta", "Makassar", "Pontianak"]

if menu == "🏠 Home":
    st.title("🏠 Dashboard History Harga Beras & Pre-Processing")
    st.markdown("---")
    
    col_atas1, col_atas2, col_atas3 = st.columns([1, 1, 2])
    
    with col_atas1:
        st.markdown("#### 📍 Wilayah Target")
        kota_pilihan = st.selectbox("Pilih Kota Pantauan:", daftar_kota)
        
        try:
            df_after = pd.read_csv(f"Data_After_Spline_{kota_pilihan}.csv")
            df_after['time'] = pd.to_datetime(df_after['time'])
            data_tersedia = True
        except FileNotFoundError:
            data_tersedia = False
            
    with col_atas2:
        st.markdown("#### 📊 Statistik Rata-rata")
        if data_tersedia:
            df_after['Tahun'] = df_after['time'].dt.year
            rata_tahunan = df_after.groupby('Tahun')['Harga Pasar'].mean().round(0).astype(int)
            for thn, harga in rata_tahunan.items():
                st.markdown(f"- **{thn}**: Rp {harga:,}".replace(',', '.'))
        else:
            st.warning("Data CSV tidak ditemukan.")
            
    with col_atas3:
        st.markdown("#### 📈 Tren Harga Beras (2023 - 2025)")
        if data_tersedia:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=df_after['time'], y=df_after['Harga Pasar'], mode='lines', line=dict(color='#558d57')))
            fig_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, xaxis_title="Tahun", yaxis_title="Rp")
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    
    col_bawah1, col_bawah2 = st.columns([1, 2])
    
    with col_bawah1:
        st.markdown("#### 🗓️ Filter Tanggal")
        start_date = st.date_input("Start Date", pd.to_datetime("2025-01-01"))
        
        # UX Improvement: Mengunci kalender End Date agar tidak bisa mundur dari Start Date
        # Nilai default (value) dibuat dinamis menjadi Start Date + 6 hari
        default_end = start_date + timedelta(days=6)
        end_date = st.date_input("End Date", value=default_end, min_value=start_date)
        
        btn_tampil = st.button("🔍 Tampilkan Data", use_container_width=True)
        
    with col_bawah2:
        st.markdown("#### 📋 Perbandingan Missing Value (Interpolasi Spline)")
        if data_tersedia and btn_tampil:
            try:
                df_before = pd.read_csv(f"Data_Before_Spline_{kota_pilihan}.csv")
                df_before['time'] = pd.to_datetime(df_before['time'])
                
                mask_before = (df_before['time'] >= pd.to_datetime(start_date)) & (df_before['time'] <= pd.to_datetime(end_date))
                mask_after = (df_after['time'] >= pd.to_datetime(start_date)) & (df_after['time'] <= pd.to_datetime(end_date))
                
                tabel_bfr = df_before.loc[mask_before, ['time', 'Harga Pasar']].copy()
                tabel_aft = df_after.loc[mask_after, ['time', 'Harga Pasar']].copy()
                
                tabel_bfr['time'] = tabel_bfr['time'].dt.strftime('%d %b %Y')
                tabel_aft['time'] = tabel_aft['time'].dt.strftime('%d %b %Y')
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.markdown("**Data Raw (Before Spline)**")
                    st.dataframe(tabel_bfr, use_container_width=True, hide_index=True)
                with subcol2:
                    st.markdown("**Data Bersih (After Spline)**")
                    st.dataframe(tabel_aft, use_container_width=True, hide_index=True)
            except FileNotFoundError:
                st.error("File 'Data_Before_Spline' tidak ditemukan.")

elif menu == "ℹ️ About":
        col_ab1, col_ab2 = st.columns([2, 1], gap="large")
        
        with col_ab1:
            st.markdown(f"""
                <div class="about-card">
                    <p class="rancangan-text">RANCANGAN</p>
                    <br>
                    <br>
                    <p class="judul-skripsi">PERANCANGAN SISTEM PREDIKSI HARGA BERAS KUALITAS MEDIUM MENGGUNAKAN XGBOOST BERBASIS PREPROCESSING INTERPOLASI SPLINE</p>
                    <br>
                    <p style="text-align: justify; color: #444; line-height: 1.6;">
                        Sistem Peringatan Dini (Early Warning System) ini dirancang untuk memantau fluktuasi harga beras secara real-time. 
                        Dengan memanfaatkan algoritma <b>XGBoost Regressor</b> yang dikombinasikan dengan <b>Interpolasi Spline</b> untuk menangani data kosong, 
                        sistem mampu memberikan proyeksi harga 7 hari ke depan dengan mempertimbangkan faktor eksternal seperti curah hujan dan siklus kalender pangan nasional.
                    </p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 25px 0;">
                    <p style="margin-bottom: 5px;"><b>Dikembangkan oleh:</b></p>
                    <p style="font-size: 20px; font-weight: 700; color: #333; margin-bottom: 0;">Timoty Wahyudi Pakpahan</p>
                    <p style="color: #666; margin-bottom: 2px;">NIM - 535220043</p>
                    <p style="color: #666; line-height: 1.2;">Program Studi - Teknik Informatika<br>Universitas Tarumanagara</p>
                    <hr/> 
                    <p style="margin-bottom: 5px;"><b>Dibimbing oleh:</b></p>
                    <p style="color: #666; margin-bottom: 2px;">Pembimbing 1: </p>
                    <p style="font-size: 20px; font-weight: 700; color: #333; margin-bottom: 0;">Prof. Dr. Ir. Dyah Erny Herwindiati, M.Si.</p>
                    <p style="color: #666; margin-bottom: 2px;">Pembimbing 2: </p>
                    <p style="font-size: 20px; font-weight: 700; color: #333; margin-bottom: 0;">Janson Hendryli, S.Kom., M.Kom.</p>
                  </div>
            """, unsafe_allow_html=True)

            try:
                with open("Manual Book Skripsi 2026.pdf", "rb") as f:
                    st.download_button(
                        label="📄 Download Manual Book",
                        data=f,
                        file_name="Manual Book Skripsi 2026.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except:
                st.info("Gaada Manual Book")
            
        with col_ab2:
            st.image("pasar beras.jpg", use_container_width=True, caption="Pasar Beras \n\n https://www.pertanian.go.id/home/?show=news&act=view&id=2142")
            st.image("beras.jpg", use_container_width=True, caption="Beras Kualitas Medium\n\n https://foto.bisnis.com/view/20260402/1963827/harga-beras-naik-sepanjang-maret-2026-premium-tembus-rp17250kg")
            st.image("el nino.jpeg", use_container_width=True, caption="Fenomena El Nino \n\n https://finance.detik.com/berita-ekonomi-bisnis/d-6864250/dampak-el-nino-kian-nyata-kekeringan-hingga-gagal-panen-hantui-ri")

elif menu == "🚨 Early Warning System":
    st.title("🚨 Dashboard Prediksi Harga Beras")
    st.markdown("---")
    st.markdown("""
    <style>
    /* suffix mm untuk curah hujan */
    div[data-testid="stNumberInput"]:has(input[aria-label*="H+"]) {
        position: relative;
    }
    div[data-testid="stNumberInput"]:has(input[aria-label*="H+"]) input {
        padding-right: 35px;
    }
    div[data-testid="stNumberInput"]:has(input[aria-label*="H+"])::after {
        content: "mm";
        position: absolute;
        right: 65px;
        bottom: 10px;
        font-size: 12px;
        color: #666;
        pointer-events: none;
    }

    /* prefix Rp untuk input harga */
    div[data-testid="stNumberInput"]:has(input[aria-label*="H-"]) {
        position: relative;
    }
    div[data-testid="stNumberInput"]:has(input[aria-label*="H-"]) input {
        padding-left: 30px;
    }
    div[data-testid="stNumberInput"]:has(input[aria-label*="H-"])::before {
        content: "Rp. ";
        position: absolute;
        left: 10px;
        bottom: 10px;
        font-size: 12px;
        color: #666;
        pointer-events: none;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("### ⚙️ 1. Pemilihan Kota & Tanggal ")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        kota_prediksi = st.selectbox("📍 Pilih Kota:", daftar_kota)
        model = load_xgb_model(kota_prediksi)
    with col_in2:
        # Menyimpan variabel waktu untuk perhitungan model di bawah (tidak terlihat di UI)
        start_date = pd.to_datetime("today")
        
        # Mengubah format tanggal menjadi string yang cantik (contoh: 14 Jun 2026)
        hari_ini_str = start_date.strftime("%d %b %Y")
        
        # Membuat label
        st.markdown("<p style='font-size: 14px; margin-bottom: 5px;'>🗓️ Tanggal Mulai (Hari Ini):</p>", unsafe_allow_html=True)
        
        # Membuat kotak indikator yang mirip form input, tapi tidak bisa diklik
        st.markdown(f"""
            <div style="
                padding: 8px 14px; 
                border: 1px solid #dcdcdc; 
                border-radius: 8px; 
                background-color: #f9f9f9; 
                color: #333; 
                font-size: 14px;
                cursor: not-allowed;
            ">
                {hari_ini_str}
            </div>
        """, unsafe_allow_html=True)
    
    if model is None:
        st.error(f"❌ Model untuk {kota_prediksi} belum tersedia!")
        st.stop()
    
    # ========================================================
    # FITUR BARU: UI TARIK DATA REFERENSI LIVE
    # ========================================================
    with st.expander(f"📡 Tarik Data Referensi API Live untuk {kota_prediksi} (Klik di sini)"):
        st.write("Gunakan fitur ini untuk menarik data historis harga dari server PIHPS dan prakiraan cuaca satelit dari wilayah sentra produksi sebagai acuan pengisian model XGBoost.")
        
        if st.button(f"🔄 Tarik Data Live {kota_prediksi}", type="secondary"):
            with st.spinner(f"Menghubungkan ke satelit meteorologi dan database pemerintah..."):
                # list_tgl tetap ditarik dari fungsi, tapi kita abaikan untuk tampilan UI
                hasil_harga, hasil_cuaca, nama_hulu, list_tgl = tarik_data_live(kota_prediksi)
                st.success("✅ Berhasil menarik data terkini!")
                
                # 1. Tabel Harga (Format Sinkron dengan Input Form)
                st.markdown(f"**📊 Harga Beras Kualitas Medium I ({kota_prediksi}) - 4 Hari Kerja Terakhir**")
                df_harga = pd.DataFrame([hasil_harga], columns=["H-3", "H-2", "H-1", "Hari Ini (H)"], index=["Harga"])
                st.dataframe(df_harga, use_container_width=True)
                
                # 2. Tabel Cuaca (Format Sinkron dengan Input Form)
                st.markdown(f"**🌦️ Prakiraan Curah Hujan di Wilayah Hulu ({nama_hulu}) - 7 Hari ke Depan**")
                df_cuaca = pd.DataFrame([hasil_cuaca], columns=["H+1", "H+2", "H+3", "H+4", "H+5", "H+6", "H+7"], index=["Curah Hujan"])
                st.dataframe(df_cuaca, use_container_width=True)
    # ========================================================

    #     
    st.markdown("### 💰 2. Input Harga Beras (3 Hari Terakhir)")
    st.info("🔎 **Sumber Data Harga:** Cek riwayat harga beras 3 hari terakhir melalui [Web PIHPS - Bank Indonesia](https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalDaerah)")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1: h3_price = st.number_input(f"H-2 ({(start_date - timedelta(days=2)).strftime('%d %b')}):", value=0, help="Masukkan harga beras **2 Hari lalu** tanpa titik.")
    with col_p2: h2_price = st.number_input(f"H-1 ({(start_date - timedelta(days=1)).strftime('%d %b')}):", value=0, help="Masukkan harga beras **1 Hari lalu** tanpa titik.")
    with col_p3: h1_price = st.number_input(f"H ({start_date.strftime('%d %b')}):", value=0, help="Masukkan harga beras **Hari Ini** lalu tanpa titik.")


    st.markdown("### ⛅ 3. Curah Hujan mm (7 Hari Kedepan)")
    st.info("🌦️ **Sumber Estimasi Cuaca:** Cek prakiraan curah hujan mingguan di [Web AccuWeather](https://www.accuweather.com/)")                           
    rain_cols = st.columns(7)
    hujan_input = []

    
    for i in range(7):
        target_date = start_date + timedelta(days=i+1)
        with rain_cols[i]:
            val = st.number_input(
                f"H+{i+1} ({target_date.strftime('%d %b')})", 
                value=10.0, 
                step=5.0, 
                key=f"r{i}",
                help="Masukkan total curah hujan harian dalam satuan milimeter (mm)."
            )
            hujan_input.append(val)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Prediksi Harga Beras", use_container_width=True, type="primary"):
        st.markdown("---")
        st.markdown("### 📊 2. HASIL ANALISIS")
        
        harga_memori = [h3_price, h2_price, h1_price]
        prediksi_7_hari = []
        tgl_prediksi = []
        
        for i in range(7):
            target_date = start_date + timedelta(days=i+1)
            tgl_prediksi.append(target_date)
            is_harvest, is_big = get_auto_calendar(target_date)
            
            d1 = harga_memori[-1] - harga_memori[-2]
            d2 = harga_memori[-2] - harga_memori[-3]
            
            input_df = pd.DataFrame([[d1, d2, hujan_input[i], is_harvest, is_big]],
                                    columns=['Delta_H-1', 'Delta_H-2', 'Curah Hujan', 'IsHarvestMonth', 'IsBigMonth'])
            
            delta_pred = model.predict(input_df)[0]
            harga_hasil = harga_memori[-1] + delta_pred
            
            prediksi_7_hari.append(harga_hasil)
            harga_memori.append(harga_hasil)
            
        selisih_total = prediksi_7_hari[-1] - h1_price
        
        col_out1, col_out2 = st.columns([1, 2])
        with col_out1:
            if selisih_total <= 100:
                st.markdown("<div class='status-aman'><h3>🟢 AMAN</h3><p>Gejolak Terkendali</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-waspada'><h3>🟡 WASPADA</h3><p>Risiko Kenaikan</p></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Prediksi Kenaikan/Penurunan:**\n\n{'+' if selisih_total > 0 else '-'} Rp {abs(int(selisih_total))} (Di hari ke-7)")
            
        with col_out2:
            fig_ews = go.Figure()
            fig_ews.add_trace(go.Scatter(x=[start_date - timedelta(days=2), start_date - timedelta(days=1), start_date], 
                                         y=[h3_price, h2_price, h1_price], mode='lines+markers', name='Harga Aktual', line=dict(color='black', width=3)))
            fig_ews.add_trace(go.Scatter(x=[start_date] + tgl_prediksi, y=[h1_price] + prediksi_7_hari, 
                                         mode='lines+markers', name='Prediksi EWS', line=dict(color='red', width=3, dash='dash')))
            fig_ews.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250, title="Trend Harga Beras 7 Hari Kedepan")
            st.plotly_chart(fig_ews, use_container_width=True)

        st.markdown("**📋 Rincian Harga 7 Hari Kedepan**")
        
        kolom_tanggal = [t.strftime('%d %b') for t in tgl_prediksi]
        baris_harga = [f"Rp {int(p):,}".replace(',', '.') for p in prediksi_7_hari]
        
        df_horizontal = pd.DataFrame([baris_harga], columns=kolom_tanggal, index=["Harga (Rp)"])
        st.dataframe(df_horizontal, use_container_width=True)
