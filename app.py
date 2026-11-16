"""
Footballistic: Streamlit Interactive Football Match Analytics Dashboard.
Features:
- Individual Player Deep-Dive: 2D Top-Down Average Positions, Heatmaps, and Movement Trails.
- Team Tactical Intelligence: Aggregated Formations, Centroid, Compactness, and Team Density.
- Full Match Replay with Clean Dynamic Motion Trails.
- Searchable Spatial Data Roster with CSV Export.
"""

import os
import tempfile
from typing import Optional, List, Dict
import cv2
import pandas as pd
import streamlit as st
import numpy as np

from main import run_pipeline
from src.analysis.heatmap import SpatialAnalytics
from src.homography.pitch_view import PitchView2D

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Footballistic • AI Match Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MINIMALIST MODERN TACTICAL UI CSS -----------------
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .app-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .brand-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F9FAFB;
        margin: 0;
    }
    .brand-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 4px 9px;
        border-radius: 6px;
    }
    .brand-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    /* Cards */
    .tactical-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    /* Player Profile Hero Banner */
    .player-hero-card {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .player-hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0;
    }
    .player-role-badge {
        display: inline-block;
        background-color: #374151;
        color: #E5E7EB;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        margin-top: 6px;
    }
    .team-badge-0 {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        font-size: 0.80rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .team-badge-1 {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        font-size: 0.80rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.98rem;
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        color: #9CA3AF;
    }
    .stTabs [aria-selected="true"] {
        color: #10B981 !important;
        border-bottom: 2px solid #10B981 !important;
    }

    /* Custom Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F9FAFB !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #9CA3AF !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SESSION STATE & AUTO-LOAD DATA -----------------
def init_session():
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # Auto-load existing previous analysis from disk if available
    if st.session_state["results"] is None:
        spatial_json = "output/spatial_data.json"
        existing_video = "output/analyzed_match.mp4"
        if os.path.exists(spatial_json):
            try:
                sa = SpatialAnalytics.load_from_json(spatial_json)
                pv = PitchView2D(pitch_width=1050, pitch_height=680)
                st.session_state["results"] = {
                    "output_video_path": existing_video if os.path.exists(existing_video) else None,
                    "spatial_analytics": sa,
                    "pitch_view": pv,
                    "pitch_2d_formation_path": "output/pitch_2d_formation.png",
                    "pitch_2d_team0_path": "output/pitch_2d_heatmap_team0.png",
                    "pitch_2d_team1_path": "output/pitch_2d_heatmap_team1.png",
                    "pitch_2d_all_path": "output/pitch_2d_heatmap_all.png",
                    "formation_path": "output/average_positions.png",
                    "overlay_path": "output/heatmap_pitch_overlay.png",
                    "team0_heatmap_path": "output/heatmap_team_0.png",
                    "team1_heatmap_path": "output/heatmap_team_1.png",
                    "first_frame_path": "output/first_frame.png",
                    "total_players": len(sa.player_tracks),
                    "video_info": {"width": sa.width, "height": sa.height, "fps": 25.0, "total_frames": sa.total_processed_frames}
                }
            except Exception as e:
                pass


init_session()

# ----------------- APP HEADER -----------------
st.markdown("""
<div class="app-brand">
    <h1 class="brand-title">⚽ Footballistic</h1>
    <span class="brand-badge">2D Tactical Engine</span>
</div>
<div class="brand-subtitle">
    Bilgisayarlı Görü (Computer Vision) Tabanlı Kuşbakışı Futbol Maç Analitiği, Isı Haritası ve Oyuncu/Takım Zekası
</div>
""", unsafe_allow_html=True)


# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("⚙️ Maç & Model Ayarları")
    
    video_source = st.radio(
        "Giriş Videosu:",
        ["Örnek Maç Klibi (DFL Bundesliga)", "Özel Video Yükle"],
        index=0
    )
    
    input_video_path = None
    sample_clip_path = os.path.join("data", "sample_videos", "sample.mp4")

    if video_source == "Örnek Maç Klibi (DFL Bundesliga)":
        if os.path.exists(sample_clip_path):
            input_video_path = sample_clip_path
            st.success("Hazır klip devrede: `sample.mp4` (1080p, 25 FPS)")
        else:
            st.warning("Örnek video klasörde bulunamadı.")
    else:
        uploaded_file = st.file_uploader(
            "Video Yükle (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name
            st.success("Özel maç videosu başarıyla yüklendi!")

    st.markdown("---")
    st.subheader("Model & Analiz Parametreleri")

    model_choice = st.selectbox(
        "Tespit Modeli:",
        ["YOLO11 (En Güncel & Hızlı - Önerilen)", "YOLOv8 (Klasik)"],
        index=0
    )
    selected_model_path = "weights/yolo11m.pt" if "YOLO11" in model_choice else "weights/yolov8m.pt"

    analyze_full = st.checkbox("Tüm Videoyu Analiz Et (30 Saniye)", value=False)
    
    if not analyze_full:
        max_frames = st.slider(
            "İşlenecek Kare Sayısı (25 kare ≈ 1 sn):",
            min_value=25,
            max_value=750,
            value=60,
            step=25,
            help="Düşük kare sayısı hızlı sonuç verir; yüksek kare sayısı daha uzun takip verisi sağlar."
        )
    else:
        max_frames = 0

    conf_thresh = st.slider(
        "Tespit Güven Eşiği (Confidence):",
        min_value=0.20,
        max_value=0.70,
        value=0.35,
        step=0.05
    )

    analyze_btn = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)


# ----------------- RUN ANALYSIS PIPELINE -----------------
if analyze_btn:
    if input_video_path is None or not os.path.exists(input_video_path):
        st.error("Lütfen geçerli bir maç videosu seçin veya yükleyin.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(curr, total, msg):
            pct = min(1.0, max(0.0, curr / total))
            progress_bar.progress(pct)
            status_text.info(f"⏳ **{msg}** ({int(pct * 100)}%)")

        try:
            with st.spinner("YOLO11, saha segmentasyonu ve 2D kuşbakışı haritalama çalışıyor..."):
                results = run_pipeline(
                    video_path=input_video_path,
                    output_dir="output",
                    max_frames=max_frames,
                    model_name=selected_model_path,
                    confidence=conf_thresh,
                    progress_callback=update_progress
                )
                st.session_state["results"] = results
                status_text.success("🎉 Analiz tamamlandı! Raporlar aşağıda güncellendi.")
                progress_bar.empty()
        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")


# ----------------- MAIN DASHBOARD -----------------
results = st.session_state["results"]

if results is not None:
    spatial_analytics: Optional[SpatialAnalytics] = results.get("spatial_analytics")
    pitch_view: Optional[PitchView2D] = results.get("pitch_view")
    if pitch_view is None:
        pitch_view = PitchView2D(1050, 680)

    # Global KPI Row
    v_info = results.get("video_info", {})
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Tespit Edilen Oyuncular", f"{results.get('total_players', 0)} Oyuncu")
    with col_kpi2:
        st.metric("Analiz Edilen Kare", f"{v_info.get('total_frames', 0)} Kare")
    with col_kpi3:
        st.metric("Video Çözünürlüğü", f"{v_info.get('width', 1920)} × {v_info.get('height', 1080)}")
    with col_kpi4:
        st.metric("2D Kuşbakışı Haritalama", "Aktif (105m × 68m)")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Main Tabs
    tab_player, tab_team, tab_video, tab_data = st.tabs([
        "🎯 Oyuncu Bazlı Analiz (Deep-Dive)",
        "🛡️ Takım Taktik Analizi",
        "🎥 Maç Yayını & Hareket Kuyrukları",
        "📋 Kapsamlı Veri Tablosu"
    ])

    # -------------------------------------------------------------
    # TAB 1: OYUNCU BAZLI ANALİZ (TEK OYUNCU SEÇİMİ & 2D KUŞBAKIŞI)
    # -------------------------------------------------------------
    with tab_player:
        st.subheader("Oyuncu Bazlı 2D Kuşbakışı Taktik İstihbarat")
        st.caption("Bir oyuncu ID'si seçtiğinizde yalnızca o oyuncunun kuşbakışı ortalama pozisyonu, hareket rotası ve ısı haritası gösterilir.")

        if spatial_analytics is not None and spatial_analytics.player_tracks:
            # Team Filter Pills
            filter_team = st.radio(
                "Oyuncu Takım Filtresi:",
                ["Tüm Takımlar", "🔴 Takım 0", "🔵 Takım 1"],
                horizontal=True
            )

            # Build player list according to filter
            all_pids = sorted(list(spatial_analytics.player_tracks.keys()))
            if filter_team == "🔴 Takım 0":
                filtered_pids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 0]
            elif filter_team == "🔵 Takım 1":
                filtered_pids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 1]
            else:
                filtered_pids = all_pids

            if filtered_pids:
                selected_pid = st.selectbox(
                    "İncelemek İstediğiniz Oyuncu ID'sini Seçin:",
                    filtered_pids,
                    format_func=lambda pid: f"Oyuncu #{pid} • {'Takım 0 (🔴 Kırmızı)' if spatial_analytics.player_teams.get(pid) == 0 else 'Takım 1 (🔵 Mavi)'} (Varlık: {spatial_analytics.get_player_metrics(pid)['presence_pct']}%)"
                )

                # Get comprehensive metrics for ONLY this player
                pm = spatial_analytics.get_player_metrics(selected_pid)
                team_id = pm["team_id"]
                badge_class = "team-badge-0" if team_id == 0 else "team-badge-1"
                team_name = "🔴 Takım 0 (Ev Sahibi / Kırmızı)" if team_id == 0 else "🔵 Takım 1 (Deplasman / Mavi)"

                # Hero Player Profile Banner
                st.markdown(f"""
                <div class="player-hero-card">
                    <div>
                        <h2 class="player-hero-title">Oyuncu #{selected_pid} Taktik Profili</h2>
                        <span class="{badge_class}">{team_name}</span>
                        <span class="player-role-badge">Taktik Bölge: {pm['tactical_zone']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metric Cards for ONLY this player
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Kat Edilen Mesafe", f"{pm['distance_m']} metre")
                with m_col2:
                    st.metric("Ortalama Koşu Hızı", f"{pm['avg_speed_kmh']} km/s")
                with m_col3:
                    st.metric("Sahada Kalma (Varlık)", f"{pm['presence_pct']}% ({pm['frames_active']} kare)")
                with m_col4:
                    st.metric("Kuşbakışı Konum (X, Y)", f"{pm['pos_meters'][0]}m, {pm['pos_meters'][1]}m")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # 2 Prominent Kuşbakışı (2D Top-Down) Pitch Panels
                p_col_left, p_col_right = st.columns(2)

                # Checkboxes to customize player focus view
                with p_col_left:
                    st.markdown("#### 📍 2D Kuşbakışı Ortalama Pozisyon & Rota")
                    st.caption("Standart 105m × 68m saha üzerinde bu oyuncunun ağırlıklı konumu ve sahada katettiği rota.")
                    
                    chk_col1, chk_col2 = st.columns(2)
                    with chk_col1:
                        show_trail = st.checkbox("Hareket İzini Çiz (Start/End)", value=True, key=f"trail_{selected_pid}")
                    with chk_col2:
                        show_teammates = st.checkbox("Takım Arkadaşlarını Göster (Soluk)", value=True, key=f"tm_{selected_pid}")

                    # Prepare teammate positions if enabled
                    teammate_dict = None
                    if show_teammates:
                        avg_2d = spatial_analytics.get_average_positions_2d()
                        teammate_dict = {
                            p: pos for p, pos in avg_2d.items()
                            if spatial_analytics.player_teams.get(p) == team_id and p != selected_pid
                        }

                    focus_pitch = pitch_view.draw_2d_player_focus(
                        tracker_id=selected_pid,
                        avg_pos_2d=pm["avg_pos_2d"],
                        team_id=team_id,
                        trail_2d=pm["positions_2d"] if show_trail else None,
                        teammate_positions=teammate_dict
                    )
                    st.image(
                        cv2.cvtColor(focus_pitch, cv2.COLOR_BGR2RGB),
                        caption=f"Oyuncu #{selected_pid} 2D Kuşbakışı Taktik Pozisyonu",
                        use_container_width=True
                    )

                with p_col_right:
                    st.markdown("#### 🔥 2D Kuşbakışı Oyuncu Isı Haritası")
                    st.caption("Yalnızca bu oyuncunun sahada topla veya topsuz baskı kurduğu ve yoğunlaştığı alanlar.")
                    
                    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
                    p_heatmap_2d = pitch_view.generate_2d_heatmap(pm["positions_2d"], sigma=22.0)
                    st.image(
                        cv2.cvtColor(p_heatmap_2d, cv2.COLOR_BGR2RGB),
                        caption=f"Oyuncu #{selected_pid} 2D Kuşbakışı Isı Yoğunluğu",
                        use_container_width=True
                    )

                st.markdown("---")
                
                # Pitch Thirds (Bölgesel Yerleşim) Breakdown
                st.markdown("#### 📐 Oyuncunun Saha Boyu Dağılımı (Pitch Thirds)")
                t_col1, t_col2, t_col3 = st.columns(3)
                with t_col1:
                    st.markdown(f"**🛡️ 1. Bölge (Savunma):** `{pm['defensive_pct']}%`")
                    st.progress(float(pm['defensive_pct']) / 100.0)
                with t_col2:
                    st.markdown(f"**⚖️ 2. Bölge (Orta Saha):** `{pm['middle_pct']}%`")
                    st.progress(float(pm['middle_pct']) / 100.0)
                with t_col3:
                    st.markdown(f"**⚡ 3. Bölge (Hücum):** `{pm['attacking_pct']}%`")
                    st.progress(float(pm['attacking_pct']) / 100.0)

            else:
                st.warning("Seçilen filtreye uygun oyuncu bulunamadı.")
        else:
            st.info("Oyuncu analizlerini görmek için soldaki menüden analizi başlatın.")

    # -------------------------------------------------------------
    # TAB 2: TAKIM TAKTİK ANALİZİ (OYUNCULARDAN AGGREGATE EDİLEN)
    # -------------------------------------------------------------
    with tab_team:
        st.subheader("Takım Taktik Zekası & Toplu Kuşbakışı Analiz")
        st.caption("Bu bölümdeki tüm ortalama pozisyon ve ısı haritası verileri, oyuncu bazlı bireysel istatistiklerin takıma özel toplanmasıyla elde edilir.")

        if spatial_analytics is not None:
            team_view_select = st.radio(
                "İncelenecek Takım:",
                ["🔴 Takım 0 Analizi", "🔵 Takım 1 Analizi", "🆚 Karşılaştırmalı Genel Saha"],
                horizontal=True
            )

            avg_positions_2d = spatial_analytics.get_average_positions_2d()

            if "Karşılaştırmalı" in team_view_select:
                # Comparison View (Both Teams Together)
                c_t0 = spatial_analytics.get_team_metrics(0)
                c_t1 = spatial_analytics.get_team_metrics(1)

                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown("#### 🔴 Takım 0 Özeti")
                    st.write(f"- **Kadro:** {c_t0['player_count']} Oyuncu")
                    st.write(f"- **Blok Yüksekliği (Centroid X):** {c_t0['centroid_m'][0]}m")
                    st.write(f"- **Takım Genişliği (Width):** {c_t0['team_width_m']}m")
                    st.write(f"- **Kompaktlık (Depth):** {c_t0['team_depth_m']}m")
                    st.write(f"- **Toplam Mesafe:** {c_t0['total_distance_m']}m")
                with col_comp2:
                    st.markdown("#### 🔵 Takım 1 Özeti")
                    st.write(f"- **Kadro:** {c_t1['player_count']} Oyuncu")
                    st.write(f"- **Blok Yüksekliği (Centroid X):** {c_t1['centroid_m'][0]}m")
                    st.write(f"- **Takım Genişliği (Width):** {c_t1['team_width_m']}m")
                    st.write(f"- **Kompaktlık (Depth):** {c_t1['team_depth_m']}m")
                    st.write(f"- **Toplam Mesafe:** {c_t1['total_distance_m']}m")

                st.markdown("---")
                comp_col_left, comp_col_right = st.columns(2)
                with comp_col_left:
                    st.markdown("#### 📍 2D Kuşbakışı Genel Diziliş (Tüm Oyuncular)")
                    all_form_img = pitch_view.draw_2d_formations(avg_positions_2d, spatial_analytics.player_teams)
                    st.image(cv2.cvtColor(all_form_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                with comp_col_right:
                    st.markdown("#### 🔥 2D Kuşbakışı Genel Maç Isı Haritası")
                    all_heat_img = pitch_view.generate_2d_heatmap(spatial_analytics.get_2d_positions(), sigma=28.0)
                    st.image(cv2.cvtColor(all_heat_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            else:
                target_team_id = 0 if "Takım 0" in team_view_select else 1
                tm = spatial_analytics.get_team_metrics(target_team_id)
                team_name = "🔴 Takım 0" if target_team_id == 0 else "🔵 Takım 1"

                # Team Metrics Row
                t_kpi1, t_kpi2, t_kpi3, t_kpi4, t_kpi5 = st.columns(5)
                with t_kpi1:
                    st.metric("Aktif Oyuncular", f"{tm['player_count']} Oyuncu")
                with t_kpi2:
                    st.metric("Takım Ağırlık Merkezi (Centroid)", f"{tm['centroid_m'][0]}m, {tm['centroid_m'][1]}m")
                with t_kpi3:
                    st.metric("Kompaktlık (Boy)", f"{tm['team_depth_m']} metre")
                with t_kpi4:
                    st.metric("Takım Eni (Genişlik)", f"{tm['team_width_m']} metre")
                with t_kpi5:
                    st.metric("Toplam Takım Mesafesi", f"{(tm['total_distance_m'] / 1000):.2f} km")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                team_col_left, team_col_right = st.columns(2)
                with team_col_left:
                    st.markdown(f"#### 📍 {team_name} 2D Kuşbakışı Dizilişi & Kompaktlık")
                    st.caption("Takım oyuncularının ortalama pozisyonları, taktiksel alan kaplama poligonu ve Takım Ağırlık Merkezi (Centroid).")
                    team_form_img = pitch_view.draw_2d_team_formation(
                        target_team_id,
                        avg_positions_2d,
                        spatial_analytics.player_teams
                    )
                    st.image(cv2.cvtColor(team_form_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                with team_col_right:
                    st.markdown(f"#### 🔥 {team_name} 2D Kuşbakışı Toplu Isı Haritası")
                    st.caption("Bu takımdaki tüm oyuncuların pozisyon verilerinin toplanmasıyla oluşan kolektif saha yoğunluğu.")
                    team_heat_img = pitch_view.generate_2d_heatmap(
                        spatial_analytics.get_2d_positions(team_id=target_team_id),
                        sigma=28.0
                    )
                    st.image(cv2.cvtColor(team_heat_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown("---")
                
                # Attack Channel & Pitch Thirds
                st.markdown(f"#### 📊 {team_name} Hücum Koridoru ve Bölge Tercihleri")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### ↔️ Enine Koridor Dağılımı (Kanat / Merkez Tercihi)")
                    st.write(f"- **Sol Koridor:** `{tm['left_flank_pct']}%`")
                    st.write(f"- **Merkez Kanal:** `{tm['center_pct']}%`")
                    st.write(f"- **Sağ Koridor:** `{tm['right_flank_pct']}%`")
                with c2:
                    st.markdown("##### ↕️ Boyuna Bölge Dağılımı (Savunma / Hücum Ağırlığı)")
                    st.write(f"- **Savunma (1. Bölge):** `{tm['defensive_pct']}%`")
                    st.write(f"- **Orta Saha (2. Bölge):** `{tm['middle_pct']}%`")
                    st.write(f"- **Hücum (3. Bölge):** `{tm['attacking_pct']}%`")

    # -------------------------------------------------------------
    # TAB 3: MAÇ YAYINI & HAREKET İZLERİ (VIDEO REPLAY)
    # -------------------------------------------------------------
    with tab_video:
        st.subheader("İşlenmiş Taktik Maç Videosu")
        st.markdown(
            "🔴 **Takım 0:** Kırmızı Kuyruk İzi | 🔵 **Takım 1:** Mavi Kuyruk İzi | ⚪ **Top:** Beyaz Uçuş Çizgisi | 🟨 **Hakem:** Sarı Elips"
        )
        
        video_path = results.get("output_video_path")
        if video_path and os.path.exists(video_path):
            with open(video_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
        else:
            st.warning("İşlenmiş video dosyası henüz üretilmedi veya bulunamadı.")

    # -------------------------------------------------------------
    # TAB 4: KAPSAMLI VERİ TABLOSU & DIŞA AKTARMA (DATA REPORT)
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("Maç Mekansal Oyuncu Raporu (Dataset)")
        st.caption("Her oyuncunun ID'si, takımı, sahada kalma oranı, koştuğu tahmini mesafe, ortalama hızı ve taktik bölgesi.")
        
        if spatial_analytics is not None:
            summary_data = spatial_analytics.get_player_summary()
            if summary_data:
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Veri Setini İndir (CSV)",
                    data=csv_data,
                    file_name="footballistic_player_analytics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
else:
    st.info("👈 Analizi başlatmak için sol menüden video seçin ve **'Analizi Başlat'** butonuna tıklayın!")
