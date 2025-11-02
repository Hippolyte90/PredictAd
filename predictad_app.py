# app.py
import streamlit as st
import tempfile
import os
from pathlib import Path
from main import treat_video_ad
from agents.synth_agent import generate_report, plot_radar
from recommandation import generate_recommendations



# ========== CSS ==========
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================== UI ===================
st.set_page_config(page_title="PredictAd — Analyse pub YouTube", layout="wide")
# load css if exists
if Path("assets/style.css").exists():
    local_css("assets/style.css")
    
st.title("🧩Avoir plus d'Impact avec vos pub Youtube🧩")
st.markdown(" Analysez vos vidéos publicitaires YouTube avec **PredictAd** et obtenez des recommandations pour maximiser leur impact.")    
st.sidebar.title("PredictAd")


st.sidebar.markdown(
    """<div style="background-color:#dcdcdc; padding:12px; border-radius:8px;">
       Choisir une vidéo (.mp4)
    </div>""",
    unsafe_allow_html=True,  
)

# File uploader : ne pas mettre unsafe_allow_html ici
uploaded = st.sidebar.file_uploader(
    label="",  # on laisse le label vide parce que l'UI est faite ci-dessus
    type=["mp4"]
)
if uploaded is None:
    st.sidebar.info("ℹ️ Upload une vidéo pour lancer l'analyse.")
else:
    tmp_dir = tempfile.mkdtemp(prefix="predictad_")
    video_path = os.path.join(tmp_dir, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"✅ Vidéo sauvegardée : {uploaded.name}")
    if st.sidebar.button("Lancer l'analyse"):
        
        with st.spinner("Analyse de la vidéo en cours..."):
            video_scores = treat_video_ad(video_path)

            if video_scores is not None:
                st.success("✅ Analyse terminée !")

                # --- 🔹 Mise en page des scores ---
                st.write("### Résultats des scores vidéo")

                # Création de colonnes pour aligner les blocs
                col1, col2, col3= st.columns(3)

                # Exemple : chaque score affiché dans un petit bloc stylé
                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color:#009fe3;padding:10px;
                        border-radius:8px;text-align:center;color:white;
                        font-weight:bold;height:130px; display:flex;
                        flex-direction:column;justify-content:center; align-items:center;">
                        <h3>🎧 Score Audio</h3> <br><span style="font-size:40px;">{video_scores.get('Audio', 0):.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    

                with col2:
                    st.markdown(
                        f"""
                        <div style="background-color:#009fe3;padding:10px;
                        border-radius:8px;text-align:center;color:white;
                        font-weight:bold; height:130px; display:flex;
                        flex-direction:column;justify-content:center; align-items:center;">
                        <h3>👁️  Score Visuel </h3><br><span style="font-size:40px;">{video_scores.get('Visuel', 0):.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div style="background-color:#009fe3;padding:10px;
                        border-radius:8px;text-align:center;color:white;
                        font-weight:bold;height:130px; display:flex;
                        flex-direction:column;justify-content:center; align-items:center;">
                        <h3>🧠 Score Linguistique </h3><span style="font-size:40px;">{video_scores.get('Linguistique', 0):.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color:#009fe3;padding:10px;
                        border-radius:8px;text-align:center;color:white;
                        font-weight:bold;width:300px; height:130px; display:flex;
                        flex-direction:column;justify-content:center; align-items:center;">
                        <h3>🌍 Score Global </h3><br><span style="font-size:40px;">{video_scores.get('Global', 0):.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("")
                    st.write("### Observation Globale :")
                    recommend = generate_report(video_scores)
                    st.markdown(f"💬**{recommend}**")
                    st.write("### Recommandations :")
                    recom_specific = generate_recommendations(video_scores)
                    for rec in recom_specific.values():
                        st.markdown(f"{rec}")
                with col2:
                    # --- 🔹 Affichage du graphique radar ---
                    st.markdown("### 🕸️ Visualisation radar des scores")

                    fig = plot_radar(video_scores)  # suppose que la fonction sauvegarde une image
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.info("Graphique radar non disponible.")
            else:
                st.error("❌ Impossible de récupérer les scores de la vidéo.")    
                

st.header("Mode d'emploi & Options")