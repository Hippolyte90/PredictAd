import streamlit as st

st.set_page_config(page_title="Interface Demo", layout="wide")

col1, spacer, col2 = st.columns([1, 0.1, 1])

with col1:
    st.subheader("🎬 Analyse Linguistique")
    st.write("Résultats du modèle linguistique : clarté, tonalité, émotion, etc.")

with col2:
    st.subheader("🎨 Analyse Visuelle")
    st.markdown("""
        <div style="background-color:#dcdcdc; padding:25px; border-radius:12px;">
            <p>Score moyen des visuels : <b>0.85</b></p>
            <p>Éléments dominants : rouge, visage humain, logo bien centré.</p>
        </div>
    """, unsafe_allow_html=True)
