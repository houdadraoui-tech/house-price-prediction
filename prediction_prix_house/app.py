import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ================== Configuration de la Page ==================
st.set_page_config(
    page_title="Prédiction Prix Immobilier USA",
    layout="centered"
)

# ================== CSS Personnalisé ==================
st.markdown("""
<style>
/* Global background */
.main {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 14px;
    box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.08);
}

/* Title */
h1 {
    text-align: center;
    color: #2563eb;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    border: none;
    margin-top: 20px;
}

/* Result box */
.result-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 12px;
    background-color: #ecfdf5;
    border-left: 6px solid #22c55e;
    text-align: center;
}

.result-title {
    font-size: 20px;
    color: #065f46;
    margin-bottom: 8px;
}

.result-value {
    font-size: 30px;
    font-weight: 700;
    color: #047857;
}
</style>
""", unsafe_allow_html=True)


# ================== Chargement du Modèle ==================
@st.cache_resource
def load_my_model():
    return joblib.load("house_model.pkl")


try:
    model = load_my_model()
except:
    st.error("Erreur : Le fichier 'house_model.pkl' est introuvable.")

# ================== Titre et Texte ==================
st.title("Prédiction du Prix des Maisons")
st.markdown(
    "<div class='subtitle'>Entrez les informations ci-dessous pour estimer le prix du marché.</div>",
    unsafe_allow_html=True
)

# ================== Champs de Saisie (Input Fields) ==================
income = st.number_input("Revenu Moyen de la Zone (Avg. Area Income)", value=60000.0)
age = st.number_input("Âge Moyen des Maisons (Avg. Area House Age)", value=5.0)
rooms = st.number_input("Nombre Moyen de Pièces (Avg. Area Number of Rooms)", value=6.0)
bedrooms = st.number_input("Nombre de Chambres (Avg. Area Number of Bedrooms)", value=3.0)
population = st.number_input("Population de la Zone (Area Population)", value=30000.0)

# ================== Prédiction ==================
if st.button("Prédire le Prix"):
    try:
        columns = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                   'Avg. Area Number of Bedrooms', 'Area Population']
        features = pd.DataFrame([[income, age, rooms, bedrooms, population]], columns=columns)

        prediction = model.predict(features)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">Prix Estimé de la Maison</div>
                <div class="result-value">${prediction[0]:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")