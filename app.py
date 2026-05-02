import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import scipy
from scipy.spatial.distance import euclidean
import sys

# --- PARCHE DEFINITIVO PARA STLITE/PYODIDE ---
# Bloqueamos pyarrow para que Plotly/Narwhals no intenten usarlo.
sys.modules['pyarrow'] = None

# --- DATOS EMBEBIDOS ---
# Datos integrados directamente para evitar errores de caché y lectura de archivos
RAW_DATA = [
    {"Nombre": "Vinícius Jr", "Edad": 24, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 200, "Goles": 18, "Asistencias": 12, "Pases_%": 82.5, "Regates": 165, "Recuperaciones": 65, "Duelos_Aereos": 15, "xG": 16.2, "Potencial": 94, "Coord_X_Media": 82, "Coord_Y_Media": 78},
    {"Nombre": "Erling Haaland", "Edad": 24, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 200, "Goles": 38, "Asistencias": 5, "Pases_%": 76.4, "Regates": 25, "Recuperaciones": 20, "Duelos_Aereos": 82, "xG": 32.4, "Potencial": 95, "Coord_X_Media": 88, "Coord_Y_Media": 50},
    {"Nombre": "Jude Bellingham", "Edad": 21, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 180, "Goles": 20, "Asistencias": 10, "Pases_%": 88.9, "Regates": 90, "Recuperaciones": 115, "Duelos_Aereos": 48, "xG": 17.1, "Potencial": 97, "Coord_X_Media": 72, "Coord_Y_Media": 52},
    {"Nombre": "Kylian Mbappé", "Edad": 25, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 180, "Goles": 30, "Asistencias": 11, "Pases_%": 84.8, "Regates": 140, "Recuperaciones": 50, "Duelos_Aereos": 22, "xG": 26.5, "Potencial": 95, "Coord_X_Media": 82, "Coord_Y_Media": 65},
    {"Nombre": "Phil Foden", "Edad": 24, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 150, "Goles": 22, "Asistencias": 14, "Pases_%": 89.2, "Regates": 110, "Recuperaciones": 75, "Duelos_Aereos": 30, "xG": 19.4, "Potencial": 94, "Coord_X_Media": 74, "Coord_Y_Media": 70},
    {"Nombre": "Bukayo Saka", "Edad": 23, "Equipo": "Arsenal", "Liga": "Premier League", "Valor_Mercado": 140, "Goles": 18, "Asistencias": 15, "Pases_%": 83.1, "Regates": 105, "Recuperaciones": 70, "Duelos_Aereos": 28, "xG": 15.8, "Potencial": 93, "Coord_X_Media": 76, "Coord_Y_Media": 22},
    {"Nombre": "Florian Wirtz", "Edad": 21, "Equipo": "Bayer Leverkusen", "Liga": "Bundesliga", "Valor_Mercado": 130, "Goles": 14, "Asistencias": 18, "Pases_%": 87.5, "Regates": 125, "Recuperaciones": 85, "Duelos_Aereos": 25, "xG": 12.4, "Potencial": 96, "Coord_X_Media": 70, "Coord_Y_Media": 55},
    {"Nombre": "Jamal Musiala", "Edad": 21, "Equipo": "Bayern Munich", "Liga": "Bundesliga", "Valor_Mercado": 130, "Goles": 12, "Asistencias": 14, "Pases_%": 86.2, "Regates": 155, "Recuperaciones": 90, "Duelos_Aereos": 20, "xG": 11.2, "Potencial": 96, "Coord_X_Media": 72, "Coord_Y_Media": 60},
    {"Nombre": "Lamine Yamal", "Edad": 17, "Equipo": "FC Barcelona", "Liga": "La Liga", "Valor_Mercado": 120, "Goles": 8, "Asistencias": 12, "Pases_%": 85.1, "Regates": 180, "Recuperaciones": 55, "Duelos_Aereos": 12, "xG": 9.4, "Potencial": 99, "Coord_X_Media": 78, "Coord_Y_Media": 82},
    {"Nombre": "Rodri", "Edad": 28, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 120, "Goles": 9, "Asistencias": 10, "Pases_%": 94.2, "Regates": 40, "Recuperaciones": 195, "Duelos_Aereos": 65, "xG": 7.2, "Potencial": 92, "Coord_X_Media": 48, "Coord_Y_Media": 50},
    {"Nombre": "Harry Kane", "Edad": 31, "Equipo": "Bayern Munich", "Liga": "Bundesliga", "Valor_Mercado": 100, "Goles": 36, "Asistencias": 12, "Pases_%": 85.4, "Regates": 35, "Recuperaciones": 50, "Duelos_Aereos": 75, "xG": 29.8, "Potencial": 90, "Coord_X_Media": 84, "Coord_Y_Media": 52},
    {"Nombre": "Cole Palmer", "Edad": 22, "Equipo": "Chelsea", "Liga": "Premier League", "Valor_Mercado": 90, "Goles": 25, "Asistencias": 15, "Pases_%": 84.2, "Regates": 95, "Recuperaciones": 65, "Duelos_Aereos": 20, "xG": 21.5, "Potencial": 94, "Coord_X_Media": 75, "Coord_Y_Media": 40},
    {"Nombre": "Declan Rice", "Edad": 25, "Equipo": "Arsenal", "Liga": "Premier League", "Valor_Mercado": 120, "Goles": 8, "Asistencias": 9, "Pases_%": 91.8, "Regates": 45, "Recuperaciones": 180, "Duelos_Aereos": 72, "xG": 6.1, "Potencial": 93, "Coord_X_Media": 52, "Coord_Y_Media": 48},
    {"Nombre": "Lautaro Martínez", "Edad": 27, "Equipo": "Inter Milan", "Liga": "Serie A", "Valor_Mercado": 110, "Goles": 27, "Asistencias": 6, "Pases_%": 79.2, "Regates": 55, "Recuperaciones": 80, "Duelos_Aereos": 58, "xG": 24.5, "Potencial": 91, "Coord_X_Media": 85, "Coord_Y_Media": 48},
    {"Nombre": "Martin Ødegaard", "Edad": 25, "Equipo": "Arsenal", "Liga": "Premier League", "Valor_Mercado": 110, "Goles": 12, "Asistencias": 14, "Pases_%": 90.5, "Regates": 70, "Recuperaciones": 105, "Duelos_Aereos": 22, "xG": 12.8, "Potencial": 92, "Coord_X_Media": 72, "Coord_Y_Media": 65},
    {"Nombre": "Antoine Griezmann", "Edad": 33, "Equipo": "Atletico Madrid", "Liga": "La Liga", "Valor_Mercado": 25, "Goles": 16, "Asistencias": 16, "Pases_%": 87.1, "Regates": 75, "Recuperaciones": 120, "Duelos_Aereos": 38, "xG": 14.2, "Potencial": 88, "Coord_X_Media": 70, "Coord_Y_Media": 50},
    {"Nombre": "Federico Valverde", "Edad": 26, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 120, "Goles": 10, "Asistencias": 10, "Pases_%": 89.5, "Regates": 85, "Recuperaciones": 150, "Duelos_Aereos": 55, "xG": 9.2, "Potencial": 93, "Coord_X_Media": 65, "Coord_Y_Media": 45},
    {"Nombre": "Rafael Leão", "Edad": 25, "Equipo": "AC Milan", "Liga": "Serie A", "Valor_Mercado": 90, "Goles": 14, "Asistencias": 12, "Pases_%": 80.5, "Regates": 170, "Recuperaciones": 45, "Duelos_Aereos": 28, "xG": 13.1, "Potencial": 92, "Coord_X_Media": 80, "Coord_Y_Media": 85},
    {"Nombre": "Kevin De Bruyne", "Edad": 33, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 45, "Goles": 6, "Asistencias": 20, "Pases_%": 91.2, "Regates": 55, "Recuperaciones": 80, "Duelos_Aereos": 20, "xG": 8.4, "Potencial": 90, "Coord_X_Media": 68, "Coord_Y_Media": 55},
    {"Nombre": "Nico Williams", "Edad": 22, "Equipo": "Athletic Club", "Liga": "La Liga", "Valor_Mercado": 70, "Goles": 10, "Asistencias": 16, "Pases_%": 81.4, "Regates": 160, "Recuperaciones": 60, "Duelos_Aereos": 18, "xG": 11.5, "Potencial": 93, "Coord_X_Media": 82, "Coord_Y_Media": 88},
    {"Nombre": "Pedri", "Edad": 21, "Equipo": "FC Barcelona", "Liga": "La Liga", "Valor_Mercado": 80, "Goles": 5, "Asistencias": 8, "Pases_%": 92.4, "Regates": 80, "Recuperaciones": 135, "Duelos_Aereos": 25, "xG": 6.2, "Potencial": 95, "Coord_X_Media": 62, "Coord_Y_Media": 48},
    {"Nombre": "Gavi", "Edad": 20, "Equipo": "FC Barcelona", "Liga": "La Liga", "Valor_Mercado": 90, "Goles": 4, "Asistencias": 6, "Pases_%": 88.5, "Regates": 70, "Recuperaciones": 165, "Duelos_Aereos": 45, "xG": 5.1, "Potencial": 95, "Coord_X_Media": 58, "Coord_Y_Media": 52},
    {"Nombre": "Aurélien Tchouaméni", "Edad": 24, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 100, "Goles": 3, "Asistencias": 5, "Pases_%": 92.8, "Regates": 30, "Recuperaciones": 175, "Duelos_Aereos": 85, "xG": 2.8, "Potencial": 93, "Coord_X_Media": 42, "Coord_Y_Media": 50},
    {"Nombre": "William Saliba", "Edad": 23, "Equipo": "Arsenal", "Liga": "Premier League", "Valor_Mercado": 80, "Goles": 2, "Asistencias": 2, "Pases_%": 93.5, "Regates": 15, "Recuperaciones": 160, "Duelos_Aereos": 92, "xG": 1.5, "Potencial": 94, "Coord_X_Media": 25, "Coord_Y_Media": 50},
    {"Nombre": "Theo Hernández", "Edad": 27, "Equipo": "AC Milan", "Liga": "Serie A", "Valor_Mercado": 60, "Goles": 6, "Asistencias": 8, "Pases_%": 84.2, "Regates": 110, "Recuperaciones": 110, "Duelos_Aereos": 65, "xG": 5.8, "Potencial": 91, "Coord_X_Media": 55, "Coord_Y_Media": 85},
    {"Nombre": "Bruno Guimarães", "Edad": 26, "Equipo": "Newcastle", "Liga": "Premier League", "Valor_Mercado": 85, "Goles": 8, "Asistencias": 12, "Pases_%": 89.1, "Regates": 75, "Recuperaciones": 170, "Duelos_Aereos": 58, "xG": 7.5, "Potencial": 92, "Coord_X_Media": 55, "Coord_Y_Media": 45},
    {"Nombre": "Victor Osimhen", "Edad": 25, "Equipo": "Napoli", "Liga": "Serie A", "Valor_Mercado": 100, "Goles": 20, "Asistencias": 4, "Pases_%": 74.5, "Regates": 45, "Recuperaciones": 35, "Duelos_Aereos": 88, "xG": 18.4, "Potencial": 93, "Coord_X_Media": 86, "Coord_Y_Media": 50},
    {"Nombre": "Alexander Isak", "Edad": 25, "Equipo": "Newcastle", "Liga": "Premier League", "Valor_Mercado": 75, "Goles": 23, "Asistencias": 5, "Pases_%": 78.2, "Regates": 75, "Recuperaciones": 40, "Duelos_Aereos": 65, "xG": 20.8, "Potencial": 92, "Coord_X_Media": 85, "Coord_Y_Media": 55},
    {"Nombre": "Kobbie Mainoo", "Edad": 19, "Equipo": "Manchester United", "Liga": "Premier League", "Valor_Mercado": 55, "Goles": 4, "Asistencias": 4, "Pases_%": 92.1, "Regates": 85, "Recuperaciones": 105, "Duelos_Aereos": 35, "xG": 4.2, "Potencial": 96, "Coord_X_Media": 52, "Coord_Y_Media": 48},
    {"Nombre": "Endrick", "Edad": 18, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 60, "Goles": 10, "Asistencias": 2, "Pases_%": 79.5, "Regates": 110, "Recuperaciones": 40, "Duelos_Aereos": 45, "xG": 9.8, "Potencial": 98, "Coord_X_Media": 84, "Coord_Y_Media": 52},
    {"Nombre": "Warren Zaïre-Emery", "Edad": 18, "Equipo": "PSG", "Liga": "Ligue 1", "Valor_Mercado": 60, "Goles": 5, "Asistencias": 8, "Pases_%": 91.8, "Regates": 70, "Recuperaciones": 125, "Duelos_Aereos": 32, "xG": 5.5, "Potencial": 96, "Coord_X_Media": 55, "Coord_Y_Media": 52},
    {"Nombre": "Pau Cubarsí", "Edad": 17, "Equipo": "FC Barcelona", "Liga": "La Liga", "Valor_Mercado": 40, "Goles": 0, "Asistencias": 2, "Pases_%": 94.5, "Regates": 25, "Recuperaciones": 95, "Duelos_Aereos": 78, "xG": 0.8, "Potencial": 97, "Coord_X_Media": 32, "Coord_Y_Media": 48},
    {"Nombre": "Savinho", "Edad": 20, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 50, "Goles": 11, "Asistencias": 12, "Pases_%": 82.4, "Regates": 145, "Recuperaciones": 58, "Duelos_Aereos": 15, "xG": 10.2, "Potencial": 93, "Coord_X_Media": 78, "Coord_Y_Media": 85},
    {"Nombre": "Alejandro Grimaldo", "Edad": 29, "Equipo": "Bayer Leverkusen", "Liga": "Bundesliga", "Valor_Mercado": 45, "Goles": 12, "Asistencias": 18, "Pases_%": 86.5, "Regates": 80, "Recuperaciones": 105, "Duelos_Aereos": 30, "xG": 9.8, "Potencial": 88, "Coord_X_Media": 58, "Coord_Y_Media": 88},
    {"Nombre": "Xavi Simons", "Edad": 21, "Equipo": "RB Leipzig", "Liga": "Bundesliga", "Valor_Mercado": 80, "Goles": 10, "Asistencias": 15, "Pases_%": 85.2, "Regates": 130, "Recuperaciones": 80, "Duelos_Aereos": 25, "xG": 12.1, "Potencial": 95, "Coord_X_Media": 74, "Coord_Y_Media": 68},
    {"Nombre": "Mohamed Salah", "Edad": 32, "Equipo": "Liverpool", "Liga": "Premier League", "Valor_Mercado": 55, "Goles": 25, "Asistencias": 14, "Pases_%": 80.1, "Regates": 92, "Recuperaciones": 45, "Duelos_Aereos": 18, "xG": 22.4, "Potencial": 89, "Coord_X_Media": 82, "Coord_Y_Media": 25},
    {"Nombre": "Virgil van Dijk", "Edad": 33, "Equipo": "Liverpool", "Liga": "Premier League", "Valor_Mercado": 30, "Goles": 3, "Asistencias": 2, "Pases_%": 91.5, "Regates": 12, "Recuperaciones": 155, "Duelos_Aereos": 95, "xG": 2.1, "Potencial": 89, "Coord_X_Media": 28, "Coord_Y_Media": 50},
    {"Nombre": "Luis Díaz", "Edad": 27, "Equipo": "Liverpool", "Liga": "Premier League", "Valor_Mercado": 75, "Goles": 12, "Asistencias": 8, "Pases_%": 82.4, "Regates": 155, "Recuperaciones": 55, "Duelos_Aereos": 22, "xG": 11.8, "Potencial": 90, "Coord_X_Media": 82, "Coord_Y_Media": 88},
    {"Nombre": "Bruno Fernandes", "Edad": 30, "Equipo": "Manchester United", "Liga": "Premier League", "Valor_Mercado": 70, "Goles": 14, "Asistencias": 15, "Pases_%": 79.5, "Regates": 65, "Recuperaciones": 110, "Duelos_Aereos": 32, "xG": 13.5, "Potencial": 89, "Coord_X_Media": 68, "Coord_Y_Media": 55},
    {"Nombre": "Alexis Mac Allister", "Edad": 25, "Equipo": "Liverpool", "Liga": "Premier League", "Valor_Mercado": 75, "Goles": 7, "Asistencias": 9, "Pases_%": 88.2, "Regates": 72, "Recuperaciones": 145, "Duelos_Aereos": 45, "xG": 6.8, "Potencial": 91, "Coord_X_Media": 58, "Coord_Y_Media": 50},
    {"Nombre": "Bernardo Silva", "Edad": 30, "Equipo": "Manchester City", "Liga": "Premier League", "Valor_Mercado": 70, "Goles": 8, "Asistencias": 12, "Pases_%": 90.5, "Regates": 95, "Recuperaciones": 115, "Duelos_Aereos": 25, "xG": 7.9, "Potencial": 89, "Coord_X_Media": 65, "Coord_Y_Media": 65},
    {"Nombre": "Vitinha", "Edad": 24, "Equipo": "PSG", "Liga": "Ligue 1", "Valor_Mercado": 55, "Goles": 7, "Asistencias": 8, "Pases_%": 91.2, "Regates": 88, "Recuperaciones": 130, "Duelos_Aereos": 28, "xG": 5.4, "Potencial": 92, "Coord_X_Media": 58, "Coord_Y_Media": 52},
    {"Nombre": "Joao Neves", "Edad": 20, "Equipo": "PSG", "Liga": "Ligue 1", "Valor_Mercado": 60, "Goles": 3, "Asistencias": 6, "Pases_%": 92.5, "Regates": 75, "Recuperaciones": 160, "Duelos_Aereos": 40, "xG": 2.5, "Potencial": 96, "Coord_X_Media": 52, "Coord_Y_Media": 50},
    {"Nombre": "Arda Güler", "Edad": 19, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 45, "Goles": 6, "Asistencias": 4, "Pases_%": 88.1, "Regates": 95, "Recuperaciones": 45, "Duelos_Aereos": 15, "xG": 4.8, "Potencial": 97, "Coord_X_Media": 75, "Coord_Y_Media": 65},
    {"Nombre": "Dušan Vlahović", "Edad": 24, "Equipo": "Juventus", "Liga": "Serie A", "Valor_Mercado": 65, "Goles": 22, "Asistencias": 4, "Pases_%": 74.1, "Regates": 42, "Recuperaciones": 25, "Duelos_Aereos": 72, "xG": 19.2, "Potencial": 91, "Coord_X_Media": 86, "Coord_Y_Media": 50},
    {"Nombre": "Nicolò Barella", "Edad": 27, "Equipo": "Inter Milan", "Liga": "Serie A", "Valor_Mercado": 80, "Goles": 5, "Asistencias": 10, "Pases_%": 87.4, "Regates": 82, "Recuperaciones": 140, "Duelos_Aereos": 35, "xG": 5.9, "Potencial": 90, "Coord_X_Media": 62, "Coord_Y_Media": 48},
    {"Nombre": "Alphonso Davies", "Edad": 23, "Equipo": "Bayern Munich", "Liga": "Bundesliga", "Valor_Mercado": 50, "Goles": 2, "Asistencias": 6, "Pases_%": 86.2, "Regates": 145, "Recuperaciones": 95, "Duelos_Aereos": 55, "xG": 2.8, "Potencial": 92, "Coord_X_Media": 60, "Coord_Y_Media": 92},
    {"Nombre": "Alessandro Bastoni", "Edad": 25, "Equipo": "Inter Milan", "Liga": "Serie A", "Valor_Mercado": 70, "Goles": 1, "Asistencias": 4, "Pases_%": 90.8, "Regates": 35, "Recuperaciones": 135, "Duelos_Aereos": 82, "xG": 1.2, "Potencial": 93, "Coord_X_Media": 32, "Coord_Y_Media": 55},
    {"Nombre": "Eduardo Camavinga", "Edad": 21, "Equipo": "Real Madrid", "Liga": "La Liga", "Valor_Mercado": 100, "Goles": 2, "Asistencias": 4, "Pases_%": 91.5, "Regates": 95, "Recuperaciones": 165, "Duelos_Aereos": 58, "xG": 2.4, "Potencial": 95, "Coord_X_Media": 55, "Coord_Y_Media": 60},
    {"Nombre": "Trent Alexander-Arnold", "Edad": 26, "Equipo": "Liverpool", "Liga": "Premier League", "Valor_Mercado": 70, "Goles": 3, "Asistencias": 12, "Pases_%": 79.8, "Regates": 62, "Recuperaciones": 105, "Duelos_Aereos": 28, "xG": 4.5, "Potencial": 91, "Coord_X_Media": 48, "Coord_Y_Media": 88},
]

# --- CARGA DE DATOS ---
def load_data():
    return pd.DataFrame(RAW_DATA)

df_raw = load_data()

if df_raw is not None:
    # Métricas para análisis
    metrics = ['Goles', 'Asistencias', 'Pases_%', 'Regates', 'Recuperaciones', 'Duelos_Aereos', 'xG']
    
    # Normalización para algoritmos
    scaler = MinMaxScaler()
    df_normalized = df_raw.copy()
    df_normalized[metrics] = scaler.fit_transform(df_raw[metrics])

    # --- CLUSTERING TÁCTICO (K-Means) ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_raw['Cluster_ID'] = kmeans.fit_predict(df_normalized[metrics])
    roles = {0: "Especialista Defensivo", 1: "Organizador de Juego", 2: "Finalizador de Área", 3: "Extremo Desequilibrante"}
    df_raw['Rol_Tactico'] = df_raw['Cluster_ID'].map(roles)

    # --- SIDEBAR: FILTROS ---
    st.sidebar.title("Elite Scouting System")
    st.sidebar.markdown("---")
    
    player_list = sorted(df_raw['Nombre'].unique())
    target_player = st.sidebar.selectbox("Jugador Objetivo", player_list)
    
    st.sidebar.subheader("Parámetros de Búsqueda")
    search_mode = st.sidebar.radio("Prioridad de Búsqueda", ["Similitud Actual", "Potencial Juvenil"])
    
    max_age = st.sidebar.slider("Límite de Edad", 16, 40, 21 if search_mode == "Potencial Juvenil" else 35)
    budget = st.sidebar.number_input("Presupuesto Máximo (M€)", 0, 250, 200)

    # Datos del Jugador Objetivo
    target_data = df_raw[df_raw['Nombre'] == target_player].iloc[0]
    target_norm = df_normalized[df_normalized['Nombre'] == target_player].iloc[0]

    # --- CABECERA ---
    st.title("Panel de Análisis de Scouting")
    st.markdown(f"Análisis detallado de **{target_player}** y búsqueda de candidatos compatibles.")
    
    # Ajustamos las proporciones para dar más espacio al Rol
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.metric("Equipo", target_data['Equipo'])
    c2.metric("Edad", f"{target_data['Edad']} años")
    c3.metric("Valor", f"{target_data['Valor_Mercado']} M€")
    c4.metric("Rol Táctico", target_data['Rol_Tactico'])

    st.markdown("---")

    # --- VISUALIZACIÓN ANALÍTICA ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Perfil Estadístico (Radar)")
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=target_norm[metrics].values.tolist(),
            theta=metrics,
            fill='toself',
            name=target_player,
            line_color='#1f77b4'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.subheader("Posicionamiento Táctico")
        # Simulación de campo de fútbol
        fig_pitch = px.scatter(
            target_data.to_frame().T,
            x='Coord_X_Media', y='Coord_Y_Media',
            range_x=[0, 100], range_y=[0, 100],
            labels={'Coord_X_Media': 'Profundidad', 'Coord_Y_Media': 'Amplitud'}
        )
        fig_pitch.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line_color="black", fillcolor="green", opacity=0.1)
        fig_pitch.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line_color="black")
        fig_pitch.update_traces(marker=dict(size=15, color='red', symbol='cross'))
        fig_pitch.update_layout(height=400)
        st.plotly_chart(fig_pitch, use_container_width=True)

    # --- MOTOR DE BÚSQUEDA ---
    st.subheader("Candidatos Identificados")
    
    # Filtrado inicial
    candidates = df_raw[
        (df_raw['Nombre'] != target_player) &
        (df_raw['Edad'] <= max_age) &
        (df_raw['Valor_Mercado'] <= budget)
    ].copy()

    if not candidates.empty:
        # Cálculo de Distancia
        target_vec = target_norm[metrics].values.astype(float)
        
        def calculate_sim(name):
            vec = df_normalized[df_normalized['Nombre'] == name][metrics].values[0].astype(float)
            return euclidean(target_vec, vec)
        
        candidates['Indice_Similitud'] = candidates['Nombre'].apply(calculate_sim)
        
        # Moneyball: Detección de Infravalorados
        candidates['Eficiencia_Coste'] = (candidates['Goles'] + candidates['Asistencias']) / (candidates['Valor_Mercado'] + 1)
        mean_eff = candidates['Eficiencia_Coste'].mean()
        candidates['Oportunidad_Mercado'] = candidates['Eficiencia_Coste'] > (mean_eff * 1.5)

        # Ordenación según modo
        if search_mode == "Potencial Juvenil":
            results = candidates.sort_values(by=['Potencial', 'Indice_Similitud'], ascending=[False, True]).head(5)
        else:
            results = candidates.sort_values(by='Indice_Similitud').head(5)

        # Presentación de Resultados
        def style_rows(row):
            return ['background-color: #e6f3ff' if row['Oportunidad_Mercado'] else '' for _ in row]

        cols_to_show = ['Nombre', 'Equipo', 'Edad', 'Valor_Mercado', 'Goles', 'Asistencias', 'Potencial', 'Rol_Tactico', 'Indice_Similitud', 'Oportunidad_Mercado']

        st.dataframe(
            results[cols_to_show].style.apply(style_rows, axis=1).hide(['Oportunidad_Mercado'], axis=1),
            use_container_width=True
        )
        st.caption("Nota: Las filas resaltadas en azul indican oportunidades de mercado basadas en rendimiento/coste.")

        # --- COMPARATIVA DETALLADA ---
        st.markdown("---")
        selected_cand = st.selectbox("Seleccione candidato para comparativa técnica", results['Nombre'])
        
        if selected_cand:
            c_norm = df_normalized[df_normalized['Nombre'] == selected_cand].iloc[0]
            c_raw = df_raw[df_raw['Nombre'] == selected_cand].iloc[0]
            
            comp1, comp2 = st.columns(2)
            with comp1:
                fig_radar_comp = go.Figure()
                fig_radar_comp.add_trace(go.Scatterpolar(r=target_norm[metrics].values.tolist(), theta=metrics, fill='toself', name=target_player))
                fig_radar_comp.add_trace(go.Scatterpolar(r=c_norm[metrics].values.tolist(), theta=metrics, fill='toself', name=selected_cand))
                fig_radar_comp.update_layout(title="Comparativa de Habilidades", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                st.plotly_chart(fig_radar_comp, use_container_width=True)
            
            with comp2:
                # Proyección (Misión 1)
                st.write(f"### Proyección de Crecimiento: {selected_cand}")
                years = [2024, 2025, 2026, 2027]
                start_val = float(c_raw['Goles'])
                target_val = float((c_raw['Potencial'] / 100) * 30)
                growth = np.linspace(start_val, target_val, 4).tolist()
                
                fig_line = px.line(x=years, y=growth, markers=True, labels={'x': 'Año', 'y': 'Goles Proyectados'})
                fig_line.add_hline(y=target_data['Goles'], line_dash="dash", annotation_text=f"Nivel {target_player}")
                st.plotly_chart(fig_line, use_container_width=True)

    else:
        st.warning("No se han encontrado candidatos con los criterios actuales.")

else:
    st.info("A la espera de datos...")
