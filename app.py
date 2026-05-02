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
# En Python, poner un módulo como None en sys.modules hace que cualquier 'import' falle.
sys.modules['pyarrow'] = None

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Elite Scouting Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("players_data.csv")
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None

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
        # (Goles + Asistencias) ajustado por posición vs Valor
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

        # Definimos las columnas que queremos mostrar (incluyendo la necesaria para el estilo)
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
                # Simulación de crecimiento hacia el potencial
                start_val = float(c_raw['Goles'])
                target_val = float((c_raw['Potencial'] / 100) * 30) # Escala ficticia
                growth = np.linspace(start_val, target_val, 4).tolist()
                
                fig_line = px.line(x=years, y=growth, markers=True, labels={'x': 'Año', 'y': 'Goles Proyectados'})
                fig_line.add_hline(y=target_data['Goles'], line_dash="dash", annotation_text=f"Nivel {target_player}")
                st.plotly_chart(fig_line, use_container_width=True)

    else:
        st.warning("No se han encontrado candidatos con los criterios actuales.")

else:
    st.info("A la espera de datos...")
