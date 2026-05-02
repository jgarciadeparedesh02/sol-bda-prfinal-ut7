# ⚽ Scouting Dashboard Inteligente - Solución

Esta es la solución propuesta para la **Práctica Final: Construcción de un Scouting Dashboard Inteligente**. La aplicación permite a los directores deportivos identificar jugadores similares (clones estadísticos) basándose en métricas de rendimiento.

## 📋 Requisitos del Proyecto
- **Streamlit**: Para la interfaz interactiva.
- **Pandas**: Para la manipulación de datos.
- **Scikit-Learn**: Para el escalado de métricas (MinMaxScaler).
- **SciPy**: Para el cálculo de distancias euclidianas.
- **Plotly**: Para los gráficos de radar interactivos.

## 🚀 Instrucciones de Instalación

1. **Instalar dependencias**:
   Asegúrate de tener Python instalado y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación**:
   Dentro de la carpeta del proyecto, lanza el comando:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Funcionalidades Implementadas

### A. Gestión de Datos
- Carga de datos optimizada con `@st.cache_data`.
- Normalización automática de métricas (0 a 1) para permitir comparaciones justas entre diferentes escalas (goles vs % pases).

### B. Interfaz de Usuario
- **Sidebar**: Filtros dinámicos por Jugador Objetivo, Edad Máxima y Presupuesto (Valor de Mercado).
- **Dashboard Principal**:
  - KPIs del jugador objetivo usando `st.metric`.
  - Gráfico de radar interactivo para visualizar el perfil de habilidades.

### C. Motor de Similitud
- Algoritmo basado en **Distancia Euclidiana** entre vectores de rendimiento.
- Clasificación de los 5 candidatos más cercanos al perfil del jugador seleccionado.

### D. Toques de Calidad (Extras)
- **Comparador Visual**: Selector para superponer el radar de un candidato sobre el del jugador objetivo.
- **Estilo Dinámico**: Resaltado en verde en la tabla para métricas donde el candidato supera al objetivo.
- **Exportación**: Botón de descarga para obtener el informe en formato CSV.

## 📁 Archivos en este repositorio
- `app.py`: Código principal de la aplicación Streamlit.
- `players_data.csv`: Dataset de ejemplo con jugadores de élite.
- `requirements.txt`: Dependencias necesarias.
- `data_generator.py`: Script opcional para generar datasets más grandes.
