import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Configuración básica
# -----------------------------
st.set_page_config(
    page_title="EDA Deportivo Sintético",
    page_icon="🏟️",
    layout="wide"
)

st.title("🏟️ EDA de Datos Sintéticos sobre Deportes")
st.write(
    "Genera un conjunto de datos sintético de temática deportiva y realiza un "
    "Análisis Exploratorio de Datos (EDA) con visualizaciones interactivas."
)

# -----------------------------
# Utilidades
# -----------------------------
@st.cache_data(show_spinner=False)
def generar_base_deportiva(n_muestras: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    deportes = ["Fútbol", "Baloncesto", "Tenis", "Atletismo", "Natación", "Voleibol", "Ciclismo"]
    paises = ["Colombia", "Argentina", "Brasil", "España", "Estados Unidos", "Francia", "Alemania", "Italia"]
    posiciones_futbol = ["Portero", "Defensa", "Mediocampo", "Delantero"]
    posiciones_basket = ["Base", "Escolta", "Alero", "Ala-Pívot", "Pívot"]
    posiciones_genericas = ["Ofensivo", "Defensivo", "Mixto"]

    # Selección de deporte principal por fila
    deporte = rng.choice(deportes, size=n_muestras)

    # Equipo ficticio por deporte (simple)
    equipos_por_deporte = {
        "Fútbol": ["Tiburones", "Águilas", "Leones", "Toros"],
        "Baloncesto": ["Halcones", "Titanes", "Cóndores", "Truenos"],
        "Tenis": ["Team Verde", "Team Azul", "Team Rojo", "Team Negro"],
        "Atletismo": ["Rayo Team", "SpeedX", "PistaMax", "FondoPlus"],
        "Natación": ["Delfines", "Truchas", "Marinos", "Sirenas"],
        "Voleibol": ["Murallas", "Arenas", "RedMax", "Saeta"],
        "Ciclismo": ["Velocistas", "Escaladores", "Rodadores", "Crono"]
    }

    equipos = [random.choice(equipos_por_deporte[d]) for d in deporte]
    pais = rng.choice(paises, size=n_muestras)

    # Posición depende del deporte
    posiciones = []
    for d in deporte:
        if d == "Fútbol":
            posiciones.append(random.choice(posiciones_futbol))
        elif d == "Baloncesto":
            posiciones.append(random.choice(posiciones_basket))
        elif d == "Tenis":
            posiciones.append(random.choice(["Individual", "Dobles"]))
        else:
            posiciones.append(random.choice(posiciones_genericas))

    # Temporadas (años recientes)
    temporada = rng.integers(2018, 2026, size=n_muestras)  # 2018–2025
    # Partidos/competencias (aprox)
    eventos_disputados = np.maximum(1, (rng.normal(30, 10, size=n_muestras)).astype(int))

    # Métricas físicas y de rendimiento
    edad = np.clip((rng.normal(25, 5, size=n_muestras)).round().astype(int), 16, 45)
    estatura_cm = np.clip((rng.normal(178, 12, size=n_muestras)).round().astype(int), 150, 210)
    peso_kg = np.clip((rng.normal(75, 10, size=n_muestras)).round().astype(int), 45, 120)

    # Puntuaciones/estadísticas
    puntuacion = np.clip((rng.normal(70, 15, size=n_muestras)), 0, 100).round(1)
    goles_puntos = np.maximum(0, (rng.poisson(5, size=n_muestras)).astype(int))  # goles/puntos totales aprox
    asistencias = np.maximum(0, (rng.poisson(3, size=n_muestras)).astype(int))
    faltas = np.maximum(0, (rng.poisson(2, size=n_muestras)).astype(int))

    # Económico
    salario_usd = (rng.lognormal(mean=10.5, sigma=0.5, size=n_muestras)).round(0).astype(int)  # ~36K–1000K+

    # Salud
    lesionado = rng.choice(["Sí", "No"], size=n_muestras, p=[0.15, 0.85])

    # Fecha (para líneas de tendencia) -> generamos una fecha aleatoria por temporada
    fechas = []
    for year in temporada:
        month = rng.integers(1, 13)
        day = int(np.clip(rng.integers(1, 29), 1, 28))
        fechas.append(pd.Timestamp(year=int(year), month=int(month), day=day))
    fecha_evento = pd.to_datetime(fechas)

    df = pd.DataFrame({
        "Deporte": deporte,
        "Equipo": equipos,
        "País": pais,
        "Posición": posiciones,
        "Temporada": temporada,
        "FechaEvento": fecha_evento,
        "EventosDisputados": eventos_disputados,
        "Edad": edad,
        "Estatura_cm": estatura_cm,
        "Peso_kg": peso_kg,
        "Puntuación": puntuacion,
        "Goles_Puntos": goles_puntos,
        "Asistencias": asistencias,
        "Faltas": faltas,
        "Salario_USD": salario_usd,
        "Lesionado": lesionado
    })

    # Aseguramos tipos
    cat_cols = ["Deporte", "Equipo", "País", "Posición", "Lesionado"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df


def columnas_por_tipo(df: pd.DataFrame, tipo: str) -> list:
    """
    tipo: 'Cuantitativas' | 'Cualitativas' | 'Mixto'
    """
    numericas = df.select_dtypes(include=[np.number, "datetime64[ns]"]).columns.tolist()
    categoricas = df.select_dtypes(include=["category", "object"]).columns.tolist()
    fechas = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    if tipo == "Cuantitativas":
        return numericas
    elif tipo == "Cualitativas":
        return categoricas
    else:
        # Mixto: preferimos 3 num + 3 cat si existen
        salida = []
        salida.extend(numericas[:3])
        salida.extend(categoricas[:3])
        # Si no hay suficientes, combinamos lo que haya
        if len(salida) == 0:
            salida = df.columns.tolist()
        return salida


# -----------------------------
# Barra lateral (controles)
# -----------------------------
with st.sidebar:
    st.header("🎛️ Controles")
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=42, step=1)
    n_muestras = st.slider("Número de muestras", min_value=10, max_value=500, value=200, step=10)
    n_columnas = st.slider("Número máximo de columnas a usar", min_value=3, max_value=6, value=6, step=1)
    tipo_vars = st.selectbox("Tipo de variables a incluir", ["Mixto", "Cuantitativas", "Cualitativas"])
    st.caption("Puedes ajustar columnas exactas debajo, tras generar.")

    generar = st.button("🔄 Generar/Actualizar Dataset")

# -----------------------------
# Generación del dataset
# -----------------------------
if "df" not in st.session_state or generar:
    st.session_state.df = generar_base_deportiva(n_muestras, seed)

df = st.session_state.df.copy()

st.success(f"Dataset generado con **{len(df)}** filas y **{df.shape[1]}** columnas.")

# Sugerencia de columnas por tipo y selección final
sugeridas = columnas_por_tipo(df, tipo_vars)
columns_max = min(n_columnas, len(df.columns))

seleccion_cols = st.multiselect(
    "Selecciona las columnas a mostrar (hasta el máximo elegido):",
    options=df.columns.tolist(),
    default=sugeridas[:columns_max],
    max_selections=columns_max
)

if not seleccion_cols:
    st.warning("Selecciona al menos una columna para continuar.")
    st.stop()

df_view = df[seleccion_cols].copy()

# -----------------------------
# Filtros rápidos (si hay categóricas)
# -----------------------------
cat_cols_view = df_view.select_dtypes(include=["category", "object"]).columns.tolist()
if cat_cols_view:
    with st.expander("🔎 Filtros rápidos (categóricas)"):
        filtros = {}
        col_f = st.columns(min(3, len(cat_cols_view)))
        for i, c in enumerate(cat_cols_view):
            with col_f[i % len(col_f)]:
                valores = sorted(df_view[c].astype(str).unique())
                seleccion = st.multiselect(f"{c}", options=valores, default=valores)
                filtros[c] = seleccion

        for c, valores in filtros.items():
            df_view = df_view[df_view[c].astype(str).isin(valores)]

# -----------------------------
# Vista de tabla y estadísticas
# -----------------------------
st.subheader("📋 Tabla de datos")
st.dataframe(df_view, use_container_width=True)

with st.expander("📈 Resumen estadístico"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Descriptivas (numéricas)**")
        num_desc = df_view.select_dtypes(include=[np.number])
        if not num_desc.empty:
            st.dataframe(num_desc.describe().T, use_container_width=True)
        else:
            st.info("No hay columnas numéricas en la selección.")

    with col2:
        st.markdown("**Valores faltantes por columna**")
        st.dataframe(df_view.isna().sum().to_frame("nulos"), use_container_width=True)

# -----------------------------
# Visualizaciones
# -----------------------------
st.subheader("📊 Visualizaciones")

tipo_graf = st.selectbox(
    "Tipo de gráfico",
    ["Línea (tendencia)", "Barras", "Dispersión", "Pastel", "Histograma"]
)

# Column helpers
num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_view.select_dtypes(include=["category", "object"]).columns.tolist()
date_cols = df_view.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# Parámetros de gráficos (dinámicos)
if tipo_graf == "Línea (tendencia)":
    eje_x = st.selectbox("Eje X (fecha o numérica)", options=date_cols + num_cols, index=0 if date_cols else 0)
    eje_y = st.selectbox("Eje Y (numérica)", options=num_cols)
    color = st.selectbox("Color (opcional, categórica)", options=["(ninguno)"] + cat_cols, index=0)
    modo = st.radio("Modo de línea", ["lines", "markers", "lines+markers"], horizontal=True)
    if eje_x in date_cols:
        df_plot = df_view.sort_values(by=eje_x)
    else:
        df_plot = df_view.copy()
    fig = px.line(
        df_plot, x=eje_x, y=eje_y,
        color=None if color == "(ninguno)" else color,
        markers=("markers" in modo)
    )
    st.plotly_chart(fig, use_container_width=True)

elif tipo_graf == "Barras":
    x_cat = st.selectbox("Categoría (X)", options=cat_cols if cat_cols else ["(ninguna)"])
    modo_barra = st.radio("¿Qué mostrar?", ["Conteo", "Media de una métrica"], horizontal=True)
    if modo_barra == "Conteo":
        fig = px.bar(df_view, x=x_cat)
    else:
        metrica = st.selectbox("Métrica (numérica)", options=num_cols)
        agg = st.selectbox("Agregación", options=["mean", "sum", "median"])
        df_agg = df_view.groupby(x_cat)[metrica].agg(agg).reset_index()
        fig = px.bar(df_agg, x=x_cat, y=metrica)
    st.plotly_chart(fig, use_container_width=True)

elif tipo_graf == "Dispersión":
    if len(num_cols) < 2:
        st.warning("Se requieren al menos dos columnas numéricas para un scatter.")
    else:
        x = st.selectbox("X (numérica)", options=num_cols)
        y = st.selectbox("Y (numérica)", options=[c for c in num_cols if c != x])
        color = st.selectbox("Color (opcional, categórica)", options=["(ninguno)"] + cat_cols, index=0)
        size = st.selectbox("Tamaño (opcional, numérica)", options=["(ninguno)"] + num_cols, index=0)
        fig = px.scatter(
            df_view, x=x, y=y,
            color=None if color == "(ninguno)" else color,
            size=None if size == "(ninguno)" else size,
            hover_data=seleccion_cols
        )
        st.plotly_chart(fig, use_container_width=True)

elif tipo_graf == "Pastel":
    if not cat_cols:
        st.warning("Necesitas al menos una columna categórica para gráfico de pastel.")
    else:
        names = st.selectbox("Categoría", options=cat_cols)
        modo_pastel = st.radio("Valor", ["Conteo", "Suma de una métrica"], horizontal=True)
        if modo_pastel == "Conteo":
            df_counts = df_view[names].value_counts().reset_index()
            df_counts.columns = [names, "Conteo"]
            fig = px.pie(df_counts, names=names, values="Conteo")
        else:
            metrica = st.selectbox("Métrica (numérica)", options=num_cols)
            df_sum = df_view.groupby(names)[metrica].sum().reset_index()
            fig = px.pie(df_sum, names=names, values=metrica)
        st.plotly_chart(fig, use_container_width=True)

elif tipo_graf == "Histograma":
    if not num_cols:
        st.warning("Necesitas al menos una columna numérica para histograma.")
    else:
        x = st.selectbox("Variable (numérica)", options=num_cols)
        color = st.selectbox("Color (opcional, categórica)", options=["(ninguno)"] + cat_cols, index=0)
        bins = st.slider("Número de bins", min_value=5, max_value=60, value=30, step=1)
        fig = px.histogram(
            df_view, x=x,
            color=None if color == "(ninguno)" else color,
            nbins=bins
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Descarga
# -----------------------------
st.download_button(
    "💾 Descargar CSV",
    data=df_view.to_csv(index=False).encode("utf-8"),
    file_name="dataset_deportivo_sintetico.csv",
    mime="text/csv"
)

st.info(
    "Tip: si necesitas repetir exactamente el mismo dataset, fija la semilla y vuelve a generar. "
    "Puedes cambiar el tipo de variables, las columnas y el tamaño (hasta 500 filas y 6 columnas)."
)
