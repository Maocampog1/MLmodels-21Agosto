import io
import re
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="App2 - Limpieza y EDA de Datos 'SUCIO→LIMPIO'",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 App2: Limpieza y EDA de Datos (SUCIO → LIMPIO)")
st.caption("Sube tu archivo 'sucio', elige reglas de limpieza y obtén un análisis exploratorio con visualizaciones interactivas.")

# -----------------------------
# Utils
# -----------------------------
def to_snake_case(s: str) -> str:
    s = s.strip()
    s = s.replace("%", "pct").replace("#", "num").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

@st.cache_data(show_spinner=False)
def read_any_file(file) -> pd.DataFrame:
    name = getattr(file, "name", "data")
    if isinstance(file, (str, type(st))):
        name = str(file)
    if str(name).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file, encoding_errors="ignore")

def detect_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return numeric_cols, object_cols + cat_cols, datetime_cols

def try_parse_dates(df: pd.DataFrame, sample_frac: float = 0.4):
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == "object":
            # heurística: si contiene '/', '-', ':' o parece fecha en muestras, intentar parseo
            sample = df2[col].dropna().astype(str).sample(
                min(len(df2[col].dropna()), max(10, int(len(df2)*sample_frac))), random_state=0
            ) if len(df2) > 0 else []
            looks_date = any(("-" in x or "/" in x or ":" in x) for x in sample[:20]) if len(sample)>0 else False
            if looks_date:
                parsed = pd.to_datetime(df2[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
                # si >20% se parsea, lo adoptamos
                if parsed.notna().mean() > 0.2:
                    df2[col] = parsed
    return df2

def clean_strings(df: pd.DataFrame, cols: list, lower=True, trim=True, normalize_accents=True):
    df2 = df.copy()
    for c in cols:
        s = df2[c].astype(str)
        if trim:
            s = s.str.strip()
        if lower:
            s = s.str.lower()
        if normalize_accents:
            s = s.apply(strip_accents)
        df2[c] = s.replace({"nan": np.nan, "none": np.nan, "": np.nan})
    return df2

def remove_duplicates(df: pd.DataFrame, subset=None):
    before = len(df)
    df2 = df.drop_duplicates(subset=subset)
    removed = before - len(df2)
    return df2, removed

def impute_missing(df: pd.DataFrame, strategy_num="median", strategy_cat="most_frequent"):
    df2 = df.copy()
    num_cols, cat_cols, _ = detect_types(df2)

    if num_cols:
        imputer_num = SimpleImputer(strategy=strategy_num)
        df2[num_cols] = imputer_num.fit_transform(df2[num_cols])

    if cat_cols:
        imputer_cat = SimpleImputer(strategy=strategy_cat)
        df2[cat_cols] = imputer_cat.fit_transform(df2[cat_cols])

    return df2

def winsorize_iqr(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower, upper)

def apply_winsorization(df: pd.DataFrame, cols: list, k: float = 1.5):
    df2 = df.copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = winsorize_iqr(df2[c].astype(float), k=k)
    return df2

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
          .sum()
          .rename("n_nulos")
          .to_frame()
          .assign(pct_nulos=lambda s: (s["n_nulos"] / len(df) * 100).round(2))
          .sort_values(by="pct_nulos", ascending=False)
    )

# -----------------------------
# Sidebar: Carga & Config
# -----------------------------
with st.sidebar:
    st.header("📥 Datos de entrada")
    file = st.file_uploader("Sube un .csv o .xlsx", type=["csv", "xlsx", "xls"])
    st.caption("También puedes arrastrar el archivo aquí.")

    st.divider()
    st.header("⚙️ Reglas de limpieza")
    rename_cols = st.checkbox("Renombrar columnas a snake_case", True)
    normalize_strings = st.checkbox("Normalizar texto (trim, lower, sin acentos)", True)
    parse_dates_flag = st.checkbox("Intentar detectar y convertir fechas", True)
    drop_dups = st.checkbox("Eliminar filas duplicadas", True)

    st.write("Imputación de nulos")
    imp_num = st.selectbox("Numéricas", ["median", "mean", "most_frequent"], index=0)
    imp_cat = st.selectbox("Categóricas", ["most_frequent", "constant_(SIN_VALOR)"], index=0)

    wins_flag = st.checkbox("Capar outliers (winsorización por IQR)", False)
    wins_k = st.slider("Intensidad IQR (k)", 1.0, 3.0, 1.5, 0.1)

    st.divider()
    st.header("🔽 Descarga")
    ready_to_download_name = st.text_input("Nombre de archivo limpio", "datos_limpios.csv")

# -----------------------------
# Carga
# -----------------------------
if file is None:
    st.info("Sube un archivo para comenzar. Si no tienes uno, prueba con tu dataset.")
    st.stop()

raw = read_any_file(file)
st.subheader("📄 Vista previa (datos originales)")
st.dataframe(raw.head(50), use_container_width=True)

st.markdown("**Resumen de nulos (original)**")
st.dataframe(missing_summary(raw), use_container_width=True)

# -----------------------------
# Pipeline de limpieza
# -----------------------------
df = raw.copy()

if rename_cols:
    df.columns = [to_snake_case(c) for c in df.columns]

# detectar tipos antes de normalizar
num_cols, cat_cols, date_cols = detect_types(df)

if normalize_strings and cat_cols:
    df = clean_strings(df, cols=cat_cols, lower=True, trim=True, normalize_accents=True)
    # re-detectar tipos por si cambió algo
    num_cols, cat_cols, date_cols = detect_types(df)

if parse_dates_flag:
    df = try_parse_dates(df)
    # re-detectar
    num_cols, cat_cols, date_cols = detect_types(df)

if drop_dups:
    df, removed = remove_duplicates(df)
    if removed > 0:
        st.success(f"Se eliminaron {removed} filas duplicadas.")

# Imputación
if imp_cat.startswith("constant"):
    # reemplazo constante "SIN_VALOR" para categóricas
    df_cat = df[cat_cols].copy() if cat_cols else pd.DataFrame()
    if not df_cat.empty:
        df_cat = df_cat.fillna("SIN_VALOR")
        df[cat_cols] = df_cat

# Imputación con SimpleImputer para el resto
df = impute_missing(df, strategy_num=imp_num, strategy_cat="most_frequent" if imp_cat != "constant_(SIN_VALOR)" else "most_frequent")

if wins_flag and num_cols:
    df = apply_winsorization(df, cols=num_cols, k=wins_k)

st.subheader("✅ Datos limpios (previa)")
st.dataframe(df.head(50), use_container_width=True)

st.markdown("**Resumen de nulos (limpio)**")
st.dataframe(missing_summary(df), use_container_width=True)

st.success(f"Filas: {len(df)} | Columnas: {df.shape[1]}")

# -----------------------------
# EDA
# -----------------------------
st.header("🔍 Análisis Exploratorio (EDA)")

num_cols, cat_cols, date_cols = detect_types(df)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Descriptivas numéricas**")
    if num_cols:
        st.dataframe(df[num_cols].describe().T, use_container_width=True)
    else:
        st.info("No hay columnas numéricas.")

with col_right:
    st.markdown("**Top categorías (si existen)**")
    if cat_cols:
        target_cat = st.selectbox("Columna categórica", options=cat_cols)
        topk = st.slider("Top K", 3, 30, 10, 1)
        vc = df[target_cat].value_counts(dropna=False).head(topk).rename_axis(target_cat).reset_index(name="conteo")
        st.dataframe(vc, use_container_width=True)
    else:
        st.info("No hay columnas categóricas.")

st.subheader("📊 Visualizaciones interactivas")
tipo = st.selectbox("Tipo de gráfico", ["Línea", "Barras", "Dispersión", "Histograma", "Pastel", "Caja (Box)"])

if tipo == "Línea":
    x_options = date_cols + num_cols
    if not x_options or not num_cols:
        st.warning("Requiere al menos 1 eje X (fecha/num) y 1 Y (num).")
    else:
        x = st.selectbox("Eje X (fecha/num)", x_options)
        y = st.selectbox("Eje Y (num)", num_cols)
        color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, index=0)
        dff = df.sort_values(by=x) if x in date_cols else df
        fig = px.line(dff, x=x, y=y, color=None if color=="(ninguno)" else color, markers=True)
        st.plotly_chart(fig, use_container_width=True)

elif tipo == "Barras":
    if not cat_cols:
        st.warning("Necesitas al menos una columna categórica.")
    else:
        x = st.selectbox("Categoría (X)", cat_cols)
        modo = st.radio("Mostrar", ["Conteo", "Media de una métrica"], horizontal=True)
        if modo == "Conteo":
            fig = px.bar(df, x=x)
        else:
            if not num_cols:
                st.warning("Se requiere una métrica numérica.")
            else:
                metrica = st.selectbox("Métrica", num_cols)
                agg = st.selectbox("Agregación", ["mean", "sum", "median"])
                dfa = df.groupby(x, dropna=False)[metrica].agg(agg).reset_index()
                fig = px.bar(dfa, x=x, y=metrica)
        st.plotly_chart(fig, use_container_width=True)

elif tipo == "Dispersión":
    if len(df.select_dtypes(include=[np.number]).columns) < 2:
        st.warning("Se requieren al menos 2 numéricas.")
    else:
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", [c for c in num_cols if c != x])
        color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, index=0)
        size = st.selectbox("Tamaño (opcional)", ["(ninguno)"] + num_cols, index=0)
        fig = px.scatter(
            df, x=x, y=y,
            color=None if color=="(ninguno)" else color,
            size=None if size=="(ninguno)" else size,
            hover_data=df.columns
        )
        st.plotly_chart(fig, use_container_width=True)

elif tipo == "Histograma":
    if not num_cols:
        st.warning("Necesitas al menos una numérica.")
    else:
        x = st.selectbox("Variable", num_cols)
        color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, index=0)
        bins = st.slider("Bins", 5, 80, 30, 1)
        fig = px.histogram(df, x=x, color=None if color=="(ninguno)" else color, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)

elif tipo == "Pastel":
    if not cat_cols:
        st.warning("Necesitas al menos una categórica.")
    else:
        names = st.selectbox("Categoría", cat_cols)
        modo = st.radio("Valor", ["Conteo", "Suma de métrica"], horizontal=True)
        if modo == "Conteo":
            dfa = df[names].value_counts(dropna=False).rename_axis(names).reset_index(name="conteo")
            fig = px.pie(dfa, names=names, values="conteo")
        else:
            if not num_cols:
                st.warning("Se requiere una métrica numérica.")
            else:
                metrica = st.selectbox("Métrica", num_cols)
                dfa = df.groupby(names, dropna=False)[metrica].sum().reset_index()
                fig = px.pie(dfa, names=names, values=metrica)
        st.plotly_chart(fig, use_container_width=True)

elif tipo == "Caja (Box)":
    if not num_cols or not cat_cols:
        st.warning("Requiere al menos una numérica y una categórica.")
    else:
        y = st.selectbox("Métrica (Y)", num_cols)
        x = st.selectbox("Categoría (X)", cat_cols)
        color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, index=0)
        fig = px.box(df, x=x, y=y, color=None if color=="(ninguno)" else color, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Descarga del dataset limpio
# -----------------------------
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Descargar CSV limpio",
    data=csv_bytes,
    file_name=ready_to_download_name,
    mime="text/csv"
)

st.caption("Consejo: ajusta las reglas en la barra lateral y observa cómo cambian la tabla y las métricas.")
