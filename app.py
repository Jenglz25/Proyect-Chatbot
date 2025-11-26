# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

from modules.retriever import Retriever
from modules.llm_engine import LLMEngine
from modules.chatbot import Chatbot
from modules import analytics

# -------------------------------------------
# CONFIG
# -------------------------------------------
CSV_PATH = "data/AI_Impact_on_Jobs_2030.csv"
MODEL_PATH = "models/llama-2-7b-32k-instruct.Q4_K_S.gguf"

st.set_page_config(
    layout="wide",
    page_title="AI Jobs Dashboard + Chatbot",
)

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    return df

df = load_data()


# ================================================================
# COLUMN DETECTION
# ================================================================
def find_column(possible, cols):
    cols_lower = {c.lower(): c for c in cols}
    for name in possible:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

col_job    = find_column(["job_title", "job title", "job"], df.columns)
col_salary = find_column(["average_salary", "salary"], df.columns)
col_ai     = find_column(["ai_exposure", "ai_exposure_index"], df.columns)

if col_ai is None:
    st.sidebar.warning("⚠ No existe columna de AI Exposure en este dataset.")

skill_cols = [c for c in df.columns if c.lower().startswith("skill_")]


# ================================================================
# CHAT SYSTEM
# ================================================================
@st.cache_resource
def init_chat():
    retr = Retriever()

    # pick best text column
    text_lengths = {}
    for col in df.columns:
        try:
            text_lengths[col] = df[col].astype(str).str.len().sum()
        except:
            text_lengths[col] = 0

    text_col = max(text_lengths, key=text_lengths.get)

    retr.build_index(df[text_col].astype(str).tolist())
    llm = LLMEngine(model_path=MODEL_PATH)
    return Chatbot(retr, llm, df)


bot = init_chat()


# ================================================================
# PANEL DINÁMICO: ANALÍTICA POR TRABAJO SELECCIONADO (GLOBAL)
# ================================================================
# Función movida a un ámbito global para evitar IndentationError
def show_job_dashboard(job_name):
    st.markdown(f"## 📊 Analítica para: **{job_name}**")

    # Filtrar registros del job
    df_job = df[df[col_job].str.contains(job_name, case=False, na=False)]

    if df_job.empty:
        st.warning("No se encontraron registros para este trabajo.")
        return

    # ----------------- MÉTRICAS -----------------
    st.subheader("📌 Métricas principales")

    c1, c2, c3 = st.columns(3)

    try:
        c1.metric("Salario promedio", f"${df_job[col_salary].astype(float).mean():,.0f}")
    except:
        c1.metric("Salario promedio", "N/A")

    try:
        c2.metric("AI Exposure", f"{df_job[col_ai].astype(float).mean():.2f}")
    except:
        c2.metric("AI Exposure", "N/A")

    c3.metric("Registros encontrados", len(df_job))


    # ----------------- TREEMAP -----------------
    if "Education_Level" in df.columns:
        st.subheader("🌳 Treemap (Job → Educación)")
        try:
            fig_tree = px.treemap(
                df_job,
                path=["Education_Level", col_job],
                values=col_salary,
                color=col_salary,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_tree, width='stretch')
        except:
            st.info("No se pudo generar el treemap por falta de datos.")


    # ----------------- RADAR CHART (Skills) -----------------
    if skill_cols:
        st.subheader("📈 Radar Chart de Skills")

        radar_df = df_job[skill_cols].astype(float).mean()

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df.values,
            theta=radar_df.index,
            fill='toself',
            name=job_name
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig_radar, width='stretch')


    # ----------------- GAUGE (AI Exposure) -----------------
    if col_ai:
        st.subheader("🎯 Gauge — AI Exposure")

        try:
            avg_ai = df_job[col_ai].astype(float).mean()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_ai,
                title={"text": "AI Exposure"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "#8fd19e"},
                        {"range": [0.3, 0.6], "color": "#ffd95e"},
                        {"range": [0.6, 1], "color": "#ff6b6b"}
                    ],
                },
            ))
            st.plotly_chart(fig_gauge, width='stretch')
        except:
            st.info("No se pudo generar el gauge de AI Exposure.")


    # ----------------- SALARIOS -----------------
    if col_salary:
        st.subheader("💰 Distribución de salarios (Job específico)")

        try:
            # Reutiliza analytics.bar_salary (asumiendo que funciona)
            st.plotly_chart(
                analytics.bar_salary(df_job, col_job, col_salary, n=min(10, len(df_job))),
                width='stretch'
            )
        except:
            st.write("Datos de salario insuficientes.")


    # ----------------- TABLA COMPLETA -----------------
    st.subheader("📋 Registros completos del Job")
    st.dataframe(df_job, width='stretch')


# ================================================================
# PAGE NAVIGATION (PÁGINAS)
# ================================================================
menu = st.sidebar.radio(
    "📌 Navegación",
    ["Dashboard", "Chatbot", "Dataset", "Acerca de"],
    index=0
)

st.sidebar.markdown("---")


# ================================================================
# SHARED FILTERS (for Dashboard)
# ================================================================
if menu == "Dashboard":

    st.sidebar.title("Filtros Avanzados")

    # Filter: education
    if "Education_Level" in df.columns:
        selected_edu = st.sidebar.multiselect(
            "Educación",
            sorted(df["Education_Level"].unique()),
            default=None
        )
    else:
        selected_edu = None

    # Filter: salary range
    # Se añade verificación de col_salary
    if col_salary and not df[col_salary].empty:
        # Asegurarse de que las columnas son float para min/max
        df[col_salary] = pd.to_numeric(df[col_salary], errors='coerce').fillna(0)
        salary_min_default = float(df[col_salary].min())
        salary_max_default = float(df[col_salary].max())
        
        salary_min, salary_max = st.sidebar.slider(
            "Rango salarial",
            salary_min_default,
            salary_max_default,
            (salary_min_default, salary_max_default)
        )
    else:
        st.sidebar.warning("⚠ No existe columna de salario o está vacía.")
        salary_min, salary_max = 0, 0


    # Filter: skill value
    if skill_cols:
        selected_skill = st.sidebar.selectbox("Skill a filtrar", skill_cols)
        min_skill = st.sidebar.slider(f"Mínimo valor de {selected_skill}", 0.0, 1.0, 0.2, 0.05)
    else:
        selected_skill = None


    # APPLY FILTERS
    df_f = df.copy()

    if selected_edu:
        df_f = df_f[df_f["Education_Level"].isin(selected_edu)]
    
    # Aplicar filtros de salario solo si la columna existe
    if col_salary and salary_max > 0:
        df_f[col_salary] = pd.to_numeric(df_f[col_salary], errors='coerce').fillna(0)
        df_f = df_f[(df_f[col_salary] >= salary_min) & (df_f[col_salary] <= salary_max)]

    # Aplicar filtros de skill solo si selected_skill existe
    if selected_skill:
        df_f[selected_skill] = pd.to_numeric(df_f[selected_skill], errors='coerce').fillna(0)
        df_f = df_f[df_f[selected_skill] >= min_skill]



# ================================================================
# PAGE 1: DASHBOARD
# ================================================================
if menu == "Dashboard":

    st.title("📊 Dashboard — AI Jobs Analytics")

    if df_f.empty:
        st.warning("No hay datos que cumplan los criterios de filtrado.")
    else:
        top_n = st.slider("Top N elementos", 5, 50, 20)

        # --- Treemap
        if "Education_Level" in df.columns:
            st.subheader("Treemap Educación → Trabajo → Salario")
            st.plotly_chart(
                analytics.treemap(df_f, col_job, col_salary),
                width='stretch'
            )
        else:
            st.info("No existe columna 'Education_Level' para el Treemap.")

        # --- AI Exposure
        st.subheader("Top Exposición a IA")
        if col_ai:
            st.plotly_chart(
                analytics.bar_ai_exposure(df_f, col_job, col_ai, top_n),
                width='stretch'
            )
        else:
            st.info("No existe columna de AI Exposure.")

        # --- Salaries
        if col_salary:
            st.subheader("Top Salarios")
            st.plotly_chart(
                analytics.bar_salary(df_f, col_job, col_salary, top_n),
                width='stretch'
            )
        else:
            st.info("No existe columna de Salario.")

        # --- Skills
        if skill_cols:
            st.subheader("Promedio de Skills")
            st.plotly_chart(
                analytics.skills_bar(df_f, skill_cols),
                width='stretch'
            )
        else:
            st.info("No se encontraron columnas de Skills (skill_*).")


        # --- Correlation Heatmap
        st.subheader(" Heatmap de correlaciones ")
        df_num = df_f.select_dtypes(include=[np.number])
        if not df_num.empty and len(df_num.columns) > 1:
            fig_corr = px.imshow(
                df_num.corr(),
                color_continuous_scale="RdBu",
                title="Mapa de correlación"
            )
            st.plotly_chart(fig_corr, width='stretch')
        else:
            st.info("Datos numéricos insuficientes para generar el Heatmap de correlación.")


        # --- Scatter
        if col_ai and col_salary:
            st.subheader("Relación Salario vs AI Exposure")
            fig_sc = px.scatter(
                df_f,
                x=col_ai,
                y=col_salary,
                color="Education_Level" if "Education_Level" in df.columns else None,
                hover_data=[col_job],
                title="Salario vs Exposición a IA"
            )
            st.plotly_chart(fig_sc, width='stretch')

        # --- Skill heatmap per job
        if col_job and skill_cols:
            st.subheader("Mapa de calor por Skills / Trabajo")
            skill_matrix = df_f.set_index(col_job)[skill_cols]
            # Convertir a float antes de la matriz de calor
            skill_matrix = skill_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            if not skill_matrix.empty and len(skill_matrix.columns) > 0 and len(skill_matrix.index) > 0:
                 fig_heat = px.imshow(
                    skill_matrix.head(top_n), # Limitar a Top N para visibilidad
                    aspect="auto",
                    color_continuous_scale="YlGnBu",
                    title=f"Skills por {col_job} (Top {top_n})"
                )
                 st.plotly_chart(fig_heat, width='stretch')
            else:
                st.info("Datos insuficientes o faltantes para el Mapa de Calor de Skills.")


# ================================================================
# PAGE 2: CHATBOT
# ================================================================
elif menu == "Chatbot":

    st.title("🤖 Chatbot Inteligente")

    question = st.text_input("Escribe tu pregunta:")

    if question:
        with st.spinner("Pensando…"):
            answer, retr_docs = bot.ask(question) # Corregida la variable 'retr' a 'retr_docs' por claridad

        st.success("Respuesta generada:")
        st.write(answer)

        st.markdown("### 📄 Documentos utilizados")
        for d in retr_docs:
            st.info(d)


# ================================================================
# PAGE 3: DATASET VIEWER
# ================================================================
elif menu == "Dataset":

    st.title("📚 Explorador del Dataset")

    st.write("Vista previa del dataset:")
    st.dataframe(df, width='stretch')

    st.download_button(
        "Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="AI_Impact_on_Jobs_2030.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("🔎 Analítica por Job Title")

    # Selector de Job Title
    if col_job:
        job_list = sorted(df[col_job].unique())
        selected_job = st.selectbox("Selecciona un Trabajo para ver su Analítica:", job_list)

        if selected_job:
            show_job_dashboard(selected_job) # Llama a la función globalmente definida
    else:
        st.warning("No se encontró la columna de Título de Trabajo.")


# ================================================================
# PAGE 4: ABOUT
# ================================================================
elif menu == "Acerca de":

    st.title("ℹ️ Acerca de la aplicación")

    st.write("""
    **AI Jobs Dashboard + Chatbot** Desarrollado con:
    - Streamlit  
    - Plotly  
    - FAISS  
    - Llama-2-7B (local)  
    - Sentence Transformers  
    """)

    st.info("Desarrollado por Jenn 💫")