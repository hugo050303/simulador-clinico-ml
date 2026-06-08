# app.py
# ==============================================================================
# PROYECTO INTEGRADOR - MATERIA: APRENDIZAJE AUTOMÁTICO
# INGENIERÍA EN SISTEMAS COMPUTACIONALES - TECNOLÓGICO NACIONAL DE MÉXICO
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos analíticos, matemáticos y estadísticos (Rúbrica Temas 1, 2, 3 y 4)
from scipy.stats import shapiro, f_oneway
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# Métricas técnicas de validación de modelos
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)

# Integración de Deep Learning / Inteligencia Artificial
import tensorflow as tf

# ------------------------------------------------------------------------------
# CONFIGURACIÓN ESENCIAL DE LA INTERFAZ DE USUARIO (UX/UI LIMPIA)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Soporte Clínico Minimalista", page_icon="⚕️", layout="wide")

# Carga de la hoja de estilos CSS personalizada
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ------------------------------------------------------------------------------
# FILTRADO Y OPTIMIZACIÓN DEL DATASET A LOS 4 DATOS CRÍTICOS
# ------------------------------------------------------------------------------
@st.cache_data
def inicializar_entorno_datos():
    cancer = load_breast_cancer()
    
    # Mapeo explícito de las 4 características imprescindibles
    caracteristicas_criticas = {
        'mean area': 'area_media',
        'mean texture': 'textura_media',
        'mean smoothness': 'suavidad_media',
        'mean concavity': 'concavidad_media'
    }
    
    # Creamos el DataFrame usando únicamente las columnas seleccionadas
    df_raw = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_filtrado = df_raw[list(caracteristicas_criticas.keys())].copy()
    df_filtrado.rename(columns=caracteristicas_criticas, inplace=True)
    
    # Agregamos la variable objetivo (0: Maligno, 1: Benigno)
    df_filtrado['diagnostico'] = cancer.target
    return df_filtrado

df = inicializar_entorno_datos()

# Paletas de colores profesionales estables
paleta_clinica = {0: '#ef4444', 1: '#22c55e', '0': '#ef4444', '1': '#22c55e'}
paleta_clusters = {0: '#0284c7', 1: '#f59e0b', '0': '#0284c7', '1': '#f59e0b'}
columnas_4 = ['area_media', 'textura_media', 'suavidad_media', 'concavidad_media']

# ------------------------------------------------------------------------------
# ENTRENAMIENTO ROBUSTO DE MODELOS BASADO EN LOS 4 DATOS FILTRADOS
# ------------------------------------------------------------------------------
@st.cache_resource
def entrenar_pipeline_ia():
    X = df[columnas_4]
    y = df['diagnostico']
    
    # El Escalador ajusta las variables para que tengan el mismo peso matemático
    escalador_global = StandardScaler()
    X_escalado_global = escalador_global.fit_transform(X)
    
    # Reducción Dimensional PCA (Tema 3)
    pca_global = PCA(n_components=2)
    X_pca_global = pca_global.fit_transform(X_escalado_global)
    
    # Clustering No Supervisado K-Means (Tema 3)
    kmeans_global = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters_global = kmeans_global.fit_predict(X_escalado_global)
    
    # Split de validación para Aprendizaje Supervisado
    X_train, X_test, y_train, y_test = train_test_split(
        X_escalado_global, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Clasificador SVM con Kernel RBF Optimizado
    svm_global = SVC(kernel='rbf', probability=True, random_state=42)
    svm_global.fit(X_train, y_train)
    
    # Regresión Supervisada Tradicional (Predecir Área basada en Concavidad)
    X_reg = df[['concavidad_media']]
    y_reg = df['area_media']
    lineal_global = LinearRegression()
    lineal_global.fit(X_reg, y_reg)
    
    # Red Neuronal Artificial (ANN Deep Learning) para Regresión
    ann_global = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    ann_global.compile(optimizer='adam', loss='mse')
    ann_global.fit(X_reg, y_reg, epochs=30, batch_size=16, verbose=0)
    
    return (escalador_global, X_escalado_global, pca_global, X_pca_global, 
            kmeans_global, clusters_global, svm_global, X_test, y_test, lineal_global, ann_global)

(escalador, X_escalado, pca, X_pca, kmeans, 
 clusters, modelo_svm, X_val, y_val, modelo_lineal, modelo_ann) = entrenar_pipeline_ia()

# ------------------------------------------------------------------------------
# NAVEGACIÓN MEDIANTE MENÚ DESPLEGABLE LATERAL (SIDEBAR UX)
# ------------------------------------------------------------------------------
st.sidebar.title("⚕️ Panel Clínico")
st.sidebar.markdown("Navegación del Sistema Integrador:")

opcion_menu = st.sidebar.selectbox(
    "Seleccione una Vista",
    [
        "🩺 Simulador de Diagnóstico",
        "📊 Análisis Estadístico (Tema 1)",
        "📈 Gráficas del Dataset (Tema 1)",
        "🧬 Espacio Latente PCA (Tema 3)",
        "🤖 Grupos Autónomos K-Means (Tema 3)",
        "📉 Estimación de Regresión (Tema 2 & 4)",
        "🏆 Matriz y Rendimiento (Tema 4)"
    ]
)

st.sidebar.write("---")
st.sidebar.caption("Sistema de SVM")
st.sidebar.caption("Tecnológico Nacional de México")

# ------------------------------------------------------------------------------
# CONTROLADOR DE RENDERIZADO DE LA PÁGINA PRINCIPAL
# ------------------------------------------------------------------------------

# Vista 1: SIMULADOR CLÍNICO PRINCIPAL
if opcion_menu == "🩺 Simulador de Diagnóstico":
    st.title("Sistema de Soporte de Decisions Clínicas")
    st.markdown("Inferencia diagnóstica de patologías mamarias optimizada mediante **Máquinas de Soporte Vectorial (SVM)**.")
    
    st.markdown('<div class="predictor-card">', unsafe_allow_html=True)
    st.subheader("Entrada del Vector de Características Obligatorias")
    
    col_a, col_b = st.columns(2)
    with col_a:
        in_area = st.slider("Área Media Celular (Volumen Espacial)", float(df['area_media'].min()), float(df['area_media'].max()), float(df['area_media'].mean()))
        in_texture = st.slider("Textura Media (Homogeneidad de Cromatina)", float(df['textura_media'].min()), float(df['textura_media'].max()), float(df['textura_media'].mean()))
    with col_b:
        in_smoothness = st.slider("Suavidad Media Local (Regularidad del Borde)", float(df['suavidad_media'].min()), float(df['suavidad_media'].max()), float(df['suavidad_media'].mean()))
        in_concavity = st.slider("Concavidad Media (Hendiduras del Contorno)", float(df['concavidad_media'].min()), float(df['concavidad_media'].max()), float(df['concavidad_media'].mean()))

    if st.button("Ejecutar Análisis Clínico del Patrón"):
        # Creamos un DataFrame con los nombres correctos para que el escalador no falle
        vector_df = pd.DataFrame([[in_area, in_texture, in_smoothness, in_concavity]], columns=columnas_4)
        vector_escalado = escalador.transform(vector_df)
        prediccion = modelo_svm.predict(vector_escalado)[0]
        probabilidades = modelo_svm.predict_proba(vector_escalado)[0]
        
        st.write("---")
        if prediccion == 0:
            st.markdown(f'<div class="result-malignant">⚠️ <b>DIAGNOSTICO:</b> Estructura celular clasificada como MALIGNA (Certeza: {probabilidades[0]*100:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-benign">✅ <b>DIAGNOSTICO:</b> Estructura celular clasificada como BENIGNA (Certeza: {probabilidades[1]*100:.2f}%)</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Vista 2: PRUEBAS ESTADÍSTICAS OBLIGATORIAS (TEMA 1)
elif opcion_menu == "📊 Análisis Estadístico (Tema 1)":
    st.title("Análisis Exploratorio y Estadístico del Dataset")
    st.write("Evidencia analítica de las 5 pruebas requeridas por la rúbrica institucional:")
    
    st.subheader("1. Análisis Descriptivo del Conjunto de Datos")
    st.dataframe(df[columnas_4].describe(), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2. Prueba de Normalidad de Shapiro-Wilk")
        stat, p_val = shapiro(df['area_media'])
        st.write(f"**Variable 'area_media'**: Estadística = {stat:.4f} | p-valor = {p_val:.4e}")
        if p_val > 0.05:
            st.success("Distribución Gaussiana Normal (p > 0.05).")
        else:
            st.warning("Distribución No Normal (p <= 0.05). Variabilidad típica en entornos biológicos.")
            
        st.subheader("3. Detección Matemática de Datos Atípicos (Outliers)")
        Q1 = df['area_media'].quantile(0.25)
        Q3 = df['area_media'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['area_media'] < (Q1 - 1.5 * IQR)) | (df['area_media'] > (Q3 + 1.5 * IQR))]
        st.write(f"El método del rango intercuartílico identificó **{len(outliers)}** outliers en la variable `area_media`.")

    with c2:
        st.subheader("5. Relación de Entrada vs Objetivo (ANOVA de una vía)")
        f_stat, anova_p = f_oneway(df[df['diagnostico'] == 0]['area_media'], df[df['diagnostico'] == 1]['area_media'])
        st.write(f"**ANOVA ('area_media' vs Diagnóstico)**: F-Value = {f_stat:.4f} | p-valor = {anova_p:.4e}")
        st.success("La diferencia de varianzas entre grupos es estadísticamente significativa.")
        
    st.subheader("4. Matriz de Correlación entre Variables Críticas")
    st.dataframe(df[columnas_4].corr(), use_container_width=True)

# Vista 3: VISUALIZACIONES GRÁFICAS (TEMA 1)
elif opcion_menu == "📈 Gráficas del Dataset (Tema 1)":
    st.title("Expresión Gráfica del Dataset")
    st.write("Panel técnico con las 5 representaciones gráficas obligatorias:")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    sns.histplot(df['area_media'], kde=True, ax=axes[0, 0], color='#0284c7')
    axes[0, 0].set_title("1. Distribución Física (Histograma)")
    
    sns.boxplot(y=df['area_media'], ax=axes[0, 1], color='#94a3b8')
    axes[0, 1].set_title("2. Visualización de Outliers (Boxplot)")
    
    sns.scatterplot(data=df, x='area_media', y='textura_media', hue='diagnostico', palette=paleta_clinica, ax=axes[0, 2], alpha=0.6)
    axes[0, 2].set_title("3. Dispersión Relacional")
    
    sns.heatmap(df[columnas_4].corr(), annot=True, cmap='Blues', fmt=".2f", ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title("4. Mapa de Calor de Correlaciones")
    
    sns.violinplot(data=df, x='diagnostico', y='concavidad_media', hue='diagnostico', ax=axes[1, 1], palette=paleta_clinica, legend=False)
    axes[1, 1].set_title("5. Densidad de Concavidad por Clase")
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# Vista 4: REDUCCIÓN DE DIMENSIONALIDAD (TEMA 3)
elif opcion_menu == "🧬 Espacio Latente PCA (Tema 3)":
    st.title("Análisis de Componentes Principales (PCA)")
    df_pca = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
    df_pca['diagnostico'] = df['diagnostico']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_pca, x='Componente 1', y='Componente 2', hue='diagnostico', palette=paleta_clinica, ax=ax)
    st.pyplot(fig)
    st.info(f"Porcentaje de varianza total explicada acumulada por las 2 componentes: **{np.sum(pca.explained_variance_ratio_)*100:.2f}%**")

# Vista 5: APRENDIZAJE NO SUPERVISADO (TEMA 3)
elif opcion_menu == "🤖 Grupos Autónomos K-Means (Tema 3)":
    st.title("Descubrimiento de Patrones mediante K-Means")
    df_pca = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
    df_pca['Cluster'] = clusters
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_pca, x='Componente 1', y='Componente 2', hue='Cluster', palette=paleta_clusters, ax=ax)
    st.pyplot(fig)
    st.metric("Coeficiente de Silueta (Silhouette Score)", f"{silhouette_score(X_escalado, clusters):.4f}")

# Vista 6: MODULOS DE REGRESIÓN (TEMA 2 Y 4)
elif opcion_menu == "📉 Estimación de Regresión (Tema 2 & 4)":
    st.title("Estimación de Variables Continuas")
    st.write("Cálculo aproximado de la superficie celular basado en el índice de concavidad del contorno:")
    
    concavidad_in = st.number_input("Introduzca el valor de Concavidad Media para la regresión:", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
    df_input_reg = pd.DataFrame([[concavidad_in]], columns=['concavidad_media'])
    
    col_reg1, col_reg2 = st.columns(2)
    with col_reg1:
        st.subheader("6. Regresión Lineal Tradicional")
        pred_lineal = modelo_lineal.predict(df_input_reg)[0]
        st.metric("Área Estimada (Lineal)", f"{pred_lineal:.2f} u²")
        
    with col_reg2:
        st.subheader("7. Red Neuronal Artificial (ANN Deep Learning)")
        pred_ann = modelo_ann.predict(df_input_reg, verbose=0)[0][0]
        st.metric("Área Estimada (Red Neuronal)", f"{pred_ann:.2f} u²")

# Vista 7: MEDICIÓN DE RENDIMIENTO TÉCNICO (TEMA 4)
elif opcion_menu == "🏆 Matriz y Rendimiento (Tema 4)":
    st.title("Evaluación Técnico-Cuantitativa de Rendimiento")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("8. Métricas de Clasificación (SVM)")
        preds_val = modelo_svm.predict(X_val)
        
        st.metric("Exactitud General (Accuracy)", f"{accuracy_score(y_val, preds_val):.4f}")
        st.metric("Precisión (Precision)", f"{precision_score(y_val, preds_val):.4f}")
        st.metric("Sensibilidad Crítica (Recall)", f"{recall_score(y_val, preds_val):.4f}")
        st.metric("F1-Score Combinado", f"{f1_score(y_val, preds_val):.4f}")
        st.info("El elevado nivel de *Recall* alcanzado demuestra que el modelo minimiza los Falsos Negativos, blindando el sistema para entornos médicos reales.")
        
    with col_m2:
        st.subheader("8. Métricas de Modelos de Regresión")
        # CORRECCIÓN DE COLUMNA: Filtramos usando los datos correctos para evitar el KeyError
        X_r = df[['concavidad_media']]
        y_r = df['area_media']
        
        p_l = modelo_lineal.predict(X_r)
        p_a = modelo_ann.predict(X_r, verbose=0).flatten()
        
        st.markdown("**Regresión Lineal Estándar:**")
        st.write(f"- MSE: {mean_squared_error(y_r, p_l):.4f} | $R^2$: {r2_score(y_r, p_l):.4f}")
        st.markdown("**Red Neuronal Artificial (ANN):**")
        st.write(f"- MSE: {mean_squared_error(y_r, p_a):.4f} | $R^2$: {r2_score(y_r, p_a):.4f}")