
from pathlib import Path
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

BASE = Path(__file__).parent
MODELO = BASE / "obas_v2_student_float16.tflite"
MAPA = BASE / "PASO24_7_mapa_puntos_calientes_colombia.html"

st.set_page_config(
    page_title="OBAS V2 | Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.18);
}
h1, h2, h3 {letter-spacing: -0.3px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_csv(nombre):
    ruta = BASE / nombre
    if not ruta.exists():
        return None
    try:
        return pd.read_csv(ruta)
    except Exception:
        return None

@st.cache_resource
def cargar_modelo(ruta):
    interpreter = tf.lite.Interpreter(
        model_path=str(ruta),
        num_threads=2
    )
    interpreter.allocate_tensors()
    return interpreter

def mostrar_tabla(nombre, titulo=None):
    df = cargar_csv(nombre)
    if df is None:
        st.warning("No se encontró el archivo: " + nombre)
        return None
    if titulo:
        st.markdown("**" + titulo + "**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    return df

def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()

def numero(df, columna, fila=0, predeterminado=None):
    if df is None or columna not in df.columns or len(df) <= fila:
        return predeterminado
    try:
        return float(df.iloc[fila][columna])
    except (ValueError, TypeError):
        return predeterminado

st.title("🔥 OBAS V2")
st.caption(
    "Observación de puntos calientes y segmentación de áreas potencialmente quemadas"
)
st.info(
    "Los puntos calientes no confirman por sí solos un incendio. "
    "La clase 1 es una predicción del modelo y no equivale automáticamente "
    "a una superficie quemada confirmada."
)

with st.sidebar:
    st.header("Panel de navegación")
    st.caption("Resultados precalculados del proyecto")
    st.divider()
    st.markdown("**Modelo de inferencia**")
    st.write("Student MobileNetV2–UNet")
    st.write("TFLite Float16")
    st.write("Entrada: 128 × 128 × 3")
    st.divider()
    st.caption("OBAS V2 · Dashboard experimental")

tab_resumen, tab_modelos, tab_geo, tab_gradcam, tab_focos, tab_pred, tab_info = st.tabs([
    "Resumen ejecutivo",
    "Comparación de modelos",
    "Análisis geoespacial",
    "Grad-CAM",
    "Puntos calientes",
    "Predicción interactiva",
    "Metodología"
])

# 1. RESUMEN EJECUTIVO
with tab_resumen:
    st.header("Resumen ejecutivo")
    st.write(
        "Vista general de las métricas, los análisis geoespaciales y "
        "la preparación del modelo para inferencia móvil."
    )

    df_tflite = cargar_csv("PASO21_2_tflite_comparacion.csv")
    df_geo = cargar_csv("PASO22_3_superficie_geografica_tolima.csv")
    df_hot = cargar_csv("PASO24_6_resumen_calidad.csv")
    df_cems = cargar_csv("PASO25_2_6_metricas.csv")
    df_decision = cargar_csv("PASO23_4_2_decision_final_modelo_movil.csv")

    tam = numero(df_tflite, "Tamaño_MB", 1)
    dice = numero(df_tflite, "Dice_F1", 1)
    iou = numero(df_tflite, "IoU", 1)
    latencia = numero(df_tflite, "Latencia_media_ms", 1)
    hectareas = None
    if df_geo is not None and "clase" in df_geo.columns:
        fila_geo = df_geo[df_geo["clase"].astype(str) == "1"]
        if not fila_geo.empty:
            hectareas = numero(fila_geo, "superficie_ha")

    if hectareas is None and df_geo is not None:
        if "superficie_ha" in df_geo.columns and len(df_geo) >= 2:
            hectareas = numero(df_geo, "superficie_ha", 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tamaño Float16", f"{tam:.2f} MB" if tam is not None else "N/D")
    c2.metric("Dice / F1", f"{dice:.4f}" if dice is not None else "N/D")
    c3.metric("IoU", f"{iou:.4f}" if iou is not None else "N/D")
    c4.metric(
        "Latencia media reportada",
        f"{latencia:.2f} ms" if latencia is not None else "N/D"
    )

    st.subheader("Hallazgos principales")
    st.markdown(
        "- **Modelo móvil:** Float16 reduce el tamaño del modelo respecto a Float32.\n"
        "- **Segmentación:** el rendimiento se interpreta mediante precisión, recall, "
        "Dice/F1 e IoU.\n"
        "- **Geoespacial:** la superficie de clase 1 es una estimación de la máscara "
        "predicha para el área analizada.\n"
        "- **Puntos calientes:** los registros son indicadores de posibles focos de calor, "
        "no confirmaciones individuales de incendios."
    )

    if hectareas is not None:
        st.metric(
            "Superficie predicha como clase 1 en el análisis geoespacial",
            f"{hectareas:,.2f} ha"
        )
        st.caption(
            "Estimación del producto analizado; no debe interpretarse como "
            "área quemada oficialmente confirmada."
        )

    st.subheader("Prueba de parche CEMS")
    if df_cems is not None:
        st.dataframe(df_cems, use_container_width=True, hide_index=True)
        st.caption(
            "Resultado de un parche individual. No representa por sí solo "
            "el rendimiento general del modelo."
        )
    else:
        st.info("No se encontró el archivo de métricas del parche CEMS.")

    st.subheader("Archivos disponibles")
    archivos_clave = [
        "PASO21_2_test_modelos.csv",
        "PASO21_2_tflite_comparacion.csv",
        "PASO22_3_superficie_geografica_tolima.csv",
        "PASO22_4_consistencia_prediccion_dNBR.csv",
        "PASO22_5_comparacion_MODIS_exploratoria.csv",
        "PASO20_3_gradcam_patch1527.png",
        "PASO24_7_mapa_puntos_calientes_colombia.html"
    ]
    existentes = sum((BASE / x).exists() for x in archivos_clave)
    st.progress(existentes / len(archivos_clave))
    st.caption(f"{existentes} de {len(archivos_clave)} archivos principales disponibles.")

# 2. COMPARACIÓN DE MODELOS
with tab_modelos:
    st.header("Comparación de modelos")
    st.write("Resultados precalculados; esta sección no vuelve a entrenar modelos.")

    st.subheader("Modelos A, B y C en test espacial")
    df_abc = mostrar_tabla("PASO21_2_test_modelos.csv")

    if df_abc is not None:
        columnas = [
            c for c in ["Accuracy", "Precision", "Recall", "F1-score", "IoU_quemado"]
            if c in df_abc.columns
        ]
        if columnas and "Variante" in df_abc.columns:
            graf = df_abc.set_index("Variante")[columnas]
            st.bar_chart(graf, y=columnas, height=350)
            st.caption("Comparación de métricas reportadas para las variantes evaluadas.")

    st.subheader("Exportación TFLite: Float32 frente a Float16")
    df_tflite = mostrar_tabla("PASO21_2_tflite_comparacion.csv")

    if df_tflite is not None:
        columnas = [
            c for c in ["Dice_F1", "IoU", "Precision", "Recall"]
            if c in df_tflite.columns
        ]
        if columnas and "Modelo" in df_tflite.columns:
            st.bar_chart(df_tflite.set_index("Modelo")[columnas], height=320)

        if "Tamaño_MB" in df_tflite.columns and "Modelo" in df_tflite.columns:
            st.subheader("Tamaño del modelo")
            st.bar_chart(df_tflite.set_index("Modelo")[["Tamaño_MB"]], height=220)

        if {"Modelo", "Tamaño_MB"}.issubset(df_tflite.columns):
            if len(df_tflite) >= 2:
                f32 = df_tflite[df_tflite["Modelo"].astype(str).str.contains("float32", case=False)]
                f16 = df_tflite[df_tflite["Modelo"].astype(str).str.contains("float16", case=False)]
                if not f32.empty and not f16.empty:
                    a = float(f32.iloc[0]["Tamaño_MB"])
                    b = float(f16.iloc[0]["Tamaño_MB"])
                    if a > 0:
                        st.metric("Reducción de tamaño Float16", f"{(a-b)/a*100:.2f}%")

    st.subheader("Decisión de modelo móvil")
    mostrar_tabla("PASO23_4_2_decision_final_modelo_movil.csv")
    st.caption(
        "La latencia reportada depende del entorno de medición. "
        "No equivale a un benchmark en un teléfono real."
    )

# 3. ANÁLISIS GEOESPACIAL
with tab_geo:
    st.header("Análisis geoespacial")
    st.warning(
        "Las superficies y asociaciones mostradas son resultados de análisis "
        "del proyecto; no constituyen confirmación independiente de incendios."
    )

    st.subheader("Superficie por clase predicha")
    df_superficie = mostrar_tabla("PASO22_3_superficie_geografica_tolima.csv")
    if df_superficie is not None:
        col_clase = "clase" if "clase" in df_superficie.columns else None
        col_area = "superficie_ha" if "superficie_ha" in df_superficie.columns else None
        if col_clase and col_area:
            datos = df_superficie.copy()
            datos[col_area] = pd.to_numeric(datos[col_area], errors="coerce")
            datos = datos.dropna(subset=[col_area])
            st.bar_chart(datos.set_index(col_clase)[[col_area]], height=280)

    st.subheader("Consistencia con dNBR")
    mostrar_tabla("PASO22_4_consistencia_prediccion_dNBR.csv")
    st.caption(
        "Una asociación entre la clase predicha y dNBR no sustituye "
        "la validación con referencias independientes."
    )

    st.subheader("Comparación exploratoria con MODIS")
    df_modis = mostrar_tabla("PASO22_5_comparacion_MODIS_exploratoria.csv")
    if df_modis is not None:
        st.info(
            "La comparación disponible es exploratoria y tiene cobertura parcial. "
            "No se debe presentar como validación independiente definitiva."
        )

# 4. GRAD-CAM
with tab_gradcam:
    st.header("Interpretabilidad: Grad-CAM")
    st.write(
        "Visualización exploratoria de activaciones del modelo para un parche "
        "específico. No demuestra causalidad ni garantiza que el modelo use "
        "exclusivamente características relacionadas con áreas quemadas."
    )

    imagen_gradcam = BASE / "PASO20_3_gradcam_patch1527.png"
    if imagen_gradcam.exists():
        st.image(
            str(imagen_gradcam),
            caption="Grad-CAM del parche 1527",
            use_container_width=True
        )
    else:
        st.warning("No se encontró la visualización Grad-CAM guardada.")

    st.subheader("Métricas de activación")
    mostrar_tabla("PASO20_4_gradcam_metricas_patch1527.csv")

    archivo_capas = BASE / "PASO20_2_capas_modelo.csv"
    if archivo_capas.exists():
        with st.expander("Inspeccionar capas del modelo"):
            mostrar_tabla("PASO20_2_capas_modelo.csv")

# 5. PUNTOS CALIENTES
with tab_focos:
    st.header("Puntos calientes de Colombia")
    st.write(
        "Mapa HTML existente y resumen de calidad de los registros. "
        "La cartografía base puede requerir conexión a internet."
    )

    if MAPA.exists():
        html_mapa = MAPA.read_text(encoding="utf-8", errors="replace")
        components.html(html_mapa, height=650, scrolling=True)
    else:
        st.error("No se encontró el mapa HTML existente.")

    st.subheader("Calidad de los registros")
    df_calidad = mostrar_tabla("PASO24_6_resumen_calidad.csv")

    st.subheader("Datos limpios")
    ruta_hotspots = BASE / "PASO24_6_puntos_calientes_colombia_limpios.csv"
    if ruta_hotspots.exists():
        try:
            df_hotspots = pd.read_csv(ruta_hotspots)
            st.caption(f"Registros disponibles: {len(df_hotspots):,}")
            st.dataframe(df_hotspots.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar CSV limpio completo",
                data=ruta_hotspots.read_bytes(),
                file_name=ruta_hotspots.name,
                mime="text/csv"
            )
            st.caption("Se muestran hasta 100 filas en pantalla; la descarga contiene el CSV completo.")
        except Exception as e:
            st.error("No se pudo leer el CSV de puntos calientes: " + str(e))
    else:
        st.warning("No se encontró el CSV limpio de puntos calientes.")

# 6. PREDICCIÓN INTERACTIVA
with tab_pred:
    st.header("Predicción interactiva")
    st.write(
        "Carga un parche NumPy compatible con el preprocesamiento del modelo. "
        "La aplicación no vuelve a entrenar el modelo."
    )

    st.warning(
        "Formato admitido: .npy con forma (128,128,3) o (3,128,128), "
        "valores finitos en [0,1] y tres canales. No se aceptan directamente "
        "imágenes JPG o PNG."
    )

    if not MODELO.exists():
        st.error("No se encuentra el modelo obas_v2_student_float16.tflite.")
    else:
        archivo = st.file_uploader(
            "Selecciona un parche .npy",
            type=["npy"],
            key="parche_obas_dashboard"
        )
        umbral = st.slider(
            "Umbral de clasificación",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="umbral_obas_dashboard"
        )

        if archivo is not None:
            try:
                imagen = np.load(archivo, allow_pickle=False).astype(np.float32)

                if imagen.shape == (3, 128, 128):
                    imagen = np.transpose(imagen, (1, 2, 0))

                if imagen.shape != (128, 128, 3):
                    st.error(
                        f"Forma no compatible: {imagen.shape}. "
                        "Se requiere (128,128,3) o (3,128,128)."
                    )
                elif not np.isfinite(imagen).all():
                    st.error("El parche contiene valores no finitos.")
                elif imagen.min() < 0 or imagen.max() > 1:
                    st.error("Los valores del parche deben estar entre 0 y 1.")
                elif st.button("Ejecutar segmentación", type="primary"):
                    interpreter = cargar_modelo(str(MODELO))
                    entrada = interpreter.get_input_details()[0]
                    salida = interpreter.get_output_details()[0]

                    interpreter.set_tensor(
                        entrada["index"],
                        imagen[np.newaxis, ...].astype(entrada["dtype"])
                    )

                    inicio = time.perf_counter()
                    interpreter.invoke()
                    latencia_ms = (time.perf_counter() - inicio) * 1000

                    probabilidades = interpreter.get_tensor(
                        salida["index"]
                    )[0, :, :, 0]
                    mascara = (probabilidades >= umbral).astype(np.uint8)
                    porcentaje = float(mascara.mean() * 100)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Latencia", f"{latencia_ms:.2f} ms")
                    c2.metric("Píxeles clase 1", f"{int(mascara.sum()):,}")
                    c3.metric("Píxeles clase 1 (%)", f"{porcentaje:.2f}%")

                    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                    ax[0].imshow(np.clip(imagen, 0, 1))
                    ax[0].set_title("Parche original")

                    im = ax[1].imshow(probabilidades, vmin=0, vmax=1)
                    ax[1].set_title("Probabilidad clase 1")
                    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

                    ax[2].imshow(mascara, vmin=0, vmax=1)
                    ax[2].set_title(f"Máscara: umbral {umbral:.2f}")

                    for eje in ax:
                        eje.axis("off")

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.download_button(
                        "Descargar máscara (.npy)",
                        data=npy_bytes(mascara),
                        file_name="obas_mascara_predicha.npy",
                        mime="application/octet-stream"
                    )
                    st.download_button(
                        "Descargar probabilidades (.npy)",
                        data=npy_bytes(probabilidades.astype(np.float32)),
                        file_name="obas_probabilidades.npy",
                        mime="application/octet-stream"
                    )

                    st.caption(
                        "El porcentaje corresponde a píxeles clasificados como clase 1. "
                        "No equivale por sí solo a hectáreas ni confirma superficie quemada."
                    )
            except Exception as e:
                st.error("No se pudo procesar el parche: " + str(e))
        else:
            st.info("Carga un parche .npy compatible para comenzar.")

# 7. METODOLOGÍA
with tab_info:
    st.header("Metodología, alcance y limitaciones")

    st.subheader("Modelo de inferencia")
    st.write("Student MobileNetV2–UNet exportado a TensorFlow Lite Float16.")
    st.write("Entrada esperada: 128 × 128 píxeles y 3 canales.")
    st.write("El modelo se usa para inferencia; el dashboard no realiza entrenamiento.")

    st.subheader("Interpretación de los resultados")
    st.markdown(
        "- **Clase 1:** píxeles que el modelo clasifica como positivos según el umbral.\n"
        "- **Dice/F1 e IoU:** métricas de coincidencia entre máscaras, según el conjunto evaluado.\n"
        "- **dNBR y MODIS:** análisis exploratorios que requieren considerar resolución, "
        "cobertura y referencia disponible.\n"
        "- **Grad-CAM:** herramienta de inspección visual de activaciones, no prueba causal.\n"
        "- **Puntos calientes:** posibles focos de calor, no confirmaciones automáticas de incendio.\n"
        "- **Hectáreas:** solo son interpretables con georreferenciación, escala y máscara "
        "correctamente alineadas.\n"
        "- **Rendimiento móvil:** las mediciones en el entorno de ejecución no sustituyen "
        "pruebas en dispositivos físicos."
    )

    st.subheader("Métricas y trazabilidad")
    for nombre in [
        "PASO21_2_validacion_modelos.csv",
        "PASO21_2_test_modelos.csv",
        "PASO21_2_tflite_comparacion.csv",
        "PASO23_3_inferencia_comparativa.csv",
        "PASO23_4_2_matriz_decision_modelo_movil.csv",
        "PASO25_1_ficha_modelo_inferencia.csv",
        "PASO25_2_6_metricas.csv"
    ]:
        with st.expander(nombre):
            mostrar_tabla(nombre)

st.divider()
st.caption(
    "OBAS V2 · Dashboard de investigación. Los resultados deben interpretarse "
    "según el conjunto de datos, la resolución y las limitaciones de validación."
)
