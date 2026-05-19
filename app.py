#instalación de liberias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CLASE PRINCIPAL - PROGRAMACIÓN ORIENTADA A OBJETOS
# =========================================================
class DataAnalyzer:
    def __init__(self, dataframe):
        """
        Constructor de la clase.
        Recibe el DataFrame principal.
        """
        self.df = dataframe

    # =====================================================
    # INFORMACIÓN GENERAL
    # =====================================================
    def dimensiones_dataset(self):
        filas, columnas = self.df.shape
        return filas, columnas
    def tipos_datos(self):
        return self.df.dtypes
    def valores_nulos(self):
        return self.df.isnull().sum()

    # =====================================================
    # CLASIFICACIÓN DE VARIABLES
    # =====================================================
    def variables_numericas(self):
        return self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    def variables_categoricas(self):
        return self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()

    # =====================================================
    # ESTADÍSTICAS DESCRIPTIVAS
    # =====================================================
    def estadisticas_descriptivas(self):
        return self.df.describe()

    # =====================================================
    # CONTEO DE CATEGORÍAS
    # =====================================================
    def conteo_categorias(self, columna):
        return self.df[columna].value_counts()

    # =====================================================
    # VISUALIZACIÓN HISTOGRAMA
    # =====================================================
    def graficar_histograma(self, columna):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(
            self.df[columna],
            kde=True,
            ax=ax
        )
        ax.set_title(f"Distribución de {columna}")
        return fig

    # =====================================================
    # VISUALIZACIÓN BOXPLOT
    # =====================================================
    def graficar_boxplot(self, x, y):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(
            data=self.df,
            x=x,
            y=y,
            ax=ax
        )
        return fig

    # =====================================================
    # MATRIZ DE CORRELACIÓN
    # =====================================================
    def matriz_correlacion(self):
        variables_num = self.df.select_dtypes(
            include=["int64", "float64"]
        )
        return variables_num.corr()

######### Análisis Exploratorio de Datos (EDA) del dataset TelcoCustomerChurn.csv
# Menú lateral
menu = st.sidebar.selectbox(
    "Navegación",
    ["Módulo 1", "Módulo 2"]
)

# Esquema de las hojas
if menu == "Módulo 1":
    st.title("🏠 Módulo 1: Home")
    st.subheader("Inicio del trabajo")
    st.write("Empezamos a desarrollar el trabajo.")
    #Logo personal, logo de Python DMC o imagen representativa
    st.image("DMC Logo.png", width=100)
    
    st.markdown("""
    Apéndice:
    - 📊 Título del proyecto: Trabajo Final  DMC Python - Estudios de Python
    - ⚙️ Breve descripción del objetivo del análisis: Análisis Exploratorio de Datos (EDA) del dataset TelcoCustomerChurn.csv
    - 📈 Nombre del estudiante: Jorge Adolfo Mondragón Arboccó
    - ⚙️ Nombre del curso: Especialización en Python for Analytics - Edición - 57
    - ⚙️ Año: 2026 (Mayo)
    - ⚙️ Breve explicación del dataset: El dataset tiene 21 variables asociadas a clientes de una empresa de Telecomunicaciones. Tiene aprox 7mil registros de clientes indicando si fueron bajas o no.
    - ⚙️ Tecnologías utilizadas: Python, Anaconda, Streamlit
                    
    **Selecciona una opción en el menú izquierdo para ver el desarrollo de los módulos.**
    """)

elif menu == "Módulo 2":
    st.title("📊 Módulo 2")
    st.subheader("Carga del dataset")
    
    # Descripción
    st.markdown("""
    Este módulo permite la carga, explorar información, estadísticas y visualizaciones del dataset Customer Churn.
    """)

    #Leemos el dataset
    archivo = st.sidebar.file_uploader(
        "📂 Seleccione un archivo CSV", type=["csv"]
    )

    # Validación de carga
    if archivo is not None:

        #Leer CSV
        df = pd.read_csv(archivo)
        st.success("✅ Archivo cargado correctamente")

        #Vista previa
        st.subheader("👀 Vista previa del dataset")
        st.dataframe(df.head(3))
    
        # Dimensiones
        #filas, columnas = df.shape
        #col1, col2, col3 = st.columns(3)
        #col1.metric("📄 Filas", filas)
        #col2.metric("📊 Columnas", columnas)
        #col3.metric("👥 Clientes", df["customerID"].nunique())
    else:
        st.warning("⚠️ Debe cargar un archivo CSV para continuar.")
        st.stop()

    # Crear objeto de la clase
    analizador = DataAnalyzer(df)

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📄 Ítem 1",
        "📈 Ítem 2",
        "📊 Ítem 3",
        "🔍 Ítem 4",
        "📈 Ítem 5",
        "🔍 Ítem 6",
        "📈 Ítem 7",
        "📈 Ítem 8",
        "📈 Ítem 9",
        "📈 Ítem 10"
    ])

    # =====================================================
    # TAB 1 - INFORMACIÓN GENERAL
    # =====================================================
    with tab1:
        st.header("📄 Ítem 1 - Información General del Dataset")
        st.markdown("""
        Este módulo permite analizar la estructura general del dataset, identificando tipos de datos y valores nulos.
        """)

        st.header("📌 Resumen")
        filas, columnas = analizador.dimensiones_dataset()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Número de Filas", filas)
        with col2:
            st.metric("Número de Columnas", columnas)

        st.subheader("📌 Tipos de Datos")
        st.dataframe(analizador.tipos_datos())

        st.subheader("📌 Conteo de valores nulos")
        st.dataframe(analizador.valores_nulos())

    # =====================================================
    # TAB 2 - CLASIFICACIÓN DE VARIABLES
    # =====================================================
    with tab2:
        st.subheader("📄 Ítem 2 - Clasificación de variables")
        st.markdown("""
        Este módulo identifica automáticamente las variables numéricas y categóricas del dataset utilizando una función personalizada.
        """)

        # =================================================
        # FUNCIÓN PERSONALIZADA
        # =================================================
        def clasificar_variables(dataframe):
            variables_numericas = dataframe.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            variables_categoricas = dataframe.select_dtypes(
                include=['object']
            ).columns.tolist()
            return variables_numericas, variables_categoricas

        # =================================================
        # EJECUCIÓN DE FUNCIÓN
        # =================================================
        numericas, categoricas = clasificar_variables(df)

        # =================================================
        # MÉTRICAS
        # =================================================
        st.subheader("📌 Conteo de Variables")
        col1, col2 = st.columns(2)
        col1.metric(
            "Variables Numéricas",
            len(numericas)
        )
        col2.metric(
            "Variables Categóricas",
            len(categoricas)
        )

        # =================================================
        # VARIABLES NUMÉRICAS
        # =================================================
        st.subheader("🔢 Variables Numéricas")
        df_numericas = pd.DataFrame({
            "Variables Numéricas": numericas
        })
        st.dataframe(df_numericas)

        # =================================================
        # VARIABLES CATEGÓRICAS
        # =================================================
        st.subheader("🔤 Variables Categóricas")
        df_categoricas = pd.DataFrame({
            "Variables Categóricas": categoricas
        })
        st.dataframe(df_categoricas)

        ##extra
        st.subheader("🔤 Conteo variables Numéricas y Categóricas")
        conteos = pd.DataFrame({
            "Tipo": ["Numéricas", "Categóricas"],
            "Cantidad": [len(numericas), len(categoricas)]
        })

        fig, ax = plt.subplots()

        sns.barplot(
            data=conteos,
            x="Tipo",
            y="Cantidad",
            ax=ax
        )
        st.pyplot(fig)

    # =====================================================
    # TAB 3 - ESTADÍSTICAS DESCRIPTIVAS
    # =====================================================
    with tab3:
        st.header("📈 Ítem 3 - Estadísticas Descriptivas")
        st.markdown("""
        Este módulo permite analizar las principales medidas estadísticas de las variables numéricas del dataset como promedio, mediana y dispersión.
        """)

        # =================================================
        # VARIABLES NUMÉRICAS
        # =================================================
        variables_numericas = df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        # =================================================
        # DESCRIBE()
        # =================================================
        st.subheader("📌 Resumen Estadístico General")
        descripcion = df[variables_numericas].describe()
        st.dataframe(descripcion)

        # =================================================
        # SELECTOR DE VARIABLE
        # =================================================
        st.subheader("📌 Análisis Individual de Variable")
        variable = st.selectbox(
            "Seleccione una variable numérica",
            variables_numericas
        )

        # =================================================
        # CÁLCULOS
        # =================================================
        media = df[variable].mean()
        mediana = df[variable].median()
        #moda = df[variable].mode()
        desviacion = df[variable].std()
        minimo = df[variable].min()
        maximo = df[variable].max()

        # =================================================
        # MÉTRICAS VISUALES
        # =================================================
        col1, col2, col3 = st.columns(3)
        col1.metric("Media", round(media, 2))
        col2.metric("Mediana", round(mediana, 2))
        col3.metric("Desv. Estándar", round(desviacion, 2))
        col4, col5 = st.columns(2)
        col4.metric("Mínimo", round(minimo, 2))
        col5.metric("Máximo", round(maximo, 2))
 
        # =================================================
        # INTERPRETACIÓN
        # =================================================
        st.subheader("📌 Interpretación Básica")
        st.write(f"""
        • La media de la variable **{variable}** es **{round(media,2)}**.

        • La mediana es **{round(mediana,2)}**, lo que representa el valor central de los datos.

        • La desviación estándar es **{round(desviacion,2)}**, indicando el nivel de dispersión de los datos respecto al promedio.

        • Los valores oscilan entre **{round(minimo,2)}** y **{round(maximo,2)}**.
        """)

        #st.subheader("📌 Distribución de la Variable")
        #fig, ax = plt.subplots(figsize=(8,4))
        #sns.histplot(
        #    df[variable],
        #    kde=True,
        #    ax=ax
        #)
        #st.pyplot(fig)

    # ====================================
    # TAB 4 - VALORES FALTANTES
    # =====================================================
    with tab4:
        st.header("⚠️ Ítem 4 - Análisis de Valores Faltantes")
        st.markdown("""
        Este módulo analiza la presencia de valores faltantes dentro del dataset, permitiendo identificar posibles problemas de calidad de datos.
        """)

        # =================================================
        # CÁLCULO DE NULOS
        # =================================================
        nulos = df.isnull().sum()
        porcentaje_nulos = (
            (nulos / len(df)) * 100
        ).round(2)

        # =================================================
        # DATAFRAME DE RESULTADOS
        # =================================================
        df_nulos = pd.DataFrame({
            "Variable": nulos.index,
            "Valores Faltantes": nulos.values,
            "Porcentaje (%)": porcentaje_nulos.values
        })

        # =================================================
        # FILTRAR SOLO VARIABLES CON NULOS
        # =================================================
        df_nulos_filtrado = df_nulos[
            df_nulos["Valores Faltantes"] > 0
        ]

        # =================================================
        # MÉTRICAS GENERALES
        # =================================================
        total_nulos = nulos.sum()
        variables_con_nulos = (
            df_nulos_filtrado.shape[0]
        )
        col1, col2 = st.columns(2)
        col1.metric(
            "Total Valores Faltantes",
            int(total_nulos)
        )
        col2.metric(
            "Variables con Nulos",
            variables_con_nulos
        )

        # =================================================
        # TABLA
        # =================================================
        st.subheader("📌 Tabla de Valores Faltantes")
        if df_nulos_filtrado.empty:
            st.success(
                "✅ El dataset no contiene valores faltantes."
            )
        else:
            st.dataframe(df_nulos_filtrado)

        # =============================================
        # VISUALIZACIÓN
        # =============================================
        st.subheader("📌 Visualización de Valores Faltantes")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(
            data=df_nulos_filtrado,
            x="Variable",
            y="Valores Faltantes",
            ax=ax
            )
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # =================================================
        # ANALISIS
        # =================================================
        st.subheader("📌 Análisis")
        if total_nulos == 0:
            st.write("""
            El dataset presenta buena calidad de datos, ya que no se identificaron valores faltantes. Esto facilita el análisis y modelado posterior.
            """)
        else:
            st.write(f"""
            Se identificaron {int(total_nulos)} valores faltantes distribuidos en {variables_con_nulos} variables.

            La presencia de datos nulos puede afectar los análisis estadísticos y modelos predictivos, por lo que podría requerirse limpieza o imputación de datos.
            """)

    # ====================================
    # TAB 5 - DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
    # =====================================================
    with tab5:
        st.header("📉 Ítem 5 - Distribución de Variables Numéricas")
        st.markdown("""
        Este módulo permite analizar visualmente la distribución de las variables numéricas mediante histogramas y curvas de densidad.
        """)

        # =================================================
        # VARIABLES NUMÉRICAS
        # =================================================
        variables_numericas = df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        # =================================================
        # SELECTOR
        # =================================================
        variable1 = st.selectbox(
            "Seleccione una variable numérica",
            variables_numericas,
            key="select_estadisticas"
        )

        # =================================================
        # MÉTRICAS
        # =================================================
        media = df[variable1].mean()
        mediana = df[variable1].median()
        desviacion = df[variable1].std()
        col1, col2, col3 = st.columns(3)
        col1.metric("Media", round(media,2))
        col2.metric("Mediana", round(mediana,2))
        col3.metric("Desv. Estándar", round(desviacion,2))

        # =================================================
        # HISTOGRAMA
        # =================================================
        st.subheader("📌 Histograma de Distribución")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(
            df[variable1],
            kde=True,
            bins=30,
            ax=ax
        )
        ax.set_title(f"Distribución de {variable1}")
        st.pyplot(fig)

        # =================================================
        # INTERPRETACIÓN
        # =================================================
        st.subheader("📌 Interpretación Visual")
        if abs(media - mediana) < 1:
            interpretacion = """
            La distribución parece relativamente simétrica, ya que la media y la mediana presentan valores similares.
            """
        elif media > mediana:
            interpretacion = """
            La distribución presenta posible sesgo positivo, indicando presencia de valores altos extremos.
            """
        else:
            interpretacion = """
            La distribución presenta posible sesgo negativo, indicando presencia de valores bajos extremos.
            """
        st.write(interpretacion)
        st.write(f"""
        La variable **{variable1}** presenta una desviación estándar de **{round(desviacion,2)}**, lo que refleja el nivel de dispersión de los datos respecto al promedio.
        """)

    # =========================================================
    # ÍTEM 6: ANÁLISIS DE VARIABLES CATEGÓRICAS
    # =========================================================
    with tab6:
        st.header("📌 Ítem 6: Análisis de Variables Categóricas")
        st.markdown("""
        En esta sección se analizan las variables categóricas del dataset.
        Se muestran:
        - Conteo de categorías
        - Proporciones porcentuales
        - Gráficos de barras
        """)

        # -----------------------------------------
        # OBTENER VARIABLES CATEGÓRICAS
        # -----------------------------------------
        variables_categoricas = df.select_dtypes(include="object").columns.tolist()

        # SELECTBOX
        variable_cat = st.selectbox(
            "Seleccione una variable categórica",
            variables_categoricas,
            key="item6_select_cat"
        )

        # -----------------------------------------
        # CONTEO DE CATEGORÍAS
        # -----------------------------------------
        st.subheader("📋 Conteo de Categorías")
        conteo_cat = df[variable_cat].value_counts().reset_index()
        conteo_cat.columns = ["Categoría", "Cantidad"]
        st.dataframe(conteo_cat)

        # -----------------------------------------
        # PROPORCIONES
        # -----------------------------------------
        st.subheader("📊 Proporciones (%)")
        proporcion_cat = round(
            (df[variable_cat].value_counts(normalize=True) * 100),
            2
        ).reset_index()
        proporcion_cat.columns = ["Categoría", "Porcentaje"]
        st.dataframe(proporcion_cat)

        # -----------------------------------------
        # GRÁFICO DE BARRAS
        # -----------------------------------------
        st.subheader("📈 Gráfico de Barras")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(
            data=df,
            x=variable_cat,
            ax=ax
        )
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # -----------------------------------------
        # INTERPRETACIÓN
        # -----------------------------------------
        st.subheader("🧠 Interpretación")
        categoria_top = df[variable_cat].value_counts().idxmax()
        cantidad_top = df[variable_cat].value_counts().max()
        st.info(
            f"La categoría más frecuente en '{variable_cat}' es "
            f"'{categoria_top}' con {cantidad_top} registros."
        )

    # =========================================================
    # ÍTEM 7: ANÁLISIS BIVARIADO
    # =========================================================
    with tab7:
        st.header("📌 Ítem 7: Análisis Bivariado")
        st.markdown("""
        Este análisis permite comparar variables numéricas contra la variable categórica Churn para identificar posibles patrones.
        """)

        # =========================================================
        # TABS
        # =========================================================
        tab1, tab2 = st.tabs([
            "MonthlyCharges vs Churn",
            "tenure vs Churn"
        ])

        # =========================================================
        # TAB 1
        # =========================================================
        with tab1:
            st.subheader("💰 MonthlyCharges vs Churn")
            col1, col2 = st.columns(2)

            # -------------------------------------
            # BOXPLOT
            # -------------------------------------
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6,5))
                sns.boxplot(
                    data=df,
                    x="Churn",
                    y="MonthlyCharges",
                    ax=ax1
                )
                st.pyplot(fig1)

            # -------------------------------------
            # PROMEDIOS
            # -------------------------------------
            with col2:
                promedio_mc = df.groupby("Churn")["MonthlyCharges"].mean()
                st.metric(
                    "Promedio NO Churn",
                    round(promedio_mc["No"], 2)
                )
                st.metric(
                    "Promedio Churn",
                    round(promedio_mc["Yes"], 2)
                )

            # -------------------------------------
            # INTERPRETACIÓN
            # -------------------------------------
            st.success("""
            Los clientes que abandonan el servicio (Churn = Yes) suelen presentar cargos mensuales más elevados.
            """)

        # =========================================================
        # TAB 2
        # =========================================================
        with tab2:
            st.subheader("📆 tenure vs Churn")
            col1, col2 = st.columns(2)

            # -------------------------------------
            # BOXPLOT
            # -------------------------------------
            with col1:
                fig2, ax2 = plt.subplots(figsize=(6,5))
                sns.boxplot(
                    data=df,
                    x="Churn",
                    y="tenure",
                    ax=ax2
                )
                st.pyplot(fig2)

            # -------------------------------------
            # PROMEDIOS
            # -------------------------------------
            with col2:
                promedio_tenure = df.groupby("Churn")["tenure"].mean()
                st.metric(
                    "Promedio NO Churn",
                    round(promedio_tenure["No"], 2)
                )
                st.metric(
                    "Promedio Churn",
                    round(promedio_tenure["Yes"], 2)
                )

            # -------------------------------------
            # INTERPRETACIÓN
            # -------------------------------------
            st.success("""
            Los clientes que abandonan el servicio generalmente tienen menor antigüedad (tenure).
            """)

    # =========================================================
    # ÍTEM 8: ANÁLISIS BIVARIADO (CATEGÓRICO VS CATEGÓRICO)
    # =========================================================
    with tab8:
        st.header("📌 Ítem 8: Análisis Bivariado: Categórico vs Categórico")
        st.markdown("""
        En esta sección se comparan variables categóricas contra la variable objetivo **Churn**.
        El análisis permite identificar patrones de abandono según:
        - Tipo de contrato
        - Servicio de internet
        """)

        # =========================================================
        # TABS
        # =========================================================
        tab1, tab2 = st.tabs([
            "Contract vs Churn",
            "InternetService vs Churn"
        ])

        # =========================================================
        # TAB 1 → CONTRACT VS CHURN
        # =========================================================
        with tab1:
            st.subheader("📄 Contract vs Churn")
            col1, col2 = st.columns(2)

            # -------------------------------------
            # TABLA CRUZADA
            # -------------------------------------
            with col1:
                tabla_contract = pd.crosstab(
                    df["Contract"],
                    df["Churn"]
                )
                st.dataframe(tabla_contract)

            # -------------------------------------
            # GRÁFICO
            # -------------------------------------
            with col2:
                fig1, ax1 = plt.subplots(figsize=(7,5))
                sns.countplot(
                    data=df,
                    x="Contract",
                    hue="Churn",
                    ax=ax1
                )
                plt.xticks(rotation=15)
                st.pyplot(fig1)

            # -------------------------------------
            # PROPORCIONES
            # -------------------------------------
            st.subheader("📊 Proporciones de Churn")
            proporciones_contract = round(
                pd.crosstab(
                    df["Contract"],
                    df["Churn"],
                    normalize="index"
                ) * 100,
                2
            )
            st.dataframe(proporciones_contract)

            # -------------------------------------
            # INTERPRETACIÓN
            # -------------------------------------
            st.success("""
            Los clientes con contratos mensuales presentan mayor probabilidad de Churn frente a contratos de largo plazo.
            """)

        # =========================================================
        # TAB 2 → INTERNETSERVICE VS CHURN
        # =========================================================
        with tab2:
            st.subheader("🌐 InternetService vs Churn")
            col1, col2 = st.columns(2)

            # -------------------------------------
            # TABLA CRUZADA
            # -------------------------------------
            with col1:
                tabla_internet = pd.crosstab(
                    df["InternetService"],
                    df["Churn"]
                )
                st.dataframe(tabla_internet)

            # -------------------------------------
            # GRÁFICO
            # -------------------------------------
            with col2:
                fig2, ax2 = plt.subplots(figsize=(7,5))
                sns.countplot(
                    data=df,
                    x="InternetService",
                    hue="Churn",
                    ax=ax2
                )
                plt.xticks(rotation=15)
                st.pyplot(fig2)

            # -------------------------------------
            # PROPORCIONES
            # -------------------------------------
            st.subheader("📊 Proporciones de Churn")
            proporciones_internet = round(
                pd.crosstab(
                    df["InternetService"],
                    df["Churn"],
                    normalize="index"
                ) * 100,
                2
            )
            st.dataframe(proporciones_internet)

            # -------------------------------------
            # INTERPRETACIÓN
            # -------------------------------------
            st.success("""
            El servicio de Fibra Óptica presenta un nivel de Churn superior frente a otros tipos de servicio.
            """)

    # =========================================================
    # ÍTEM 9: Análisis basado en parámetros seleccionados
    # =========================================================
    with tab9:
        st.header("📌 Ítem 9: Análisis basado en parámetros seleccionados")
        st.markdown("""
        Este módulo permite realizar análisis dinámicos seleccionando variables de manera interactiva mediante widgets.
        """)

        # =========================================================
        # SELECCIÓN DE VARIABLES
        # =========================================================
        col1, col2 = st.columns(2)
        with col1:
            variable_numerica = st.selectbox(
                "Seleccione una variable numérica",
                df.select_dtypes(include=["int64", "float64"]).columns,
                key="item9_variable_numerica"
            )

        with col2:
            variable_categorica = st.selectbox(
                "Seleccione una variable categórica",
                df.select_dtypes(include="object").columns,
                key="item9_variable_categorica"
            )

        # =========================================================
        # MULTISELECT
        # =========================================================
        #columnas_mostrar = st.multiselect(
        #    "Seleccione columnas para visualizar",
        #    df.columns.tolist(),
        #    default=[
        #        variable_categorica,
        #        variable_numerica
        #    ],
        #    key="item9_multiselect"
        #)

        # =========================================================
        # VISTA PREVIA
        # =========================================================
        #st.subheader("📋 Vista Previa")
        #if len(columnas_mostrar) > 0:
        #    st.dataframe(df[columnas_mostrar].head())
        #else:
        #    st.warning("Seleccione al menos una columna.")

        # =========================================================
        # GRÁFICO DINÁMICO
        # =========================================================
        st.subheader("📈 Análisis Dinámico")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(
            data=df,
            x=variable_categorica,
            y=variable_numerica,
            ax=ax
        )
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # =========================================================
        # MÉTRICAS
        # =========================================================
        st.subheader("📊 Estadísticas Generales")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Promedio",
                round(df[variable_numerica].mean(), 2)
            )
        with col2:
            st.metric(
                "Mediana",
                round(df[variable_numerica].median(), 2)
            )
        with col3:
            st.metric(
                "Desviación Estándar",
                round(df[variable_numerica].std(), 2)
            )

        # =========================================================
        # INTERPRETACIÓN
        # =========================================================
        st.info(f"""
        El gráfico permite comparar la distribución de '{variable_numerica}' según las categorías de '{variable_categorica}'.
        """)

    # =========================================================
    # ÍTEM 10: HALLAZGOS CLAVE
    # =========================================================
    with tab10:
        st.header("📌 Ítem 10: Hallazgos clave del EDA")
        st.markdown("""
        En esta sección se presentan los principales insights obtenidos durante el Análisis Exploratorio de Datos (EDA).
        El objetivo es resumir los hallazgos más relevantes del comportamiento de los clientes y su relación con el Churn.
        """)

        # =========================================================
        # MÉTRICAS PRINCIPALES
        # =========================================================
        st.subheader("📊 Indicadores Generales")
        col1, col2, col3, col4 = st.columns(4)

        # Total clientes
        total_clientes = df.shape[0]

        # Total churn
        total_churn = df[df["Churn"] == "Yes"].shape[0]

        # Porcentaje churn
        porcentaje_churn = round(
            (total_churn / total_clientes) * 100,
            2
        )

        # Promedio MonthlyCharges
        promedio_monthly = round(
            df["MonthlyCharges"].mean(),
            2
        )

        with col1:
            st.metric(
                "Total Clientes",
                total_clientes
            )

        with col2:
            st.metric(
                "Clientes Churn",
                total_churn
            )

        with col3:
            st.metric(
                "% Churn",
                f"{porcentaje_churn}%"
            )

        with col4:
            st.metric(
                "Promedio MonthlyCharges",
                promedio_monthly
            )

        # =========================================================
        # VISUALIZACIÓN RESUMEN
        # =========================================================
        st.subheader("📈 Resumen Visual")
        col1, col2 = st.columns(2)

        # =========================================================
        # GRÁFICO 1 → CHURN
        # =========================================================
        with col1:
            fig1, ax1 = plt.subplots(figsize=(5,5))
            churn_counts = df["Churn"].value_counts()
            ax1.pie(
                churn_counts,
                labels=churn_counts.index,
                autopct='%1.1f%%'
            )
            ax1.set_title("Distribución de Churn")
            st.pyplot(fig1)

        # =========================================================
        # GRÁFICO 2 → CONTRACT VS CHURN
        # =========================================================
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,5))
            sns.countplot(
                data=df,
                x="Contract",
                hue="Churn",
                ax=ax2
            )
            plt.xticks(rotation=15)
            st.pyplot(fig2)

        # =========================================================
        # HALLAZGOS PRINCIPALES
        # =========================================================
        st.subheader("🧠 Insights Principales")

        st.success("""
        ✅ Los clientes con contratos mensuales presentan una mayor tasa de abandono (Churn).

        ✅ Los clientes con mayor antigüedad (tenure) tienden a permanecer más tiempo en la empresa.

        ✅ Los clientes con cargos mensuales elevados presentan una mayor probabilidad de cancelar el servicio.

        ✅ El servicio de Fibra Óptica muestra mayor proporción de Churn respecto a otros servicios.

        ✅ El porcentaje general de Churn representa un indicador importante para estrategias de retención.
        """)

        # =========================================================
        # CONCLUSIÓN GENERAL
        # =========================================================
        st.subheader("📌 Conclusión General")

        st.info(f"""
        El análisis exploratorio permitió identificar patrones importantes relacionados con la cancelación de clientes.
        Se observó que factores como: tipo de contrato, antigüedad del cliente, servicio de internet y cargos mensuales influyen significativamente en el comportamiento de Churn.
        El porcentaje actual de abandono es de {porcentaje_churn}%, por lo que la empresa podría enfocar estrategias de fidelización         en clientes con contratos mensuales y altos cargos mensuales.
        """)