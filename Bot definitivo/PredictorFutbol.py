# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, log_loss, f1_score, roc_auc_score
from concurrent.futures import ThreadPoolExecutor
import io
import warnings
from typing import List, Dict, Optional, Any, Tuple
import traceback

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Iqscore Football Predictor (RF)",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ignore common warnings ---
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- np.bool handling for compatibility ---
if hasattr(np, 'bool'):
    if type(np.bool) is type: np.bool = np.bool_ # type: ignore
    else: pass
else: np.bool_ = bool

# --- Initialize Session State for Theme ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light' # Default to light mode

# --- Predictor Class (Adapted for Streamlit) ---
class StreamlitFootballRandomForestPredictor:
    """
    Predicts football match outcomes using RandomForest models - Iqscore Version.
    Adapted for Streamlit, handling data uploads, preprocessing,
    model training, and prediction display with Streamlit UI elements.
    Includes predictions for detailed match statistics.
    """
    # --- Configuration Constants ---
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_ESTIMATORS: int = 150
    MAX_DEPTH: Optional[int] = 20
    MIN_SAMPLES_SPLIT: int = 10
    MIN_SAMPLES_LEAF: int = 5
    MAX_FEATURES: Optional[str] = 'sqrt'
    N_JOBS: int = -1
    CONFIDENCE_THRESHOLDS: Dict[str, List[Tuple[float, str, str]]] = {
        'result': [(0.70, 'Fuerte', '🔥'), (0.55, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'over_under': [(0.75, 'Fuerte', '🔥'), (0.60, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'btts': [(0.75, 'Fuerte', '🔥'), (0.60, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'ht_result': [(0.70, 'Fuerte', '🔥'), (0.55, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'yellow_cards': [(5.0, 'Fuerte', '🔥'), (4.0, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'corners': [(11.0, 'Fuerte', '🔥'), (9.5, 'Moderada', '👍'), (0.0, 'Baja', '👀')],
        'total_shots': [(25.0, 'Alto', '🎯'), (20.0, 'Medio', '⚽'), (0.0, 'Bajo', '🛡️')],
        'total_shots_on_target': [(10.0, 'Alto', '🎯'), (7.0, 'Medio', '⚽'), (0.0, 'Bajo', '🛡️')],
        'total_fouls': [(24.0, 'Alto', '🟨'), (20.0, 'Medio', '✋'), (0.0, 'Bajo', '✔️')],
        'total_red_cards': [(0.6, 'Alto', '🟥'), (0.0, 'Bajo', '✔️')],
    }
    OVER_UNDER_THRESHOLD: float = 2.5

    # --- Class Attributes ---
    def __init__(self):
        # Use Streamlit session state to persist objects across reruns
        if 'predictor_models' not in st.session_state:
            st.session_state.predictor_models = {}
        if 'predictor_numeric_features' not in st.session_state:
            st.session_state.predictor_numeric_features = []
        if 'predictor_categorical_features' not in st.session_state:
            st.session_state.predictor_categorical_features = []
        if 'predictor_features' not in st.session_state:
            st.session_state.predictor_features = []
        if 'predictor_preprocessor' not in st.session_state:
            st.session_state.predictor_preprocessor = None
        if 'predictor_data' not in st.session_state:
            st.session_state.predictor_data = None
        if 'predictor_X_processed_shape' not in st.session_state:
            st.session_state.predictor_X_processed_shape = None
        if 'predictor_data_loaded' not in st.session_state:
            st.session_state.predictor_data_loaded = False
        if 'predictor_trained' not in st.session_state:
            st.session_state.predictor_trained = False
        if 'available_teams' not in st.session_state:
            st.session_state.available_teams = []
        if 'available_referees' not in st.session_state:
            st.session_state.available_referees = [''] # Start with blank

    # --- Accessors for Session State Attributes ---
    @property
    def models(self) -> Dict[str, Any]: return st.session_state.predictor_models
    @models.setter
    def models(self, value: Dict[str, Any]): st.session_state.predictor_models = value

    @property
    def numeric_features(self) -> List[str]: return st.session_state.predictor_numeric_features
    @numeric_features.setter
    def numeric_features(self, value: List[str]): st.session_state.predictor_numeric_features = value

    @property
    def categorical_features(self) -> List[str]: return st.session_state.predictor_categorical_features
    @categorical_features.setter
    def categorical_features(self, value: List[str]): st.session_state.predictor_categorical_features = value

    @property
    def features(self) -> List[str]: return st.session_state.predictor_features
    @features.setter
    def features(self, value: List[str]): st.session_state.predictor_features = value

    @property
    def preprocessor(self) -> Optional[ColumnTransformer]: return st.session_state.predictor_preprocessor
    @preprocessor.setter
    def preprocessor(self, value: Optional[ColumnTransformer]): st.session_state.predictor_preprocessor = value

    @property
    def data(self) -> Optional[pd.DataFrame]: return st.session_state.predictor_data
    @data.setter
    def data(self, value: Optional[pd.DataFrame]): st.session_state.predictor_data = value

    @property
    def X_processed_shape(self) -> Optional[Tuple[int, int]]: return st.session_state.predictor_X_processed_shape
    @X_processed_shape.setter
    def X_processed_shape(self, value: Optional[Tuple[int, int]]): st.session_state.predictor_X_processed_shape = value

    @property
    def data_loaded(self) -> bool: return st.session_state.predictor_data_loaded
    @data_loaded.setter
    def data_loaded(self, value: bool): st.session_state.predictor_data_loaded = value

    @property
    def trained(self) -> bool: return st.session_state.predictor_trained
    @trained.setter
    def trained(self, value: bool): st.session_state.predictor_trained = value

    @property
    def available_teams(self) -> List[str]: return st.session_state.available_teams
    @available_teams.setter
    def available_teams(self, value: List[str]): st.session_state.available_teams = value

    @property
    def available_referees(self) -> List[str]: return st.session_state.available_referees
    @available_referees.setter
    def available_referees(self, value: List[str]): st.session_state.available_referees = value

    # --- Data Loading Methods (Adapted for Streamlit Uploads) ---
    def load_and_process_uploaded_files(self, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
        """Loads data from files uploaded via Streamlit, prepares, and trains."""
        if not uploaded_files:
            st.warning("Por favor, sube uno o más archivos CSV o Excel.")
            self.reset_state() # Reset if no files are uploaded
            return

        progress_bar = st.progress(0, text="Cargando archivos...")
        data_frames = []
        num_files = len(uploaded_files)

        def load_single_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
            """Loads a single uploaded file content into a DataFrame."""
            file_name = uploaded_file.name
            st.write(f"Intentando cargar: `{file_name}`")
            df = None
            try:
                file_content = uploaded_file.getvalue() # Read file content
                if file_name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(io.BytesIO(file_content), delimiter=',', encoding='utf-8', on_bad_lines='warn', low_memory=False)
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        try:
                            df = pd.read_csv(io.BytesIO(file_content), delimiter=';', encoding='utf-8', on_bad_lines='warn', low_memory=False)
                        except (pd.errors.ParserError, UnicodeDecodeError):
                            try:
                                df = pd.read_csv(io.BytesIO(file_content), delimiter=',', encoding='latin1', on_bad_lines='warn', low_memory=False)
                            except (pd.errors.ParserError, UnicodeDecodeError):
                                try:
                                    df = pd.read_csv(io.BytesIO(file_content), delimiter=';', encoding='latin1', on_bad_lines='warn', low_memory=False)
                                except Exception as e_inner:
                                    st.error(f"Error al parsear CSV `{file_name}` con configuraciones comunes: {e_inner}")
                                    return None

                elif file_name.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(file_content))
                else:
                    st.warning(f"Formato no soportado para `{file_name}`. Ignorando.")
                    return None

                if df is not None:
                    # Basic processing (similar to Colab)
                    odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
                                   'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA',
                               'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']
                    for col in odds_cols:
                        col_lower = col.lower().replace(' ', '_').replace('.', '')
                        if col in df.columns:
                             df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif col_lower in df.columns:
                            df[col_lower] = pd.to_numeric(df[col_lower], errors='coerce')
                    st.success(f"Archivo `{file_name}` cargado.")
                    return df
                else:
                     return None

            except Exception as e:
                st.error(f"Error general al cargar `{file_name}`: {e}")
                return None

        # Use ThreadPoolExecutor for potentially faster loading (optional)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_single_file, f) for f in uploaded_files]
            for i, future in enumerate(futures):
                df_result = future.result()
                if df_result is not None:
                    data_frames.append(df_result)
                progress_bar.progress((i + 1) / num_files, text=f"Cargando archivo {i+1}/{num_files}...")

        if not data_frames:
            st.error("No se pudieron cargar datos válidos de ningún archivo.")
            self.reset_state()
            progress_bar.empty()
            return

        try:
            progress_bar.progress(1.0, text="Concatenando datos...")
            self.data = pd.concat(data_frames, ignore_index=True, sort=False)
            self.data_loaded = True
            self.trained = False # Reset trained status on new data
            st.success(f"¡Datos combinados! Total de partidos: {len(self.data)}")

            # --- Trigger Preparation and Training ---
            progress_bar.progress(0.1, text="Preparando datos...")
            self.prepare_data(progress_bar) # Pass progress bar

            if self.data is not None and self.preprocessor is not None:
                progress_bar.progress(0.5, text="Entrenando modelos...")
                self.train_models(progress_bar) # Pass progress bar
                if self.models:
                    self.trained = True
                    # Populate dropdown options
                    self.update_dropdown_options()
                    st.success("¡Modelos entrenados y listos para predicción!")
                else:
                     st.error("Falló el entrenamiento de modelos. Verifica los datos.")
                     self.trained = False
            else:
                 st.error("Falló la preparación de datos. No se puede entrenar.")
                 self.trained = False

        except Exception as e:
            st.error(f"Error procesando archivos: {e}")
            traceback.print_exc()
            self.reset_state()
        finally:
             progress_bar.empty() # Remove progress bar

    def reset_state(self):
        """Resets the predictor's state in session_state."""
        st.session_state.predictor_models = {}
        st.session_state.predictor_numeric_features = []
        st.session_state.predictor_categorical_features = []
        st.session_state.predictor_features = []
        st.session_state.predictor_preprocessor = None
        st.session_state.predictor_data = None
        st.session_state.predictor_X_processed_shape = None
        st.session_state.predictor_data_loaded = False
        st.session_state.predictor_trained = False
        st.session_state.available_teams = []
        st.session_state.available_referees = ['']

    def update_dropdown_options(self):
        """Updates team and referee lists for Streamlit dropdowns."""
        if self.data is not None and not self.data.empty:
            try:
                teams_h = self.data['hometeam'].unique() if 'hometeam' in self.data.columns else []
                teams_a = self.data['awayteam'].unique() if 'awayteam' in self.data.columns else []
                teams = sorted(list(set(teams_h) | set(teams_a)))
                self.available_teams = [str(t) for t in teams if pd.notna(t)]

                referees_list = [''] # Blank option
                if 'referee' in self.data.columns:
                    referees_list.extend(sorted([r for r in self.data['referee'].unique() if pd.notna(r) and r != '']))
                self.available_referees = [str(r) for r in referees_list if pd.notna(r)]

            except Exception as e:
                st.error(f"Error actualizando opciones de dropdown: {e}")
                self.available_teams = ["Error Cargando Equipos"]
                self.available_referees = ["Error Cargando Árbitros"]
        else:
             self.available_teams = []
             self.available_referees = ['']

    # --- prepare_data Method ---
    def prepare_data(self, progress_bar: Optional[st.progress] = None):
        if self.data is None or self.data.empty:
            st.error("Error: No hay datos para preparar.")
            return

        st.info("--- Iniciando Preparación de Datos ---")
        log_area = st.expander("Logs de Preparación de Datos", expanded=False)
        log_messages = []

        try:
            data = self.data.copy()
            data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)

            if 'date' not in data.columns:
                raise ValueError("La columna 'date' es requerida para el análisis.")
            try:
                data['date'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True)
            except Exception as date_e:
                 log_messages.append(f"Advertencia: Error al convertir la columna 'date': {date_e}. Intentando inferir formato.")
                 try:
                     data['date'] = pd.to_datetime(data['date'], errors='coerce', infer_datetime_format=True)
                 except Exception as date_e2:
                      log_messages.append(f"Error: Falló la conversión de fecha incluso con inferencia: {date_e2}. Eliminando filas con fechas inválidas.")
                      data.dropna(subset=['date'], inplace=True)

            n_null_dates_before = data['date'].isnull().sum()
            if n_null_dates_before > 0:
                most_recent_date = data['date'].max()
                if pd.isna(most_recent_date): most_recent_date = pd.Timestamp.now().normalize()
                data['date'].fillna(most_recent_date, inplace=True)
                log_messages.append(f"Se imputaron {n_null_dates_before} fechas nulas con la fecha más reciente ({most_recent_date.date()}).")

            core_target_cols = ['fthg', 'ftag', 'ftr']
            missing_core = [col for col in core_target_cols if col not in data.columns]
            if missing_core: raise ValueError(f"Faltan columnas objetivo esenciales: {missing_core}. Verifica tu archivo.")

            for col in ['fthg', 'ftag']: data[col] = pd.to_numeric(data[col], errors='coerce')

            initial_rows = len(data)
            data.dropna(subset=['fthg', 'ftag'], inplace=True)
            if 'ftr' in data.columns:
                 data['ftr'] = data['ftr'].astype(str).str.upper().str.strip()
                 valid_ftr = ['H', 'D', 'A']
                 invalid_ftr_mask = ~data['ftr'].isin(valid_ftr)
                 if invalid_ftr_mask.any():
                      log_messages.append(f"Advertencia: Se encontraron {invalid_ftr_mask.sum()} valores inválidos en 'ftr' (ej: {data.loc[invalid_ftr_mask, 'ftr'].unique()[:5]}). Eliminando filas.")
                      data = data[data['ftr'].isin(valid_ftr)]
            else: raise ValueError("La columna 'ftr' es requerida.")

            if len(data) < initial_rows: log_messages.append(f"Se eliminaron {initial_rows - len(data)} filas por falta de FTHG, FTAG o FTR inválido.")
            if data.empty: raise ValueError("No quedan datos después de eliminar filas con FTHG/FTAG faltantes o FTR inválido.")

            self.numeric_features = [
                'b365h', 'b365d', 'b365a', 'bwh', 'bwd', 'bwa', 'iwh', 'iwd', 'iwa',
                'psh', 'psd', 'psa', 'whh', 'whd', 'wha', 'vch', 'vcd', 'vca',
                'maxh', 'maxd', 'maxa', 'avgh', 'avgd', 'avga',
                'hs', 'as', 'hst', 'ast', 'hf', 'af', 'hc', 'ac', 'hy', 'ay', 'hr', 'ar'
            ]
            self.categorical_features = ['hometeam', 'awayteam', 'referee']

            available_numeric = [f for f in self.numeric_features if f in data.columns]
            missing_numeric = [f for f in self.numeric_features if f not in data.columns]
            if missing_numeric: log_messages.append(f"Advertencia: Faltan características numéricas predefinidas: {missing_numeric}. No se usarán.")
            self.numeric_features = available_numeric

            available_categorical = [f for f in self.categorical_features if f in data.columns]
            missing_categorical = [f for f in self.categorical_features if f not in data.columns]
            if missing_categorical: log_messages.append(f"Advertencia: Faltan características categóricas predefinidas: {missing_categorical}. No se usarán.")
            self.categorical_features = available_categorical

            if 'hometeam' not in self.categorical_features: raise ValueError("Falta la columna 'hometeam'.")
            if 'awayteam' not in self.categorical_features: raise ValueError("Falta la columna 'awayteam'.")

            # Handle 'referee' more robustly
            if 'referee' not in data.columns:
                data['referee'] = 'Desconocido'
                log_messages.append("Columna 'referee' no encontrada, se creó una columna dummy 'Desconocido'.")
                if 'referee' not in self.categorical_features:
                     self.categorical_features.append('referee')
            else:
                data['referee'] = data['referee'].astype(str).fillna('Desconocido').str.strip()
                if 'referee' not in self.categorical_features:
                    self.categorical_features.append('referee')


            if progress_bar: progress_bar.progress(0.2, text="Imputando valores...")

            if self.categorical_features:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                data[self.categorical_features] = categorical_imputer.fit_transform(data[self.categorical_features])
                if 'referee' in self.categorical_features: data['referee'] = data['referee'].fillna('Desconocido')

            log_messages.append(f"Convirtiendo a numérico e imputando {len(self.numeric_features)} características numéricas...")
            for col in self.numeric_features: data[col] = pd.to_numeric(data[col], errors='coerce')

            if self.numeric_features:
                numeric_imputer = SimpleImputer(strategy='mean')
                data[self.numeric_features] = numeric_imputer.fit_transform(data[self.numeric_features])

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

            self.features = self.numeric_features + self.categorical_features

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)],
                remainder='drop',
                verbose_feature_names_out=False)

            if progress_bar: progress_bar.progress(0.3, text="Ajustando preprocesador...")

            X_temp = data[self.features].copy()

            # Impute before fitting preprocessor
            if self.categorical_features:
                cat_imputer_fit = SimpleImputer(strategy='most_frequent')
                X_temp[self.categorical_features] = cat_imputer_fit.fit_transform(X_temp[self.categorical_features])
            if self.numeric_features:
                num_imputer_fit = SimpleImputer(strategy='mean')
                X_temp[self.numeric_features] = num_imputer_fit.fit_transform(X_temp[self.numeric_features])

            if X_temp.isnull().any().any():
                log_messages.append("Advertencia CRÍTICA: Valores nulos encontrados ANTES de ajustar el preprocesador. Rellenando con 0/Unknown.")
                for col in X_temp.select_dtypes(include=np.number).columns:
                    X_temp[col].fillna(0, inplace=True)
                for col in X_temp.select_dtypes(include='object').columns:
                     X_temp[col].fillna('Unknown', inplace=True)


            X_processed_temp = self.preprocessor.fit_transform(X_temp)
            self.X_processed_shape = X_processed_temp.shape
            try:
                feature_names_out = self.preprocessor.get_feature_names_out()
                log_messages.append(f"Ejemplo de características procesadas ({len(feature_names_out)} total): {feature_names_out[:5]}...{feature_names_out[-5:]}")
            except Exception as e_feat_names:
                log_messages.append(f"No se pudieron obtener los nombres de las características procesadas: {e_feat_names}")

            self.data = data
            log_messages.append("\n--- Preparación de Datos Completada ---")
            log_messages.append(f"- Total partidos para modelado: {len(self.data)}")
            log_messages.append(f"- Características numéricas: {len(self.numeric_features)}")
            log_messages.append(f"- Características categóricas (OHE): {len(self.categorical_features)}")
            if 'hometeam' in data.columns and 'awayteam' in data.columns:
                 unique_teams = pd.concat([data['hometeam'], data['awayteam']]).nunique()
                 log_messages.append(f"- Equipos únicos: {unique_teams}")
            if 'referee' in data.columns:
                 unique_referees = data['referee'].nunique()
                 log_messages.append(f"- Árbitros únicos: {unique_referees}")
            log_messages.append(f"- Dimensiones procesadas: {self.X_processed_shape[1] if self.X_processed_shape else 'N/A'}")
            st.success("Preparación de Datos Completada.")

        except Exception as e:
            st.error(f"Error durante la preparación de datos: {e}")
            log_messages.append(f"\nERROR FATAL: {e}")
            traceback.print_exc(file=io.StringIO()) # Capture traceback
            log_messages.append(traceback.format_exc())
            self.data = None # Reset data if prep fails
            self.preprocessor = None
        finally:
            with log_area:
                st.code("\n".join(log_messages), language='text')

    # --- train_models Method ---
    def train_models(self, progress_bar: Optional[st.progress] = None):
        if self.data is None or self.data.empty or self.preprocessor is None or self.X_processed_shape is None:
            st.error("Error: Datos no preparados o preprocesador no ajustado para entrenar.")
            return

        st.info("--- Iniciando Entrenamiento de Modelos RandomForest ---")
        log_area = st.expander("Logs de Entrenamiento", expanded=False)
        log_messages = []

        try:
            missing_in_data = [f for f in self.features if f not in self.data.columns]
            if missing_in_data: raise ValueError(f"Faltan columnas requeridas ANTES de transformar para entrenamiento: {missing_in_data}")

            X = self.data[self.features].copy()

            # Impute before transforming for training
            if self.categorical_features:
                 cat_imputer_transform = SimpleImputer(strategy='most_frequent')
                 X[self.categorical_features] = cat_imputer_transform.fit_transform(X[self.categorical_features])
            if self.numeric_features:
                 num_imputer_transform = SimpleImputer(strategy='mean')
                 X[self.numeric_features] = num_imputer_transform.fit_transform(X[self.numeric_features])

            if X.isnull().any().any():
                log_messages.append("Advertencia: Valores nulos encontrados antes de transformar para entrenamiento. Rellenando con 0/Unknown.")
                for col in X.select_dtypes(include=np.number).columns:
                     X[col].fillna(0, inplace=True)
                for col in X.select_dtypes(include='object').columns:
                     X[col].fillna('Unknown', inplace=True)


            X_processed = self.preprocessor.transform(X)
            X_processed_df = pd.DataFrame(X_processed, index=self.data.index) # Use original data index

            models_to_train = {
                'result': {'type': 'classification', 'target': 'ftr_encoded'},
                'home_goals': {'type': 'regression', 'target': 'fthg'},
                'away_goals': {'type': 'regression', 'target': 'ftag'},
                'ht_result': {'type': 'classification', 'target': 'ht_lead'},
                'btts': {'type': 'classification', 'target': 'btts_flag'},
                'yellow_cards': {'type': 'regression', 'target': 'total_yellows'},
                'total_corners': {'type': 'regression', 'target': 'total_corners'},
                'total_shots': {'type': 'regression', 'target': 'total_shots'},
                'total_shots_on_target': {'type': 'regression', 'target': 'total_shots_on_target'},
                'total_fouls': {'type': 'regression', 'target': 'total_fouls'},
                'total_red_cards': {'type': 'regression', 'target': 'total_red_cards'},
            }

            log_messages.append("Preparando columnas objetivo...")
            target_creation_errors = []
            rows_before_target_prep = len(self.data)

            # Ensure data alignment before creating targets
            self.data = self.data.loc[X_processed_df.index].copy()

            # Create target columns
            ftr_map = {'H': 0, 'D': 1, 'A': 2}
            if 'ftr' in self.data.columns:
                self.data['ftr'] = self.data['ftr'].astype(str).str.upper().str.strip()
                self.data = self.data[self.data['ftr'].isin(ftr_map.keys())] # Filter invalid FTR early
                self.data['ftr_encoded'] = self.data['ftr'].map(ftr_map).astype('Int64')
                if self.data['ftr_encoded'].isnull().any():
                    null_count = self.data['ftr_encoded'].isnull().sum()
                    log_messages.append(f"Advertencia CRÍTICA: {null_count} nulos en 'ftr_encoded'. Eliminando filas.")
                    self.data.dropna(subset=['ftr_encoded'], inplace=True)
            else: target_creation_errors.append('ftr for ftr_encoded')

            def safe_impute_median(df, col_name):
                if col_name in df.columns:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    median_val = df[col_name].median()
                    if pd.isna(median_val): median_val = 0 # Default to 0 if median is NaN
                    df[col_name].fillna(median_val, inplace=True)
                    return True
                return False

            # Create ht_lead
            if safe_impute_median(self.data, 'hthg') and safe_impute_median(self.data, 'htag'):
                 self.data['ht_lead'] = (self.data['hthg'] > self.data['htag']).astype(int)
            else: target_creation_errors.append('hthg/htag for ht_lead')

            # Create btts_flag
            if safe_impute_median(self.data, 'fthg') and safe_impute_median(self.data, 'ftag'):
                self.data['btts_flag'] = ((self.data['fthg'] > 0) & (self.data['ftag'] > 0)).astype(int)
            else: target_creation_errors.append('fthg/ftag for btts_flag')

            # Create total_yellows
            if safe_impute_median(self.data, 'hy') and safe_impute_median(self.data, 'ay'):
                self.data['total_yellows'] = self.data['hy'] + self.data['ay']
                safe_impute_median(self.data, 'total_yellows') # Impute the derived column
            else: target_creation_errors.append('hy/ay for total_yellows')

            # Create total_corners
            if safe_impute_median(self.data, 'hc') and safe_impute_median(self.data, 'ac'):
                 self.data['total_corners'] = self.data['hc'] + self.data['ac']
                 safe_impute_median(self.data, 'total_corners')
            else: target_creation_errors.append('hc/ac for total_corners')

            # Create total_shots
            if safe_impute_median(self.data, 'hs') and safe_impute_median(self.data, 'as'):
                self.data['total_shots'] = self.data['hs'] + self.data['as']
                safe_impute_median(self.data, 'total_shots')
            else: target_creation_errors.append('hs/as for total_shots')

            # Create total_shots_on_target
            if safe_impute_median(self.data, 'hst') and safe_impute_median(self.data, 'ast'):
                self.data['total_shots_on_target'] = self.data['hst'] + self.data['ast']
                safe_impute_median(self.data, 'total_shots_on_target')
            else: target_creation_errors.append('hst/ast for total_shots_on_target')

            # Create total_fouls
            if safe_impute_median(self.data, 'hf') and safe_impute_median(self.data, 'af'):
                self.data['total_fouls'] = self.data['hf'] + self.data['af']
                safe_impute_median(self.data, 'total_fouls')
            else: target_creation_errors.append('hf/af for total_fouls')

            # Create total_red_cards
            if safe_impute_median(self.data, 'hr') and safe_impute_median(self.data, 'ar'):
                self.data['total_red_cards'] = self.data['hr'] + self.data['ar']
                safe_impute_median(self.data, 'total_red_cards')
            else: target_creation_errors.append('hr/ar for total_red_cards')


            if target_creation_errors:
                log_messages.append("\nAdvertencia: No se pudieron crear/imputar las siguientes columnas objetivo:")
                for err in target_creation_errors: log_messages.append(f"  - {err}")

            # Final alignment after potential row drops during target creation
            common_index = X_processed_df.index.intersection(self.data.index)
            if len(common_index) < len(X_processed_df):
                log_messages.append(f"Re-alineando X_processed ({len(X_processed_df)}) con self.data ({len(self.data)}) después de crear objetivos")
                X_processed_df = X_processed_df.loc[common_index]
                self.data = self.data.loc[common_index]
                log_messages.append(f"Nuevo tamaño alineado: {len(X_processed_df)}")

            if X_processed_df.empty or self.data.empty:
                 raise ValueError("No quedan datos después de la preparación del objetivo y la alineación.")

            # --- Train Loop ---
            temp_models = {}
            valid_targets = [cfg['target'] for name, cfg in models_to_train.items() if cfg['target'] in self.data.columns and not self.data[cfg['target']].isnull().all()]
            num_models_to_train = len(valid_targets)
            trained_count = 0

            for name, config in models_to_train.items():
                target_col_name = config['target']

                if target_col_name not in valid_targets:
                    if target_col_name not in self.data.columns:
                        log_messages.append(f"\nSaltando modelo '{name}': Objetivo '{target_col_name}' no existe.")
                    else: # Exists but all null
                        log_messages.append(f"\nSaltando modelo '{name}': Objetivo '{target_col_name}' solo tiene nulos.")
                    continue

                y_target_series = self.data[target_col_name].copy() # Already aligned index

                # Impute target NaNs (should be minimal after safe_impute_median)
                if y_target_series.isnull().any():
                    is_numeric = pd.api.types.is_numeric_dtype(y_target_series)
                    # Use mode for classification, median for regression
                    fill_val = y_target_series.mode()[0] if not is_numeric and not y_target_series.mode().empty else (y_target_series.median() if is_numeric else None)
                    # Fallback if mode/median is also NaN or empty
                    if pd.isna(fill_val): fill_val = 0 if is_numeric else 'Unknown'
                    log_messages.append(f"Imputando {y_target_series.isnull().sum()} NaNs restantes en objetivo '{target_col_name}' con {fill_val}")
                    y_target_series.fillna(fill_val, inplace=True)

                trained_count += 1
                train_progress = 0.5 + 0.5 * (trained_count / num_models_to_train) if num_models_to_train > 0 else 1.0
                if progress_bar: progress_bar.progress(train_progress, text=f"Entrenando modelo: {name.replace('_', ' ').title()} ({trained_count}/{num_models_to_train})")
                log_messages.append(f"\n{trained_count}/{num_models_to_train}. Entrenando modelo: {name.replace('_', ' ').title()}...")

                y_target = y_target_series.copy()

                # Ensure correct dtype for model
                if config['type'] == 'classification':
                    try: y_target = y_target.astype(int)
                    except ValueError as e: log_messages.append(f"Error convirtiendo '{target_col_name}' a int: {e}. Saltando."); continue
                elif config['type'] == 'regression':
                    try: y_target = y_target.astype(float)
                    except ValueError as e: log_messages.append(f"Error convirtiendo '{target_col_name}' a float: {e}. Saltando."); continue

                if y_target.nunique() <= 1 and config['type'] != 'regression': # Regression can have 1 unique value
                     log_messages.append(f"Advertencia: Objetivo '{target_col_name}' tiene <= 1 valor único ({y_target.unique()}).")
                     # Don't skip, model might still be useful if it predicts that single value

                # Stratification logic
                stratify_target = None
                if config['type'] == 'classification' and y_target.nunique() > 1 :
                     min_class_count = y_target.value_counts().min()
                     # Check if test_size * total_samples < number of classes
                     n_splits_required = 2 # Default for train_test_split
                     if min_class_count >= n_splits_required :
                          stratify_target = y_target
                     else:
                          log_messages.append(f"Advertencia: No se puede estratificar '{name}', clase minoritaria ({min_class_count}) < n_splits ({n_splits_required}).")

                X_input_for_split = X_processed_df.values
                y_target_for_split = y_target.values

                try:
                     X_train, X_test, y_train, y_test = train_test_split(
                         X_input_for_split, y_target_for_split, test_size=self.TEST_SIZE,
                         random_state=self.RANDOM_STATE, stratify=stratify_target)
                except ValueError as split_err:
                     log_messages.append(f"Error en split (posiblemente estratificado): {split_err}. Intentando sin estratificación...")
                     X_train, X_test, y_train, y_test = train_test_split(
                         X_input_for_split, y_target_for_split, test_size=self.TEST_SIZE,
                         random_state=self.RANDOM_STATE, stratify=None)

                # Define model
                model = None
                common_params = {'n_estimators': self.N_ESTIMATORS, 'max_depth': self.MAX_DEPTH,
                                 'min_samples_split': self.MIN_SAMPLES_SPLIT, 'min_samples_leaf': self.MIN_SAMPLES_LEAF,
                                 'random_state': self.RANDOM_STATE, 'n_jobs': self.N_JOBS}
                try:
                    if config['type'] == 'classification':
                        # Use balanced for imbalanced classes, otherwise None might be better
                        class_weight = 'balanced' if y_target.nunique() > 1 and y_target.value_counts(normalize=True).min() < 0.4 else None
                        model = RandomForestClassifier(**common_params, max_features=self.MAX_FEATURES, class_weight=class_weight)
                    elif config['type'] == 'regression':
                         model = RandomForestRegressor(**common_params, max_features=None) # Default for RF Regressor is all features
                    else:
                        log_messages.append(f"Error: Tipo de modelo desconocido '{config['type']}'. Saltando."); continue

                    # Train and evaluate
                    model.fit(X_train, y_train)
                    temp_models[name] = model # Store in temp dict first
                    log_messages.append(f"Modelo {name.replace('_', ' ').title()} Entrenado.")

                    y_pred = model.predict(X_test)
                    log_messages.append("Evaluación Test Set:")
                    if config['type'] == 'classification':
                        acc = accuracy_score(y_test, y_pred); f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        log_messages.append(f"     Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)
                            # Ensure probabilities match shape of y_test classes for log_loss
                            try:
                                log_messages.append(f"    Log Loss: {log_loss(y_test, y_proba, labels=model.classes_):.4f}")
                            except ValueError as logloss_err:
                                log_messages.append(f"    Log Loss: N/A ({logloss_err})")

                            # AUC only makes sense for binary or one-vs-rest multiclass
                            if len(model.classes_) == 2:
                                 try: log_messages.append(f"    AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
                                 except ValueError as auc_err: log_messages.append(f"    AUC: N/A ({auc_err})")
                            elif len(model.classes_) > 2:
                                try:
                                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted', labels=model.classes_)
                                    log_messages.append(f"    AUC (ovr, weighted): {auc:.4f}")
                                except ValueError as auc_err: log_messages.append(f"    AUC (ovr, weighted): N/A ({auc_err})")


                    elif config['type'] == 'regression':
                        mae = mean_absolute_error(y_test, y_pred); mse = mean_squared_error(y_test, y_pred); rmse = np.sqrt(mse)
                        log_messages.append(f"    MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

                except Exception as train_eval_e:
                     log_messages.append(f"ERROR durante entrenamiento/evaluación de {name}: {train_eval_e}")
                     # traceback.print_exc() # Avoid printing traceback directly to UI

            self.models = temp_models # Assign to session state *after* loop finishes

            if not self.models:
                 st.error("¡Error! No se entrenó ningún modelo. Verifica los datos y configuraciones.")
                 log_messages.append("\n--- ¡Error! No se entrenó ningún modelo. ---")
            else:
                 st.success(f"¡Entrenamiento de {len(self.models)} modelos RandomForest completado!")
                 log_messages.append(f"\n--- ¡Entrenamiento de {len(self.models)} modelos RandomForest completado! ---")

        except Exception as e:
            st.error(f"Error durante el entrenamiento de modelos: {e}")
            log_messages.append(f"\nERROR FATAL en entrenamiento: {e}")
            traceback.print_exc(file=io.StringIO()) # Log traceback to string
            log_messages.append(traceback.format_exc())
            self.models = {} # Reset models on error
        finally:
             with log_area:
                  st.code("\n".join(log_messages), language='text')


    # --- Helper Functions ---
    def calculate_over_probability(self, threshold: float, predicted_total_goals: float) -> float:
        K = 0.8 # Sigmoid steepness factor
        try: predicted_total_goals = float(predicted_total_goals)
        except (ValueError, TypeError): return 0.5 # Default probability if prediction is invalid
        # Sigmoid function centered around the threshold
        probability = 1 / (1 + np.exp(-K * (predicted_total_goals - threshold)))
        return np.clip(probability, 0.01, 0.99) # Clip probabilities to avoid extremes

    def get_match_history(self, home_team: str, away_team: str) -> str:
        if self.data is None or self.data.empty: return "<p>No hay datos históricos disponibles.</p>"
        try:
            if 'hometeam' not in self.data.columns or 'awayteam' not in self.data.columns:
                return "<p>Columnas 'hometeam'/'awayteam' no encontradas.</p>"
            history = self.data[
                ((self.data['hometeam'] == home_team) & (self.data['awayteam'] == away_team)) |
                ((self.data['hometeam'] == away_team) & (self.data['awayteam'] == home_team))
            ].sort_values(by='date', ascending=False).head(10)
        except Exception as e: return f"<p>Error buscando historial: {e}</p>"
        if history.empty: return "<p>No hay historial reciente de partidos directos.</p>"

        history_html = """<h3 class="card-subtitle">📜 Historial Directo (Últimos 10)</h3><div class="table-responsive"><table class="results-table"><thead><tr><th>Fecha</th><th>Local</th><th>Visitante</th><th>G.L.</th><th>G.V.</th><th>Res.</th></tr></thead><tbody>"""
        for _, row in history.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
            home_goals = int(row['fthg']) if pd.notna(row['fthg']) else '-'; away_goals = int(row['ftag']) if pd.notna(row['ftag']) else '-'
            result = row['ftr'] if pd.notna(row['ftr']) else '-'
            ht = row.get('hometeam', 'N/A'); at = row.get('awayteam', 'N/A')
            history_html += f"""<tr><td>{date_str}</td><td>{ht}</td><td>{at}</td><td class="text-center">{home_goals}</td><td class="text-center">{away_goals}</td><td class="text-center">{result}</td></tr>"""
        history_html += """</tbody></table></div>"""
        return history_html

    def get_recent_matches(self, team: str, num_matches: int = 10) -> str:
        if self.data is None or self.data.empty: return f"<p>No hay datos históricos para {team}.</p>"
        try:
            if 'hometeam' not in self.data.columns or 'awayteam' not in self.data.columns:
                return f"<p>Columnas de equipo no encontradas para {team}.</p>"
            team_matches = self.data[(self.data['hometeam'] == team) | (self.data['awayteam'] == team)].sort_values(by='date', ascending=False).head(num_matches)
        except Exception as e: return f"<p>Error buscando partidos recientes para {team}: {e}</p>"
        if team_matches.empty: return f"<p>No hay partidos recientes para {team}.</p>"

        recent_matches_html = f"""<h3 class="card-subtitle">📅 Últimos {num_matches} Partidos de {team}</h3><div class="table-responsive"><table class="results-table"><thead><tr><th>Fecha</th><th>Local</th><th>Visitante</th><th>G.L.</th><th>G.V.</th><th>Res.</th></tr></thead><tbody>"""
        for _, row in team_matches.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
            home_goals = int(row['fthg']) if pd.notna(row['fthg']) else '-'; away_goals = int(row['ftag']) if pd.notna(row['ftag']) else '-'
            result = row['ftr'] if pd.notna(row['ftr']) else '-'
            ht = row.get('hometeam', 'N/A'); at = row.get('awayteam', 'N/A')
            # Highlight the team in the row
            home_td_style = "font-weight: 600; color: var(--text-color-highlight);" if ht == team else ""
            away_td_style = "font-weight: 600; color: var(--text-color-highlight);" if at == team else ""
            recent_matches_html += f"""<tr><td>{date_str}</td><td style="{home_td_style}">{ht}</td><td style="{away_td_style}">{at}</td><td class="text-center">{home_goals}</td><td class="text-center">{away_goals}</td><td class="text-center">{result}</td></tr>"""
        recent_matches_html += """</tbody></table></div>"""
        return recent_matches_html

    def get_top_teams_by_referee(self, referee: str, num_teams: int = 5) -> str:
        if self.data is None or self.data.empty: return "<p>No hay datos de árbitros.</p>"
        if 'referee' not in self.data.columns: return "<p>Columna 'referee' no encontrada.</p>"
        try: matches = self.data[self.data['referee'] == referee]
        except Exception as e: return f"<p>Error filtrando partidos por árbitro '{referee}': {e}</p>"
        if matches.empty: return f"<p>No hay partidos dirigidos por {referee}.</p>"
        if 'ftr' not in matches.columns or 'hometeam' not in matches.columns or 'awayteam' not in matches.columns:
             return f"<p>Faltan columnas ('ftr', 'hometeam', 'awayteam') para calcular victorias.</p>"
        try:
            home_wins = matches[matches['ftr'] == 'H'].groupby('hometeam').size()
            away_wins = matches[matches['ftr'] == 'A'].groupby('awayteam').size()
            total_wins = home_wins.add(away_wins, fill_value=0).astype(int).sort_values(ascending=False)
        except Exception as e: return f"<p>Error calculando victorias por árbitro '{referee}': {e}</p>"
        if total_wins.empty: return f"<p>No se registraron victorias bajo {referee}.</p>"
        top_teams = total_wins.head(num_teams)

        top_teams_html = f"""<h3 class="card-subtitle">🏆 Top {num_teams} equipos con más victorias (Árbitro: {referee})</h3><div class="table-responsive"><table class="results-table simple-table"><thead><tr><th>Equipo</th><th class="text-center">Victorias</th></tr></thead><tbody>"""
        for team_name, wins in top_teams.items(): top_teams_html += f"""<tr><td>{team_name}</td><td class="text-center">{wins}</td></tr>"""
        top_teams_html += """</tbody></table></div>"""
        return top_teams_html

    def _get_recommendation_strength(self, pred_type: str, value: float) -> Tuple[str, str]:
        thresholds = self.CONFIDENCE_THRESHOLDS.get(pred_type, [])
        try: value = float(value)
        except (ValueError, TypeError): return 'Info', 'ℹ️' # Handle non-float inputs
        for threshold, strength, icon in thresholds:
             if value >= threshold: return strength, icon
        # Return lowest confidence if no threshold met
        if thresholds: return thresholds[-1][1], thresholds[-1][2] # ('Baja', '👀') or similar
        else: return 'Info', 'ℹ️' # Default if no thresholds defined

    # --- predict_match Method ---
    def predict_match(self, home_team: str, away_team: str, referee: Optional[str] = None) -> str:
        """Generates detailed HTML predictions for a specific match using trained RF models."""
        log_messages = [] # For potential debugging within Streamlit logs if needed

        # --- Initial Checks ---
        if not self.models: return "<div class='error-box'>Error: Los modelos no están entrenados. Carga datos primero.</div>"
        if not self.preprocessor: return "<div class='error-box'>Error: El preprocesador no está listo. Carga datos primero.</div>"
        if self.data is None or self.data.empty: return "<div class='error-box'>Error: No hay datos cargados para predecir.</div>"
        if not self.features: return "<div class='error-box'>Error: La lista de características está vacía.</div>"

        log_messages.append(f"Iniciando predicción para: {home_team} vs {away_team} (Árbitro: {referee or 'No especificado'})")

        try:
            # --- Validate Inputs against *available* options from state ---
            if home_team not in self.available_teams:
                 raise ValueError(f"Equipo local '{home_team}' no válido o no encontrado en los datos cargados.")
            if away_team not in self.available_teams:
                 raise ValueError(f"Equipo visitante '{away_team}' no válido o no encontrado.")

            # Handle referee input, check against available referees if provided
            referee_to_use = 'Desconocido' # Default if not provided or not found
            if referee: # If a referee name is given
                if referee in self.available_referees:
                     referee_to_use = referee
                else:
                     log_messages.append(f"Advertencia: Árbitro '{referee}' no encontrado en la lista. Usando 'Desconocido'.")
                     # Keep referee_to_use as 'Desconocido' which OHE should handle if trained with it
            else: # No referee provided explicitly
                 # Use 'Missing' to match categorical imputer fill_value, or fallback
                 referee_imputed_value = 'Missing'
                 if referee_imputed_value in self.available_referees:
                     referee_to_use = referee_imputed_value
                 else:
                     # If 'Missing' wasn't seen during training, OHE might ignore it.
                     # Using the most frequent from training or 'Desconocido' might be safer.
                     # For simplicity, we stick to 'Desconocido' as the fallback.
                     log_messages.append(f"Nota: Árbitro no especificado. Usando '{referee_to_use}'.")


            # --- Prepare Input Data ---
            # Start with categorical features
            match_data_dict: Dict[str, Any] = {
                'hometeam': [home_team],
                'awayteam': [away_team],
                'referee': [referee_to_use]
            }

            # Add numeric features using average from training data or 0 as fallback
            if self.numeric_features and self.data is not None:
                try:
                    # Use means calculated during preprocessor fitting if available
                    numeric_transformer_pipeline = self.preprocessor.named_transformers_['num']
                    imputer_step = numeric_transformer_pipeline.named_steps['imputer']
                    means = pd.Series(imputer_step.statistics_, index=self.numeric_features)

                    # Fill missing means with 0 just in case
                    means = means.reindex(self.numeric_features, fill_value=0)
                    for feature in self.numeric_features:
                         match_data_dict[feature] = [means.get(feature, 0)]

                except Exception as e:
                     log_messages.append(f"Advertencia calculando/recuperando medias numéricas: {e}. Usando 0.")
                     for feature in self.numeric_features: match_data_dict[feature] = [0]
            else:
                 for feature in self.numeric_features: match_data_dict[feature] = [0] # Fallback if no numeric features defined


            # Create DataFrame ensuring correct feature order
            try:
                # Ensure all features needed are present in the dict
                for f in self.features:
                     if f not in match_data_dict:
                            # This case should ideally not happen if numeric features are handled above
                            log_messages.append(f"Advertencia: Característica '{f}' faltante en dict, añadiendo 0/Unknown.")
                            match_data_dict[f] = [0] if f in self.numeric_features else ['Unknown']

                match_df = pd.DataFrame(match_data_dict, index=[0])[self.features] # Enforce column order
            except KeyError as e: raise KeyError(f"Error creando DataFrame de predicción. Falta/Discrepancia de característica: {e}.")
            except Exception as df_e: raise Exception(f"Error inesperado creando DataFrame de predicción: {df_e}")

            # Final check for NaNs before transform (should be unlikely now)
            if match_df.isnull().any().any():
                log_messages.append("Advertencia: NaN encontrado en DataFrame ANTES de transformar para predicción. Rellenando.")
                for col in match_df.select_dtypes(include=np.number).columns:
                    match_df[col].fillna(0, inplace=True)
                for col in match_df.select_dtypes(include='object').columns:
                     match_df[col].fillna('Unknown', inplace=True)

            # --- Preprocess ---
            try: match_processed = self.preprocessor.transform(match_df)
            except Exception as transform_e: raise Exception(f"Error durante transformación de datos para predicción: {transform_e}")

            # --- Make Predictions ---
            log_messages.append("Realizando predicciones con RandomForest...")
            predictions = {}
            missing_models = []

            for model_name, model in self.models.items():
                try:
                    if isinstance(model, RandomForestClassifier):
                        pred_proba = model.predict_proba(match_processed)[0]
                        if model_name == 'result': # H=0, D=1, A=2
                            if len(pred_proba) == len(model.classes_): # Check expected number of classes
                                # Create dict mapping class index to probability
                                prob_dict = {cls: prob for cls, prob in zip(model.classes_, pred_proba)}
                                # Ensure H, D, A probabilities are present, default to 1/3 if not
                                predictions[model_name] = np.array([prob_dict.get(0, 1/3), prob_dict.get(1, 1/3), prob_dict.get(2, 1/3)])
                            else:
                                log_messages.append(f"Advertencia: Predicción de 'result' devolvió {len(pred_proba)} clases, esperado {len(model.classes_)}. Usando fallback.")
                                predictions[model_name] = np.array([1/3, 1/3, 1/3]) # Fallback

                        elif model_name == 'ht_result': # Assuming 0=No Home Lead, 1=Home Lead
                             if len(pred_proba) == len(model.classes_):
                                 prob_dict = {cls: prob for cls, prob in zip(model.classes_, pred_proba)}
                                 predictions[model_name] = prob_dict.get(1, 0.5) # Probability of class 1 (Home Lead)
                             else: predictions[model_name] = 0.5 # Fallback

                        elif model_name == 'btts': # Assuming 0=No, 1=Yes
                             if len(pred_proba) == len(model.classes_):
                                 prob_dict = {cls: prob for cls, prob in zip(model.classes_, pred_proba)}
                                 predictions[model_name] = prob_dict.get(1, 0.5) # Probability of class 1 (BTTS=Yes)
                             else: predictions[model_name] = 0.5 # Fallback
                        else: # Other potential classifiers
                             if len(pred_proba) == len(model.classes_):
                                 prob_dict = {cls: prob for cls, prob in zip(model.classes_, pred_proba)}
                                 # Store probability of the positive class (usually 1) if binary
                                 predictions[model_name] = prob_dict.get(1, 0.5) if len(model.classes_) == 2 else pred_proba
                             else: predictions[model_name] = 0.5 # Fallback


                    elif isinstance(model, RandomForestRegressor):
                        pred_value = model.predict(match_processed)[0]
                        # Ensure non-negative predictions for counts/stats
                        predictions[model_name] = max(0, float(pred_value))
                    else:
                         predictions[model_name] = None; missing_models.append(model_name)
                except Exception as model_pred_e:
                    log_messages.append(f"Error prediciendo con modelo '{model_name}': {model_pred_e}")
                    predictions[model_name] = None; missing_models.append(model_name)

            if missing_models: log_messages.append(f"Advertencia: Fallaron predicciones para: {', '.join(missing_models)}")

            essential_models = ['result', 'home_goals', 'away_goals']
            failed_essentials = [m for m in essential_models if predictions.get(m) is None]
            if failed_essentials: raise RuntimeError(f"Fallaron predicciones esenciales: {', '.join(failed_essentials)}. No se puede continuar.")

            # --- Process Predictions & Calculate Derived Values ---
            result_probs = predictions['result'] # Should be a numpy array [P(H), P(D), P(A)]
            # Normalize probabilities if they don't sum to 1
            prob_sum = np.sum(result_probs)
            if not np.isclose(prob_sum, 1.0):
                 log_messages.append(f"Advertencia: Suma de probabilidad de resultado ({prob_sum:.3f}) != 1. Normalizando.")
                 if prob_sum > 1e-9: result_probs = result_probs / prob_sum
                 else: result_probs = np.array([1/3, 1/3, 1/3]) # Avoid division by zero

            pred_home_goals = float(predictions.get('home_goals', 0))
            pred_away_goals = float(predictions.get('away_goals', 0))
            pred_total_goals = pred_home_goals + pred_away_goals
            # Round goals for display, but use raw predictions for other calcs
            rounded_home_goals = int(round(pred_home_goals))
            rounded_away_goals = int(round(pred_away_goals))

            # Get probabilities from predictions dict, default to 0.5 if missing
            prob_ht_lead = float(predictions.get('ht_result', 0.5)) # P(Home leads HT)
            prob_btts = float(predictions.get('btts', 0.5)); prob_no_btts = 1.0 - prob_btts # P(BTTS=Yes)
            prob_over_25 = self.calculate_over_probability(self.OVER_UNDER_THRESHOLD, pred_total_goals)
            prob_under_25 = 1.0 - prob_over_25

            # Get stat predictions, default to 0 if missing
            pred_yellow_cards = float(predictions.get('yellow_cards', 0))
            pred_total_corners = float(predictions.get('total_corners', 0))
            pred_total_shots = float(predictions.get('total_shots', 0))
            pred_total_shots_on_target = float(predictions.get('total_shots_on_target', 0))
            pred_total_fouls = float(predictions.get('total_fouls', 0))
            pred_total_red_cards = float(predictions.get('total_red_cards', 0))

            # --- Generate Recommendations ---
            log_messages.append("Generando recomendaciones...")
            recommendations = []
            rec_map = {0: f"Gana {home_team}", 1: "Empate", 2: f"Gana {away_team}"}

            # Main Result Recommendation
            main_bet_idx = np.argmax(result_probs) # 0, 1, or 2
            main_bet_conf = result_probs[main_bet_idx]
            strength, icon = self._get_recommendation_strength('result', main_bet_conf)
            recommendations.append({'type': 'Resultado Final', 'prediction': rec_map[main_bet_idx], 'confidence_val': main_bet_conf, 'confidence_disp': f"{main_bet_conf:.1%}", 'strength': strength, 'icon': icon})

            # Over/Under Recommendation
            if prob_over_25 > prob_under_25:
                 strength, icon = self._get_recommendation_strength('over_under', prob_over_25)
                 recommendations.append({'type': f'Goles Totales', 'prediction': f"Más de {self.OVER_UNDER_THRESHOLD}", 'confidence_val': prob_over_25, 'confidence_disp': f"{prob_over_25:.1%}", 'strength': strength, 'icon': icon})
            else:
                strength, icon = self._get_recommendation_strength('over_under', prob_under_25)
                recommendations.append({'type': f'Goles Totales', 'prediction': f"Menos de {self.OVER_UNDER_THRESHOLD}", 'confidence_val': prob_under_25, 'confidence_disp': f"{prob_under_25:.1%}", 'strength': strength, 'icon': icon})

            # BTTS Recommendation
            if prob_btts > prob_no_btts:
                strength, icon = self._get_recommendation_strength('btts', prob_btts)
                recommendations.append({'type': 'Ambos Marcan', 'prediction': "Sí 👍", 'confidence_val': prob_btts, 'confidence_disp': f"{prob_btts:.1%}", 'strength': strength, 'icon': icon})
            else:
                 strength, icon = self._get_recommendation_strength('btts', prob_no_btts)
                 recommendations.append({'type': 'Ambos Marcan', 'prediction': "No 👎", 'confidence_val': prob_no_btts, 'confidence_disp': f"{prob_no_btts:.1%}", 'strength': strength, 'icon': icon})

            # HT Result Recommendation (only if probability is reasonably confident)
            if abs(prob_ht_lead - 0.5) > 0.10: # Only recommend if P > 60% or < 40%
                 ht_pred_text = f"{home_team} Lidera 1T" if prob_ht_lead > 0.5 else f"{home_team} NO Lidera 1T"
                 ht_conf = max(prob_ht_lead, 1.0 - prob_ht_lead)
                 strength, icon = self._get_recommendation_strength('ht_result', ht_conf)
                 recommendations.append({'type': 'Resultado 1T', 'prediction': ht_pred_text, 'confidence_val': ht_conf, 'confidence_disp': f"{ht_conf:.1%}", 'strength': strength, 'icon': icon})

            # Stat Predictions (presented as estimations)
            stat_predictions = [
                {'type': 'Tarjetas Amarillas', 'value': pred_yellow_cards, 'key': 'yellow_cards', 'icon': '🟨'},
                {'type': 'Córners Totales', 'value': pred_total_corners, 'key': 'corners', 'icon': '🚩'},
                {'type': 'Tiros Totales', 'value': pred_total_shots, 'key': 'total_shots', 'icon': '🎯'},
                {'type': 'Tiros a Puerta', 'value': pred_total_shots_on_target, 'key': 'total_shots_on_target', 'icon': '⚽'},
                {'type': 'Faltas Totales', 'value': pred_total_fouls, 'key': 'total_fouls', 'icon': '✋'},
                {'type': 'Tarjetas Rojas', 'value': pred_total_red_cards, 'key': 'total_red_cards', 'icon': '🟥'},
            ]

            for stat in stat_predictions:
                 if stat['key'] in predictions and predictions[stat['key']] is not None:
                     strength, _ = self._get_recommendation_strength(stat['key'], stat['value'])
                     recommendations.append({'type': stat['type'], 'prediction': f"~{stat['value']:.1f} {stat['icon']}",
                                             'confidence_val': -1, # Sort stats below probability bets
                                             'confidence_disp': f"Estimación: {stat['value']:.1f}",
                                             'strength': strength, 'icon': stat['icon'], 'is_stat': True})

            recommendations.sort(key=lambda x: x['confidence_val'], reverse=True)

            # --- Generate HTML Output ---
            log_messages.append("Generando HTML...")
            # Add theme class to the main container
            html_output = f"<div class='predictor-container theme-{st.session_state.theme}'>"
            html_output += f"""
                <div class="predictor-header">
                    <h2>{home_team} <span style="font-weight: 400;">vs</span> {away_team}</h2>
                    <p>Análisis y Predicciones con RandomForest | Iqscore</p>
                </div>
                <div class="predictor-content">
                    <div class="card">
                        <div class="card-header"><i class="fas fa-chart-pie"></i> Probabilidades y Marcador Estimado</div>
                        <div class="card-content">
                             <div class="prediction-grid">
                                <div class="stat-box"><div class="team-name">{home_team} (Local)</div><div class="probability prob-home">{result_probs[0]:.1%} <i class="fas fa-home"></i></div></div>
                                <div class="stat-box"><div class="team-name">Empate</div><div class="probability prob-draw">{result_probs[1]:.1%} <i class="fas fa-handshake"></i></div></div>
                                <div class="stat-box"><div class="team-name">{away_team} (Visitante)</div><div class="probability prob-away">{result_probs[2]:.1%} <i class="fas fa-plane-departure"></i></div></div>
                             </div>
                             <div style="text-align: center; padding-top: 15px; font-size: 1em; font-weight: 600;" class="estimated-score">⚽ Marcador Estimado (Regresión): {rounded_home_goals} - {rounded_away_goals} 🥅</div>
                             <div class='explanation-note'><strong>Nota Importante:</strong> Las probabilidades (Local/Empate/Visitante) y el Marcador Estimado provienen de modelos diferentes entrenados en los mismos datos. El marcador es una estimación directa de goles, mientras que las probabilidades reflejan la confianza del modelo en cada uno de los tres resultados posibles (1X2). Pueden no coincidir perfectamente (p.ej., alta prob. de Empate pero marcador 1-0). Use ambas como guías complementarias.</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header"><i class="fas fa-star"></i> ¡Picks Recomendados!</div>
                        <div class="card-content">
                          <div class="table-responsive"><table class="results-table"><thead><tr><th>Mercado</th><th>Predicción Sugerida</th><th>Confianza / Estimación</th><th class="text-center">Nivel</th></tr></thead><tbody>
            """
            # Add probability-based recommendations
            for rec in recommendations:
                 if not rec.get('is_stat', False):
                    strength_class = f"strength-{rec['strength']}"
                    html_output += f"""<tr><td>{rec['type']}</td><td>{rec['prediction']}</td><td class="text-center">{rec['confidence_disp']}</td><td class="text-center strength-cell"><span class="{strength_class}">{rec['icon']} {rec['strength']}</span></td></tr>"""

            html_output += """</tbody></table><p class="disclaimer">*Apuesta con responsabilidad. Las predicciones son estimaciones basadas en datos históricos y no garantizan resultados.</p></div></div></div>""" # Close picks card

            # Stats and History Card
            html_output += """
                    <div class="card">
                        <div class="card-header"><i class="fas fa-book-open"></i> Estadísticas Clave y Forma</div>
                        <div class="card-content">
                            <h3 class="card-subtitle">📊 Otras Estadísticas Estimadas</h3><ul class="predicted-stats-list">"""
            stats_added = 0
            # Add stat-based estimations to the list
            for rec in recommendations:
                 if rec.get('is_stat', True): # Include stats here
                     strength_class = f"strength-{rec['strength']}" # Use strength class even for stats
                     html_output += f'<li><span>{rec["type"]}</span> <span class="strength-cell"><span class="{strength_class}">{rec["prediction"]} ({rec["strength"]})</span></span></li>'
                     stats_added += 1
            if stats_added == 0: html_output += "<li>No hay estimaciones estadísticas disponibles.</li>"
            html_output += "</ul><hr class='section-divider'>"

            # Add History, Recent Form, Referee Stats
            html_output += self.get_match_history(home_team, away_team)
            html_output += "<hr class='section-divider'>"
            html_output += self.get_recent_matches(home_team)
            html_output += "<hr class='section-divider'>"
            html_output += self.get_recent_matches(away_team)
            # Only add referee stats if a valid referee was found and used
            if referee and referee_to_use != 'Desconocido' and referee_to_use != 'Missing':
                 try:
                     if 'referee' in self.data.columns and referee_to_use in self.data['referee'].unique():
                        html_output += "<hr class='section-divider'>"
                        html_output += self.get_top_teams_by_referee(referee_to_use)
                     else: log_messages.append(f"Nota: Árbitro '{referee_to_use}' no encontrado para stats (posiblemente filtrado).")
                 except Exception as ref_e: log_messages.append(f"Advertencia: No se pudieron generar stats del árbitro '{referee_to_use}': {ref_e}")

            html_output += """</div></div></div></div>""" # Close stats card, content, container
            log_messages.append("Predicción HTML generada exitosamente.")
            # Optional: Log messages to Streamlit console if needed
            # print("\n".join(log_messages))
            return html_output

        except ValueError as ve: return f"<div class='error-box'>Error en la predicción: {str(ve)}</div>"
        except KeyError as ke: return f"<div class='error-box'>Error: Falta información o columna ({str(ke)}). Revisa los datos cargados.</div>"
        except RuntimeError as re: return f"<div class='error-box'>Error Crítico en Predicción: {str(re)}</div>"
        except Exception as e:
            # Log detailed error for debugging in Streamlit's console/logs
            print(f"Error inesperado en predict_match: {e}")
            traceback.print_exc()
            return f"<div class='error-box'>Error inesperado generando la predicción. Revisa la consola para detalles: {str(e)}</div>"

# --- Initialize predictor (accesses/creates session state) ---
predictor = StreamlitFootballRandomForestPredictor()

# --- CSS Function to get theme-specific styles ---
def get_themed_css(theme):
    # Base styles (common to both) - Using Poppins as requested
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'); /* Font Awesome */

        /* Apply Poppins globally */
        body, .stApp, input, textarea, button, select, p, div, span, li, td, th, h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
        }

        /* General body/app styling */
        body {
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .stApp { /* Target the main app container */
            background-color: var(--app-bg);
            color: var(--text-color);
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            color: var(--header-text); /* Use specific header text color */
            transition: color 0.3s ease;
        }
        p, div, span, li, td, th {
            transition: color 0.3s ease, background-color 0.3s ease;
        }

        /* --- Predictor Output Specific Styles --- */
        .predictor-container {
            max-width: 950px; margin: 25px auto; border-radius: 15px;
            box-shadow: 0 6px 20px var(--shadow-color);
            overflow: hidden; border: 1px solid var(--border-color);
            background-color: var(--container-bg);
            transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .predictor-header {
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); color: #ffffff; /* Keep original gradient */
            padding: 25px 30px; text-align: center; border-bottom: 5px solid #fdbb2d; /* Keep original border */
        }
        .predictor-header h2 { margin: 0 0 8px 0; font-size: 2.1em; font-weight: 700; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); color: #ffffff; } /* Ensure header text is white */
        .predictor-header p { margin: 0; color: #f0f0f0; font-style: normal; font-size: 1.05em; }
        .predictor-content { padding: 30px; background-color: var(--content-bg); transition: background-color 0.3s ease; }
        .card {
             background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; margin-bottom: 30px;
             box-shadow: 0 4px 15px var(--shadow-color-light); overflow: hidden;
             transition: transform 0.2s ease-in-out, background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
             transform: translateY(-4px);
             box-shadow: 0 8px 25px var(--shadow-color);
        }
        .card-header {
            background: linear-gradient(to right, var(--card-header-bg-start), var(--card-header-bg-end)); color: var(--card-header-text);
            padding: 15px 20px; border-bottom: 1px solid var(--border-color); font-size: 1.3em; font-weight: 600; border-radius: 12px 12px 0 0; display: flex; align-items: center;
            transition: border-color 0.3s ease, background 0.3s ease, color 0.3s ease;
        }
        .card-header i.fas { margin-right: 10px; font-size: 1.1em; vertical-align: middle;}
        .card-content { padding: 25px; }
        .card-subtitle {
            font-size: 1.15em; font-weight: 600; color: var(--text-color); margin-top: 0; margin-bottom: 18px;
            border-bottom: 2px solid var(--primary-accent); padding-bottom: 8px;
            transition: color 0.3s ease, border-color 0.3s ease;
         }
        .prediction-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; text-align: center; margin: 20px 0; padding: 15px 0; border-top: 1px solid var(--border-light); border-bottom: 1px solid var(--border-light); transition: border-color 0.3s ease; }
        .stat-box { padding: 15px; background-color: var(--stat-box-bg); border-radius: 8px; transition: background-color 0.3s ease; box-shadow: inset 0 1px 2px var(--shadow-color-inset); }
        .stat-box .team-name { font-weight: 600; margin-bottom: 8px; font-size: 1.1em; color: var(--text-muted); word-wrap: break-word; transition: color 0.3s ease; }
        .stat-box .probability { font-size: 2.8em; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; justify-content: center; transition: color 0.3s ease; }
        .stat-box .probability i.fas { font-size: 0.7em; margin-left: 8px; vertical-align: middle;}
        .prob-home { color: var(--prob-home-color); } .prob-draw { color: var(--prob-draw-color); } .prob-away { color: var(--prob-away-color); }
        .results-table { width: 100%; border-collapse: separate; border-spacing: 0 5px; font-size: 0.98em; margin-bottom: 15px; }
        .results-table th, .results-table td { border-bottom: 1px solid var(--border-color); padding: 12px 15px; text-align: left; vertical-align: middle; color: var(--text-color); transition: border-color 0.3s ease, color 0.3s ease, background-color 0.3s ease; }
        .results-table th { background-color: var(--table-header-bg); font-weight: 600; color: var(--text-muted); border-top: 1px solid var(--border-color); }
        .results-table tbody tr { background-color: var(--table-row-bg); border-radius: 5px; transition: background-color 0.2s ease; }
        .results-table tbody tr:hover { background-color: var(--table-row-hover-bg); }
        .results-table td:first-child { border-left: 3px solid transparent; }
        .results-table tr:hover td:first-child { border-left-color: var(--primary-accent); }
        .results-table .text-center { text-align: center; }
        .strength-cell span { display: inline-block; padding: 5px 12px; border-radius: 15px; font-weight: 600; font-size: 0.9em; white-space: nowrap; transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; }
        .strength-Fuerte { background-color: var(--strength-fuerte-bg); color: var(--strength-fuerte-text); border: 1px solid var(--strength-fuerte-border); }
        .strength-Moderada { background-color: var(--strength-moderada-bg); color: var(--strength-moderada-text); border: 1px solid var(--strength-moderada-border); }
        .strength-Baja { background-color: var(--strength-baja-bg); color: var(--strength-baja-text); border: 1px solid var(--strength-baja-border); }
        .strength-Alto { background-color: var(--strength-alto-bg); color: var(--strength-alto-text); border: 1px solid var(--strength-alto-border); }
        .strength-Medio { background-color: var(--strength-medio-bg); color: var(--strength-medio-text); border: 1px solid var(--strength-medio-border); }
        .strength-Info { background-color: var(--strength-info-bg); color: var(--strength-info-text); border: 1px solid var(--strength-info-border); }
        .table-responsive { overflow-x: auto; padding-bottom: 5px;}
        .simple-table th { background-color: var(--table-header-bg); } .simple-table td { background-color: var(--table-row-bg); }
        .error-box { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; padding: 18px; border-radius: 8px; margin: 20px; text-align: center; font-weight: bold; font-size: 1.1em;} /* Keep error box light for visibility */
        .predicted-stats-list { list-style: none; padding-left: 0; margin-top: 15px; }
        .predicted-stats-list li { background-color: var(--stat-list-item-bg); margin-bottom: 8px; padding: 10px 15px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 0.95em; transition: background-color 0.3s ease; }
        .predicted-stats-list li span:first-child { font-weight: 500; color: var(--text-muted); transition: color 0.3s ease;}
        .predicted-stats-list li span.strength-cell { font-weight: 600; }
        hr.section-divider { margin: 30px 0; border: none; border-top: 1px solid var(--border-light); transition: border-color 0.3s ease; }
        .explanation-note { font-size: 0.9em; color: var(--text-muted); background-color: var(--note-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; margin-top: 15px; margin-bottom: 10px; text-align: justify; line-height: 1.5; transition: color 0.3s ease, background-color 0.3s ease, border-color 0.3s ease;}
        .explanation-note strong { color: var(--text-color); transition: color 0.3s ease;}
        .disclaimer { font-size: 0.85em; text-align: center; color: var(--text-muted); margin-top: 15px; }

        /* Estimated score specific class for theming */
        .estimated-score { color: var(--text-muted); transition: color 0.3s ease; }

        /* --- Sidebar Styles --- */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            transition: background-color 0.3s ease;
            border-right: 1px solid var(--border-color);
        }
         /* Style sidebar header */
        [data-testid="stSidebar"] .sidebar-header {
             font-size: 1.4em;
             font-weight: 700;
             padding: 15px 20px;
             margin: 0 -20px 10px -20px; /* Extend background */
             border-bottom: 1px solid var(--border-color);
             background-color: var(--sidebar-header-bg);
             color: var(--header-text);
             transition: color 0.3s ease, border-color 0.3s ease, background-color 0.3s ease;
         }
         [data-testid="stSidebar"] .sidebar-description {
             font-size: 0.95em;
             padding: 0 0px 15px 0px; /* Adjust padding */
             color: var(--sidebar-text);
             line-height: 1.6;
             transition: color 0.3s ease;
         }
         [data-testid="stSidebar"] .sidebar-description b {
            color: var(--sidebar-text-bold);
            transition: color 0.3s ease;
         }
         /* Style sidebar status box */
         [data-testid="stSidebar"] .sidebar-status {
             padding: 10px 15px;
             margin: 10px 0px; /* Adjust margin */
             border-radius: 5px;
             font-weight: 500;
             text-align: center;
             transition: background-color 0.3s ease, color 0.3s ease;
             border: 1px solid transparent; /* Add base border */
         }
         .status-success { background-color: var(--status-success-bg); color: var(--status-success-text); border-color: var(--status-success-border); }
         .status-warning { background-color: var(--status-warning-bg); color: var(--status-warning-text); border-color: var(--status-warning-border); }
         .status-info { background-color: var(--status-info-bg); color: var(--status-info-text); border-color: var(--status-info-border); }

        /* --- Streamlit Specific Overrides --- */
        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            border: 1px solid var(--button-border-color);
            background-color: var(--button-bg);
            color: var(--button-text);
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
         .stButton>button:hover {
             border-color: var(--button-hover-border-color);
             background-color: var(--button-hover-bg);
             color: var(--button-hover-text);
         }
         .stButton>button:active {
             border-color: var(--button-active-border-color);
             background-color: var(--button-active-bg);
             color: var(--button-active-text);
         }
         .stButton>button:disabled {
             border-color: var(--button-disabled-border-color);
             background-color: var(--button-disabled-bg);
             color: var(--button-disabled-text);
             opacity: 0.65;
         }
         /* Primary Button */
         .stButton>button[kind="primary"] {
             border-color: var(--button-primary-border-color);
             background-color: var(--button-primary-bg);
             color: var(--button-primary-text);
         }
         .stButton>button[kind="primary"]:hover {
             border-color: var(--button-primary-hover-border-color);
             background-color: var(--button-primary-hover-bg);
             color: var(--button-primary-hover-text);
         }
          .stButton>button[kind="primary"]:active {
             border-color: var(--button-primary-active-border-color);
             background-color: var(--button-primary-active-bg);
             color: var(--button-primary-active-text);
         }
         .stButton>button[kind="primary"]:disabled {
             border-color: var(--button-primary-disabled-border-color);
             background-color: var(--button-primary-disabled-bg);
             color: var(--button-primary-disabled-text);
         }


        /* Inputs & Labels */
        .stSelectbox label, .stFileUploader label, .stTextInput label {
             color: var(--text-color);
             transition: color 0.3s ease;
             font-weight: 600; /* Make labels slightly bolder */
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div { /* Target inner divs for bg/border */
             background-color: var(--input-bg);
             border-color: var(--input-border-color) !important; /* Use important to override base styles */
             transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        /* Input text color */
        [data-baseweb="select"] input, [data-baseweb="input"] input, [data-baseweb="input"] textarea, .stSelectbox div[data-baseweb="tag"] {
             color: var(--text-color) !important;
              transition: color 0.3s ease;
        }
        /* Selectbox dropdown menu */
         [data-baseweb="popover"] ul[role="listbox"] {
              background-color: var(--dropdown-bg);
              border: 1px solid var(--border-color);
          }
          [data-baseweb="popover"] ul[role="listbox"] li {
               color: var(--text-color);
               background-color: var(--dropdown-item-bg);
           }
           [data-baseweb="popover"] ul[role="listbox"] li:hover {
               background-color: var(--dropdown-item-hover-bg);
           }

        /* Toggle Switch */
        [data-testid="stToggle"] label {
            color: var(--text-color); /* Ensure toggle label follows theme */
        }
        [data-testid="stCaption"] {
            color: var(--text-muted); /* Theme the caption */
        }
        [data-testid="stDivider"] {
             border-top-color: var(--border-light); /* Theme dividers */
         }
         [data-testid="stSpinner"] > div > div{
              border-top-color: var(--primary-accent); /* Theme spinner color */
         }
         [data-testid="stExpander"] summary { /* Style expander header */
             background-color: var(--expander-header-bg);
             border: 1px solid var(--border-color);
             color: var(--text-color);
          }
          [data-testid="stExpander"] summary:hover {
              background-color: var(--expander-header-hover-bg);
          }
          [data-testid="stExpander"] summary svg { /* Expander arrow color */
               fill: var(--text-color);
           }

    """

    # Theme-specific variables
    if theme == 'dark':
        # Improved Dark Theme Palette (Example: Dracula-inspired)
        theme_css = """
        :root {
            --app-bg: #282a36; /* Dark background */
            --sidebar-bg: #21222c; /* Slightly darker sidebar */
            --sidebar-header-bg: #191a21;
            --container-bg: #3a3c4e; /* Card/container background */
            --content-bg: #282a36; /* Background inside containers */
            --card-bg: #44475a; /* Individual card background */
            --stat-box-bg: #3a3c4e; /* Background for stat boxes */
            --stat-list-item-bg: #44475a; /* Background for list items */
            --table-header-bg: #3a3c4e;
            --table-row-bg: #44475a;
            --table-row-hover-bg: #5a5c72;
            --note-bg: #3a3c4e;
            --input-bg: #3a3c4e;
            --dropdown-bg: #3a3c4e;
            --dropdown-item-bg: #44475a;
            --dropdown-item-hover-bg: #5a5c72;
            --expander-header-bg: #44475a;
            --expander-header-hover-bg: #5a5c72;

            --text-color: #f8f8f2; /* Light text */
            --text-muted: #bd93f9; /* Purple for muted/secondary text */
            --header-text: #f8f8f2; /* Light text for headers */
            --sidebar-text: #e0e0e0;
            --sidebar-text-bold: #f8f8f2;
            --card-header-text: #ffffff;
            --button-text: #f8f8f2;
            --button-hover-text: #ffffff;
            --button-active-text: #ffffff;
            --button-disabled-text: #aaaaaa;
            --button-primary-text: #ffffff;
            --button-primary-hover-text: #ffffff;
            --button-primary-active-text: #ffffff;
            --button-primary-disabled-text: #bbbbbb;


            --border-color: #6272a4; /* Muted blue/purple border */
            --border-light: #44475a; /* Lighter border for dividers */
            --input-border-color: #6272a4;
            --button-border-color: #6272a4;
            --button-hover-border-color: #bd93f9;
            --button-active-border-color: #ff79c6;
            --button-disabled-border-color: #44475a;
            --button-primary-border-color: #bd93f9;
            --button-primary-hover-border-color: #ff79c6;
            --button-primary-active-border-color: #ff79c6;
            --button-primary-disabled-border-color: #6272a4;


            --primary-accent: #ff79c6; /* Pink accent */
            --card-header-bg-start: #4d5066;
            --card-header-bg-end: #6272a4;

            --prob-home-color: #8be9fd; /* Cyan */
            --prob-draw-color: #f1fa8c; /* Yellow */
            --prob-away-color: #ffb86c; /* Orange */

            --shadow-color: rgba(0, 0, 0, 0.3);
            --shadow-color-light: rgba(0, 0, 0, 0.2);
            --shadow-color-inset: rgba(0, 0, 0, 0.1);

             /* Button Backgrounds */
             --button-bg: #6272a4;
             --button-hover-bg: #7e8cc4;
             --button-active-bg: #5a5c72;
             --button-disabled-bg: #44475a;
             --button-primary-bg: #bd93f9; /* Purple */
             --button-primary-hover-bg: #d6acff;
             --button-primary-active-bg: #a97de8;
             --button-primary-disabled-bg: #6272a4;


            /* Strength colors (adjust for dark mode contrast) */
            --strength-fuerte-bg: #3b5b47; --strength-fuerte-text: #50fa7b; --strength-fuerte-border: #50fa7b; /* Green */
            --strength-moderada-bg: #6e6032; --strength-moderada-text: #f1fa8c; --strength-moderada-border: #f1fa8c; /* Yellow */
            --strength-baja-bg: #7a3e5e; --strength-baja-text: #ff79c6; --strength-baja-border: #ff79c6; /* Pink */
            --strength-alto-bg: #5b7bb4; --strength-alto-text: #8be9fd; --strength-alto-border: #8be9fd; /* Cyan */
            --strength-medio-bg: #716799; --strength-medio-text: #bd93f9; --strength-medio-border: #bd93f9; /* Purple */
            --strength-info-bg: #6272a4; --strength-info-text: #f8f8f2; --strength-info-border: #f8f8f2; /* Grey/Blue */

            /* Status colors */
            --status-success-bg: #3b5b47; --status-success-text: #50fa7b; --status-success-border: #50fa7b;
            --status-warning-bg: #6e6032; --status-warning-text: #f1fa8c; --status-warning-border: #f1fa8c;
            --status-info-bg: #6272a4; --status-info-text: #f8f8f2; --status-info-border: #bd93f9;
        }
        """
    else: # Light theme (default - refined slightly)
        theme_css = """
        :root {
            --app-bg: #f9f9f9; /* Slightly off-white */
            --sidebar-bg: #f0f2f6;
            --sidebar-header-bg: #e9ecef;
            --container-bg: #ffffff;
            --content-bg: #ffffff;
            --card-bg: #ffffff;
            --stat-box-bg: #f8f9fa;
            --stat-list-item-bg: #f8f9fa;
            --table-header-bg: #f1f3f5;
            --table-row-bg: #fff;
            --table-row-hover-bg: #f8f9fa;
            --note-bg: #f8f9fa;
            --input-bg: #ffffff;
            --dropdown-bg: #ffffff;
            --dropdown-item-bg: #ffffff;
            --dropdown-item-hover-bg: #f0f2f6;
            --expander-header-bg: #f8f9fa;
            --expander-header-hover-bg: #e9ecef;

            --text-color: #212529; /* Darker text */
            --text-muted: #6c757d;
            --header-text: #212529;
            --sidebar-text: #495057;
            --sidebar-text-bold: #212529;
            --card-header-text: #ffffff;
            --button-text: #495057;
            --button-hover-text: #212529;
            --button-active-text: #212529;
            --button-disabled-text: #6c757d;
            --button-primary-text: #ffffff;
            --button-primary-hover-text: #ffffff;
            --button-primary-active-text: #ffffff;
            --button-primary-disabled-text: #ffffff;


            --border-color: #dee2e6;
            --border-light: #e9ecef;
            --input-border-color: #ced4da;
            --button-border-color: #ced4da;
            --button-hover-border-color: #adb5bd;
            --button-active-border-color: #6c757d;
            --button-disabled-border-color: #dee2e6;
            --button-primary-border-color: #6a11cb;
            --button-primary-hover-border-color: #5e0dad;
            --button-primary-active-border-color: #520b94;
            --button-primary-disabled-border-color: #9b5de5;


            --primary-accent: #6a11cb; /* Original Purple */
            --card-header-bg-start: #6a11cb;
            --card-header-bg-end: #2575fc; /* Original Gradient */

            --prob-home-color: #0d6efd; /* Blue */
            --prob-draw-color: #ffc107; /* Amber */
            --prob-away-color: #dc3545; /* Red */

            --shadow-color: rgba(0, 0, 0, 0.1);
            --shadow-color-light: rgba(0, 0, 0, 0.05);
            --shadow-color-inset: rgba(0, 0, 0, 0.03);

             /* Button Backgrounds */
             --button-bg: #ffffff;
             --button-hover-bg: #f8f9fa;
             --button-active-bg: #e9ecef;
             --button-disabled-bg: #ffffff;
             --button-primary-bg: #6a11cb; /* Purple */
             --button-primary-hover-bg: #5e0dad;
             --button-primary-active-bg: #520b94;
             --button-primary-disabled-bg: #9b5de5;


            /* Strength colors */
            --strength-fuerte-bg: #d1e7dd; --strength-fuerte-text: #0f5132; --strength-fuerte-border: #a3cfbb;
            --strength-moderada-bg: #fff3cd; --strength-moderada-text: #664d03; --strength-moderada-border: #ffe69c;
            --strength-baja-bg: #f8d7da; --strength-baja-text: #842029; --strength-baja-border: #f5c2c7;
            --strength-alto-bg: #cfe2ff; --strength-alto-text: #0a367a; --strength-alto-border: #b6d4fe;
            --strength-medio-bg: #e2e3e5; --strength-medio-text: #41464b; --strength-medio-border: #d3d6d8;
            --strength-info-bg: #e2e3e5; --strength-info-text: #41464b; --strength-info-border: #d3d6d8;

            /* Status colors */
            --status-success-bg: #d1e7dd; --status-success-text: #0f5132; --status-success-border: #a3cfbb;
            --status-warning-bg: #fff3cd; --status-warning-text: #664d03; --status-warning-border: #ffe69c;
            --status-info-bg: #cfe2ff; --status-info-text: #0a367a; --status-info-border: #b6d4fe;
        }
        """

    return base_css + theme_css + "</style>"

# --- Inject CSS based on session state ---
st.markdown(get_themed_css(st.session_state.theme), unsafe_allow_html=True)

# --- Sidebar for Data Upload and Training ---
with st.sidebar:

    # --- Theme Toggle ---
    # Use columns for better layout
    col1_theme, col2_theme = st.columns([3,1])
    with col1_theme:
         st.markdown("Modo Visualización:")
    with col2_theme:
        # Use label_visibility="collapsed" to hide the default label
        is_dark = st.toggle("Modo Oscuro", value=(st.session_state.theme == 'dark'), key="theme_toggle", label_visibility="collapsed")

    # Update theme based on toggle state
    if is_dark and st.session_state.theme != 'dark':
        st.session_state.theme = 'dark'
        st.rerun() # Rerun to apply theme immediately
    elif not is_dark and st.session_state.theme != 'light':
        st.session_state.theme = 'light'
        st.rerun() # Rerun to apply theme immediately


    st.divider()

    # --- Header ---
    st.markdown('<div class="sidebar-header"><i class="fas fa-cogs"></i> Iqscore Predictor </div>', unsafe_allow_html=True)

    # --- Description ---
    st.markdown("""
    <div class="sidebar-description">
    Bienvenido al predictor de partidos de fútbol. <br>
    1️⃣ <b>Carga tus datos históricos</b> (CSV/Excel) con resultados y estadísticas de partidos. <br>
    2️⃣ El bot <b>entrenará modelos</b> de Machine Learning (RandomForest). <br>
    3️⃣ Una vez entrenado, <b>selecciona los equipos</b> y (opcionalmente) el árbitro en el panel principal para <b>generar un análisis</b> detallado del partido.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "📁 Cargar Archivos (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Sube uno o más archivos con datos históricos de partidos. Columnas comunes: hometeam, awayteam, fthg, ftag, ftr, date, hs, as, etc."
    )

    # --- Processing Logic ---
    process_button_disabled = not uploaded_files
    if uploaded_files:
        if st.button("⚙️ Procesar y Entrenar", key="process_button", type="primary", disabled=process_button_disabled, use_container_width=True):
            predictor.load_and_process_uploaded_files(uploaded_files)
    elif predictor.data_loaded:
        # Offer option to clear data/models
        if st.button("🧹 Limpiar Datos y Modelos", key="clear_button", use_container_width=True):
            predictor.reset_state()
            st.rerun() # Rerun to reflect the cleared state

    st.divider()

    # --- Status Display ---
    st.markdown("📊 **Estado del Modelo**", unsafe_allow_html=True)
    if predictor.trained:
        st.markdown('<div class="sidebar-status status-success">✅ Modelos Entrenados y Listos</div>', unsafe_allow_html=True)
    elif predictor.data_loaded:
        st.markdown('<div class="sidebar-status status-warning">⚠️ Datos Cargados, Modelos Pendientes</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-status status-info">⏳ Esperando carga de archivos...</div>', unsafe_allow_html=True)


# --- Main Area for Prediction Input and Output ---
st.title("🤖 Sistema de Predicción de Fútbol ⚽")
st.markdown("Selecciona un partido para obtener análisis detallados, estadísticas y predicciones generadas por modelos RandomForest.")

if not predictor.trained:
    st.warning("👈 Por favor, carga tus datos y entrena los modelos usando el panel lateral izquierdo primero.")
else:
    # --- Prediction Form ---
    st.subheader("🏟️ Selecciona el Partido")
    col1, col2, col3 = st.columns(3)

    with col1:
        home_team = st.selectbox(
            "🏠 Equipo Local:",
            options=predictor.available_teams,
            index=0 if predictor.available_teams else None,
            key="home_team_select"
        )
    with col2:
        # Ensure the default away team is different from home team if possible
        available_away = [t for t in predictor.available_teams if t != home_team]
        away_index = 0 if available_away else None
        away_team = st.selectbox(
            "✈️ Equipo Visitante:",
            options=available_away,
             index=away_index,
            key="away_team_select"
        )
    with col3:
        referee = st.selectbox(
            "👤 Árbitro (Opcional):",
            options=predictor.available_referees,
             index=0, # Default to blank option
            key="referee_select"
        )

    # --- Prediction Button ---
    # Disable button if no teams selected or teams are the same
    predict_button_disabled = not home_team or not away_team or home_team == away_team
    if st.button("⚡ Generar Análisis", key="predict_button", type="primary", disabled=predict_button_disabled):
        if not home_team or not away_team:
             st.error("❌ Debes seleccionar ambos equipos.")
        elif home_team == away_team:
            st.error("❌ Los equipos local y visitante deben ser diferentes.")
        else:
            referee_val = referee if referee else None # Use None if blank selected
            with st.spinner(f"⏳ Analizando {home_team} vs {away_team}..."):
                 prediction_html = predictor.predict_match(home_team, away_team, referee_val)
                 # Display the HTML output
                 st.markdown(prediction_html, unsafe_allow_html=True)
                 st.success("✅ Análisis generado.")

# --- Footer/Info ---
st.divider()
st.caption("⚽ Predictor Iqscore vRF | Adaptado para Streamlit")