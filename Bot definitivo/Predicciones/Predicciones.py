# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# --- NUEVAS IMPORTACIONES ---
try:
    import lightgbm as lgb
except ImportError:
    print("*"*50)
    print("¡Error! La librería LightGBM no está instalada.")
    print("Por favor, instálala ejecutando: pip install lightgbm")
    print("*"*50)
    raise # Detener la ejecución si no está instalada

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, log_loss, f1_score, roc_auc_score
# --- FIN NUEVAS IMPORTACIONES ---
from concurrent.futures import ThreadPoolExecutor
import io
import warnings
from typing import List, Dict, Optional, Any, Tuple
import traceback # Import traceback for better error reporting
import time # Needed for interface delay

# Ignore common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# np.bool handling for compatibility
if hasattr(np, 'bool'):
    if type(np.bool) is type: np.bool = np.bool_ # type: ignore
    else: pass
else: np.bool_ = bool

# Conditional imports for Colab environment
try:
    from google.colab import files
    from IPython.display import display, HTML
    import matplotlib.pyplot as plt
    from ipywidgets import widgets
    IS_COLAB = True
except ImportError:
     IS_COLAB = False
     print("Warning: Not running in a Colab environment. File uploads and Colab-specific display will not work.")
     plt = None
     widgets = None
     class files:
         @staticmethod
         def upload(): print("File upload only available in Google Colab."); return {}
     def display(obj): print(obj) # Fallback display is just print
     def HTML(text): return text # Fallback HTML returns the raw string
     if widgets is None: # Define dummy widgets if ipywidgets is not available
         class widgets:
             @staticmethod
             def Text(**kwargs): return type("DummyText", (), {"value": "", "description": kwargs.get("description","")})()
             @staticmethod
             def Button(**kwargs): return type("DummyButton", (), {"on_click": lambda func: None, "description": kwargs.get("description","")})()
             @staticmethod
             def Output(): return type("DummyOutput", (), {"append_stdout": print})()
             @staticmethod
             def VBox(children): return children # Just return the list of children
             @staticmethod
             def Dropdown(**kwargs): return type("DummyDropdown", (), {"value": kwargs.get('options', [''])[0] if kwargs.get('options') else '', "description": kwargs.get("description","")})()


# --- CLASE RENOMBRADA ---
class ColabFootballLGBMPredictor:
    """
    Predicts football match outcomes using LightGBM models - Iqscore Version.
    """
    # --- Configuration Constants ---
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # --- LightGBM Hyperparameters (Adjust as needed) ---
    N_ESTIMATORS: int = 1000
    LEARNING_RATE: float = 0.05
    NUM_LEAVES: int = 31
    MAX_DEPTH: int = -1
    REG_ALPHA: float = 0.1
    REG_LAMBDA: float = 0.1
    COLSAMPLE_BYTREE: float = 0.8
    SUBSAMPLE: float = 0.8
    SUBSAMPLE_FREQ: int = 1
    EARLY_STOPPING_ROUNDS: int = 50
    N_JOBS: int = -1

    # Recommendation Thresholds (Unchanged)
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
        self.models: Dict[str, Any] = {}
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.features: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self.data: Optional[pd.DataFrame] = None
        self.X_processed_shape: Optional[Tuple[int, int]] = None
        self.best_iterations_: Dict[str, Optional[int]] = {}

        # Decide action based on environment
        if IS_COLAB:
            print("Running in Colab. Attempting data upload...")
            self.upload_data() # Attempt upload if in Colab
            # If upload successful and data is present, proceed
            if self.data is not None and not self.data.empty:
                try:
                    print("\nData loaded via upload, preparing and training...")
                    self.prepare_data()
                    self.train_models()
                except Exception as e:
                     print(f"\n❌ Error during initial data processing/training in Colab: {e}\n{traceback.format_exc()}")
                     self.data = None; self.models = {}; self.preprocessor = None
            else:
                print("Data not loaded via upload in Colab.")
        else:
            # Outside Colab: Do NOT load data automatically. User must call load methods.
            print("Running outside Colab. Load data manually using 'load_data_from_df' or 'load_data_from_file'.")

    # --- Data Loading Methods ---
    def load_data_from_df(self, df: pd.DataFrame):
        """Loads data from a pandas DataFrame."""
        try:
            self.data = df.copy(); print(f"Data loaded from DataFrame: {len(self.data)} rows")
        except Exception as e:
            print(f"Error loading data from DataFrame: {e}"); self.data = None

    def load_data_from_file(self, file_path: str):
        """Loads data from a CSV or Excel file path."""
        try:
            print(f"Attempting to load data from: {file_path}")
            if file_path.lower().endswith('.csv'):
                try:
                    # Try standard comma delimiter first
                    df = pd.read_csv(file_path, delimiter=',', low_memory=False)
                except Exception as e1:
                    print(f"  > Failed with comma delimiter ({e1}), trying semicolon...")
                    # Fallback to semicolon and latin1 encoding
                    df = pd.read_csv(file_path, delimiter=';', encoding='latin1', low_memory=False)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel (.xlsx, .xls).")

            # Standardize and convert odds columns
            odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
                         'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA',
                         'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']
            converted_count = 0
            original_cols = df.columns.tolist() # Keep original case for checking
            df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False).str.replace(' ', '_') # Standardize column names AFTER getting original

            for col_orig in original_cols:
                col_std = col_orig.strip().lower().replace('.', '', regex=False).replace(' ', '_')
                # Check if the original column name (before standardization) matches an odds column
                if col_orig in odds_cols:
                    if col_std in df.columns: # Ensure standardized column exists
                        try:
                            df[col_std] = pd.to_numeric(df[col_std], errors='coerce')
                            converted_count += 1
                        except Exception as conv_e:
                             print(f"  > Warning: Could not convert column '{col_std}' (original: '{col_orig}') to numeric: {conv_e}")
                    else:
                         print(f"  > Warning: Standardized column '{col_std}' (original: '{col_orig}') not found after cleaning names.")


            print(f"  > Converted {converted_count} odds columns to numeric.")
            self.data = df; print(f"Successfully loaded '{file_path}': {len(self.data)} rows, {len(self.data.columns)} columns.")

        except FileNotFoundError:
            print(f"❌ Error: File not found at '{file_path}'. Please check the path.")
            self.data = None
        except ValueError as ve:
             print(f"❌ Error loading file: {ve}")
             self.data = None
        except Exception as e:
            print(f"❌ An unexpected error occurred loading '{file_path}': {e}")
            traceback.print_exc()
            self.data = None

    def upload_data(self):
        """Handles file uploads specifically in Google Colab."""
        if not IS_COLAB:
            print("File upload functionality is only available in Google Colab."); return

        print("Please upload your CSV or Excel data file(s).")
        try:
            uploaded_files = files.upload() # This triggers the Colab upload dialog
            if not uploaded_files:
                print("No files were uploaded."); self.data = None; return

            loaded_dfs = []
            # Define a helper function to load each file safely
            def process_uploaded_file(filename, file_content):
                print(f"Processing uploaded file: {filename}")
                try:
                    if filename.lower().endswith('.csv'):
                        # Try common encodings and delimiters for CSV
                        try:
                            df = pd.read_csv(io.BytesIO(file_content), delimiter=',', encoding='utf-8', on_bad_lines='warn', low_memory=False)
                        except (UnicodeDecodeError, Exception):
                            print(f"  > Failed UTF-8/comma, trying Latin-1/semicolon for {filename}...")
                            df = pd.read_csv(io.BytesIO(file_content), delimiter=';', encoding='latin1', on_bad_lines='warn', low_memory=False)
                    elif filename.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(io.BytesIO(file_content))
                    else:
                        print(f"  > Skipping '{filename}': Unsupported file format.")
                        return None

                    if df is not None:
                        # Standardize and convert odds columns (same logic as load_data_from_file)
                        odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
                                     'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA',
                                     'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']
                        converted_count = 0
                        original_cols = df.columns.tolist()
                        df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False).str.replace(' ', '_')
                        for col_orig in original_cols:
                            col_std = col_orig.strip().lower().replace('.', '', regex=False).replace(' ', '_')
                            if col_orig in odds_cols and col_std in df.columns:
                                try:
                                     df[col_std] = pd.to_numeric(df[col_std], errors='coerce')
                                     converted_count += 1
                                except Exception as conv_e:
                                     print(f"  > Warning (upload): Could not convert '{col_std}' in {filename}: {conv_e}")

                        print(f"  > Successfully loaded '{filename}'. Converted {converted_count} odds columns.")
                        return df
                except Exception as e:
                    print(f"❌ Error loading or processing uploaded file '{filename}': {e}")
                    traceback.print_exc()
                    return None
                return None # Should not be reached, but for safety

            # Use ThreadPoolExecutor for potentially faster loading if multiple files are uploaded
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all file processing tasks
                future_to_file = {executor.submit(process_uploaded_file, fn, fc): fn for fn, fc in uploaded_files.items()}
                # Collect results as they complete
                for future in future_to_file:
                    result_df = future.result()
                    if result_df is not None:
                        loaded_dfs.append(result_df)

            # Concatenate all successfully loaded DataFrames
            if not loaded_dfs:
                print("No valid data could be loaded from the uploaded files."); self.data = None; return

            try:
                self.data = pd.concat(loaded_dfs, ignore_index=True, sort=False)
                print(f"\n✅ Data loaded successfully from uploads! Total matches: {len(self.data)}")
            except Exception as e:
                print(f"❌ Error concatenating data from uploaded files: {e}"); self.data = None

        except Exception as e:
            print(f"An error occurred during the file upload process: {e}")
            traceback.print_exc()
            self.data = None
    # --- End Data Loading ---

    # --- PREPARE_DATA ---
    def prepare_data(self):
        """Preprocesses the loaded data: cleans columns, handles dates, imputes missing values, defines features."""
        if self.data is None or self.data.empty:
            print("Error: Cannot prepare data, DataFrame is missing or empty."); return

        print("\n--- Starting Data Preparation ---")
        data = self.data.copy() # Work on a copy

        # 1. Standardize Column Names
        print("  Standardizing column names...")
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
        print(f"  Column names standardized. Example: {data.columns[:5].tolist()}...")

        # 2. Handle 'date' column
        if 'date' not in data.columns:
            raise ValueError("Critical Error: Required 'date' column not found in the data.")
        print("  Processing 'date' column...")
        try:
            # Attempt parsing with dayfirst=True (common in European formats)
            data['date'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True)
        except Exception as e_dayfirst:
            print(f"    > Parsing with dayfirst=True failed ({e_dayfirst}), trying infer_datetime_format...")
            try:
                # Fallback to inferring the format
                data['date'] = pd.to_datetime(data['date'], errors='coerce', infer_datetime_format=True)
            except Exception as e_infer:
                 raise ValueError(f"Critical Error: Failed to parse 'date' column even with inference: {e_infer}")

        # Impute any NaT values in 'date' after parsing attempts
        n_null_dates = data['date'].isnull().sum()
        if n_null_dates > 0:
            fill_date = data['date'].max() if data['date'].notna().any() else pd.Timestamp.now().normalize()
            data['date'].fillna(fill_date, inplace=True)
            print(f"    > Imputed {n_null_dates} null date values with '{fill_date.strftime('%Y-%m-%d')}'.")
        data.sort_values(by='date', inplace=True) # Sort by date after cleaning

        # 3. Handle Core Target Columns ('fthg', 'ftag', 'ftr')
        print("  Processing core target columns (fthg, ftag, ftr)...")
        core_targets = ['fthg', 'ftag', 'ftr']
        missing_core = [c for c in core_targets if c not in data.columns]
        if missing_core:
            raise ValueError(f"Critical Error: Missing required core target columns: {missing_core}")

        # Convert goals to numeric, coercing errors
        for c in ['fthg', 'ftag']:
            initial_non_numeric = pd.to_numeric(data[c], errors='coerce').isnull().sum()
            data[c] = pd.to_numeric(data[c], errors='coerce')
            if initial_non_numeric > 0:
                 print(f"    > Coerced {initial_non_numeric} non-numeric values in '{c}' to NaN.")

        # Clean and validate 'ftr' (Full Time Result)
        data['ftr'] = data['ftr'].astype(str).str.upper().str.strip()
        valid_ftr = ['H', 'D', 'A']
        initial_rows = len(data)
        # Drop rows where fthg/ftag are NaN OR ftr is invalid
        data.dropna(subset=['fthg', 'ftag'], inplace=True)
        data = data[data['ftr'].isin(valid_ftr)]
        rows_after_cleaning = len(data)
        if rows_after_cleaning < initial_rows:
            print(f"    > Removed {initial_rows - rows_after_cleaning} rows due to missing/invalid core targets (fthg, ftag, ftr).")

        if data.empty:
            raise ValueError("Critical Error: No valid data remaining after cleaning core target columns.")

        # 4. Define Feature Sets (Numeric and Categorical)
        print("  Defining feature sets...")
        # Base numeric features (mostly odds and match stats)
        potential_numeric = [
            'b365h', 'b365d', 'b365a', 'bwh', 'bwd', 'bwa', 'iwh', 'iwd', 'iwa',
            'psh', 'psd', 'psa', 'whh', 'whd', 'wha', 'vch', 'vcd', 'vca',
            'maxh', 'maxd', 'maxa', 'avgh', 'avgd', 'avga',
            'hs', 'as', 'hst', 'ast', 'hf', 'af', 'hc', 'ac', 'hy', 'ay', 'hr', 'ar' # Home/Away Stats
        ]
        # Base categorical features
        potential_categorical = ['hometeam', 'awayteam', 'referee']

        # Filter to keep only features actually present in the data
        self.numeric_features = [f for f in potential_numeric if f in data.columns]
        self.categorical_features = [f for f in potential_categorical if f in data.columns]

        print(f"    > Identified {len(self.numeric_features)} numeric features present in data.")
        print(f"    > Identified {len(self.categorical_features)} categorical features present in data.")

        # Ensure essential categorical features are present
        if 'hometeam' not in self.categorical_features or 'awayteam' not in self.categorical_features:
            raise ValueError("Critical Error: Missing 'hometeam' or 'awayteam' column, cannot proceed.")

        # Handle 'referee' specifically: If exists, ensure it's string and fill NaNs. If not, create a dummy.
        if 'referee' in self.categorical_features:
             data['referee'] = data['referee'].astype(str).fillna('Desconocido').str.strip()
             print("    > Cleaned existing 'referee' column.")
        elif 'referee' in data.columns: # It exists but wasn't initially selected (e.g., wrong type)
            print("    > 'referee' column found but not initially selected. Attempting to add...")
            data['referee'] = data['referee'].astype(str).fillna('Desconocido').str.strip()
            self.categorical_features.append('referee')
            print("    > Added and cleaned 'referee' column.")
        else:
            print("    > 'referee' column not found. Creating a dummy 'Desconocido' referee.")
            data['referee'] = 'Desconocido'
            self.categorical_features.append('referee')

        # 5. Impute Missing Values (using SimpleImputer)
        print("  Imputing missing values...")
        # Impute categorical features with the most frequent value
        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[self.categorical_features] = cat_imputer.fit_transform(data[self.categorical_features])
            print(f"    > Imputed categorical features using strategy: 'most_frequent'.")

        # Impute numeric features with the mean
        if self.numeric_features:
            # First, ensure all numeric columns are indeed numeric, coercing if necessary
            num_imputed_count = 0
            for col in self.numeric_features:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    print(f"      > Warning: Column '{col}' is not numeric. Attempting conversion...")
                    original_nan_count = data[col].isnull().sum()
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    new_nan_count = data[col].isnull().sum()
                    if new_nan_count > original_nan_count:
                         print(f"        > Coercion created {new_nan_count - original_nan_count} new NaNs in '{col}'.")
            # Now apply mean imputation
            num_imputer = SimpleImputer(strategy='mean')
            data[self.numeric_features] = num_imputer.fit_transform(data[self.numeric_features])
            num_imputed_count = data[self.numeric_features].isnull().sum().sum() # Should be 0 after imputation
            print(f"    > Imputed numeric features using strategy: 'mean'. Final NaNs in numeric: {num_imputed_count}")


        # 6. Define Preprocessing Pipelines and ColumnTransformer
        print("  Defining preprocessing pipelines...")
        # Pipeline for numeric features: Impute (mean) -> Scale (StandardScaler)
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        # Pipeline for categorical features: Impute (constant 'Missing') -> Encode (OneHotEncoder)
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Changed for dense output
        ])

        # Combine pipelines using ColumnTransformer
        self.features = self.numeric_features + self.categorical_features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop', # Drop columns not specified in features
            verbose_feature_names_out=False # Keep original feature names where possible
        )

        # 7. Fit the Preprocessor and Check Output Shape
        print("  Fitting the preprocessor...")
        # Create a temporary copy of the features for fitting to avoid modifying 'data' yet
        X_temp_fit = data[self.features].copy()

        # Re-apply imputation just before fitting preprocessor as a safeguard
        if self.categorical_features:
             X_temp_fit[self.categorical_features] = SimpleImputer(strategy='most_frequent').fit_transform(X_temp_fit[self.categorical_features])
        if self.numeric_features:
             X_temp_fit[self.numeric_features] = SimpleImputer(strategy='mean').fit_transform(X_temp_fit[self.numeric_features])

        # Final check for NaNs before fitting
        if X_temp_fit.isnull().any().any():
            print("    > WARNING: NaNs detected in data just before fitting preprocessor. Filling with 0 as fallback.")
            X_temp_fit.fillna(0, inplace=True) # Fallback imputation

        try:
            # Fit the preprocessor
            self.preprocessor.fit(X_temp_fit)
            # Transform the temporary data to check output shape and names
            X_processed_check = self.preprocessor.transform(X_temp_fit)
            self.X_processed_shape = X_processed_check.shape
            print(f"    > Preprocessor fitted successfully. Processed data shape: {self.X_processed_shape}")
            # Try to get feature names after transformation (useful for debugging)
            try:
                processed_feature_names = self.preprocessor.get_feature_names_out()
                print(f"    > Processed feature names ({len(processed_feature_names)}): {processed_feature_names[:3]}...{processed_feature_names[-3:]}")
            except Exception as e_names:
                print(f"    > Could not retrieve processed feature names: {e_names}")
        except Exception as e_fit:
            print(f"❌ Critical Error fitting the preprocessor: {e_fit}")
            traceback.print_exc()
            self.preprocessor = None # Invalidate preprocessor on error
            raise # Re-raise the exception to stop execution

        # 8. Finalize
        self.data = data # Update self.data with the cleaned & imputed version
        print("\n--- Data Preparation Complete ---")
        print(f"  - Final match count: {len(self.data)}")
        if self.X_processed_shape:
            print(f"  - Number of features after preprocessing: {self.X_processed_shape[1]}")
        else:
             print(f"  - Number of features after preprocessing: Error determining shape.")
    # --- End prepare_data ---

    # --- TRAIN_MODELS ---
    def train_models(self):
        """Trains all the required LightGBM models using the preprocessed data."""
        if self.data is None or self.data.empty:
            print("Error: Cannot train models, data is not loaded or is empty."); return
        if self.preprocessor is None:
            print("Error: Cannot train models, preprocessor is not fitted."); return
        if not self.features:
             print("Error: Cannot train models, feature list is empty."); return

        print("\n--- Starting LightGBM Model Training ---")
        try:
            # 1. Prepare Input Features (X)
            print("  Preparing input features (X) using the fitted preprocessor...")
            # Ensure all required features exist in the final data
            missing_features = [f for f in self.features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Critical Error: Features required by preprocessor are missing from data: {missing_features}")

            X_train_raw = self.data[self.features].copy()

            # Safeguard: Check for NaNs *before* transforming (should have been handled in prepare_data)
            if X_train_raw.isnull().any().any():
                 print("    > WARNING: NaNs detected in raw features before transforming for training. Applying emergency fillna(0).")
                 # Attempt to fill based on type, otherwise use 0
                 for col in X_train_raw.columns[X_train_raw.isnull().any()]:
                      if pd.api.types.is_numeric_dtype(X_train_raw[col]):
                           X_train_raw[col].fillna(0, inplace=True) # Or use mean/median if available
                      else:
                           X_train_raw[col].fillna("Missing", inplace=True) # Or use mode if available
                 # X_train_raw.fillna(0, inplace=True) # Simple fallback

            # Transform the features using the *already fitted* preprocessor
            X_processed = self.preprocessor.transform(X_train_raw)
            X_processed_df = pd.DataFrame(X_processed, index=self.data.index) # Maintain index alignment
            print(f"  Input features processed. Shape: {X_processed_df.shape}")


            # 2. Define Model Configurations
            models_def = {
                # Classification Models
                'result': {'type': 'classification', 'target': 'ftr_encoded', 'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3},
                'ht_result': {'type': 'classification', 'target': 'ht_lead', 'objective': 'binary', 'metric': 'logloss'}, # Home team leads at HT (1) or not (0)
                'btts': {'type': 'classification', 'target': 'btts_flag', 'objective': 'binary', 'metric': 'logloss'}, # Both teams score (1) or not (0)

                # Regression Models
                'home_goals': {'type': 'regression', 'target': 'fthg', 'objective': 'regression_l1', 'metric': 'mae'}, # Predict Home Goals
                'away_goals': {'type': 'regression', 'target': 'ftag', 'objective': 'regression_l1', 'metric': 'mae'}, # Predict Away Goals
                'yellow_cards': {'type': 'regression', 'target': 'total_yellows', 'objective': 'regression_l1', 'metric': 'mae'}, # Predict Total Yellow Cards
                'total_corners': {'type': 'regression', 'target': 'total_corners', 'objective': 'regression_l1', 'metric': 'mae'}, # Predict Total Corners
                'total_shots': {'type': 'regression', 'target': 'total_shots', 'objective': 'regression_l1', 'metric': 'mae'},
                'total_shots_on_target': {'type': 'regression', 'target': 'total_shots_on_target', 'objective': 'regression_l1', 'metric': 'mae'},
                'total_fouls': {'type': 'regression', 'target': 'total_fouls', 'objective': 'regression_l1', 'metric': 'mae'},
                'total_red_cards': {'type': 'regression', 'target': 'total_red_cards', 'objective': 'regression_l1', 'metric': 'mae'},
            }

            # 3. Prepare Target Variables (y)
            print("  Preparing target variables (y)...")
            target_creation_errors = []
            initial_rows_targets = len(self.data)

            # Helper function to safely convert to numeric and impute with median/0
            def safe_impute_numeric(df, col_name):
                if col_name in df.columns:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    median_val = df[col_name].median()
                    fill_val = median_val if pd.notna(median_val) else 0
                    nan_count = df[col_name].isnull().sum()
                    if nan_count > 0: print(f"      > Imputing {nan_count} NaNs in '{col_name}' with {fill_val:.2f}")
                    df[col_name].fillna(fill_val, inplace=True)
                    return True
                print(f"      > Column '{col_name}' not found for imputation.")
                return False

            # Result ('ftr_encoded')
            ftr_map = {'H': 0, 'D': 1, 'A': 2}
            if 'ftr' in self.data.columns:
                 # Ensure 'ftr' is clean before mapping (already done in prepare_data, but double-check)
                 self.data['ftr'] = self.data['ftr'].astype(str).str.upper().str.strip()
                 self.data = self.data[self.data['ftr'].isin(ftr_map.keys())] # Ensure only valid values remain
                 self.data['ftr_encoded'] = self.data['ftr'].map(ftr_map)
                 # Drop rows where mapping might have failed (though previous check should prevent this)
                 self.data.dropna(subset=['ftr_encoded'], inplace=True)
                 self.data['ftr_encoded'] = self.data['ftr_encoded'].astype(int) # Ensure integer type
                 print("    > Created 'ftr_encoded' target.")
            else: target_creation_errors.append('ftr_encoded (base ftr missing)')

            # HT Lead ('ht_lead') - Requires 'hthg', 'htag'
            if safe_impute_numeric(self.data, 'hthg') and safe_impute_numeric(self.data, 'htag'):
                self.data['ht_lead'] = (self.data['hthg'] > self.data['htag']).astype(int)
                print("    > Created 'ht_lead' target.")
            else: target_creation_errors.append('ht_lead (base hthg/htag missing or invalid)')

            # Both Teams To Score ('btts_flag') - Requires 'fthg', 'ftag'
            # These should be clean from prepare_data, but run safe_impute again just in case
            if safe_impute_numeric(self.data, 'fthg') and safe_impute_numeric(self.data, 'ftag'):
                self.data['btts_flag'] = ((self.data['fthg'] > 0) & (self.data['ftag'] > 0)).astype(int)
                print("    > Created 'btts_flag' target.")
            else: target_creation_errors.append('btts_flag (base fthg/ftag missing or invalid)') # Should not happen if prepare_data worked

            # Total Yellows ('total_yellows') - Requires 'hy', 'ay'
            if safe_impute_numeric(self.data, 'hy') and safe_impute_numeric(self.data, 'ay'):
                self.data['total_yellows'] = self.data['hy'] + self.data['ay']
                safe_impute_numeric(self.data, 'total_yellows') # Impute the sum if needed
                print("    > Created 'total_yellows' target.")
            else: target_creation_errors.append('total_yellows (base hy/ay missing or invalid)')

            # Total Corners ('total_corners') - Requires 'hc', 'ac'
            if safe_impute_numeric(self.data, 'hc') and safe_impute_numeric(self.data, 'ac'):
                self.data['total_corners'] = self.data['hc'] + self.data['ac']
                safe_impute_numeric(self.data, 'total_corners')
                print("    > Created 'total_corners' target.")
            else: target_creation_errors.append('total_corners (base hc/ac missing or invalid)')

            # Total Shots ('total_shots') - Requires 'hs', 'as'
            if safe_impute_numeric(self.data, 'hs') and safe_impute_numeric(self.data, 'as'):
                self.data['total_shots'] = self.data['hs'] + self.data['as']
                safe_impute_numeric(self.data, 'total_shots')
                print("    > Created 'total_shots' target.")
            else: target_creation_errors.append('total_shots (base hs/as missing or invalid)')

            # Total Shots on Target ('total_shots_on_target') - Requires 'hst', 'ast'
            if safe_impute_numeric(self.data, 'hst') and safe_impute_numeric(self.data, 'ast'):
                self.data['total_shots_on_target'] = self.data['hst'] + self.data['ast']
                safe_impute_numeric(self.data, 'total_shots_on_target')
                print("    > Created 'total_shots_on_target' target.")
            else: target_creation_errors.append('total_shots_on_target (base hst/ast missing or invalid)')

            # Total Fouls ('total_fouls') - Requires 'hf', 'af'
            if safe_impute_numeric(self.data, 'hf') and safe_impute_numeric(self.data, 'af'):
                self.data['total_fouls'] = self.data['hf'] + self.data['af']
                safe_impute_numeric(self.data, 'total_fouls')
                print("    > Created 'total_fouls' target.")
            else: target_creation_errors.append('total_fouls (base hf/af missing or invalid)')

            # Total Red Cards ('total_red_cards') - Requires 'hr', 'ar'
            if safe_impute_numeric(self.data, 'hr') and safe_impute_numeric(self.data, 'ar'):
                self.data['total_red_cards'] = self.data['hr'] + self.data['ar']
                safe_impute_numeric(self.data, 'total_red_cards')
                print("    > Created 'total_red_cards' target.")
            else: target_creation_errors.append('total_red_cards (base hr/ar missing or invalid)')

            # Report any errors during target creation
            if target_creation_errors:
                print(f"  > WARNING: Could not create the following target variables: {target_creation_errors}")

            # Realign X and y after potential row drops during target cleaning/creation
            final_rows_targets = len(self.data)
            if final_rows_targets < initial_rows_targets:
                print(f"  > Data rows reduced from {initial_rows_targets} to {final_rows_targets} during target preparation.")
                common_index = X_processed_df.index.intersection(self.data.index)
                X_processed_df = X_processed_df.loc[common_index]
                self.data = self.data.loc[common_index]
                print(f"  > X and y realigned to {len(common_index)} common rows.")

            if X_processed_df.empty or self.data.empty:
                 raise ValueError("Critical Error: No data left after target preparation and alignment.")

            # 4. Train Each Model
            self.models = {} # Reset models dict
            self.best_iterations_ = {} # Reset best iterations
            available_targets = [config['target'] for name, config in models_def.items() if config['target'] in self.data.columns]
            num_models_to_train = len(available_targets)
            trained_count = 0

            for name, config in models_def.items():
                target_col = config['target']

                # Skip if target column doesn't exist or is all null
                if target_col not in self.data.columns:
                    print(f"\nSkipping '{name}': Target column '{target_col}' not found in data.")
                    continue
                if self.data[target_col].isnull().all():
                    print(f"\nSkipping '{name}': Target column '{target_col}' contains only null values.")
                    continue

                trained_count += 1
                print(f"\n--- {trained_count}/{num_models_to_train}. Training Model: {name.replace('_',' ').title()} ---")

                # Get target series, aligned with processed X
                y_series = self.data.loc[X_processed_df.index, target_col].copy()

                # Final check and imputation for the target variable itself
                if y_series.isnull().any():
                    print(f"  > Warning: NaNs found in target '{target_col}'. Imputing...")
                    is_numeric = pd.api.types.is_numeric_dtype(y_series)
                    if is_numeric:
                        fill_val = y_series.median() if pd.notna(y_series.median()) else 0
                    else: # Assuming categorical/object
                        mode_val = y_series.mode()
                        fill_val = mode_val[0] if not mode_val.empty else 'Missing'
                    y_series.fillna(fill_val, inplace=True)
                    print(f"    > Imputed with: {fill_val}")


                # Prepare y for the model (correct type)
                y_target = y_series.copy()
                if config['type'] == 'classification':
                    try:
                         y_target = y_target.astype(int)
                    except ValueError as e:
                         print(f"  > Error converting target '{target_col}' to int for classification: {e}. Skipping model.")
                         continue
                    # Check for sufficient classes for stratification
                    if y_target.nunique() <= 1:
                         print(f"  > Warning: Target '{target_col}' has only {y_target.nunique()} unique value(s). Stratification disabled.")
                         stratify_param = None
                    elif y_target.value_counts().min() < 2:
                          print(f"  > Warning: Some classes in '{target_col}' have < 2 samples. Stratification disabled.")
                          stratify_param = None
                    else:
                          stratify_param = y_target
                elif config['type'] == 'regression':
                    try:
                        y_target = y_target.astype(float)
                        stratify_param = None
                    except ValueError as e:
                         print(f"  > Error converting target '{target_col}' to float for regression: {e}. Skipping model.")
                         continue
                else:
                    print(f"  > Skipping '{name}': Unknown model type '{config['type']}'.")
                    continue


                # Prepare X and y arrays for splitting
                X_input = X_processed_df.values
                y_input = y_target.values

                # Split data into training and testing sets
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_input, y_input,
                        test_size=self.TEST_SIZE,
                        random_state=self.RANDOM_STATE,
                        stratify=stratify_param # Use stratify_param determined above
                    )
                    print(f"  Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")
                except ValueError as e_split:
                    print(f"  > Stratified split failed for '{target_col}': {e_split}. Retrying without stratification...")
                    try:
                         X_train, X_test, y_train, y_test = train_test_split(
                              X_input, y_input,
                              test_size=self.TEST_SIZE,
                              random_state=self.RANDOM_STATE,
                              stratify=None
                         )
                         print(f"  Data split (unstratified): Train shape {X_train.shape}, Test shape {X_test.shape}")
                    except Exception as e_split_nostrat:
                          print(f"  > ERROR: Failed to split data even without stratification: {e_split_nostrat}. Skipping model '{name}'.")
                          continue


                # Define LightGBM parameters
                lgbm_params = {
                    'objective': config['objective'],
                    'metric': config['metric'],
                    'n_estimators': self.N_ESTIMATORS,
                    'learning_rate': self.LEARNING_RATE,
                    'num_leaves': self.NUM_LEAVES,
                    'max_depth': self.MAX_DEPTH,
                    'reg_alpha': self.REG_ALPHA,
                    'reg_lambda': self.REG_LAMBDA,
                    'colsample_bytree': self.COLSAMPLE_BYTREE,
                    'subsample': self.SUBSAMPLE,
                    'subsample_freq': self.SUBSAMPLE_FREQ,
                    'random_state': self.RANDOM_STATE,
                    'n_jobs': self.N_JOBS,
                    'verbose': -1 # Suppress LightGBM's own verbosity during training
                }
                if config['objective'] == 'multiclass':
                    lgbm_params['num_class'] = config['num_class']


                # Initialize and Train the Model
                try:
                    # Select model type based on config
                    if config['type'] == 'classification':
                        model = lgb.LGBMClassifier(**lgbm_params)
                    elif config['type'] == 'regression':
                        model = lgb.LGBMRegressor(**lgbm_params)
                    else: # Should have been caught earlier, but safety check
                        continue

                    # Define callbacks (only early stopping here)
                    early_stopping_callback = lgb.early_stopping(
                        stopping_rounds=self.EARLY_STOPPING_ROUNDS,
                        verbose=False # Don't print early stopping messages
                    )

                    print(f"  Training LightGBM model with early stopping (rounds={self.EARLY_STOPPING_ROUNDS})...")
                    # Train the model
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        eval_metric=config['metric'], # Use the specific metric for evaluation
                        callbacks=[early_stopping_callback]
                    )

                    self.models[name] = model
                    self.best_iterations_[name] = model.best_iteration_
                    print(f"  Model trained successfully. Best iteration: {model.best_iteration_ or 'N/A (No early stopping)'}")

                    # Evaluate on the test set using the best iteration
                    best_iter = model.best_iteration_
                    num_iter_predict = best_iter if best_iter and best_iter > 0 else None # Use None if no early stopping or best_iter is 0

                    y_pred = model.predict(X_test, num_iteration=num_iter_predict)

                    print("  Evaluating performance on the test set:")
                    if config['type'] == 'classification':
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        print(f"    Accuracy: {acc:.4f}")
                        print(f"    F1 Score (Weighted): {f1:.4f}")

                        # Calculate probabilities and Log Loss / AUC if applicable
                        if hasattr(model, "predict_proba"):
                             try:
                                 y_prob = model.predict_proba(X_test, num_iteration=num_iter_predict)
                                 # Log Loss (works for binary and multiclass)
                                 try:
                                     logloss = log_loss(y_test, y_prob)
                                     print(f"    Log Loss: {logloss:.4f}")
                                 except ValueError as e_ll:
                                     print(f"    Log Loss: N/A (Possibly only one class in y_test: {e_ll})")

                                 # AUC (only for binary classification)
                                 if config['objective'] == 'binary':
                                     # Ensure y_prob has the expected shape for binary
                                     if y_prob.shape[1] == 2:
                                         try:
                                             auc = roc_auc_score(y_test, y_prob[:, 1]) # Use prob of positive class
                                             print(f"    AUC: {auc:.4f}")
                                         except ValueError as e_auc:
                                              print(f"    AUC: N/A (Possibly only one class in y_test: {e_auc})")
                                         except Exception as e_auc_other:
                                              print(f"    AUC: N/A (Calculation error: {e_auc_other})")
                                     else:
                                         print(f"    AUC: N/A (Expected 2 columns for binary probabilities, got shape {y_prob.shape})")

                             except Exception as e_proba:
                                 print(f"    Could not calculate probabilities or derived metrics: {e_proba}")

                    elif config['type'] == 'regression':
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        print(f"    Mean Absolute Error (MAE): {mae:.4f}")
                        print(f"    Mean Squared Error (MSE): {mse:.4f}")
                        print(f"    Root Mean Squared Error (RMSE): {rmse:.4f}")

                except Exception as e_train_eval:
                    print(f"❌ ERROR during training or evaluation for model '{name}': {e_train_eval}")
                    traceback.print_exc()
                    # Remove partially trained model if error occurred
                    if name in self.models: del self.models[name]
                    if name in self.best_iterations_: del self.best_iterations_[name]

            # 5. Final Check
            if not self.models:
                print("\n--- Training Error! No models were successfully trained. ---")
            else:
                print(f"\n--- LightGBM Training Complete: {len(self.models)} models trained successfully ---")

        except Exception as e_outer:
            print(f"\n❌ An unexpected error occurred during the overall model training process: {e_outer}")
            traceback.print_exc()
            # Clear any potentially inconsistent state
            self.models = {}
            self.best_iterations_ = {}
    # --- End train_models ---

    # --- Helper functions ---
    def calculate_over_probability(self, threshold: float, predicted_total_goals: float) -> float:
        """
        Calculates a heuristic probability for 'Over' threshold goals based on the predicted total.
        Uses a sigmoid function centered around the threshold.

        Args:
            threshold: The Over/Under line (e.g., 2.5).
            predicted_total_goals: The sum of predicted home and away goals.

        Returns:
            The calculated probability of the total goals being OVER the threshold (clipped between 0.01 and 0.99).
            Returns 0.5 if the input prediction is invalid.
        """
        K = 0.8 # Sigmoid steepness factor (can be tuned based on observed accuracy)
        try:
            # Ensure the predicted value is a valid float
            pred_value = float(predicted_total_goals)
            # Sigmoid function: 1 / (1 + exp(-K * (x - threshold)))
            probability_over = 1 / (1 + np.exp(-K * (pred_value - threshold)))
            # Clip the probability to avoid exact 0 or 1, which can be problematic
            return np.clip(probability_over, 0.01, 0.99)
        except (ValueError, TypeError):
            # Handle cases where prediction is not a valid number (e.g., None, string)
            print(f"  > Warning: Invalid input '{predicted_total_goals}' for calculate_over_probability. Returning default 0.5.")
            return 0.5

    def _get_recommendation_strength(self, prediction_type: str, value: float) -> Tuple[str, str]:
        """
        Determines the strength level and icon for a recommendation based on its value (probability or statistic).

        Args:
            prediction_type: The key corresponding to the prediction type in CONFIDENCE_THRESHOLDS (e.g., 'result', 'over_under').
            value: The numeric value of the prediction (e.g., probability, predicted corners).

        Returns:
            A tuple containing the strength description (e.g., 'Fuerte', 'Moderada') and the associated icon (e.g., '🔥').
            Returns ('Info', 'ℹ️') if the type is unknown or value is invalid.
        """
        thresholds = self.CONFIDENCE_THRESHOLDS.get(prediction_type, [])
        if not thresholds:
            print(f"  > Warning: No confidence thresholds defined for prediction type '{prediction_type}'.")
            return 'Info', 'ℹ️' # Default if type is unknown

        try:
            # Ensure value is a float for comparison
            numeric_value = float(value)
        except (ValueError, TypeError):
             print(f"  > Warning: Invalid value '{value}' for determining recommendation strength for '{prediction_type}'.")
             return 'Info', 'ℹ️' # Default if value is not numeric

        # Iterate through thresholds (which are sorted descending by value)
        for threshold_value, strength_label, icon in thresholds:
            if numeric_value >= threshold_value:
                return strength_label, icon

        # If value is lower than all defined thresholds, return the lowest defined level
        # (Assumes the last entry in the list is the 'lowest' confidence level, e.g., 'Baja')
        # If thresholds list was somehow empty (already checked, but for safety), return default.
        return thresholds[-1][1], thresholds[-1][2] if thresholds else ('Info', 'ℹ️')


    # --- HTML Generation Helpers (for predict_match output) ---

    def _generate_html_table(self, title: str, headers: List[str], rows: List[List[str]], table_class: str = "results-table") -> str:
        """Generates a standard HTML table string."""
        if not rows:
            return f'<h3 class="card-subtitle">{title}</h3><p>No hay datos disponibles.</p>'

        html = f'<h3 class="card-subtitle">{title}</h3>'
        html += '<div class="table-responsive">'
        html += f'<table class="{table_class}"><thead><tr>'
        for header in headers:
            html += f'<th>{header}</th>'
        html += '</tr></thead><tbody>'
        for row_data in rows:
            html += '<tr>'
            for i, cell in enumerate(row_data):
                # Apply text-center class to numeric-like columns (heuristic)
                cell_str = str(cell).strip()
                is_numeric_like = cell_str.replace('.', '', 1).isdigit() or (cell_str.startswith('-') and cell_str[1:].replace('.', '', 1).isdigit())
                text_center_class = ' class="text-center"' if is_numeric_like and "Equipo" not in headers[i] and "Local" not in headers[i] and "Visitante" not in headers[i] else ''
                html += f'<td{text_center_class}>{cell}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'
        return html

    def get_match_history(self, home_team: str, away_team: str, num: int = 10) -> str:
        """Generates an HTML table for the direct match history between two teams."""
        title = f"📜 Historial Directo ({home_team} vs {away_team} - Últimos {num})"
        if self.data is None or self.data.empty :
            return f'<h3 class="card-subtitle">{title}</h3><p>No hay datos históricos cargados.</p>'
        if 'hometeam' not in self.data.columns or 'awayteam' not in self.data.columns:
            return f'<h3 class="card-subtitle">{title}</h3><p>Columnas de equipo ("hometeam", "awayteam") no encontradas.</p>'

        try:
            history_df = self.data[
                ((self.data['hometeam'].str.lower() == home_team.lower()) & (self.data['awayteam'].str.lower() == away_team.lower())) |
                ((self.data['hometeam'].str.lower() == away_team.lower()) & (self.data['awayteam'].str.lower() == home_team.lower()))
            ].sort_values(by='date', ascending=False).head(num)
        except Exception as e:
            print(f"  > Error filtering match history: {e}")
            return f'<h3 class="card-subtitle">{title}</h3><p>Error al buscar el historial de partidos.</p>'

        if history_df.empty:
            return f'<h3 class="card-subtitle">{title}</h3><p>No se encontraron partidos directos recientes entre estos equipos.</p>'

        headers = ["Fecha", "Local", "Visitante", "G.L.", "G.V.", "Res."]
        rows = []
        for _, row in history_df.iterrows():
            dt_str = row.get('date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'N/A'
            ht = str(row.get('hometeam', 'N/A'))
            at = str(row.get('awayteam', 'N/A'))
            hg = int(row.get('fthg', -1)) if pd.notna(row.get('fthg')) else '-'
            ag = int(row.get('ftag', -1)) if pd.notna(row.get('ftag')) else '-'
            res = str(row.get('ftr', '-'))
            rows.append([dt_str, ht, at, hg, ag, res])

        return self._generate_html_table(title, headers, rows)

    def get_recent_matches(self, team: str, num: int = 10) -> str:
        """Generates an HTML table for the recent matches of a specific team."""
        title = f"📅 Últimos {num} Partidos ({team})"
        if self.data is None or self.data.empty :
             return f'<h3 class="card-subtitle">{title}</h3><p>No hay datos históricos cargados.</p>'
        if 'hometeam' not in self.data.columns or 'awayteam' not in self.data.columns:
             return f'<h3 class="card-subtitle">{title}</h3><p>Columnas de equipo ("hometeam", "awayteam") no encontradas.</p>'

        try:
            team_lower = team.lower()
            recent_matches_df = self.data[
                (self.data['hometeam'].str.lower() == team_lower) | (self.data['awayteam'].str.lower() == team_lower)
            ].sort_values(by='date', ascending=False).head(num)
        except Exception as e:
            print(f"  > Error filtering recent matches for {team}: {e}")
            return f'<h3 class="card-subtitle">{title}</h3><p>Error al buscar los partidos recientes.</p>'

        if recent_matches_df.empty:
            return f'<h3 class="card-subtitle">{title}</h3><p>No se encontraron partidos recientes para {team}.</p>'

        headers = ["Fecha", "Local", "Visitante", "G.L.", "G.V.", "Res."]
        rows = []
        for _, row in recent_matches_df.iterrows():
            dt_str = row.get('date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'N/A'
            ht = str(row.get('hometeam', 'N/A'))
            at = str(row.get('awayteam', 'N/A'))
            hg = int(row.get('fthg', -1)) if pd.notna(row.get('fthg')) else '-'
            ag = int(row.get('ftag', -1)) if pd.notna(row.get('ftag')) else '-'
            res = str(row.get('ftr', '-'))
            # Highlight the team name
            ht_styled = f'<strong style="color:#2575fc;">{ht}</strong>' if ht.lower() == team_lower else ht
            at_styled = f'<strong style="color:#c33764;">{at}</strong>' if at.lower() == team_lower else at
            rows.append([dt_str, ht_styled, at_styled, hg, ag, res])

        return self._generate_html_table(title, headers, rows)


    def get_top_teams_by_referee(self, referee_name: str, num: int = 5) -> str:
        """Generates an HTML table for teams with the most wins under a specific referee."""
        title = f"🏆 Top {num} Equipos con Más Victorias (Árbitro: {referee_name})"
        if self.data is None or self.data.empty:
             return f'<h3 class="card-subtitle">{title}</h3><p>No hay datos históricos cargados.</p>'
        if 'referee' not in self.data.columns:
            return f'<h3 class="card-subtitle">{title}</h3><p>Columna "referee" no encontrada.</p>'
        if not all(c in self.data for c in ['hometeam', 'awayteam', 'ftr']):
             return f'<h3 class="card-subtitle">{title}</h3><p>Faltan columnas ("hometeam", "awayteam", "ftr") para calcular estadísticas de árbitro.</p>'

        try:
            ref_lower = referee_name.lower()
            ref_matches = self.data[self.data['referee'].str.lower() == ref_lower].copy()
        except Exception as e:
            print(f"  > Error filtering matches for referee {referee_name}: {e}")
            return f'<h3 class="card-subtitle">{title}</h3><p>Error al buscar los partidos para este árbitro.</p>'

        if ref_matches.empty:
            return f'<h3 class="card-subtitle">{title}</h3><p>No se encontraron partidos para el árbitro {referee_name}.</p>'

        try:
            # Calculate wins for home teams under this ref
            home_wins = ref_matches[ref_matches['ftr'] == 'H'].groupby('hometeam').size()
            # Calculate wins for away teams under this ref
            away_wins = ref_matches[ref_matches['ftr'] == 'A'].groupby('awayteam').size()
            # Combine the wins, filling missing teams with 0 wins
            total_wins = home_wins.add(away_wins, fill_value=0).astype(int).sort_values(ascending=False)
        except Exception as e:
            print(f"  > Error calculating wins for referee {referee_name}: {e}")
            return f'<h3 class="card-subtitle">{title}</h3><p>Error al calcular las victorias para este árbitro.</p>'

        if total_wins.empty:
            return f'<h3 class="card-subtitle">{title}</h3><p>No se registraron victorias de equipos con el árbitro {referee_name}.</p>'

        top_teams = total_wins.head(num)
        headers = ["Equipo", "Victorias"]
        rows = []
        for team, wins_count in top_teams.items():
            rows.append([str(team), int(wins_count)])

        return self._generate_html_table(title, headers, rows, table_class="results-table simple-table")


    # --- PREDICT_MATCH ---
    def predict_match(self, home_team: str, away_team: str, referee: Optional[str] = None) -> str:
        """
        Generates detailed HTML predictions for a given match using the trained LightGBM models.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            referee: Name of the referee (optional). If None or not found, a default/missing value is used.

        Returns:
            An HTML string containing the formatted predictions and analysis.
            Returns an error HTML string if prediction is not possible.
        """
        # --- Initial Checks ---
        error_prefix = "<div class='error-box'><strong>Error en Predicción:</strong> "
        error_suffix = "</div>"
        if not self.models: return f"{error_prefix}No hay modelos entrenados disponibles.{error_suffix}"
        if not self.preprocessor: return f"{error_prefix}El preprocesador de datos no está listo.{error_suffix}"
        if self.data is None or self.data.empty: return f"{error_prefix}No hay datos cargados para obtener información base.{error_suffix}"
        if not self.features: return f"{error_prefix}La lista de features está vacía.{error_suffix}"
        if not self.X_processed_shape: return f"{error_prefix}La forma de los datos procesados es desconocida.{error_suffix}"

        print(f"\n--- Iniciando Predicción LGBM para: {home_team} vs {away_team} (Árbitro: {referee or 'N/A'}) ---")

        try:
            # --- 1. Validate Inputs and Prepare Match Data ---
            print("  Validando equipos y árbitro...")
            # Use lower case for comparison to avoid case sensitivity issues
            home_team_lower = home_team.lower()
            away_team_lower = away_team.lower()
            referee_lower = referee.lower() if referee else None

            available_teams = pd.concat([self.data['hometeam'].str.lower(), self.data['awayteam'].str.lower()]).unique()
            available_referees = self.data['referee'].str.lower().unique()

            if home_team_lower not in available_teams:
                 # Try to find similar team names as suggestions
                 # from difflib import get_close_matches
                 # close_matches = get_close_matches(home_team_lower, available_teams, n=3, cutoff=0.7)
                 # suggestion = f" Equipos similares encontrados: {', '.join(close_matches)}." if close_matches else ""
                 # return f"{error_prefix}Equipo local '{home_team}' no encontrado en los datos históricos.{suggestion}{error_suffix}"
                 # Simplified warning:
                 print(f"  > WARNING: Equipo local '{home_team}' no encontrado en los datos históricos. La predicción usará promedios generales.")
                 # We can still proceed by using average stats, but the categorical feature will be unknown

            if away_team_lower not in available_teams:
                 print(f"  > WARNING: Equipo visitante '{away_team}' no encontrado en los datos históricos. La predicción usará promedios generales.")
                 # We can still proceed

            if home_team_lower == away_team_lower:
                return f"{error_prefix}El equipo local y visitante no pueden ser el mismo ('{home_team}').{error_suffix}"


            # Determine the referee value to use
            referee_to_use = 'Desconocido' # Default if no referee provided or found
            if referee_lower:
                if referee_lower in available_referees:
                    # Find the original casing of the referee name to use in the input dict
                    # This is important because the OneHotEncoder was fitted with original casing (or imputed value)
                    original_ref_casing = self.data.loc[self.data['referee'].str.lower() == referee_lower, 'referee'].iloc[0]
                    referee_to_use = original_ref_casing
                    print(f"  Árbitro '{referee}' encontrado y validado como '{referee_to_use}'.")
                else:
                    print(f"  > WARNING: Árbitro '{referee}' no encontrado en los datos históricos. Usando '{referee_to_use}' como valor.")
            else:
                 print(f"  No se proporcionó árbitro. Usando '{referee_to_use}'.")


            # Create the input dictionary for the match
            print("  Creando DataFrame de entrada para la predicción...")
            match_input_dict: Dict[str, Any] = {}

            # Add categorical features
            match_input_dict['hometeam'] = home_team # Use original casing provided by user
            match_input_dict['awayteam'] = away_team # Use original casing
            match_input_dict['referee'] = referee_to_use

            # Add numeric features: Use the average value from the training data as a default
            # This provides a reasonable baseline if specific stats for this matchup aren't directly available.
            # The model learned patterns based on these averages during training for missing values.
            avg_numeric_stats = self.data[self.numeric_features].mean().fillna(0) # Calculate means and fill any NaNs with 0
            for feature in self.numeric_features:
                 match_input_dict[feature] = avg_numeric_stats.get(feature, 0) # Get average, default to 0 if somehow missing

            # Ensure all features expected by the preprocessor are present
            for f in self.features:
                if f not in match_input_dict:
                     # This case should ideally not happen if numeric/categorical features are handled above
                     print(f"  > WARNING: Feature '{f}' missing from input dict. Adding default value (0 or 'Missing').")
                     default_val = 0 if f in self.numeric_features else 'Missing'
                     match_input_dict[f] = default_val


            # Create a DataFrame with the correct feature order
            try:
                match_df = pd.DataFrame([match_input_dict], columns=self.features) # Ensure columns are in the same order as during fitting
            except Exception as e_df:
                 return f"{error_prefix}Error al crear el DataFrame de entrada: {e_df}{error_suffix}"

            # --- 2. Preprocess the Match Data ---
            print("  Preprocesando datos del partido con el transformador ajustado...")
            # Safeguard: Check for NaNs *before* transforming (should use imputed averages/modes)
            if match_df.isnull().any().any():
                print("    > WARNING: NaNs detected in input DataFrame before transforming prediction! Applying emergency fill.")
                for col in match_df.columns[match_df.isnull().any()]:
                    if pd.api.types.is_numeric_dtype(match_df[col]): match_df[col].fillna(0, inplace=True)
                    else: match_df[col].fillna("Missing", inplace=True)

            try:
                match_processed = self.preprocessor.transform(match_df)
                # Check if the output shape matches the expected number of features
                if match_processed.shape[1] != self.X_processed_shape[1]:
                     return f"{error_prefix}Discrepancia en el número de features después del preprocesamiento. Esperado: {self.X_processed_shape[1]}, Obtenido: {match_processed.shape[1]}.{error_suffix}"
                print(f"  Preprocesamiento completado. Shape: {match_processed.shape}")
            except Exception as e_proc:
                 return f"{error_prefix}Error al preprocesar los datos del partido: {e_proc}<br/>{traceback.format_exc()}{error_suffix}"


            # --- 3. Make Predictions with Trained Models ---
            print("  Realizando predicciones con los modelos LGBM...")
            predictions = {}
            failed_predictions = []

            for name, model in self.models.items():
                model_type = 'Classifier' if isinstance(model, lgb.LGBMClassifier) else 'Regressor' if isinstance(model, lgb.LGBMRegressor) else 'Unknown'
                # print(f"    Predicting for '{name}' ({model_type})...") # Verbose logging if needed
                try:
                    # Get the best iteration for prediction (if available from early stopping)
                    best_iter = self.best_iterations_.get(name)
                    num_iter_predict = best_iter if best_iter and best_iter > 0 else None

                    if model_type == 'Classifier':
                        # Predict probabilities
                        predicted_proba = model.predict_proba(match_processed, num_iteration=num_iter_predict)[0] # Get the probabilities for the first (only) sample

                        if name == 'result': # Multiclass: [Prob_H, Prob_D, Prob_A]
                             # Ensure probabilities sum roughly to 1
                             if not np.isclose(np.sum(predicted_proba), 1.0):
                                  # print(f"      > Warning: Raw probabilities for 'result' don't sum to 1 ({np.sum(predicted_proba):.3f}). Normalizing.")
                                  predicted_proba = predicted_proba / np.sum(predicted_proba) if np.sum(predicted_proba) > 1e-9 else np.array([0.333, 0.334, 0.333]) # Normalize or default
                             # Check expected number of classes
                             if len(predicted_proba) != 3:
                                 print(f"      > ERROR: Expected 3 probabilities for 'result', got {len(predicted_proba)}. Using default.")
                                 predictions[name] = np.array([0.333, 0.334, 0.333]) # Default probabilities
                             else:
                                predictions[name] = predicted_proba

                        elif name == 'ht_result' or name == 'btts': # Binary: [Prob_0, Prob_1]
                             if len(predicted_proba) == 2:
                                 predictions[name] = predicted_proba[1] # Store probability of the positive class (1)
                             elif len(predicted_proba) == 1:
                                # Handle case where model only predicts one class (e.g., if training data was all one class)
                                 predicted_class = model.classes_[0]
                                 predictions[name] = 1.0 if predicted_class == 1 else 0.0
                                 print(f"      > Warning: Binary model '{name}' predicted only one class ({predicted_class}). Probability set to {predictions[name]:.1f}.")
                             else:
                                 print(f"      > ERROR: Expected 1 or 2 probabilities for binary '{name}', got {len(predicted_proba)}. Using default 0.5.")
                                 predictions[name] = 0.5 # Default probability
                        else: # Other unexpected classifier?
                             print(f"      > Warning: Unhandled classifier '{name}'. Storing raw probabilities: {predicted_proba}")
                             predictions[name] = predicted_proba # Store raw output

                    elif model_type == 'Regressor':
                        predicted_value = model.predict(match_processed, num_iteration=num_iter_predict)[0]
                        # Ensure regression predictions are non-negative (e.g., goals, cards)
                        predictions[name] = max(0.0, float(predicted_value))
                    else:
                        print(f"      > ERROR: Model '{name}' has an unknown type. Cannot predict.")
                        predictions[name] = None
                        failed_predictions.append(name)

                except Exception as e_pred:
                    print(f"    ❌ Error prediciendo con el modelo '{name}': {e_pred}")
                    # traceback.print_exc() # Optional: print full traceback for debugging
                    predictions[name] = None # Store None if prediction failed
                    failed_predictions.append(name)

            if failed_predictions:
                 print(f"  > WARNING: Fallaron las predicciones para los siguientes modelos: {', '.join(failed_predictions)}")

            # Check if essential predictions failed
            essential_preds = ['result', 'home_goals', 'away_goals']
            missing_essentials = [m for m in essential_preds if predictions.get(m) is None]
            if missing_essentials:
                 return f"{error_prefix}Predicciones esenciales fallidas: {', '.join(missing_essentials)}. No se puede continuar.{error_suffix}"


            # --- 4. Process Predictions and Calculate Derived Values ---
            print("  Procesando predicciones y calculando valores derivados...")

            # Full Time Result Probabilities (already handled potential normalization)
            result_probs = predictions['result'] # Should be a numpy array [P(H), P(D), P(A)]

            # Predicted Goals
            pred_home_goals = float(predictions['home_goals'])
            pred_away_goals = float(predictions['away_goals'])
            pred_total_goals = pred_home_goals + pred_away_goals

            # Over/Under Probability (using helper function)
            prob_over_25 = self.calculate_over_probability(self.OVER_UNDER_THRESHOLD, pred_total_goals)
            prob_under_25 = 1.0 - prob_over_25

            # Both Teams To Score Probability
            # Use the direct prediction if available, otherwise default to 0.5
            prob_btts_yes = float(predictions.get('btts', 0.5)) # Get prediction or default
            prob_btts_no = 1.0 - prob_btts_yes

            # Home Team Leads at HT Probability
            # Use the direct prediction if available, otherwise default to 0.5
            prob_ht_home_lead = float(predictions.get('ht_result', 0.5)) # Get prediction or default

            # Other Stats (ensure float type and handle potential None)
            pred_yellow_cards = float(predictions.get('yellow_cards', 0.0))
            pred_corners = float(predictions.get('total_corners', 0.0))
            pred_shots = float(predictions.get('total_shots', 0.0))
            pred_shots_target = float(predictions.get('total_shots_on_target', 0.0))
            pred_fouls = float(predictions.get('total_fouls', 0.0))
            pred_red_cards = float(predictions.get('total_red_cards', 0.0))


            # --- Optional: Debugging Output ---
            # print("-" * 20 + " DEBUG INFO - Predictions " + "-" * 20)
            # print(f"  Result Probs (H/D/A): {result_probs[0]:.3f} / {result_probs[1]:.3f} / {result_probs[2]:.3f}")
            # print(f"  Predicted Goals (H/A/T): {pred_home_goals:.2f} / {pred_away_goals:.2f} / {pred_total_goals:.2f}")
            # print(f"  Prob Over {self.OVER_UNDER_THRESHOLD}: {prob_over_25:.3f}")
            # print(f"  Prob Under {self.OVER_UNDER_THRESHOLD}: {prob_under_25:.3f}")
            # print(f"  Prob BTTS (Yes/No): {prob_btts_yes:.3f} / {prob_btts_no:.3f}")
            # print(f"  Prob HT Home Lead: {prob_ht_home_lead:.3f}")
            # print(f"  Predicted Yellows: {pred_yellow_cards:.2f}")
            # print(f"  Predicted Corners: {pred_corners:.2f}")
            # print(f"  Predicted Shots: {pred_shots:.2f}")
            # print(f"  Predicted SOT: {pred_shots_target:.2f}")
            # print(f"  Predicted Fouls: {pred_fouls:.2f}")
            # print(f"  Predicted Reds: {pred_red_cards:.2f}")
            # print("-" * 60)
            # --- End Debugging Output ---


            # --- 5. Generate Recommendations ---
            print("  Generando recomendaciones basadas en umbrales...")
            recommendations = []
            result_map = {0: f"Gana {home_team} (L)", 1: "Empate", 2: f"Gana {away_team} (V)"}

            # a) Final Result (1X2)
            most_likely_outcome_index = np.argmax(result_probs)
            confidence_result = result_probs[most_likely_outcome_index]
            strength_result, icon_result = self._get_recommendation_strength('result', confidence_result)
            recommendations.append({
                'type': 'Resultado Final (1X2)',
                'prediction': result_map[most_likely_outcome_index],
                'confidence_value': confidence_result,
                'confidence_display': f"{confidence_result:.1%}",
                'strength': strength_result,
                'icon': icon_result,
                'is_stat': False
            })

            # b) Over/Under Total Goals
            if prob_over_25 > prob_under_25:
                prediction_ou = f"Más de {self.OVER_UNDER_THRESHOLD} goles"
                confidence_ou = prob_over_25
                strength_ou, icon_ou = self._get_recommendation_strength('over_under', confidence_ou)
            else:
                prediction_ou = f"Menos de {self.OVER_UNDER_THRESHOLD} goles"
                confidence_ou = prob_under_25
                strength_ou, icon_ou = self._get_recommendation_strength('over_under', confidence_ou)
            recommendations.append({
                'type': f"Goles Totales (Más/Menos {self.OVER_UNDER_THRESHOLD})",
                'prediction': prediction_ou,
                'confidence_value': confidence_ou,
                'confidence_display': f"{confidence_ou:.1%}",
                'strength': strength_ou,
                'icon': icon_ou,
                'is_stat': False
            })

            # c) Both Teams To Score (BTTS)
            if prob_btts_yes > prob_btts_no:
                prediction_btts = "Sí (Ambos equipos marcan)"
                confidence_btts = prob_btts_yes
                strength_btts, icon_btts = self._get_recommendation_strength('btts', confidence_btts)
            else:
                prediction_btts = "No (Al menos uno NO marca)"
                confidence_btts = prob_btts_no
                strength_btts, icon_btts = self._get_recommendation_strength('btts', confidence_btts)
            recommendations.append({
                'type': 'Ambos Equipos Marcan (BTTS)',
                'prediction': prediction_btts,
                'confidence_value': confidence_btts,
                'confidence_display': f"{confidence_btts:.1%}",
                'strength': strength_btts,
                'icon': icon_btts,
                'is_stat': False
            })

            # d) Half Time Result (Home Lead vs Not Lead) - Only recommend if prediction is confident enough (deviates from 0.5)
            ht_confidence_threshold = 0.10 # Minimum deviation from 0.5 to make a recommendation
            if abs(prob_ht_home_lead - 0.5) > ht_confidence_threshold:
                if prob_ht_home_lead > 0.5:
                    prediction_ht = f"{home_team} lidera al descanso"
                    confidence_ht = prob_ht_home_lead
                else:
                    prediction_ht = f"{home_team} NO lidera al descanso"
                    confidence_ht = 1.0 - prob_ht_home_lead # Confidence in the "NO lead" outcome
                strength_ht, icon_ht = self._get_recommendation_strength('ht_result', confidence_ht)
                recommendations.append({
                    'type': 'Resultado al Descanso (Local)',
                    'prediction': prediction_ht,
                    'confidence_value': confidence_ht,
                    'confidence_display': f"{confidence_ht:.1%}",
                    'strength': strength_ht,
                    'icon': icon_ht,
                    'is_stat': False
                })

            # e) Statistical Predictions (Cards, Corners, etc.)
            stat_predictions = [
                {'key': 'yellow_cards', 'value': pred_yellow_cards, 'label': 'Tarjetas Amarillas', 'icon': '🟨'},
                {'key': 'corners', 'value': pred_corners, 'label': 'Córners Totales', 'icon': '🚩'},
                {'key': 'total_shots', 'value': pred_shots, 'label': 'Tiros Totales', 'icon': '🎯'},
                {'key': 'total_shots_on_target', 'value': pred_shots_target, 'label': 'Tiros a Puerta Totales', 'icon': '⚽'},
                {'key': 'total_fouls', 'value': pred_fouls, 'label': 'Faltas Totales', 'icon': '✋'},
                {'key': 'total_red_cards', 'value': pred_red_cards, 'label': 'Tarjetas Rojas Totales', 'icon': '🟥'},
            ]
            for stat in stat_predictions:
                 # Check if the prediction exists (wasn't None)
                 if predictions.get(stat['key']) is not None:
                     strength_stat, icon_stat_strength = self._get_recommendation_strength(stat['key'], stat['value'])
                     recommendations.append({
                         'type': stat['label'],
                         'prediction': f"~ {stat['value']:.1f}", # Display estimated value
                         'confidence_value': -1, # Not probability-based, use -1 to sort last
                         'confidence_display': f"Estimado: {stat['value']:.1f}",
                         'strength': strength_stat, # Strength based on the *value* (e.g., Alto, Medio)
                         'icon': stat['icon'],
                         'is_stat': True
                     })

            # Sort recommendations: Primary: non-stats first. Secondary: by confidence value (desc).
            recommendations.sort(key=lambda x: (x['is_stat'], -x['confidence_value']))

            # print(f"  DEBUG: Final list of recommendations (before HTML): {recommendations}")


            # --- 6. Generate HTML Output ---
            print("  Generando salida HTML...")

            # CSS Styles (Minified) - Consider moving to a separate CSS file if preferred
            html_css = """
<style>
/* Minified CSS */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
body { font-family: 'Poppins', sans-serif; background-color: #f0f2f5; } /* Added body style */
.predictor-container{font-family:'Poppins',sans-serif;max-width:950px;margin:25px auto;background-color:#fff;padding:0;border-radius:15px;box-shadow:0 10px 30px rgba(0,0,0,.1);overflow:hidden;border:1px solid #e0e0e0}
.predictor-header{background:linear-gradient(135deg,#1a2a6c,#b21f1f,#fdbb2d);color:#fff;padding:25px 30px;text-align:center;border-bottom:5px solid #fdbb2d}
.predictor-header h2{margin:0 0 8px;font-size:2.1em;font-weight:700;text-shadow:1px 1px 3px rgba(0,0,0,.3)}
.predictor-header p{margin:0;color:#f0f0f0;font-style:normal;font-size:1.05em}
.predictor-content{padding:20px 30px;background-color:#f9f9f9} /* Adjusted padding */
.card{background-color:#fff;border:1px solid #e9ecef;border-radius:12px;margin-bottom:25px;box-shadow:0 4px 15px rgba(0,0,0,.06);overflow:hidden} /* Adjusted margin */
.card-header{background:linear-gradient(to right,#6a11cb,#2575fc);color:#fff;padding:15px 20px;border-bottom:1px solid #dee2e6;font-size:1.3em;font-weight:600;border-radius:12px 12px 0 0;display:flex;align-items:center}
.card-header i.fas{margin-right:10px;font-size:1.1em;vertical-align:middle} /* FontAwesome example */
.card-content{padding:20px} /* Adjusted padding */
.card-subtitle{font-size:1.1em;font-weight:600;color:#343a40;margin-top:0;margin-bottom:15px;border-bottom:2px solid #6a11cb;padding-bottom:8px} /* Adjusted subtitle */
.prediction-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;text-align:center;margin:15px 0;padding:10px 0;border-top:1px solid #eee;border-bottom:1px solid #eee} /* Adjusted grid */
.stat-box{padding:12px;background-color:#f8f9fa;border-radius:8px} /* Adjusted padding */
.stat-box .team-name{font-weight:600;margin-bottom:8px;font-size:1em;color:#495057;word-wrap:break-word} /* Adjusted font size */
.stat-box .probability{font-size:2.5em;font-weight:700;margin-bottom:5px;display:flex;align-items:center;justify-content:center} /* Adjusted font size */
.stat-box .probability i.fas{font-size:.7em;margin-left:8px}
.prob-home{color:#2575fc}
.prob-draw{color:#ff7e5f}
.prob-away{color:#c33764}
.results-table{width:100%;border-collapse:separate;border-spacing:0 5px;font-size:.95em;margin-bottom:15px} /* Adjusted font size */
.results-table td,.results-table th{border-bottom:1px solid #e9ecef;padding:10px 12px;text-align:left;vertical-align: middle;} /* Adjusted padding */
.results-table th{background-color:#f8f9fa;font-weight:600;color:#495057;border-top:1px solid #e9ecef;white-space: nowrap;}
.results-table tbody tr{background-color:#fff;border-radius:5px}
.results-table tbody tr:hover{background-color:#f1f3f5;} /* Added hover effect */
.results-table .text-center{text-align:center}
.strength-cell span{display:inline-block;padding:4px 10px;border-radius:15px;font-weight:600;font-size:.85em;white-space:nowrap; border: 1px solid transparent;} /* Adjusted padding/font */
.strength-Fuerte{background-color:#d1e7dd;color:#0f5132;border-color:#a3cfbb}
.strength-Moderada{background-color:#fff3cd;color:#664d03;border-color:#ffe69c}
.strength-Baja{background-color:#f8d7da;color:#842029;border-color:#f5c2c7}
.strength-Alto{background-color:#cfe2ff;color:#0a367a;border-color:#b6d4fe} /* Added borders */
.strength-Medio{background-color:#e2e3e5;color:#41464b;border-color:#d3d6d8}
.strength-Bajo{background-color:#f8f9fa;color:#6c757d;border-color:#dee2e6}
.strength-Info{background-color:#e2e3e5;color:#41464b; border-color:#d3d6d8;} /* Style for Info */
.table-responsive{overflow-x:auto; margin-bottom: 10px;} /* Ensure space below table */
.error-box{background-color:#f8d7da;color:#842029;border:1px solid #f5c2c7;padding:18px;border-radius:8px;margin:20px auto;text-align:center;font-weight:600; max-width: 800px;} /* Centered error box */
.predicted-stats-list{list-style:none;padding-left:0;margin-top:15px}
.predicted-stats-list li{background-color:#f8f9fa;margin-bottom:8px;padding:10px 15px;border-radius:6px;display:flex;justify-content:space-between;align-items:center; font-size: 0.95em;}
.predicted-stats-list li span:first-child{font-weight:500; color: #343a40;}
.predicted-stats-list li span:last-child{font-weight:600;} /* Ensure estimate value is bold */
hr.section-divider{margin:25px 0;border:none;border-top:1px solid #eee} /* Adjusted margin */
.footer-note{font-size:.85em;text-align:center;color:#888;margin-top:15px;}
</style>
"""

            # Start HTML Document
            html_output = f"<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>Predicción: {home_team} vs {away_team}</title>{html_css}</head><body>"
            html_output += "<div class='predictor-container'>"

            # Header
            html_output += f"""
<div class='predictor-header'>
    <h2>{home_team} vs {away_team}</h2>
    <p>Análisis Predictivo LightGBM | Iqscore</p>
</div>
<div class='predictor-content'>
"""

            # Card 1: Probabilities and Estimated Score
            html_output += """
<div class='card'>
    <div class='card-header'>📊 Probabilidades del Resultado y Marcador Estimado</div>
    <div class='card-content'>
        <div class='prediction-grid'>
            <div class='stat-box'>
                <div class='team-name'>Victoria Local<br/>({home_team})</div>
                <div class='probability prob-home'>{result_probs[0]:.1%}</div>
            </div>
            <div class='stat-box'>
                <div class='team-name'>Empate</div>
                <div class='probability prob-draw'>{result_probs[1]:.1%}</div>
            </div>
            <div class='stat-box'>
                <div class='team-name'>Victoria Visitante<br/>({away_team})</div>
                <div class='probability prob-away'>{result_probs[2]:.1%}</div>
            </div>
        </div>
        <div style='text-align:center; padding-top:15px; font-weight:600; font-size: 1.1em;'>
            ⚽ Marcador Estimado: {pred_home_goals:.1f} - {pred_away_goals:.1f} 🥅
        </div>
    </div>
</div>
""".format(home_team=home_team, away_team=away_team, result_probs=result_probs, pred_home_goals=pred_home_goals, pred_away_goals=pred_away_goals)

            # Card 2: Recommended Picks (Non-Stats)
            html_output += """
<div class='card'>
    <div class='card-header'>💡 Picks Recomendados (Basados en Probabilidad)</div>
    <div class='card-content'>
        <div class='table-responsive'>
            <table class='results-table'>
                <thead><tr><th>Mercado</th><th>Predicción</th><th>Probabilidad</th><th class='text-center'>Confianza</th></tr></thead>
                <tbody>
"""
            # Populate recommended picks table
            picks_rendered = 0
            for rec in recommendations:
                 if not rec['is_stat']: # Only non-stat recommendations here
                      html_output += f"""
<tr>
    <td>{rec['type']}</td>
    <td>{rec['prediction']}</td>
    <td class='text-center'>{rec['confidence_display']}</td>
    <td class='text-center strength-cell'><span class='strength-{rec['strength']}'>{rec['icon']} {rec['strength']}</span></td>
</tr>"""
                      picks_rendered += 1

            if picks_rendered == 0:
                 html_output += "<tr><td colspan='4' style='text-align:center; font-style:italic;'>No hay picks específicos con suficiente confianza probabilística para recomendar.</td></tr>"

            html_output += """
                </tbody>
            </table>
        </div>
        <p class='footer-note'>*Las probabilidades y la confianza se basan en el modelo. Apuesta con responsabilidad.</p>
    </div>
</div>
"""

            # Card 3: Other Stats Estimates and Historical Data
            html_output += """
<div class='card'>
    <div class='card-header'>📈 Estadísticas Estimadas y Datos Históricos</div>
    <div class='card-content'>
"""
            # Section: Other Statistical Estimates
            html_output += "<h3 class='card-subtitle'>📊 Otras Estimaciones Estadísticas</h3>"
            html_output += "<ul class='predicted-stats-list'>"
            stats_rendered = 0
            for rec in recommendations:
                if rec['is_stat']:
                    # For stats, the 'strength' describes the predicted value level (Alto, Medio, Bajo)
                    html_output += f"<li><span>{rec['type']}</span> <span class='strength-cell'><span class='strength-{rec['strength']}'>{rec['prediction']} ({rec['strength']}) {rec['icon']}</span></span></li>"
                    stats_rendered += 1

            if stats_rendered == 0:
                html_output += "<li>No hay estimaciones estadísticas disponibles.</li>"
            html_output += "</ul><hr class='section-divider'>"

            # Section: Historical Head-to-Head
            html_output += self.get_match_history(home_team, away_team)
            html_output += "<hr class='section-divider'>"

            # Section: Recent Matches Home Team
            html_output += self.get_recent_matches(home_team)
            html_output += "<hr class='section-divider'>"

            # Section: Recent Matches Away Team
            html_output += self.get_recent_matches(away_team)

            # Section: Referee Stats (Optional)
            if referee_to_use != 'Desconocido': # Only show if a valid ref was used
                html_output += "<hr class='section-divider'>"
                html_output += self.get_top_teams_by_referee(referee_to_use) # Use the validated referee name

            html_output += """
    </div>
</div>
"""

            # Close main content, container, and body/html
            html_output += "</div></div></body></html>" # Close predictor-content, predictor-container

            print("✅ HTML Prediction Generated Successfully.")
            return html_output

        except Exception as e:
            print(f"❌ Error fatal durante predict_match: {e}")
            traceback.print_exc()
            # Return a user-friendly error message within the HTML structure
            return f"<!DOCTYPE html><html><head><title>Error</title>{html_css}</head><body><div class='error-box'><strong>Error Crítico en la Predicción:</strong><br/>Ocurrió un problema inesperado al generar el análisis.<br/>Detalle: {e}</div></body></html>"

    # --- run_colab_predictor_interface (Only functional in Colab) ---
    def run_colab_predictor_interface(self):
        """Runs the interactive widget interface (requires Colab environment)."""
        if not IS_COLAB:
            print("Interfaz interactiva solo disponible en Google Colab."); return
        if self.data is None or self.data.empty:
            print("Error: No hay datos cargados para iniciar la interfaz."); return
        if not self.models:
            print("Error: Los modelos no están entrenados. No se puede iniciar la interfaz."); return

        # Ensure required Colab/IPython modules are loaded (should be if IS_COLAB is True)
        try:
            from IPython.display import display, HTML, clear_output
            import ipywidgets as widgets
        except ImportError:
             print("Error crítico: Faltan módulos de IPython/ipywidgets necesarios para la interfaz en Colab.")
             return

        # Display Header for the Interface
        display(HTML("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <div style="font-family: 'Poppins', sans-serif; max-width: 850px; margin: 25px auto; padding: 30px; background: linear-gradient(140deg, #1d4350, #a43931); color: white; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.2);">
            <h1 style="text-align: center; margin: 0 0 15px 0; font-size: 2.4em; font-weight: 700;"><i class="fas fa-bolt"></i> Sistema de Predicción (LightGBM) <span style="color: #fdbb2d; font-weight: bold;">Iqscore</span></h1>
            <p style="text-align: center; font-size: 1.15em; margin-bottom: 20px; color: #e0e0e0;">Introduce los detalles del partido para recibir análisis y predicciones generadas por LightGBM.</p>
            <ol style="margin-left: 25px; line-height: 1.7; font-size: 1.05em;">
                <li>Verifica que los datos (<strong style="color: #fdbb2d;">CSV/Excel</strong>) se hayan cargado y los modelos entrenado correctamente.</li>
                <li>Selecciona los equipos <strong style="color: #fdbb2d;">local</strong> y <strong style="color: #fdbb2d;">visitante</strong> de las listas.</li>
                <li>Elige el <strong style="color: #fdbb2d;">árbitro</strong> (opcional, si no se elige se usará un valor por defecto).</li>
                <li>Presiona el botón <strong style="color: #fdbb2d;">'Generar Análisis'</strong>.</li>
            </ol>
        </div>"""))

        # Prepare Dropdown Options
        try:
            # Get unique teams and sort them alphabetically
            all_teams = pd.concat([self.data['hometeam'], self.data['awayteam']]).dropna().unique()
            sorted_teams = sorted([str(t) for t in all_teams])

            # Get unique referees, add an empty option for 'None', and sort
            all_referees = self.data['referee'].dropna().unique()
            # Ensure 'Desconocido' (or imputed value) is handled if present
            sorted_referees = [''] + sorted([str(r) for r in all_referees if str(r).strip() and str(r).lower() != 'desconocido'])
            if 'Desconocido' in all_referees: sorted_referees.append('Desconocido') # Ensure it's an option if used

        except Exception as e_dropdown:
            print(f"❌ Error preparando las opciones de los menús desplegables: {e_dropdown}")
            # Provide dummy options to prevent crashing the interface
            sorted_teams = ["Error al cargar equipos"]
            sorted_referees = ["", "Error al cargar árbitros"]

        # Define Widgets
        widget_style = {'description_width': 'initial'} # Adjust width of description labels
        widget_layout = {'width': 'auto', 'min_width': '350px'} # Ensure widgets take reasonable space

        home_team_input = widgets.Dropdown(options=sorted_teams, description='🏠 Equipo Local:', style=widget_style, layout=widget_layout)
        away_team_input = widgets.Dropdown(options=sorted_teams, description='✈️ Equipo Visitante:', style=widget_style, layout=widget_layout)
        referee_input = widgets.Dropdown(options=sorted_referees, value='', description='👤 Árbitro (Opcional):', style=widget_style, layout=widget_layout)

        predict_button = widgets.Button(
            description='⚡ Generar Análisis ⚡',
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Haz clic para obtener las predicciones del partido seleccionado',
            icon='futbol', # Example FontAwesome icon name
            layout={'height': '45px', 'width': 'auto', 'min_width': '220px', 'margin': '15px 0 10px 0'} # Button layout
        )

        output_area = widgets.Output() # Area to display results or errors

        # Define Button Click Handler
        def on_predict_button_clicked(button_instance):
             with output_area: # Capture output within this area
                 clear_output(wait=True) # Clear previous results

                 home = home_team_input.value
                 away = away_team_input.value
                 ref = referee_input.value or None # Use None if empty string selected

                 # --- Input Validation ---
                 if not home or home == "Error al cargar equipos":
                      display(HTML("<div class='error-box'>⚠️ Error: Debes seleccionar un equipo local válido.</div>"))
                      return
                 if not away or away == "Error al cargar equipos":
                      display(HTML("<div class='error-box'>⚠️ Error: Debes seleccionar un equipo visitante válido.</div>"))
                      return
                 if home == away:
                      display(HTML("<div class='error-box'>⚠️ Error: El equipo local y visitante no pueden ser el mismo.</div>"))
                      return
                 # Optional: Validate referee choice against available options if needed
                 if ref and ref not in sorted_referees and ref != "Error al cargar árbitros":
                       display(HTML(f"<div class='error-box'>⚠️ Advertencia: El árbitro '{ref}' no parece estar en la lista original. Se intentará usar de todas formas.</div>"))
                       # We let predict_match handle unknown referees gracefully

                 print(f"⏳ Analizando partido con LightGBM: {home} vs {away} (Árbitro: {ref or 'N/A'}). Por favor, espera...")
                 time.sleep(0.5) # Brief pause for user feedback

                 # --- Call Prediction Logic ---
                 try:
                      # Call the main prediction method
                      prediction_html_result = self.predict_match(str(home), str(away), str(ref) if ref else None)
                      # Display the generated HTML
                      display(HTML(prediction_html_result))
                 except Exception as e:
                      # Display any unexpected error during prediction
                      print(f"❌ Error grave durante la generación del análisis:")
                      traceback.print_exc() # Print traceback in the output area
                      display(HTML(f"<div class='error-box'><strong>Error Inesperado:</strong> {str(e)}</div>"))

        # Link button click event to the handler function
        predict_button.on_click(on_predict_button_clicked)

        # Arrange Widgets in Layout
        input_box = widgets.VBox(
             [home_team_input, away_team_input, referee_input],
             layout={'border': '1px solid #ccc', 'padding': '15px', 'border_radius': '8px', 'margin_bottom': '15px'}
        )

        # Display the complete interface
        display(widgets.VBox([input_box, predict_button, output_area]))


# --- Execution Block (Handles both Colab and Local Execution) ---
if __name__ == "__main__":

    # --- COLAB EXECUTION PATH ---
    if IS_COLAB:
        print("--- Ambiente Colab Detectado ---")
        try:
            print("--- Iniciando Sistema Predictor (Colab) ---")
            # In Colab, __init__ attempts upload and potentially training
            predictor_instance = ColabFootballLGBMPredictor()

            # Check if initialization was successful (data loaded AND models trained)
            if predictor_instance.data is not None and not predictor_instance.data.empty and predictor_instance.models and predictor_instance.preprocessor:
                 print("\n--- Inicialización OK. Iniciando Interfaz Interactiva ---")
                 # Load FontAwesome CSS for icons in the interface
                 display(HTML("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'>"))
                 # Run the widget-based interface
                 predictor_instance.run_colab_predictor_interface()
            else:
                 # Initialization failed, provide feedback
                 print("\n❌ Fallo en la Inicialización Automática (Colab). No se puede iniciar la interfaz.")
                 if predictor_instance.data is None or predictor_instance.data.empty:
                      print("   - Motivo: Datos no cargados o vacíos.")
                 if not predictor_instance.models:
                      print("   - Motivo: Modelos no entrenados.")
                 if not predictor_instance.preprocessor:
                      print("   - Motivo: Preprocesador no listo.")
                 print("   -> Por favor, revisa los pasos de carga y entrenamiento o ejecuta manualmente.")

        except ImportError:
            # This should be caught earlier, but as a fallback
            print("\n❌ Error de Importación Crítico (Colab). Asegúrate de que las librerías (lightgbm, ipywidgets, etc.) estén disponibles.")
        except Exception as e_colab:
            print(f"\n❌ Error Fatal durante la ejecución en Colab: {e_colab}")
            traceback.print_exc()

    # --- LOCAL PYTHON EXECUTION PATH ---
    else:
        print("--- Ambiente Python Local Detectado ---")
        print("La interfaz interactiva de Colab no está disponible.")
        print("Se requiere la carga manual de datos y ejecución de predicciones.")
        print("Asegúrate de tener instaladas las librerías: pip install pandas numpy scikit-learn lightgbm")
        print("*"*60)

        try:
            print("\n--- Creando instancia del predictor (Local) ---")
            # When created locally, __init__ does NOT load data or train automatically.
            predictor_instance = ColabFootballLGBMPredictor()

            # --- MANUAL STEPS REQUIRED LOCALLY ---

            # 1. Specify the path to your data file
            #    vvv ----- ¡¡¡ MODIFICA ESTA LÍNEA !!! ----- vvv
            data_file_path = "ruta/completa/a/tu/archivo_datos.csv"  # <-- PON AQUÍ LA RUTA A TU CSV O EXCEL
            #    ^^^ ----- ¡¡¡ MODIFICA ESTA LÍNEA !!! ----- ^^^

            # 2. Load data from the specified file
            print(f"\n--- Cargando datos desde: {data_file_path} ---")
            try:
                predictor_instance.load_data_from_file(data_file_path)
            except Exception as load_err:
                # load_data_from_file prints specific errors, but we catch general ones too
                print(f"❌ Error general al intentar cargar el archivo: {load_err}")
                print("   Verifica que la ruta sea correcta y el archivo sea un CSV o Excel válido.")
                exit() # Stop execution if data loading fails

            # 3. Check if data loaded successfully
            if predictor_instance.data is None or predictor_instance.data.empty:
                print("❌ Error Crítico: No se pudieron cargar los datos. El DataFrame está vacío o es None.")
                print("   Deteniendo la ejecución.")
                exit() # Stop execution
            else:
                print(f"✅ Datos cargados exitosamente: {len(predictor_instance.data)} filas encontradas.")

                # 4. Prepare data and Train models (must be called manually)
                try:
                    print("\n--- Iniciando preparación de datos (Manual) ---")
                    predictor_instance.prepare_data() # Call data preparation
                    print("\n--- Iniciando entrenamiento de modelos (Manual) ---")
                    predictor_instance.train_models()   # Call model training
                except Exception as train_err:
                    print(f"❌ Error durante la preparación de datos o el entrenamiento: {train_err}")
                    print("   Detalles del error:")
                    traceback.print_exc()
                    print("   Deteniendo la ejecución.")
                    exit() # Stop if preparation/training fails

                # 5. Verify that models and preprocessor are ready
                if predictor_instance.models and predictor_instance.preprocessor:
                    print("\n✅ ¡Preparación y entrenamiento completados exitosamente!")

                    # 6. Specify the match details for prediction
                    #    vvv ----- ¡¡¡ MODIFICA ESTAS LÍNEAS !!! ----- vvv
                    home_team_name = "Equipo Local A"  # <-- PON AQUÍ EL NOMBRE EXACTO DEL EQUIPO LOCAL
                    away_team_name = "Equipo Visitante B" # <-- PON AQUÍ EL NOMBRE EXACTO DEL EQUIPO VISITANTE
                    referee_name = None                # <-- PON AQUÍ EL NOMBRE DEL ÁRBITRO (o déjalo None si no aplica/no se conoce)
                    #    ^^^ ----- ¡¡¡ MODIFICA ESTAS LÍNEAS !!! ----- ^^^

                    print(f"\n--- Realizando predicción para: {home_team_name} vs {away_team_name} (Árbitro: {referee_name or 'N/A'}) ---")

                    # 7. Call the prediction method
                    try:
                        prediction_html = predictor_instance.predict_match(home_team_name, away_team_name, referee_name)

                        # 8. Handle the output (print or save)
                        print("\n--- Resultado de la Predicción (HTML crudo) ---")
                        # Displaying raw HTML in the console might be messy.
                        # Consider saving to a file for better viewing.
                        # print(prediction_html) # Uncomment to print raw HTML to console

                        # --- Opción: Guardar el HTML en un archivo ---
                        try:
                             output_filename = f"prediccion_{home_team_name.replace(' ','_')}_vs_{away_team_name.replace(' ','_')}.html"
                             with open(output_filename, "w", encoding="utf-8") as f:
                                 f.write(prediction_html)
                             print(f"\n✅ Resultado de la predicción guardado en el archivo: '{output_filename}'")
                             print("   Puedes abrir este archivo en tu navegador web para ver el análisis formateado.")
                        except Exception as save_err:
                             print(f"\n⚠️ Error al guardar el archivo HTML: {save_err}")
                             print("   Imprimiendo HTML crudo en la consola como alternativa:")
                             print(prediction_html)
                        # --- Fin Opción Guardar ---

                    except Exception as pred_err:
                         print(f"❌ Error durante la ejecución de la predicción: {pred_err}")
                         traceback.print_exc()

                else:
                     # Training or preparation likely failed earlier, but double-check state
                     print("\n❌ Error Crítico: Los modelos o el preprocesador no se inicializaron correctamente después del entrenamiento.")
                     print("   Revisa los mensajes de error anteriores en las fases de preparación y entrenamiento.")
                     print(f"   - Estado Modelos: {'OK' if predictor_instance.models else 'FALLIDO'}")
                     print(f"   - Estado Preprocesador: {'OK' if predictor_instance.preprocessor else 'FALLIDO'}")
                     print("   Deteniendo la ejecución.")
                     exit()

        except ImportError:
             # Should have been caught by the check at the top of the file
             print("\n❌ Error de Importación Crítico. Asegúrate de que 'lightgbm' y otras dependencias estén instaladas (`pip install ...`).")
        except Exception as e_local:
            print(f"\n❌ Error Fatal durante la ejecución local: {e_local}")
            traceback.print_exc()