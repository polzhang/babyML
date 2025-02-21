import io
import os
import pandas as pd
import numpy as np
import time
import logging
import redis
import pickle
import traceback
from datetime import timedelta
from flask import Flask, request, jsonify, Response, stream_with_context, session
from flask_session import Session
from flask_cors import CORS, cross_origin
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import colorama
import queue
import json
from colorama import Fore, Style
from flaml.automl.logger import logger as flaml_logger
from urllib.parse import urlparse

#IMPORTANT df = pd.DataFrame(session['uploaded_file_data'])


log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# Clear any existing handlers
flaml_logger.handlers.clear()

# Add queue handler
queue_handler = QueueHandler(log_queue)
flaml_logger.addHandler(queue_handler)
flaml_logger.setLevel(logging.INFO)

# Add console handler for immediate console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
flaml_logger.addHandler(console_handler)

colorama.init()
app = Flask(__name__)
CORS(app, origins=["https://baby-ml.vercel.app"])
     
REDIS_URL = os.getenv('REDIS_URL')

global_state = {
    'uploaded_file_data': None,
    'test_data': None,
    'automl_instance': None,
    'config': None
}

def get_redis_client():
    try:
        url = urlparse(REDIS_URL)
        return redis.Redis(
            host='enjoyed-treefrog-57366.upstash.io',
            port=url.port or 6379,
            username=url.username or 'default',
            password=url.password,  # This will come from your environment variable
            ssl=True,  # Upstash requires SSL
            decode_responses=False,
            socket_timeout=5
        )
    except Exception as e:
        print(f"Redis connection error: {e}")
        return None

redis_client = get_redis_client()

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis_client  # Your existing redis_client
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
Session(app)

@app.route('/stream-logs')
def stream_logs_endpoint():
    def generate():
        # Get log queue for current session
        session_logs = session.get('log_queue', [])
        while True:
            try:
                if session_logs:
                    log_entry = session_logs.pop(0)
                    session.modified = True
                    yield f"data: {log_entry}\n\n"
                else:
                    yield f"data: heartbeat\n\n"
                time.sleep(0.1)
            except Exception:
                yield f"data: heartbeat\n\n"
                time.sleep(0.1)
    
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response



@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read file into pandas DataFrame
        file_content = file.read()
        file_stream = io.BytesIO(file_content)
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_stream)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_stream)
        else:
            return jsonify({"error": "Unsupported file type. Please upload CSV or Excel file"}), 400

        # Store DataFrame in session
        session['uploaded_file_data'] = df.to_dict()  # Convert DataFrame to dict for session storage
        session['columns'] = list(df.columns)
        
        return jsonify({
            "message": "File successfully uploaded",
            "columns": list(df.columns)
        }), 200

    except pd.errors.EmptyDataError:
        return jsonify({"error": "The uploaded file is empty"}), 400
    except pd.errors.ParserError as e:
        return jsonify({"error": "Error parsing file. Please check the file format"}), 400
    except Exception as e:
        return jsonify({"error": "Server error processing upload"}), 500

@app.route('/get-columns', methods=['GET'])
def get_columns():
    try:
        if 'columns' not in session:
            return jsonify({"error": "No data uploaded"}), 404
        
        columns = session['columns']
        return jsonify({"columns": columns}), 200
        
    except Exception as e:
        return jsonify({"error": "Server error retrieving columns"}), 500

def detect_and_encode_categorical(df, max_unique_ratio=0.05):
    categorical_columns = []
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        unique_ratio = len(df[column].unique()) / len(df)
        
        is_categorical = (
            pd.api.types.is_string_dtype(df[column]) or 
            pd.api.types.is_object_dtype(df[column]) or 
            pd.api.types.is_categorical_dtype(df[column]) or
            unique_ratio <= max_unique_ratio
        )
        
        if is_categorical:
            categorical_columns.append(column)
    
    df_encoded = df.copy()
    
    if categorical_columns:
        print(f"\n{Fore.YELLOW}=== Detected Categorical Columns ==={Style.RESET_ALL}")
        print(categorical_columns)
        
        for column in categorical_columns:
            n_unique = len(df[column].unique())
            
            if n_unique == 2 or n_unique > 10:
                print(f"Applying Label Encoding to: {column}")
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df[column].astype(str))
            else:
                print(f"Applying One-Hot Encoding to: {column}")
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = ohe.fit_transform(df[[column]])
                new_columns = [f"{column}_{cat}" for cat in ohe.categories_[0]]
                
                for i, new_col in enumerate(new_columns):
                    df_encoded[new_col] = encoded_features[:, i]
                
                df_encoded.drop(column, axis=1, inplace=True)
    
    return df_encoded


def detect_and_encode_categorical(df):
    """Helper function to encode categorical columns"""
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes
    return df

@app.route('/setup-training', methods=['POST'])
def setup_training():
    if redis_client is None:
        logging.error("Redis client is not initialized")
        return jsonify({"error": "Database connection error"}), 503

    print("Received request data:", request.json)
    print("Request content type:", request.content_type)
    print(f"\n{Fore.GREEN}=== Received Training Setup Request ==={Style.RESET_ALL}")

    # Clear log queue
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break

    # Clear previous AutoML instance from Redis if it exists
    if redis_client.exists('automl_instance'):
        redis_client.delete('automl_instance')

    try:
        # Get data from Redis
        pickled_data = redis_client.get('uploaded_file_data')
        if pickled_data is None:
            print("Error: No data uploaded setup")
            return jsonify({"error": "No data uploaded setup"}), 400

        # Load the DataFrame
        df = pickle.loads(pickled_data)

        # Store configuration in Redis
        config = request.json
        redis_client.set('training_config', pickle.dumps(config))
        print(f"\n{Fore.YELLOW}Received Configuration:{Style.RESET_ALL}")
        print(config)

        # Extract required values from config with defaults
        target_column = config.get('target_variable')
        problem_type = config.get('problem_type', 'classification')
        validation_settings = config.get('validation', {})
        validation_method = validation_settings.get('method', 'holdout')
        split_ratio = validation_settings.get('split_ratio', 0.8)

        if not target_column:
            print("target var not specified")
            return jsonify({"error": "Target variable not specified"}), 400

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Classification preprocessing
        label_mapping = None
        if problem_type == 'classification':
            if pd.api.types.is_numeric_dtype(y):
                print(f"\n{Fore.YELLOW}Converting numeric target to categorical for classification{Style.RESET_ALL}")
                y = y.astype(str)
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
            
            # Store label encoder in Redis for later use
            redis_client.set('label_encoder', pickle.dumps(le))
            
            print(f"\n{Fore.CYAN}Label Mapping (Numeric to Original Class):{Style.RESET_ALL}")
            for num, label in label_mapping.items():
                print(f"{num} -> {label}")

        elif problem_type == 'regression':
            if not pd.api.types.is_numeric_dtype(y):
                print("targetvar must be numeric for regression")
                return jsonify({"error": "Target variable must be numeric for regression tasks"}), 400
            y = y.astype(float)

        # Handle missing data
        preprocessing_config = config.get('preprocessing', {})
        missing_data_config = preprocessing_config.get('missing_data', {})
        missing_strategy = missing_data_config.get('strategy')

        if missing_strategy == 'imputation':
            imputer_method = missing_data_config.get('imputation_method', 'mean')
            if imputer_method == 'constant':
                constant_value = missing_data_config.get('constant_value', 0)
                imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            else:
                imputer = SimpleImputer(strategy=imputer_method)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        elif missing_strategy == 'drop_rows':
            X = X.dropna()
            y = y[X.index]

        # Feature preprocessing
        X = detect_and_encode_categorical(X)
        
        # Scaling
        scaler = StandardScaler()
        scaler2 = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        X = pd.DataFrame(scaler2.fit_transform(X), columns=X.columns)

        # Store preprocessors in Redis
        redis_client.set('scalers', pickle.dumps((scaler, scaler2)))

        # Feature reduction
        if preprocessing_config.get('feature_reduction') == 'pca':
            pca = PCA(n_components=0.95)
            X = pd.DataFrame(pca.fit_transform(X))
            redis_client.set('pca', pickle.dumps(pca))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=split_ratio,
            random_state=42,
            stratify=y if problem_type == 'classification' else None
        )

        print(f"\n{Fore.CYAN}=== Training Data Shape ==={Style.RESET_ALL}")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")

        # FLAML setup
        automl = AutoML()
        settings = {
            'time_budget': config.get('time_budget'),
            'metric': validation_settings.get('metric', 'accuracy' if problem_type == 'classification' else 'r2'),
            'task': problem_type,
            'n_jobs': -1,
            'estimator_list': config.get('models', {}).get('selected', ['lgbm', 'rf', 'xgboost', 'extra_tree']),
            'eval_method': validation_method
        }

        if validation_method == 'holdout':
            settings['split_ratio'] = split_ratio
        elif validation_method == 'cv':
            settings['n_splits'] = int(validation_settings.get('k_folds', 5))

        print(f"\n{Fore.YELLOW}=== Starting FLAML Training ==={Style.RESET_ALL}")
        print("Settings:", settings)

        automl.fit(
            X_train=X_train,
            y_train=y_train,
            **settings
        )

        # Store trained model and test data in Redis
        redis_client.set('automl_instance', pickle.dumps(automl))
        redis_client.set('test_data', pickle.dumps((X_test, y_test)))

        best_model = automl.model.estimator

        # Calculate metrics
        if problem_type == 'classification':
            y_pred = automl.predict(X_test)
            y_pred_proba = automl.predict_proba(X_test)
            
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            try:
                if len(np.unique(y)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None
            
            metrics = {
                "classification_report": class_report,
                "confusion_matrix": conf_matrix,
                "roc_auc_score": roc_auc,
                "training_time": automl.time_to_find_best_model,
                "models_trained": len(automl.estimator_list),
                "best_iteration": automl.best_iteration
            }
            
            print(f"\n{Fore.CYAN}=== Detailed Metrics ==={Style.RESET_ALL}")
            metrics_log = [
                f"ROC AUC Score: {roc_auc:.4f}" if roc_auc else "ROC AUC: N/A",
                f"Accuracy: {class_report['accuracy']:.4f}",
                f"Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}",
                f"Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}"
            ]
            
            for log_msg in metrics_log:
                print(log_msg)
                log_queue.put(log_msg)
            
        else:  # regression
            y_pred = automl.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            try:
                mape = mean_absolute_percentage_error(y_test, y_pred)
            except:
                mape = None
                
            metrics = {
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "mape": mape,
                "training_time": automl.time_to_find_best_model,
                "models_trained": len(automl.estimator_list),
                "best_iteration": automl.best_iteration
            }
            
            metrics_log = [
                f"RMSE: {rmse:.4f}",
                f"MAE: {mae:.4f}",
                f"R²: {r2:.4f}"
            ]
            
            if mape is not None:
                metrics_log.append(f"MAPE: {mape:.4f}")
            
            for log_msg in metrics_log:
                print(log_msg)
                log_queue.put(log_msg)

        # Get feature importance
        try:
            feature_importance = automl.feature_importance()
            if isinstance(feature_importance, pd.Series):
                feature_importance = feature_importance.to_dict()
        except:
            feature_importance = None

        # Final logs
        print(f"\n{Fore.GREEN}=== Training Complete ==={Style.RESET_ALL}")
        print(f"Best ML learner: {best_model.__class__.__name__}")
        print(f"Best hyperparameter config: {automl.best_config}")
        print(f"Training time: {automl.time_to_find_best_model:.2f} seconds")
        print(f"Models trained: {len(automl.estimator_list)}")

        response = {
            "status": "success",
            "best_estimator": best_model.__class__.__name__,
            "best_config": automl.best_config,
            "metrics": metrics,
            "feature_importance": feature_importance
        }

        return jsonify(response), 200

    except Exception as e:
        error_msg = str(e)
        print(f"{Fore.RED}Error in training setup: {error_msg}{Style.RESET_ALL}")
        log_queue.put(f"{Fore.RED}Error in training setup: {error_msg}{Style.RESET_ALL}")
        return jsonify({
            "error": error_msg,
            "status": "error",
            "traceback": traceback.format_exc()
        }), 500


@app.route('/start-training', methods=['POST'])
def start_training():
    """Simulated endpoint for starting the training process."""
    # Here you can invoke the terminal command or script that runs your training.
    return jsonify({"message": "Training setup started!"}), 200




@app.route('/upload-and-predict', methods=['POST'])
def upload_and_predict():
    """Handle both test data upload and prediction."""
    if redis_client is None:
        logging.error("Redis client is not initialized")
        return jsonify({"error": "Database connection error"}), 503

    print(f"\n{Fore.CYAN}=== Starting Prediction Process ==={Style.RESET_ALL}")

    try:
        # Get configuration from Redis
        config_data = redis_client.get('training_config')
        if config_data is None:
            print(f"{Fore.RED}Error: No configuration found{Style.RESET_ALL}")
            return jsonify({"error": "No configuration found. Please set up training first."}), 400
        
        config = pickle.loads(config_data)
        print(f"Current config state: {config}")

        # Verify target variable exists in config
        target_column = config.get('target_variable')
        if not target_column:
            print(f"{Fore.RED}Error: 'target_variable' not found in config{Style.RESET_ALL}")
            return jsonify({
                "error": "'target_variable' not found in config",
                "current_config": config
            }), 400

        print(f"\n{Fore.GREEN}=== Received Request for Upload and Prediction ==={Style.RESET_ALL}")

        if 'file' not in request.files:
            print(f"{Fore.RED}Error: No file part in request{Style.RESET_ALL}")
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            print(f"{Fore.RED}Error: No selected file{Style.RESET_ALL}")
            return jsonify({"error": "No selected file"}), 400

        # Read the file
        file_content = file.read()
        file.seek(0)
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            file_stream = io.StringIO(file_content.decode("utf-8"))
            test_data = pd.read_csv(file_stream)
        elif file_extension in ['xls', 'xlsx']:
            file_stream = io.BytesIO(file_content)
            test_data = pd.read_excel(file_stream)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Get trained model from Redis
        model_data = redis_client.get('automl_instance')
        if model_data is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        automl = pickle.loads(model_data)

        # Preprocess test data
        preprocessing_config = config.get('preprocessing', {})
        missing_data_config = preprocessing_config.get('missing_data', {})

        try:
            print("\nStarting preprocessing steps...")
            X_test = test_data.copy()

            # Handle missing values
            missing_strategy = missing_data_config.get('strategy')
            if missing_strategy == 'imputation':
                imputer_method = missing_data_config.get('imputation_method', 'mean')
                if imputer_method == 'constant':
                    constant_value = missing_data_config.get('constant_value', 0)
                    imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
                else:
                    imputer = SimpleImputer(strategy=imputer_method)
                X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
            elif missing_strategy == 'drop_rows':
                X_test = X_test.dropna()

            # Encode categorical variables
            X_test = detect_and_encode_categorical(X_test)
            
            # Get scalers from Redis and apply
            scalers_data = redis_client.get('scalers')
            if scalers_data is not None:
                scaler, scaler2 = pickle.loads(scalers_data)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
                X_test = pd.DataFrame(scaler2.transform(X_test), columns=X_test.columns)

            # Apply PCA if used in training
            if preprocessing_config.get('feature_reduction') == 'pca':
                pca_data = redis_client.get('pca')
                if pca_data is not None:
                    pca = pickle.loads(pca_data)
                    X_test = pd.DataFrame(pca.transform(X_test))

        except Exception as e:
            print(f"{Fore.RED}Error during preprocessing:{Style.RESET_ALL}")
            traceback.print_exc()
            raise ValueError(f"Error during data preprocessing: {str(e)}")

        # Generate predictions
        try:
            raw_predictions = automl.predict(X_test)
            
            # For classification, convert predictions back to original labels
            if config.get('problem_type') == 'classification':
                le_data = redis_client.get('label_encoder')
                if le_data is not None:
                    le = pickle.loads(le_data)
                    predictions = le.inverse_transform(raw_predictions)
                else:
                    predictions = raw_predictions
                
                # Get probabilities if available
                try:
                    probabilities = automl.predict_proba(X_test)
                except:
                    probabilities = None
            else:
                predictions = raw_predictions
                probabilities = None

        except Exception as e:
            print(f"{Fore.RED}Error during prediction:{Style.RESET_ALL}")
            traceback.print_exc()
            raise ValueError(f"Error generating predictions: {str(e)}")

        # Combine predictions with original data
        try:
            prediction_column = f'Predicted_{target_column}'
            test_data[prediction_column] = predictions

            if config.get('problem_type') == 'classification' and probabilities is not None:
                for i, prob in enumerate(probabilities.T):
                    test_data[f'Probability_Class_{i}'] = prob

            response_data = {
                "headers": test_data.columns.tolist(),
                "rows": test_data.to_dict('records'),
                "num_predictions": len(predictions),
                "prediction_column": prediction_column
            }

            # Store predictions in Redis
            redis_client.set('latest_predictions', pickle.dumps({
                'test_data': test_data,
                'predictions': predictions,
                'probabilities': probabilities
            }))

            print(f"\n{Fore.GREEN}Successfully completed prediction process{Style.RESET_ALL}")
            return jsonify(response_data), 200

        except Exception as e:
            raise ValueError(f"Error formatting prediction results: {str(e)}")

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"\n{Fore.RED}=== Error in Upload or Prediction Process ==={Style.RESET_ALL}")
        print(f"Error message: {error_msg}")
        print(f"Traceback:\n{traceback_str}")
        
        return jsonify({
            "error": error_msg,
            "traceback": traceback_str,
            "status": "error"
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    
    