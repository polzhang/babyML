import io
import pandas as pd
import numpy as np
import time
import logging
import traceback
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import colorama
import queue
import json
from colorama import Fore, Style
from flaml.automl.logger import logger as flaml_logger


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
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})


uploaded_file_data = None #TRAIN DATA
automl_instance = None




@app.route('/stream-logs')
def stream_logs_endpoint():
    def generate():
        while True:
            try:
                # Non-blocking queue get with timeout
                log_entry = log_queue.get_nowait()
                yield f"data: {log_entry}\n\n"
            except queue.Empty:
                # Send heartbeat if no logs
                yield f"data: heartbeat\n\n"
                time.sleep(0.1)  # Reduced sleep time for more responsive logging
    
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/start-training', methods=['POST'])
def start_training():
    """Simulated endpoint for starting the training process."""
    # Here you can invoke the terminal command or script that runs your training.
    return jsonify({"message": "Training setup started!"}), 200

@app.route('/get-columns', methods=['GET'])
def get_columns():
    global uploaded_file_data
    print("get_columns called, uploaded_file_data is:", "None" if uploaded_file_data is None else "present")
    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400
    columns = list(uploaded_file_data.columns)
    print("Returning columns:", columns)
    return jsonify({"columns": columns}), 200

@app.route('/set-config', methods=['POST'])
def set_config():
    global config
    config = request.json
    print("Received configuration:", config)
    return jsonify({"message": "Configuration received successfully!"}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_data
    print(f"\n{Fore.GREEN}=== Received File Upload Request ==={Style.RESET_ALL}")

    if 'file' not in request.files:
        print(f"{Fore.RED}Error: No file part in request{Style.RESET_ALL}")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        print(f"{Fore.RED}Error: No selected file{Style.RESET_ALL}")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the file content
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        print(f"File name: {file.filename}")
        print(f"File content length: {len(file_content)} bytes")
        
        file_extension = file.filename.split('.')[-1].lower()
        print(f"File extension: {file_extension}")

        if file_extension == 'csv':
            # For CSV files, use StringIO
            file_stream = io.StringIO(file_content.decode("utf-8"))
            uploaded_file_data = pd.read_csv(file_stream)
        elif file_extension in ['xls', 'xlsx']:
            # For Excel files, use BytesIO
            file_stream = io.BytesIO(file_content)
            uploaded_file_data = pd.read_excel(file_stream)
        else:
            print(f"{Fore.RED}Error: Unsupported file type: {file_extension}{Style.RESET_ALL}")
            return jsonify({"error": "Unsupported file type"}), 400

        print(f"\n{Fore.CYAN}=== DataFrame Info ==={Style.RESET_ALL}")
        print(f"Shape: {uploaded_file_data.shape}")
        print("\nColumns:")
        for col in uploaded_file_data.columns:
            print(f"- {col}")

        return jsonify({
            "message": "File uploaded successfully",
            "columns": list(uploaded_file_data.columns),
            "rows": len(uploaded_file_data)
        }), 200

    except UnicodeDecodeError as e:
        print(f"{Fore.RED}Error: File encoding issue: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": "File encoding error. Please ensure the file is properly encoded"}), 500
    except pd.errors.EmptyDataError:
        print(f"{Fore.RED}Error: The file is empty{Style.RESET_ALL}")
        return jsonify({"error": "The uploaded file is empty"}), 500
    except pd.errors.ParserError as e:
        print(f"{Fore.RED}Error parsing file: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": "Error parsing file. Please check the file format"}), 500
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

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

@app.route('/setup-training', methods=['POST'])
def setup_training():
    global uploaded_file_data, automl_instance, config
    print(f"\n{Fore.GREEN}=== Received Training Setup Request ==={Style.RESET_ALL}")
    
    # Clear the log queue before starting new training
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
    
    # Clear previous automl instance
    if automl_instance is not None:
        del automl_instance
        automl_instance = None


    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        config = request.json
        print(f"\n{Fore.YELLOW}Received Configuration:{Style.RESET_ALL}")
        print(config)

        # Extract required values from config with defaults
        target_column = config.get('target_variable')
        problem_type = config.get('problem_type', 'classification')
        validation_settings = config.get('validation', {})
        validation_method = validation_settings.get('method', 'holdout')
        split_ratio = validation_settings.get('split_ratio', 0.8)  # Default to 0.8 if not provided
        
        if not target_column:
            return jsonify({"error": "Target variable not specified"}), 400

        X = uploaded_file_data.drop(columns=[target_column])
        y = uploaded_file_data[target_column]

        # Classification preprocessing
        if problem_type == 'classification':
            if pd.api.types.is_numeric_dtype(y):
                print(f"\n{Fore.YELLOW}Converting numeric target to categorical for classification{Style.RESET_ALL}")
                y = y.astype(str)
            global le 
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    
                # Print the label mapping
            print(f"\n{Fore.CYAN}Label Mapping (Numeric to Original Class):{Style.RESET_ALL}")
            for num, label in label_mapping.items():
                print(f"{num} -> {label}")
            
        elif problem_type == 'regression':
            if not pd.api.types.is_numeric_dtype(y):
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

        # Feature reduction
        if preprocessing_config.get('feature_reduction') == 'pca':
            pca = PCA(n_components=0.95)
            X = pd.DataFrame(pca.fit_transform(X))

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
        automl_instance = AutoML()
        settings = {
            'time_budget': config.get('time_budget') ,  # seconds
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

        automl_instance.fit(
            X_train=X_train,
            y_train=y_train,
            **settings
        )

        best_model = automl_instance.model.estimator
        best_model = automl_instance.model.estimator
        
        # Calculate additional metrics based on problem type
        if problem_type == 'classification':
            from sklearn.metrics import (
                classification_report, confusion_matrix, 
                roc_auc_score, precision_recall_fscore_support
            )
            
            y_pred = automl_instance.predict(X_test)
            y_pred_proba = automl_instance.predict_proba(X_test)
            
            # Get classification metrics
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            # Calculate ROC AUC (handle binary/multiclass)
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
                "training_time": automl_instance.time_to_find_best_model,
                "models_trained": len(automl_instance.estimator_list),
                "best_iteration": automl_instance.best_iteration
            }
            
            print(f"\n{Fore.CYAN}=== Detailed Metrics ==={Style.RESET_ALL}")
            print(f"ROC AUC Score: {roc_auc:.4f}" if roc_auc else "ROC AUC: N/A")
            print(f"Accuracy: {class_report['accuracy']:.4f}")
            print(f"Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
            print(f"Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}")
            
            log_queue.put(f"\n{Fore.CYAN}=== Detailed Metrics ==={Style.RESET_ALL}")
            log_queue.put(f"ROC AUC Score: {roc_auc:.4f}" if roc_auc else "ROC AUC: N/A")
            log_queue.put(f"Accuracy: {class_report['accuracy']:.4f}")
            log_queue.put(f"Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
            log_queue.put(f"Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}")

            
        else:  # regression
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, 
                r2_score, mean_absolute_percentage_error
            )
            
            y_pred = automl_instance.predict(X_test)
            
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
                "training_time": automl_instance.time_to_find_best_model,
                "models_trained": len(automl_instance.estimator_list),
                "best_iteration": automl_instance.best_iteration
            }
            
            print(f"\n{Fore.CYAN}=== Detailed Metrics ==={Style.RESET_ALL}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")

            log_queue.put(f"\n{Fore.CYAN}=== Detailed Metrics ==={Style.RESET_ALL}")
            log_queue.put(f"RMSE: {rmse:.4f}")
            log_queue.put(f"MAE: {mae:.4f}")
            log_queue.put(f"R²: {r2:.4f}")


            if mape is not None:
                print(f"MAPE: {mape:.4f}")
                log_queue.put(f"MAPE: {mape:.4f}")

        # Get feature importance if available
        try:
            feature_importance = automl_instance.feature_importance()
            if isinstance(feature_importance, pd.Series):
                feature_importance = feature_importance.to_dict()
        except:
            feature_importance = None

        print(f"\n{Fore.GREEN}=== Training Complete ==={Style.RESET_ALL}")
        print(f"Best ML learner: {best_model.__class__.__name__}")
        print(f"Best hyperparameter config: {automl_instance.best_config}")
        print(f"Training time: {automl_instance.time_to_find_best_model:.2f} seconds")
        print(f"Models trained: {len(automl_instance.estimator_list)}")

        response = {
            "status": "success",
            "best_estimator": best_model.__class__.__name__,
            "best_config": automl_instance.best_config,
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

@app.route('/upload-and-predict', methods=['POST'])
def upload_and_predict():
    """Handle both test data upload and prediction."""
    global test_data, config, automl_instance
    print("upload_and_predict called")
    print("uploaded_file_data before:", "None" if uploaded_file_data is None else "present")
    target_column = config.get('target_variable')
    if not target_column:
        print(f"{Fore.RED}Error: 'target_variable' not found in config{Style.RESET_ALL}")
        return jsonify({"error": "'target_variable' not found in config"}), 400


    print(f"\n{Fore.GREEN}=== Received Request for Upload and Prediction ==={Style.RESET_ALL}")

    if 'file' not in request.files:
        print(f"{Fore.RED}Error: No file part in request{Style.RESET_ALL}")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        print(f"{Fore.RED}Error: No selected file{Style.RESET_ALL}")
        return jsonify({"error": "No selected file"}), 400

    print("uploaded_file_data after:", "None" if uploaded_file_data is None else "present")
    try:
        # Save the file content
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        print(f"File name: {file.filename}")
        print(f"File content length: {len(file_content)} bytes")
        
        file_extension = file.filename.split('.')[-1].lower()
        print(f"File extension: {file_extension}")

        # Upload file
        if file_extension == 'csv':
            file_stream = io.StringIO(file_content.decode("utf-8"))
            test_data = pd.read_csv(file_stream)
        elif file_extension in ['xls', 'xlsx']:
            file_stream = io.BytesIO(file_content)
            test_data = pd.read_excel(file_stream)
        else:
            print(f"{Fore.RED}Error: Unsupported file type: {file_extension}{Style.RESET_ALL}")
            return jsonify({"error": "Unsupported file type"}), 400

        print(f"\n{Fore.CYAN}=== Uploaded Test DataFrame Info ==={Style.RESET_ALL}")
        print(f"Shape: {test_data.shape}")
        print("\nColumns:")
        for col in test_data.columns:
            print(f"- {col}")

        # Get config from form data
        config_str = request.form.get('config', '{}')
        try:
            config = json.loads(config_str)
        except:
            config = {}

        preprocessing_config = config.get('preprocessing', {})
        missing_data_config = preprocessing_config.get('missing_data', {})
        
        # Check for automl_instance
        if automl_instance is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # 1. Preprocess test data
        try:
            print("\nStarting preprocessing steps...")
            X_test = test_data.copy()

            # Handle missing values
            print("Handling missing values...")
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
            print("Encoding categorical variables...")
            X_test = detect_and_encode_categorical(X_test)
            print(f"Columns after categorical encoding: {X_test.columns.tolist()}")

            # Scale features
            print("Scaling features...")
            scaler = StandardScaler()
            scaler2 = MinMaxScaler()
            X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
            X_test = pd.DataFrame(scaler2.fit_transform(X_test), columns=X_test.columns)

            # Feature reduction if used in training
            if preprocessing_config.get('feature_reduction') == 'pca':
                print("Applying PCA feature reduction...")
                pca = PCA(n_components=0.95)
                X_test = pd.DataFrame(pca.fit_transform(X_test))

            print(f"Final preprocessed data shape: {X_test.shape}")
            print(f"Final columns: {X_test.columns.tolist()}")

        except Exception as e:
            print(f"{Fore.RED}Error during preprocessing:{Style.RESET_ALL}")
            traceback.print_exc()
            raise ValueError(f"Error during data preprocessing: {str(e)}")

        # 2. Generate predictions
        try:
            print("\nGenerating predictions...")
            predictions = le.inverse_transform(automl_instance.predict(X_test))
            print(f"Successfully generated {len(predictions)} predictions")

            # For classification problems, get probabilities if available
            if config.get('problem_type') == 'classification':
                try:
                    probabilities = automl_instance.predict_proba(X_test)
                    print("Generated prediction probabilities")
                except:
                    probabilities = None
                    print("Could not generate prediction probabilities")

        except Exception as e:
            print(f"{Fore.RED}Error during prediction:{Style.RESET_ALL}")
            traceback.print_exc()
            raise ValueError(f"Error generating predictions: {str(e)}")

        # 3. Combine predictions with original data
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
    
    app.run(debug=True)

    
    