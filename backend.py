import io
import pandas as pd
import numpy as np
import time
import subprocess
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import colorama
from colorama import Fore, Style
import threading


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

uploaded_file_data = None
automl_instance = None
log_output = []

def stream_terminal_output():
    """Capture terminal output in real-time from a subprocess."""
    process = subprocess.Popen(
        ['python', 'backend.py'],  # Replace with the command/script whose output you want to capture
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )
    
    # Read and stream stdout and stderr from the subprocess
    while True:
        output = process.stdout.readline()
        if output:
            log_output.append(output.strip())  # Capture the output
        error_output = process.stderr.readline()
        if error_output:
            log_output.append(error_output.strip())  # Capture any error output
        if process.poll() is not None:
            break
        time.sleep(0.1)

def generate_log_stream():
    """Generator function to stream logs to the frontend."""
    global log_output
    while True:
        if log_output:
            log_entry = log_output.pop(0)  # Take the first log entry
            yield f"data: {log_entry}\n\n"
        time.sleep(1)

@app.route('/stream-logs')
def stream_logs_endpoint():
    def generate():
        try:
            while True:
                # Check if there are any logs to send
                if log_output:
                    log_entry = log_output.pop(0)
                    yield f"data: {log_entry}\n\n"
                else:
                    # Send a heartbeat every 15 seconds to keep connection alive
                    yield f"data: heartbeat\n\n"
                time.sleep(1)
        except GeneratorExit:
            print("Client disconnected from event stream")
    
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
    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400
    columns = list(uploaded_file_data.columns)
    return jsonify({"columns": columns}), 200

@app.route('/set-config', methods=['POST'])
def set_config():
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
    global uploaded_file_data, automl_instance
    print(f"\n{Fore.GREEN}=== Received Training Setup Request ==={Style.RESET_ALL}")
    
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
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"\n{Fore.CYAN}Unique classes in target: {le.classes_}{Style.RESET_ALL}")
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
            'time_budget': 20,  # seconds
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
        print(f"\n{Fore.GREEN}=== Training Complete ==={Style.RESET_ALL}")
        print(f"Best ML learner: {best_model.__class__.__name__}")
        print(f"Best hyperparameter config: {automl_instance.best_config}")
        print(f"Best score: {automl_instance.best_loss}")

        response = {
            "status": "success",
            "best_estimator": best_model.__class__.__name__,
            "best_config": automl_instance.best_config,
            "best_score": automl_instance.best_loss,
            "feature_importance": automl_instance.feature_importance() if hasattr(automl_instance, 'feature_importance') else None
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"{Fore.RED}Error in training setup: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": f"Error in training setup: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting terminal output stream...")
    
    # Start the terminal output capturing in a separate thread
    log_thread = threading.Thread(target=stream_terminal_output)
    log_thread.daemon = True  # Daemonize the thread so it ends with the main program
    log_thread.start()

    app.run(debug=True)
