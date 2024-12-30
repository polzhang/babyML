import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import logging
import colorama
from colorama import Fore, Style

colorama.init()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

uploaded_file_data = None
automl_instance = None

@app.route('/get-columns', methods=['GET'])
def get_columns():
    global uploaded_file_data
    
    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400
    
    # Return the columns as a list
    columns = list(uploaded_file_data.columns)
    return jsonify({"columns": columns}), 200

@app.route('/set-config', methods=['POST'])
def set_config():
    # Extract the JSON payload sent from the frontend
    config = request.json

    # You can now access the 'config' dictionary, e.g., print it to the console
    print("Received configuration:", config)

    # Process the config as needed, for example, you can store it in a database or file
    # Or you can return some kind of success message
    return jsonify({"message": "Configuration received successfully!"}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_data

    print(f"\n{Fore.GREEN}=== Received File Upload Request ==={Style.RESET_ALL}")

    # Check if the file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Check file extension
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            file_stream = io.StringIO(file.stream.read().decode("utf-8"))
            uploaded_file_data = pd.read_csv(file_stream)
        elif file_extension in ['xls', 'xlsx']:
            uploaded_file_data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Print the head of the DataFrame
        print(f"\n{Fore.CYAN}=== DataFrame Head ==={Style.RESET_ALL}")
        print(uploaded_file_data.head())

        return jsonify({"message": "File uploaded successfully", "columns": list(uploaded_file_data.columns)}), 200

    except Exception as e:
        print(f"{Fore.RED}Error processing file: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/setup-training', methods=['GET'])
def setup_training():
    global uploaded_file_data, automl_instance
    
    print(f"\n{Fore.GREEN}=== Received Training Setup Request ==={Style.RESET_ALL}")
    
    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        config = request.json
        print(f"\n{Fore.YELLOW}Received Configuration:{Style.RESET_ALL}")
        print(config)

        # Extract configuration
        target_column = config['target_variable']
        problem_type = config['problem_type']
        train_split = config.get('train_test_split', 0.8)
        random_state = config.get('random_state', 42)

        # Prepare data
        X = uploaded_file_data.drop(columns=[target_column])
        y = uploaded_file_data[target_column]

        # Handle missing data
        if config['preprocessing']['missing_data']['strategy'] == 'imputation':
            imputer_method = config['preprocessing']['missing_data']['imputation_method']
            imputer = SimpleImputer(strategy=imputer_method)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        elif config['preprocessing']['missing_data']['strategy'] == 'drop_rows':
            X = X.dropna()
            y = y[X.index]

        # Apply standardisation & normalisation
        
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Apply feature reduction if selected
        if config['preprocessing']['feature_reduction'] == 'pca':
            pca = PCA(n_components=0.95)  # Preserve 95% variance
            X = pd.DataFrame(pca.fit_transform(X))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_split,
            random_state=random_state
        )

        print(f"\n{Fore.CYAN}=== Training Data Shape ==={Style.RESET_ALL}")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")

        # Setup FLAML
        automl_instance = AutoML()
        
        # Configure settings based on problem type
        settings = {
            'time_budget': 3600,  # 1 hour time budget
            'metric': config['validation']['metric'],
            'task': problem_type,
            'n_jobs': -1,
            'estimator_list': config['models']['selected'] if config['models']['selected'] else None,
            'eval_method': config['validation']['method'],
            'n_splits': config['validation']['k_folds'] if config['validation']['method'] == 'kfold' else 5,
            'verbose': 2
        }

        print(f"\n{Fore.YELLOW}=== Starting FLAML Training ==={Style.RESET_ALL}")
        print("Settings:", settings)

        # Train the model
        automl_instance.fit(
            X_train=X_train,
            y_train=y_train,
            **settings
        )

        # Get results
        best_model = automl_instance.model.estimator
        print(f"\n{Fore.GREEN}=== Training Complete ==={Style.RESET_ALL}")
        print(f"Best ML leaner: {best_model.__class__.__name__}")
        print(f"Best hyperparmeter config: {automl_instance.best_config}")
        print(f"Best score: {automl_instance.best_loss}")

        # Prepare response
        response = {
            "status": "success",
            "best_estimator": best_model.__class__.__name__,
            "best_config": automl_instance.best_config,
            "best_score": automl_instance.best_loss,
            "feature_importance": automl_instance.feature_importance() if hasattr(automl_instance, 'feature_importance') else None
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"{Fore.RED}Error during training: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"""
{Fore.GREEN}=======================================
🚀 FLAML AutoML Server is Running!
=======================================
{Fore.CYAN}
Server Details:
- URL: http://localhost:5000
- CORS: Enabled for http://localhost:3000
- AutoML Ready
{Fore.YELLOW}
Endpoints:
- POST /upload : Upload and process data files
- POST /setup-training : Configure and start training
{Style.RESET_ALL}
""")
    app.run(debug=True)
