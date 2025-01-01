import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
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
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            file_stream = io.StringIO(file.stream.read().decode("utf-8"))
            uploaded_file_data = pd.read_csv(file_stream)
        elif file_extension in ['xls', 'xlsx']:
            uploaded_file_data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        print(f"\n{Fore.CYAN}=== DataFrame Head ==={Style.RESET_ALL}")
        print(uploaded_file_data.head())
        return jsonify({"message": "File uploaded successfully", "columns": list(uploaded_file_data.columns)}), 200

    except Exception as e:
        print(f"{Fore.RED}Error processing file: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

def detect_and_encode_categorical(df, max_unique_ratio=0.05):
    """
    Automatically detect and encode categorical columns in the dataframe.
    
    Args:
        df: Input DataFrame
        max_unique_ratio: Maximum ratio of unique values to total values to consider a column categorical
        
    Returns:
        Encoded DataFrame
    """
    categorical_columns = []
    
    for column in df.columns:
        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        # Calculate ratio of unique values
        unique_ratio = len(df[column].unique()) / len(df)
        
        # Check if column is categorical based on dtype or unique ratio
        is_categorical = (
            pd.api.types.is_string_dtype(df[column]) or 
            pd.api.types.is_object_dtype(df[column]) or 
            pd.api.types.is_categorical_dtype(df[column]) or
            unique_ratio <= max_unique_ratio
        )
        
        if is_categorical:
            categorical_columns.append(column)
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    if categorical_columns:
        print(f"\n{Fore.YELLOW}=== Detected Categorical Columns ==={Style.RESET_ALL}")
        print(categorical_columns)
        
        for column in categorical_columns:
            n_unique = len(df[column].unique())
            
            # Use Label Encoding for binary categories or when there are many categories
            if n_unique == 2 or n_unique > 10:
                print(f"Applying Label Encoding to: {column}")
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df[column].astype(str))
            
            # Use One-Hot Encoding for categorical variables with few categories
            else:
                print(f"Applying One-Hot Encoding to: {column}")
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = ohe.fit_transform(df[[column]])
                
                # Create new column names for one-hot encoded features
                new_columns = [f"{column}_{cat}" for cat in ohe.categories_[0]]
                
                # Add encoded features to dataframe
                for i, new_col in enumerate(new_columns):
                    df_encoded[new_col] = encoded_features[:, i]
                
                # Drop original column
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

        target_column = config['target_variable']
        problem_type = config['problem_type']
        validation_method = config['validation']['method']
        split_ratio = config['validation'].get('split_ratio', 0.8)
        
        if isinstance(split_ratio, int):
            split_ratio = split_ratio / 100

        X = uploaded_file_data.drop(columns=[target_column])
        y = uploaded_file_data[target_column]

        # Convert target variable based on problem type
        if problem_type == 'classification':
            # Check if target is numeric but should be treated as categorical
            if pd.api.types.is_numeric_dtype(y):
                print(f"\n{Fore.YELLOW}Converting numeric target to categorical for classification{Style.RESET_ALL}")
                y = y.astype(str)
            
            # Use LabelEncoder for classification tasks
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"\n{Fore.CYAN}Unique classes in target: {le.classes_}{Style.RESET_ALL}")
        elif problem_type == 'regression':
            # Ensure target is numeric for regression
            if not pd.api.types.is_numeric_dtype(y):
                return jsonify({"error": "Target variable must be numeric for regression tasks"}), 400
            y = y.astype(float)

        # Handle missing data
        if config['preprocessing']['missing_data']['strategy'] == 'imputation':
            imputer_method = config['preprocessing']['missing_data']['imputation_method']
            if imputer_method == 'constant':
                constant_value = config.get('preprocessing', {}).get('missing_data', {}).get('constant_value', 0)
                imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            else:
                imputer = SimpleImputer(strategy=imputer_method)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        elif config['preprocessing']['missing_data']['strategy'] == 'drop_rows':
            X = X.dropna()
            y = y[X.index]

        # Automatically detect and encode categorical variables
        X = detect_and_encode_categorical(X)
        
        # Scale numerical features
        scaler = StandardScaler()
        scaler2 = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        X = pd.DataFrame(scaler2.fit_transform(X), columns=X.columns)

        if config['preprocessing']['feature_reduction'] == 'pca':
            pca = PCA(n_components=0.95)
            X = pd.DataFrame(pca.fit_transform(X))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=split_ratio,
            random_state=42,
            stratify=y if problem_type == 'classification' else None
        )

        print(f"\n{Fore.CYAN}=== Training Data Shape ==={Style.RESET_ALL}")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        if problem_type == 'classification':
            print(f"Number of classes: {len(np.unique(y))}")
            print(f"Class distribution: {np.bincount(y)}")

        automl_instance = AutoML()
        settings = {
            'time_budget': 20,
            'metric': config['validation']['metric'],
            'task': problem_type,
            'n_jobs': -1,
            'estimator_list': config['models'].get('selected'),
            'eval_method': validation_method,
            'verbose': 3
        }

        # Add appropriate validation parameters based on method
        if validation_method == 'holdout':
            settings['split_ratio'] = split_ratio
        elif validation_method == 'kfold':
            settings['n_splits'] = int(config['validation']['k_folds'])

        print(f"\n{Fore.YELLOW}=== Starting FLAML Training ==={Style.RESET_ALL}")
        print("Settings:", settings)

        automl_instance.fit(
            X_train=X_train,
            y_train=y_train,
            **settings
        )

        best_model = automl_instance.model.estimator
        print(f"\n{Fore.GREEN}=== Training Complete ==={Style.RESET_ALL}")
        print(f"Best ML leaner: {best_model.__class__.__name__}")
        print(f"Best hyperparmeter config: {automl_instance.best_config}")
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


    except Exception as e:
        print(f"{Fore.RED}Error in training setup: {str(e)}{Style.RESET_ALL}")
        return jsonify({"error": f"Error in training setup: {str(e)}"}), 500


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