import io
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flaml import AutoML

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # React frontend URL

uploaded_file_data = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_data

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_content = file.read()
        if file.filename.endswith('.csv'):
            uploaded_file_data = pd.read_csv(io.BytesIO(file_content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            uploaded_file_data = pd.read_excel(io.BytesIO(file_content))
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Return column names as JSON
        return jsonify({"columns": list(uploaded_file_data.columns)}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


@app.route('/setup-training', methods=['POST'])
def setup_training():
    global uploaded_file_data

    if uploaded_file_data is None:
        return jsonify({"error": "No data uploaded"}), 400




if __name__ == '__main__':
    app.run(debug=True)
