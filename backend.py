import io
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # React frontend URL

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the file content into memory
        file_content = file.read()

        # Check if it's a CSV file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        
        # Check if it's an Excel file (xlsx or xls)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
        
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Print the DataFrame (for testing purposes)
        print(df)

        # No message, just return status 200 (OK)
        return '', 200

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
