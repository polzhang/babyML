import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, Download, AlertCircle } from "lucide-react";
import FileDropZone2 from "@/components/FileDropZone2";  // Import FileDropzone2 component

interface ExportPredictProps {
  onBack: () => void;
}

const ExportPredict: React.FC<ExportPredictProps> = ({ onBack }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [predictResults, setPredictResults] = useState(null);

  const handleFileDrop = (uploadedFiles: File[]) => {
    if (uploadedFiles.length > 0) {
      const uploadedFile = uploadedFiles[0];
      if (!uploadedFile.name.match(/\.(csv|xlsx)$/)) {
        setError('Please upload a CSV or Excel file');
        return;
      }
      setFile(uploadedFile);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please upload a file first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }

      const results = await response.json();
      setPredictResults(results);

      if (results.predictions) {
        const csvContent = "data:text/csv;charset=utf-8," 
          + results.predictions.map(row => row.join(",")).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "predictions.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl">Predict & Export
            
        </CardTitle>
        <CardDescription  >
        Generate predictions by applying the trained model to your test data
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-4">
          <FileDropZone2 
            onFileSelect={handleFileDrop}
            acceptedFileTypes={['.csv', '.xlsx']}
            maxFiles={1}
          />
          
          {file && (
            <div className="flex justify-end mt-4">
              <Button
                onClick={handlePredict}
                disabled={loading}
                className="min-w-[120px]"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <Upload className="animate-spin" size={16} />
                    Processing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Download size={16} />
                    Predict
                  </span>
                )}
              </Button>
            </div>
          )}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {predictResults && (
          <Alert>
            <AlertDescription>
              Predictions generated successfully! The file has been automatically downloaded.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex justify-start mt-6">
          <Button 
            variant="outline" 
            onClick={onBack}
            className="hover:bg-gray-100"
          >
            Back to Training
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExportPredict;