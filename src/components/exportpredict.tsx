import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, AlertCircle } from "lucide-react";
import FileUploadViewer from "@/components/FileUploadViewer";

interface ExportPredictProps {
  onBack: () => void;
}

const ExportPredict: React.FC<ExportPredictProps> = ({ onBack }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [detailedError, setDetailedError] = useState('');
  const [predictResults, setPredictResults] = useState<{
    headers: string[];
    rows: Record<string, string>[];
  } | null>(null);

  const handleFileUpload = async (data: { headers: string[]; rows: Record<string, string>[] }, fileName: string) => {
    if (!fileName) return;

    setLoading(true);
    setError('');
    setDetailedError('');
    console.log('Starting prediction process with file:', fileName);

    try {
      // Create CSV from data
      const csvContent = [
        data.headers.join(','),
        ...data.rows.map(row => data.headers.map(header => row[header] || '').join(','))
      ].join('\n');

      const file = new File([csvContent], fileName, { type: 'text/csv' });
      const formData = new FormData();
      formData.append('file', file);

      // Make prediction request
      console.log('Sending prediction request...');
      const response = await fetch('http://localhost:5000/upload-and-predict', {
        method: 'POST',
        body: formData,
      });

      const responseData = await response.json();
      console.log('Received response:', responseData);

      if (!response.ok) {
        throw new Error(responseData.error || 'Failed to generate predictions');
      }

      if (responseData.status === 'error') {
        throw new Error(responseData.error || 'Error in prediction process');
      }

      if (!responseData.headers || !responseData.rows) {
        throw new Error('Invalid response format from server');
      }

      setPredictResults({
        headers: responseData.headers,
        rows: responseData.rows,
      });

    } catch (err) {
      console.error('Prediction error:', err);
      
      if (err instanceof Error) {
        setError(err.message);
        // If the error response contains a traceback, show it
        const errorResponse = err as any;
        if (errorResponse.traceback) {
          setDetailedError(errorResponse.traceback);
        }
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl">Predict & Export</CardTitle>
        <CardDescription>
          Upload your test data to generate predictions
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-4">
          <FileUploadViewer
            onFileUpload={handleFileUpload}
            initialData={predictResults}
            className="w-full"
          />

          {loading && (
            <Alert>
              <Upload className="h-4 w-4 animate-spin" />
              <AlertDescription>Processing predictions...</AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <div className="font-medium">Error: {error}</div>
                {detailedError && (
                  <div className="mt-2 text-xs whitespace-pre-wrap font-mono">
                    {detailedError}
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}

          {predictResults && (
            <div className="flex justify-end space-x-4">
              <Button
                onClick={() => {
                  const csvContent = [
                    predictResults.headers.join(','),
                    ...predictResults.rows.map(row => 
                      predictResults.headers.map(header => row[header] || '').join(',')
                    )
                  ].join('\n');

                  const blob = new Blob([csvContent], { type: 'text/csv' });
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'predictions.csv';
                  a.click();
                  window.URL.revokeObjectURL(url);
                }}
              >
                Download Predictions
              </Button>
            </div>
          )}
        </div>

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