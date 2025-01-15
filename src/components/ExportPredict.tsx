import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, AlertCircle } from "lucide-react";
import Predictionsviewer from "@/components/Predictionsviewer";

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
  const [fileUploaded, setFileUploaded] = useState(false);

  const handleFileUpload = async (data: { headers: string[]; rows: Record<string, string>[] } | null, fileName: string) => {
    if (!data || !fileName) {
      setPredictResults(null);
      setFileUploaded(false);
      return;
    }

    setLoading(true);
    setError('');
    setDetailedError('');
    setFileUploaded(true);
    console.log('Starting prediction process with file:', fileName);

    try {
      const csvContent = [
        data.headers.join(','),
        ...data.rows.map(row => data.headers.map(header => row[header] || '').join(','))
      ].join('\n');

      const file = new File([csvContent], fileName, { type: 'text/csv' });
      const formData = new FormData();
      formData.append('file', file);

      console.log('Sending prediction request...');
      const response = await fetch('https://babyml.onrender.com/upload-and-predict', {
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

  const downloadPredictions = () => {
    if (!predictResults) return;

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
  };

  return (
    <Card className="w-full mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl">Predict & Export</CardTitle>
        <CardDescription>
          Upload your test data to generate predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div> 
          <Predictionsviewer
            onFileUpload={handleFileUpload}
            initialData={predictResults}
            className="w-full"
          />

          {loading && (
            <Alert>
              <Upload className="h-4 w-4 animate-spin" />
              <AlertDescription>Generating predictions...</AlertDescription>
            </Alert>
          )}

          {error && fileUploaded && (
            <Alert variant="destructive" className="mt-10">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <div className="mt-1.5 font-medium">Error: {error}</div>
                {detailedError && (
                  <div className="text-xs whitespace-pre-wrap font-mono">
                    {detailedError}
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <div className="flex justify-between mt-12">
          <Button 
            variant="outline" 
            onClick={onBack}
            className="hover:bg-gray-100"
          >
            Back to Training
          </Button>

          {predictResults && (
            <Button
              onClick={downloadPredictions}
            >
              Download Predictions
            </Button>
          )}
        </div>

      </CardContent>
    </Card>
  );
};

export default ExportPredict;
