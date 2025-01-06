"use client";
import * as React from 'react';
import { Box, Stepper, Step, StepLabel, Button, Typography } from '@mui/material';
import FileUploadViewer from '@/components/FileUploadViewer';
import Component2 from '@/components/Component2';
import ExportPredict from '@/components/ExportPredict'; 

// Types
interface FileData {
  headers: string[];
  rows: Record<string, string>[];
}

interface StepComponentProps {
  onNext: () => void;
  onBack?: () => void;
  fileData: FileData | null;
  setFileData: (data: FileData | null) => void;
  fileName: string;
  setFileName: (name: string) => void;
}

const STEPS = ['Upload Training Data', 'Select Training Options', 'Predict/Export'] as const;

// Step Components
const Step1Component: React.FC<StepComponentProps> = ({ 
  onNext, 
  fileData, 
  setFileData, 
  fileName,
  setFileName 
}) => (
  <div className="mt-5">
    <FileUploadViewer
      className="relative p-4 rounded-lg shadow-lg"
      handleNext={onNext}
      initialData={fileData}
      initialFileName={fileName}
      onFileUpload={(data, name) => {
        setFileData(data);
        setFileName(name);
      }}
    />
  </div>
);

const Step2Component: React.FC<StepComponentProps> = ({ onNext, onBack }) => (
  <Component2
    className="p-4 rounded-lg"
    handleNext={onNext}
    handleBack={onBack}
  />
);

const Step3Component: React.FC<StepComponentProps> = ({ onBack }) => (
  <ExportPredict
    onBack={onBack}/>
);

export default function PageContainer() {
  // State
  const [activeStep, setActiveStep] = React.useState(0);
  const [fileData, setFileData] = React.useState<FileData | null>(null);
  const [fileName, setFileName] = React.useState('');

  // Handlers
  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setFileData(null);
    setFileName('');
  };

  // Step components mapping with consistent props
  const stepComponents = [
    <Step1Component
      key="step1"
      onNext={handleNext}
      fileData={fileData}
      setFileData={setFileData}
      fileName={fileName}
      setFileName={setFileName}
    />,
    <Step2Component
      key="step2"
      onNext={handleNext}
      onBack={handleBack}
      fileData={fileData}
      setFileData={setFileData}
      fileName={fileName}
      setFileName={setFileName}
    />,
    <Step3Component 
      key="step3" 
      onBack={handleBack}
      onNext={handleNext}
      fileData={fileData}
      setFileData={setFileData}
      fileName={fileName}
      setFileName={setFileName}
     />,
  ];

  // Log state changes for debugging
  React.useEffect(() => {
    console.log('File Data:', fileData);
    console.log('File Name:', fileName);
    console.log('Active Step:', activeStep);
  }, [fileData, fileName, activeStep]);

  return (
    <div className="min-h-screen bg-white">
      <div className="px-4 py">
        <img
          src="/logo.png"
          alt="Logo"
          className="w-48 h-auto mb-8"
        />
        
        <Box className="max-w-7xl mx-auto">
          <Stepper activeStep={activeStep}>
            {STEPS.map((label) => (
              <Step key={label}>
                <StepLabel 
                  sx={{ '& .MuiStepLabel-label': { fontSize: '18px' } }}
                >
                  {label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          <div className="mt-8">
            {activeStep === STEPS.length ? (
              <div>
                <Typography sx={{ mt: 2, mb: 1 }}>
                  All steps completed successfully
                </Typography>
                <Box sx={{ display: 'flex', pt: 2 }}>
                  <Box sx={{ flex: '1 1 auto' }} />
                  <Button
                    variant="contained"
                    onClick={handleReset}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Reset
                  </Button>
                </Box>
              </div>
            ) : (
              <div>{stepComponents[activeStep]}</div>
            )}
          </div>
        </Box>
      </div>
    </div>
  );
}