/* eslint-disable */
"use client";
import * as React from 'react';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import FileUploadViewer from '@/components/FileUploadViewer';
import Component2 from '@/components/Component2';

// Define your steps as you have
const steps = ['Upload Training Data', 'Select Training Parameters', 'Predict/Export'];
interface StepProps {
  onNext: () => void;  // Add prop for handling next step
  onBack?: () => void;
}
interface FileData {
  headers: string[];
  rows: { [key: string]: string }[];
}



// Example custom components for each step
const Step1Component: React.FC<StepProps> = ({ onNext, fileData, setFileData, setFileName }) => (
  <div>
    <FileUploadViewer
      className="relative top-[20px] p-4 rounded-lg shadow-lg w"
      handleNext={onNext}
      initialData={fileData}  // Pass initialData to FileUploadViewer
      onFileUpload={(data, name) => {
        setFileData(data);  // Update the file data when the file is uploaded
        setFileName(name);  // Set the file name
      }}
    />
  </div>
);
const Step2Component: React.FC<StepProps> = ({ onNext, onBack }) => (
  <Component2
    className="p-4 rounded-lg"
    handleNext={onNext}
    handleBack={onBack}
  />  
);
const Step3Component = () => <div>Step 3 Content</div>;

export default function PageContainer() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [skipped, setSkipped] = React.useState(new Set<number>());
  const [fileData, setFileData] = React.useState<{ headers: string[]; rows: { [key: string]: string }[] } | null>(null);
  const [fileName, setFileName] = React.useState<string>('');

  const isStepOptional = (step: number) => step === 1;
  const isStepSkipped = (step: number) => skipped.has(step);

  const handleNext = () => {
    let newSkipped = skipped;
    if (isStepSkipped(activeStep)) {
      newSkipped = new Set(newSkipped.values());
      newSkipped.delete(activeStep);
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped(newSkipped);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleSkip = () => {
    if (!isStepOptional(activeStep)) {
      throw new Error("You can't skip a step that isn't optional.");
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped((prevSkipped) => {
      const newSkipped = new Set(prevSkipped.values());
      newSkipped.add(activeStep);
      return newSkipped;
    });
  };

  const handleReset = () => {
    setActiveStep(0);
    setFileData(null);  // Reset the file data when resetting
    setFileName('');
  };

  // Create an array of the step components you want to render
  const stepComponents = [
    <Step1Component
      key="step1"
      onNext={() => setActiveStep((prevStep) => prevStep + 1)}
      fileData={fileData}  // Pass fileData to Step1Component
      setFileData={setFileData}  // Pass setter to Step1Component
      setFileName={setFileName}  // Pass setter to set the file name
    />,
    <Step2Component
      key="step2"
      onNext={() => setActiveStep((prevStep) => prevStep + 1)}
      onBack={handleBack}
    />,
    <Step3Component />,
  ];

  return (
    <div>
      <img
        src="/logo.png"
        alt="Logo"
        className="left-3 relative w-48 h-auto"
      />
      <div className="mt-6 ml-12 flex items-center justify-center w-[90vw]">
        <Box sx={{ width: '100%' }}>
          <Stepper activeStep={activeStep}>
            {steps.map((label, index) => {
              const stepProps: { completed?: boolean } = {};
              const labelProps: {
                optional?: React.ReactNode;
              } = {};
              if (isStepOptional(index)) {
                labelProps.optional = (
                  <Typography variant="caption"></Typography>
                );
              }
              if (isStepSkipped(index)) {
                stepProps.completed = false;
              }
              return (
                <Step key={label} {...stepProps}>
                  <StepLabel {...labelProps} sx={{ '& .MuiStepLabel-label': { fontSize: '18px' } }}>
                    {label}
                  </StepLabel>
                </Step>
              );
            })}
          </Stepper>

          {activeStep === steps.length ? (
            <React.Fragment>
              <Typography sx={{ mt: 2, mb: 1 }}>
                All steps completed - you're finished
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
                <Box sx={{ flex: '1 1 auto' }} />
                <Button onClick={handleReset}>Reset</Button>
              </Box>
            </React.Fragment>
          ) : (
            <React.Fragment>
              {stepComponents[activeStep]}
            </React.Fragment>
          )}
        </Box>
      </div>
    </div>
  );
}
