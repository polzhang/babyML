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

// Define your steps as you have
const steps = ['Upload Training Data', 'Select Model Parameters', 'Predict/Export'];
interface Step1Props {
  onNext: () => void;  // Add prop for handling next step
}

// Example custom components for each step
const Step1Component: React.FC<Step1Props> = ({ onNext }) => (
  <div>
    <FileUploadViewer
      className="relative top-[20px] p-4 rounded-lg shadow-lg w"
      handleNext={onNext}
    />
  </div>
);
const Step2Component = () => <div>Step 2 Content</div>;
const Step3Component = () => <div>Step 3 Content</div>;

export default function PageContainer() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [skipped, setSkipped] = React.useState(new Set<number>());

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
  };

  // Create an array of the step components you want to render
  const stepComponents = [
    <Step1Component key="step1" onNext={() => setActiveStep((prevStep) => prevStep + 1)} />,
    <Step2Component />, // Content for Step 2
    <Step3Component />, // Content for Step 3
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
                  {/* Apply sx to the StepLabel to adjust font size while keeping the default styling */}
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
                All steps completed - you&apos;re finished
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