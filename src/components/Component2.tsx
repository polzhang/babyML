'use client';
/* eslint-disable no-console, no-unused-vars */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface ModelParametersProps {
  className?: string;
  handleNext?: () => void;
  handleBack?: () => void;
}

const ModelParameters: React.FC<ModelParametersProps> = ({ className, handleNext, handleBack }) => {
  const [modelType, setModelType] = useState('neural-network');
  const [epochs, setEpochs] = useState([100]);
  const [learningRate, setLearningRate] = useState([0.001]);
  const [useValidation, setUseValidation] = useState(true);

  const handleSubmit = () => {
    if (handleNext) handleNext();
  };

  return (
    <Card className="mt-5 rounded-lg shadow-sm">
      <CardContent className="p-6">
        <h2 className="text-2xl font-semibold text-center mb-6">Training Parameters</h2>
        
        <div className="space-y-6">
          <div className="space-y-2">
            <Label>Model Type</Label>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger>
                <SelectValue placeholder="Select model type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="neural-network">Neural Network</SelectItem>
                <SelectItem value="random-forest">Random Forest</SelectItem>
                <SelectItem value="svm">SVM</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Number of Epochs: {epochs}</Label>
            <Slider
              value={epochs}
              onValueChange={setEpochs}
              min={10}
              max={1000}
              step={10}
            />
          </div>

          <div className="space-y-2">
            <Label>Learning Rate: {learningRate}</Label>
            <Slider
              value={learningRate}
              onValueChange={setLearningRate}
              min={0.0001}
              max={0.1}
              step={0.0001}
            />
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              checked={useValidation}
              onCheckedChange={setUseValidation}
            />
            <Label>Use Validation Split</Label>
          </div>

          <div className="flex justify-between mt-6">
            <Button
              onClick={handleBack}
              variant="outline"
              size="lg"
            >
              Back
            </Button>
            <Button
              onClick={handleSubmit}
              size="lg"
            >
              Next
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelParameters;