import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// Combined component with all features (including imputation & preprocessing)
const Component2 = ({ handleNext, handleBack }: { handleNext: () => void; handleBack: () => void }) => {
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [problemType, setProblemType] = useState<'classification' | 'regression'>('classification');
  const [randomState, setRandomState] = useState<string>('42');
  const [selectedModels, setSelectedModels] = useState<string[]>([]); // multiple model selection
  const [imputationMethod, setImputationMethod] = useState<'mean' | 'median' | 'mode'>('mean');
  const [scalingMethod, setScalingMethod] = useState<'none' | 'standardization' | 'normalization'>('none');

  const handleTargetChange = (value: string) => setSelectedTarget(value);
  const handleProblemTypeChange = (value: 'classification' | 'regression') => setProblemType(value);
  const handleRandomStateChange = (e: React.ChangeEvent<HTMLInputElement>) => setRandomState(e.target.value);
  const handleModelSelection = (value: string[]) => setSelectedModels(value);
  const handleImputationChange = (value: 'mean' | 'median' | 'mode') => setImputationMethod(value);
  const handleScalingChange = (value: 'none' | 'standardization' | 'normalization') => setScalingMethod(value);

  return (
    <Card>
      <CardHeader>
        <CardTitle>ML Setup</CardTitle>
        <CardDescription>Set up your machine learning task</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex">
          <div className="space-y-4">
            <div>
              <Label htmlFor="targetVariable">Target Variable</Label>
              <Select onValueChange={handleTargetChange} value={selectedTarget}>
                <SelectTrigger>
                  <SelectValue placeholder="Select target variable" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="target1">Target 1</SelectItem>
                  <SelectItem value="target2">Target 2</SelectItem>
                  <SelectItem value="target3">Target 3</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="problemType">Problem Type</Label>
              <Select onValueChange={handleProblemTypeChange} value={problemType}>
                <SelectTrigger>
                  <SelectValue placeholder="Select problem type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="randomState">Random State</Label>
              <Input
                type="text"
                id="randomState"
                value={randomState}
                onChange={handleRandomStateChange}
                placeholder="Enter seed value"
              />
            </div>
          </div>

          <div className="space-y-4 ml-10">
            <div>
              <Label htmlFor="imputationMethod">Imputation Method</Label>
              <Select onValueChange={handleImputationChange} value={imputationMethod}>
                <SelectTrigger>
                  <SelectValue placeholder="Select imputation method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mean">Mean</SelectItem>
                  <SelectItem value="median">Median</SelectItem>
                  <SelectItem value="mode">Mode</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="scalingMethod">Scaling Method</Label>
              <Select onValueChange={handleScalingChange} value={scalingMethod}>
                <SelectTrigger>
                  <SelectValue placeholder="Select scaling method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="standardization">Standardization</SelectItem>
                  <SelectItem value="normalization">Normalization</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="models">Model Selection</Label>
              <Select multiple onValueChange={handleModelSelection} value={selectedModels}>
                <SelectTrigger>
                  <SelectValue placeholder="Select models" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="modelA">Model A</SelectItem>
                  <SelectItem value="modelB">Model B</SelectItem>
                  <SelectItem value="modelC">Model C</SelectItem>
                  <SelectItem value="modelD">Model D</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <div className="flex justify-between mt-4">
          <Button variant="secondary" onClick={handleBack}>
            Back
          </Button>
          <Button onClick={handleNext}>
            Next
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default Component2;
