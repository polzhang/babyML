
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import axios from 'axios';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ConsoleOutputProps {
  logs: string[];  // Explicitly define the type for logs as string array
}

const ConsoleOutput: React.FC<ConsoleOutputProps> = ({ logs }) => {
  const bottomRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getColor = (text:string) => {
    if (text.includes('=== Training Complete ===')) return 'text-green-500';
    if (text.includes('Error')) return 'text-red-500';
    if (text.includes('===')) return 'text-cyan-500';
    if (text.includes('Starting')) return 'text-yellow-500';
    return 'text-black-200';
  };

  return (
    <Card className="bg-gray-100">
      <CardContent className="p-4">
        <ScrollArea className="h-[400px]">
          <div className="ml-5 font-mono text-sm space-y-1">
            {logs.map((log, index) => (
              <div key={index} className={`${getColor(log)} whitespace-pre-wrap`}>
                {log}
              </div>
            ))}
            <div ref={bottomRef} /> {/* Invisible element at the bottom */}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};


const Component2: React.FC<{ 
  handleNext: () => void; 
  handleBack: () => void;
  refreshTrigger?: number
}> = ({ handleNext, handleBack, refreshTrigger }) => {
  // Basic Setup State
  const [selectedTarget, setSelectedTarget] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [trainTestSplit, setTrainTestSplit] = useState(80); // Default train-test split at 80%
  const [timeBudget, setTimeBudget] = useState('10'); 
  // Data Preprocessing State
  const [missingDataStrategy, setMissingDataStrategy] = useState('drop_rows');
  const [imputationMethod, setImputationMethod] = useState('mean');
  const [featureReduction, setFeatureReduction] = useState('none');
  const [constantValue, setConstantValue] = useState('');

  // Model Selection State
  const [enableEnsemble, setEnableEnsemble] = useState(false);
  const [validationMethod, setValidationMethod] = useState('auto'); // Default to Auto
  const [kFold, setKFold] = useState('5'); // Default k-fold value
  const [optimizationMetric, setOptimizationMetric] = useState('f1'); // Default metric for classification

  const [trainingComplete, setTrainingComplete] = useState(false);

  useEffect(() => {
    // Dynamically update optimizationMetric when problemType changes
    setOptimizationMetric(problemType === 'classification' ? 'f1' : 'r2');
  }, [problemType]);


  // Columns State for Target Variable Select
  const [columns, setColumns] = useState([]);

  const [logs, setLogs] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
    const handleBackButtonClick = () => { setIsTraining(false); };
  useEffect(() => {
    const eventSource = new EventSource('http://localhost:5000/stream-logs');
    
    eventSource.onmessage = (event) => {
      if (event.data !== 'heartbeat') {
        setLogs(prevLogs => [...prevLogs, event.data]); // Update logs state with new messages
      }
    };
  

  
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, []);

  useEffect(() => {
    // Fetch column names from the backend (Flask API)
    axios.get('http://localhost:5000/get-columns')
      .then(response => {
        console.log("Received columns from server:", response.data.columns);
        setColumns(response.data.columns);  // Set the column names in state
      })
      .catch(error => {
        console.error('Error fetching columns:', error);
      });
  }, [refreshTrigger]);
  
  

  // Handle k-fold input change
  const handleInputChange = (e) => {
    const value = e.target.value;
    if (value === '' || (Number(value) >= 0 && Number(value) <= 99999999999)) {
      setKFold(value); // Set the input value (empty or valid number between 2 and 10)
    }
  };

  // Handle Submit
  const handleSubmit = () => {
    setIsTraining(true);
    setLogs([]);
    const config = {
      target_variable: selectedTarget,
      problem_type: problemType,
      train_test_split: trainTestSplit / 100,
      time_budget: parseInt(timeBudget,10), 
      preprocessing: {
        missing_data: {
          strategy: missingDataStrategy,
          imputation_method: imputationMethod
        },
        feature_reduction: featureReduction,
      },
      models: {
        ensemble: enableEnsemble,
        selected: [] // Add model selection if needed
      },
      validation: {
        method: validationMethod,
        k_folds: parseInt(kFold),
        metric: optimizationMetric
      }
    };
    

    axios.post('http://localhost:5000/setup-training', config)
      .then(response => {
        console.log('Training results:', response.data);
        const fullLogs = response.data.logs || [];
          setLogs(prev => [...prev, ...fullLogs,
            '=== Training Complete ===',
            `Best Model: ${response.data.best_estimator}`,
            `Best Score: ${response.data.best_score}`,
            'Best Configuration:',
            JSON.stringify(response.data.best_config, null, 2)
        ]);
        // Handle success - update UI with training results
        setTrainingComplete(true); // Set training complete
      })
      .catch(error => {
        console.error('Error in training:', error);
        // Handle error - show error message to user
        setTrainingComplete(false);
      });
  };
  if (isTraining) {
    return (
      <div>
        <ConsoleOutput logs={logs} />
        <div className="mt-4 flex justify-between">
          <Button variant="outline" onClick={handleBackButtonClick}>Back</Button>
          {trainingComplete &&  (
            <Button onClick={handleNext}>
              Next
            </Button>
          )}
        </div>
      </div>
    );
  }
  

  return (
    
    <div className="grid place-items-center">
      <Card className="w-full max-w-6xl">
        <CardHeader>
          <CardTitle className='text-2xl'>AutoML Setup</CardTitle>
          <CardDescription>Configure your automated machine learning pipeline</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="basic" className="space-y-4">
            <TabsList>
              <TabsTrigger value="basic">Basic Setup</TabsTrigger>
              <TabsTrigger value="preprocessing">
                <TooltipProvider>
                  <div>
                    {/* Tooltip wrapping the Preprocessing tab */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        {/* Your tab trigger, e.g., "Preprocessing" */}
                        <button style={{ border: "none", background: "none", cursor: "pointer", fontSize: "14px" }}>
                          Preprocessing

                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="top" align="center" className="tooltip-content">
                        <p>Standardisation and normalisation are automatically applied to the training data</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                </TooltipProvider>
              </TabsTrigger>
              <TabsTrigger value="modeling"><TooltipProvider>
                  <div>
                    {/* Tooltip wrapping the Preprocessing tab */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        {/* Your tab trigger, e.g., "Preprocessing" */}
                        <button style={{ border: "none", background: "none", cursor: "pointer", fontSize: "14px" }}>
                          Data Split & Optimisation

                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="top" align="center" className="tooltip-content">
                        <p>FLAML will train and optimise multiple models based on your selections below</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                </TooltipProvider></TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              <div>
                <Label>Target Variable *</Label>
                <Select onValueChange={setSelectedTarget} value={selectedTarget}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select target variable" />
                  </SelectTrigger>
                  <SelectContent>
                    {columns.map((column, index) => (
                      <SelectItem key={index} value={column}>
                        {column}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Problem Type</Label>
                <Select onValueChange={setProblemType} value={problemType}>
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
            <Label>Time Budget (seconds)</Label>
            <Input
              type="number"
              placeholder="Enter value"
              value={timeBudget}
              onChange={(e) => setTimeBudget(e.target.value)}
              min="10"
            />
          </div>
            </TabsContent>

            <TabsContent value="preprocessing" className="space-y-4">
            <div>
    <Label>Missing Data Strategy</Label>
    <Select
      onValueChange={(value) => {
        setMissingDataStrategy(value);
        if (value !== "imputation") {
          setImputationMethod(""); // Clear imputation method if strategy is not "imputation"
          setConstantValue(""); // Clear constant value if strategy is not "imputation"
        }
      }}
      value={missingDataStrategy}
    >
      <SelectTrigger>
        <SelectValue placeholder="Select strategy" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="imputation">Imputation</SelectItem>
        <SelectItem value="drop_rows">Drop Rows</SelectItem>
      </SelectContent>
    </Select>
  </div>

  {missingDataStrategy === "imputation" && (
    <div>
      <Label>Imputation Method</Label>
      <Select
        onValueChange={(value) => {
          setImputationMethod(value);
          if (value !== "constant") {
            setConstantValue(""); // Clear constant value if not using "constant" method
          }
        }}
        value={imputationMethod}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select method" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="mean">Mean</SelectItem>
          <SelectItem value="median">Median</SelectItem>
          <SelectItem value="most_frequent">Mode</SelectItem>
          <SelectItem value="constant">Constant</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )}

  {missingDataStrategy === "imputation" && imputationMethod === "constant" && (
    <div>
      <Label>Constant Value</Label>
      <Input
        type="text"
        placeholder="Enter constant value"
        value={constantValue}
        onChange={(e) => setConstantValue(e.target.value)}
        className="input-class"
      />
    </div>
  )}
              

              <div>
                <Label>Feature Reduction</Label>
                <Select onValueChange={setFeatureReduction} value={featureReduction}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="pca">PCA</SelectItem>
                  
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>

            <TabsContent value="modeling" className="space-y-4">

              
              <div>
                <Label>Validation Method</Label>
                <Select onValueChange={setValidationMethod} value={validationMethod}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select Method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Default)</SelectItem>
                    <SelectItem value="cv">N-Fold Cross Validation</SelectItem>
                    <SelectItem value="holdout">Train-Test Split</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {validationMethod === 'holdout' && (
                <div>
                  <Label>Train-Test Split ({trainTestSplit}%)</Label>
                  <Slider
                    value={[trainTestSplit]}
                    onValueChange={([value]) => setTrainTestSplit(value)}
                    min={50}
                    max={90}
                    step={1}
                    className="w-full"
                  />
                </div>
              )}

              {validationMethod === 'cv' && (
                <div>
                  <Label>Number of Folds</Label>
                  <Input
                    type="text" // Change to "text" to allow empty input
                    value={kFold}
                    onChange={handleInputChange}
                    className="w-full"
                    placeholder="Enter no. of folds"
                    min={2}
                    max={10}
                  />
                </div>
              )}

<div>
  <Label>Optimization Metric</Label>
  <Select
    onValueChange={setOptimizationMetric}
    value={optimizationMetric}
  >
    <SelectTrigger>
      <SelectValue placeholder="Select metric" />
    </SelectTrigger>
    <SelectContent>
      {problemType === "regression" ? (
        <>
          <SelectItem value="r2">R2 (Default)</SelectItem>
          <SelectItem value="mae">Mean Absolute Error (MAE)</SelectItem>
          <SelectItem value="rmse">Root Mean Squared Error (RMSE)</SelectItem>
          <SelectItem value="mse">Mean Squared Error (MSE)</SelectItem>
        </>
      ) : problemType === "classification" ? (
        <>
          <SelectItem value="f1">F1 Score (Default)</SelectItem>
          <SelectItem value="log_loss">Log Loss</SelectItem>
          <SelectItem value="precision">Precision</SelectItem>
          <SelectItem value="accuracy">Accuracy</SelectItem>
          <SelectItem value="roc_auc">ROC AUC</SelectItem>
        </>
      ) : (
        <SelectItem value="" disabled>
          Please select a problem type
        </SelectItem>
      )}
    </SelectContent>
  </Select>
</div>


              <div className="flex items-center space-x-2">
                <Switch
                  id="ensemble"
                  checked={enableEnsemble}
                  onCheckedChange={setEnableEnsemble}
                />
                <Label htmlFor="ensemble">Enable Ensemble Models</Label>
              </div>
            </TabsContent>
          </Tabs>

          <div className="flex mt-5 justify-between space-x-2">
            <Button variant="outline" onClick={handleBack}>Back</Button>
            <Button onClick={handleSubmit}>Finalise and Train</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Component2;
