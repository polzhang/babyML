'use client';

import React, { useState, useEffect } from 'react';
import FileDropZone from '@/components/FileDropZone';
import FileInfo from './FileInfo';
import Table from './Table';
import * as XLSX from 'xlsx';


interface PredictionsviewerProps {
  className?: string;
  handleNext?: () => void;
  onFileUpload?: (data: { headers: string[]; rows: { [key: string]: string }[] }, fileName: string) => void;
  initialData?: { headers: string[]; rows: { [key: string]: string }[] } | null;
  initialFileName?: string;
}

const Predictionsviewer: React.FC<PredictionsviewerProps> = ({ 
  className,
  onFileUpload,
  initialData,
  initialFileName
}) => {
  // Initialize state with initial values
  const [data, setData] = useState<typeof initialData>(null);
  const [fileName, setFileName] = useState('');
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: '', direction: 'asc' });
  const [loading, setLoading] = useState(false);

  // Update local state when props change
  useEffect(() => {
    console.log('Initial Data Changed:', initialData);
    console.log('Initial FileName Changed:', initialFileName);
    
    if (initialData) {
      setData(initialData);
    }
    if (initialFileName) {
      setFileName(initialFileName);
    }
  }, [initialData, initialFileName]);

  const parseExcel = async (file: File) => {
    setLoading(true);
    try {
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer);
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      const headers = (jsonData[0] as string[]).map((header) => header?.trim());
      const rows = (jsonData.slice(1) as Array<(string | undefined)[]>)
        .map((row) => {
          return headers.reduce((obj: { [key: string]: string }, header: string, index: number) => {
            obj[header] = row[index] !== undefined && row[index] !== null ? String(row[index]) : '';
            return obj;
          }, {});
        })
        .filter((row) => Object.values(row).some((value) => value !== ''));

      const newData = { headers, rows };
      setData(newData);
      setFileName(file.name);
      onFileUpload?.(newData, file.name);
      
      await uploadFileToBackend(file);
    } catch (error) {
      console.error('Error parsing Excel file:', error);
      alert('Error parsing Excel file. Please make sure it\'s a valid Excel file.');
    } finally {
      setLoading(false);
    }
  };

  const uploadFileToBackend = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('https://babyml.onrender.com/upload-and-predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleSort = (column: string) => {
    const direction = sortConfig.key === column && sortConfig.direction === 'asc' ? 'desc' : 'asc';
    setSortConfig({ key: column, direction });

    const currentData = data || initialData;
    if (currentData?.rows) {
      const sortedRows = [...currentData.rows].sort((a, b) => {
        if (a[column] < b[column]) return direction === 'asc' ? -1 : 1;
        if (a[column] > b[column]) return direction === 'asc' ? 1 : -1;
        return 0;
      });

      setData({
        headers: currentData.headers,
        rows: sortedRows,
      });
    }
  };

  const resetFile = () => {
    setData(null);
    setFileName('');
    onFileUpload?.(null, '');
  };

  // Determine which data to display
  const displayData = data || initialData;
  const displayFileName = fileName || initialFileName;

  return (
    <div className={`bg-white ${className}`}>
      {/* Fixed height container with padding */}
      <div className="h-[300px] flex flex-col">
        {!data && !fileName ? (
          /* Wrap FileDropZone in a div that takes remaining height */
          <div className="flex-1 items-center justify-center ">
            
            <FileDropZone onFileSelect={parseExcel} />

          </div>
        ) : (
          <div className="mt-4 h-full">
            <FileInfo fileName={displayFileName || ''} onReset={resetFile}/>
            <div className="flex-1 overflow-auto rounded-md ">
              <Table 
                headers={displayData.headers} 
                rows={displayData.rows} 
                sortConfig={sortConfig} 
                onSort={handleSort} 
              />
            </div>
            
            <p className="relative mt-2 text-sm text-gray-500"> {displayData.rows.length > 100 ? `Showing first 100 rows of ${displayData.rows.length} total rows` : `Showing all ${displayData.rows.length} rows`} </p>
            
      
          </div>
        )}
      </div>
    </div>
  );
};
export default Predictionsviewer;