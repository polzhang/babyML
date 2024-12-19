/* eslint-disable */
'use client';
import React, { useState } from 'react';
import FileDropZone from './FileDropZone';
import FileInfo from './FileInfo';
import Table from './Table';
import * as XLSX from 'xlsx';
import {Button} from '@/components/ui/button';

type RowData = { [key: string]: string };
type Data = {
  headers: string[];
  rows: RowData[];
};

interface FileUploadViewerProps {
  className?: string;
  handleNext?: () => void;  // New prop for stepper control
}

const FileUploadViewer: React.FC<FileUploadViewerProps> = ({ 
  className,
  handleNext 
}) => {
  const [data, setData] = useState<Data | null>(null);
  const [fileName, setFileName] = useState('');
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: '', direction: 'asc' });
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const parseExcel = async (file: File) => {
    setLoading(true); // Show loading spinner
    try {
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer);
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      const headers = (jsonData[0] as string[]).map((header) => header?.trim());
      const rows = (jsonData.slice(1) as Array<(string | undefined)[]>).map((row) => {
        return headers.reduce((obj: RowData, header: string, index: number) => {
          obj[header] = row[index] !== undefined && row[index] !== null ? row[index] : '';
          return obj;
        }, {}); 
      }).filter((row) => Object.values(row).some((value) => value !== ''));

      setData({ headers, rows });
      setFileName(file.name);
      setUploadedFile(file);  // Store the uploaded file in the state
    } catch (error) {
      console.error('Error parsing Excel file:', error);
      alert('Error parsing Excel file. Please make sure it\'s a valid Excel file.');
    } finally {
      setLoading(false); // Hide loading spinner
    }
  };
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const parseCsv = async (file: File) => {
    setLoading(true); // Show loading spinner
    try {
      const text = await file.text();
      const rows = text.split('\n').map((row) => row.trim());
      const headers = rows[0].split(',').map((header) => header.trim());
      const tableData = rows.slice(1).map((row) => {
        const values = row.split(',').map((value) => value.trim());
        return headers.reduce((obj: RowData, header: string, index: number) => {
          obj[header] = values[index] !== undefined && values[index] !== null ? values[index] : '';
          return obj;
        }, {}); 
      });

      setData({ headers, rows: tableData });
      setFileName(file.name);
      setUploadedFile(file);  // Store the uploaded file in the state
    } catch (error) {
      console.error('Error parsing CSV file:', error);
      alert('Error parsing file. Please make sure it\'s a valid CSV file.');
    } finally {
      setLoading(false); // Hide loading spinner
    }
  };

  const handleFile = async (file: File) => {
    const fileType = file.name.split('.').pop()?.toLowerCase();
    if (['xlsx', 'xls', 'csv'].includes(fileType!)) {
      await parseExcel(file); // This is your existing logic to parse the file
      // Send the file to the backend
      await uploadFileToBackend(file);
    } else {
      alert('Please upload a valid Excel (.xlsx, .xls) or CSV file');
    }
  };
  
  const uploadFileToBackend = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
  
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
  
      // Ensure the response is valid and check content type
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      // Check if the response is JSON before attempting to parse it
      const contentType = response.headers.get('Content-Type');
      if (contentType && contentType.includes('application/json')) {
        await response.json(); // Parse the response (if needed for future use)
      }
  
      // No further action, no message displayed
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };
  

  const handleSort = (column: string) => {
    const direction = sortConfig.key === column && sortConfig.direction === 'asc' ? 'desc' : 'asc';
    setSortConfig({ key: column, direction });

    // Check if data and rows are available
    if (data && data.rows) {
      const sortedRows = [...data.rows].sort((a, b) => {
        if (a[column] < b[column]) return direction === 'asc' ? -1 : 1;
        if (a[column] > b[column]) return direction === 'asc' ? 1 : -1;
        return 0;
      });

      setData({
        ...data, // Keep existing data
        rows: sortedRows,
      });
    }
  };

  const resetFile = () => {
    setData(null);
    setFileName('');
    setUploadedFile(null); // Reset the uploaded file when the file is reset
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border ${className}`}>
      <div className="p-6">
        {!data ? (
          <FileDropZone onFileSelect={handleFile} />
        ) : (
          <div>
            <FileInfo fileName={fileName} onReset={resetFile} />
            <div className="overflow-hidden border rounded-md relative">
              {loading && (
                <div className="absolute inset-0 flex justify-center items-center">
                  <div className="w-12 h-12 border-4 border-t-4 border-blue-500 rounded-full animate-spin"></div>
                </div>
              )}
              <Table headers={data.headers} rows={data.rows} sortConfig={sortConfig} onSort={handleSort} />
            </div>
            {data.rows.length > 100 && (
              <p className="mt-3 text-sm text-gray-500">
                Showing first 100 rows of {data.rows.length} total rows
              </p>
            )}
            {/* Next button appears only when data is loaded successfully */}
            <div className="flex justify-end">
              <Button
                onClick={handleNext}
                variant="default"
                color="primary"
                size="lg"
              >
                NEXT
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploadViewer;