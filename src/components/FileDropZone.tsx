"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { Upload } from 'lucide-react';

interface FileDropZone2Props {
  onFileSelect: (file: File) => void;
}

const FileDropZone = ({ onFileSelect }: FileDropZone2Props) => {
  const [isDragging, setIsDragging] = useState(false);
  const dropZoneRef = useRef<HTMLDivElement | null>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <div className="w-full h-full flex items-center justify-center p-2">
      <div
        ref={dropZoneRef}
        className={`
          w-full 
          h-full 
          border-2 
          border-dashed 
          rounded-lg 
          transition-colors
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          onChange={handleFileChange}
          accept=".csv,.xlsx,.xls"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center justify-center h-full gap-4 px-4">
          <Upload className={`w-12 h-12 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
          <div className="text-center">
            <p className="text-xl font-medium text-gray-700">Drag and drop your file here</p>
            <p className="mt-1 text-l text-gray-500">or click to browse</p>
            <p className="mt-2 text-l text-gray-400">Supports CSV, Excel (.xlsx, .xls)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileDropZone;