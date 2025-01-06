"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { Upload } from 'lucide-react';

interface FileDropZone2Props {
  onFileSelect: (file: File) => void;
}

const FileDropZone2 = ({ onFileSelect }: FileDropZone2Props) => {
  const [isDragging, setIsDragging] = useState(false);
  const dropZoneRef = useRef(null);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
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
    <div className="flex justify-center items-center p-0.5">
      <label 
        className={`
          relative
          w-full
          h-80
          flex
          flex-col
          items-center
          justify-center
          p-6
          border-2
          border-dashed
          rounded-lg
          cursor-pointer
          transition-colors
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        ref={dropZoneRef}
      >
        <input
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept=".csv,.xlsx,.xls"
        />
        <Upload className="w-12 h-12 text-gray-400 mb-4" />
        <div className="text-center">
          <p className="text-lg font-medium text-gray-700 mb-2">
            Drag and drop your file here
          </p>
          <p className="text-sm text-gray-500 mb-2">
            or click to browse
          </p>
          <p className="text-xs text-gray-400">
            Supports CSV, Excel (.xlsx, .xls)
          </p>
        </div>
      </label>
    </div>
  );
};

export default FileDropZone2;