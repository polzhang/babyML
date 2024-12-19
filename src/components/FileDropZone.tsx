"use client";
import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { Upload } from 'lucide-react';

// Explicitly type the `onFileSelect` prop to accept a function that takes a `File` and returns `void`
interface FileDropZoneProps {
  onFileSelect: (file: File) => void;
}

const FileDropZone = ({ onFileSelect }: FileDropZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const dropZoneRef = useRef<HTMLDivElement | null>(null);

  // Explicitly type the `e` parameter as DragEvent for drag events
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  // Explicitly type the `e` parameter as DragEvent for drag events
  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  // Explicitly type the `e` parameter as DragEvent for drop events
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  };

  // Explicitly type the `e` parameter as ChangeEvent for file input events
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <div
      ref={dropZoneRef}
      className={`relative h-[60vh] border-2 border-dashed rounded-lg ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
      }`}
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
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <Upload className={`w-12 h-12 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
        <div className="text-center">
          <p className="text-sm font-medium text-gray-700">Drag and drop your file here</p>
          <p className="mt-1 text-xs text-gray-500">or click to browse</p>
          <p className="mt-2 text-xs text-gray-400">Supports CSV, Excel (.xlsx, .xls)</p>
        </div>
      </div>
    </div>
  );
};

export default FileDropZone;
