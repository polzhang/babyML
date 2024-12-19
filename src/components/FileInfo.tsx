// components/FileInfo.tsx
'use client';
import React from 'react';
import { X, FileText } from 'lucide-react';

const FileInfo = ({ fileName, onReset }: { fileName: string; onReset: () => void }) => (
  <div className="relative bottom-[20px] flex items-center justify-between">
    <div className="flex items-center gap-2">
      <FileText className="w-5 h-5 text-blue-500" />
      <span className="font-medium">{fileName}</span>
    </div>
    <button onClick={onReset} className="p-1 text-gray-500 hover:text-gray-700 transition-colors">
      <X className="w-5 h-5" />
    </button>
  </div>
);

export default FileInfo;
