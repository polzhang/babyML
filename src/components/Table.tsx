'use client';
import React from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

/* eslint-disable  @typescript-eslint/no-explicit-any */
const Table = ({
  headers,
  rows,
  sortConfig,
  onSort,
}: {
  headers: string[];
  rows: any[];
  sortConfig: { key: string; direction: 'asc' | 'desc' };
  onSort: (column: string) => void;
}) => (
  <div
    className="relative bottom border rounded-lg overflow-y-auto"
    style={{ maxHeight: '16rem' }}
  >
    <table className="w-full border-collapse">
      <thead className="bg-white-50">
        <tr>
          {headers.map((header, index) => (
            <th
              key={index}
              className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b sticky bg-gray-50 cursor-pointer"
              onClick={() => onSort(header)}
            >
              <div className="flex items-center justify-between">
                <span>{header}</span>
                {sortConfig.key === header && (
                  <span
                    className={`ml-2 ${
                      sortConfig.direction === 'asc'
                        ? 'text-blue-500'
                        : 'text-red-500'
                    }`}
                  >
                    {sortConfig.direction === 'asc' ? (
                      <ChevronUp className="w-4 h-4 inline" />
                    ) : (
                      <ChevronDown className="w-4 h-4 inline" />
                    )}
                  </span>
                )}
              </div>
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="bg-white divide-y divide-gray-200">
        {rows.slice(0, 100).map((row, rowIndex) => (
          <tr key={rowIndex} className="hover:bg-gray-50">
            {headers.map((header, colIndex) => (
              <td
                key={colIndex}
                className="px-4 py-2 text-sm text-gray-900 whitespace-nowrap"
              >
                {/* Explicitly check for undefined/null */}
                {row[header] !== undefined && row[header] !== null
                  ? row[header]
                  : ''}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default Table;
