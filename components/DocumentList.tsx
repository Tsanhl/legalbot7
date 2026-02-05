import React from 'react';
import { UploadedDocument } from '../types';

interface DocumentListProps {
  documents: UploadedDocument[];
  onRemove: (id: string) => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({ documents, onRemove }) => {
  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-legal-400">
        <p className="text-sm">No specific documents added.</p>
        <p className="text-[10px] mt-1">AI will use general legal knowledge and Google Search grounding.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {documents.map((doc) => (
        <div 
          key={doc.id} 
          className="flex items-center justify-between p-3 bg-white rounded-lg border border-legal-200 shadow-sm group hover:border-legal-400 transition-colors"
        >
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className={`flex-shrink-0 p-2 rounded-md ${doc.type === 'link' ? 'bg-blue-100' : 'bg-red-100'}`}>
              {doc.type === 'link' ? (
                <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              )}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium text-legal-900 truncate" title={doc.name}>
                {doc.name}
              </p>
              <p className="text-xs text-legal-500">
                {doc.type === 'link' ? 'Web Resource' : `${(doc.size / 1024).toFixed(1)} KB`}
              </p>
            </div>
          </div>
          <button 
            onClick={() => onRemove(doc.id)}
            className="text-legal-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity p-1"
            title="Remove"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
};