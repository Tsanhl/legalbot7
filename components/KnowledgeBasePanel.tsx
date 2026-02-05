import React, { useEffect, useState } from 'react';
import { loadLawResourceIndex, LawResourceEntry, LawResourceIndex, getResourcesByCategory } from '../services/knowledgeBaseService';

interface KnowledgeBasePanelProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectResources: (resources: LawResourceEntry[]) => void;
}

export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({ 
  isOpen, 
  onClose, 
  onSelectResources 
}) => {
  const [index, setIndex] = useState<LawResourceIndex | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedResources, setSelectedResources] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadLawResourceIndex().then(idx => {
      setIndex(idx);
      setLoading(false);
    });
  }, []);

  const filteredResources = React.useMemo(() => {
    if (!index) return [];
    
    let resources = index.resources;
    
    if (selectedCategory) {
      resources = resources.filter(r => r.category === selectedCategory);
    }
    
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      resources = resources.filter(r => 
        r.name.toLowerCase().includes(query) ||
        r.category.toLowerCase().includes(query) ||
        r.subcategory.toLowerCase().includes(query)
      );
    }
    
    return resources.slice(0, 100); // Limit for performance
  }, [index, selectedCategory, searchQuery]);

  const toggleResource = (id: string) => {
    setSelectedResources(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleApply = () => {
    if (!index) return;
    const resources = index.resources.filter(r => selectedResources.has(r.id));
    onSelectResources(resources);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-legal-200 flex items-center justify-between bg-legal-50">
          <div>
            <h2 className="text-lg font-bold text-legal-900">📚 Law Resources Knowledge Base</h2>
            <p className="text-sm text-legal-500">
              {index ? `${index.totalFiles} documents available` : 'Loading...'}
            </p>
          </div>
          <button 
            onClick={onClose}
            className="text-legal-400 hover:text-legal-700 p-2"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Filters */}
        <div className="p-4 border-b border-legal-100 flex gap-4 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search resources..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-2 border border-legal-300 rounded-md text-sm focus:ring-2 focus:ring-yellow-500 focus:border-yellow-500"
            />
          </div>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-legal-300 rounded-md text-sm focus:ring-2 focus:ring-yellow-500"
          >
            <option value="">All Categories</option>
            {index?.categories.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>

        {/* Resource List */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center py-8 text-legal-500">Loading knowledge base...</div>
          ) : filteredResources.length === 0 ? (
            <div className="text-center py-8 text-legal-500">No resources found</div>
          ) : (
            <div className="space-y-2">
              {filteredResources.map(resource => (
                <div 
                  key={resource.id}
                  onClick={() => toggleResource(resource.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedResources.has(resource.id) 
                      ? 'border-yellow-500 bg-yellow-50' 
                      : 'border-legal-200 hover:border-legal-400'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 mt-0.5 ${
                      selectedResources.has(resource.id) 
                        ? 'bg-yellow-500 border-yellow-500' 
                        : 'border-legal-300'
                    }`}>
                      {selectedResources.has(resource.id) && (
                        <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-legal-900 truncate">{resource.name}</p>
                      <p className="text-xs text-legal-500 mt-0.5">
                        {resource.category}
                        {resource.subcategory && ` › ${resource.subcategory}`}
                      </p>
                    </div>
                    <span className="text-xs text-legal-400 flex-shrink-0">
                      {(resource.size / 1024).toFixed(0)} KB
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-legal-200 flex items-center justify-between bg-legal-50">
          <span className="text-sm text-legal-600">
            {selectedResources.size} resources selected
          </span>
          <div className="flex gap-3">
            <button
              onClick={() => setSelectedResources(new Set())}
              className="px-4 py-2 text-sm text-legal-600 hover:text-legal-800"
            >
              Clear
            </button>
            <button
              onClick={handleApply}
              disabled={selectedResources.size === 0}
              className="px-4 py-2 text-sm bg-legal-900 text-white rounded-md hover:bg-legal-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add to Context
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

