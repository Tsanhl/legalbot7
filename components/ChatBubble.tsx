import React from 'react';
import { ChatMessage, MessageRole, Citation } from '../types';

interface ChatBubbleProps {
  message: ChatMessage;
  onCitationClick?: (citation: Citation) => void;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ message, onCitationClick }) => {
  const isUser = message.role === MessageRole.USER;

  // Parser for the custom JSON citation format: [[{...}]]
  const renderContent = (text: string) => {
    // 1. First, strip markdown bolding (**) and italics (*) that strictly should not be there per user request
    // We regex replace **text** and *text* markers.
    const cleanText = text.replace(/\*\*/g, '').replace(/\*/g, '');

    // 2. Split by the custom delimiter [[ ... ]]
    const parts = cleanText.split(/(\[\[\{.*?\}\]\])/g);

    return parts.map((part, index) => {
      if (part.startsWith('[[') && part.endsWith(']]')) {
        try {
          // Extract JSON string
          const jsonStr = part.slice(2, -2);
          const citationData = JSON.parse(jsonStr) as Citation;
          
          return (
            <button 
              key={index}
              onClick={() => onCitationClick?.(citationData)}
              className="inline-flex items-center mx-1 px-1.5 py-0.5 rounded text-xs font-medium bg-yellow-100 text-legal-900 border-b-2 border-yellow-400 hover:bg-yellow-200 hover:border-yellow-500 transition-colors cursor-pointer select-none"
              title={`Click to view source in ${citationData.doc}`}
            >
              <span className="mr-1">¶</span>
              {citationData.ref}
            </button>
          );
        } catch (e) {
          // Fallback if JSON parse fails
          console.error("Failed to parse citation:", part);
          return <span key={index} className="text-red-500 text-xs">[Citation Error]</span>;
        }
      }
      // Regular text
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`max-w-[85%] rounded-lg shadow-sm ${
        isUser 
          ? 'bg-legal-800 text-white rounded-br-none' 
          : 'bg-white border border-legal-200 text-legal-900 rounded-bl-none'
      }`}>
        <div className="px-5 py-4">
            <p className={`text-xs mb-2 font-bold uppercase tracking-wider ${isUser ? 'text-legal-300' : 'text-legal-500'}`}>
                {isUser ? 'You' : 'LexCitator AI'}
            </p>
            <div className={`prose ${isUser ? 'prose-invert' : 'prose-slate'} max-w-none font-serif leading-relaxed whitespace-pre-wrap text-sm`}>
                {renderContent(message.text)}
            </div>
            
            {/* Grounding Sources (Web) */}
            {!isUser && message.groundingUrls && message.groundingUrls.length > 0 && (
              <div className="mt-4 pt-3 border-t border-legal-200">
                <p className="text-xs font-semibold text-legal-500 mb-2">Web Sources Used:</p>
                <div className="flex flex-wrap gap-2">
                  {message.groundingUrls.map((url, i) => (
                    <a 
                      key={i} 
                      href={url.uri} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:underline bg-blue-50 px-2 py-1 rounded border border-blue-100"
                    >
                      {url.title}
                    </a>
                  ))}
                </div>
              </div>
            )}
        </div>
      </div>
    </div>
  );
};