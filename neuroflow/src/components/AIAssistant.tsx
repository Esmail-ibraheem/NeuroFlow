import React, { useState } from 'react';
import { MessageSquare, Send, Sparkles, X, Loader2 } from 'lucide-react';
import { generatePipeline, chatWithAI } from '../services/geminiService';

interface AIAssistantProps {
  onGenerateGraph: (nodes: any[], edges: any[]) => void;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ onGenerateGraph }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant', text: string }[]>([
    { role: 'assistant', text: "Hi! I'm NeuroBot. I can build your AI training pipeline automatically. Try saying 'Fine-tune Llama 3 on a medical dataset', or just chat with me!" }
  ]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;

    const userMsg = prompt;
    setPrompt("");
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setLoading(true);

    try {
      const lowerMsg = userMsg.toLowerCase();
      // Heuristic to detect building intent
      const isBuildRequest = lowerMsg.includes('build') ||
        lowerMsg.includes('generate') ||
        lowerMsg.includes('create') ||
        lowerMsg.includes('design') ||
        lowerMsg.includes('setup') ||
        lowerMsg.includes('pipeline') ||
        lowerMsg.includes('workflow') ||
        lowerMsg.includes('architecture') ||
        lowerMsg.includes('fine-tune');

      if (isBuildRequest) {
        setMessages(prev => [...prev, { role: 'assistant', text: "Architecting your pipeline... Please wait." }]);
        const { nodes, edges, explanation } = await generatePipeline(userMsg);
        onGenerateGraph(nodes, edges);
        setMessages(prev => [...prev, { role: 'assistant', text: explanation }]);
      } else {
        const response = await chatWithAI(userMsg);
        setMessages(prev => [...prev, { role: 'assistant', text: response }]);
      }
    } catch (error: any) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Sorry, I couldn't process that: ${error.message || 'Check your API Key or try again.'}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Trigger Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-blue-600 hover:bg-blue-500 text-white p-4 rounded-full shadow-2xl z-50 transition-all hover:scale-110 flex items-center justify-center gap-2 group"
        >
          <Sparkles size={24} className="group-hover:rotate-12 transition-transform" />
          <span className="font-semibold pr-1">AI Build</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[500px] bg-[#18181b] border border-[#27272a] rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300">
          {/* Header */}
          <div className="p-4 bg-gradient-to-r from-blue-900/50 to-purple-900/50 border-b border-[#27272a] flex justify-between items-center">
            <div className="flex items-center gap-2 text-white font-bold">
              <Sparkles size={18} className="text-blue-400" />
              NeuroBot Architect
            </div>
            <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[85%] rounded-lg p-3 text-sm leading-relaxed ${msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-[#27272a] text-gray-200 rounded-bl-none border border-[#3f3f46]'
                    }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-[#27272a] text-gray-200 rounded-lg p-3 rounded-bl-none border border-[#3f3f46] flex items-center gap-2">
                  <Loader2 size={14} className="animate-spin text-blue-400" />
                  <span className="text-xs text-gray-400">Designing architecture...</span>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="p-3 bg-[#09090b] border-t border-[#27272a] flex gap-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your training pipeline..."
              className="flex-1 bg-[#18181b] text-white border border-[#27272a] rounded-md px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={loading || !prompt.trim()}
              className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default AIAssistant;