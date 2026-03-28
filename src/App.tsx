import React, { useState, useEffect } from 'react';
import { Settings, Play, CheckCircle, AlertCircle, Download, ExternalLink, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import axios from 'axios';

// Types
interface FileRecord {
  id: string;
  originalFilename: string;
  secureUrl: string;
  section: string;
  status: 'Pending' | 'AlreadyCopy';
  wordCount: number;
  sizeBytes: number;
  uploadTimestamp: number;
  completedTimestamp?: number;
  generatedUrl?: string; // Added locally after generation
}

interface CloudinaryConfig {
  cloudName: string;
  apiKey: string;
  apiSecret: string;
  geminiApiKeys: string;
  model: string;
}

const DEFAULT_CLOUDINARY_CONFIG: CloudinaryConfig = {
  cloudName: 'dky6urpy2',
  apiKey: '164368975898265',
  apiSecret: '4toK14OkJY-ePc-NFE_YsB2vrFc',
  geminiApiKeys: '',
  model: 'gemini-3-flash-preview',
};

const AVAILABLE_MODELS = [
  { value: 'gemini-3-flash-preview', label: 'Gemini 3 Flash Preview (Recommended)' },
  { value: 'gemini-3-pro-preview', label: 'Gemini 3 Pro Preview' },
  { value: 'gemini-2.0-flash-exp', label: 'Gemini 2.0 Flash Experimental' },
  { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash' },
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
];

export default function App() {
  const [files, setFiles] = useState<FileRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cloudinaryConfig, setCloudinaryConfig] = useState<CloudinaryConfig>(() => {
    const saved = localStorage.getItem('cloudinaryConfig');
    // Merge with default to ensure new fields like 'model' exist
    return saved ? { ...DEFAULT_CLOUDINARY_CONFIG, ...JSON.parse(saved) } : DEFAULT_CLOUDINARY_CONFIG;
  });
  const [showSettings, setShowSettings] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const stopGenerationRef = React.useRef(false);
  const [currentFileId, setCurrentFileId] = useState<string | null>(null);
  const [generatedFiles, setGeneratedFiles] = useState<FileRecord[]>([]);
  const [generatedUrls, setGeneratedUrls] = useState<Record<string, string>>(() => {
    const saved = localStorage.getItem('generatedUrls');
    return saved ? JSON.parse(saved) : {};
  });
  const [activeTab, setActiveTab] = useState<'pending' | 'generated'>('pending');
  const [externalApiStatus, setExternalApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [generationStatus, setGenerationStatus] = useState<Record<string, { step: string; progress: number; message: string }>>({});
  const [queueStats, setQueueStats] = useState<{ activeJobs: number; completedJobs: number; failedJobs: number; totalJobs: number } | null>(null);
  
  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    checkExternalApi();
    fetchFiles();
    fetchQueueStats();
    
    // Poll stats every 5 seconds
    const interval = setInterval(fetchQueueStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Reset pagination when tab changes
  useEffect(() => {
    setCurrentPage(1);
  }, [activeTab]);

  const fetchQueueStats = async () => {
    try {
      const response = await axios.get('/api/stats');
      setQueueStats(response.data);
    } catch (err) {
      console.error('Failed to fetch queue stats', err);
    }
  };

  const checkExternalApi = async () => {
    try {
      const response = await axios.get('/api/proxy/health');
      if (response.data.message === 'CloudText Backend is Running') {
        setExternalApiStatus('online');
      } else {
        setExternalApiStatus('offline');
      }
    } catch (err) {
      setExternalApiStatus('offline');
    }
  };

  useEffect(() => {
    localStorage.setItem('cloudinaryConfig', JSON.stringify(cloudinaryConfig));
  }, [cloudinaryConfig]);

  useEffect(() => {
    localStorage.setItem('generatedUrls', JSON.stringify(generatedUrls));
  }, [generatedUrls]);

  const fetchFiles = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/proxy/files');
      const apiFiles = response.data;
      
      // Merge with local generated URLs
      const mergedFiles = apiFiles.map((f: FileRecord) => ({
        ...f,
        generatedUrl: generatedUrls[f.id] || f.generatedUrl
      }));
      
      setFiles(mergedFiles);
      setGeneratedFiles(mergedFiles.filter((f: FileRecord) => f.status === 'AlreadyCopy'));
    } catch (err: any) {
      setError(err.message || 'Failed to fetch files');
    } finally {
      setLoading(false);
    }
  };

  const stopGeneration = () => {
    stopGenerationRef.current = true;
    setIsGenerating(false);
  };

  const startGeneration = async () => {
    if (isGenerating) {
      stopGeneration();
      return;
    }

    setIsGenerating(true);
    stopGenerationRef.current = false;
    const pendingFiles = files.filter(f => f.status === 'Pending');
    
    for (const file of pendingFiles) {
      if (stopGenerationRef.current) break;
      setCurrentFileId(file.id);
      
      // Setup SSE for progress
      const eventSource = new EventSource(`/api/progress/${file.id}`);
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setGenerationStatus(prev => ({
            ...prev,
            [file.id]: data
          }));
        } catch (e) {
          console.error('Failed to parse SSE message', e);
        }
      };

      try {
        const response = await axios.post('/api/generate', {
          fileId: file.id,
          fileUrl: file.secureUrl,
          cloudinaryConfig,
          geminiApiKeys: cloudinaryConfig.geminiApiKeys.split(',').map(k => k.trim()).filter(k => k),
          model: cloudinaryConfig.model || 'gemini-3-flash-preview'
        }, {
          timeout: 300000 // 5 minutes timeout
        });

        if (response.data.success) {
          const newUrl = response.data.generatedUrl;
          
          // Update local URL map
          setGeneratedUrls(prev => ({
            ...prev,
            [file.id]: newUrl
          }));

          setFiles(prev => prev.map(f => 
            f.id === file.id 
              ? { ...f, status: 'AlreadyCopy', generatedUrl: newUrl } 
              : f
          ));
          
          setGeneratedFiles(prev => [
            { ...file, status: 'AlreadyCopy', generatedUrl: newUrl },
            ...prev.filter(f => f.id !== file.id) // Avoid duplicates
          ]);
        }
        
        // Add a small delay between requests to be nice to APIs
        await new Promise(resolve => setTimeout(resolve, 1000));

      } catch (err: any) {
        console.error(`Failed to generate for file ${file.id}:`, err);
        const errorMessage = err.response?.data?.error || err.message;
        // Don't set global error, just log it and maybe show a toast/notification in a real app
        // For now, we'll set a temporary error state that clears or append to a log
        console.log(`Error details: ${JSON.stringify(err.response?.data)}`);
        setError(`Failed to generate for file ${file.originalFilename}: ${errorMessage}`);
        
        // Wait a bit longer on error before continuing
        await new Promise(resolve => setTimeout(resolve, 2000));
      } finally {
        eventSource.close();
        // Clear status after a delay
        setTimeout(() => {
          setGenerationStatus(prev => {
            const newState = { ...prev };
            delete newState[file.id];
            return newState;
          });
        }, 5000);
      }
    }
    setIsGenerating(false);
    setCurrentFileId(null);
    stopGenerationRef.current = false;
  };

  const handleDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename.replace('.txt', '.mdx'); // Suggest MDX extension
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Pagination Logic
  const pendingFilesList = files.filter(f => f.status === 'Pending');
  const currentList = activeTab === 'pending' ? pendingFilesList : generatedFiles;
  const totalPages = Math.ceil(currentList.length / itemsPerPage);
  const paginatedList = currentList.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
              AI
            </div>
            <h1 className="text-xl font-semibold text-slate-900">Content Generator</h1>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
              externalApiStatus === 'online' ? 'bg-green-100 text-green-700' : 
              externalApiStatus === 'offline' ? 'bg-red-100 text-red-700' : 
              'bg-slate-100 text-slate-600'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                externalApiStatus === 'online' ? 'bg-green-500' : 
                externalApiStatus === 'offline' ? 'bg-red-500' : 
                'bg-slate-400'
              }`} />
              {externalApiStatus === 'online' ? 'API Connected' : 
               externalApiStatus === 'offline' ? 'API Offline' : 
               'Checking API...'}
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-full transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden mb-6"
            >
              <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h2 className="text-lg font-medium mb-4">Configuration</h2>
                
                <div className="mb-6">
                  <label className="block text-sm font-medium text-slate-700 mb-1">Gemini Model</label>
                  <select
                    value={cloudinaryConfig.model || 'gemini-3-flash-preview'}
                    onChange={e => setCloudinaryConfig({...cloudinaryConfig, model: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    {AVAILABLE_MODELS.map(model => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-500 mt-1">Select the model to use for generation.</p>
                </div>

                <div className="mb-6">
                  <label className="block text-sm font-medium text-slate-700 mb-1">Gemini API Keys (Comma Separated)</label>
                  <input
                    type="text"
                    value={cloudinaryConfig.geminiApiKeys}
                    onChange={e => setCloudinaryConfig({...cloudinaryConfig, geminiApiKeys: e.target.value})}
                    placeholder="AIzaSy..., AIzaSy..."
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                  <p className="text-xs text-slate-500 mt-1">Provide multiple keys to rotate through them automatically.</p>
                </div>

                <h3 className="text-md font-medium mb-3 text-slate-600">Cloudinary Settings</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Cloud Name</label>
                    <input
                      type="text"
                      value={cloudinaryConfig.cloudName}
                      onChange={e => setCloudinaryConfig({...cloudinaryConfig, cloudName: e.target.value})}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">API Key</label>
                    <input
                      type="text"
                      value={cloudinaryConfig.apiKey}
                      onChange={e => setCloudinaryConfig({...cloudinaryConfig, apiKey: e.target.value})}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">API Secret</label>
                    <input
                      type="password"
                      value={cloudinaryConfig.apiSecret}
                      onChange={e => setCloudinaryConfig({...cloudinaryConfig, apiSecret: e.target.value})}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Queue Stats */}
        {queueStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Active Jobs</div>
              <div className="text-2xl font-semibold text-blue-600">{queueStats.activeJobs}</div>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Completed</div>
              <div className="text-2xl font-semibold text-green-600">{queueStats.completedJobs}</div>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Failed</div>
              <div className="text-2xl font-semibold text-red-600">{queueStats.failedJobs}</div>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Total Processed</div>
              <div className="text-2xl font-semibold text-slate-700">{queueStats.totalJobs}</div>
            </div>
          </div>
        )}

        {/* Stats / Actions */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div className="flex gap-2 bg-white p-1 rounded-lg border border-slate-200 shadow-sm">
            <button
              onClick={() => setActiveTab('pending')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'pending' 
                  ? 'bg-indigo-50 text-indigo-700' 
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              Pending Prompts ({files.filter(f => f.status === 'Pending').length})
            </button>
            <button
              onClick={() => setActiveTab('generated')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'generated' 
                  ? 'bg-indigo-50 text-indigo-700' 
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              Available Downloads ({generatedFiles.length})
            </button>
          </div>

          <div className="flex gap-3">
            <button
              onClick={fetchFiles}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 font-medium transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={startGeneration}
              disabled={files.filter(f => f.status === 'Pending').length === 0 && !isGenerating}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-white transition-colors ${
                isGenerating 
                  ? 'bg-red-500 hover:bg-red-600 shadow-sm hover:shadow' 
                  : 'bg-indigo-600 hover:bg-indigo-700 shadow-sm hover:shadow'
              }`}
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Stop Generation
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Generation Queue
                </>
              )}
            </button>
          </div>
        </div>

        {/* Content Table */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          {loading && files.length === 0 ? (
            <div className="p-12 text-center text-slate-500">Loading files...</div>
          ) : error ? (
            <div className="p-12 text-center text-red-500 flex flex-col items-center gap-2">
              <AlertCircle className="w-8 h-8" />
              {error}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-4 font-medium text-slate-500">Filename</th>
                    <th className="px-6 py-4 font-medium text-slate-500">Status</th>
                    <th className="px-6 py-4 font-medium text-slate-500">Size</th>
                    <th className="px-6 py-4 font-medium text-slate-500">Date</th>
                    <th className="px-6 py-4 font-medium text-slate-500 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {paginatedList.map((file) => (
                    <tr key={file.id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4 font-medium text-slate-900">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded bg-slate-100 flex items-center justify-center text-slate-400">
                            <span className="text-xs font-mono">TXT</span>
                          </div>
                          {file.originalFilename}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        {file.id === currentFileId ? (
                          <div className="w-full max-w-xs">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-blue-700">
                                {generationStatus[file.id]?.message || 'Starting...'}
                              </span>
                              <span className="text-xs font-medium text-blue-700">
                                {generationStatus[file.id]?.progress || 0}%
                              </span>
                            </div>
                            <div className="w-full bg-blue-100 rounded-full h-2">
                              <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                                style={{ width: `${generationStatus[file.id]?.progress || 5}%` }}
                              />
                            </div>
                          </div>
                        ) : file.status === 'AlreadyCopy' ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-green-50 text-green-700">
                            <CheckCircle className="w-3 h-3" />
                            Completed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
                            Pending
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {(file.sizeBytes / 1024).toFixed(1)} KB
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {new Date(file.uploadTimestamp).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {file.generatedUrl && (
                            <>
                              <button
                                onClick={() => handleDownload(file.generatedUrl!, file.originalFilename)}
                                className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                                title="Download Generated Article"
                              >
                                <Download className="w-4 h-4" />
                              </button>
                              <a
                                href={file.generatedUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors"
                                title="View Generated Article"
                              >
                                <ExternalLink className="w-4 h-4" />
                              </a>
                            </>
                          )}
                          <a
                            href={file.secureUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                            title="View Source Prompt"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {paginatedList.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-6 py-12 text-center text-slate-400">
                        No files found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
          
          {/* Pagination Controls */}
          {!loading && !error && totalPages > 1 && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 bg-slate-50">
              <div className="text-sm text-slate-500">
                Showing <span className="font-medium">{(currentPage - 1) * itemsPerPage + 1}</span> to <span className="font-medium">{Math.min(currentPage * itemsPerPage, currentList.length)}</span> of <span className="font-medium">{currentList.length}</span> results
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm font-medium text-slate-600 bg-white border border-slate-300 rounded-md hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 text-sm font-medium text-slate-600 bg-white border border-slate-300 rounded-md hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
