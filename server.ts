import 'dotenv/config';
import express from 'express';
import axios from 'axios';
import { v2 as cloudinary } from 'cloudinary';
import { GoogleGenAI } from '@google/genai';
import cors from 'cors';
import helmet from 'helmet';
import { randomUUID } from 'crypto';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// External API Configuration
const EXTERNAL_API_BASE = 'https://cloud-text-manager-server.vercel.app';
const EXTERNAL_API_URL = `${EXTERNAL_API_BASE}/api/files`;

// --- Processing System (In-Memory Job Store) ---
interface Job {
  id: string;
  fileId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  result?: any;
  error?: string;
  startTime: number;
  endTime?: number;
}

const jobs = new Map<string, Job>();
const clients = new Map<string, express.Response>();

// Helper to initialize Gemini
const getGemini = (providedKeys?: string[]) => {
  if (providedKeys && Array.isArray(providedKeys) && providedKeys.length > 0) {
    const validKeys = providedKeys.filter(k => k && !k.includes('YOUR_API_KEY') && k !== 'MY_GEMINI_API_KEY');
    if (validKeys.length > 0) {
      const randomKey = validKeys[Math.floor(Math.random() * validKeys.length)];
      return new GoogleGenAI({ apiKey: randomKey });
    }
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not defined in environment variables and no valid keys were provided in the request.');
  }
  if (apiKey === 'MY_GEMINI_API_KEY' || apiKey.includes('YOUR_API_KEY')) {
    throw new Error('Invalid GEMINI_API_KEY detected. Please configure your API key.');
  }
  return new GoogleGenAI({ apiKey });
};

// Helper to update job progress and notify clients
const updateJob = (jobId: string, updates: Partial<Job>) => {
  const job = jobs.get(jobId);
  if (!job) return;

  Object.assign(job, updates);
  if (updates.status === 'completed' || updates.status === 'failed') {
    job.endTime = Date.now();
  }

  // Notify SSE clients subscribed to this fileId
  const client = clients.get(job.fileId);
  if (client) {
    client.write(`data: ${JSON.stringify({ 
      step: job.status, 
      progress: job.progress, 
      message: job.message,
      jobId: job.id
    })}\n\n`);
  }
};

// --- API Routes ---

// Root Endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Content Generation Server is Running',
    version: '1.0.0',
    endpoints: {
      health: '/api/health',
      stats: '/api/stats',
      generate: '/api/generate (POST)',
      progress: '/api/progress/:fileId (SSE)'
    }
  });
});

// Health Check Endpoint
app.get('/api/health', async (req, res) => {
  const health = {
    status: 'online',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    externalApi: 'unknown'
  };

  try {
    const response = await axios.get(EXTERNAL_API_BASE, { timeout: 5000 });
    health.externalApi = response.status === 200 ? 'online' : 'offline';
  } catch (error: any) {
    health.externalApi = 'offline';
  }

  res.json(health);
});

// Stats Endpoint
app.get('/api/stats', (req, res) => {
  const activeJobs = Array.from(jobs.values()).filter(j => j.status === 'processing').length;
  const completedJobs = Array.from(jobs.values()).filter(j => j.status === 'completed').length;
  const failedJobs = Array.from(jobs.values()).filter(j => j.status === 'failed').length;

  res.json({
    activeJobs,
    completedJobs,
    failedJobs,
    totalJobs: jobs.size,
    uptime: process.uptime()
  });
});

// Proxy: Get Files
app.get('/api/proxy/files', async (req, res) => {
  try {
    const response = await axios.get(EXTERNAL_API_URL);
    res.json(response.data);
  } catch (error: any) {
    console.error('Error fetching files:', error.message);
    res.status(500).json({ error: 'Failed to fetch files from external API' });
  }
});

// SSE Endpoint for Progress
app.get('/api/progress/:fileId', (req, res) => {
  const { fileId } = req.params;
  
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  clients.set(fileId, res);
  
  // Send initial connection message
  res.write(`data: ${JSON.stringify({ step: 'init', progress: 0, message: 'Connected' })}\n\n`);

  // If there's an active job for this file, send current status
  const existingJob = Array.from(jobs.values()).find(j => j.fileId === fileId && j.status === 'processing');
  if (existingJob) {
    res.write(`data: ${JSON.stringify({ 
      step: existingJob.status, 
      progress: existingJob.progress, 
      message: existingJob.message 
    })}\n\n`);
  }

  req.on('close', () => {
    clients.delete(fileId);
  });
});

// --- Automation System ---

// Configuration for Automation (Fallback to Env Vars)
const AUTO_CONFIG = {
  cloudName: process.env.CLOUDINARY_CLOUD_NAME || 'dky6urpy2',
  apiKey: process.env.CLOUDINARY_API_KEY || '164368975898265',
  apiSecret: process.env.CLOUDINARY_API_SECRET || '4toK14OkJY-ePc-NFE_YsB2vrFc',
  geminiApiKeys: process.env.GEMINI_API_KEYS ? process.env.GEMINI_API_KEYS.split(',') : [],
  model: process.env.GEMINI_MODEL || 'gemini-3-flash-preview',
  pollInterval: 60000 // 60 seconds
};

// Core Processing Logic (Refactored)
const processFile = async (fileId: string, fileUrl: string, config: any) => {
  const jobId = randomUUID();
  const job: Job = {
    id: jobId,
    fileId,
    status: 'pending',
    progress: 0,
    message: 'Job created',
    startTime: Date.now()
  };
  jobs.set(jobId, job);
  
  updateJob(jobId, { status: 'processing', progress: 5, message: 'Starting processing...' });

  try {
    // 1. Fetch prompt content
    updateJob(jobId, { progress: 10, message: 'Fetching prompt from source...' });
    console.log(`[${jobId}] Fetching prompt from: ${fileUrl}`);
    
    let promptText = '';
    try {
      const promptResponse = await axios.get(fileUrl);
      promptText = typeof promptResponse.data === 'string' 
        ? promptResponse.data 
        : JSON.stringify(promptResponse.data);
    } catch (fetchError: any) {
      throw new Error(`Failed to fetch prompt file: ${fetchError.message}`);
    }

    // 2. Generate content with Gemini
    updateJob(jobId, { progress: 30, message: 'Generating content with Gemini...' });
    const ai = getGemini(config.geminiApiKeys);
    const selectedModel = config.model || 'gemini-3-flash-preview';
    
    let generatedContent = '';
    try {
      const result = await ai.models.generateContent({
        model: selectedModel,
        contents: promptText,
        config: { responseMimeType: 'text/plain' }
      });
      generatedContent = result.text || '';
      if (!generatedContent) throw new Error('Gemini returned empty content');
    } catch (geminiError: any) {
      throw new Error(`Gemini generation failed: ${geminiError.message}`);
    }

    // 3. Upload to Cloudinary
    updateJob(jobId, { progress: 70, message: 'Uploading to Cloudinary...' });
    let uploadResult;
    try {
      cloudinary.config({
        cloud_name: config.cloudName,
        api_key: config.apiKey,
        api_secret: config.apiSecret,
      });

      const base64Content = Buffer.from(generatedContent).toString('base64');
      const dataUri = `data:text/markdown;base64,${base64Content}`;
      
      uploadResult = await cloudinary.uploader.upload(dataUri, {
        resource_type: 'raw',
        folder: 'generated_articles',
        public_id: `article_${fileId}_${Date.now()}.mdx`,
      });
    } catch (cloudinaryError: any) {
      throw new Error(`Cloudinary upload failed: ${cloudinaryError.message}`);
    }

    // 4. Update External API Status
    updateJob(jobId, { progress: 90, message: 'Updating external API status...' });
    try {
      await axios.put(`${EXTERNAL_API_URL}/${fileId}`, {
        status: 'AlreadyCopy',
        completedTimestamp: Date.now()
      });
    } catch (apiError: any) {
      console.warn(`[${jobId}] Warning: Failed to update status on external API.`);
    }

    // Complete Job
    updateJob(jobId, { 
      status: 'completed', 
      progress: 100, 
      message: 'Process completed successfully!',
      result: { generatedUrl: uploadResult.secure_url }
    });

    return {
      success: true,
      jobId,
      fileId,
      generatedUrl: uploadResult.secure_url,
      content: generatedContent
    };

  } catch (error: any) {
    console.error(`[${jobId}] Job failed:`, error.message);
    updateJob(jobId, { 
      status: 'failed', 
      progress: 0, 
      message: error.message || 'An error occurred',
      error: error.message
    });
    throw error;
  }
};

// Automation Loop
const runAutomation = async () => {
  console.log('[Automation] Checking for pending files...');
  try {
    const response = await axios.get(EXTERNAL_API_URL);
    const files = response.data;
    const pendingFiles = files.filter((f: any) => f.status === 'Pending');

    console.log(`[Automation] Found ${pendingFiles.length} pending files.`);

    for (const file of pendingFiles) {
      // Check if already processing
      const isProcessing = Array.from(jobs.values()).some(j => j.fileId === file.id && (j.status === 'processing' || j.status === 'pending'));
      if (isProcessing) {
        console.log(`[Automation] File ${file.id} is already being processed. Skipping.`);
        continue;
      }

      console.log(`[Automation] Starting processing for file ${file.id}...`);
      
      // Use AUTO_CONFIG for automation
      // Note: We need to ensure geminiApiKeys is passed correctly. 
      // If AUTO_CONFIG.geminiApiKeys is empty, getGemini will fall back to process.env.GEMINI_API_KEY
      
      try {
        await processFile(file.id, file.secureUrl, AUTO_CONFIG);
        console.log(`[Automation] File ${file.id} processed successfully.`);
      } catch (err) {
        console.error(`[Automation] Failed to process file ${file.id}:`, err);
      }
      
      // Wait a bit between files to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  } catch (error: any) {
    console.error('[Automation] Error in automation loop:', error.message);
  }
};

// Start Automation Loop
setInterval(runAutomation, AUTO_CONFIG.pollInterval);
// Run once on startup after a delay
setTimeout(runAutomation, 5000);

// Generate Article Endpoint (Manual Trigger)
app.post('/api/generate', async (req, res) => {
  const { fileId, fileUrl, cloudinaryConfig, geminiApiKeys, model } = req.body;

  if (!fileId || !fileUrl || !cloudinaryConfig) {
    return res.status(400).json({ error: 'Missing required parameters' });
  }

  try {
    const result = await processFile(fileId, fileUrl, {
      ...cloudinaryConfig,
      geminiApiKeys,
      model
    });
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ 
      error: error.message || 'An error occurred during generation',
      jobId: jobs.size > 0 ? Array.from(jobs.keys()).pop() : undefined // Best guess at job ID if failed early
    });
  }
});

// Start Server
app.listen(PORT, () => {
  console.log(`Content Generation Server running on port ${PORT}`);
});
