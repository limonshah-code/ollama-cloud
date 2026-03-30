import 'dotenv/config';
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import nodemailer from 'nodemailer';

// ================= CONFIG =================
const EXTERNAL_API_BASE = 'https://cloud-text-manager-server.vercel.app';
const EXTERNAL_API_URL = `${EXTERNAL_API_BASE}/api/all-files`;
const GENERATED_DIR = path.join(process.cwd(), 'generated-content');

const BATCH_SIZE = 10;
const CONCURRENCY_LIMIT = 1;
const REQUEST_DELAY = 1000;
const MAX_RETRIES = 5;

// Ensure output directory exists
if (!fs.existsSync(GENERATED_DIR)) {
  fs.mkdirSync(GENERATED_DIR, { recursive: true });
}

// ================= GEMINI MULTI-KEY =================
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || 'stepfun/step-3.5-flash:free';

if (!OPENROUTER_API_KEY) {
  throw new Error('OPENROUTER_API_KEY is not defined.');
}

const sendChat = async (messages: any[], model: string) => {
  const url = 'https://openrouter.ai/api/v1/chat/completions';
  const headers = {
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/Shah-Limon/ollama-cloud',
    'X-Title': 'Content Generation Script TS',
  };

  const payload = {
    model,
    messages,
    extra_body: { reasoning: { enabled: true } },
  };

  try {
    const response = await axios.post(url, payload, { headers, timeout: 120000 });
    const choice = response.data.choices?.[0];
    if (!choice) throw new Error('No choices in OpenRouter response');

    return {
      content: choice.message.content || '',
      role: choice.message.role || 'assistant',
      reasoning_details: choice.message.reasoning_details,
    };
  } catch (error: any) {
    const errorData = error.response?.data;
    throw new Error(`OpenRouter API error: ${errorData ? JSON.stringify(errorData) : error.message}`);
  }
};

// ================= SLUG GENERATOR =================
const generateSafeFilename = (originalFilename: string): string => {
  const baseName = path.parse(originalFilename).name;
  return (
    baseName
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[^\w\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '') + '.mdx'
  );
};

// ================= EMAIL =================
const sendEmail = async (subject: string, text: string) => {
  if (!process.env.EMAIL_HOST) return;

  const transporter = nodemailer.createTransport({
    host: process.env.EMAIL_HOST,
    port: Number(process.env.EMAIL_PORT) || 587,
    secure: process.env.EMAIL_SECURE === 'true',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  });

  await transporter.sendMail({
    from: process.env.EMAIL_FROM || process.env.EMAIL_USER,
    to: process.env.EMAIL_TO || process.env.EMAIL_USER,
    subject,
    text,
  });
};

// ================= UTILS =================
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

// ================= RETRY + KEY ROTATION =================
const generateWithRetry = async (promptText: string, model: string) => {
  let attempt = 0;

  while (attempt < MAX_RETRIES) {
    try {
      await sleep(REQUEST_DELAY);

      const messages = [{ role: 'user', content: promptText }];
      const response = await sendChat(messages, model);

      if (!response.content) throw new Error('Empty response');
      return response.content;

    } catch (error: any) {
      attempt++;
      console.log(`Retry ${attempt}/${MAX_RETRIES} - ${error.message}`);
      await sleep(2000 * attempt);

      if (attempt >= MAX_RETRIES) throw error;
    }
  }
};

// ================= FILE PROCESS =================
const processFile = async (file: any, model: string) => {
  console.log(`Processing: ${file.originalFilename}`);

  try {
    const promptResponse = await axios.get(file.secureUrl);
    const promptText =
      typeof promptResponse.data === 'string'
        ? promptResponse.data
        : JSON.stringify(promptResponse.data);

    const generatedContent = await generateWithRetry(promptText, model);

    const safeFilename = generateSafeFilename(file.originalFilename);
    const filePath = path.join(GENERATED_DIR, safeFilename);

    fs.writeFileSync(filePath, generatedContent, 'utf-8');

    await axios.put(`${EXTERNAL_API_URL}/${file._id}`, {
      status: 'AlreadyCopy',
      completedTimestamp: Date.now(),
    });

    console.log(`✅ Done: ${safeFilename}`);
    return { success: true, name: safeFilename };

  } catch (err: any) {
    console.error(`❌ Failed: ${file.originalFilename}`, err.message);
    return { success: false, name: file.originalFilename, error: err.message };
  }
};

// ================= PARALLEL QUEUE =================
const runQueue = async (files: any[]) => {
  const model = OPENROUTER_MODEL;
  const results: any[] = [];
  let index = 0;

  async function worker() {
    while (true) {
      if (index >= files.length) break;
      const currentIndex = index++;
      const file = files[currentIndex];
      const result = await processFile(file, model);
      results.push(result);
    }
  }

  const workers = Array.from({ length: CONCURRENCY_LIMIT }, () => worker());
  await Promise.all(workers);

  return results;
};

// ================= FOLDER STATS =================
const getFolderStats = (dir: string) => {
  if (!fs.existsSync(dir)) return { totalFiles: 0, mdxFiles: 0 };
  const files = fs.readdirSync(dir);
  const mdxFiles = files.filter((f) => f.endsWith('.mdx')).length;
  return { totalFiles: files.length, mdxFiles };
};

// ================= SEND BATCH EMAIL =================
const sendBatchEmail = async (success: any[], failed: any[]) => {
  // Bangladesh time
  const now = new Date();
  const dateTimeStr = now.toLocaleString('en-US', { timeZone: 'Asia/Dhaka' });

  const folderStats = getFolderStats(GENERATED_DIR);

  const subject = `OpenRouter Content Generation Report - ${dateTimeStr} - ${success.length} Success, ${failed.length} Failed`;

  const body = `
OpenRouter Content Generation Report
----------------------------------
Date & Time (BDT): ${dateTimeStr}
Total Processed: ${success.length + failed.length}
Success: ${success.length}
Failed: ${failed.length}

Successful Files:
${success.map((f) => `- ${f.name}`).join('\n')}

Failed Files:
${failed.map((f) => `- ${f.name} (${f.error})`).join('\n')}

Folder Stats:
- Total files in folder: ${folderStats.totalFiles}
- Total MDX files: ${folderStats.mdxFiles}

Generated content has been saved to the repository.
`;

  await sendEmail(subject, body);
};

// ================= MAIN =================
const run = async () => {
  console.log('🚀 Starting OpenRouter Processing...');

  try {
    const response = await axios.get("https://cloud-text-manager-server.vercel.app/api/all-files?section=General");
    const pendingFiles = response.data
      .filter((f: any) => f.status === 'Pending')
      .slice(0, BATCH_SIZE);

    if (!pendingFiles.length) {
      console.log('No pending files.');
      return;
    }

    console.log(`Found ${pendingFiles.length} files.`);

    const results = await runQueue(pendingFiles);

    const success = results.filter((r) => r.success);
    const failed = results.filter((r) => !r.success);

    await sendBatchEmail(success, failed);

    console.log('🎉 Batch Completed Successfully');

  } catch (error: any) {
    console.error('Fatal error:', error.message);
    await sendEmail('Batch Failed', error.message);
    process.exit(1);
  }
};


run();
