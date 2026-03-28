# Content Generation Server (Ollama Cloud)

This project has been migrated from Gemini API to **Ollama Cloud** using the official Ollama Python client.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed.
- `OLLAMA_API_KEY` set in your `.env` file.

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Start the Server
Run the FastAPI server on port 3000:
```bash
# Using npm
npm run start:python

# Or directly
python server.py
```

#### Batch Content Generation
To process pending files in batches:
```bash
# Using npm
npm run generate:python

# Or directly
python scripts/generate.py
```

## 🛠️ Tech Stack
- **AI Backend**: Ollama Cloud (`gpt-oss:120b-cloud`, `qwen3-coder:480b-cloud`, `kimi-k2-thinking`)
- **Server**: FastAPI (Python)
- **Storage**: Cloudinary (for generated content)
- **Real-time Updates**: Server-Sent Events (SSE)

## 📁 Key Files
- `server.py`: Main FastAPI server with automation loop.
- `scripts/generate.py`: Batch processing utility.
- `.env`: Environment configuration.
- `requirements.txt`: Python package dependencies.

## 📝 Integration Details
The system automatically selects the best model based on the prompt's intent:
- **Coding**: `qwen3-coder:480b-cloud`
- **Reasoning**: `kimi-k2-thinking`
- **Default**: `gpt-oss:120b-cloud`
