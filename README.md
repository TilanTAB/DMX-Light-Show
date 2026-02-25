# 💡 DMX Light Show — AI-Powered Concert Lighting Engine

An AI-powered DMX lighting controller that generates synchronized light shows from YouTube music using real-time audio analysis and Azure OpenAI. Designed for uDMX USB adapters with RGB par lights.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What It Does

1. **Paste a YouTube URL** → Downloads audio via `yt-dlp`
2. **Analyzes the music** → FFT spectral analysis detects BPM, sections (verse/chorus/drop), energy levels, and drum patterns
3. **AI generates a lighting script** → Azure GPT creates a timestamped cue list with colors, strobe rules, and energy levels per section
4. **Beat-reactive DMX output** → Lights flash on every kick drum and snare hit, with colors driven by the AI cue list
5. **Live loopback mode** → Capture system audio (Spotify, browser, anything) and react to it in real-time without pre-analysis

### Key Features

- 🥁 **Spectral flux onset detection** — Lights punch on every drum hit using frame-to-frame FFT comparison (not just simple thresholds)
- 🎨 **AI-curated color palettes** — GPT designs kick drum and snare colors with high contrast per section
- ⚡ **Instant attack, fast decay** — Professional strobe-punch feel (0.95 attack on beats)
- 🔄 **Duplicate detection** — Recognizes previously generated shows by YouTube video ID
- 📊 **Real-time progress** — Live download percentage, analysis status, and elapsed timer in the UI
- 📝 **Full logging** — Every generation run logged to `logs/generation.log` with timestamps

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│           (Vite dev server — port 5173)                  │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐ │
│  │ Library │ │ Generate │ │  Play/    │ │  Progress  │ │
│  │  Grid   │ │  Button  │ │  Stop     │ │   Bar      │ │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └─────┬──────┘ │
└───────┼──────────┼───────────┼───────────────┼──────────┘
        │          │           │               │
        ▼          ▼           ▼               ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend (port 8000)                │
│                       app.py                             │
│  ┌──────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │ Show     │ │  Generation  │ │  Playback Control │   │
│  │ Library  │ │  Pipeline    │ │  (subprocess)     │   │
│  └──────────┘ └──────┬───────┘ └───────────────────┘   │
└──────────────────────┼──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────────┐
  │ yt-dlp   │  │ youtube_   │  │ llm_designer │
  │ + curl   │  │ analyzer   │  │   (Azure     │
  │ download │  │ (FFT/BPM)  │  │    GPT)      │
  └──────────┘  └────────────┘  └──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  music_light   │
              │  (DMX engine)  │
              │  ┌──────────┐  │
              │  │ uDMX USB │  │
              │  └──────────┘  │
              └────────────────┘
```

---

## 📦 Prerequisites

Before starting, ensure you have these installed:

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Backend, audio analysis, DMX control |
| **Node.js** | 18+ | React frontend build |
| **yt-dlp** | Latest | YouTube audio download |
| **ffmpeg** | Latest | Audio format conversion |
| **uDMX adapter** | — | USB DMX interface (Vendor: `0x16C0`, Product: `0x05DC`) |

### Azure OpenAI (Required for AI mode)

You need an Azure OpenAI resource with a deployed model (GPT-4o-mini or GPT-5 Nano recommended):
1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Create a resource and deploy a chat model
3. Copy the endpoint, API key, and deployment name

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/TilanTAB/DMX-Light-Show.git
cd DMX-Light-Show
```

### 2. Python Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic python-dotenv numpy pyaudiowpatch pyusb requests
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Azure OpenAI credentials
# Windows: notepad .env
# Linux/Mac: nano .env
```

Fill in your `.env`:
```env
AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE.openai.azure.com"
AZURE_OPENAI_API_KEY="your-api-key-here"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
```

### 5. Download Required Binaries

Download and place these in the project root directory:

- **yt-dlp**: [Download yt-dlp.exe](https://github.com/yt-dlp/yt-dlp/releases/latest) → place `yt-dlp.exe` in project root
- **ffmpeg**: [Download ffmpeg](https://ffmpeg.org/download.html) → place `ffmpeg.exe` in project root (or ensure it's on your PATH)

### 6. USB Driver Setup (Windows)

For the uDMX adapter to work on Windows, you need the **libusb** driver:

1. Download [Zadig](https://zadig.akeo.ie/)
2. Plug in your uDMX adapter
3. In Zadig: select the uDMX device → Install **WinUSB** driver
4. Verify: the device should appear when you run `python test_dmx.py`

---

## ▶️ Running the Application

You need **two terminals** running simultaneously:

### Terminal 1 — Backend API Server

```bash
# From project root
.venv\Scripts\python.exe app.py
```

You should see:
```
==================================================
  DMX Show Manager API (FastAPI)
  API:  http://localhost:8000/api/shows
  Docs: http://localhost:8000/docs
==================================================
```

### Terminal 2 — Frontend Dev Server

```bash
cd frontend
npx vite
```

You should see:
```
  VITE v6.x.x  ready in xxxms
  ➜  Local:   http://localhost:5173/
```

### Open the App

Navigate to **http://localhost:5173** in your browser.

---

## 🎮 Usage Guide

### Generating a Show (AI Mode)

1. Paste a YouTube URL in the input field
2. Click **⚡ Generate**
3. Watch the progress bar:
   - Downloading audio... 15% → 45% → 100%
   - Converting to WAV format...
   - Analyzing audio structure (FFT)...
   - Sending to Azure GPT for lighting design...
   - AI lighting plan received!
   - Done!
4. The show appears in the **Show Library**

### Playing a Show (Synced Mode)

1. Connect your uDMX adapter
2. Click **▶ Play** on any show card
3. Audio plays through your speakers while DMX lights sync to every drum beat
4. Click **⏹ Stop** to end

### Live Loopback Mode

1. Click **🎧 Loopback** (global or per-show)
2. Play music from **any source** (Spotify, YouTube, VLC, etc.)
3. Lights react in real-time to whatever audio is playing
4. Uses WASAPI loopback capture — works with any audio output device

### Regenerating a Show

If you paste a URL that already has a show:
- A dialog asks: **"Show exists. Regenerate AI plan?"**
- **Regenerate** = re-runs the AI with the cached audio (skips the 2-minute download)
- **Use Existing** = loads the previously generated show

---

## 📁 Project Structure

```
DMX-Light-Show/
├── app.py                  # FastAPI REST API (show library, generation, playback)
├── youtube_analyzer.py     # Audio pipeline: download → FFT analysis → AI call → save
├── llm_designer.py         # Azure OpenAI prompt engineering for lighting design
├── music_light.py          # DMX engine: beat detection, color mixing, uDMX output
├── .env.example            # Template for Azure credentials (copy to .env)
├── .gitignore              # Excludes secrets, binaries, audio files, logs
│
├── frontend/               # React UI (Vite)
│   ├── src/
│   │   ├── App.jsx         # Main component: library, controls, generator
│   │   ├── api.js          # API service layer (all fetch calls)
│   │   ├── index.css       # Dark glassmorphism design system
│   │   └── main.jsx        # React entry point
│   ├── index.html          # HTML shell
│   ├── vite.config.js      # Vite configuration
│   └── package.json        # Node dependencies
│
├── test_dmx.py             # Hardware test: cycles through RGB on uDMX
├── test_azure.py           # Connectivity test: verifies Azure OpenAI access
├── test_hello_world.py     # Basic DMX write test
├── test_white.py           # Full white output test
└── test_media.py           # Audio playback test
```

---

## 🔧 Configuration & Tuning

### Drum Sensitivity

Edit the top of `music_light.py` to adjust:

```python
KICK_GAIN   = 6.0   # Higher = more reactive to kick drums
SNARE_GAIN  = 4.0   # Higher = more reactive to snares
ONSET_THRESHOLD = 1.5  # Lower = triggers on softer hits
KICK_MIN_FLASH  = 0.6  # Minimum brightness on kick (0-1)
SNARE_MIN_FLASH = 0.5  # Minimum brightness on snare (0-1)
```

### Color Palettes

The AI generates palettes automatically, but you can edit the fallback palettes:

```python
PALETTES = [
    ((255, 0, 50),  (0, 150, 255)),   # (kick_color, snare_color)
    ((255, 50, 0),  (100, 0, 255)),
    ((0, 255, 100), (255, 0, 200)),
]
```

### DMX Channel Map

```python
# CH1=Master, CH2=Red, CH3=Green, CH4=Blue, CH5=White, CH6=Strobe
send_dmx(master, red, green, blue, white, strobe)
```

---

## 🔍 Troubleshooting

| Problem | Solution |
|---|---|
| `uDMX not found!` | Install WinUSB driver via Zadig. Check USB connection. |
| `GPT-5 Nano Connection Timeout` | Azure may be slow. Timeout is 120s with 2 retries. Check `.env` credentials. |
| `Download failed` | Ensure `yt-dlp.exe` and `ffmpeg.exe` are in the project root. |
| `No WAV file found after download` | ffmpeg conversion failed. Check ffmpeg is accessible. |
| Lights too dim | Increase `KICK_GAIN` and `SNARE_GAIN` in `music_light.py`. |
| Lights too chaotic | Decrease gains or increase `ONSET_COOLDOWN` (default: 0.12s). |
| Progress bar stuck | Check `logs/generation.log` for the actual error. |
| Shows not appearing | Check `logs/generation.log` — likely an Azure timeout or credential issue. |

### Viewing Logs

```bash
# Full generation log with timestamps
cat logs/generation.log

# Azure AI-specific log
cat dmx_ai.log
```

---

## 🛠️ API Documentation

FastAPI auto-generates interactive docs at: **http://localhost:8000/docs**

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/shows` | List all saved shows |
| `POST` | `/api/shows/generate` | Start AI show generation |
| `GET` | `/api/shows/generate/status` | Poll generation progress |
| `DELETE` | `/api/shows/{show_id}` | Delete a show |
| `POST` | `/api/play` | Start synced WAV playback |
| `POST` | `/api/loopback` | Start live loopback mode |
| `POST` | `/api/stop` | Stop any active playback |
| `GET` | `/api/status` | Get current playback status |

---

## 🧠 How the Beat Detection Works

The DMX engine uses **spectral flux onset detection** — an industry-standard technique:

1. **FFT** decomposes each audio frame into frequency bands
2. **Spectral flux** = difference between current and previous frame's FFT magnitudes
3. A sudden **positive flux** in the kick band (30-150 Hz) = kick drum hit
4. A sudden flux in the snare band (150-400 Hz) = snare hit
5. **AGC** (Auto Gain Control) adapts to the song's volume over time
6. The onset score must exceed the AGC baseline to register as a beat
7. **Cooldown** (120ms) prevents double-triggering on the same hit

This approach reacts to *transients* (sudden energy changes) rather than absolute volume — which is why it works across quiet acoustic songs and loud EDM equally well.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio extraction
- [ffmpeg](https://ffmpeg.org/) — Audio format conversion
- [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) — WASAPI loopback capture
- [PyUSB](https://github.com/pyusb/pyusb) — USB communication with uDMX
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) — AI lighting plan generation
