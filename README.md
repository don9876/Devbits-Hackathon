# Real-Estate CPaaS

A production-ready automated phone-based real-estate sales agent that leverages cutting-edge AI technologies to handle inbound real-estate inquiries. Built with [Groq](https://groq.com/) for fast LLM inference, [Whisper](https://github.com/openai/whisper) for speech-to-text transcription, and [@rhasspy/piper](https://github.com/rhasspy/piper) for high-quality text-to-speech synthesis.

## Overview

This CPaaS server listens on a dedicated TCP port and handles a single incoming call from a remote CPaaS client. It orchestrates a multi-stage pipeline:

1. **Speech Detection** – Uses Silero VAD (Voice Activity Detection) to identify when the user is speaking
2. **Transcription** – Converts audio to text using Whisper's distil-medium.en model
3. **Dialogue Generation** – Sends transcribed text to Groq's Llama 3.3 70B model with a custom real-estate sales prompt
4. **Text-to-Speech** – Streams responses through Piper for natural-sounding audio synthesis
5. **Lead Capture** – Records all interactions and generates automated lead summaries

Perfect for real-estate companies looking to automate first-contact sales calls, lead qualification, and property information delivery.

## Repository layout

```
.
├── main.py                      # core server implementation with CallHandler
├── en_US-lessac-high.onnx       # TTS voice model (large; not committed by default)
├── en_US-lessac-high.onnx.json  # model metadata and configuration
├── real_estate_data.json        # company info and property listings
├── Real_Estate_Call_Logs.txt    # automated lead summaries (append-only)
├── requirements.txt             # Python package dependencies
├── .gitignore                   # version control exclusions
└── README.md                    # this file
```

## Quick Start

```bash
git clone https://github.com/don9876/Devbits-Hackathon.git
cd Devbits-Hackathon
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sk_your_real_key_here"
export PIPER_EXE="/path/to/piper"
python main.py
```

## Prerequisites

### System Requirements

* **Python**: 3.10 or higher (3.11+ recommended for best performance)
* **Operating System**: Linux, macOS, or Windows (with WSL2 recommended for Windows)
* **RAM**: Minimum 4GB; 8GB+ recommended for smooth concurrent model operations
* **Network**: Stable internet connection for initial model downloads (~500MB for Whisper)

### External Dependencies

* **[piper](https://github.com/rhasspy/piper)** – A high-performance TTS engine. Must be installed separately and made available on your `PATH` or configured via the `PIPER_EXE` environment variable.
  
* **[GROQ API Key](https://groq.com/)** – Required for LLM inference. Sign up at Groq and generate an API key; set as the `GROQ_API_KEY` environment variable.

* **CPaaS Client** – A compatible SIP/telephony client that speaks the `MSG_AUDIO_8KHZ` protocol (not included in this repo). This is the incoming call interface.

## Installation

```bash
git clone https://github.com/don9876/Devbits-Hackathon.git
cd Devbits-Hackathon
python -m venv venv
source venv/bin/activate        # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` contains the packages required by `main.py`:

```
torch>=2.0.0            # CPU or GPU build, install nightly if you need CUDA
faster-whisper>=0.5.0   # ASR frontend
groq>=1.0.0             # Groq LLM client
numpy
scipy
```

### Piper and model files

The ONNX TTS model is large (~200 MB) and therefore not checked in by default (see `.gitignore`). If you choose to commit it to the repo you can remove the ignore line; otherwise download it yourself and place it in the repo root or update `MODEL_PATH` in `main.py` to wherever you store it.

Piper itself is a separate project. Build from source with `cargo` or grab a binary release from the [GitHub releases page](https://github.com/rhasspy/piper/releases). Once installed, either add it to your `PATH` or set the `PIPER_EXE` constant in `main.py` to the full executable path. The server will spawn Piper to produce 8 kHz audio for the CPaaS client.

## Usage

Before starting the server you must set a valid Groq API key. The example below uses a placeholder; substitute your real key:

```bash
# POSIX style
export GROQ_API_KEY="sk_your_real_key_here"

# On Windows PowerShell use setx or the Environment Variables UI
setx GROQ_API_KEY "sk_your_real_key_here"

python main.py
```

The server listens on `0.0.0.0:9092` and accepts a single call. Lead summaries are appended to `Real_Estate_Call_Logs.txt` automatically.

A sample `real_estate_data.json` might look like:

```json
{
  "company_name": "Horizon Estates",
  "address": "123 Main St, Noida",
  "listings": {
    "ATS Pristine": {
      "type": "3BHK Apartment",
      "details": "Eco-friendly project with clubhouse, pool, gym",
      "price_range": "1.8–2.2 Cr"
    }
  }
}
```

## Development

* Add new listings to `real_estate_data.json`.
* Update the system prompt in `CallHandler.build_system_prompt()`.
* Extend/modularise `CallHandler` as needed; it’s currently monolithic.

## GitHub

Add `.gitignore` with:

```
venv/
__pycache__/
*.pyc
`en_US-lessac-high.onnx`
whisper-*
*.onnx
```

and push the repo as usual:

```sh
git init
git add .
git commit -m "initial real-estate CPaaS server"
git remote add origin git@github.com:youruser/yourrepo.git
git push -u origin main
```
