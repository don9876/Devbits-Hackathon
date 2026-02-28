# Real-Estate CPaaS

Automated phone-based real-estate sales agent built around [Groq](https://groq.com/), [Whisper](https://github.com/openai/whisper) and [@rhasspy/piper](https://github.com/rhasspy/piper).

## Repository layout

```
.
├── main.py                      # server implementation
├── en_US-lessac-high.onnx       # TTS model (large; not checked in)
├── en_US-lessac-high.onnx.json  # model metadata
├── real_estate_data.json        # company/listings data
├── Real_Estate_Call_Logs.txt    # rolling log of leads
├── requirements.txt             # Python dependencies
└── README.md                    # this file
```

## Prerequisites

* Python 3.10+ (3.11 recommended).
* A working CPaaS client that speaks the simple `MSG_AUDIO_8KHZ` protocol (not included).
* [piper](https://github.com/rhasspy/piper) binary installed; set `PIPER_EXE` in `main.py` or export the same name as an environment variable (`export PIPER_EXE=/path/to/piper` on POSIX, `setx PIPER_EXE "C:\path\to\piper.exe"` on Windows).  The server will spawn this executable for text-to-speech, so ensure the path is valid or have it on your `PATH`.
* `GROQ_API_KEY` environment variable (get one from https://groq.com)
* Network access to download Whisper models (first run).

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
