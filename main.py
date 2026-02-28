#!/usr/bin/env python3

import socket
import struct
import threading
import queue
import numpy as np
import torch
import subprocess
import os
import json
import time
import re
from faster_whisper import WhisperModel
from groq import Groq
from scipy import signal

# ================= CONFIG =================
"""API key for Groq LLM access.

The code will read from the `GROQ_API_KEY` environment variable and
expects you to set one before starting the server.  A hardcoded key is
dangerous to commit, so we simply use a clear placeholder here.  When
cloning the repository you should run::

    export GROQ_API_KEY="your_real_key_here"   # POSIX shells
    setx GROQ_API_KEY "your_real_key_here"     # Windows PowerShell (persist)

and avoid putting the actual secret in source control.
"""
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
PIPER_EXE = os.environ.get("PIPER_EXE", "piper")
MODEL_PATH = os.path.abspath("./en_US-lessac-high.onnx")
DATA_FILE = "real_estate_data.json" 
MOM_LOG_FILE = "Real_Estate_Call_Logs.txt" 

WHISPER_MODEL_SIZE = "distil-medium.en" 
AST_RATE = 8000
AI_RATE = 16000
PIPER_RATE = 22050
MSG_AUDIO_8KHZ = 0x10

# ================= HARDWARE DETECTION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"[*] CPaaS Engine Active: Using {DEVICE.upper()} acceleration.")

# ================= DATA LOADING =================
def load_real_estate_data():
    """Return contents of `DATA_FILE` or a small default dict.

    This is called once at startup; if the file cannot be read a fallback
    with a generic company name is returned so the server can still run.
    """
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON Error loading {DATA_FILE}: {e}")
        return {"company_name": "Horizon Estates", "listings": {}, "address": "Global HQ"}

REAL_ESTATE_DATA = load_real_estate_data()

# ================= AI MODELS =================
print(f"Loading Models ({WHISPER_MODEL_SIZE}, VAD)...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
groq_client = Groq(api_key=GROQ_API_KEY)

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
vad_model = vad_model.to(DEVICE)
(get_speech_timestamps, _, _, VADIterator, _) = utils

# ================= CALL HANDLER =================
class CallHandler:
    """Encapsulates state and threads for a single call.

    A `CallHandler` object wraps all of the queues, subprocesses and
    threads required to receive audio from the CPaaS client, transcribe it,
    generate an LLM response and send back TTS audio.  It also manages
    lead summary generation and connection teardown.
    """
    def __init__(self, conn):
        self.conn = conn
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_ai_talking = False
        self.chat_history = [] 

        self.company_info = REAL_ESTATE_DATA
        self.system_prompt = self.build_system_prompt()
        
        self.piper_proc = subprocess.Popen(
            [PIPER_EXE, "--model", MODEL_PATH, "--output_file", "-", "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        
        threading.Thread(target=self.receiver, daemon=True).start()
        threading.Thread(target=self.processor, daemon=True).start()
        threading.Thread(target=self.sender, daemon=True).start()
        threading.Thread(target=self.tts_worker, daemon=True).start()
        threading.Thread(target=self.piper_stdout_reader, daemon=True).start()
        
        threading.Thread(target=self.initial_greeting, daemon=True).start()

    def build_system_prompt(self):
        items = self.company_info.get('listings', {})
        listing_str = "\n".join([f"- {name}: Type: {d['type']}, Details: {d['details']}, Price: {d['price_range']}" for name, d in items.items()])
        
        return f"""You are an intelligent sales agent for {self.company_info['company_name']}. 
Do not just read a script; understand the user's context.

CORE TERMINATION RULE:
- If the user says "No", "No thank you", or "Nothing else" when asked if they need more help, you MUST append the tag [HANGUP] at the very end of your final response.
- Once a site visit is confirmed and you have provided the professional closing, you MUST append [HANGUP].

CORE MEMORY PROTOCOL:
- Before asking for Name, Number, or Budget, check the conversation history. If provided, DO NOT ask again.
- Address the user by name once it is provided to show you are listening.

USER PROFILING PROTOCOL:
When a user calls, you must ask questions to build a profile. Ask only one question at a time from the user. Do not ask more than 2 questions at once:
1. Where do they stay?
2. What are they looking for? (3BHK vs 4BHK, Villa vs Apartment)
3. What is their budget?
4. When do they plan to move?

CONTEXTUAL LOGIC:
- If the caller is from the NCR, prioritize suggesting properties in Noida or Gurgaon.
- If they ask for a 'Villa', do NOT pitch an apartment. Match their 'Type' exactly.
- If they ask for as 'Apartment', do NOT pitch a Villa. Match their 'Type' exactly.
- If the query is complex or involves 'Luxury' above 5Cr, mention you can transfer them to a 'Specialized Luxury Expert'.
- Do not talk about any specific protocols being followed.

CURRENT DATASET:
{listing_str}

CONVERSATIONAL PROTOCOL:
1. USER PROFILE: First, complete building user profile by the rules stated in "USER PROFILING PROTOCOL" above.
1. FORMATTING: Speak in plain text ONLY. No markdown. Use professional, welcoming English.
2. INQUIRY PHASE: Provide property details, pricing, and amenities freely. Do not extend any reply to more than 4 lines. Keep it small and to the point.
3. LEAD TRIGGER: Only start collecting personal data if the user wants to "visit", "view", "book a tour", or "speak to an agent."
4. DATA COLLECTION (After Trigger):
   - CHECK MEMORY: If Name/Number were provided earlier, do not ask again.
   - If Name missing -> Ask for Name.
   - If Mobile missing -> Ask for Mobile.
   - If both present -> Confirm the site visit schedule.
   - If a mobile number has been provided, read it back to the user in digits independently.
5. SMART FOLLOW-UPS:
   - Occasionally ask if they want more info on pricing, location, or if they'd like to schedule a visit.
6. DO NOT explicitly ask 'Is there anything else I can assist you with' in every turn; only use it to transition when a user seems finished with a topic.
7. TERMINATION:
- Only use [HANGUP] if the user explicitly says "No" or "No Thank you" to an assistance request like "Is there anything else I can do for you?", or if a site visit is FULLY confirmed with a date and time, after providing a professional closing.
- If the user's input is unclear (like "See" or "Okay"), do NOT hang up. Ask for clarification or ask: "Is there anything else I can assist you with regarding our departments, timings, or fees?"
"""

    def generate_mom(self):
        print(f"\n[*] Appending Lead Summary to {MOM_LOG_FILE}...")
        if not self.chat_history: return
        transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.chat_history])
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"Generate a professional Sales Lead Summary from this real estate call:\n{transcript}"
        try:
            res = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
            mom_content = res.choices[0].message.content
            with open(MOM_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n{'#'*60}\n SALES LEAD RECORD - {timestamp}\n{'#'*60}\n{mom_content}\n\n")
        except Exception as e:
            print(f"MoM Error: {e}")

    def initial_greeting(self):
        time.sleep(0.5)
        greeting = f"Hello, thank you for calling {self.company_info['company_name']}. I am your automated property assistant. How can I help you with your real estate search today?"
        print(f"AI: {greeting}")
        self.tts_queue.put(greeting)

    def sanitize_for_tts(self, text):
        text = re.sub(r'[*_~`#"]', '', text)
        text = text.replace('-', ', ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tts_worker(self):
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text:
                    # Check for hangup tag
                    is_final_msg = "[HANGUP]" in text.upper()
                    clean_text = text.replace("[HANGUP]", "").replace("[hangup]", "").strip()
                    
                    if clean_text:
                        try:
                            self.piper_proc.stdin.write((clean_text + "\n").encode("utf-8"))
                            self.piper_proc.stdin.flush()
                        except (BrokenPipeError, OSError): pass
                    
                    self.tts_queue.task_done()

                    # If this was the hangup message, trigger termination AFTER processing
                    if is_final_msg:
                        self.trigger_hangup()
                        break 
            except queue.Empty: continue

    # NOTE: a second trigger_hangup definition appears later; the later one
    # is more complete, so the earlier version above is left here purely for
    # historical context and will not be called.  The duplicate has been
    # removed in the next patch segment.

    def piper_stdout_reader(self):
        leftover_bytes = b''
        raw_buffer = b''
        while not self.stop_event.is_set():
            try:
                raw = self.piper_proc.stdout.read(4096)
                if not raw:
                    if self.piper_proc.poll() is not None: break
                    time.sleep(0.01); continue
                raw_buffer += raw
                if len(raw_buffer) % 2 != 0:
                    bytes_to_process, raw_buffer = raw_buffer[:-1], raw_buffer[-1:]
                else:
                    bytes_to_process, raw_buffer = raw_buffer, b''
                if not bytes_to_process: continue
                data_s16 = np.frombuffer(bytes_to_process, dtype=np.int16)
                x_old = np.linspace(0, 1, len(data_s16))
                x_new = np.linspace(0, 1, int(len(data_s16) * (AST_RATE / PIPER_RATE)))
                resampled_s16 = np.interp(x_new, x_old, data_s16).astype(np.int16)
                combined = leftover_bytes + resampled_s16.tobytes()
                num_frames = len(combined) // 320
                for i in range(num_frames):
                    self.output_queue.put(combined[i*320 : (i+1)*320])
                leftover_bytes = combined[num_frames * 320:]
            except: break

    def trigger_hangup(self):
        """Gracefully terminate the call.

        This method ensures all pending TTS audio is generated and sent
        before closing the socket and killing the piper subprocess.
        It may be invoked when the dialogue contains a `[HANGUP]` tag or
        when the remote side disconnects.
        """
        # 1. Wait for TTS queue to be processed by tts_worker
        self.tts_queue.join()

        # 2. Wait for audio output queue to be fully sent by sender thread
        print("[*] Draining audio buffers for graceful exit...")
        while not self.output_queue.empty() and not self.stop_event.is_set():
            time.sleep(0.1)

        # 3. Final safety buffer to ensure last chunk plays on client side
        time.sleep(1.0)

        print("[!] AI hanging up.")
        self.stop_event.set()
        try:
            self.conn.sendall(struct.pack(">BH", 0x00, 0)) # Hangup signal
        except:
            pass
        self.cleanup()

    def cleanup(self):
        self.stop_event.set()
        try: self.piper_proc.terminate()
        except: pass
        try: self.conn.shutdown(socket.SHUT_RDWR); self.conn.close()
        except: pass

    def sender(self):
        try:
            while not self.stop_event.is_set():
                try:
                    payload = self.output_queue.get(timeout=0.1)
                    self.conn.sendall(struct.pack(">BH", MSG_AUDIO_8KHZ, len(payload)) + payload)
                    time.sleep(0.019) 
                except queue.Empty: continue
        finally: self.cleanup()

    def receiver(self):
        try:
            while not self.stop_event.is_set():
                header = self.conn.recv(3)
                if not header or len(header) < 3: break
                msg_type, p_len = struct.unpack(">BH", header)
                payload = b''
                while len(payload) < p_len:
                    chunk = self.conn.recv(p_len - len(payload))
                    if not chunk: break
                    payload += chunk
                if msg_type == MSG_AUDIO_8KHZ and not self.is_ai_talking:
                    self.input_queue.put(payload)
        finally: self.cleanup()

    def processor(self):
        vad_iter = VADIterator(vad_model, sampling_rate=AI_RATE, min_silence_duration_ms=300)
        audio_history = []
        vad_buf = np.array([], dtype=np.float32)
        
        if self.company_info.get("company_name") == "Horizon Estates":
            print("[!] WARNING: Running in Fallback Mode. Check if 'real_estate_data.json' exists.")

        is_user_speaking = False

        while not self.stop_event.is_set():
            try:
                chunk = self.input_queue.get(timeout=0.1)
                f32_8k = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                f32_16k = signal.resample(f32_8k, len(f32_8k) * 2)
                vad_buf = np.concatenate([vad_buf, f32_16k])

                while len(vad_buf) >= 512:
                    current, vad_buf = vad_buf[:512], vad_buf[512:]
                    tensor_chunk = torch.from_numpy(current).to(DEVICE)
                    speech_dict = vad_iter(tensor_chunk, return_seconds=True)

                    if speech_dict:
                        if "start" in speech_dict:
                            is_user_speaking = True
                            audio_history = [] 
                            print("[VAD] Speech started...")
                        elif "end" in speech_dict:
                            is_user_speaking = False
                            print("[VAD] Speech ended. Processing...")
                            
                            if len(audio_history) > 15: 
                                self.is_ai_talking = True
                                full_audio = np.concatenate(audio_history)
                                audio_history = []
                                segments, _ = whisper_model.transcribe(full_audio, beam_size=1)
                                user_text = " ".join([s.text for s in segments]).strip()
                                
                                if user_text:
                                    print(f"User: {user_text}")
                                    self.chat_history.append({"role": "user", "content": user_text})
                                    stream = groq_client.chat.completions.create(
                                        model="llama-3.3-70b-versatile", 
                                        messages=[{"role": "system", "content": self.system_prompt}] + self.chat_history[-15:],
                                        stream=True 
                                    )
                                    full_response = ""
                                    sentence_buffer = ""
                                    for chunk_res in stream:
                                        token = chunk_res.choices[0].delta.content
                                        if token:
                                            sentence_buffer += token
                                            full_response += token
                                            if any(punct in sentence_buffer for punct in ".?!:"):
                                                clean_chunk = re.sub(r'\[HANGUP\]', '', sentence_buffer, flags=re.IGNORECASE).strip()
                                                if clean_chunk:
                                                    self.tts_queue.put(clean_chunk)
                                                sentence_buffer = ""
                                    
                                    if sentence_buffer.strip():
                                        self.tts_queue.put(sentence_buffer.strip())

                                    print(f"AI: {full_response.replace('[HANGUP]', '').strip()}")
                                    self.chat_history.append({"role": "assistant", "content": full_response})
                                    
                                    if "[HANGUP]" in full_response.upper():
                                        print("[*] AI triggered end of call signal.")
                                        self.tts_queue.put("[HANGUP]")

                                self.is_ai_talking = False
                            
                            audio_history = [] 
                            vad_iter.reset_states()   
                    else:
                        if is_user_speaking:
                            audio_history.append(current)
            except queue.Empty: continue
            except Exception as e:
                print(f"Processor Error: {e}")

# ================= SERVER =================
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 9092))
    server.listen(1)
    print(f"Real Estate CPaaS Active. Listening for calls...")
    try:
        conn, addr = server.accept()
        handler = CallHandler(conn)
        while not handler.stop_event.is_set(): time.sleep(0.5)
        handler.generate_mom()
    except KeyboardInterrupt: pass
    finally: server.close()

if __name__ == "__main__":
    start_server()