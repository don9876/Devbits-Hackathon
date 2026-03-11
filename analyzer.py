import os
import time
import json
import shutil
import glob
from sarvamai import SarvamAI
from groq import Groq

# ================= CONFIG =================
GROQ_API_KEY = "GROQ_API_KEY_HERE"  # Set your Groq API key here
SARVAM_API_KEY = "SARVAM_API_KEY_HERE"  # Set your Sarvam API key here

LOG_FILE = "Real_Estate_Call_Logs.txt"
TEMP_OUTPUT_DIR = "./sarvam_batch_results"

def run_complete_analysis(audio_path):
    if not os.path.exists(audio_path):
        print(f"[X] File not found: {audio_path}")
        return

    # Prepare temp directory
    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
    os.makedirs(TEMP_OUTPUT_DIR)

    print(f"\n--- Batch Processing: {os.path.basename(audio_path)} ---")
    client_sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    try:
        # 1. Job Creation & Execution
        print("[*] Creating Job (Saaras v3)...")
        job = client_sarvam.speech_to_text_job.create_job(
            model="saaras:v3",
            language_code="hi-IN",
            mode="translate"
        )
        job.upload_files(file_paths=[audio_path])
        job.start()

        print("[*] Waiting for completion...")
        job.wait_until_complete()
        
        print(f"[*] Downloading results to {TEMP_OUTPUT_DIR}...")
        job.download_outputs(output_dir=TEMP_OUTPUT_DIR)

        # 2. Flexible File Finding
        # Sarvam may name it 'testaudio.mp3.json' or 'testaudio.json'
        # We search for any .json file in the output folder
        json_files = glob.glob(os.path.join(TEMP_OUTPUT_DIR, "*.json"))
        
        if not json_files:
            print(f"[X] No JSON result found in {TEMP_OUTPUT_DIR}")
            return
            
        result_file = json_files[0] # Take the first matching result
        print(f"[✓] Reading result from: {os.path.basename(result_file)}")

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            transcript = data.get('transcript', '')

        if not transcript:
            print("[X] Transcript field is empty.")
            return

        print(f"[✓] Transcript extracted ({len(transcript)} chars)")

    except Exception as e:
        print(f"[X] Sarvam Error: {e}")
        return

    # 3. Groq MOM Generation
    print("[*] Generating MOM with Groq...")
    client_groq = Groq(api_key=GROQ_API_KEY)
    
    system_prompt = (
        "You are a professional Real Estate Assistant. Create a Minutes of Meeting (MOM) "
        "from the provided transcript. Extract: Customer Identity, Requirement, Budget, "
        "and Next Steps. Format clearly; use 'Not Mentioned' for missing info."
    )

    try:
        completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {transcript}"}
            ],
            model="llama-3.3-70b-versatile",
        )
        
        mom_result = completion.choices[0].message.content
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*60}\nFILE: {audio_path}\n{mom_result}\n")
        
        print(f"\n--- FINAL MOM ---\n{mom_result}")
        shutil.rmtree(TEMP_OUTPUT_DIR) # Final cleanup

    except Exception as e:
        print(f"[X] Groq Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_complete_analysis(sys.argv[1])