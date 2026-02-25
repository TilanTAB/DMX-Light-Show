import os
import json
import logging
import requests
from dotenv import load_dotenv

# Set up comprehensive logging to file
logging.basicConfig(
    filename='dmx_ai.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load variables from .env
load_dotenv()

def get_gpt_lighting_plan(audio_features):
    """
    Sends LIVE rhythm telemetry (BPM, bass density, energy) to GPT-5 Nano 
    to generate a custom JSON lighting plan.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([endpoint, api_key, api_version, deployment_name]):
        print("[-] Missing Azure credentials in .env file.")
        return None

    print(f"\n[+] Sending Rhythm Telemetry to Azure GPT-5 Nano...")

    # Trim the spectral timeline to avoid exceeding token limits
    # Send every 3rd data point (6-second resolution) instead of every 2s
    trimmed_timeline = audio_features.get("spectral_timeline", [])[::3]
    
    # Build a compact telemetry payload for the LLM
    compact_telemetry = {
        "song_metrics": audio_features.get("song_metrics", {}),
        "structural_sections": audio_features.get("structural_sections", []),
        "events": audio_features.get("events", {}),
        "spectral_samples": trimmed_timeline[:40],  # Cap at 40 data points
    }

    prompt = f"""You are an elite professional concert lighting director with 20+ years of experience 
designing DMX shows for festivals like Tomorrowland, Ultra, and Coachella. You understand music structure 
deeply: verses, choruses, build-ups, drops, breakdowns, bridges, intros, and outros.

I am giving you precise audio telemetry data from a song. Use this data to generate a 
TIMESTAMPED lighting cue list that a DMX controller will execute in perfect sync with the music.

=== AUDIO TELEMETRY ===
{json.dumps(compact_telemetry, indent=2)}

=== YOUR TASK ===
Generate a JSON lighting show that maps EACH detected structural section to specific lighting behaviors.

INDUSTRY BEST PRACTICES TO FOLLOW:
1. INTRO/OUTRO: Slow, ambient washes. Deep blues/purples. Low master dimmer. No strobes. Slow fade (2-4s).
2. VERSE: Gentle breathing colors synced to vocal presence. Warm tones. Medium brightness. 
3. PRE-CHORUS/BUILD-UP: Rising tension! Colors shift cooler. Strobe speed increases gradually. 
   Brightness ramps from 40% to 80%. This MUST feel like anticipation building.
4. CHORUS/DROP: MAXIMUM IMPACT. Pure white blast for 0.5s, then saturated vibrant colors. 
   Full brightness. Strobes allowed. Fast attack (instant on), medium decay.
5. BREAKDOWN: Strip back to single color wash. Slow breathing. Moody. Purple/teal tones.
6. VOCAL SECTIONS: When vocals are present, use warmer tones (amber, pink, soft white). 
   Smooth transitions. Never strobe during clean vocals.
7. DRUM-HEAVY SECTIONS: When beat_density is high, use complementary color pairs that alternate 
   on each beat. Snappy attack/decay for punchy feel.
8. PALETTE RULE: Never use more than 2-3 colors simultaneously. Each section should have a 
   deliberate, curated palette that contrasts with adjacent sections.

=== REQUIRED JSON OUTPUT FORMAT ===
{{
  "show_name": "descriptive name for this show",
  "cues": [
    {{
      "section_name": "Intro",
      "start_time": 0.0,
      "end_time": 16.0,
      "color_1": [R, G, B],
      "color_2": [R, G, B],
      "master_dimmer_percent": 30,
      "fade_speed_seconds": 3.0,
      "strobe_allowed": false,
      "strobe_speed": 0,
      "energy_level": 2,
      "behavior": "slow_breathe",
      "notes": "Deep ambient blue wash, gentle pulse"
    }},
    ...more cues for EVERY detected section...
  ],
  "phrases": [
    {{
      "name": "section name",
      "color_1": [R, G, B],
      "color_2": [R, G, B],
      "strobe_allowed": false,
      "energy": 3
    }}
  ]
}}

BEHAVIOR OPTIONS: "slow_breathe", "fast_pulse", "color_chase", "strobe_blast", 
"static_wash", "rainbow_sweep", "buildup_ramp", "instant_flash"

=== DRUM-REACTIVE ENGINE ===
IMPORTANT: The DMX controller is BEAT-REACTIVE. It detects drum hits in real-time using FFT onset detection:
- "color_1" = KICK DRUM color (flashes on every bass drum hit)
- "color_2" = SNARE DRUM color (flashes on every snare/clap hit)
- "energy_level" (1-10) = Controls how AGGRESSIVELY lights react to drums:
  * 1-3: Subtle — only big hits register. Good for intros, outros, breakdowns.
  * 4-6: Moderate — most beats visible. Good for verses.
  * 7-8: Aggressive — every beat punches hard. Good for choruses.
  * 9-10: Maximum — violent strobing on every hit. Use ONLY for drops.

COLOR PAIRING RULE: color_1 and color_2 MUST have HIGH CONTRAST so kick and snare 
produce visually DIFFERENT flashes. Bad: red + orange (too similar). Good: red + cyan, 
purple + yellow, blue + white. The audience should SEE the difference between kick and snare.

CRITICAL: Generate a cue for EVERY structural section. Use the exact start/end timestamps 
from the telemetry. The "phrases" array is a simplified summary for backward compatibility.
Be creative with colors — avoid boring primaries. Use rich, curated palettes."""
    
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a professional JSON data generator for DMX lighting shows."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    # Retry with backoff for transient failures
    import time as _time
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Sending prompt to Azure via python requests (Model: {deployment_name}, attempt {attempt+1})...")
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code != 200:
                logger.error(f"Azure returned status code {response.status_code}: {response.text}")
                print(f"[-] Azure API Error {response.status_code}: {response.text}")
                if attempt < max_retries and response.status_code >= 500:
                    print(f"[~] Retrying in {5 * (attempt+1)}s...")
                    _time.sleep(5 * (attempt + 1))
                    continue
                return None
                
            json_data = response.json()
            logger.info("Azure API response received successfully.")
            
            content = json_data['choices'][0]['message']['content']
            lighting_plan = json.loads(content)
            
            return lighting_plan

        except requests.exceptions.Timeout:
            logger.error(f"Azure API request timed out (attempt {attempt+1}).")
            print(f"[-] GPT-5 Nano Timeout (attempt {attempt+1}/{max_retries+1}).")
            if attempt < max_retries:
                print(f"[~] Retrying in {5 * (attempt+1)}s...")
                _time.sleep(5 * (attempt + 1))
                continue
            print("[-] All retries exhausted.")
            return None
        except Exception as e:
            logger.error(f"Azure API Call Failed: {e}", exc_info=True)
            print(f"[-] GPT-5 Nano Connection Error: {e}")
            return None

if __name__ == "__main__":
    # Test script execution
    print("Testing Azure connection with Live Rhythm Data (Make sure .env is filled out!)")
    
    # Example telemetry that our music_light.py script will calculate live:
    sample_rhythm_data = {
        "bpm": 142.5,
        "bass_density": "extremely high",
        "transient_speed": "aggressive",
        "mid_frequency_presence": "low",
        "estimated_vibe": "hard techno / heavy bass"
    }
    
    plan = get_gpt_lighting_plan(sample_rhythm_data)
    if plan:
        print("\n[+] Success! GPT generated the following DMX sequence based on rhythm:")
        print(json.dumps(plan, indent=2))
