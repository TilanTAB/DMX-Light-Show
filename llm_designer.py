import os
import json
import time
import logging
import requests
from dotenv import load_dotenv

# Load .env before any os.getenv() calls
load_dotenv()

# Set up comprehensive logging to file
logging.basicConfig(
    filename='dmx_ai.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level boto3 client cache — created once on first Bedrock call
_bedrock_client = None


def _extract_json_string(raw_text):
    """Strip markdown code fences if present. No-op for clean JSON.
    Needed because Claude models sometimes wrap JSON in ```json ... ``` blocks
    even when instructed not to.
    """
    text = raw_text.strip()
    if text.startswith("```") and "\n" in text:
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _call_azure(system_message, prompt):
    """Send a chat completion request to Azure OpenAI.
    Returns the raw content string. Raises on HTTP or network errors.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        # B3 NOTE: temperature/max_tokens/top_p removed — this Azure deployment
        # (GPT-5 Nano) rejects them as unsupported. Model uses its own defaults.
    }

    logger.info(f"Sending prompt to Azure (deployment: {deployment_name})...")
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        logger.error(f"Azure returned {response.status_code}: {response.text}")
        raise RuntimeError(f"Azure API Error {response.status_code}: {response.text}")

    json_data = response.json()
    logger.info("Azure API response received successfully.")
    return json_data["choices"][0]["message"]["content"]


def _call_bedrock(system_message, prompt):
    """Send a chat completion request to Amazon Bedrock via the Converse API.
    The Converse API is model-agnostic — works for Claude, Titan, and Nova.
    Returns the raw content string. Raises on errors.
    """
    global _bedrock_client
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        # Re-raise as ImportError (not RuntimeError) so the retry loop can
        # distinguish a missing package (permanent) from a transient network error.
        raise ImportError(
            "boto3 is required for Bedrock support. Install it with: pip install boto3"
        )

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0")

    # Cache the client — boto3.client() loads service model JSON and resolves
    # credentials; creating it on every retry attempt wastes ~100-200ms each time.
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(read_timeout=120, retries={"max_attempts": 0}),
        )

    logger.info(f"Sending prompt to Bedrock (model: {model_id}, region: {region})...")

    response = _bedrock_client.converse(
        modelId=model_id,
        system=[{"text": system_message}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4096},
    )

    logger.info("Bedrock API response received successfully.")
    return response["output"]["message"]["content"][0]["text"]


def _build_section_descriptions(sections):
    """B1 FIX: Convert enriched section data (from A1) into natural language
    prose that the LLM can reason about. LLMs understand language far better
    than numeric tables — this produces dramatically better behavior assignments.
    
    Example output:
      Section 1: Intro/Outro (0s-16s) — balanced, sparse beat, 0% vocals, low energy
      Section 2: Chorus/Drop (16s-48s) — bass-heavy, fast beats, 20% vocals, HIGH energy
    """
    if not sections:
        return "No sections detected."
    
    lines = []
    # Find max energy for relative comparison
    max_energy = max((s.get("avg_energy", 0.001) for s in sections), default=0.001)
    
    for i, sec in enumerate(sections):
        start = sec.get("start", 0)
        end = sec.get("end", 0)
        sec_type = sec.get("type", "Unknown")
        character = sec.get("character", "balanced")
        
        # Relative energy label
        rel_energy = sec.get("avg_energy", 0) / max(max_energy, 0.001)
        if rel_energy < 0.35:
            energy_label = "very low energy"
        elif rel_energy < 0.6:
            energy_label = "low-medium energy"
        elif rel_energy < 0.85:
            energy_label = "medium-high energy"
        else:
            energy_label = "HIGH energy"
        
        # Beat density description
        beat_d = sec.get("avg_beat_density", 0)
        if beat_d < 2:
            beat_label = "sparse/no clear beat"
        elif beat_d < 5:
            beat_label = "steady beat"
        elif beat_d < 8:
            beat_label = "fast beats"
        else:
            beat_label = "very fast/dense beats"
        
        # Vocal description
        vocal_pct = sec.get("vocal_pct", 0)
        if vocal_pct > 70:
            vocal_label = f"strong vocals ({vocal_pct}%)"
        elif vocal_pct > 30:
            vocal_label = f"some vocals ({vocal_pct}%)"
        else:
            vocal_label = f"instrumental ({vocal_pct}% vocal)"
        
        duration = round(end - start, 1)
        lines.append(
            f"  Section {i+1}: {sec_type} ({start:.0f}s-{end:.0f}s, {duration}s long) "
            f"— {character}, {beat_label}, {vocal_label}, {energy_label}"
        )
    
    return "\n".join(lines)

def get_gpt_lighting_plan(audio_features):
    """
    Sends audio telemetry to the configured LLM provider to generate a custom
    JSON lighting plan. Provider is selected via LLM_PROVIDER env var:
      "azure"   — Azure OpenAI (default, backward-compatible)
      "bedrock" — Amazon Bedrock (Claude, Titan, or Nova via Converse API)
    """
    provider = os.getenv("LLM_PROVIDER", "azure").lower().strip()

    # Validate credentials for the chosen provider
    if provider == "bedrock":
        # Fast-fail before building the 2000-token prompt if boto3 isn't installed
        try:
            import boto3  # noqa: F401
        except ImportError:
            print("[-] boto3 is required for Bedrock. Install: pip install boto3")
            return None
        if not os.getenv("AWS_DEFAULT_REGION"):
            print("[-] Missing AWS_DEFAULT_REGION in .env file for Bedrock.")
            return None
        print(f"\n[+] Sending Rhythm Telemetry to Bedrock ({os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-sonnet-4-20250514-v1:0')})...")
    else:
        # Default: azure
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not all([endpoint, api_key, api_version, deployment_name]):
            print("[-] Missing Azure credentials in .env file.")
            return None
        print(f"\n[+] Sending Rhythm Telemetry to Azure ({deployment_name})...")

    # B1 FIX: Replace raw spectral timeline (40 rows of numbers, ~2000 tokens)
    # with prose summaries the LLM can actually reason about (~300 tokens).
    # LLMs are terrible at numerical table reasoning but excellent at prose.
    sections = audio_features.get("structural_sections", [])
    section_descriptions = _build_section_descriptions(sections)

    # Build a compact telemetry payload — now with enriched sections (A1)
    # and no raw timeline (B1). Saves ~1700 tokens per request.
    compact_telemetry = {
        "song_metrics": audio_features.get("song_metrics", {}),
        "structural_sections": sections,
        "events": audio_features.get("events", {}),
    }

    prompt = f"""You are an elite professional concert lighting director with 20+ years of experience 
designing DMX shows for festivals like Tomorrowland, Ultra, and Coachella.

=== AUDIO TELEMETRY ===
{json.dumps(compact_telemetry, indent=2)}

=== SECTION CHARACTER SUMMARY ===
{section_descriptions}

=== YOUR TASK ===
Analyze the ENTIRE song structure first. Identify the emotional arc: 
where does tension build? Where does it release? Where are the quiet moments?
Then assign lighting behaviors that CREATE A NARRATIVE — not just reactive blinking.

=== INDUSTRY LIGHTING PHILOSOPHY ===
A great light show follows these principles:
1. CONTRAST creates impact. A drop only feels massive if the buildup was restrained.
2. SILENCE is a tool. Blackout before a drop makes the flash 10x more powerful.
3. WHITE LIGHT on bass/kick is the #1 most impactful technique. Use it generously.
4. NEVER strobe during vocals. The audience needs to SEE the performer.
5. Color temperature tells a story: warm (amber/pink) = intimate, cool (blue/cyan) = epic.
6. Each section should feel DIFFERENT from its neighbors. Same color twice in a row = lazy.
7. Buildups should RAMP — start dim and restrained, end bright and chaotic.
8. Drops should EXPLODE — instant white blast → full color within 0.5 seconds.

=== SECTION-TO-BEHAVIOR GUIDE ===
Analyze the telemetry and pick the BEST behavior for each section:

"slow_breathe" — Gentle sinusoidal fade between two colors. 
  USE FOR: Intros, outros, breakdowns, ambient sections.
  FEEL: Dreamy, floating. Like breathing underwater.

"static_wash" — Hold one color steady with subtle volume-linked breathing. Snare → brief white accent.
  USE FOR: Verses with vocals. Let the singer be the focus.
  FEEL: Calm, understated. The room glows but doesn't distract.

"beat_reactive" — Kick→color_1, Snare→color_2, Kick→WHITE blast. Standard concert mode.
  USE FOR: Choruses, medium-energy sections with clear beat.
  FEEL: Alive, punchy. Audience feels every beat.

"bass_white_blast" — WHITE LED explodes on every kick. Colored wash underneath from mids.
  USE FOR: Heavy choruses, bass-driven sections, EDM drops.
  FEEL: Raw power. The white punch cuts through everything.

"blackout_punch" — Complete DARKNESS between beats. Instant color+white flash on each hit.
  USE FOR: Drops, intense moments, dramatic reveals after breakdowns.
  FEEL: Aggressive, cinematic. Like lightning in a dark sky.

"color_chase" — Alternates between color_1 and color_2 on every beat.
  USE FOR: High-energy dance sections, fun/upbeat moments.
  FEEL: Dynamic, playful. Two-tone flickering.

"fast_pulse" — Rapid beat-synced pulses. Kick→color_1, Snare→color_2, hi-hat→white shimmer.
  USE FOR: Energetic sections where beat is fast and relentless.
  FEEL: Intense but not chaotic. Controlled energy.

"buildup_ramp" — Brightness ramps 20%→100% over the section duration. Strobe increases.
  USE FOR: Tension-building moments before a big drop.
  FEEL: Anticipation. The audience knows something big is coming.

"strobe_blast" — Full white strobe at maximum rate. Use VERY SPARINGLY (max 4-8 seconds).
  USE FOR: The absolute peak moment of the song. The biggest drop.
  FEEL: Sensory overload. Use once or twice per song MAX.

"rainbow_sweep" — Slow hue rotation through the spectrum. Beats cause brightness pulse.
  USE FOR: Festive moments, euphoric buildups, celebratory sections.
  FEEL: Colorful, joyful. Like a festival sunset.

=== AMBIENT / CHILL BEHAVIORS ===
Use these for quiet sections, breakdowns, lo-fi, acoustic, and ambient music.

"ocean_drift" — Slow wave undulations. Two overlapping sine waves create gentle color swelling. Bass adds subtle warmth but never flashes.
  USE FOR: Ambient intros, zen breakdowns, downtempo sections.
  FEEL: Dreamy, oceanic. Like light underwater.

"candlelight" — Warm amber flicker with organic noise. Volume modulates flicker intensity.
  USE FOR: Acoustic sets, ballads, intimate moments.
  FEEL: Warm, intimate. Like a room lit by candles.

"sunset_fade" — Slow cinematic crossfade from color_1 to color_2 over the section. No beat reaction at all.
  USE FOR: Long vocal sections, outros, emotional transitions.
  FEEL: Cinematic, emotional. A slow color journey.

"aurora_shimmer" — Three independent sine waves on R, G, B create a slowly evolving color field that never repeats.
  USE FOR: Ambient electronic, chillwave, post-rock, atmospheric sections.
  FEEL: Ethereal, mesmerizing. Like northern lights.

"abyssal_bloom" — Near-black drifting floor with rare, slow teal blooms that swell and dissolve, and rarer white glints. Bass makes blooms more frequent and slightly brighter, but never lets it get busy.
  USE FOR: Deep breakdowns, the quietest intros/outros, meditative or minimal-techno sections.
  FEEL: Vast, restrained, oceanic-depths. Darkness as the canvas.

=== NARRATIVE ARC TEMPLATE ===
A well-designed show follows this emotional curve:
  Intro: ocean_drift or aurora_shimmer (gentle, mysterious)
  → Verse 1: candlelight or static_wash (warm, focused)
  → Pre-Chorus: buildup_ramp (rising tension!)
  → Chorus 1: bass_white_blast (IMPACT! White on kicks!)
  → Post-Chorus: beat_reactive (ride the energy)
  → Verse 2: sunset_fade (contrast: pull back cinematically)
  → Hook: buildup_ramp (tension again)
  → Chorus 2: blackout_punch (even MORE intense than Chorus 1!)
  → Bridge/Breakdown: ocean_drift or aurora_shimmer (dramatic reset)
  → Final Chorus: strobe_blast 2s → bass_white_blast (ultimate climax)
  → Outro: sunset_fade (fade to darkness)

=== REQUIRED JSON FORMAT ===
{{
  "show_name": "descriptive name",
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
      "energy_level": 2,
      "behavior": "slow_breathe",
      "mood": "warm|cool|neon|euphoric|dark"
    }},
    ...one cue per structural section...
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

=== COLOR RULES ===
- color_1 = KICK DRUM color (flashes on bass hits + drives WHITE channel)
- color_2 = SNARE/CLAP color (flashes on transients)
- They MUST have HIGH CONTRAST: red+cyan, purple+yellow, blue+white, magenta+green.
- Avoid boring primaries. Use curated palettes: deep ocean blue [10,30,180], volcanic amber [255,100,20], 
  electric violet [130,0,255], arctic cyan [0,220,255], neon magenta [255,0,120].
- Each section should use a DIFFERENT palette from its neighbors.

=== ENERGY_LEVEL (1-10) ===
Controls how aggressively the lights react to drum onsets:
1-3: Subtle. Only big transients register. (intros, outros, breakdowns)
4-6: Moderate. Most beats visible. (verses, bridges)
7-8: Aggressive. Every beat punches hard. (choruses)
9-10: Maximum. Every onset triggers full flash. (drops ONLY)

=== HARD CONSTRAINTS (NEVER VIOLATE) ===
1. NEVER use the same behavior for two adjacent cues. Contrast is mandatory.
2. NEVER use strobe_blast for more than 6 seconds total across the entire show.
3. NEVER set energy_level > 7 for sections with vocal_pct > 50% (vocals need calm lighting).
4. start_time of cue N+1 MUST EXACTLY EQUAL end_time of cue N. No gaps, no overlaps.
5. The FIRST cue MUST start at time 0.0.
6. The LAST cue MUST end at the song's total_duration_seconds.
7. master_dimmer_percent for Intro/Outro sections MUST be <= 40.
8. Use buildup_ramp at least once before the highest-energy Chorus/Drop section.
9. Use the section "character" field from the telemetry to guide behavior selection:
   - character="bass-heavy" → bass_white_blast or blackout_punch
   - character="vocal-driven" → static_wash or beat_reactive (never strobe!)
   - character="bright/atmospheric" → rainbow_sweep or slow_breathe
   - character="rhythmic" → fast_pulse or color_chase
10. Assign a "mood" to every cue (warm|cool|neon|euphoric|dark). Adjacent cues SHOULD use contrasting moods. The mood + color_1 seed the lighting palette, so pick an evocative color_1 per section.

CRITICAL: Generate EXACTLY one cue for EVERY structural section from the telemetry.
The "phrases" array is a backward-compatibility summary."""

    # B4 FIX: Stronger system message with step-by-step reasoning framework
    system_message = """You are a professional concert lighting director who outputs ONLY valid JSON lighting plans for DMX fixtures.

REASONING PROCESS (follow this order):
1. Read the song metrics (BPM, energy, duration) to understand the overall vibe.
2. Read the SECTION CHARACTER SUMMARY to understand each section's sonic identity.
3. For each section, pick a behavior that matches its character AND contrasts with neighbors.
4. Choose color palettes that tell an emotional story (warm intro → cool drop → warm outro).
5. Set energy_level proportional to the section's relative energy and beat density.
6. Validate: no gaps between cues, full song coverage, no adjacent duplicate behaviors.

Output ONLY the JSON object. No explanation, no markdown, no comments."""

    # Retry with backoff for transient failures
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"LLM request via {provider} (attempt {attempt+1})...")

            if provider == "bedrock":
                raw_content = _call_bedrock(system_message, prompt)
            else:
                raw_content = _call_azure(system_message, prompt)

            # _extract_json_string strips markdown fences Claude sometimes adds
            content = _extract_json_string(raw_content)
            lighting_plan = json.loads(content)

            # I2 FIX: Validate and repair AI response before returning
            lighting_plan = _validate_and_repair_plan(lighting_plan)

            return lighting_plan

        except json.JSONDecodeError as e:
            # Malformed JSON is never retried — same response on retry
            logger.error(f"AI returned invalid JSON: {e}")
            print(f"[-] AI response was not valid JSON: {e}")
            return None
        except ImportError as e:
            # Missing package — retry won't help, fail immediately
            logger.error(f"Missing dependency: {e}")
            print(f"[-] Missing dependency: {e}")
            return None
        except Exception as e:
            # Covers: requests.exceptions.Timeout (Azure), botocore ReadTimeoutError
            # (Bedrock), RuntimeError (HTTP 4xx/5xx), NoCredentialsError, etc.
            # All are retried — permanent failures (bad creds, 401) will fail
            # consistently and exhaust retries, which is acceptable.
            logger.error(f"LLM call failed ({provider}, attempt {attempt+1}): {e}", exc_info=True)
            print(f"[-] LLM Error ({provider}): {e}")
            if attempt < max_retries:
                print(f"[~] Retrying in {5 * (attempt+1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            return None


VALID_BEHAVIORS = {
    "slow_breathe", "fast_pulse", "color_chase", "strobe_blast",
    "static_wash", "rainbow_sweep", "buildup_ramp", "instant_flash",
    "beat_reactive", "bass_white_blast", "blackout_punch",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer", "abyssal_bloom",
}


def _validate_rgb(value, default=(255, 255, 255)):
    """Ensure value is a valid [R, G, B] list with 0-255 ints."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return list(default)
    try:
        return [max(0, min(255, int(v))) for v in value]
    except (ValueError, TypeError):
        return list(default)


def _default_mood_for_energy(energy):
    """Fallback mood when the LLM omits one. Matches loopback's energy->mood map."""
    e = int(energy or 0)
    if e <= 3:
        return "warm"
    if e <= 6:
        return "cool"
    if e <= 8:
        return "neon"
    return "euphoric"


def _validate_and_repair_plan(plan):
    """
    Validate AI lighting plan structure. Repair common LLM mistakes
    instead of crashing at runtime. Log everything we fix.
    """
    if not isinstance(plan, dict):
        logger.error("AI plan is not a dict — returning empty plan")
        return {"show_name": "Unknown", "cues": [], "phrases": []}

    # Validate cues
    cues = plan.get("cues", [])
    if not isinstance(cues, list):
        cues = []

    valid_cues = []
    for i, cue in enumerate(cues):
        if not isinstance(cue, dict):
            logger.warning(f"Cue {i} is not a dict — skipping")
            continue
        # Repair colors
        cue["color_1"] = _validate_rgb(cue.get("color_1"))
        cue["color_2"] = _validate_rgb(cue.get("color_2"))
        # Repair behavior
        behavior = cue.get("behavior", "beat_reactive")
        if behavior not in VALID_BEHAVIORS:
            logger.warning(f"Cue {i}: unknown behavior '{behavior}' → beat_reactive")
            cue["behavior"] = "beat_reactive"
        # Defaults for missing fields
        cue.setdefault("start_time", 0)
        cue.setdefault("end_time", cue["start_time"] + 10)
        cue.setdefault("energy_level", 5)
        cue.setdefault("master_dimmer_percent", 80)
        # P2-2 FIX: Clamp to valid ranges — LLM might return energy_level: 50
        cue["energy_level"] = max(1, min(10, int(cue["energy_level"])))
        cue["master_dimmer_percent"] = max(0, min(100, int(cue["master_dimmer_percent"])))
        # mood seeds the VarietyEngine palette family; default from energy if absent
        if not cue.get("mood"):
            cue["mood"] = _default_mood_for_energy(cue["energy_level"])
        cue.setdefault("strobe_allowed", False)
        cue.setdefault("fade_speed_seconds", 1.0)
        cue.setdefault("section_name", f"Section {i+1}")
        valid_cues.append(cue)

    plan["cues"] = valid_cues

    # C3 FIX: Validate timeline coverage — no gaps, no overlaps.
    # Without this, the LLM might leave 5-10 second dark patches between cues.
    if len(valid_cues) >= 2:
        # Sort by start_time to ensure correct ordering
        valid_cues.sort(key=lambda c: c.get("start_time", 0))
        
        # Ensure first cue starts at 0.0
        if valid_cues[0].get("start_time", 0) > 0.5:
            logger.warning(f"C3: First cue starts at {valid_cues[0]['start_time']}s, forcing to 0.0")
            valid_cues[0]["start_time"] = 0.0
        
        # Close gaps: set each cue's start_time = previous cue's end_time
        for i in range(1, len(valid_cues)):
            prev_end = valid_cues[i - 1].get("end_time", 0)
            curr_start = valid_cues[i].get("start_time", 0)
            gap = curr_start - prev_end
            if gap > 0.5:  # More than 0.5s gap = needs fixing
                logger.warning(f"C3: {gap:.1f}s gap before cue {i} ('{valid_cues[i].get('section_name','')}') — closing")
                valid_cues[i]["start_time"] = prev_end
            elif gap < -0.5:  # Overlap
                logger.warning(f"C3: {-gap:.1f}s overlap at cue {i} — trimming")
                valid_cues[i]["start_time"] = prev_end

    # Validate phrases
    phrases = plan.get("phrases", [])
    if isinstance(phrases, list):
        for p in phrases:
            if isinstance(p, dict):
                p["color_1"] = _validate_rgb(p.get("color_1"))
                p["color_2"] = _validate_rgb(p.get("color_2"))

    logger.info(f"Validated plan: {len(valid_cues)} cues, {len(phrases)} phrases")
    return plan

if __name__ == "__main__":
    # Test script execution — uses LLM_PROVIDER from .env (default: azure)
    _provider = os.getenv("LLM_PROVIDER", "azure")
    print(f"Testing {_provider.upper()} connection (make sure .env is filled out!)")

    sample_rhythm_data = {
        "bpm": 142.5,
        "bass_density": "extremely high",
        "transient_speed": "aggressive",
        "mid_frequency_presence": "low",
        "estimated_vibe": "hard techno / heavy bass"
    }

    plan = get_gpt_lighting_plan(sample_rhythm_data)
    if plan:
        print(f"\n[+] Success! {_provider.upper()} generated the following DMX sequence:")
        print(json.dumps(plan, indent=2))
