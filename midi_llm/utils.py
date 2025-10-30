"""
Utility functions for MIDI-LLM.

This module contains helper functions for audio synthesis, MIDI conversion,
and other supporting operations. Users can safely skip this file when learning
the codebase - start with generate_vllm.py or train.py instead.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch

# Core dependency - required
try:
    from anticipation.convert import events_to_midi
except ImportError:
    print("Error: anticipation package not found. Please install it for MIDI conversion.")
    print("Install with: pip install anticipation")
    sys.exit(1)

# Optional dependencies for audio synthesis
SYNTHESIS_AVAILABLE = False
LOUDNESS_NORM_AVAILABLE = False
try:
    import midi2audio
    import librosa
    import librosa.effects
    import soundfile as sf
    SYNTHESIS_AVAILABLE = True
    
    # Optional loudness normalization
    try:
        import pyloudnorm as pyln
        LOUDNESS_NORM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

AMT_GPT2_BOS_ID = 55026
LLAMA_VOCAB_SIZE = 128256
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# MIDI tokens are in the extended vocabulary range
ALLOWED_TOKEN_IDS = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID))


# ============================================================================
# Validation
# ============================================================================

def has_excessive_notes_at_any_time(
    tokens: Union[torch.Tensor, List[int]], 
    max_notes_per_time: int = 64
) -> bool:
    """
    Check if generated MIDI has excessive simultaneous notes at any time point.
    
    This validation helps filter out invalid or unrealistic generations that have
    too many notes playing at once, which can indicate a failure mode.
    
    Args:
        tokens: Token sequence (torch.Tensor or list of ints)
        max_notes_per_time: Maximum allowed notes at any single time point
        
    Returns:
        True if excessive notes detected, False otherwise
    """
    # Convert to tensor if needed
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens)
    
    # Extract time tokens (every 3rd token in the sequence: time, duration, note)
    times = tokens[::3]
    
    # Use torch.bincount for efficient counting
    # bincount returns counts for indices 0 to max_value
    counts = torch.bincount(times)
    
    # Check if any time has more than max_notes_per_time notes
    return torch.any(counts > max_notes_per_time).item()


# ============================================================================
# Audio Synthesis
# ============================================================================

def synthesize_midi_to_audio(
    midi_path: str, 
    soundfont_path: str,
    save_mp3: bool = True,
    samplerate: Optional[int] = None,
    target_loudness: float = -18.0
) -> bool:
    """
    Synthesize MIDI file to audio (WAV/MP3) using FluidSynth with loudness normalization.
    
    Args:
        midi_path: Path to MIDI file
        soundfont_path: Path to SoundFont (.sf2) file
        save_mp3: If True, convert to MP3 and delete WAV
        samplerate: Optional sample rate for audio
        target_loudness: Target loudness in LUFS (default: -14.0, Spotify standard)
        
    Returns:
        True if successful, False otherwise
    """
    if not SYNTHESIS_AVAILABLE:
        print("Warning: Audio synthesis libraries not available. Skipping synthesis.")
        print("Install with: conda install conda-forge::fluidsynth conda-forge::ffmpeg")
        print("              pip install midi2audio librosa soundfile pyloudnorm")
        return False
    
    try:
        wav_path = midi_path.replace(".mid", ".wav")
        
        # Initialize FluidSynth
        fs = midi2audio.FluidSynth(soundfont_path)
        if samplerate is not None:
            fs.sample_rate = samplerate
        
        # Synthesize MIDI to WAV
        fs.midi_to_audio(midi_path, wav_path)
        
        # Load and trim silence from audio
        wav, sr = librosa.load(wav_path)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        
        # Apply loudness normalization
        if LOUDNESS_NORM_AVAILABLE:
            try:
                # Measure the loudness
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(wav)
                
                # Normalize to target loudness
                wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                
                # Prevent clipping
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
            except Exception as e:
                print(f"Warning: Loudness normalization failed: {e}")
        
        # Write normalized audio
        sf.write(wav_path, wav, sr)
        
        if save_mp3:
            # Convert WAV to MP3 using ffmpeg
            mp3_path = midi_path.replace(".mid", ".mp3")
            if samplerate is None:
                cmd = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 {mp3_path} -y >/dev/null 2>&1"
            else:
                cmd = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 -ar {samplerate} {mp3_path} -y >/dev/null 2>&1"
            
            os.system(cmd)
            
            # Remove WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)
        
        return True
    
    except Exception as e:
        print(f"Error synthesizing MIDI to audio: {e}")
        return False


# ============================================================================
# MIDI Generation and Saving
# ============================================================================

def save_generation(
    tokens: List[int],
    prompt: str,
    output_dir: Path,
    generation_idx: int,
    soundfont_path: Optional[str] = None,
    synthesize: bool = False,
    validate: bool = True
) -> bool:
    """
    Save generated tokens as MIDI file (and optionally audio).
    
    Args:
        tokens: List of generated token IDs (already shifted from LLAMA vocab)
        prompt: Original text prompt
        output_dir: Directory to save outputs
        generation_idx: Index of this generation (for multiple outputs)
        soundfont_path: Path to SoundFont file for synthesis
        synthesize: Whether to synthesize to audio
        validate: Whether to validate tokens before saving (checks for excessive notes)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate tokens before saving
        if validate:
            if has_excessive_notes_at_any_time(tokens, max_notes_per_time=64):
                print(f"  ✗ Generation {generation_idx}: Failed validation (excessive simultaneous notes)")
                return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompt text
        prompt_file = output_dir / "prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        # Save token sequence
        tokens_file = output_dir / f"gen_{generation_idx}_tokens.txt"
        with open(tokens_file, "w") as f:
            for token in tokens:
                f.write(f"{token}\n")
        
        # Convert tokens to MIDI
        midi_obj = events_to_midi(tokens)
        midi_file = output_dir / f"gen_{generation_idx}.mid"
        midi_obj.save(str(midi_file))
        
        print(f"  ✓ Saved MIDI: {midi_file}")
        
        # Optionally synthesize to audio
        if synthesize and soundfont_path:
            success = synthesize_midi_to_audio(
                str(midi_file),
                soundfont_path,
                save_mp3=True
            )
            if success:
                print(f"  ✓ Synthesized audio: {midi_file.with_suffix('.mp3')}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error saving generation {generation_idx}: {e}")
        return False

