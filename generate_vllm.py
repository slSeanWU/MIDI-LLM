#!/usr/bin/env python3
"""
This script generates MIDI files from text prompts using the MIDI-LLM model with vLLM backend.
vLLM provides faster inference compared to standard HuggingFace model.generate() mixin.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
import tqdm
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

# Import helper functions and constants
from midi_llm.utils import (
    save_generation,
    synthesize_midi_to_audio,
    has_excessive_notes_at_any_time,
    AMT_GPT2_BOS_ID,
    LLAMA_VOCAB_SIZE,
    LLAMA_MODEL_NAME,
    ALLOWED_TOKEN_IDS,
    SYNTHESIS_AVAILABLE,
)

# Default generation parameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.98
DEFAULT_MAX_TOKENS = 2046
DEFAULT_N_OUTPUTS = 4  # give more outputs for variability


def prepare_vllm_model(
    model_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_outputs: int,
    do_fp8_quantization: bool = False,
    gpu_memory_utilization: float = 0.9
) -> tuple:
    """
    Initialize vLLM model and sampling parameters.
    
    Args:
        model_path: Path to model checkpoint
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        n_outputs: Number of outputs per prompt
        do_fp8_quantization: Whether to use FP8 quantization
        gpu_memory_utilization: Fraction of GPU memory to use
        
    Returns:
        Tuple of (model, sampling_params)
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n_outputs,
        max_tokens=max_tokens,
        allowed_token_ids=ALLOWED_TOKEN_IDS,
    )
    
    print(f"\n{'='*70}")
    print("Model Configuration")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Quantization: {'FP8' if do_fp8_quantization else 'None (BF16)'}")
    print(f"GPU memory utilization: {gpu_memory_utilization:.1%}")
    print(f"\nSampling Parameters:")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Outputs per prompt: {n_outputs}")
    print(f"{'='*70}\n")
    
    model = LLM(
        model=model_path,
        quantization="fp8" if do_fp8_quantization else None,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    print(f"✓ Model loaded successfully\n")
    
    return model, sampling_params


def generate_from_prompts(
    model: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    sampling_params: SamplingParams,
    output_dir: Path,
    soundfont_path: Optional[str] = None,
    synthesize: bool = False,
    system_prompt: Optional[str] = None
) -> dict:
    """
    Generate MIDI from text prompts and save results.
    
    Args:
        model: vLLM model
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts
        sampling_params: vLLM sampling parameters
        output_dir: Base output directory (timestamped subdirs will be created inside)
        soundfont_path: Path to SoundFont file
        synthesize: Whether to synthesize to audio
        system_prompt: Optional system prompt prefix
        
    Returns:
        Dictionary with generation statistics and output files
    """
    # Default system prompt
    if system_prompt is None:
        system_prompt = "You are a world-class composer. Please compose some music according to the following description: "
    
    stats = {
        "total_prompts": len(prompts),
        "successful_generations": 0,
        "failed_generations": 0,
        "generation_times": [],
        "output_files": []  # Track all generated files
    }
    
    for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Generating")):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")
        
        # Create output directory for this prompt with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_output_dir = output_dir / f"{timestamp}_prompt_{idx+1}"
        
        # Prepare full prompt
        # add space to the end of each prompt to match training
        full_prompt = system_prompt + prompt + " "
        
        # Tokenize
        llama_input = tokenizer(full_prompt, padding=False)
        input_ids = llama_input["input_ids"]
        
        # Add MIDI BOS token (AMT_GPT2_BOS_ID in extended vocab)
        input_ids.append(AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE)
        
        # Generate
        start_time = time.time()
        vllm_input = [TokensPrompt(prompt_token_ids=input_ids)]
        outputs = model.generate(vllm_input, sampling_params)
        generation_time = time.time() - start_time
        
        if idx > 0:  # Skip first generation for timing (warmup)
            stats["generation_times"].append(generation_time)
        
        print(f"Generation time: {generation_time:.2f}s")
        
        # Save all outputs for this prompt
        n_outputs = len(outputs[0].outputs)
        successful_outputs = 0
        prompt_files = []
        
        for output_idx in range(n_outputs):
            # Extract tokens and shift back to MIDI vocab range
            token_ids = outputs[0].outputs[output_idx].token_ids
            midi_tokens = [t - LLAMA_VOCAB_SIZE for t in token_ids]
            
            # Save generation
            success = save_generation(
                tokens=midi_tokens,
                prompt=prompt,
                output_dir=prompt_output_dir,
                generation_idx=output_idx + 1,
                soundfont_path=soundfont_path,
                synthesize=synthesize
            )
            
            if success:
                successful_outputs += 1
                # Track output files
                midi_file = prompt_output_dir / f"gen_{output_idx + 1}.mid"
                prompt_files.append(str(midi_file))
                if synthesize and soundfont_path:
                    mp3_file = prompt_output_dir / f"gen_{output_idx + 1}.mp3"
                    if mp3_file.exists():
                        prompt_files.append(str(mp3_file))
        
        print(f"Successfully saved {successful_outputs}/{n_outputs} outputs")
        stats["successful_generations"] += successful_outputs
        stats["failed_generations"] += (n_outputs - successful_outputs)
        stats["output_files"].extend(prompt_files)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI files from text prompts using MIDI-LLM with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from a single prompt (there will be 4 outputs by default)
  python generate_vllm.py --model path/to/checkpoint \\
      --prompt "A cheerful piano melody"
  
  # Generate single output without synthesis
  python generate_vllm.py --model path/to/checkpoint \\
      --prompt "A relaxing jazz piece" \\
      --n_outputs 1 \\
      --no-synthesize
  
  # Interactive mode (with initial prompt)
  python generate_vllm.py --model path/to/checkpoint \\
      --prompt "A cheerful melody" \\
      --interactive
  
  # Interactive-only mode (no initial prompt)
  python generate_vllm.py --model path/to/checkpoint \\
      --interactive
  
  # Generate from prompts file with FP8 quantization
  python generate_vllm.py --model path/to/checkpoint \\
      --prompts_file prompts.txt \\
      --fp8 \\
      --temperature 1.0 \\
      --top_p 0.98
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MIDI-LLM model checkpoint"
    )
    
    # Input arguments (not required if using --interactive only)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single text prompt for generation"
    )
    input_group.add_argument(
        "--prompts_file",
        type=str,
        help="Path to file containing prompts (one per line)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_root",
        type=str,
        default="./generated_outputs",
        help="Root directory for outputs (timestamped subdirs will be created inside, default: ./generated_outputs)"
    )
    parser.add_argument(
        "--n_outputs",
        type=int,
        default=DEFAULT_N_OUTPUTS,
        help=f"Number of outputs to generate per prompt (default: {DEFAULT_N_OUTPUTS})"
    )
    
    # Synthesis arguments
    parser.add_argument(
        "--no-synthesize",
        dest="synthesize",
        action="store_false",
        help="Skip audio synthesis (only generate MIDI files)"
    )
    parser.set_defaults(synthesize=True)
    parser.add_argument(
        "--soundfont",
        type=str,
        default="./soundfonts/FluidR3_GM/FluidR3_GM.sf2",
        help="Path to SoundFont file for synthesis (default: ./soundfonts/FluidR3_GM/FluidR3_GM.sf2)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling threshold (default: {DEFAULT_TOP_P})"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    
    # Model arguments
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use FP8 quantization for faster inference (requires compatible GPU)"
    )
    parser.add_argument(
        "--gpu_memory",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: $HF_HOME or ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after initial generation (keep generating until empty prompt)"
    )
    
    args = parser.parse_args()
    
    # Validate that either prompts are provided or interactive mode is enabled
    if not args.prompt and not args.prompts_file and not args.interactive:
        parser.error("Either --prompt, --prompts_file, or --interactive must be specified")
    
    # Load prompts (if provided)
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    # Check synthesis requirements
    if args.synthesize:
        soundfont_path = Path(args.soundfont)
        if not soundfont_path.exists():
            print(f"Error: SoundFont not found at {soundfont_path}")
            print("Please download a SoundFont or disable synthesis with --no-synthesize")
            sys.exit(1)
        
        if not SYNTHESIS_AVAILABLE:
            print("Warning: Audio synthesis libraries not available.")
            print("Synthesis will be skipped. Install dependencies:")
            print("  conda install conda-forge::fluidsynth conda-forge::ffmpeg")
            print("  pip install midi2audio librosa soundfile")
            args.synthesize = False
    
    # Create output root directory with timestamp
    output_root = Path(args.output_root)
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_root / session_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=args.cache_dir,
        pad_token="<|eot_id|>",
    )
    
    # Load model
    model, sampling_params = prepare_vllm_model(
        model_path=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_outputs=args.n_outputs,
        do_fp8_quantization=args.fp8,
        gpu_memory_utilization=args.gpu_memory
    )
    
    # Generate from initial prompts (if provided)
    if prompts:
        print(f"Starting generation for {len(prompts)} prompt(s)...\n")
        start_time = time.time()
        
        stats = generate_from_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            sampling_params=sampling_params,
            output_dir=output_dir,
            soundfont_path=args.soundfont if args.synthesize else None,
            synthesize=args.synthesize
        )
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*70}")
        print("Generation Summary")
        print(f"{'='*70}")
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Successful generations: {stats['successful_generations']}")
        print(f"Failed generations: {stats['failed_generations']}")
        print(f"Total time: {total_time:.2f}s")
        
        if stats['generation_times']:
            avg_time = sum(stats['generation_times']) / len(stats['generation_times'])
            print(f"Average generation time: {avg_time:.2f}s (excluding warmup)")
        
        print(f"\nOutputs saved to: {output_dir.absolute()}")
        
        # Print generated files
        if stats['output_files']:
            print(f"\nGenerated files:")
            for file_path in stats['output_files']:
                file_type = "🎵 MIDI" if file_path.endswith('.mid') else "🎧 Audio"
                print(f"  {file_type}: {file_path}")
        
        print(f"{'='*70}\n")
        
        # Save stats to JSON
        stats_file = output_dir / "generation_stats.json"
        with open(stats_file, "w") as f:
            json.dump({
                **stats,
                "total_time": total_time,
                "average_time": sum(stats['generation_times']) / len(stats['generation_times']) if stats['generation_times'] else 0,
                "config": {
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "n_outputs": args.n_outputs,
                    "fp8": args.fp8,
                }
            }, f, indent=2)
    else:
        print(f"No initial prompts provided. Starting in interactive mode...\n")
    
    # Interactive mode
    if args.interactive:
        print(f"\n{'='*70}")
        print("Interactive Mode")
        print(f"{'='*70}")
        print("Enter prompts to generate more MIDI files.")
        print("Press Enter with empty prompt to exit.\n")
        
        while True:
            try:
                # Get user input
                user_prompt = input("Prompt: ").strip()
                
                # Exit if empty
                if not user_prompt:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                # Generate from the new prompt
                print()
                interactive_stats = generate_from_prompts(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[user_prompt],
                    sampling_params=sampling_params,
                    output_dir=output_dir,
                    soundfont_path=args.soundfont if args.synthesize else None,
                    synthesize=args.synthesize
                )
                
                # Print mini summary
                print(f"\n✓ Generated {interactive_stats['successful_generations']}/{args.n_outputs} outputs")
                if interactive_stats['generation_times']:
                    print(f"  Generation time: {interactive_stats['generation_times'][0]:.2f}s")
                
                # Print file paths
                if interactive_stats['output_files']:
                    for file_path in interactive_stats['output_files']:
                        file_type = "🎵" if file_path.endswith('.mid') else "🎧"
                        print(f"  {file_type} {file_path}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting interactive mode.")
                break
            except EOFError:
                print("\n\nExiting interactive mode.")
                break


if __name__ == "__main__":
    main()

