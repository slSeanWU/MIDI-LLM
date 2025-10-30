# MIDI-LLM

Built on **Llama 3.2 (1B)** with an extended vocabulary for MIDI tokens.

## Research Paper
- Shih-Lun Wu, Yoon Kim, and Cheng-Zhi Anna Huang.  
  "**MIDI-LLM: Adapting large language models for text-to-MIDI music generation**."  
  NeurIPS AI4Music Workshop, 2025.

## Setup

- A GPU with 16GB+ VRAM and CUDA 12.x is recommended

- Install [Miniconda / Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

- Create and activate Python 3.11 conda environment
```bash
conda create -n midi-llm python=3.11
conda activate midi-llm
```

- Install packages + download soundfont for MIDI-to-audio synthesis
```bash
# Conda pkgs for audio processing & synthesis
conda install conda-forge::ffmpeg
conda install conda-forge::fluidsynth

# Soundfont (credit -- '@Frank Wen' https://member.keymusician.com/Member/FluidR3_GM/README.html)
wget https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
mkdir -p soundfonts
unzip FluidR3_GM.zip -d ./soundfonts/FluidR3_GM
rm FluidR3_GM.zip
```

- Install [PyTorch](https://pytorch.org/get-started/locally/) with `pip`
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
(Note: this is an example for CUDA 12.6, check PyTorch website if you're on other CUDA versions)

- Check if PyTorch works correctly on CUDA GPU

```bash
python -c "import torch; x = torch.randn(30, 30).cuda(); y = x.clone(); z = torch.mm(x, y); print(f'GPU works correctly, output shape: {z.shape}')"
```

- Install other dependencies
```bash
pip install -r requirements.txt
```

- Verify all installation
```bash
python -c "import torch; from vllm import LLM; from anticipation.convert import events_to_midi; print('Setup successful')"
```

## Run Inference with vLLM
### Example 1: Single prompt
```bash
python generate_vllm.py \
    --model slseanwu/MIDI-LLM_Llama-3.2-1B # will pull from huggingface hub \
    --prompt "A cheerful piano melody"
```
This will output 4 MIDIs (and the synthesized MP3s) conditioned on the same input prompt

### Example 2: Batch generation from file

```bash
python generate_vllm.py \
    --model slseanwu/MIDI-LLM_Llama-3.2-1B \
    --prompts_file some_example_prompts.txt \
    --fp8 \
    --no-synthesize
```
- `some_example_prompts.txt` should contain one prompt per line.
- `--fp8` performs dynamic weight quantization for faster inference.
- `--no-synthesize` skips audio synthesis (i.e., outputs MIDI only).

### Example 3: Interactive mode

```bash
python generate_vllm.py \
    --model slseanwu/MIDI-LLM_Llama-3.2-1B \
    --output_root generations_interactive/ \
    --interactive
```
- Outputs will be saved under `generations_interactive/`
This loads the model once, then lets you enter prompts interactively. Press Enter with empty prompt to exit.

### More options
See full options with:
```bash
python generate_vllm.py --help
```

### Inference Output Structure
```
[output_root]/
└── 2025-10-30_143022/           # Session timestamp
    ├── 20251030_143022_prompt_1/
    │   ├── prompt.txt
    │   ├── gen_1.mid
    │   ├── gen_1.mp3
    │   └── ...
    └── generation_stats.json
```