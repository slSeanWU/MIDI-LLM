# MIDI-LLM

### 🎸 [Live Demo](https://midi-llm-demo.vercel.app) | 🤗 [Model](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) | 📑 Paper (coming soon)

- Shih-Lun Wu, Yoon Kim, and Cheng-Zhi Anna Huang.  
  "**MIDI-LLM: Adapting Large Language Models for Text-to-MIDI Music Generation**."  
  NeurIPS AI4Music Workshop, 2025.

Built on **Llama 3.2 (1B)** with an extended vocabulary for MIDI tokens.


- **[Setup](#setup)**
- **[Inference (Generation) Usage](#inference-generation-usage)**
- **[Example Prompts](#example-prompts)**
- **[Training Guidelines](#training-guidelines)**
- **[Citation](#citation)**

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

## Inference (Generation) Usage

**IMPORTANT**: We provide two inference backends with different trade-offs:
- **vLLM** (`generate_vllm.py`): Faster token generation but more complex setup and longer initialization. **Recommended for batch inference (multiple prompts) or interactive sessions.**
- **Transformers** (`generate_transformers.py`): Simpler setup and faster initialization, but slower generation. **Recommended for quick single-prompt testing.**

Both scripts share the same arguments (except for `--fp8` quantization, which only works in vLLM) and output format.

### Example 1: Single prompt (use transformers)
```bash
python generate_transformers.py \
    --prompt "A cheerful rock song with bright electric guitars"
```
Outputs 4 MIDIs (and synthesized MP3s) conditioned on the same prompt by default.

### Example 2: Batch generation from file (use vLLM)

```bash
python generate_vllm.py \
    --prompts_file assets/example_prompts.txt \
    --fp8 \
    --no-synthesize
```
- [`assets/example_prompts.txt`](assets/example_prompts.txt) contains 4 example prompts (one per line).
- `--fp8` performs FP8 quantization for faster inference.
- `--no-synthesize` skips audio synthesis (outputs MIDI only).

### Example 3: Interactive mode (use vLLM)

```bash
python generate_vllm.py \
    --interactive \
    --output_root generations_interactive/ \
    --n_outputs 1
```
Loads the model once, then lets you enter prompts continuously. Press Enter with an empty prompt to exit.

- Outputs will be stored under `generations_interactive/`
- `--n_outputs 1` generates only 1 output for each prompt

### More options
See full options for either script with:
```bash
python generate_transformers.py --help # or
python generate_vllm.py --help
```

### Inference output structure
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

## Example Prompts

Here are some example prompts to get you started. The model can work with both detailed descriptions similar to what's seen at training, and creative free-form prompts.

### In-Domain Examples (from validation set)

<details>
<summary><b>Example 1: Rock with pop influence</b></summary>

```
A melodic and energetic rock song with a touch of pop influence, featuring synth 
strings, piano, distortion guitar, synth voice, and drums, all contributing to a 
blend of happy and dark moods. Set in the key of A minor with a 4/4 time signature, 
this fast-paced track showcases a chord progression of Bm, Cmaj7, and Gmaj7.
```

</details>

<details>
<summary><b>Example 2: Classical soundtrack</b></summary>

```
A slow and relaxing classical piece featuring a church organ and French horn, likely 
to be used as a soundtrack in a dramatic or emotional film. Written in A minor and 4/4 
time. The chord progression of E7, Am, and E contributes to the piece's sentimental 
atmosphere.
```

</details>

### Creative Custom Prompts

<details>
<summary><b>Example 3: Road trip song</b></summary>

```
An energetic and motivating pop song you love to hear on a long road trip.
```

</details>

<details>
<summary><b>Example 4: Sunday picnic jazz</b></summary>

```
Upbeat and playful jazz music with lively saxophones, like you're going out on a 
Sunday picnic.
```

</details>

## Training Guidelines

We provide high-level guidance for researchers interested in training their own models. If there is sufficient interest from the community, we will consider releasing the full data processing and training pipeline.

<details>
<summary><b>Data Preparation</b></summary>

1. **Collect MIDI data**: E.g., download the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
2. **Tokenize MIDI files**: Use the [Anticipation](https://github.com/jthickstun/anticipation/) library to convert MIDI files to token sequences
3. **Collect text prompts**: Obtain text descriptions for your MIDI files (e.g., [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) in our use case)
4. **Match text-MIDI examples**: Ensure you can map each text prompt to its corresponding MIDI file

**Note**: The 896 LakhMIDI IDs used for evaluation in our paper are available in [`assets/evaluation_set_lakh_ids.txt`](assets/evaluation_set_lakh_ids.txt).

</details>

<details>
<summary><b>Training Process</b></summary>

1. **Create training dataloader**: Write a PyTorch [Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to generate paired text-MIDI training examples
2. **Setup environment**: Install and configure [Accelerate](https://huggingface.co/docs/accelerate/en/basic_tutorials/install)
3. **Start training**: Use the [HuggingFace Trainer](https://huggingface.co/learn/llm-course/en/chapter3/3) with our pretrained model at [slseanwu/MIDI-LLM_Llama-3.2-1B](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) as the starting point
4. **Optional optimizations**:
   - Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) for memory and speed improvements
   - See [multi-GPU training guide](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch) for distributed training

</details>

## Citation

If you find our repo and model useful, please cite our research as
```bibtex
@inproceedings{wu2025midillm,
  title={{MIDI-LLM}: Adapting large language models for text-to-{MIDI} music generation},
  author={Wu, Shih-Lun and Kim, Yoon and Huang, Cheng-Zhi Anna},
  booktitle={Proc. NeurIPS AI4Music Workshop},
  year={2025}
}
```
