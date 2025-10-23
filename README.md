# MIDI-LLM
[NeurIPS AI4Music '25] MIDI-LLM: Adapting LLMs for text-to-MIDI music generation.

Built with Llama 3.2 (1B) LLM.

# Setup

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