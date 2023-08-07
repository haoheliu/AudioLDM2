# AudioLDM 2

[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audioldm2/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm2-text2audio-text2music)  

This repo currently support Text-to-Audio Generation (including Music)

<hr>

## Web APP

1. Prepare running environment
```shell
conda create -n audioldm python=3.8; conda activate audioldm
pip3 install audioldm
git clone https://github.com/haoheliu/AudioLDM2; cd AudioLDM2
```
2. Start the web application (powered by Gradio)
```shell
python3 app.py
```
3. A link will be printed out. Click the link to open the browser and play.

## Commandline Usage
Prepare running environment
```shell
# Optional
conda create -n audioldm python=3.8; conda activate audioldm
# Install AudioLDM
pip3 install git+https://github.com/haoheliu/AudioLDM2.git
```

- Generate based on a text prompt

```shell
audioldm2 -t "Musical constellations twinkling in the night sky, forming a cosmic melody."
```

- Generate based on a list of text

```shell
audioldm2 -tl batch.lst
```

## Random Seed Matters

Sometimes model may not perform well (sounds wired or low quality) when changing into a different hardware. In this case, please adjust the random seed and find the optimal one for your hardware. 
```shell
audioldm2 --seed 1234 -t "Musical constellations twinkling in the night sky, forming a cosmic melody."
```

## Pretrained Models

You can choose model checkpoint by setting up "model_name":

```shell
audioldm2 --model_name "audioldm2-full-large-650k" -t "Musical constellations twinkling in the night sky, forming a cosmic melody."
```

We have three checkpoints you can choose for now:
1. **audioldm2-full** (default): This checkpoint can perform both sound effect and music generation. 
2. **audioldm2-music-665k**: This checkpoint is specialized on music generation. 
3. **audioldm2-full-large-650k**: This checkpoint is the larger version of audioldm2-full. 

Evaluation result on AudioCaps and MusicCaps evaluation set:

Coming soon.

## Other options
```shell
  usage: audioldm2 [-h] [-t TEXT] [-tl TEXT_LIST] [-s SAVE_PATH] [--model_name {audioldm2-full,audioldm2-music-665k,audioldm2-full-large-650k}] [-b BATCHSIZE] [--ddim_steps DDIM_STEPS] [-gs GUIDANCE_SCALE]
                  [-n N_CANDIDATE_GEN_PER_TEXT] [--seed SEED]

  optional arguments:
    -h, --help            show this help message and exit
    -t TEXT, --text TEXT  Text prompt to the model for audio generation
    -tl TEXT_LIST, --text_list TEXT_LIST
                          A file that contains text prompt to the model for audio generation
    -s SAVE_PATH, --save_path SAVE_PATH
                          The path to save model output
    --model_name {audioldm2-full,audioldm2-music-665k,audioldm2-full-large-650k}
                          The checkpoint you gonna use
    -b BATCHSIZE, --batchsize BATCHSIZE
                          Generate how many samples at the same time
    --ddim_steps DDIM_STEPS
                          The sampling step for DDIM
    -gs GUIDANCE_SCALE, --guidance_scale GUIDANCE_SCALE
                          Guidance scale (Large => better quality and relavancy to text; Small => better diversity)
    -n N_CANDIDATE_GEN_PER_TEXT, --n_candidate_gen_per_text N_CANDIDATE_GEN_PER_TEXT
                          Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with
                          heavier computation
    --seed SEED           Change this value (any integer number) will lead to a different generation result.
```

## Cite this work
If you found this tool useful, please consider citing

```bibtex
    AudioLDM 2 paper coming soon
```

```bibtex
@article{liu2023audioldm,
  title={AudioLDM: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2301.12503},
  year={2023}
}
```

