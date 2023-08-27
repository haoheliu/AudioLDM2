import os
import re

import yaml
import torch
import torchaudio

import audioldm2.latent_diffusion.modules.phoneme_encoder.text as text
from audioldm2.latent_diffusion.models.ddpm import LatentDiffusion
from audioldm2.latent_diffusion.util import get_vits_phoneme_ids_no_padding
from audioldm2.utils import default_audioldm_config, download_checkpoint
import os

# CACHE_DIR = os.getenv(
#     "AUDIOLDM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/audioldm2")
# )

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def text2phoneme(data):
    return text._clean_text(re.sub(r'<.*?>', '', data), ["english_cleaners2"])

def text_to_filename(text):
    return text.replace(" ", "_").replace("'", "_").replace('"', "_")

def extract_kaldi_fbank_feature(waveform, sampling_rate, log_mel_spec):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]

def make_batch_for_text_to_audio(text, transcription="", waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if(transcription):
        transcription = text2phoneme(transcription)
    transcription = [transcription] * batchsize

    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")

    if fbank is None:
        fbank = torch.zeros(
            (batchsize, 1024, 64)
        )  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize

    stft = torch.zeros((batchsize, 1024, 512))  # Not used
    phonemes = get_vits_phoneme_ids_no_padding(transcription)

    if waveform is None:
        waveform = torch.zeros((batchsize, 160000))  # Not used
        ta_kaldi_fbank = torch.zeros((batchsize, 1024, 128))
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, 16000, fbank)

    batch = {
        "text": text,  # list
        "fname": [text_to_filename(t) for t in text],  # list
        "waveform": waveform,
        "stft": stft,
        "log_mel_spec": fbank,
        "ta_kaldi_fbank": ta_kaldi_fbank,
    }
    batch.update(phonemes)
    return batch


def round_up_duration(duration):
    return int(round(duration / 2.5) + 1) * 2.5


# def split_clap_weight_to_pth(checkpoint):
#     if os.path.exists(os.path.join(CACHE_DIR, "clap.pth")):
#         return
#     print("Constructing the weight for the CLAP model.")
#     include_keys = "cond_stage_models.0.cond_stage_models.0.model."
#     new_state_dict = {}
#     for each in checkpoint["state_dict"].keys():
#         if include_keys in each:
#             new_state_dict[each.replace(include_keys, "module.")] = checkpoint[
#                 "state_dict"
#             ][each]
#     torch.save({"state_dict": new_state_dict}, os.path.join(CACHE_DIR, "clap.pth"))


def build_model(ckpt_path=None, config=None, device=None, model_name="audioldm2-full"):

    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print("Loading AudioLDM-2: %s" % model_name)
    print("Loading model on %s" % device)

    ckpt_path = download_checkpoint(model_name)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else: 
        config = default_audioldm_config(model_name)

    # # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    # config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)

    latent_diffusion.load_state_dict(checkpoint["state_dict"])
    
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)
    
    return latent_diffusion

def text_to_audio(
    latent_diffusion,
    text,
    transcription="",
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=3.5,
    n_candidate_gen_per_text=3,
    latent_t_per_second=25.6,
    config=None,
):

    seed_everything(int(seed))
    waveform = None

    batch = make_batch_for_text_to_audio(text, transcription=transcription, waveform=waveform, batchsize=batchsize)

    latent_diffusion.latent_t_size = int(duration * latent_t_per_second)

    with torch.no_grad():
        waveform = latent_diffusion.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_candidate_gen_per_text,
            duration=duration,
        )

    return waveform
