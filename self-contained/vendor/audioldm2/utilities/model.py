import torch

import audioldm2.hifigan as hifigan


def get_vocoder_config():
    return {
        "resblock": "1",
        "num_gpus": 6,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [5, 4, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "upsample_initial_channel": 1024,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "segment_size": 8192,
        "num_mels": 64,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 1024,
        "sampling_rate": 16000,
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
        "num_workers": 4,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1,
        },
    }

def get_vocoder_config_48k():
    return {
        "resblock": "1",
        "num_gpus": 8,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,

        "upsample_rates": [6,5,4,2,2],
        "upsample_kernel_sizes": [12,10,8,4,4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3,7,11,15],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5], [1,3,5]],

        "segment_size": 15360,
        "num_mels": 256,
        "n_fft": 2048,
        "hop_size": 480,
        "win_size": 2048,

        "sampling_rate": 48000,

        "fmin": 20,
        "fmax": 24000,
        "fmax_for_loss": None,

        "num_workers": 8,

        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:18273",
            "world_size": 1
        }
    }


def get_available_checkpoint_keys(model, ckpt):
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if "_orig_mod." in key:
            new_state_dict["generator"][key.replace("_orig_mod.", "")] = state_dict[
                "generator"
            ][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict


def get_vocoder(config, device, mel_bins):
    name = "HiFi-GAN"
    speaker = ""
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if(mel_bins == 64):
            config = get_vocoder_config()
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator_old(config)
            # print("Load hifigan/g_01080000")
            # ckpt = torch.load(os.path.join(ROOT, "hifigan/g_01080000"))
            # ckpt = torch.load(os.path.join(ROOT, "hifigan/g_00660000"))
            # ckpt = torch_version_orig_mod_remove(ckpt)
            # vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
        else:
            config = get_vocoder_config_48k()
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator_old(config)
            # print("Load hifigan/g_01080000")
            # ckpt = torch.load(os.path.join(ROOT, "hifigan/g_01080000"))
            # ckpt = torch.load(os.path.join(ROOT, "hifigan/g_00660000"))
            # ckpt = torch_version_orig_mod_remove(ckpt)
            # vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs
