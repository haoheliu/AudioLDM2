# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")

import hashlib
import os

import requests
from tqdm import tqdm

URL_MAP = {
    "vggishish_lpaps": "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggishish16.pt",
    "vggishish_mean_std_melspec_10s_22050hz": "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/train_means_stds_melspec_10s_22050hz.txt",
    "melception": "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melception-21-05-10T09-28-40.pt",
}

CKPT_MAP = {
    "vggishish_lpaps": "vggishish16.pt",
    "vggishish_mean_std_melspec_10s_22050hz": "train_means_stds_melspec_10s_22050hz.txt",
    "melception": "melception-21-05-10T09-28-40.pt",
}

MD5_MAP = {
    "vggishish_lpaps": "197040c524a07ccacf7715d7080a80bd",
    "vggishish_mean_std_melspec_10s_22050hz": "f449c6fd0e248936c16f6d22492bb625",
    "melception": "a71a41041e945b457c7d3d814bbcf72d",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data


def read_json(dataset_json_file):
    with open(dataset_json_file, "r") as fp:
        data_json = json.load(fp)
    return data_json["data"]


def copy_test_subset_data(metadata, testset_copy_target_path):
    # metadata = read_json(testset_metadata)
    os.makedirs(testset_copy_target_path, exist_ok=True)
    if len(os.listdir(testset_copy_target_path)) == len(metadata):
        return
    else:
        # delete files in folder testset_copy_target_path
        for file in os.listdir(testset_copy_target_path):
            try:
                os.remove(os.path.join(testset_copy_target_path, file))
            except Exception as e:
                print(e)

    print("Copying test subset data to {}".format(testset_copy_target_path))
    for each in tqdm(metadata):
        cmd = "cp {} {}".format(each["wav"], os.path.join(testset_copy_target_path))
        os.system(cmd)


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


def get_restore_step(path):
    checkpoints = os.listdir(path)
    if os.path.exists(os.path.join(path, "final.ckpt")):
        return "final.ckpt", 0
    elif not os.path.exists(os.path.join(path, "last.ckpt")):
        steps = [int(x.split(".ckpt")[0].split("step=")[1]) for x in checkpoints]
        return checkpoints[np.argmax(steps)], np.max(steps)
    else:
        steps = []
        for x in checkpoints:
            if "last" in x:
                if "-v" not in x:
                    fname = "last.ckpt"
                else:
                    this_version = int(x.split(".ckpt")[0].split("-v")[1])
                    steps.append(this_version)
                    if len(steps) == 0 or this_version > np.max(steps):
                        fname = "last-v%s.ckpt" % this_version
        return fname, 0


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(logger, step=None, fig=None, audio=None, sampling_rate=22050, tag=""):
    # if losses is not None:
    #     logger.add_scalar("Loss/total_loss", losses[0], step)
    #     logger.add_scalar("Loss/mel_loss", losses[1], step)
    #     logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
    #     logger.add_scalar("Loss/pitch_loss", losses[3], step)
    #     logger.add_scalar("Loss/energy_loss", losses[4], step)
    #     logger.add_scalar("Loss/duration_loss", losses[5], step)
    #     if(len(losses) > 6):
    #         logger.add_scalar("Loss/disc_loss", losses[6], step)
    #         logger.add_scalar("Loss/fmap_loss", losses[7], step)
    #         logger.add_scalar("Loss/r_loss", losses[8], step)
    #         logger.add_scalar("Loss/g_loss", losses[9], step)
    #         logger.add_scalar("Loss/gen_loss", losses[10], step)
    #         logger.add_scalar("Loss/diff_loss", losses[11], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        audio = audio / (max(abs(audio)) * 1.1)
        logger.add_audio(
            tag,
            audio,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample_val(
    targets, predictions, vocoder, model_config, preprocess_config
):
    index = np.random.choice(list(np.arange(targets[6].size(0))))

    basename = targets[0][index]
    src_len = predictions[8][index].item()
    mel_len = predictions[9][index].item()
    mel_target = targets[6][index, :mel_len].detach().transpose(0, 1)

    mel_prediction = predictions[0][index, :mel_len].detach().transpose(0, 1)
    postnet_mel_prediction = predictions[1][index, :mel_len].detach().transpose(0, 1)
    duration = targets[11][index, :src_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = predictions[2][index, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = predictions[2][index, :mel_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = predictions[3][index, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = predictions[3][index, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    # from datetime import datetime
    # now = datetime.now()
    # current_time = now.strftime("%D:%H:%M:%S")
    # np.save(("mel_pred_%s.npy" % current_time).replace("/","-"), mel_prediction.cpu().numpy())
    # np.save(("postnet_mel_prediction_%s.npy" % current_time).replace("/","-"), postnet_mel_prediction.cpu().numpy())
    # np.save(("mel_target_%s.npy" % current_time).replace("/","-"), mel_target.cpu().numpy())

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (postnet_mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        [
            "Raw mel spectrogram prediction",
            "Postnet mel prediction",
            "Ground-Truth Spectrogram",
        ],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            postnet_mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_one_sample(mel_input, mel_prediction, labels, vocoder):
    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_input.permute(0, 2, 1),
            vocoder,
        )
        wav_prediction = vocoder_infer(
            mel_prediction.permute(0, 2, 1),
            vocoder,
        )
    else:
        wav_reconstruction = wav_prediction = None

    return wav_reconstruction, wav_prediction


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):
    # (diff_output, diff_loss, latent_loss) = diffusion

    basenames = targets[0]

    for i in range(len(predictions[1])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        # diff_output = diff_output[i, :mel_len].detach().transpose(0, 1)
        # duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            # pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            # energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram by PostNet"],
        )
        # np.save("{}_postnet.npy".format(basename), mel_prediction.cpu().numpy())
        plt.savefig(os.path.join(path, "{}_postnet_2.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower", aspect="auto")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
