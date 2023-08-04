import os
import torch
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

CACHE = {
    "get_vits_phoneme_ids": {
        "PAD_LENGTH": 310,
        "_pad": "_",
        "_punctuation": ';:,.!?¡¿—…"«»“” ',
        "_letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "_letters_ipa": "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
        "_special": "♪☎☒☝⚠",
    }
}

CACHE["get_vits_phoneme_ids"]["symbols"] = (
    [CACHE["get_vits_phoneme_ids"]["_pad"]]
    + list(CACHE["get_vits_phoneme_ids"]["_punctuation"])
    + list(CACHE["get_vits_phoneme_ids"]["_letters"])
    + list(CACHE["get_vits_phoneme_ids"]["_letters_ipa"])
    + list(CACHE["get_vits_phoneme_ids"]["_special"])
)
CACHE["get_vits_phoneme_ids"]["_symbol_to_id"] = {
    s: i for i, s in enumerate(CACHE["get_vits_phoneme_ids"]["symbols"])
}


def get_vits_phoneme_ids(config, dl_output, metadata):
    pad_token_id = 0
    pad_length = CACHE["get_vits_phoneme_ids"]["PAD_LENGTH"]
    _symbol_to_id = CACHE["get_vits_phoneme_ids"]["_symbol_to_id"]

    assert (
        "phonemes" in metadata.keys()
    ), "You must provide vits phonemes on using addon get_vits_phoneme_ids"
    clean_text = metadata["phonemes"]
    sequence = []

    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]

    inserted_zero_sequence = [0] * (len(sequence) * 2)
    inserted_zero_sequence[1::2] = sequence
    inserted_zero_sequence = inserted_zero_sequence + [0]

    def _pad_phonemes(phonemes_list):
        return phonemes_list + [pad_token_id] * (pad_length - len(phonemes_list))

    return {"phoneme_idx": torch.LongTensor(_pad_phonemes(inserted_zero_sequence))}


def get_vits_phoneme_ids_no_padding(config, dl_output, metadata):
    pad_token_id = 0
    pad_length = CACHE["get_vits_phoneme_ids"]["PAD_LENGTH"]
    _symbol_to_id = CACHE["get_vits_phoneme_ids"]["_symbol_to_id"]

    assert (
        "phonemes" in metadata.keys()
    ), "You must provide vits phonemes on using addon get_vits_phoneme_ids"
    clean_text = metadata["phonemes"] + "⚠"
    sequence = []

    for symbol in clean_text:
        if symbol not in _symbol_to_id.keys():
            print("%s is not in the vocabulary. %s" % (symbol, clean_text))
            symbol = "_"
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]

    def _pad_phonemes(phonemes_list):
        return phonemes_list + [pad_token_id] * (pad_length - len(phonemes_list))

    sequence = sequence[:pad_length]

    return {"phoneme_idx": torch.LongTensor(_pad_phonemes(sequence))}


def calculate_relative_bandwidth(config, dl_output, metadata):
    assert "stft" in dl_output.keys()

    # The last dimension of the stft feature is the frequency dimension
    freq_dimensions = dl_output["stft"].size(-1)

    freq_energy_dist = torch.sum(dl_output["stft"], dim=0)
    freq_energy_dist = torch.cumsum(freq_energy_dist, dim=0)
    total_energy = freq_energy_dist[-1]

    percentile_5th = total_energy * 0.05
    percentile_95th = total_energy * 0.95

    lower_idx = torch.argmin(torch.abs(percentile_5th - freq_energy_dist))
    higher_idx = torch.argmin(torch.abs(percentile_95th - freq_energy_dist))

    lower_idx = int((lower_idx / freq_dimensions) * 1000)
    higher_idx = int((higher_idx / freq_dimensions) * 1000)

    return {"freq_energy_percentile": torch.LongTensor([lower_idx, higher_idx])}


def calculate_mel_spec_relative_bandwidth_as_extra_channel(config, dl_output, metadata):
    assert "stft" in dl_output.keys()
    linear_mel_spec = torch.exp(torch.clip(dl_output["log_mel_spec"], max=10))

    # The last dimension of the stft feature is the frequency dimension
    freq_dimensions = linear_mel_spec.size(-1)
    freq_energy_dist = torch.sum(linear_mel_spec, dim=0)
    freq_energy_dist = torch.cumsum(freq_energy_dist, dim=0)
    total_energy = freq_energy_dist[-1]

    percentile_5th = total_energy * 0.05
    percentile_95th = total_energy * 0.95

    lower_idx = torch.argmin(torch.abs(percentile_5th - freq_energy_dist))
    higher_idx = torch.argmin(torch.abs(percentile_95th - freq_energy_dist))

    latent_t_size = config["model"]["params"]["latent_t_size"]
    latent_f_size = config["model"]["params"]["latent_f_size"]

    lower_idx = int(latent_f_size * float((lower_idx / freq_dimensions)))
    higher_idx = int(latent_f_size * float((higher_idx / freq_dimensions)))

    bandwidth_condition = torch.zeros((latent_t_size, latent_f_size))
    bandwidth_condition[:, lower_idx:higher_idx] += 1.0

    return {
        "mel_spec_bandwidth_cond_extra_channel": bandwidth_condition,
        "freq_energy_percentile": torch.LongTensor([lower_idx, higher_idx]),
    }


def waveform_rs_48k(config, dl_output, metadata):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]

    if sampling_rate != 48000:
        waveform_48k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=48000
        )
    else:
        waveform_48k = waveform

    return {"waveform_48k": waveform_48k}


def extract_vits_phoneme_and_flant5_text(config, dl_output, metadata):
    assert (
        "phoneme" not in metadata.keys()
    ), "The metadata of speech you use seems belong to fastspeech. Please check dataset_root.json"

    if "phonemes" in metadata.keys():
        new_item = get_vits_phoneme_ids_no_padding(config, dl_output, metadata)
        new_item["text"] = ""  # We assume TTS data does not have text description
    else:
        fake_metadata = {"phonemes": ""}  # Add empty phoneme sequence
        new_item = get_vits_phoneme_ids_no_padding(config, dl_output, fake_metadata)

    return new_item


def extract_fs2_phoneme_and_flant5_text(config, dl_output, metadata):
    if "phoneme" in metadata.keys():
        new_item = extract_fs2_phoneme_g2p_en_feature(config, dl_output, metadata)
        new_item["text"] = ""
    else:
        fake_metadata = {"phoneme": []}
        new_item = extract_fs2_phoneme_g2p_en_feature(config, dl_output, fake_metadata)
    return new_item


def extract_fs2_phoneme_g2p_en_feature(config, dl_output, metadata):
    PAD_LENGTH = 135

    phonemes_lookup_dict = {
        "K": 0,
        "IH2": 1,
        "NG": 2,
        "OW2": 3,
        "AH2": 4,
        "F": 5,
        "AE0": 6,
        "IY0": 7,
        "SH": 8,
        "G": 9,
        "W": 10,
        "UW1": 11,
        "AO2": 12,
        "AW2": 13,
        "UW0": 14,
        "EY2": 15,
        "UW2": 16,
        "AE2": 17,
        "IH0": 18,
        "P": 19,
        "D": 20,
        "ER1": 21,
        "AA1": 22,
        "EH0": 23,
        "UH1": 24,
        "N": 25,
        "V": 26,
        "AY1": 27,
        "EY1": 28,
        "UH2": 29,
        "EH1": 30,
        "L": 31,
        "AA2": 32,
        "R": 33,
        "OY1": 34,
        "Y": 35,
        "ER2": 36,
        "S": 37,
        "AE1": 38,
        "AH1": 39,
        "JH": 40,
        "ER0": 41,
        "EH2": 42,
        "IY2": 43,
        "OY2": 44,
        "AW1": 45,
        "IH1": 46,
        "IY1": 47,
        "OW0": 48,
        "AO0": 49,
        "AY0": 50,
        "EY0": 51,
        "AY2": 52,
        "UH0": 53,
        "M": 54,
        "TH": 55,
        "T": 56,
        "OY0": 57,
        "AW0": 58,
        "DH": 59,
        "Z": 60,
        "spn": 61,
        "AH0": 62,
        "sp": 63,
        "AO1": 64,
        "OW1": 65,
        "ZH": 66,
        "B": 67,
        "AA0": 68,
        "CH": 69,
        "HH": 70,
    }
    pad_token_id = len(phonemes_lookup_dict.keys())

    assert (
        "phoneme" in metadata.keys()
    ), "The dataloader add-on extract_phoneme_g2p_en_feature will output phoneme id, which is not specified in your dataset"

    phonemes = [
        phonemes_lookup_dict[x]
        for x in metadata["phoneme"]
        if (x in phonemes_lookup_dict.keys())
    ]

    if (len(phonemes) / PAD_LENGTH) > 5:
        print(
            "Warning: Phonemes length is too long and is truncated too much! %s"
            % metadata
        )

    phonemes = phonemes[:PAD_LENGTH]

    def _pad_phonemes(phonemes_list):
        return phonemes_list + [pad_token_id] * (PAD_LENGTH - len(phonemes_list))

    return {"phoneme_idx": torch.LongTensor(_pad_phonemes(phonemes))}


def extract_phoneme_g2p_en_feature(config, dl_output, metadata):
    PAD_LENGTH = 250

    phonemes_lookup_dict = {
        " ": 0,
        "AA": 1,
        "AE": 2,
        "AH": 3,
        "AO": 4,
        "AW": 5,
        "AY": 6,
        "B": 7,
        "CH": 8,
        "D": 9,
        "DH": 10,
        "EH": 11,
        "ER": 12,
        "EY": 13,
        "F": 14,
        "G": 15,
        "HH": 16,
        "IH": 17,
        "IY": 18,
        "JH": 19,
        "K": 20,
        "L": 21,
        "M": 22,
        "N": 23,
        "NG": 24,
        "OW": 25,
        "OY": 26,
        "P": 27,
        "R": 28,
        "S": 29,
        "SH": 30,
        "T": 31,
        "TH": 32,
        "UH": 33,
        "UW": 34,
        "V": 35,
        "W": 36,
        "Y": 37,
        "Z": 38,
        "ZH": 39,
    }
    pad_token_id = len(phonemes_lookup_dict.keys())

    assert (
        "phoneme" in metadata.keys()
    ), "The dataloader add-on extract_phoneme_g2p_en_feature will output phoneme id, which is not specified in your dataset"
    phonemes = [
        phonemes_lookup_dict[x]
        for x in metadata["phoneme"]
        if (x in phonemes_lookup_dict.keys())
    ]

    if (len(phonemes) / PAD_LENGTH) > 5:
        print(
            "Warning: Phonemes length is too long and is truncated too much! %s"
            % metadata
        )

    phonemes = phonemes[:PAD_LENGTH]

    def _pad_phonemes(phonemes_list):
        return phonemes_list + [pad_token_id] * (PAD_LENGTH - len(phonemes_list))

    return {"phoneme_idx": torch.LongTensor(_pad_phonemes(phonemes))}


def extract_kaldi_fbank_feature(config, dl_output, metadata):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]
    log_mel_spec_hifigan = dl_output["log_mel_spec"]

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

    TARGET_LEN = log_mel_spec_hifigan.size(0)

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


def extract_kaldi_fbank_feature_32k(config, dl_output, metadata):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]
    log_mel_spec_hifigan = dl_output["log_mel_spec"]

    if sampling_rate != 32000:
        waveform_32k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=32000
        )
    else:
        waveform_32k = waveform

    waveform_32k = waveform_32k - waveform_32k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_32k,
        htk_compat=True,
        sample_frequency=32000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec_hifigan.size(0)

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


# Use the beat and downbeat information as music conditions
def extract_drum_beat(config, dl_output, metadata):
    def visualization(conditional_signal, mel_spectrogram, filename):
        import soundfile as sf

        sf.write(
            os.path.basename(dl_output["fname"]),
            np.array(dl_output["waveform"])[0],
            dl_output["sampling_rate"],
        )
        plt.figure(figsize=(10, 10))

        plt.subplot(211)
        plt.imshow(np.array(conditional_signal).T, aspect="auto")
        plt.title("Conditional Signal")

        plt.subplot(212)
        plt.imshow(np.array(mel_spectrogram).T, aspect="auto")
        plt.title("Mel Spectrogram")

        plt.savefig(filename)
        plt.close()

    assert "sample_rate" in metadata and "beat" in metadata and "downbeat" in metadata

    sampling_rate = metadata["sample_rate"]
    duration = dl_output["duration"]
    # The dataloader segment length before performing torch resampling
    original_segment_length_before_resample = int(sampling_rate * duration)

    random_start_sample = int(dl_output["random_start_sample_in_original_audio_file"])

    # The sample idx for beat and downbeat, relatively to the segmented audio
    beat = [
        x - random_start_sample
        for x in metadata["beat"]
        if (
            x - random_start_sample >= 0
            and x - random_start_sample <= original_segment_length_before_resample
        )
    ]
    downbeat = [
        x - random_start_sample
        for x in metadata["downbeat"]
        if (
            x - random_start_sample >= 0
            and x - random_start_sample <= original_segment_length_before_resample
        )
    ]

    latent_shape = (
        config["model"]["params"]["latent_t_size"],
        config["model"]["params"]["latent_f_size"],
    )
    conditional_signal = torch.zeros(latent_shape)

    # beat: -0.5
    # downbeat: +1.0
    # 0: none; -0.5: beat; 1.0: downbeat; 0.5: downbeat+beat
    for each in beat:
        beat_index = int(
            (each / original_segment_length_before_resample) * latent_shape[0]
        )
        beat_index = min(beat_index, conditional_signal.size(0) - 1)

        conditional_signal[beat_index, :] -= 0.5

    for each in downbeat:
        beat_index = int(
            (each / original_segment_length_before_resample) * latent_shape[0]
        )
        beat_index = min(beat_index, conditional_signal.size(0) - 1)

        conditional_signal[beat_index, :] += 1.0

    # visualization(conditional_signal, dl_output["log_mel_spec"], filename = os.path.basename(dl_output["fname"])+".png")

    return {"cond_beat_downbeat": conditional_signal}
