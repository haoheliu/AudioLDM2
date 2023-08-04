import torch
import os

import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np
from audioldm2.latent_diffusion.modules.ema import *

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from torch.optim.lr_scheduler import LambdaLR
from audioldm2.latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder
from audioldm2.latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from audioldm2.latent_diffusion.util import instantiate_from_config
import soundfile as sf

from audioldm2.utilities.model import get_vocoder
from audioldm2.utilities.tools import synth_one_sample

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()
        self.automatic_optimization=False
        assert "mel_bins" in ddconfig.keys(), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = None
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, "cpu", num_mel)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)
        # print("Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None

    def get_log_dir(self):
        if self.logger_save_dir is None and self.logger_exp_name is None:
            return os.path.join(self.logger.save_dir, self.logger._project)
        else:
            return os.path.join(self.logger_save_dir, self.logger_exp_name)

    def set_log_dir(self, save_dir, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_name = exp_name

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        # x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # bs, ch, shuffled_timesteps, fbins = dec.size()
        # dec = self.time_unshuffle_operation(dec, bs, int(ch*shuffled_timesteps), fbins)
        # dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):
        from audioldm2.utilities.model import vocoder_infer

        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def visualize_latent(self, input):
        import matplotlib.pyplot as plt

        # for i in range(10):
        #     zero_input = torch.zeros_like(input) - 11.59
        #     zero_input[:,:,i * 16: i * 16 + 16,:16] += 13.59

        #     posterior = self.encode(zero_input)
        #     latent = posterior.sample()
        #     avg_latent = torch.mean(latent, dim=1)[0]
        #     plt.imshow(avg_latent.cpu().detach().numpy().T)
        #     plt.savefig("%s.png" % i)
        #     plt.close()

        np.save("input.npy", input.cpu().detach().numpy())
        # zero_input = torch.zeros_like(input) - 11.59
        time_input = input.clone()
        time_input[:, :, :, :32] *= 0
        time_input[:, :, :, :32] -= 11.59

        np.save("time_input.npy", time_input.cpu().detach().numpy())

        posterior = self.encode(time_input)
        latent = posterior.sample()
        np.save("time_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("freq_%s.png" % i)
            plt.close()

        freq_input = input.clone()
        freq_input[:, :, :512, :] *= 0
        freq_input[:, :, :512, :] -= 11.59

        np.save("freq_input.npy", freq_input.cpu().detach().numpy())

        posterior = self.encode(freq_input)
        latent = posterior.sample()
        np.save("freq_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("time_%s.png" % i)
            plt.close()

    def get_input(self, batch):
        fname, text, label_indices, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["label_vector"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )
        # if(self.time_shuffle != 1):
        #     if(fbank.size(1) % self.time_shuffle != 0):
        #         pad_len = self.time_shuffle - (fbank.size(1) % self.time_shuffle)
        #         fbank = torch.nn.functional.pad(fbank, (0,0,0,pad_len))

        ret = {}

        ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
            fbank.unsqueeze(1),
            stft.unsqueeze(1),
            fname,
            waveform.unsqueeze(1),
        )

        return ret


    def save_wave(self, batch_wav, fname, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for wav, name in zip(batch_wav, fname):
            name = os.path.basename(name)

            sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
        log = dict()
        x = batch.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(posterior.sample())
            log["reconstructions"] = xrec

        log["inputs"] = x
        wavs = self._log_img(log, train=train, index=0, waveform=waveform)
        return wavs

    def _log_img(self, log, train=True, index=0, waveform=None):
        images_input = self.tensor2numpy(log["inputs"][index, 0]).T
        images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
        images_samples = self.tensor2numpy(log["samples"][index, 0]).T

        if train:
            name = "train"
        else:
            name = "val"

        if self.logger is not None:
            self.logger.log_image(
                "img_%s" % name,
                [images_input, images_reconstruct, images_samples],
                caption=["input", "reconstruct", "samples"],
            )

        inputs, reconstructions, samples = (
            log["inputs"],
            log["reconstructions"],
            log["samples"],
        )

        if self.image_key == "fbank":
            wav_original, wav_prediction = synth_one_sample(
                inputs[index],
                reconstructions[index],
                labels="validation",
                vocoder=self.vocoder,
            )
            wav_original, wav_samples = synth_one_sample(
                inputs[index], samples[index], labels="validation", vocoder=self.vocoder
            )
            wav_original, wav_samples, wav_prediction = (
                wav_original[0],
                wav_samples[0],
                wav_prediction[0],
            )
        elif self.image_key == "stft":
            wav_prediction = (
                self.decode_to_waveform(reconstructions)[index, 0]
                .cpu()
                .detach()
                .numpy()
            )
            wav_samples = (
                self.decode_to_waveform(samples)[index, 0].cpu().detach().numpy()
            )
            wav_original = waveform[index, 0].cpu().detach().numpy()

        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "original_%s"
                    % name: wandb.Audio(
                        wav_original, caption="original", sample_rate=self.sampling_rate
                    ),
                    "reconstruct_%s"
                    % name: wandb.Audio(
                        wav_prediction, caption="reconstruct", sample_rate=self.sampling_rate
                    ),
                    "samples_%s"
                    % name: wandb.Audio(
                        wav_samples, caption="samples", sample_rate=self.sampling_rate
                    ),
                }
            )

        return wav_original, wav_prediction, wav_samples

    def tensor2numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
