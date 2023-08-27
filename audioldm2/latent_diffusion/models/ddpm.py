from multiprocessing.sharedctypes import Value
import os

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from audioldm2.latent_diffusion.modules.encoders.modules import *

from audioldm2.latent_diffusion.util import (
    exists,
    default,
    count_params,
    instantiate_from_config,
)
from audioldm2.latent_diffusion.modules.ema import LitEma
from audioldm2.latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)

# from latent_encoder.autoencoder import (
#     VQModelInterface,
#     IdentityFirstStage,
#     AutoencoderKL,
# )

from audioldm2.latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)

from audioldm2.latent_diffusion.models.ddim import DDIMSampler
from audioldm2.latent_diffusion.models.plms import PLMSSampler
import soundfile as sf
import os

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}

# CACHE_DIR = os.getenv(
#     "AUDIOLDM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/audioldm2")
# )


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        sampling_rate=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        evaluator=None,
        device=None,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.state = None
        self.device = device
        # print(
        #     f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        # )
        assert sampling_rate is not None
        self.validation_folder_name = "temp_name"
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.sampling_rate = sampling_rate

        self.clap = CLAPAudioEmbeddingClassifierFreev2(
            pretrained_path="",
            enable_cuda=self.device=="cuda",
            sampling_rate=self.sampling_rate,
            embed_mode="audio",
            amodel="HTSAT-base",
        )

        self.initialize_param_check_toolkit()

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            # print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

        self.label_indices_total = None
        # To avoid the system cannot find metric value for checkpoint
        self.metrics_buffer = {
            "val/kullback_leibler_divergence_sigmoid": 15.0,
            "val/kullback_leibler_divergence_softmax": 10.0,
            "val/psnr": 0.0,
            "val/ssim": 0.0,
            "val/inception_score_mean": 1.0,
            "val/inception_score_std": 0.0,
            "val/kernel_inception_distance_mean": 0.0,
            "val/kernel_inception_distance_std": 0.0,
            "val/frechet_inception_distance": 133.0,
            "val/frechet_audio_distance": 32.0,
        }
        self.initial_learning_rate = None
        self.test_data_subset_path = None

    def get_log_dir(self):
        return os.path.join(
            self.logger_save_dir, self.logger_exp_group_name, self.logger_exp_name
        )

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            #     print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                #     print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        shape = (batch_size, channels, self.latent_t_size, self.latent_f_size)
        self.channels
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        # fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
        # fbank, stft, label_indices, fname, waveform, text = batch
        fname, text, waveform, stft, fbank, phoneme_idx = (
            batch["fname"],
            batch["text"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
            batch["phoneme_idx"]
        )
        # for i in range(fbank.size(0)):
        #     fb = fbank[i].numpy()
        #     seg_lb = seg_label[i].numpy()
        #     logits = np.mean(seg_lb, axis=0)
        #     index = np.argsort(logits)[::-1][:5]
        #     plt.imshow(seg_lb[:,index], aspect="auto")
        #     plt.title(index)
        #     plt.savefig("%s_label.png" % i)
        #     plt.close()
        #     plt.imshow(fb, aspect="auto")
        #     plt.savefig("%s_fb.png" % i)
        #     plt.close()
        ret = {}

        ret["fbank"] = (
            fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        )
        ret["stft"] = stft.to(memory_format=torch.contiguous_format).float()
        # ret["clip_label"] = clip_label.to(memory
        # _format=torch.contiguous_format).float()
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
        ret["phoneme_idx"] = phoneme_idx.to(memory_format=torch.contiguous_format).long()
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if p.requires_grad:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        optimize_ddpm_parameter=True,
        unconditional_prob_cfg=0.1,
        warmup_steps=10000,
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        batchsize=None,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        *args,
        **kwargs,
    ):
        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.warmup_steps = warmup_steps

        if optimize_ddpm_parameter:
            if unconditional_prob_cfg == 0.0:
                "You choose to optimize DDPM. The classifier free guidance scale should be 0.1"
                unconditional_prob_cfg = 0.1
        else:
            if unconditional_prob_cfg == 0.1:
                "You choose not to optimize DDPM. The classifier free guidance scale should be 0.0"
                unconditional_prob_cfg = 0.0

        self.evaluation_params = evaluation_params
        assert self.num_timesteps_cond <= kwargs["timesteps"]

        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = "concat" if concat_mode else "crossattn"
        # if cond_stage_config == "__is_unconditional__":
        #     conditioning_key = None

        conditioning_key = list(cond_stage_config.keys())

        self.conditioning_key = conditioning_key

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.optimize_ddpm_parameter = optimize_ddpm_parameter
        # if(not optimize_ddpm_parameter):
        #     print("Warning: Close the optimization of the latent diffusion model")
        #     for p in self.model.parameters():
        #         p.requires_grad=False

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.unconditional_prob_cfg = unconditional_prob_cfg
        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.conditional_dry_run_finished = False
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        for each in self.cond_stage_models:
            params = params + list(
                each.parameters()
            )  # Add the parameter from the conditional stage

        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        # if self.use_scheduler:
        #     assert "target" in self.scheduler_config
        #     scheduler = instantiate_from_config(self.scheduler_config)

        #     print("Setting up LambdaLR scheduler...")
        #     scheduler = [
        #         {
        #             "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
        #             "interval": "step",
        #             "frequency": 1,
        #         }
        #     ]
        #     return [opt], scheduler
        return opt

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_factor == 1
            and self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            # assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}
        for i, cond_model_key in enumerate(config.keys()):
            if "params" in config[cond_model_key] and "device" in config[cond_model_key]["params"]:
                config[cond_model_key]["params"]["device"] = self.device
            model = instantiate_from_config(config[cond_model_key])
            model = model.to(self.device)
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            # when the cond_stage_key is "all", pick one random element out
            if isinstance(c, dict):
                c = c[list(c.keys())[0]]

            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()

            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c

    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_decoding_output=False,
        return_encoder_input=False,
        return_encoder_output=False,
        unconditional_prob_cfg=0.1,
    ):
        x = super().get_input(batch, k)

        x = x.to(self.device)

        if return_first_stage_encode:
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
        else:
            z = None
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(
                unconditional_prob_cfg
            ):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]

                if cond_model_key in cond_dict.keys():
                    continue

                # if not self.training:
                #     if isinstance(
                #         self.cond_stage_models[
                #             self.cond_stage_model_metadata[cond_model_key]["model_idx"]
                #         ],
                #         CLAPAudioEmbeddingClassifierFreev2,
                #     ):
                #         print(
                #             "Warning: CLAP model normally should use text for evaluation"
                #         )

                # The original data for conditioning
                # If cond_model_key is "all", that means the conditional model need all the information from a batch

                if cond_stage_key != "all":
                    xc = super().get_input(batch, cond_stage_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = batch

                # if cond_stage_key is "all", xc will be a dictionary containing all keys
                # Otherwise xc will be an entry of the dictionary
                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )

                # cond_dict will be used to condition the diffusion model
                # If one conditional model return multiple conditioning signal
                if isinstance(c, dict):
                    for k in c.keys():
                        cond_dict[k] = c[k]
                else:
                    cond_dict[cond_model_key] = c

        # If the key is accidently added to the dictionary and not in the condition list, remove the condition
        # for k in list(cond_dict.keys()):
        #     if(k not in self.cond_stage_model_metadata.keys()):
        #         del cond_dict[k]

        out = [z, cond_dict]

        if return_decoding_output:
            xrec = self.decode_first_stage(z)
            out += [xrec]

        if return_encoder_input:
            out += [x]

        if return_encoder_output:
            out += [encoder_posterior]

        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True

        # Output is a dictionary, where the value could only be tensor or tuple
        return out

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1.0 / self.scale_factor * z
            decoding = self.first_stage_model.decode(z)
        return decoding

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def encode_first_stage(self, x):
        with torch.no_grad():
            return self.first_stage_model.encode(x)

    def extract_possible_loss_in_cond_dict(self, cond_dict):
        # This function enable the conditional module to return loss function that can optimize them

        assert isinstance(cond_dict, dict)
        losses = {}

        for cond_key in cond_dict.keys():
            if "loss" in cond_key and "noncond" in cond_key:
                assert cond_key not in losses.keys()
                losses[cond_key] = cond_dict[cond_key]

        return losses

    def filter_useful_cond_dict(self, cond_dict):
        new_cond_dict = {}
        for key in cond_dict.keys():
            if key in self.cond_stage_model_metadata.keys():
                new_cond_dict[key] = cond_dict[key]

        # All the conditional key in the metadata should be used
        for key in self.cond_stage_model_metadata.keys():
            assert key in new_cond_dict.keys(), "%s, %s" % (
                key,
                str(new_cond_dict.keys()),
            )

        return new_cond_dict

    def shared_step(self, batch, **kwargs):
        if self.training:
            # Classifier-free guidance
            unconditional_prob_cfg = self.unconditional_prob_cfg
        else:
            unconditional_prob_cfg = 0.0  # TODO possible bug here

        x, c = self.get_input(
            batch, self.first_stage_key, unconditional_prob_cfg=unconditional_prob_cfg
        )

        if self.optimize_ddpm_parameter:
            loss, loss_dict = self(x, self.filter_useful_cond_dict(c))
        else:
            loss_dict = {}
            loss = None

        additional_loss_for_cond_modules = self.extract_possible_loss_in_cond_dict(c)
        assert isinstance(additional_loss_for_cond_modules, dict)

        loss_dict.update(additional_loss_for_cond_modules)

        if len(additional_loss_for_cond_modules.keys()) > 0:
            for k in additional_loss_for_cond_modules.keys():
                if loss is None:
                    loss = additional_loss_for_cond_modules[k]
                else:
                    loss = loss + additional_loss_for_cond_modules[k]

        # for k,v in additional_loss_for_cond_modules.items():
        #     self.log(
        #         "cond_stage/"+k,
        #         float(v),
        #         prog_bar=True,
        #         logger=True,
        #         on_step=True,
        #         on_epoch=True,
        #     )
        if self.training:
            assert loss is not None

        return loss, loss_dict

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        # assert c is not None
        # c = self.get_learned_conditioning(c)

        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        return loss, loss_dict

    def reorder_cond_dict(self, cond_dict):
        # To make sure the order is correct
        new_cond_dict = {}
        for key in self.conditioning_key:
            new_cond_dict[key] = cond_dict[key]
        return new_cond_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        cond = self.reorder_cond_dict(cond)

        x_recon = self.model(x_noisy, t, cond_dict=cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        # print(model_output.size(), target.size())
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )

        # if return_codebook_ids:
        #     return model_mean + nonzero_mask * (
        #         0.5 * model_log_variance
        #     ).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    def save_waveform(self, waveform, savepath, name="outwav"):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            todo_waveform = waveform[i, 0]
            todo_waveform = (
                todo_waveform / np.max(np.abs(todo_waveform))
            ) * 0.8  # Normalize the energy of the generation output
            sf.write(path, todo_waveform, samplerate=self.sampling_rate)

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim and not use_plms:
            ddim_sampler = DDIMSampler(self, device=self.device)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        elif use_plms:
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    @torch.no_grad()
    def generate_batch(
        self,
        batch,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None

        # with self.ema_scope("Plotting"):
        for i in range(1):
            z, c = self.get_input(
                batch,
                self.first_stage_key,
                unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
            )

            c = self.filter_useful_cond_dict(c)

            text = super().get_input(batch, "text")

            # Generate multiple samples
            batch_size = z.shape[0] * n_gen

            # Generate multiple samples at a time and filter out the best
            # The condition to the diffusion wrapper can have many format
            for cond_key in c.keys():
                if isinstance(c[cond_key], list):
                    for i in range(len(c[cond_key])):
                        c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                elif isinstance(c[cond_key], dict):
                    for k in c[cond_key].keys():
                        c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                else:
                    c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

            text = text * n_gen

            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = {}
                for key in self.cond_stage_model_metadata:
                    model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                    unconditional_conditioning[key] = self.cond_stage_models[
                        model_idx
                    ].get_unconditional_condition(batch_size)

            fnames = list(super().get_input(batch, "fname"))
            samples, _ = self.sample_log(
                cond=c,
                batch_size=batch_size,
                x_T=x_T,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                use_plms=use_plms,
            )

            mel = self.decode_first_stage(samples)

            waveform = self.mel_spectrogram_to_waveform(
                mel, savepath="", bs=None, name=fnames, save=False
            )

            if n_gen > 1:
                best_index = []
                similarity = self.clap.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text:")
                print(' '.join('{:.2f}'.format(num) for num in similarity.detach().cpu().tolist()))
                print("Choose the following indexes as the output:", best_index)

            return waveform

    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name=None,
        use_plms=False,
        limit_num=None,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        if name is None:
            name = self.get_validation_folder_name()

        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)

        if (
            "audiocaps" in waveform_save_path
            and len(os.listdir(waveform_save_path)) >= 964
        ):
            print("The evaluation has already been done at %s" % waveform_save_path)
            return waveform_save_path

        with self.ema_scope("Plotting"):
            for i, batch in enumerate(batchs):
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                )

                if limit_num is not None and i * z.size(0) > limit_num:
                    break

                c = self.filter_useful_cond_dict(c)

                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen

                # Generate multiple samples at a time and filter out the best
                # The condition to the diffusion wrapper can have many format
                for cond_key in c.keys():
                    if isinstance(c[cond_key], list):
                        for i in range(len(c[cond_key])):
                            c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                    elif isinstance(c[cond_key], dict):
                        for k in c[cond_key].keys():
                            c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                    else:
                        c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = {}
                    for key in self.cond_stage_model_metadata:
                        model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                        unconditional_conditioning[key] = self.cond_stage_models[
                            model_idx
                        ].get_unconditional_condition(batch_size)

                fnames = list(super().get_input(batch, "fname"))
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                if n_gen > 1:
                    try:
                        best_index = []
                        similarity = self.clap.cos_similarity(
                            torch.FloatTensor(waveform).squeeze(1), text
                        )
                        for i in range(z.shape[0]):
                            candidates = similarity[i :: z.shape[0]]
                            max_index = torch.argmax(candidates).item()
                            best_index.append(i + max_index * z.shape[0])

                        waveform = waveform[best_index]

                        print("Similarity between generated audio and text", similarity)
                        print("Choose the following indexes:", best_index)
                    except Exception as e:
                        print("Warning: while calculating CLAP score (not fatal), ", e)
                self.save_waveform(waveform, waveform_save_path, name=fnames)
        return waveform_save_path


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

        self.conditioning_key = conditioning_key

        for key in self.conditioning_key:
            if (
                "concat" in key
                or "crossattn" in key
                or "hybrid" in key
                or "film" in key
                or "noncond" in key
            ):
                continue
            else:
                raise Value("The conditioning key %s is illegal" % key)

        self.being_verbosed_once = False

    def forward(self, x, t, cond_dict: dict = {}):
        x = x.contiguous()
        t = t.contiguous()

        # x with condition (or maybe not)
        xc = x

        y = None
        context_list, attn_mask_list = [], []

        conditional_keys = cond_dict.keys()

        for key in conditional_keys:
            if "concat" in key:
                xc = torch.cat([x, cond_dict[key].unsqueeze(1)], dim=1)
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    y = torch.cat([y, cond_dict[key].squeeze(1)], dim=-1)
            elif "crossattn" in key:
                # assert context is None, "You can only have one context matrix, got %s" % (cond_dict.keys())
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][
                                k
                            ]  # crossattn_audiomae_pooled: torch.Size([12, 128, 768])
                else:
                    assert len(cond_dict[key]) == 2, (
                        "The context condition for %s you returned should have two element, one context one mask"
                        % (key)
                    )
                    context, attn_mask = cond_dict[key]

                # The input to the UNet model is a list of context matrix
                context_list.append(context)
                attn_mask_list.append(attn_mask)

            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                raise NotImplementedError()

        # if(not self.being_verbosed_once):
        #     print("The input shape to the diffusion model is as follows:")
        #     print("xc", xc.size())
        #     print("t", t.size())
        #     for i in range(len(context_list)):
        #         print("context_%s" % i, context_list[i].size(), attn_mask_list[i].size())
        #     if(y is not None):
        #         print("y", y.size())
        #     self.being_verbosed_once = True
        out = self.diffusion_model(
            xc, t, context_list=context_list, y=y, context_attn_mask_list=attn_mask_list
        )
        return out
        self.warmup_step()

        if (
            self.state is None
            and len(self.trainer.optimizers[0].state_dict()["state"].keys()) > 0
        ):
            self.state = (
                self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"].clone()
            )
        elif self.state is not None and batch_idx % 1000 == 0:
            assert (
                torch.sum(
                    torch.abs(
                        self.state
                        - self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"]
                    )
                )
                > 1e-7
            ), "Optimizer is not working"

        if len(self.metrics_buffer.keys()) > 0:
            for k in self.metrics_buffer.keys():
                self.log(
                    k,
                    self.metrics_buffer[k],
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
                print(k, self.metrics_buffer[k])
            self.metrics_buffer = {}

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_abs",
            float(lr),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )


if __name__ == "__main__":
    import yaml

    model_config = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable-diffusion/models/ldm/text2img256/config.yaml"
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    latent_diffusion = LatentDiffusion(**model_config["model"]["params"])

    import ipdb

    ipdb.set_trace()
