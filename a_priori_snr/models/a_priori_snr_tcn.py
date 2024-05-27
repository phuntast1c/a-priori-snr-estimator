import math
from typing import Optional, Union

import torch

from .. import building_blocks as bb
from .. import losses, utils
from . import BaseLitModel


EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class APrioriSNREstimator(BaseLitModel):
    def __init__(
        self,
        learning_rate: float = 0.0003,
        batch_size: int = 8,
        metrics_test: Union[
            tuple, str
        ] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,DNSMOS,SISDR",
        metrics_val: Union[tuple, str] = "",
        frame_length: int = 1024,
        shift_length: int = 512,
        window_type: str = "hann",
        layer: int = 4,
        stack: int = 2,
        kernel: int = 3,
        hidden_dim: Optional[int] = None,
        fs: int = 16000,
        use_batchnorm: bool = True,
        use_log: bool = True,
        my_lr_scheduler: str = "ReduceLROnPlateau",
        limit_frequencies: bool = False,
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="SPPEstimator",
            my_lr_scheduler=my_lr_scheduler,
        )

        self.frame_length = frame_length
        self.shift_length = shift_length
        self.window_type = window_type
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.frequency_bins = int(self.frame_length / 2) + 1
        self.bn_dim = self.frequency_bins // 4 if hidden_dim is None else hidden_dim
        self.fs = fs
        self.use_batchnorm = use_batchnorm
        self.use_log = use_log

        # load pre-saved mean and std for compression and expansion
        self.mu_k = torch.load(
            "a_priori_snr/saved/a_priori_snr_mean.pt", map_location="cuda"
        )
        self.sigma_k = torch.load(
            "a_priori_snr/saved/a_priori_snr_std.pt", map_location="cuda"
        )

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=self.window_type,
            sqrt=self.window_type == "hann",
        )

        self.loss = losses.SNRCrossEntropy(
            frame_length=frame_length,
            shift_length=shift_length,
            limit_frequencies=limit_frequencies,
            fs=fs,
        )

        self.limit_frequencies = limit_frequencies
        if self.limit_frequencies:
            self.freq_indices = self.get_freq_indices()
            self.loss.freq_indices = self.freq_indices
            self.mu_k = self.mu_k[self.freq_indices]
            self.sigma_k = self.sigma_k[self.freq_indices]
            self.output_size = 3  # only three freqs considered
        else:
            self.output_size = self.frequency_bins

        self.loss.mu_k = self.mu_k
        self.loss.sigma_k = self.sigma_k

        self.estimator = bb.TCNEstimator(
            input_dim=3 * self.frequency_bins,  # mag, cos, sin
            output_dim=self.output_size,
            BN_dim=self.bn_dim,
            hidden_dim=4 * self.bn_dim,
            layer=int(self.layer),
            stack=int(self.stack),
            kernel=int(self.kernel),
        )

        if self.use_batchnorm:
            self.batchnorm1d_noisy = (
                torch.nn.BatchNorm1d(  # used for feature normalization
                    num_features=self.frequency_bins,
                )
            )

        self.num_params = self.count_parameters()
        self.save_hyperparameters()

    def forward(self, batch):
        """
        # dimensions convention:
        batch_size x channels x F x T x filter_length x filter_length
        """
        noisy = batch["input"]
        noisy = torch.stack([self.stft.get_stft(x) for x in noisy])
        noisy = noisy.squeeze(1)

        try:
            self.testing = self.trainer.testing
        except RuntimeError:
            self.testing = True

        # use (log) magnitude and phase spectra
        noisy_mag = noisy.abs() + EPS
        if self.use_log:
            noisy_mag = noisy_mag.log10()
        noisy_angle = noisy.angle()
        noisy_phase_cos = noisy_angle.cos()
        noisy_phase_sin = noisy_angle.sin()

        if self.use_batchnorm:
            noisy_mag = self.batchnorm1d_noisy(noisy_mag)

        features_cat = torch.cat([noisy_mag, noisy_phase_cos, noisy_phase_sin], dim=1)
        snr_estimate = self.estimator(
            features_cat
        )  # sigmoid activation is included in BCEWithLogitsLoss and thus only required for inference

        return {
            "snr_estimate": snr_estimate,
        }

    def perform_inference(self, batch):
        snr_estimate = self.forward(batch)["snr_estimate"].sigmoid()
        # SNR estimate needs to be expanded from [0, 1]
        snr_estimate = utils.expand(snr_estimate, self.mu_k, self.sigma_k)
        return snr_estimate

    def get_freq_indices(self):
        freqs_desired = [1250, 2250, 3500]  # desired by Daniel
        # find frequency bins closest to the desired frequencies
        freqs_stft = (
            torch.arange(self.stft.num_bins) / (self.stft.num_bins - 1) * self.fs / 2
        )
        freqs_stft_idx_desired = []
        for freq in freqs_desired:
            idx = torch.argmin((freqs_stft - freq).abs())
            freqs_stft_idx_desired.append(int(idx.item()))
        print(
            f"using only following freqs in loss computation: {freqs_stft[freqs_stft_idx_desired]}"
        )
        return freqs_stft_idx_desired
