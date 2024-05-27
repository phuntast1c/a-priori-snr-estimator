import torch
from torch import nn

from .. import utils

EPS = torch.finfo(torch.get_default_dtype()).eps


class SNRCrossEntropy(nn.Module):
    """
    compute oracle ML-like SPP, and use that as the target
    """

    def __init__(
        self,
        frame_length: int = 1024,
        shift_length: int = 512,
        fs: int = 16000,
        limit_frequencies: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.fs = fs
        self.limit_frequencies = limit_frequencies

        self.stft = utils.STFTTorch(
            frame_length=frame_length,
            overlap_length=frame_length - shift_length,
            window=torch.hann_window,
            sqrt=True,
        )

    def forward(self, outputs, extras, meta):
        power_speech = self.stft.get_stft(extras["input"]).abs().pow(2)
        power_noise = self.stft.get_stft(extras["interference"]).abs().pow(2)

        if self.limit_frequencies:
            power_speech = power_speech[:, :, self.freq_indices, :]
            power_noise = power_noise[:, :, self.freq_indices, :]

        a_priori_snr = 10 * (power_speech / power_noise.clamp(EPS)).log10().squeeze(1)
        # assert not torch.any(torch.isnan(a_priori_snr))
        a_priori_snr = utils.compress(a_priori_snr, self.mu_k, self.sigma_k)
        a_priori_snr_estimate = outputs["snr_estimate"]
        loss = nn.functional.binary_cross_entropy_with_logits(
            a_priori_snr_estimate,
            a_priori_snr,
        )

        return {
            "loss": loss,
            "snr_target": a_priori_snr,
            "snr_estimate": a_priori_snr_estimate,
        }
