import math
from typing import Union

import matplotlib
import torch

from .. import losses, utils
from . import BaseLitModel

matplotlib.use("agg")

EPS = torch.finfo(torch.get_default_dtype()).eps
PI = math.pi


class MeanVarianceEstimator(BaseLitModel):
    """
    estimate statistics of a-priori SNR
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 4,
        loss: str = "CompressedLoss",
        metrics_test: Union[
            tuple, str
        ] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,DNSMOS,SISDR",
        metrics_val: Union[tuple, str] = "",
        my_optimizer: str = "AdamW",
        my_lr_scheduler: str = "ReduceLROnPlateau",
        normalize_utterance: bool = False,
        fs: int = 16000,
        frame_length: int = 128,
        shift_length: int = 32,
        window_type: str = "hann",
        **kwargs,
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="PassThrough",
            normalize_utterance=normalize_utterance,
            my_optimizer=my_optimizer,
            my_lr_scheduler=my_lr_scheduler,
        )
        self.normalize_utterance = normalize_utterance
        self.fs = fs
        self.loss = getattr(losses, loss)()

        self.frame_length = frame_length
        self.shift_length = shift_length
        self.window_type = window_type

        self.num_params = 0
        self.save_hyperparameters()
        # give the model a dummy trainable parameter
        self.dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=self.window_type,
            sqrt=self.window_type == "hann",
        )

        self.a_priori_snrs = []

    def forward(
        self,
        x,
    ):
        # estimate instantaneous a-priori SNR
        # add statistics to model
        # get current instantaneous a-priori SNR
        power_speech = self.stft.get_stft(x["input"]).abs().pow(2)
        power_noise = self.stft.get_stft(x["interference"]).abs().pow(2)
        a_priori_snr = 10 * (power_speech / power_noise.clamp(min=EPS)).log10()
        self.a_priori_snrs.append(a_priori_snr.detach().cpu())
        return {"input_proc": x["input_eval"] + self.dummy}

    def on_train_epoch_end(self) -> None:
        a_priori_snrs = torch.cat(self.a_priori_snrs, dim=0)
        a_priori_snr_mean = a_priori_snrs.mean((0, 1, 3))
        a_priori_snr_std = a_priori_snrs.std((0, 1, 3))
        print(f"overall mean: {a_priori_snrs.mean()}, std: {a_priori_snrs.std()}")
        torch.save(a_priori_snr_mean, "a_priori_snr/saved/a_priori_snr_mean.pt")
        torch.save(a_priori_snr_std, "a_priori_snr/saved/a_priori_snr_std.pt")
        # complete run
        raise NotImplementedError
