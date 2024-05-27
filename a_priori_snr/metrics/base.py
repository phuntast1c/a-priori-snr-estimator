import torch

from .. import utils

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class BaseSELoss(torch.nn.Module):
    def __init__(
        self,
        use_stft: bool = False,
        multichannel_handling: str = "average",
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_stft = use_stft
        self.multichannel_handling = multichannel_handling
        self.kwargs = kwargs

        if self.use_stft:
            self.stft = utils.STFTTorch(
                frame_length=self.kwargs["frame_length"],
                overlap_length=self.kwargs["overlap_length"],
                window=self.kwargs["window_fn"],
                sqrt=self.kwargs["sqrt"],
            )

    def forward(self, outputs: dict, batch: dict, meta: dict = None):
        target = batch["target"]
        estimate = outputs["input_proc"]

        assert target.ndim <= 3
        multichannel = target.ndim == 3  # (B x M x T)

        if self.use_stft:
            if multichannel:
                target = torch.stack([self.stft.get_stft(x) for x in target], dim=0)
                estimate = torch.stack([self.stft.get_stft(x) for x in estimate], dim=0)
            else:
                target = self.stft.get_stft(target)
                estimate = self.stft.get_stft(estimate)

        if multichannel:
            if self.multichannel_handling == "cat":
                # concatenate channels temporally
                target = torch.cat(
                    [target[:, idx] for idx in torch.arange(target.shape[1])], dim=-1
                )
                estimate = torch.cat(
                    [estimate[:, idx] for idx in torch.arange(estimate.shape[1])],
                    dim=-1,
                )
            if self.multichannel_handling != "average":
                raise ValueError(
                    f"unknown multichannel handling type {self.multichannel_handling}!"
                )

        assert target.shape == estimate.shape

        return {"loss": self.get_loss(target, estimate)}

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
