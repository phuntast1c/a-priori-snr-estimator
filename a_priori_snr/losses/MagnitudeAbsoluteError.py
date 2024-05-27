import torch

from . import BaseSELoss

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class MagnitudeAbsoluteError(BaseSELoss):
    def __init__(
        self,
        frame_length=512,
        overlap_length=None,
        window_fn=torch.hann_window,
        sqrt=True,
        use_mask=False,
        beta: float = 0.4,
        kind: str = "combined",
        **kwargs,
    ):
        self.kind = kind
        self.beta = beta
        self.overlap_length = (
            int(0.5 * frame_length) if overlap_length is None else overlap_length
        )

        super().__init__(
            use_stft=True,
            frame_length=frame_length,
            overlap_length=self.overlap_length,
            use_mask=use_mask,
            beta=beta,
            kind=kind,
            window_fn=window_fn,
            sqrt=sqrt,
        )

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        loss_magnitude = (estimate.abs() - target.abs()).abs().mean()
        loss_complex = (estimate - target).abs().mean()

        if self.kind == "combined":
            loss = self.beta * loss_complex + (1.0 - self.beta) * loss_magnitude
        elif self.kind == "complex":
            loss = loss_complex
        elif self.kind == "magnitude":
            loss = loss_magnitude
        else:
            raise ValueError(f"unknown loss kind {self.kind}!")
        return loss
