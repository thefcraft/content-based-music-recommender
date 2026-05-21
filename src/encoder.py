import torch
import torch.nn as nn
from typing import Annotated, Self


class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # (Bx1xHxW) => (Bx64xH'xW')
            nn.BatchNorm2d(num_features=64),  # (Bx64xH'xW') => (Bx64xH'xW')
            nn.ReLU(inplace=True),  # (Bx64xH'xW') => (Bx64xH'xW')
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # (Bx64xH'xW') => (Bx128xH''xW'')
            nn.BatchNorm2d(128),  # (Bx128xH''xW'') => (Bx128xH''xW'')
            nn.ReLU(inplace=True),  # (Bx128xH''xW'') => (Bx128xH''xW'')
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # (Bx128xH''xW'') => (Bx256xH'''xW''')
            nn.BatchNorm2d(256),  # (Bx256xH'''xW''') => (Bx256xH'''xW''')
            nn.ReLU(inplace=True),  # (Bx256xH'''xW''') => (Bx256xH'''xW''')
            nn.AdaptiveAvgPool2d((1, 1)),  # (Bx256xH'''xW''') => (Bx256x1x1)
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(
        self, x: Annotated[torch.Tensor, "shape: {batch}x1x{n_mels}x{num_frames}"]
    ) -> Annotated[torch.Tensor, "shape: {batch}x{embedding_dim}"]:
        x = self.features(x)  # (Bx1xHxW) => (Bx256x1x1)
        x = torch.flatten(x, start_dim=1)  # (Bx256x1x1) => (Bx256)
        x = self.fc(x)  # (Bx256) => (Bx{embedding_dim})
        return x

    @classmethod
    def from_self(cls, other: Self) -> Self:
        return cls(other.embedding_dim)
