import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 4096,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.ouout_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Predictor(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 4096,
        out_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BYOL[T: nn.Module](nn.Module):
    def __init__(
        self,
        encoder: T,
        target_encoder: T,
        embedding_dim: int,
        projection_dim: int,
        hidden_dim: int,
        moving_average_decay: float,
    ) -> None:
        super().__init__()

        self.moving_average_decay = moving_average_decay

        # online network
        self.online_encoder = encoder

        self.online_projector = MLPHead(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

        self.online_predictor = Predictor(
            in_dim=projection_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

        # NOTE: target network does not require gradients
        self.target_encoder = target_encoder
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.target_projector = MLPHead(
            in_dim=self.online_projector.in_dim,
            hidden_dim=self.online_projector.hidden_dim,
            out_dim=self.online_projector.ouout_dim,
        )
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self) -> None:
        for online, target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data = (
                self.moving_average_decay * target.data
                + (1 - self.moving_average_decay) * online.data
            )

        for online, target in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target.data = (
                self.moving_average_decay * target.data
                + (1 - self.moving_average_decay) * online.data
            )

    @staticmethod
    def loss_fn(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)

        return 2 - 2 * (prediction * target).sum(dim=-1).mean()

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:

        # online branch
        online_y1 = self.online_encoder(view1)
        online_z1 = self.online_projector(online_y1)
        online_p1 = self.online_predictor(online_z1)

        online_y2 = self.online_encoder(view2)
        online_z2 = self.online_projector(online_y2)
        online_p2 = self.online_predictor(online_z2)

        # target branch
        with torch.no_grad():
            target_y1 = self.target_encoder(view1)
            target_z1 = self.target_projector(target_y1)

            target_y2 = self.target_encoder(view2)
            target_z2 = self.target_projector(target_y2)

        loss1 = self.loss_fn(online_p1, target_z2)
        loss2 = self.loss_fn(online_p2, target_z1)

        loss = loss1 + loss2

        return loss
