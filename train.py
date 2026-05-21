import os

import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from typing import Sequence

from src import utils
from src.config import Config
from src.dataset import MusicDataset
from src.encoder import AudioEncoder
from src.byol import BYOL


def main() -> None:
    config = Config.from_dotenv()
    dataset = MusicDataset.from_process_audio_dir_and_build_chunks(config=config)
    dataloader = utils.cast(
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        Sequence[tuple[torch.Tensor, torch.Tensor]],
        "item_shape: {batch_size}x1x{n_mels}x{num_frames}",
    )
    encoder = AudioEncoder(
        embedding_dim=config.embedding_dim,
    ).to(config.device)
    target_encoder = AudioEncoder.from_self(encoder).to(config.device)

    byol_model = BYOL(
        encoder=encoder,
        target_encoder=target_encoder,
        embedding_dim=config.embedding_dim,
        projection_dim=config.byol_projection_dim,
        hidden_dim=config.byol_hidden_dim,
        moving_average_decay=config.byol_moving_average_decay,
    ).to(config.device)

    optimizer = AdamW(
        byol_model.parameters(),
        lr=config.optm_lr,
        weight_decay=config.optm_weight_decay,
    )

    for epoch in range(config.train_epochs):
        byol_model.train()

        total_loss = 0.0

        for step, (view1, view2) in enumerate(dataloader):
            view1 = view1.to(config.device, non_blocking=True)
            view2 = view2.to(config.device, non_blocking=True)

            optimizer.zero_grad()

            loss = utils.cast(byol_model(view1, view2), torch.Tensor)

            loss.backward() # pyright: ignore[reportUnknownMemberType]

            optimizer.step() # pyright: ignore[reportUnknownMemberType]

            byol_model.update_target_network()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"epoch={epoch} step={step}/{len(dataloader)} loss={loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)

        print(f"[epoch {epoch}/{config.train_epochs}] avg_loss={avg_loss:.4f}")

        torch.save(
            {
                "encoder": byol_model.online_encoder.state_dict(),
                "model": byol_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },  # pyright: ignore[reportUnknownArgumentType]
            config.train_checkpoint_dir.joinpath(f"checkpoint_{epoch}.pt"),
        )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
