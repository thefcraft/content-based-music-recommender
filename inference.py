import os
import json
import chromadb
import torch
import torchaudio  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from tqdm import tqdm
from typing import Annotated, Iterator, Literal

from src import utils
from src.dataset import MusicDataset
from src.config import Config
from src.encoder import AudioEncoder


class InferenceAudioProcessor:
    def __init__(self, config: Config) -> None:
        self.mel_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.mel_spectrogram_sample_rate,
            n_fft=config.mel_spectrogram_n_fft,
            hop_length=config.mel_spectrogram_hop_length,
            n_mels=config.mel_spectrogram_n_mels,
            center=False,
        )
        self.config = config
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def waveform_to_feature(
        self, waveform: Annotated[torch.Tensor, "shape: 1x{chunk_duration_samples}"]
    ) -> Annotated[torch.Tensor, "shape: 1x{n_mels}x{num_frames}"]:
        mel = self.mel_transformation(waveform)
        # NOTE: Log mel spectrogram
        mel = torch.log(mel + self.config.epsilon)
        # NOTE: Normalize mel spectrogram
        mel = (mel - mel.mean()) / (mel.std() + self.config.epsilon)
        return mel

    def process(
        self, filepath: os.PathLike[str] | str
    ) -> Annotated[Iterator[torch.Tensor], "item_shape: 1x{n_mels}x{num_frames}"]:
        chunks = MusicDataset.process_audio_for_chunks(
            path=filepath, config=self.config, resamplers=self.resamplers
        )
        for waveform in chunks:
            feature = self.waveform_to_feature(waveform)
            yield feature.to(self.config.device)

    def process_batch(
        self, filepath: os.PathLike[str] | str
    ) -> Annotated[
        Iterator[torch.Tensor], "item_shape: {batch_size}x1x{n_mels}x{num_frames}"
    ]:
        items: list[torch.Tensor] = []  # max_size: self.config.batch_size
        for item in self.process(filepath):
            if len(items) == self.config.batch_size:
                yield torch.stack(items).to(self.config.device)
                items = []
            items.append(item)
        if len(items) != 0:
            yield torch.stack(items).to(self.config.device)


@torch.no_grad
def add_audio_to_chroma(
    config: Config,
    collection: chromadb.Collection,
    audio_encoder: AudioEncoder,
    audio_processor: InferenceAudioProcessor,
):
    for filename, filepath in tqdm(
        utils.get_files_in_directory(config.music_dir, return_filename=True)
    ):
        result = collection.get(ids=[f"{filename}-{0}"])
        if len(result["ids"]) > 0:
            continue

        embeddings: list[np.ndarray] = []
        for chunk in audio_processor.process_batch(filepath):
            chunk_embeddings = utils.cast(audio_encoder(chunk), torch.Tensor)
            chunk_embeddings = torch.nn.functional.normalize(
                chunk_embeddings,
                dim=-1,
            )
            embeddings.extend(
                [chunk_embedding.cpu().numpy() for chunk_embedding in chunk_embeddings]
            )
        collection.add(
            ids=[f"{filename}-{chunk}" for chunk in range(len(embeddings))],
            embeddings=embeddings,
            metadatas=[
                {
                    "filename": filename,
                    "chunk": chunk,
                    "num_chunks": len(embeddings),
                }
                for chunk in range(len(embeddings))
            ],
        )


def aggregate_query_results(
    results: chromadb.QueryResult,
    aggregate_key: Literal[
        "score_avg",
        "score_avg_top3",
        "score_avg_top5",
        "score_best_match",
        "num_matches",
        "num_unique_matches",
    ] = "num_unique_matches",
):
    # NOTE: cosine distance: smaller means better
    song_scores: dict[str, list[float]] = {}
    matched_chunks: dict[str, set[int]] = {}
    for metas, distances in zip(  # pyright: ignore[reportUnknownVariableType]
        results["metadatas"],  # pyright: ignore[reportArgumentType]
        results["distances"],  # pyright: ignore[reportArgumentType]
    ):
        for meta, distance in zip(
            utils.cast(metas, Iterator[chromadb.Metadata]),
            utils.cast(distances, Iterator[float]),
        ):
            filename = utils.cast(meta["filename"], str)
            chunk = utils.cast(meta["chunk"], int)
            if filename not in song_scores:
                song_scores[filename] = []
            song_scores[filename].append(distance)
            if filename not in matched_chunks:
                matched_chunks[filename] = set()
            matched_chunks[filename].add(chunk)

    aggregated: list[dict[str, str | float | int | list[int]]] = []
    for filename, distances in song_scores.items():
        distances = sorted(
            distances,
            reverse=False,  # NOTE: we want smallest
        )
        aggregated.append(
            {
                "filename": filename,
                "score_best_match": float(min(distances)),
                "score_avg": float(np.mean(distances)),
                "score_avg_top3": float(np.mean(distances[:3])),
                "score_avg_top5": float(np.mean(distances[:5])),
                "matched_chunks": sorted(
                    matched_chunks[filename],
                    reverse=False,  # NOTE: ascending order
                ),
                "num_matches": len(distances),
                "num_unique_matches": len(matched_chunks[filename]),
            }
        )
    return sorted(
        aggregated,
        key=lambda result: result[aggregate_key],
        reverse=False
        if aggregate_key
        in ("score_avg", "score_avg_top3", "score_avg_top5", "score_best_match")
        else True,
    )


@torch.no_grad
def main() -> None:
    config = Config.from_dotenv()
    audio_processor = InferenceAudioProcessor(config=config)

    chroma_client = chromadb.PersistentClient(path=config.vector_db_dir)
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.vector_db_collection_name
    )

    checkpoint = torch.load(
        config.train_checkpoint_dir.joinpath(config.inference_model_checkpoint_name)
    )
    model = AudioEncoder(embedding_dim=config.embedding_dim).to(config.device)
    model.load_state_dict(state_dict=checkpoint["encoder"])
    model.eval()

    add_audio_to_chroma(
        config=config,
        collection=chroma_collection,
        audio_encoder=model,
        audio_processor=audio_processor,
    )

    filename: str = "lenkatv - Lenka - Everything At Once (Official Video).opus"
    embeddings: list[np.ndarray] = []
    for chunk in audio_processor.process_batch(config.music_dir.joinpath(filename)):
        chunk_embeddings = utils.cast(model(chunk), torch.Tensor)
        chunk_embeddings = torch.nn.functional.normalize(
            chunk_embeddings,
            dim=-1,
        )
        embeddings.extend(
            [chunk_embedding.cpu().numpy() for chunk_embedding in chunk_embeddings]
        )
    raw_results = chroma_collection.query(
        query_embeddings=[*embeddings, np.mean(embeddings, axis=0, keepdims=False)],
        n_results=5,
        where={
            "filename": {"$ne": filename},
        },
    )
    results = aggregate_query_results(raw_results, aggregate_key='num_matches')
    with open("inference.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
