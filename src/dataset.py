import os
# import librosa
import random
import torch
from torch.utils.data import Dataset

from functools import lru_cache
import torchaudio  # pyright: ignore[reportMissingTypeStubs]
import shelve
from typing import Iterator, Annotated, TypedDict, Self

from .config import Config
from .utils import cast, annotate, get_files_in_directory
from tqdm import tqdm


class AudioFileMetadata(TypedDict):
    filename: str
    num_chunks: int


class MusicDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, config: Config):
        self.mel_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.mel_spectrogram_sample_rate,
            n_fft=config.mel_spectrogram_n_fft,
            hop_length=config.mel_spectrogram_hop_length,
            n_mels=config.mel_spectrogram_n_mels,
            center=False,
        )
        self.config = config
        metadata_db = config.music_chunks_dir.joinpath("metadata.db")
        with shelve.open(metadata_db) as db:
            assert db["sample_rate"] == config.sample_rate, (
                "Sample rate mismatch. Please check your configuration."
            )
            assert db["chunk_duration"] == config.chunk_duration, (
                "Chunk duration mismatch. Please check your configuration."
            )
            assert db["chunk_overlap"] == config.chunk_overlap, (
                "Chunk overlap mismatch. Please check your configuration."
            )
            self.filename_to_num_chunks: dict[str, int] = db["filename_to_num_chunks"]
        self.post_init()

    def post_init(self) -> None:
        self.index_to_filename_chunk: list[tuple[str, int]] = [
            (filename, chunk)
            for filename, num_chunks in self.filename_to_num_chunks.items()
            for chunk in range(num_chunks)
        ]
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=self.config.time_mask_param
        )

        self.frequency_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.config.freq_mask_param
        )

    def __len__(self) -> int:
        return len(self.index_to_filename_chunk)

    @staticmethod
    def process_audio_for_chunks(
        path: str | os.PathLike[str],
        config: Config,
        resamplers: dict[int, torchaudio.transforms.Resample] | None = None,
    ) -> Annotated[Iterator[torch.Tensor], "item_shape: 1x{chunk_duration_samples}"]:
        signal, sr = torchaudio.load(path)  # pyright: ignore[reportUnknownMemberType]
        # NOTE: Resample if required
        if sr != config.sample_rate:
            if resamplers is not None:
                resampler = resamplers.get(sr, None)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=config.sample_rate
                    )
                    resamplers[sr] = resampler
            else:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=config.sample_rate
                )
            signal = cast(resampler(signal), torch.Tensor)
        signal = annotate(signal, shape="{channels}x{length}")
        # NOTE: mix down if required
        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = annotate(signal, shape="1x{length}")
        signal_samples = signal.shape[1]

        chunk_duration_samples = int(config.chunk_duration * config.sample_rate)
        chunk_overlap_samples = int(config.chunk_overlap * config.sample_rate)
        for start in range(
            0, signal_samples, chunk_duration_samples - chunk_overlap_samples
        ):
            chunk = signal[:, start : start + chunk_duration_samples]
            chunk_samples = chunk.shape[1]
            # NOTE: Right padding if required
            if chunk_samples < chunk_duration_samples:
                num_missing_samples = chunk_duration_samples - chunk_samples
                # NOTE: for two d tensor: (left, right, top, bottom)
                last_dim_padding = (0, num_missing_samples, 0, 0)
                chunk = torch.nn.functional.pad(chunk, last_dim_padding)
            yield chunk

    @classmethod
    def from_process_audio_dir_and_build_chunks(cls, config: Config) -> Self:
        resamplers: dict[int, torchaudio.transforms.Resample] = {}
        metadata_db = config.music_chunks_dir.joinpath("metadata.db")
        with shelve.open(metadata_db) as db:
            if "sample_rate" not in db:
                db["sample_rate"] = config.sample_rate
            else:
                assert db["sample_rate"] == config.sample_rate, (
                    "Sample rate mismatch. Please check your configuration."
                )
            if "chunk_duration" not in db:
                db["chunk_duration"] = config.chunk_duration
            else:
                assert db["chunk_duration"] == config.chunk_duration, (
                    "Chunk duration mismatch. Please check your configuration."
                )
            if "chunk_overlap" not in db:
                db["chunk_overlap"] = config.chunk_overlap
            else:
                assert db["chunk_overlap"] == config.chunk_overlap, (
                    "Chunk overlap mismatch. Please check your configuration."
                )
            if "filename_to_num_chunks" not in db:
                filename_to_num_chunks: dict[str, int] = {}
            else:
                filename_to_num_chunks: dict[str, int] = db["filename_to_num_chunks"]
            for filename, filepath in tqdm(
                get_files_in_directory(config.music_dir, return_filename=True)
            ):
                chunk_filepath = config.music_chunks_dir.joinpath(filename + ".pt")
                if chunk_filepath.exists():
                    continue
                chunks = cls.process_audio_for_chunks(
                    path=filepath, config=config, resamplers=resamplers
                )
                chunks = torch.stack(list(chunks))
                torch.save(chunks, chunk_filepath)
                filename_to_num_chunks[filename] = len(chunks)
            db["filename_to_num_chunks"] = filename_to_num_chunks
        self = cls.__new__(cls)
        self.mel_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.mel_spectrogram_sample_rate,
            n_fft=config.mel_spectrogram_n_fft,
            hop_length=config.mel_spectrogram_hop_length,
            n_mels=config.mel_spectrogram_n_mels,
        )
        self.config = config
        self.filename_to_num_chunks = filename_to_num_chunks
        self.post_init()
        return self

    def waveform_augment(
        self,
        waveform: Annotated[torch.Tensor, "shape: 1x{chunk_duration_samples}"],
    ) -> Annotated[torch.Tensor, "shape: 1x{chunk_duration_samples}"]:
        # NOTE: Random gain
        if random.random() < 0.8:
            gain = (
                torch.empty(1)
                .uniform_(random.randint(5, 9) / 10, 1 + random.randint(2, 5) / 10)
                .item()
            )
            waveform = waveform * gain

        # NOTE: Add gaussian noise
        if random.random() < 0.5:
            noise_strength = torch.empty(1).uniform_(0.001, 0.01).item()
            noise = torch.randn_like(waveform) * noise_strength
            waveform = waveform + noise

        # NOTE: Random polarity inversion
        if random.random() < 0.3:
            waveform = -waveform

        # NOTE: Random time shift
        if random.random() < 0.5:
            shift = random.randint(
                0,
                int(0.2 * waveform.shape[1]),
            )
            waveform = torch.roll(
                waveform,
                shifts=shift,
                dims=1,
            )

        # NOTE: Random Crop
        if random.random() < 0.2:
            target_crop = int(waveform.shape[1] * 0.8)

            start = random.randint(
                0,
                waveform.shape[1] - target_crop,
            )

            cropped = waveform[:, start : start + target_crop]

            waveform = torch.nn.functional.pad(
                cropped,
                (0, waveform.shape[1] - target_crop, 0, 0),
            )

        # NOTE: Clamp to valid range
        waveform = waveform.clamp(-1.0, 1.0)

        return waveform

    def spectrogram_augment(
        self,
        mel: Annotated[torch.Tensor, "shape: 1x{n_mels}x{num_frames}"],
    ) -> Annotated[torch.Tensor, "shape: 1x{n_mels}x{num_frames}"]:
        if random.random() < 0.8:
            mel = self.time_masking(mel)

        if random.random() < 0.8:
            mel = self.frequency_masking(mel)
        return mel

    def waveform_to_feature(
        self, waveform: Annotated[torch.Tensor, "shape: 1x{chunk_duration_samples}"]
    ) -> Annotated[torch.Tensor, "shape: 1x{n_mels}x{num_frames}"]:
        mel = self.mel_transformation(waveform)
        # NOTE: Log mel spectrogram
        mel = torch.log(mel + self.config.epsilon)
        # NOTE: Normalize mel spectrogram
        mel = (mel - mel.mean()) / (mel.std() + self.config.epsilon)
        return mel

    @staticmethod
    @lru_cache(maxsize=64)
    def load_chunks(filepath: str) -> torch.Tensor:
        return torch.load(filepath)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, chunk = self.index_to_filename_chunk[index]
        chunk_filepath: str = str(
            self.config.music_chunks_dir.joinpath(filename + ".pt")
        )
        chunks = self.load_chunks(
            chunk_filepath
        )  # NOTE: lru cache as neighboring index corresponds to same filepath
        waveform: torch.Tensor = chunks[chunk]

        view1 = self.spectrogram_augment(
            self.waveform_to_feature(
                self.waveform_augment(waveform.clone()),
            ),
        )
        view2 = self.spectrogram_augment(
            self.waveform_to_feature(
                self.waveform_augment(waveform),
            ),
        )
        return view1, view2
