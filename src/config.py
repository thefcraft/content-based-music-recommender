import torch
from pydantic_settings import BaseSettings
from pydantic import Field, DirectoryPath
from typing import Self, IO, Literal
from functools import cached_property
from os import PathLike



class Config(BaseSettings):
    music_dir: DirectoryPath = Field(
        alias="MUSIC_DIR",
        description="Path to the directory containing music files",
    )
    music_chunks_dir: DirectoryPath = Field(
        alias="MUSIC_CHUNKS_DIR",
        description="Path to the directory where music chunks will be saved",
    )

    sample_rate: int = Field(
        alias="SAMPLE_RATE",
        description="Sample rate for audio processing",
        default=22050,
    )

    chunk_duration: float = Field(
        alias="CHUNK_DURATION",
        description="Duration of each music chunk in seconds",
        default=10.0,
    )
    
    chunk_overlap: float = Field(
        alias="CHUNK_OVERLAP",
        description="Overlap between consecutive music chunks in seconds",
        default=3.0,
    )

    @property
    def mel_spectrogram_sample_rate(self) -> int:
        "Sample rate for Mel spectrogram computation"
        return self.sample_rate

    mel_spectrogram_n_fft: int = Field(
        alias="MEL_SPECTROGRAM_N_FFT",
        description="Number of FFT points for Mel spectrogram computation",
        default=1024,
    )
    mel_spectrogram_hop_length: int = Field(
        alias="MEL_SPECTROGRAM_HOP_LENGTH",
        description="Hop length for Mel spectrogram computation",
        default=512,
    )
    mel_spectrogram_n_mels: int = Field(
        alias="MEL_SPECTROGRAM_N_MELS",
        description="Number of Mel bands for Mel spectrogram computation",
        default=64,
    )
    device_type: Literal["gpu", "cpu", "auto"] = Field(
        default="auto",
        alias="DEVICE",
        description="Device to use for music processing",
    )

    epsilon: float = Field(
        alias="EPSILON",
        description="Small value to avoid division by zero in log computation",
        default=1e-8,
    )
    time_mask_param: int = Field(
        alias="TIME_MASK_PARAM",
        description="Parameter for time masking in Mel spectrogram computation",
        default=20,
    )
    freq_mask_param: int = Field(
        alias="FREQ_MASK_PARAM",
        description="Parameter for frequency masking in Mel spectrogram computation",
        default=10,
    )

    batch_size: int = Field(
        alias="BATCH_SIZE",
        default=32,
    )
    num_workers: int = Field(
        alias="NUM_WORKERS",
        default=4,
    )
    embedding_dim: int = Field(
        alias="EMBEDDING_DIM",
        default=512,
    )

    byol_projection_dim: int = Field(
        alias="BYOL_PROJECTION_DIM",
        default=256,
    )
    byol_hidden_dim: int = Field(
        alias="BYOL_HIDDEN_DIM",
        default=4096,
    )
    byol_moving_average_decay: float = Field(
        alias="BYOL_MOVING_AVERAGE_DECAY",
        default=0.99,
    )

    optm_lr: float = Field(
        alias="OPTM_LR",
        default=1e-4,
    )
    optm_weight_decay: float = Field(
        alias="OPTM_WEIGHT_DECAY",
        default=1e-5,
    )
    train_epochs: int = Field(
        alias="TRAIN_EPOCHS",
        default=100,
    )

    train_checkpoint_dir: DirectoryPath = Field(
        alias="TRAIN_CHECKPOINT_DIR",
    )
    vector_db_dir: DirectoryPath = Field(
        alias="VECTOR_DB_DIR",
    )
    vector_db_collection_name: str = Field(
        alias="VECTOR_DB_COLLECTION_NAME",
        default="music"
    )
    inference_model_checkpoint_name: str = Field(
        alias="INFERENCE_MODEL_CHECKPOINT_NAME"
    )

    @cached_property
    def device(self) -> torch.device:
        "Device to use for music processing"
        if self.device_type == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_type == "gpu":
            return torch.device("cuda")
        elif self.device_type == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Invalid device: {self.device_type}")

    @classmethod
    def from_env(cls) -> Self:
        return cls()  # pyright: ignore[reportCallIssue]

    @classmethod
    def from_dotenv(
        cls,
        dotenv_path: PathLike[str] | str | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> Self:
        from dotenv import load_dotenv

        load_dotenv(
            dotenv_path=dotenv_path,
            stream=stream,
            verbose=verbose,
            override=override,
            interpolate=interpolate,
            encoding=encoding,
        )
        return cls.from_env()
