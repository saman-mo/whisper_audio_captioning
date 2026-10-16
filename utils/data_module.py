from typing import Optional, List, Dict, Any
import os
import logger

import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio

from padding_data_collator_with_text import PaddingDataCollatorWithText
from audio_engine import UnifiedAudioEngine



class WhisperLightningDataModule(L.LightningDataModule):
    """
    Production-grade LightningDataModule for Whisper pipelines.
    Leverages a pre-instantiated UnifiedAudioEngine to stream, resample,
    and preprocess Hugging Face or custom datasets seamlessly across worker splits.
    """
    def __init__(
        self,
        audio_engine: UnifiedAudioEngine = None,
        dataset_name: str = "mozilla-foundation/common_voice_11_0",
        dataset_config: str = "en",
        batch_size: int = 16,
        num_workers: int = 4,
        dataset_split: List[str] = ['train', 'validation'],
        dataset_streaming: bool = False,
        path_to_local_dataset: str = None
        ):
        super().__init__()
        if audio_engine is None:
            raise AttributeError("Audio engine must be provided!")
        self.audio_engine = audio_engine
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_split = dataset_split
        self.dataset_streaming = dataset_streaming
        self.path_to_local_dataset = path_to_local_dataset
        

        self.data_collator = PaddingDataCollatorWithText(processor=self.audio_engine.processor)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        if self.path_to_local_dataset:
            if not os.path.isdir(self.path_to_local_dataset):
                raise FileNotFoundError(f"Local data folder path does not exist: {self.path_to_local_dataset}")
        else:
            if not self.dataset_streaming:
                load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=self.dataset_split[0],
                    streaming=self.dataset_streaming
                )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.path_to_local_dataset:
            full_dataset_dict = load_dataset(self.path_to_local_dataset, streaming=self.dataset_streaming)
        else:
            config_arg = self.dataset_config if self.dataset_config else None
            full_dataset_dict = load_dataset(
                self.dataset_name, 
                config_arg, 
                streaming=self.dataset_streaming
            )
        text_col = "sentence" if "sentence" in full_dataset_dict["train"].column_names else "text"


        if stage == "fit" or stage is None:
            if 'train' in self.dataset_split:
                self.train_dataset = full_dataset_dict["train"]
                self.train_dataset = self.train_dataset.cast_column(
                    "audio",
                    Audio(sampling_rate=self.audio_engine.sample_rate)
                )
                self.train_dataset = self.train_dataset.map(
                    lambda x: self._map_preprocessing(x, text_col),
                    remove_columns=self.train_dataset.column_names,
                    num_proc=None if self.dataset_streaming else self.num_workers
                )
                
            if 'validation' in self.dataset_split or 'val' in self.dataset_split:
                v_key = 'validation' if 'validation' in full_dataset_dict else 'val'
                self.val_dataset = full_dataset_dict[v_key]
                self.val_dataset = self.val_dataset.cast_column(
                    "audio",
                    Audio(sampling_rate=self.audio_engine.sample_rate)
                )
                self.val_dataset = self.val_dataset.map(
                    lambda x: self._map_preprocessing(x, text_col),
                    remove_columns=self.val_dataset.column_names,
                    num_proc=None if self.dataset_streaming else self.num_workers
                )

        if stage == "test" or stage is None:
            if 'test' in self.dataset_split and 'test' in full_dataset_dict:
                self.test_dataset = full_dataset_dict["test"]
                self.test_dataset = self.test_dataset.cast_column(
                    "audio",
                    Audio(sampling_rate=self.audio_engine.sample_rate)
                )
                self.test_dataset = self.test_dataset.map(
                    lambda x: self._map_preprocessing(x, text_col),
                    remove_columns=self.test_dataset.column_names,
                    num_proc=None if self.dataset_streaming else self.num_workers
                )

    def _map_preprocessing(self, batch: Dict[str, Any], text_column_name: str) -> Dict[str, Any]:
        return self.audio_engine.extract_features(
            audio_array=batch["audio"]["array"],
            text_target=batch[text_column_name]
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=0 if self.dataset_streaming else self.num_workers,
            shuffle=False if self.dataset_streaming else True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise KeyError("Validation split requested but data source does not contain it!")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=0 if self.dataset_streaming else self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            logger.warning(
                    "⚠️ Test split was requested, but no 'test' data file "
                    "or configuration was found in your dataset source. Skipping testing phase!"
                )
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=0 if self.dataset_streaming else self.num_workers,
            pin_memory=True
        )
        
        

        