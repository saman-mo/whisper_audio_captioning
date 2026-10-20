from typing import Any, List, Dict

from dataclasses import dataclass
import torch


@dataclass
class PaddingDataCollatorWithText:
    """
    Data Collator that dynamically pads audio features and text labels per batch.
    It passes raw text string references directly to WhisperEvaluationCallback.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        decoder_input_ids = self.processor.tokenizer.shift_tokens_right(
            labels_batch["input_ids"], 
            decoder_start_token_id=self.processor.tokenizer.bos_token_id
        )
        batch["decoder_input_ids"] = decoder_input_ids

        batch["text"] = [feature["text"] for feature in features]

        return batch