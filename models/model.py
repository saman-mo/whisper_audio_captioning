from typing import Optional

import lightning as L
from transformers import WhisperForConditionalGeneration
import torch.nn as nn
import torch.optim as optim
from torchmetrics.text import WordErrorRate, CharErrorRate
from peft import LoraConfig, get_peft_model



class AudioTranscriber(L.LightningModule):
    def __init__(self,
                whisper_for_conditional_generation_model: WhisperForConditionalGeneration = None,
                learning_rate: float = None,
                lora_config: LoraConfig = None,
                ignore_index_for_training: int = -100,
                max_tokens: int = 448,
                lr_scheduler: Optional[callable] = None,
                lr_scheduler_name: str = 'lr_scheduler',
                optimizer_weight_decay: float = 0.01,
                scheduler_interval: str = "step",
                scheduler_frequency: int = 1
                ):
        super().__init__()
        self.save_hyperparameters(ignore=['lr_scheduler', 'whisper_for_conditional_generation_model'])

        self.max_tokens = max_tokens

        self.optimizer_weight_decay = optimizer_weight_decay

        if not lora_config:
            raise AttributeError("LoRA configuration is not provided!")

        if not whisper_for_conditional_generation_model:
                raise AttributeError("WhisperForConditionalGeneration model is not provided!")

        self.transformer = get_peft_model(whisper_for_conditional_generation_model, lora_config)
    
        if learning_rate is None:
            raise AttributeError("Learning rate is not provided!")
        self.learning_rate = learning_rate

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_frequency = scheduler_frequency
        self.lr_scheduler_interval = scheduler_interval

        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index_for_training)

        self.batch_size = None
        
        self.val_wer_score = WordErrorRate()
        self.test_wer_score = WordErrorRate()
        self.val_cer_score = CharErrorRate()
        self.test_cer_score = CharErrorRate()

        self.val_step_outputs = []
        self.test_step_outputs = []


    def forward(self, input_features, labels=None, decoder_input_ids=None):
        self.batch_size = input_features.shape[0]
        return self.transformer(
            input_features=input_features,
            labels=labels,
            decoder_input_ids=decoder_input_ids
        )

    def training_step(self, batch, batch_idx):
        labels = batch.get("labels")
        logits = self(
            input_features=batch.get("input_features"),
            labels=labels,
        ).logits

        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        loss = self.loss_function(
            logits_shifted.view(-1, self.transformer.config.vocab_size),
            labels_shifted.view(-1)
        )

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_features = batch.get("input_features")
        labels = batch.get("labels")
        logits = self(
            input_features=input_features,
            labels=labels,
        ).logits

        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        loss = self.loss_function(
            logits_shifted.view(-1, self.transformer.config.vocab_size),
            labels_shifted.view(-1)
        )

        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        predicted_ids = self.transformer.generate(
            input_features=input_features,
            max_new_tokens=self.max_tokens,
        )

        output = {
            "predicted_ids": predicted_ids.detach().cpu(),
            "target_text": batch.get("text")  
        }
        self.val_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        input_features = batch.get("input_features")
        labels = batch.get("labels")
        logits = self(
            input_features=input_features,
            labels=labels,
        ).logits

        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        loss = self.loss_function(
            logits_shifted.view(-1, self.transformer.config.vocab_size),
            labels_shifted.view(-1)
        )

        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        predicted_ids = self.transformer.generate(
            input_features=input_features,
            max_new_tokens=self.max_tokens,
        )

        output = {
            "predicted_ids": predicted_ids.detach().cpu(),
            "target_text": batch.get("text")  
        }
        self.test_step_outputs.append(output)
        return output

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.optimizer_weight_decay
        )

        if self.lr_scheduler == None:
            return optimizer

        scheduler_config = {
            "scheduler": self.lr_scheduler(optimizer),
            "name": self.lr_scheduler_name,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
            "monitor": "val_loss"
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }







        
