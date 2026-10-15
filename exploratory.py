from transformers import pipeline, WhisperForConditionalGeneration, WhisperModel, WhisperProcessor,WhisperFeatureExtractor, WhisperTokenizer
import torch
from datasets import load_dataset
import numpy



# pipeline = pipeline(
#     task='automatic-speech-recognition',
#     model='openai/whisper-small',
#     device=0
# )

dst = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small')
feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
model_caption = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
model = WhisperModel.from_pretrained('openai/whisper-small')

processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)












if __name__ == "__main__":
    # sample = "sample_med.flac"
    # res = pipeline(sample, return_timestamps=True)
    # print(res["text"])
    inputs = processor(audio=dst[0]['audio']['array'], return_tensors='pt', sample_rate=16000).input_features
    labels = processor(text=dst[0]['text'], return_tensors='pt', padding="max_length", max_length=448, add_special_tokens=False).input_ids
    # labels[labels == processor.tokenizer.pad_token_id] = -100

    
    # outputs = model(input_features=inputs, decoder_input_ids=labels)
    # print(outputs.last_hidden_state.shape)

    # print(inputs.shape)
    # print(labels.shape)
    outputs_caption = model_caption(input_features=inputs, labels=None)
    print(outputs_caption.logits.shape)
    # generated_ids = model_caption.generate(input_features=inputs)
    # caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(caption)
    # decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    
    