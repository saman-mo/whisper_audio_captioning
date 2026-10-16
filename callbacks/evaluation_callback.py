import lightning as L




class WhisperEvaluationCallback(L.Callback):
    def __init__(self, processor, normalizer_fn):
        super().__init__()
        self.processor = processor
        self.normalizer = normalizer_fn


    def _evaluate_metrics(self, outputs, metric_wer, metric_cer, log_prefix, pl_module):
        for step_output in outputs:
            preds_tensor = step_output["predicted_ids"]
            target_strings = step_output["target_text"]

            decoded_preds = self.processor.batch_decode(preds_tensor, skip_special_tokens=True)
            
            clean_predictions = [self.normalizer(p) for p in decoded_preds]
            clean_references = [self.normalizer(t) for t in target_strings]

            metric_wer.update(clean_predictions, clean_references)
            metric_cer.update(clean_predictions, clean_references)

        final_wer = metric_wer.compute()
        final_cer = metric_cer.compute()

        pl_module.log(f"{log_prefix}_wer", final_wer, prog_bar=True)
        pl_module.log(f"{log_prefix}_cer", final_cer, prog_bar=True)

        metric_wer.reset()
        metric_cer.reset()
        outputs.clear()


    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "val_step_outputs") or not pl_module.val_step_outputs:
            return
        self._evaluate_metrics(
            outputs=pl_module.val_step_outputs,
            metric_wer=pl_module.val_wer_score,
            metric_cer=pl_module.val_cer_score,
            log_prefix="val",
            pl_module=pl_module
        )


    def on_test_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "test_step_outputs") or not pl_module.test_step_outputs:
            return
        self._evaluate_metrics(
            outputs=pl_module.test_step_outputs,
            metric_wer=pl_module.test_wer_score,
            metric_cer=pl_module.test_cer_score,
            log_prefix="test",
            pl_module=pl_module
        )

    