import torch

class EarlyExit:
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def evaluate(self, logits_or_probs, is_probs=False):
        """
        Evaluates whether a sample can exit the processing pipeline early.
        Arguments:
            logits_or_probs (Tensor): Model output logits or probabilities.
            is_probs (bool): True if inputs are already probabilities (softmax-ed).
        """
        with torch.no_grad():
            if not is_probs:
                probs = torch.softmax(logits_or_probs, dim=-1)
            else:
                probs = logits_or_probs

            confidence, prediction = torch.max(probs, dim=-1)
            
            # Extract scalar values safely
            conf_val = float(confidence.item())
            pred_val = int(prediction.item())
            
            return {
                "confidence": conf_val,
                "prediction": pred_val,
                "should_exit": conf_val >= self.threshold
            }