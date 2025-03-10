from typing import Dict, List, Union
import torch
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from contrastors.dataset.transform import image_transform
from contrastors.config import TransformsConfig
from transformers import AutoTokenizer
import torch.nn as nn

class DictTensor:
    """
    Enable to do `tokenizer(texts).to(device)`
    """
    def __init__(self, d: Dict[str, torch.Tensor]):
        self.d = d

    def to(self, device):
        return {k: v.to(device) for k, v in self.d.items()}


class ContrastorsForBenchmark(nn.Module):
    """
    Enable model.encode_text(dict_tensor) and model.encode_image(image)
    """
    def __init__(self, model: DualEncoder):
        super().__init__()
        self.model = model

    def encode_text(self, dict_tensor):
        return self.model.encode_text(dict_tensor, normalize=True)

    def encode_image(self, image):
        # DualEncoder expects a dict with pixel_values
        vision_inputs = {"pixel_values": image}
        return self.model.encode_image(vision_inputs, normalize=True)


class ContrastorsTokenizerForBenchmark:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # TODO: should we eval with longer?
        self.tokenizer.model_max_length = 77
        self.search_query_prefix = "search_query: "

    def __call__(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        
        # Add search_query prefix to all texts
        prefixed_texts = [self.search_query_prefix + text for text in texts]
        
        # Tokenize with the contrastors tokenizer
        inputs = self.tokenizer(prefixed_texts, return_tensors="pt", padding=True, truncation=True)
        return DictTensor(inputs)

    def __len__(self):
        return len(self.tokenizer)


def load_contrastors(model_name: str, device="cpu", **kwargs):
    """
    Load a Contrastors model for benchmarking.
    The tokenizer will automatically append 'search_query' as a prefix to all input texts.
    """
    try:
        # Load the dual encoder model
        config = DualEncoderConfig.from_pretrained(model_name)
        model = DualEncoder.from_pretrained(model_name, config=config)
        model = model.to(device)
    except ImportError:
        raise ImportError("Please install contrastors package")
    
    # Setup processor components
    transform = image_transform(**TransformsConfig().dict(), is_train=False)
    tokenizer = ContrastorsTokenizerForBenchmark(AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True))

    return ContrastorsForBenchmark(model), transform, tokenizer
