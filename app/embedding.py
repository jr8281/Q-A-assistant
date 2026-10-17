import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)

# BGE models perform better when queries (not documents) carry this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class PyTorchEmbedder:
    """Generates normalised text embeddings locally with a Hugging Face model."""

    def __init__(self, model_name: str = settings.embed_model):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_embeddings(
        self, texts: list[str], batch_size: int = 32, is_query: bool = False
    ) -> np.ndarray:
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        batches = []
        for i in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[i:i + batch_size],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                # BGE uses the [CLS] token (first position) as the sentence embedding
                embeddings = output[0][:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            batches.append(embeddings.cpu().numpy())

        return np.vstack(batches)