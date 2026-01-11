import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from transformer_model import Transformer


class HuggingFaceToyTransformer(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="https://github.com/SauravP97/toy-transformer",
    pipeline_tag="text-to-text",
    license="mit",
):
    def __init__(
        self,
        vocab_size: int,
        embedding_dimension: int,
        block_size: int,
        n_head: int,
        n_layer: int,
        device: str,
    ):
        super().__init__()
        self.transformer = Transformer(
            vocab_size,
            embedding_dimension,
            block_size,
            n_head,
            n_layer,
            device,
        )

    def forward(self, x, target=None):
        return self.transformer(x, target)

    def generate(self, start_token, max_new_tokens=1000):
        return self.transformer.generate(start_token, max_new_tokens)
