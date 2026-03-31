"""Trainer class for the Language Model."""

import torch

from transformer_model import Transformer
from hugging_face.hf_model import HuggingFaceToyTransformer


class Trainer:
    """Trainer module."""

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        train_test_split: float,
        data: list[int],
        embedding_dimension: int,
        vocab_size: int,
        n_head: int,
        n_layer: int,
        max_iterations: int,
        learning_rate: float,
        eval_interval: int,
        use_hf_version: bool,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.n_head = n_head  # Number of Transformer Heads
        self.n_layer = n_layer  # Number of Transformer Blocks (layered sequentially)
        transformed_data = torch.tensor(data, dtype=torch.long)
        self.threshold = int(train_test_split * len(transformed_data))
        self.train_set = transformed_data[: self.threshold]
        self.validation_set = transformed_data[self.threshold :]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transformer_model = self._get_transformer_model(use_hf_version)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.transformer_model.parameters(), lr=self.learning_rate
        )
        self.eval_interval = eval_interval

    def _get_transformer_model(self, use_hf_version: bool):
        if not use_hf_version:
            return Transformer(
                self.vocab_size,
                self.embedding_dimension,
                self.block_size,
                self.n_head,
                self.n_layer,
                self.device,
            )
        else:
            return HuggingFaceToyTransformer(
                self.vocab_size,
                self.embedding_dimension,
                self.block_size,
                self.n_head,
                self.n_layer,
                self.device,
            )

    def get_train_data(self):
        """Get the Training split."""
        return self.train_set

    def get_validation_data(self):
        """Get the Validation split."""
        return self.validation_set

    def get_model(self):
        return self.transformer_model

    def _get_batch_of_train_or_test_split(self, split):
        """
        Return batch of Train or Validation split.
        The batch holds the input and the target list of a block size.
        Eg. For a block size of 5
        And input: [5, 1, 4, 18, 12, 10, 30]
        Return -> [5, 1, 4, 18, 12, 10], [1, 4, 18, 12, 10, 30]
        The Target array holds the target index (to the right) of the input.
        """
        data = self.train_set if split == "train" else self.validation_set
        # For a 'batch_size' of 5
        # Return 5 random indices in the range of [0, len(data) - block_size]
        # We take the 'block_size' offset coz we need to generate the input and target
        # of length block_size. If it goes more than that then we can fall into an Index Error.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        return x, y

    def _get_current_validation_loss(self) -> float:
        """Get the validation loss as per the current state of the trained model."""
        Xb_valid, Yb_valid = self._get_batch_of_train_or_test_split("valid")
        Xb_valid.to(self.device)
        Yb_valid.to(self.device)

        logits, loss = self.transformer_model(Xb_valid, Yb_valid)

        return loss.item()

    def execute_training_loop(self, track_loss=True) -> float:
        """Execute training loop and return final Loss"""
        print(f"Training on Device: {self.device}")
        print("Executing Training Loop")
        losses_per_eval_interval = []
        for iter in range(self.max_iterations):
            if iter % self.eval_interval == 0 or iter == (self.max_iterations - 1):
                if losses_per_eval_interval and track_loss:
                    average_loss_per_eval_interval = sum(
                        losses_per_eval_interval
                    ) / len(losses_per_eval_interval)
                    cur_validation_loss = self._get_current_validation_loss()
                    print(
                        f"Iteration: {iter} | Loss: {average_loss_per_eval_interval} | Validation Loss: {cur_validation_loss}"
                    )
                losses_per_eval_interval = []

            Xb, Yb = self._get_batch_of_train_or_test_split("train")
            Xb.to(self.device)
            Yb.to(self.device)

            logits, loss = self.transformer_model(Xb, Yb)
            self.optimizer.zero_grad(set_to_none=True)
            losses_per_eval_interval.append(loss.item())
            loss.backward()
            self.optimizer.step()

        validation_loss = self._get_current_validation_loss()

        return loss.item(), validation_loss

    def generate(self, tokens_to_generate=2000, kv_cache_enabled=False) -> list:
        """Generate text from the trained model"""
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        if kv_cache_enabled:
            return self.transformer_model.generate_with_kv_cache_enabled(
                context, max_new_tokens=tokens_to_generate
            )[0].tolist()
        else:
            return self.transformer_model.generate(
                context, max_new_tokens=tokens_to_generate
            )[0].tolist()
