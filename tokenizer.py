"""Tokenizer for the Transformer model."""


from typing import Union


class Tokenizer():
    """Tokenizer module."""
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self._get_data_from_path(dataset_path)
        self.vocab = self._get_vocab()
        self.vocab_size = len(self.vocab)

        self.character_to_index = {ch:i for i,ch in enumerate(self.vocab)}
        self.index_to_character = {i:ch for i,ch in enumerate(self.vocab)}

    def _get_data_from_path(self, dataset_path: str) -> str:
        text = ""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return text

    def _get_vocab(self) -> list:
        return sorted(list(set(self.dataset)))

    def get_dataset(self):
        """Return the dataset for building the Tokenizer."""
        return self.dataset

    def encode(self, characters: Union[list[str], str]) -> list[int]:
        """Encode a List of characters or String into Tokenized indices."""
        return [self.character_to_index[character] for character in characters]

    def decode(self, indices: list[int], stringify=False) -> Union[list[str], str]:
        """Decode the Tokenized indices back to a String or a List of characters."""
        decoded_list = [self.index_to_character[index] for index in indices]
        if stringify:
            return ''.join(decoded_list)
        return decoded_list
