# A decoder only Transformer implementation

The Transformer implemented in this repo is a Decoder only transformer which uses masked attention. Meaning the tokens in the training phase do not see the tokens in the future by implementing a masked attention block.

## Tokenizer

A Tokenizer class capable of taking a parent dataset and tokenize a stream of characters or a string. It can also convert back the tokenized indices to the original string or stream of characters.


### Codeblock:

```
tokenizer = Tokenizer(dataset_path='./dataset/shakespear-text.txt')
dataset = tokenizer.get_dataset()

print(dataset[:200])

encoded_value = tokenizer.encode('Hey How are You')
print(encoded_value)
print(tokenizer.decode(encoded_value))
print(tokenizer.decode(encoded_value, True))
```

### Output

```
>   First Citizen:
    Before we proceed any further, hear me speak.

    All:
    Speak, speak.

    First Citizen:
    You are all resolved rather to die than to famish?

    All:
    Resolved. resolved.

    First Citizen:
    First, you
>   [20, 43, 63, 1, 20, 53, 61, 1, 39, 56, 43, 1, 37, 53, 59]
>   ['H', 'e', 'y', ' ', 'H', 'o', 'w', ' ', 'a', 'r', 'e', ' ', 'Y', 'o', 'u']
>   Hey How are You
```