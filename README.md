# Toy-Transformer (decoder-only) :snowman:

A toy [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning)) model implementation from scratch! If you have a block of text on which you want to train a transformer model and let the model generate text afterwards, then you can use this repo to achieve all of it on your local machine without a GPU.

If you want to get straight to the show, clone this repo and run the below command to train the transformer model on a block of text (documenting the work of Shakespeare) and let the model generate more content afterwards (given you have [Pytorch](https://pytorch.org/) library installed).

```python
python main.py
```

<img src="./media/toy-transformer.png" width="500" alt="Alt text description">

## :small_orange_diamond: Introduction

The Transformer implemented in this repo is a [Decoder-only](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) transformer which uses masked attention. Meaning the tokens in the training phase do not see the tokens in the future by implementing a masked attention block.

Apart from the Transformer implementation, the repo also includes other modules which will be a requirement for building a base language model end to end.

Inspired from [@karpathy](https://github.com/karpathy)'s nanogpt implementation :raised_hands:

> This repo also supports training the model on a `GPU` (***if you want***). The repo internally implements the logic to move the model parameters and the training dataset to the currently enabled device and runs on it. Meaning if you have a CPU it runs on it, if you have a GPU it seamlessly flies on it!

### Model Architecture

#### Details:
  - Number of Heads (per Attention block) = 4
  - Embedding Dimension = 64
  - Number of Attention Blocks (sequential) = 4
  - Head Size = (Embedding Dimension // Number of Heads) = 16

#### Visualizing the Model:

<img src="./media/model-tranformer-decoder.png" width="800" alt="Alt text description">

Below is the glimpse of Training and Validation loss curve over `10,000` iterations.

<img src="./media/loss-epochs.png" width="600" alt="Loss Epochs">

## :small_orange_diamond: Training and Inference with Transformer model

This repo enables you to train your own block of text data on a decoder-only transformer model and run predictions on your machine. You do not need a GPU to run training iterations and inference (predictions) using this report but can still get a gist of how the transformer model works.

### :small_blue_diamond: Step 1: Tokenize your dataset before feeding it to the Transformer model.

The below 2 lines of code takes in your text block and tokenize them through a character level tokenizer.

```python
tokenizer = Tokenizer(dataset_path="<path-to-your-text-block>")
tokenized_dataset = tokenizer.get_encoded_dataset()
```

### :small_blue_diamond: Step 2: Run the training iterations

The below 2 lines of code Trains the model on previously tokenized dataset (in Step 1). You can tune multiple parameters below however you like. If you're a beginner and wants to just see the model in action, keeping the below parameters as it is should be okay.

The `max_iterations` parameter decides how many training iterations will the model runs. The `eval_interval` decides on how often you want the Trainer to spit out the Training loss, so that you can keep a track of the training process.

At the end of the Training process, you will receive the final Training and Validation Loss. The Validation loss tells you how your trained model is doing on the unseen data. Below we set `train_test_split` parameter to **0.9**, that reserves **10%** of the tokenized dataset to be used as validation set.

```python
trainer = Trainer(
    batch_size=16,
    block_size=32,
    train_test_split=0.9,
    data=tokenized_dataset,
    embedding_dimension=64,
    vocab_size=tokenizer.vocab_size,
    n_head=4,
    n_layer=4,
    max_iterations=10000,
    learning_rate=1e-3,
    eval_interval=500,
)
training_loss, validation_loss = trainer.execute_training_loop()
```

Once this step is completed, you have a mini **Base Language Model** with yourself!

### :small_blue_diamond: Step 3: Perform Inference (Predictions)

Since, you have a trained Transformer model now, you can start generating data from it. It's a base language model which is an auto-complete engine. Meaning you provide it with some characters to start with, and it will generate characters moving forward.

Below 2 lines of code doest that!

You can pass the number of tokens you want the model to generate. In the below snippet we are generating 2000 tokens from the trained model.

The model sees and understands tokenized data which can be hard for humans to interpret. Hence, once we get the predicted tokens, we will use the same Tokenizer to decode and get the original text back and understand the predicted text block.

```python
predicted_tokens = trainer.generate(2000)
predicted_text = tokenizer.decode(predicted_tokens, stringify=True)
```

If you follow the above steps for Shakespeare text block and run `10,000` training iterations, then model would generate text blocks like these.

<img src="./media/predicted-text-block.png" width="600" alt="Loss Epochs">

> Below I have explained individual modules in detail and demonstrated how they can be understood independently.

## :small_orange_diamond: Tokenizer

<img src="./media/tokenizer-character-level.png" width="600" alt="Alt text description">

The `Tokenizer` class capable of taking a parent dataset and tokenize a stream of characters or a string. It can also convert back the tokenized indices to the original string or stream of characters.

**Note**: Currently this repo implements a character level tokenizer. Hence the provided text block is tokenized character-by-character and the prepared dataset is also character level.


### Codeblock:

```python
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

## :small_orange_diamond: Dataset processing into Training and Validation sets

Trainer module can be used to get the train-test split for a provided split value. A split value of `0.9` means `90%` of the data is for training and the rest `10%` for validation.


### Codeblock

```python
trainer = Trainer(
    batch_size=3, block_size=8, train_test_split=0.9, data=tokenizer.encode(dataset)
)

xb, yb = trainer.get_batch_of_train_or_test_split(split='Train')
print(f'Input: {xb}')
print(f'Target: {yb}')

print('Decoded the batch')

print(f'Input : {[tokenizer.decode(x, stringify=True) for x in xb.numpy()]}')
print(f'Target: {[tokenizer.decode(y, stringify=True) for y in yb.numpy()]}')
```


### Output

```
>   Input: tensor([[46, 43, 43, 49,  6,  0, 16, 39],
            [61,  1, 47, 57,  1, 57, 46, 39],
            [53, 42,  1, 51, 39, 57, 58, 43]])
>   Target: tensor([[43, 43, 49,  6,  0, 16, 39, 57],
            [ 1, 47, 57,  1, 57, 46, 39, 56],
            [42,  1, 51, 39, 57, 58, 43, 56]])
>   Decoded the batch
>   Input : ['heek,\nDa', 'w is sha', 'od maste']
>   Target: ['eek,\nDas', ' is shar', 'd master']
```
