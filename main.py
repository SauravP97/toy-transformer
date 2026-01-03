from tokenizer import Tokenizer

tokenizer = Tokenizer(dataset_path='./dataset/shakespear-text.txt')
dataset = tokenizer.get_dataset()

print(dataset[:200])

encoded_value = tokenizer.encode('Hey How are You')
print(encoded_value)
print(tokenizer.decode(encoded_value))
print(tokenizer.decode(encoded_value, True))
