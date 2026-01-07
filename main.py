import time

from tokenizer import Tokenizer
from train import Trainer

start_time = time.perf_counter()

tokenizer = Tokenizer(dataset_path="./dataset/shakespear-text.txt")
tokenized_dataset = tokenizer.get_encoded_dataset()

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
print(f"\nFinal Training Loss: {training_loss}")
print(f"Final Validation Loss: {validation_loss}")

predicted_tokens = trainer.generate(2000)
predicted_text = tokenizer.decode(predicted_tokens, stringify=True)

print("\nPredictions:")
print(predicted_text)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nExecution time: {elapsed_time:.4f} seconds")
