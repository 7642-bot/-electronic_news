from app.finetuning.dataset_builder import build
from app.finetuning.train_qlora import train

if __name__ == "__main__":
    build()
    train()

# This is the entry point for fine-tuning a language model using QLoRA. It first builds the dataset by calling the `build()` function from the `dataset_builder` module, and then it trains the model by calling the `train()` function from the `train_qlora` module.