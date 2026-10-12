from app.finetuning.dataset_builder import build
from app.finetuning.train_qlora import train

if __name__ == "__main__":
    build()
    train()
