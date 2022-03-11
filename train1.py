import argparse, os
import logging
from model import GPTSingleHead
from data import SrcCodeDataset
from evaluate import SingleCLMEvaluator
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torchmetrics import BLEUScore

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
MODEL_MAP = {"distilgpt2": "distilgpt2", "gpt2": "gpt2", "gpt2_medium": "gpt2-medium",
             "gpt2_large": "gpt2-large"}

class LitGPTSingleHeadModel(pl.LightningModule):
    def __init__(self, model, optimizer_params):
        super().__init__(self)
        self.model = model
        self.lr = optimizer_params['lr'] 
          
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    # TODO: implement the other functions in the LightningModule like the validation step and the test step 
    def configure_optimizers(self):
        # TODO: add scheduler here
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--model_select', type=str, default="distilgpt2",
                        help='model select from distilgpt2, gpt2_medium, gpt2, or gpt2_large')
    parser.add_argument('--dataset_name', type=str, default="source_code",
                        help='dataset name whatever name you put into the ./dataset directory (by default: source_code)')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=4,
                        help='input batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=8,
                        help='input batch size for development')
    parser.add_argument('--num_epochs_train', type=int, default=16,
                        help='number of epochs to train')
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help='maximum sequence length of samples in a batch for training')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.2,
                        help='warmup_ratio')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='early_stop')
    parser.add_argument('--scheduler', type=str, default="warmuplinear",
                        help='scheduler')
    parser.add_argument('--seed', type=int, default=122,
                        help='random seed')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='accumulation steps if you want large batch size but can not fit in the memory allowed')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='number of gpu for training')
    parser.add_argument('--visiable_device', type=str, default="0",
                        help='visiable gpus for training, should be consistent with n_gpu')
    parser.add_argument('--evaluation_steps', type=int, default=200,
                        help='evaluation_steps')
    parser.add_argument('--wandb_project_name', type=str, default="code_generate",
                        help='project name for wandb')
    parser.add_argument(
        "--restore_training", action="store_true", help="restore training if a saved checkopint exists"
    )
    parser.add_argument(
        "--with_wandb", action="store_true", help="Train with wandb tracking."
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")
    dataset_folder = f"dataset/{args.dataset_name}/json/"
    assert args.model_select in MODEL_MAP.keys(), (f"model has to be in {MODEL_MAP.keys()}")
    output_path = f"model/{args.model_select}_fine_tuned_coder"
    logger.info("{} for dataset in: {}".format(output_path, dataset_folder))
    logger.info(
        f"*****************model select: {args.model_select} for code generation using dataset: {args.dataset_name}******************")
    # add more params for wandb
    args.wandb_run_name = output_path
    #initialize model by model name (the same as used in transformers lib)
    model = GPTSingleHead(MODEL_MAP[args.model_select], max_seq_length=args.max_seq_length)
    #add special tokens for controlling code generation by different programming language
    model.add_special_words({"pad_token": "<pad>", "additional_special_tokens": ["<python>", "<java>"]})
    #load training dataset
    file_path = dataset_folder + "train.jsonl"
    train_dataset = SrcCodeDataset(file_path, model, cache_path=os.path.join(".cache", output_path, "train"))
    #load developlemt dataset
    file_path = dataset_folder + "dev.jsonl"
    dev_dataset = SrcCodeDataset(file_path, model, cache_path=os.path.join(".cache", output_path, "dev"))
    # initialize development evaluator
    dev_evaluator = SingleCLMEvaluator()
    # initialize model trainer
    
    
    # TODO: create dataloaders for train_dataset and dev_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.dev_batch_size)
    
    
    
    trainer = pl.Trainer(
        gpus=args.n_gpu,
        max_epochs=args.num_epochs_train,
        check_val_every_n_epoch=args.evaluation_steps
    )
    trainer.fit(model, train_dataloader, dev_dataloader)
    trainer.validate(model, dev_dataloader)
