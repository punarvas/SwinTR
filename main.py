from utils.datautils import prepare_dataloader, TomogramDataset, load_tomodataset
from utils.model import Trainer, ddp_setup, load_swinunetr  # type: ignore
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import argparse
import pandas as pd  # type: ignore
import os
import json
import warnings

warnings.filterwarnings("ignore")


def main(rank: int, dataset: TomogramDataset, model: torch.nn.Module, optimizer: torch.nn.Module,
         criterion: torch.nn.Module, world_size: int, args: argparse.ArgumentParser):
    ddp_setup(rank, world_size)
    train_data = prepare_dataloader(dataset, args.batch_size)
    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(model, train_data, optimizer, rank, args.save_every, criterion, args.save_dir, args.use_lr_scheduler)
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic segmentation of Tomogram for identifying Ribosomes and Nucleosomes')
    
    parser.add_argument("--roi_size", type=int, default=64, help="Size of the patch to be obtained from tomogram (default: 64)")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size on each device (default: 1)")
    parser.add_argument("--dataset_path", type=str, help="Path to the Dataset JSON")
    parser.add_argument("--input_dim", type=int, help="Number of channels in input tomogram")
    parser.add_argument("--output_dim", type=int, help="Number of channels in segmentation map (i.e., number of classes)")
    parser.add_argument("--feature_size", type=int, default=36, help="Feature size of the SwinUNETR")
    parser.add_argument("--total_epochs", type=int, default=500, help="Total epochs to train the model (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--save_every", type=int, help="How often to save a snapshot of model weights")
    parser.add_argument("--scale_x", type=bool, default=True, help="Whether to perform min-max scaling on the inputs")
    parser.add_argument("--to_one_hot", type=bool, default=True, help="Whether to transform outputs to one-hot encoding (recommended: True)")
    parser.add_argument("--save_dir", type=str, default="~/Desktop/pretrained_model", help="Directory to save the learned weights and other results (optional)")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Whether to use exponential learning rate scheduler during training")
    args = parser.parse_args()

    # Load dataset file
    t = None
    try:
        j = json.load(open(args.dataset_path, "r"))
        t = pd.DataFrame(j["training"])
        t = t.loc[:, ["image", "label"]]
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified file: {args.dataset_path} could not be located.")
    
    train_dataset = load_tomodataset(t, args.roi_size, args.input_dim, args.output_dim)
    model, optimizer, criterion = load_swinunetr(
        roi_size=(args.roi_size, args.roi_size, args.roi_size),
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        depths=[1, 1, 1, 1],
        num_heads=[1, 3, 6, 12],
        feature_size=args.feature_size, 
        drop_rate=0.1,
        attn_drop_rate=0.1,
        learning_rate=args.learning_rate)

    world_size = torch.cuda.device_count()
    print(f"!!! Found {world_size} GPU(s) !!!")
    mp.spawn(main, args=(train_dataset, model, optimizer, criterion, 
                         world_size, args), nprocs=world_size)
