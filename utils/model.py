import torch
from monai.networks.nets import SwinUNETR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.functional as F
from monai.losses import DiceLoss
from torch.distributed import init_process_group, destroy_process_group
import os
import json
from torch.optim.lr_scheduler import ExponentialLR


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        criterion: torch.nn.Module,
        save_dir: str,
        use_lr_scheduler: bool
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.criterion = criterion
        self.save_dir = save_dir
        self.use_lr_scheduler = use_lr_scheduler

        if self.use_lr_scheduler:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            epoch_loss += self._run_batch(source, targets)

        aggregated_loss = torch.tensor([epoch_loss], device=self.gpu_id)
        torch.distributed.all_reduce(aggregated_loss, op=torch.distributed.ReduceOp.SUM)
        num_samples = len(self.train_data.dataset)
        aggregated_loss = aggregated_loss.item() / num_samples

        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Epoch loss: {aggregated_loss:.5f}")
        return aggregated_loss

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        save_path = os.path.join(self.save_dir, "trained_model.pt")
        torch.save(ckp, save_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        loss_history = []
        for epoch in range(max_epochs):
            epoch_loss = self._run_epoch(epoch)
            loss_history.append(epoch_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            if self.use_lr_scheduler:
                self.scheduler.step()
        
        save_path = os.path.join(self.save_dir, "loss_history.json")
        history = {"train_loss": loss_history}
        with open(save_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Training loss history saved to {save_path}")



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_swinunetr(roi_size, input_dim, output_dim, depths, num_heads, feature_size, 
                   drop_rate=0.1, attn_drop_rate=0.1, learning_rate=1e-4):
    model = SwinUNETR(
        img_size=roi_size,
        in_channels=input_dim,
        out_channels=output_dim,  # Number of segmentation classes
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        use_checkpoint=True  # Saves GPU memory
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = DiceLoss(to_onehot_y=False, sigmoid=False, softmax=True)
    return model, optimizer, criterion
