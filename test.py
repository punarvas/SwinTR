import os
from functools import partial
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.datautils import prepare_dataloader, load_tomodataset
import argparse
import pandas as pd  # type: ignore
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import mrcfile
import numpy as np
import time
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic segmentation of Tomogram for identifying Ribosomes and Nucleosomes')
    
    parser.add_argument("--roi_size", type=int, default=64, help="Size of the patch to be obtained from tomogram (default: 64)")
    parser.add_argument("--pretrained_path", type=str, help="Path to the directory where pretrained model is saved")
    parser.add_argument("--dataset_path", type=str, help="Path to the test dataset JSON")
    parser.add_argument("--input_dim", type=int, help="Number of channels in input tomogram")
    parser.add_argument("--output_dim", type=int, help="Number of channels in segmentation map (i.e., number of classes)")
    parser.add_argument("--feature_size", type=int, default=36, help="Feature size of the SwinUNETR")
    parser.add_argument("--scale_x", type=bool, default=True, help="Whether to perform min-max scaling on the inputs")
    parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
    parser.add_argument("--save_dir", type=str, default="test1", help="Directory to save the learned weights and other results (optional)")
    args = parser.parse_args()

    assert os.path.exists(args.dataset_path) == True
    # Load dataset file
    t = None
    try:
        j = json.load(open(args.dataset_path, "r"))
        t = pd.DataFrame(j["test"])
        t = t.loc[:, ["image", "label"]]
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified file: {args.dataset_path} could not be located.")
    
    use_tomodataset = False
    test_dataset = load_tomodataset(t, args.roi_size, args.input_dim, 
                                    args.output_dim, monai_transform=False, use_tomodataset=use_tomodataset)
    test_loader = prepare_dataloader(test_dataset, batch_size=1, distributed=False)

    model = SwinUNETR(
        img_size=(args.roi_size, args.roi_size, args.roi_size),
        in_channels=args.input_dim,
        out_channels=args.output_dim,  # Number of segmentation classes
        feature_size=args.feature_size,
        depths=[1, 1, 1, 1],
        num_heads=[1, 3, 6, 12],
        drop_rate=0.1,
        attn_drop_rate=0.1,
        use_checkpoint=True  # Saves GPU memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_path = os.path.join(args.pretrained_path, "trained_model.pt")
    model_dict = torch.load(pretrained_model_path)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    print(f"!!! Model loaded successfully: {pretrained_model_path} !!!")

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_size, args.roi_size, args.roi_size],
        sw_batch_size=1,
        predictor=model
    )
  
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()
    with torch.no_grad():
        for i, (source, _) in enumerate(test_loader):
            tomo_path = t.iloc[i, 0]
            print(f"Running inference on {tomo_path}...")
            tomo_name = tomo_path.split("/")[-1].split(".")[0]
            source = source.to(device)
            if not use_tomodataset:    # Will apply min-max scaling manually
                source_min, source_max = torch.min(source), torch.max(source)
                source = (source - source_min) / (source_max - source_min)
            outputs = model_inferer_test(source)
            prob = torch.softmax(outputs, dim=1)
            p = prob.cpu().numpy()
            save_path = os.path.join(args.save_dir, f"{tomo_name}.npz")
            np.savez(save_path, p)
            print(f"Segmentation saved to: {save_path}")

    total_time_m = (time.time() - start_time) / 60    
    print(f"Execution completed. Time taken: {total_time_m:2f} minute(s)")
