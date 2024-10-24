import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
)
from cloudcasting.dataset import SatelliteDataset
from torch.utils.data import DataLoader

import ocf_tsimagemixer.imagemixer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    training_data_path: str,
    batch_size: int,
    history_steps: int,
    forecast_steps: int,
    num_epochs: int,
    device: str,
    num_workers: int = 0,
) -> None:
    # Load the training dataset
    dataset = SatelliteDataset(
        zarr_path=training_data_path,
        start_time="2022-01-31",
        end_time=None,
        history_mins=(history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )

    # Construct a DataLoader
    gen = torch.Generator()
    gen.manual_seed(0)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    # Create the model
    model = ocf_tsimagemixer.ImageMixer(
        batch_size,
        history_steps,
        forecast_steps,
        NUM_CHANNELS,
        IMAGE_SIZE_TUPLE[0],
        IMAGE_SIZE_TUPLE[1],
    )
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    best_loss = 999
    best_model = None
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        for X, y in tqdm.tqdm(train_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X.to(device))

            # Calculate the loss
            loss = criterion(outputs, y.to(device))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best loss {best_loss:.4f}"
        )
        torch.save(
            best_model.state_dict(),
            f"models/best-model-epoch-{epoch}-loss-{best_loss:.3g}.state-dict.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--train", action="store_true", help="Run training")
    cmd_group.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--model-state-dict", help="Path to model state dict")
    args = parser.parse_args()

    # Get the appropriate PyTorch device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.train:
        train(
            training_data_path="/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2022_training_nonhrv.zarr",
            batch_size=5,
            history_steps=96,
            forecast_steps=1,
            num_epochs=10,
            device=device,
        )
    if args.validate:
        print("validate")
