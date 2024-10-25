import argparse
import pathlib
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
from cloudcasting.dataset import SatelliteDataset, ValidationSatelliteDataset
from cloudcasting.utils import numpy_validation_collate_fn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import ocf_tsimagemixer.imagemixer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    batch_size: int,
    device: str,
    forecast_steps: int,
    history_steps: int,
    num_epochs: int,
    output_directory: pathlib.Path,
    training_data_path: str,
    num_workers: int = 0,
) -> None:
    # Batch size must be greater than 1
    if batch_size < 2:
        print("Batch size must be 2 or greater")
        return

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
    save_model = False
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        for X, y in tqdm.tqdm(train_dataloader):
            # All batches must be the same size
            if X.shape[0] != batch_size:
                print(f"Skipping batch with size {X.shape[0]}")
                continue

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
                save_model = True

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best loss {best_loss:.4f}"
        )
        if save_model:
            torch.save(
                best_model.state_dict(),
                output_directory / f"best-model-epoch-{epoch}-loss-{best_loss:.3g}.state-dict.pt",
            )
            save_model = False


def validation_plot(y: np.array, y_hat: np.array, output_path: pathlib.Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(y[0][0][0], cmap="gray")
    ax2.imshow(y_hat[0][0][0], cmap="gray")
    fig.savefig(output_path)


def validate(
    batch_size: int,
    device: str,
    forecast_steps: int,
    history_steps: int,
    output_directory: pathlib.Path,
    state_dict_path: str,
    validation_data_path: str,
    num_workers: int = 0,
):
    # Create the model
    model = ocf_tsimagemixer.ImageMixer(
        batch_size,
        history_steps,
        forecast_steps,
        NUM_CHANNELS,
        IMAGE_SIZE_TUPLE[0],
        IMAGE_SIZE_TUPLE[1],
    )
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Set up the validation dataset
    valid_dataset = ValidationSatelliteDataset(
        zarr_path=validation_data_path,
        history_mins=(model.history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=model.forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_validation_collate_fn,
        drop_last=False,
    )

    for idx, (X, y) in enumerate(tqdm.tqdm(valid_dataloader)):
        if idx % 100 == 0:
            y_hat = model(torch.from_numpy(X).to(device))
            validation_plot(y, y_hat.detach().cpu(), output_directory / f"cloud-{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--train", action="store_true", help="Run training")
    cmd_group.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=2)
    parser.add_argument("--data-path", type=str, help="Path to the input data")
    parser.add_argument("--num-history-steps", type=int, help="History steps", default=24)
    parser.add_argument("--num-epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--model-state-dict", type=str, help="Path to model state dict")
    parser.add_argument("--output-directory", type=str, help="Path to save outputs to")
    args = parser.parse_args()

    # Get the appropriate PyTorch device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Ensure output directory exists
    output_directory=pathlib.Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.train:
        train(
            batch_size=args.batch_size,
            device=device,
            forecast_steps=1,
            history_steps=args.num_history_steps,
            num_epochs=args.num_epochs,
            output_directory=output_directory,
            training_data_path=args.data_path,
        )
    if args.validate:
        validate(
            batch_size=args.batch_size,
            device=device,
            forecast_steps=1,
            history_steps=args.num_history_steps,
            output_directory=output_directory,
            state_dict_path=args.model_state_dict,
            validation_data_path=args.data_path,
        )
