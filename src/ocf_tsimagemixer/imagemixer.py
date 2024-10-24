import torch
from torchtsmixer import TSMixer


class ImageMixer(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        history_steps: int,
        forecast_steps: int,
        num_channels: int,
        img_height: int,
        img_width: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.forecast_steps = forecast_steps
        self.history_steps = history_steps
        self.img_height = img_height
        self.img_width = img_width
        self.input_channels = num_channels * img_height * img_width
        self.num_channels = num_channels
        self.output_channels = num_channels * img_height * img_width
        # Construct the mixer
        self.ts_mixer = TSMixer(
            history_steps, forecast_steps, self.input_channels, self.output_channels
        )

    def forward(self, x_as_image: torch.Tensor) -> torch.Tensor:
        # We convert between:
        # - image space: (batch_size, channels, time, height, width)
        # - model space: (batch_size, time, channels)
        x_as_model = x_as_image.swapaxes(1, 2).reshape(
            self.batch_size, self.history_steps, self.input_channels
        )
        output_as_model = self.ts_mixer(x_as_model)
        output_as_image = output_as_model.reshape(
            self.batch_size,
            self.forecast_steps,
            self.num_channels,
            self.img_height,
            self.img_width,
        ).swapaxes(1, 2)
        return output_as_image
