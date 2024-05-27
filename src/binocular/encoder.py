import torch
import numpy.typing as npt

from PIL import Image
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights


class Encoder:
    def __init__(self):
        # Load pre-trained model
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

        # Extract only the encoding layers
        self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
        self.encoder.eval()

        # Define transforms
        self.transform = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()

    def encode(self, image: Image) -> npt.NDArray:
        with torch.no_grad():
            image = self.transform(image)
            image = image.unsqueeze(0)

            out = self.encoder(image)
            out = out.squeeze().numpy()

        return out
