from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import ExifTags, Image

from binocular.encoder import Encoder


class Processor:
    def __init__(self):
        self.encoder = Encoder()

    def process_image(self, image_path: str | Path) -> dict:
        return self._extract_metadata_compute_features(image_path=image_path)

    def process_dir(
        self, dir: str | Path, ext: str, normalise: None | str = "MinMaxScaler", sort_by: str = "DateTime"
    ) -> list[dict]:
        if isinstance(dir, str):
            dir = Path(dir)

        image_paths = dir.glob(f"*.{ext}")

        data = []
        for image_path in image_paths:
            data.append(self.process_image(image_path=image_path))

        if normalise:
            data = self._normalise(data, normalise)

        data = sorted(data, key=lambda x: x[sort_by])

        return data

    def _normalise(self, data: list[dict], normalise: str) -> list[str]:
        if normalise == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=(0, 1))
        if normalise == "StandardScaler":
            scaler = StandardScaler(with_mean=True, with_std=True)

        all_features = [image["FeatureVector"] for image in data]
        all_features = scaler.fit_transform(all_features).tolist()
        for image, norm_feature in zip(data, all_features):
            image["FeatureVector"] = norm_feature

        return data

    def _extract_metadata_compute_features(self, image_path: str | Path) -> dict:
        """Load image, extract metadata and compute the feature vector.

        Args:
            image_path (str | Path): _description_

        Returns:
            dict: Dictionary containing metadata and feature vector.
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        # Extract metadata
        image = Image.open(image_path)
        metadata = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS}
        metadata["ImagePath"] = str(image_path)
        metadata["ImageName"] = image_path.name

        # Rotate image if orientation is portrait
        if metadata.get("Orientation") != 1:
            image = image.rotate(90)

        # Compute feature vector
        metadata["FeatureVector"] = self.encoder.encode(image).tolist()

        return metadata
