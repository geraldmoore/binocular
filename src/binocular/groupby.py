from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from binocular.similarity import compute_cosine_similarity


class GroupBy:
    def __init__(
        self, time_threshold: timedelta, similarity_threshold: float, datetime_key: str = "DateTime"
    ):
        self.time_threshold = time_threshold
        self.similarity_threshold = similarity_threshold
        self.datetime_key = datetime_key
        self.similarity_matrix = None

    @staticmethod
    def _strptime(s: str, format: str = "%Y:%m:%d %H:%M:%S") -> datetime:
        return datetime.strptime(s, format)

    def _assign_group(self, data: list[dict], similar_image_idxs: npt.NDArray) -> dict:
        # Iterate over pairwise image indices and assign groups based on time threshold
        map_image_group = {}
        for ref_image_idx, sim_image_idx in similar_image_idxs:
            ref_image = data[ref_image_idx]
            sim_image = data[sim_image_idx]

            if (
                self._strptime(ref_image[self.datetime_key]) + self.time_threshold
                >= self._strptime(sim_image[self.datetime_key])
                >= self._strptime(ref_image[self.datetime_key]) - self.time_threshold
            ):
                # Check if reference image already has a group
                if ref_group := map_image_group.get(ref_image_idx):
                    map_image_group[sim_image_idx] = ref_group
                else:
                    # Assign a new group if reference doesn't have one and time threshold met
                    map_image_group[sim_image_idx] = ref_image_idx
                    map_image_group[ref_image_idx] = (
                        ref_image_idx  # Assign group to reference image as well
                    )
            else:
                # Similar image outside time window - assign its own group
                map_image_group[sim_image_idx] = sim_image_idx

        # Sort keys to maintain same order as the input list
        map_image_group = dict(sorted(map_image_group.items()))

        for image, group in zip(data, map_image_group.values()):
            image.update({"Group": group})

        return data

    def compute_similarity_matrix(self, data: list[dict]):
        feature_vector = np.array([image["FeatureVector"] for image in data])

        # Compute similarity matrix
        return compute_cosine_similarity(feature_vector)

    def apply(self, data: list[dict]) -> list[dict]:
        """Group the list of images into similar groups using a similarity matrix and a time
        based threshold.

        `datetime_key` is used to specify which key in the metadata information should be used to
        grab the image creation datetime, in order to apply a time based threshold.

        Args:
            data (list[dict]): List of image data as a dict containing image metadata and feature
                vector.
            datetime_key (str, optional): The metadata key to use for the datetime information.
                Defaults to "DateTime".
        """
        # Compute similarity matrix
        self.similarity_matrix = self.compute_similarity_matrix(data)

        # Threshold similarity matrix
        thresh_similarity_matrix = self.similarity_matrix > self.similarity_threshold

        # Pairwise image indices where similarity threshold is met
        similar_image_idxs = np.argwhere(thresh_similarity_matrix)

        return self._assign_group(data=data, similar_image_idxs=similar_image_idxs)
