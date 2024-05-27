import numpy as np
import numpy.typing as npt


def compute_cosine_similarity(features: npt.NDArray) -> npt.NDArray:
    """
    Computes the Cosine Similarity from the input features.
    """
    
    num_rows = features.shape[0]

    # Calculate pairwise cosine similarities
    similarity_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i, num_rows):  # Avoid redundant calculations in the upper triangle
            similarity_matrix[i, j] = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j])
            )

    # Fill the lower triangle by mirroring the upper triangle for efficiency
    similarity_matrix += similarity_matrix.T - np.diag(similarity_matrix.diagonal())

    return similarity_matrix
