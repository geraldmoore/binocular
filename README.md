# BinOcular

![Python](https://img.shields.io/badge/python-%3E=3.11-blue.svg)

A Python package which groups similar camera imagery using a cosine similarity matrix and a
time-based threshold.

## TODO

- [ ] Speed up feature encoding. Currently feature extraction uses a model that runs serially for
each image. This can be batched to massively speed up how long it takes to process a large number 
of images.

## Example

Images to group should sit inside a separate directory, for example `"/path/to/image/dir"`. Use the 
`ext` argument to specify the image extension type, for example `ext="JPG"`. 

Grouping uses both time and a cosing similarity metric. To specify the time window for two images to
be allowed to be considered similar, provide a datetime key which corresponds to the image creation
key in the metadata for the images. As these could be different based on different camera models,
this value can be specified. For example `datetime_key = "DateTime"` indicates that the key
`DateTime` in the metadata for the imagery is to be used to identify when the image was taken. To
specify the similarity threshold, use the argument `similarity_threshold`. The higher this value,
the more constrained similar images will be and the smaller the group size. The lower this value,
the larger the groups. This value is applied to the cosine similarity matrix so use this to
understand what the appropriate threshold should be.

To run grouping on an input directory containing images:

```python
from binocular.processor import Processor
from binocular.groupby import GroupBy


datetime_key = "DateTime"
time_threshold = timedelta(minutes=30)
similarity_threshold = 0.8

# Read in all metadata and compute features
data = Processor().process_dir(
    dir="/path/to/image/dir", 
    ext="JPG", 
    normalise="MinMaxScaler",  # Choose between MinMaxScaler or StandardScaler
    sort_by=datetime_key
)  # Results in a list containing the metadata and extracted feature vectors for each image

# Apply grouping
data = GroupBy(
    time_threshold=time_threshold,
    similarity_threshold=similarity_threshold,
    datetime_key=datetime_key,
).apply(data=data)  # Results in the same list as above, but with an additional group key
```

If you would like to get the similarity matrix first in order to understand the best threshold
value, extract out the feature vectors into an array and use `compute_cosine_similarity`:

```python
from binocular.processor import Processor
from binocular.similarity import compute_cosine_similarity


datetime_key = "DateTime"
time_threshold = timedelta(minutes=30)
similarity_threshold = 0.8

# Read in all metadata and compute features
data = Processor().process_dir(
    dir="/path/to/image/dir", 
    ext="JPG", 
    normalise="MinMaxScaler", 
    sort_by=datetime_key
)  # Results in a list containing the metadata and extracted feature vectors for each image

# Extract feature vectors for each image
feature_vector = np.array([image["FeatureVector"] for image in data])

# Compute similarity matrix
similarity_matrix = compute_cosine_similarity(feature_vector)

# Plot it out
plt.imshow(similarity_matrix)
```
