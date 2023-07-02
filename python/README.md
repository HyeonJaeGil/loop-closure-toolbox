# Python API of DBoW3 and VLAD

This is pybind version of DBoW3 and VLAD C++ libraries. 
Since python is a very powerful language for deep learning, 
**this implementation can bind Loop Clousre Detection with Deep Learning Global descriptors.**

## What's inside?
We offer two main libraries: **pydbow** and **pyvlad**.
Since two libraries have almost identical API, we will only explain with pydbow here.

## pydbow
For DBoW3, We offer (almost) the same API as C++. 
Here's an simple example.

### Create vocabulary
We can either load vocabulary from path or create with HKMeans.
```python
import pydbow

# create Vocabulary instance from file path
voc_load = pydbow.Vocabulary("./config/sthereo_01_rgb_4_3.yaml")

# or create Vocabulary later
voc_created = pydbow.Vocabulary(4,3) # k=4, l=3
training_features = ... # list of local descriptors to be used for train
voc_created.create(training_features)
```

### managing database
Users can use `add` and `query` API just like C++.
```python
import pydbow

# create Database with Vocabulary instance (voc)
db = pydbow.Database(voc, False, 0)

# add entries to Database
for image in images:
    descs = get_descriptors(image) # user's implementation
    db.add(descs)


for query_image in query_images:
    q_descs = get_descriptors(query_image) # user's implementation
    results = db.query(q_descs, 5) # retrieve best 5 results

    # result is a tuple of (entry_id, score)
    for result in results:
        print(f'idx: {result[0]}, score: {result[1]}')

```

### pairwise score
This one is **new features** (originally not included in DBoW3 libraries).
For evaluation or visualization, users can get pairwise score between entries.
Value lies between [0,1], and all values are double.
*Computations are parallelized for speed!*
```python

pscore = db.compute_pairwise_score()
cv2.imshow('pscore', pscore)
cv2.waitKey(0)

```
