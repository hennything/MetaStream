## Table of contents
* [General info](#general-info)
* [Usage](#usage)
* [References] (#references)

## General info
MetaStream Python implementation for streaming data

## Usage
To run this project, create or run a virtual environment.

### Install the necessary requirements:

```
$ pip install -r requirements.txt
```

### Import MetaStream:

```Python
from meta_stream.py import MetaStream
```

### Parameters:

* meta_learner:
* learners:
* base_window:
* base_sel_window_size:
* meta_window:

### Run base-fit (note that the size of the meta-table generated must be larger than the number of parameters):

Parameters:

*

```Python
meta = MetaStream(meta_learner, learners)

meta.base_train(data=df, target='nswdemand')
```

### Run meta-fit:

Scoring methods:

* meta-stream: prediction at the base level is performed by the learner recommended by the meta-learner
* default (optional): prediction at the base level is performed by the learner most frequent in the meta-table
* ensemble (optional): prediction at the base level is performed by all learners and averaged accross predictions

```Python
meta.meta_train(ådata=df, target='nswdemand')
```

## References

.. [Neuro2014] Rossi, André Luis Debiaso, et al. "MetaStream: A meta-learning based method for periodic algorithm selection in time-changing data." Neurocomputing 127 (2014): 52-64
