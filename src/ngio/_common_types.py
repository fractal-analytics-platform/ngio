import numpy as np
import zarr
from dask import array as da

ArrayLike = np.ndarray | da.core.Array | zarr.Array
