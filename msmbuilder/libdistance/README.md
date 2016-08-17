Libdistance
===========

Implement `pdist` and `cdist` *without* having to cast to double. `scipy`
necessarily casts all data to `double`, which might use a lot of RAM.

`src/distance_kernels.h` is based off of `scipy/spatial/src/distance_impl.h`
with `name_distance_float()` methods added.
