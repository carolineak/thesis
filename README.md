# GPU Acceleration of Block Sparse LU Factorisation

## Problem statement
This project seeks to examine block sparse
LU factorisation for near-field MoM matrices and whether its performance can be enhanced by offloading computations to a GPU.

## Building and Running the Code

The repository contains three main directories with runnable C code:

### `initial_block_sparse_format` 

This directory contains the first, non-general version of the
block-sparse LU factorisation with fill-in elimination. A demo can be
run with

    $ make run

which uses default arguments for block size, block structure, and
printing level. Custom arguments can be supplied, for example:

``` {.objectivec language="C"}
$ make run ARGS="8 1 1"
```

A test suite of synthetic test cases (varying sizes and block
structures) can be executed with

``` {.objectivec language="C"}
$ make test
```

### `flat_block_sparse_format`

This directory contains the fully general factorisation implemented on
flattened data. The same commands as above are available:

``` {.objectivec language="C"}
$ make run
$ make test
```

In addition, the command

``` {.objectivec language="C"}
$ make binary_test
```

runs a test suite on realistic antenna cases loaded from binary files.
These full-scale antenna cases are only available on TICRA's server; on
other systems, the command falls back to two smaller test examples
generated from realistic antenna data.

### `kernel_block_sparse_format`

This directory holds the fully general factorisation with selected
kernels offloaded to the GPU. The same three commands

``` {.objectivec language="C"}
$ make run
$ make test
$ make binary_test
```

are supported and exercise the GPU-enabled implementation.\
\
In all three directories, the command

``` {.objectivec language="C"}
$ make clean
```

removes compiled binaries and dependency files.\
\
The directory `experiments` contains the Python scripts used to parse
result files and generate the tables and plots shown in the experimental
chapter. All generated plots and tables are stored there as well.\
\
The directory `deprecated_code` contains code that was used earlier in
the development but is no longer maintained. This includes the initial
Fortran prototype, an abandoned CUDA data-layout experiment, and small
kernel examples used to test CUDA behaviour and server configurations.
