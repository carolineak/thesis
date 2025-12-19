# GPU Acceleration of Block Sparse LU Factorisation

## Problem statement
This project seeks to examine block sparse
LU factorisation for near-field MoM matrices and whether its performance can be enhanced by offloading computations to a GPU.

## Abstract
This thesis investigates block sparse LU factorisation for near-field Method of Moments matrices and evaluates whether its performance can be improved by offloading computations to a GPU. On the algorithmic side, the work analyses fill-in behaviour and presents an elimination strategy inspired by the “Compress and Eliminate” method, showing that fill-in blocks can be deferred and handled at the end of the factorisation. This preserves the sparsity structure and avoids dynamic memory operations, making the algorithm more predictable and better suited to GPU execution. On the implementation side, a complete CPU version in C is developed and validated, then used to identify the dominant computational kernels through experiments on realistic antenna test cases. Based on this analysis, three kernels: the two triangular solves and the Schur complement update, are offloaded to the GPU in a hybrid CPU–GPU design where the CPU manages block-level bookkeeping and the GPU performs the computational kernels via cuBLAS. The resulting implementation maintains numerical accuracy and achieves an end-to-end speed-up of up to 3.6x over the CPU baseline, with significantly larger speed-ups at the kernel level, demonstrating that block sparse LU with eliminated fill-in handling is both practical and promising for GPU-accelerated electromagnetic simulation.

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
