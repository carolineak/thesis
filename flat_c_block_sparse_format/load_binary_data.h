#ifndef LOAD_BINARY_DATA_H
#define LOAD_BINARY_DATA_H

int load_block_sparse_from_bin(const char *path, block_sparse_format *bsf);

void check_block_sparse_format(const block_sparse_format *bsf);

void debug_print_input_bin(const char *path);

#endif