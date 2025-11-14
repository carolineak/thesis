#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include "block_sparse_format.h"


// Compare function for qsort: sort by start, then end
static int cmp_range(const void *a, const void *b)
{
    const int_range *ra = (const int_range *)a;
    const int_range *rb = (const int_range *)b;

    if (ra->start < rb->start) return -1;
    if (ra->start > rb->start) return  1;
    if (ra->end   < rb->end)   return -1;
    if (ra->end   > rb->end)   return  1;
    return 0;
}

// Build row/col slices with ranges sorted by (start, end)
static int build_slices_from_ranges(
    int num_blocks,
    const int *start_arr,
    const int *end_arr,
    int *out_num_slices,
    block_slice **out_slices,
    int *out_block_indices  // length = num_blocks
)
{
    if (num_blocks <= 0) {
        *out_num_slices = 0;
        *out_slices = NULL;
        return 0;
    }

    // 1) Collect all ranges
    int_range *all_ranges = malloc((size_t)num_blocks * sizeof *all_ranges);
    if (!all_ranges) return -1;

    for (int k = 0; k < num_blocks; ++k) {
        all_ranges[k].start = start_arr[k];
        all_ranges[k].end   = end_arr[k];
    }

    // 2) Sort them
    qsort(all_ranges, (size_t)num_blocks, sizeof *all_ranges, cmp_range);

    // 3) Deduplicate to unique sorted ranges
    int_range *unique = malloc((size_t)num_blocks * sizeof *unique);
    int *counts       = calloc((size_t)num_blocks, sizeof *counts);
    if (!unique || !counts) {
        free(all_ranges);
        free(unique);
        free(counts);
        return -2;
    }

    int num_slices = 0;
    unique[0] = all_ranges[0];
    num_slices = 1;

    for (int i = 1; i < num_blocks; ++i) {
        if (cmp_range(&all_ranges[i], &unique[num_slices - 1]) != 0) {
            unique[num_slices++] = all_ranges[i];
        }
    }

    free(all_ranges);

    // 4) For each block, find which unique range it belongs to and count
    for (int k = 0; k < num_blocks; ++k) {
        int_range r = { start_arr[k], end_arr[k] };

        // linear search is fine unless you have *huge* numbers of slices
        int slice_idx = -1;
        for (int s = 0; s < num_slices; ++s) {
            if (unique[s].start == r.start && unique[s].end == r.end) {
                slice_idx = s;
                break;
            }
        }
        if (slice_idx < 0) {
            // should never happen
            free(unique);
            free(counts);
            return -3;
        }
        out_block_indices[k] = slice_idx;
        counts[slice_idx]++;
    }

    // 5) Allocate slices in sorted range order
    block_slice *slices = calloc((size_t)num_slices, sizeof *slices);
    if (!slices) {
        free(unique);
        free(counts);
        return -4;
    }

    for (int s = 0; s < num_slices; ++s) {
        slices[s].range      = unique[s];        // sorted ranges
        slices[s].num_blocks = counts[s];
        slices[s].indices    = malloc((size_t)counts[s] * sizeof *slices[s].indices);
        if (!slices[s].indices) {
            for (int j = 0; j < s; ++j) free(slices[j].indices);
            free(slices);
            free(unique);
            free(counts);
            return -5;
        }
        counts[s] = 0; // reuse as fill position
    }

    // 6) Fill slice->indices using the precomputed mapping
    for (int k = 0; k < num_blocks; ++k) {
        int s = out_block_indices[k];
        int pos = counts[s]++;
        slices[s].indices[pos] = k;
    }

    free(unique);
    free(counts);

    *out_num_slices = num_slices;
    *out_slices     = slices;
    return 0;
}


int load_block_sparse_from_bin(const char *path, block_sparse_format *bsf)
{
    if (!bsf) return -1;
    memset(bsf, 0, sizeof *bsf);

    FILE *f = fopen(path, "rb");
    if (!f) return -2;

    // 1) Read num_blocks (Int32)
    int32_t num_blocks_i32 = 0;
    if (fread(&num_blocks_i32, sizeof num_blocks_i32, 1, f) != 1) {
        fclose(f);
        return -3;
    }
    if (num_blocks_i32 <= 0) {
        fclose(f);
        return -4;
    }

    int num_blocks = (int)num_blocks_i32;

    // 2) Read the four index arrays (Int32 each)
    int32_t *row_start32 = malloc((size_t)num_blocks * sizeof *row_start32);
    int32_t *row_stop32  = malloc((size_t)num_blocks * sizeof *row_stop32);
    int32_t *col_start32 = malloc((size_t)num_blocks * sizeof *col_start32);
    int32_t *col_stop32  = malloc((size_t)num_blocks * sizeof *col_stop32);

    if (!row_start32 || !row_stop32 || !col_start32 || !col_stop32) {
        free(row_start32); free(row_stop32);
        free(col_start32); free(col_stop32);
        fclose(f);
        return -5;
    }

    if (fread(row_start32, sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(row_stop32,  sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(col_start32, sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(col_stop32,  sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks) {
        free(row_start32); free(row_stop32);
        free(col_start32); free(col_stop32);
        fclose(f);
        return -6;
    }

    // 3) Convert to int and compute global dimensions
    int *row_start = malloc((size_t)num_blocks * sizeof *row_start);
    int *row_stop  = malloc((size_t)num_blocks * sizeof *row_stop);
    int *col_start = malloc((size_t)num_blocks * sizeof *col_start);
    int *col_stop  = malloc((size_t)num_blocks * sizeof *col_stop);

    if (!row_start || !row_stop || !col_start || !col_stop) {
        free(row_start32); free(row_stop32);
        free(col_start32); free(col_stop32);
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return -7;
    }

    int m = 0;
    int n = 0;

    for (int k = 0; k < num_blocks; ++k) {
        // Convert from 1-indexed to 0-indexed by subtracting 1
        row_start[k] = (int)row_start32[k] - 1;
        row_stop[k]  = (int)row_stop32[k] - 1;
        col_start[k] = (int)col_start32[k] - 1;
        col_stop[k]  = (int)col_stop32[k] - 1;

        int last_row = row_stop[k] + 1;
        int last_col = col_stop[k] + 1;

        if (last_row > m) m = last_row;
        if (last_col > n) n = last_col;
    }

    free(row_start32); free(row_stop32);
    free(col_start32); free(col_stop32);

    // 4) Allocate internal arrays in bsf
    bsf->row_indices      = malloc((size_t)num_blocks * sizeof *bsf->row_indices);
    bsf->col_indices      = malloc((size_t)num_blocks * sizeof *bsf->col_indices);
    bsf->block_sizes      = malloc((size_t)num_blocks * sizeof *bsf->block_sizes);
    bsf->offsets          = malloc((size_t)num_blocks * sizeof *bsf->offsets);
    bsf->relies_on_fillin = calloc((size_t)num_blocks, sizeof *bsf->relies_on_fillin);

    if (!bsf->row_indices || !bsf->col_indices ||
        !bsf->block_sizes || !bsf->offsets || !bsf->relies_on_fillin) {

        free(row_start); free(row_stop); free(col_start); free(col_stop);
        free(bsf->row_indices);
        free(bsf->col_indices);
        free(bsf->block_sizes);
        free(bsf->offsets);
        free(bsf->relies_on_fillin);
        fclose(f);
        memset(bsf, 0, sizeof *bsf);
        return -8;
    }

    // 5) Build row and column slices
    if (build_slices_from_ranges(num_blocks, row_start, row_stop,
                                 &bsf->num_rows, &bsf->rows, bsf->row_indices) != 0 ||
        build_slices_from_ranges(num_blocks, col_start, col_stop,
                                 &bsf->num_cols, &bsf->cols, bsf->col_indices) != 0) {

        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        bsf_free(bsf);
        return -9;
    }

    // 6) Compute block_sizes and offsets, and total_elems
    size_t total_elems = 0;
    for (int k = 0; k < num_blocks; ++k) {
        // *** IMPORTANT: adjust here if row_stop/col_stop are inclusive ***
        int rows = row_stop[k] - row_start[k] + 1;
        int cols = col_stop[k] - col_start[k] + 1;
        int sz   = rows * cols;

        if (rows <= 0 || cols <= 0 || sz <= 0) {
            free(row_start); free(row_stop); free(col_start); free(col_stop);
            fclose(f);
            bsf_free(bsf);
            return -10;
        }

        bsf->block_sizes[k] = sz;
        bsf->offsets[k]     = (int)total_elems;  // assumes total_elems fits in int
        total_elems        += (size_t)sz;
    }

    free(row_start); free(row_stop); free(col_start); free(col_stop);

    // 7) Read the remaining data as float complex
    bsf->flat_data = malloc(total_elems * sizeof *bsf->flat_data);
    if (!bsf->flat_data) {
        fclose(f);
        bsf_free(bsf);
        return -11;
    }

    if (fread(bsf->flat_data, sizeof(float complex), total_elems, f) != total_elems) {
        fclose(f);
        bsf_free(bsf);
        return -12;
    }

    fclose(f);

    // 8) Fill top-level meta
    bsf->m            = m;
    bsf->n            = n;
    bsf->num_blocks   = num_blocks;
    bsf->global_pivot = NULL;   // not factorized

    return 0;
}


void check_block_sparse_format(const block_sparse_format *bsf)
{
    if (!bsf) {
        printf("[check_bsf] ERROR: bsf pointer is NULL\n");
        return;
    }

    printf("=========================================================\n");
    printf("Block-sparse matrix summary\n");
    printf("---------------------------------------------------------\n");
    printf(" Global size      : %d x %d\n", bsf->m, bsf->n);
    printf(" Number of blocks : %d\n", bsf->num_blocks);
    printf(" Row slices       : %d\n", bsf->num_rows);
    printf(" Col slices       : %d\n", bsf->num_cols);
    printf("---------------------------------------------------------\n");

    // Basic array presence
    if (!bsf->row_indices || !bsf->col_indices ||
        !bsf->block_sizes || !bsf->offsets || !bsf->flat_data) {
        printf("[check_bsf] ERROR: One or more essential arrays are NULL\n");
        return;
    }

    // Check monotonic offsets and positive sizes
    size_t total = 0;
    int bad_offsets = 0;
    for (int k = 0; k < bsf->num_blocks; ++k) {
        int sz = bsf->block_sizes[k];
        int off = bsf->offsets[k];
        if (sz <= 0) {
            printf("[check_bsf] WARNING: block %d has nonpositive size %d\n", k, sz);
        }
        if (k > 0 && off < bsf->offsets[k-1]) {
            printf("[check_bsf] WARNING: block %d has nonmonotonic offset %d < prev %d\n",
                   k, off, bsf->offsets[k-1]);
            bad_offsets = 1;
        }
        total = (size_t)off + (size_t)sz;
    }

    printf(" Total elements (from offsets) : %zu\n", total);

    // Check row/col slice coverage and report first few
    printf("\nRow slices (%d):\n", bsf->num_rows);
    for (int i = 0; i < bsf->num_rows && i < 8; ++i) {
        printf("  Slice %2d: range [%d,%d], num_blocks=%d\n",
               i, bsf->rows[i].range.start, bsf->rows[i].range.end, bsf->rows[i].num_blocks);
    }
    if (bsf->num_rows > 8) printf("  ... (%d more)\n", bsf->num_rows - 5);

    printf("\nCol slices (%d):\n", bsf->num_cols);
    for (int j = 0; j < bsf->num_cols && j < 8; ++j) {
        printf("  Slice %2d: range [%d,%d], num_blocks=%d\n",
               j, bsf->cols[j].range.start, bsf->cols[j].range.end, bsf->cols[j].num_blocks);
    }
    if (bsf->num_cols > 8) printf("  ... (%d more)\n", bsf->num_cols - 5);

    // Verify each block maps to a valid slice
    int bad_rows = 0, bad_cols = 0;
    for (int k = 0; k < bsf->num_blocks; ++k) {
        int ri = bsf->row_indices[k];
        int ci = bsf->col_indices[k];
        if (ri < 0 || ri >= bsf->num_rows) bad_rows++;
        if (ci < 0 || ci >= bsf->num_cols) bad_cols++;
    }
    if (bad_rows) printf("[check_bsf] WARNING: %d blocks have invalid row slice indices\n", bad_rows);
    if (bad_cols) printf("[check_bsf] WARNING: %d blocks have invalid col slice indices\n", bad_cols);

    // Print first few blocks
    printf("\nSample blocks:\n");
    int num_print = (bsf->num_blocks < 8) ? bsf->num_blocks : 8;
    for (int k = 0; k < num_print; ++k) {
        int off = bsf->offsets[k];
        int sz  = bsf->block_sizes[k];
        printf(" Block %2d: offset=%d size=%d (slice_row=%d, slice_col=%d)",
               k, off, sz, bsf->row_indices[k], bsf->col_indices[k]);

        if (bsf->flat_data && sz > 0) {
            float complex v0 = bsf->flat_data[off];
            printf(" first entry = (%g + %gi)\n", crealf(v0), cimagf(v0));
        } else {
            printf("\n");
        }
    }
    if (bsf->num_blocks > num_print)
        printf("  ... (%d more blocks)\n", bsf->num_blocks - num_print);

    // Quick final consistency report
    printf("---------------------------------------------------------\n");
    if (!bad_offsets && !bad_rows && !bad_cols)
        printf("Structure looks consistent\n");
    else
        printf("Detected inconsistencies (see warnings above)\n");
    printf("=========================================================\n\n");
}

void debug_print_input_bin(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("[debug_print_input_bin] fopen");
        return;
    }

    // 1) Read num_blocks (Int32)
    int32_t num_blocks32 = 0;
    if (fread(&num_blocks32, sizeof num_blocks32, 1, f) != 1) {
        printf("[debug_print_input_bin] ERROR: could not read num_blocks\n");
        fclose(f);
        return;
    }

    if (num_blocks32 <= 0) {
        printf("[debug_print_input_bin] ERROR: num_blocks=%d (must be >0)\n", num_blocks32);
        fclose(f);
        return;
    }
    int num_blocks = (int)num_blocks32;

    // 2) Allocate arrays for indices
    int32_t *row_start = malloc((size_t)num_blocks * sizeof *row_start);
    int32_t *row_stop  = malloc((size_t)num_blocks * sizeof *row_stop);
    int32_t *col_start = malloc((size_t)num_blocks * sizeof *col_start);
    int32_t *col_stop  = malloc((size_t)num_blocks * sizeof *col_stop);

    if (!row_start || !row_stop || !col_start || !col_stop) {
        printf("[debug_print_input_bin] ERROR: memory allocation failed for index arrays\n");
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    // 3) Read the four index arrays
    if (fread(row_start, sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(row_stop,  sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(col_start, sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks ||
        fread(col_stop,  sizeof(int32_t), (size_t)num_blocks, f) != (size_t)num_blocks) {

        printf("[debug_print_input_bin] ERROR: could not read index arrays\n");
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    // 4) Determine how many complex values there are from file size
    long header_bytes = sizeof(num_blocks32)
                      + 4L * num_blocks * (long)sizeof(int32_t);

    if (fseek(f, 0, SEEK_END) != 0) {
        printf("[debug_print_input_bin] ERROR: fseek to end failed\n");
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        printf("[debug_print_input_bin] ERROR: ftell failed\n");
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    if (file_size < header_bytes) {
        printf("[debug_print_input_bin] ERROR: file too small (%ld bytes) for header (%ld bytes)\n",
               file_size, header_bytes);
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    long values_bytes = file_size - header_bytes;
    size_t num_values = (size_t)values_bytes / sizeof(float complex);

    if (values_bytes % (long)sizeof(float complex) != 0) {
        printf("[debug_print_input_bin] WARNING: values_bytes (%ld) not divisible by sizeof(float complex) (%zu)\n",
               values_bytes, sizeof(float complex));
    }

    // 5) Read the complex values
    if (fseek(f, header_bytes, SEEK_SET) != 0) {
        printf("[debug_print_input_bin] ERROR: fseek to values section failed\n");
        free(row_start); free(row_stop); free(col_start); free(col_stop);
        fclose(f);
        return;
    }

    float complex *values = NULL;
    if (num_values > 0) {
        values = malloc(num_values * sizeof *values);
        if (!values) {
            printf("[debug_print_input_bin] ERROR: memory allocation failed for values\n");
            free(row_start); free(row_stop); free(col_start); free(col_stop);
            fclose(f);
            return;
        }

        if (fread(values, sizeof(float complex), num_values, f) != num_values) {
            printf("[debug_print_input_bin] ERROR: could not read values array\n");
            free(row_start); free(row_stop); free(col_start); free(col_stop);
            free(values);
            fclose(f);
            return;
        }
    }

    fclose(f);

    // 6) Print a nice summary

    printf("=========================================================\n");
    printf("Debug print of binary matrix file: %s\n", path);
    printf("---------------------------------------------------------\n");
    printf("num_blocks = %d\n", num_blocks);
    printf("num_values = %zu (computed from file size)\n", num_values);
    printf("---------------------------------------------------------\n");

    int show = (num_blocks < 44) ? num_blocks : 44;

    printf("row_start[0..%d]:\n  ", show);
    for (int i = 0; i < show; ++i) {
        printf("%d ", row_start[i]);
    }
    if (num_blocks > show) printf("... (%d more)", num_blocks - show);
    printf("\n\n");

    printf("row_stop[0..%d]:\n  ", show);
    for (int i = 0; i < show; ++i) {
        printf("%d ", row_stop[i]);
    }
    if (num_blocks > show) printf("... (%d more)", num_blocks - show);
    printf("\n\n");

    printf("col_start[0..%d]:\n  ", show);
    for (int i = 0; i < show; ++i) {
        printf("%d ", col_start[i]);
    }
    if (num_blocks > show) printf("... (%d more)", num_blocks - show);
    printf("\n\n");

    printf("col_stop[0..%d]:\n  ", show);
    for (int i = 0; i < show; ++i) {
        printf("%d ", col_stop[i]);
    }
    if (num_blocks > show) printf("... (%d more)", num_blocks - show);
    printf("\n\n");

    // Optional: compute expected total values from block shapes
    // assuming ranges are [start, stop) (end-exclusive).
    size_t expected_values = 0;
    for (int b = 0; b < num_blocks; ++b) {
        int rows = row_stop[b] - row_start[b] + 1;
        int cols = col_stop[b] - col_start[b] + 1;
        if (rows <= 0 || cols <= 0) {
            printf("[debug_print_input_bin] WARNING: block %d has nonpositive size (%d x %d)\n",
                   b, rows, cols);
        } else {
            expected_values += (size_t)rows * (size_t)cols;
        }
    }
    printf("Expected num_values from index ranges ([start,stop]) = %zu\n",
           expected_values);
    if (expected_values != num_values) {
        printf("[debug_print_input_bin] WARNING: expected_values != num_values\n");
    }
    printf("---------------------------------------------------------\n");

    // Print a small sample of the complex values
    if (values && num_values > 0) {
        size_t show_vals = (num_values < 10) ? num_values : 10;
        printf("First %zu values (flattened):\n", show_vals);
        for (size_t i = 0; i < show_vals; ++i) {
            float complex v = values[i];
            printf("  values[%zu] = (%g + %gi)\n", i,
                   crealf(v), cimagf(v));
        }
        if (num_values > show_vals) {
            printf("  ... (%zu more values)\n", num_values - show_vals);
        }
    } else {
        printf("No values data present (num_values = %zu)\n", num_values);
    }

    printf("=========================================================\n");

    // 7) Cleanup
    free(row_start); free(row_stop); free(col_start); free(col_stop);
    free(values);
}