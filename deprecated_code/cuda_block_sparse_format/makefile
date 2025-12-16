# Makefile for compiling main.cu with dependency on block_sparse_format.cu.h

CUDA_NVCC := nvcc
TARGET := main
SRC := main.cu
DEPS := block_sparse_format.cu.h test_matrix.cu.h

$(TARGET): $(SRC) $(DEPS)
	$(CUDA_NVCC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# TODO: Fix dependencies
