# ==== Compilers ====
CC      = gcc
NVCC    = nvcc                          

# ==== CUDA paths (adjust CUDA_HOME if needed) ====
CUDA_HOME ?= /usr/local/cuda
CUDA_INC   = $(CUDA_HOME)/include
CUDA_LIB   = $(CUDA_HOME)/lib64

# ==== Flags ====
CFLAGS = -Wall -Wextra -O2 -MMD -MP \
         -I/usr/include \
         -I/usr/include/lapacke \
         -I/usr/include/openblas \
         -I$(CUDA_INC)

NVCCFLAGS = -O2 -Xcompiler "-Wall -Wextra" -I$(CUDA_INC)

# ==== Libraries ====
LDLIBS = -L$(CUDA_LIB) -lcudart -lcublas \
         -lopenblas -llapacke -llapack -lm

# ==== Sources and objects ====
SRCS_C   = block_sparse_format.c test_matrix.c block_sparse_test_runner.c load_binary_data.c main.c
SRCS_CU  = kernels.cu              

OBJS_C   = $(SRCS_C:.c=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)         

OBJS     = $(OBJS_C) $(OBJS_CU)      
DEPS_C   = $(OBJS_C:.o=.d)
DEPS_CU  = $(OBJS_CU:.o=.d)

# ==== Target executable ====
TARGET  = demo

.PHONY: all run debug clean

# ==== Default rule ====
all: $(TARGET)

# ==== Link (use NVCC so cudart is linked correctly) ====
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDLIBS)

# ==== Compile .c to .o (auto-generates .d dependency files) ====
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ==== Compile .cu to .o (no .d to avoid tmpxft issue) ====
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==== Include dependency files ====
-include $(DEPS_C) $(DEPS_CU)

# ==== Run the demo ====
run: $(TARGET)
	./$(TARGET)

# ==== Debug build ====
debug: CFLAGS += -g -O0
debug: NVCCFLAGS += -G -Xcompiler "-g -O0"
debug: clean all

# ==== Cleanup ====
clean:
	rm -f $(OBJS) $(DEPS_C) $(DEPS_CU) $(TARGET)
