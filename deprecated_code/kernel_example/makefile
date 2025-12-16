# ==== Compilers ====
CC      = gcc
NVCC    = nvcc

# ==== CUDA root (adjust if needed) ====
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64

# ==== Flags ====
CFLAGS    = -Wall -Wextra -O2 -I$(CUDA_INC)
NVCCFLAGS = -O2 -I$(CUDA_INC)

# ==== Files ====
TARGET  = demo
SRCS_C   = main.c
SRCS_CU  = kernels.cu
OBJS_C   = $(SRCS_C:.c=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)
OBJS     = $(OBJS_C) $(OBJS_CU)

# ==== Libraries ====
LDFLAGS =
LDLIBS  = -L$(CUDA_LIB) -lcudart -lcublas -lm

# ==== Default rule ====
all: $(TARGET)

# ==== Linking ====
# Use nvcc for linking so CUDA runtime is correctly included
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

# ==== Compile C sources ====
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ==== Compile CUDA sources ====
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==== Run ====
run: $(TARGET)
	./$(TARGET)

# ==== Clean ====
clean:
	rm -f $(OBJS) $(TARGET)
