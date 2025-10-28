# ==== Compilers ====
CC      = gcc
NVCC    = nvcc

# ==== Flags ====
CFLAGS    = -Wall -Wextra -O2
NVCCFLAGS = -O2

# ==== Files ====
TARGET  = demo
SRCS_C   = main.c
SRCS_CU  = kernels.cu
OBJS_C   = $(SRCS_C:.c=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)
OBJS     = $(OBJS_C) $(OBJS_CU)

# ==== Libraries ====
LDLIBS = -lcudart -lm

# ==== Default rule ====
all: $(TARGET)

# ==== Linking ====
# Use nvcc for linking so CUDA runtime is correctly included
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDLIBS)

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
