# Compilers
CC      = gcc
NVCC    = nvcc

# Flags
CFLAGS  = -Wall -Wextra -O2
NVCCFLAGS = -O2

# Target
TARGET  = demo

# Sources and objects
SRCS_C   = main.c
SRCS_CU  = kernels.cu
OBJS_C   = $(SRCS_C:.c=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)
OBJS     = $(OBJS_C) $(OBJS_CU)

# Libraries (add more if needed)
LDLIBS = -lm

# Default rule
all: $(TARGET)

# Link step — use nvcc so CUDA runtime is linked automatically
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDLIBS)

# Compile C source
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(OBJS) $(TARGET)
