# Compiler and flags
CC      = gcc
NVCC    = nvcc                          
CFLAGS  = -Wall -Wextra -O2 -MMD -MP
NVCCFLAGS = -O2 -Xcompiler "-Wall -Wextra"  

# Libraries
LDLIBS = -lopenblas -llapacke -lm

# Sources and objects
SRCS_C   = block_sparse_format.c test_matrix.c block_sparse_test_runner.c main.c
SRCS_CU  = kernels.cu              

OBJS_C   = $(SRCS_C:.c=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)         

OBJS     = $(OBJS_C) $(OBJS_CU)      
DEPS     = $(OBJS_C:.o=.d)           # only C files generate deps

# Target executable
TARGET  = demo

.PHONY: all run debug clean

# Default rule
all: $(TARGET)

# Link (use NVCC so cudart is linked automatically)
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDLIBS)

# Compile .c to .o (auto-generates .d dependency files)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile .cu to .o (no .d to avoid tmpxft issue)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Include dependency files
-include $(DEPS)

# Run the demo
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -O0
debug: NVCCFLAGS += -G -Xcompiler "-g -O0"  
debug: clean all

# Cleanup
clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)
