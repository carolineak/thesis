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
         -lcusolver \
         -lopenblas -llapacke -llapack -lm

# Optional command-line arguments for the demo binary
ARGS ?=

# ==== Sources and objects (C) ====
# Core C sources used by demo and both test drivers
CORE_SRCS_C = block_sparse_format.c \
              test_matrix.c \
              block_sparse_test_runner.c \
              load_binary_data.c

# Demo program sources
APP_SRCS_C  = $(CORE_SRCS_C) main.c

# Test program sources
TEST_SRCS_C = $(CORE_SRCS_C) run_tests.c

# Binary test program sources
BINARY_TEST_SRCS_C = $(CORE_SRCS_C) run_binary_tests.c

# ==== Sources and objects (CUDA) ====
SRCS_CU  = kernels.cu

# Object files
APP_OBJS_C         = $(APP_SRCS_C:.c=.o)
TEST_OBJS_C        = $(TEST_SRCS_C:.c=.o)
BINARY_TEST_OBJS_C = $(BINARY_TEST_SRCS_C:.c=.o)
OBJS_CU            = $(SRCS_CU:.cu=.o)

DEPS_C  = $(APP_OBJS_C:.o=.d) $(TEST_OBJS_C:.o=.d) $(BINARY_TEST_OBJS_C:.o=.d)
DEPS_CU = $(OBJS_CU:.o=.d)

# ==== Target executables ====
TARGET             = demo
TEST_TARGET        = run_tests
BINARY_TEST_TARGET = run_binary_tests

.PHONY: all run test binary_test debug clean

# ==== Default rule ====
all: $(TARGET)

# ==== Link (use NVCC so cudart is linked correctly) ====
$(TARGET): $(APP_OBJS_C) $(OBJS_CU)
	$(NVCC) -o $@ $(APP_OBJS_C) $(OBJS_CU) $(LDLIBS)

$(TEST_TARGET): $(TEST_OBJS_C) $(OBJS_CU)
	$(NVCC) -o $@ $(TEST_OBJS_C) $(OBJS_CU) $(LDLIBS)

$(BINARY_TEST_TARGET): $(BINARY_TEST_OBJS_C) $(OBJS_CU)
	$(NVCC) -o $@ $(BINARY_TEST_OBJS_C) $(OBJS_CU) $(LDLIBS)

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
	./$(TARGET) $(ARGS)

# ==== Run tests ====
test: $(TEST_TARGET)
	./$(TEST_TARGET)

binary_test: $(BINARY_TEST_TARGET)
	./$(BINARY_TEST_TARGET)

# ==== Debug build ====
debug: CFLAGS += -g -O0
debug: NVCCFLAGS += -G -Xcompiler "-g -O0"
debug: clean all

# ==== Cleanup ====
clean:
	rm -f $(APP_OBJS_C) $(TEST_OBJS_C) $(BINARY_TEST_OBJS_C) \
	      $(OBJS_CU) $(DEPS_C) $(DEPS_CU) \
	      $(TARGET) $(TEST_TARGET) $(BINARY_TEST_TARGET)
