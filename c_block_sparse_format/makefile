# Compiler and flags
CC      = gcc
CFLAGS = -Wall -Wextra -O2 -MMD -MP \
         -I/usr/include \
         -I/usr/include/lapacke \
         -I/usr/include/openblas

# Libraries
LDLIBS = -lopenblas -llapacke -llapack -lm

# Sources and objects
SRCS    = block_sparse_format.c test_matrix.c dense_functions.c check_correctness.c block_sparse_test_runner.c main.c
OBJS    = $(SRCS:.c=.o)
DEPS    = $(OBJS:.o=.d)

# Target executable
TARGET  = demo

.PHONY: all run debug clean

# Default rule
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

# Compile .c to .o (auto-generates .d dependency files)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Include auto-generated dependency files (if present)
-include $(DEPS)

# Run the demo
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -O0
debug: clean all

# Cleanup
clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)
