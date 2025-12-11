# Compiler and flags
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -MMD -MP \
         -I/usr/include \
         -I/usr/include/lapacke \
         -I/usr/include/openblas

# Libraries
LDLIBS = -lopenblas -llapacke -llapack -lm

# Optional command-line arguments to pass to ./demo via `make run ARGS="..."`
ARGS ?=

# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------

# Core sources used by both demo and tests
CORE_SRCS = block_sparse_format.c \
            test_matrix.c \
            dense_functions.c \
            check_correctness.c \
            block_sparse_test_runner.c

# Demo program
APP_SRCS  = $(CORE_SRCS) main.c

# Test program (driver + core + helpers)
TEST_SRCS = $(CORE_SRCS) run_tests.c

APP_OBJS   = $(APP_SRCS:.c=.o)
TEST_OBJS  = $(TEST_SRCS:.c=.o)

DEPS       = $(APP_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

# Target executables
TARGET      = demo
TEST_TARGET = run_tests

.PHONY: all run test debug clean

# Default rule
all: $(TARGET)

# ----------------------------------------------------------------------
# Link targets
# ----------------------------------------------------------------------

# Link demo
$(TARGET): $(APP_OBJS)
	$(CC) $(CFLAGS) -o $@ $(APP_OBJS) $(LDLIBS)

# Link test runner
$(TEST_TARGET): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LDLIBS)

# ----------------------------------------------------------------------
# Test target
# ----------------------------------------------------------------------

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# ----------------------------------------------------------------------
# Compilation and deps
# ----------------------------------------------------------------------

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------

run: $(TARGET)
	./$(TARGET) $(ARGS)

debug: CFLAGS += -g -O0
debug: clean all

clean:
	rm -f $(APP_OBJS) $(TEST_OBJS) $(DEPS) $(TARGET) $(TEST_TARGET)
