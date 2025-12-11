# Compiler and flags
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -MMD -MP \
         -I/usr/include \
         -I/usr/include/lapacke \
         -I/usr/include/openblas

# Libraries
LDLIBS = -lopenblas -llapacke -llapack -lm

# Optional command-line arguments for the demo binary
ARGS ?=

# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------

# Core sources used by both demo and tests
CORE_SRCS = block_sparse_format.c \
            test_matrix.c \
            block_sparse_test_runner.c \
            load_binary_data.c

# Demo program (normal executable)
APP_SRCS  = $(CORE_SRCS) main.c

# Test program (separate test driver)
TEST_SRCS = $(CORE_SRCS) run_tests.c

# Binary test program (another class of tests)
BINARY_TEST_SRCS = $(CORE_SRCS) run_binary_tests.c

APP_OBJS         = $(APP_SRCS:.c=.o)
TEST_OBJS        = $(TEST_SRCS:.c=.o)
BINARY_TEST_OBJS = $(BINARY_TEST_SRCS:.c=.o)

DEPS = $(APP_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BINARY_TEST_OBJS:.o=.d)

# Target executables
TARGET             = demo
TEST_TARGET        = run_tests
BINARY_TEST_TARGET = run_binary_tests

.PHONY: all run test binary_test debug clean

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

# Link binary test runner
$(BINARY_TEST_TARGET): $(BINARY_TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $(BINARY_TEST_OBJS) $(LDLIBS)

# ----------------------------------------------------------------------
# Test targets
# ----------------------------------------------------------------------

test: $(TEST_TARGET)
	./$(TEST_TARGET)

binary_test: $(BINARY_TEST_TARGET)
	./$(BINARY_TEST_TARGET)

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
	rm -f $(APP_OBJS) $(TEST_OBJS) $(BINARY_TEST_OBJS) $(DEPS) \
	      $(TARGET) $(TEST_TARGET) $(BINARY_TEST_TARGET)
