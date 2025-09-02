# Compiler and flags
CC     = gcc
CFLAGS = -Wall -Wextra -O2

# Librarys
LDLIBS = -lopenblas -llapacke -lm

# Sources and objects
SRCS = block_sparse_format.c main.c
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = demo

# Default rule
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

# Compile .c to .o
%.o: %.c block_sparse_format.h
	$(CC) $(CFLAGS) -c $< -o $@

# Run the demo
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -O0
debug: clean all

# Cleanup
clean:
	rm -f $(OBJS) $(TARGET)
