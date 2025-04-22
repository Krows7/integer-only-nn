# Compiler and Flags
CC       := gcc
# CFLAGS: -Wall (warnings), -O2 (optimization), -Isrc/api (include path for headers), -g (debug symbols)
# CFLAGS   := -Wall -O3 -Isrc/api -Isrc/ext -fsanitize=address -g -Wextra
CFLAGS   := -Wall -O3 -Isrc/api -Isrc/ext -g -Wextra
# LDFLAGS: -lm (link math library for functions like sqrtf, powf, log2f, etc.)
LDFLAGS  := -lm

# Directories
SRCDIR		:= src
API_SRCDIR  := $(SRCDIR)/api
EXT_SRCDIR	:= $(SRCDIR)/ext
TEST_SRCDIR := $(SRCDIR)/tests
OBJDIR      := obj
BINDIR      := build

# Find source files
API_SRC     := $(wildcard $(API_SRCDIR)/*.c)
EXT_SRC		:= $(wildcard $(EXT_SRCDIR)/*.c)
TEST_SRC    := $(wildcard $(TEST_SRCDIR)/*.c)

# Generate object file names from API source files
API_OBJ     := $(patsubst $(API_SRCDIR)/%.c,$(OBJDIR)/%.o,$(API_SRC))

EXT_OBJ     := $(patsubst $(EXT_SRCDIR)/%.c,$(OBJDIR)/%.o,$(EXT_SRC))

# Generate executable names from test source files (e.g., src/tests/foo.c -> bin/foo)
TEST_EXECS  := $(patsubst $(TEST_SRCDIR)/%.c,$(BINDIR)/%,$(TEST_SRC))

# Default target: build all test executables
.PHONY: all
all: $(TEST_EXECS)

# Rule to link a test executable
# Depends on the specific test source file and *all* API object files
# Uses order-only prerequisite for the bin directory
$(BINDIR)/%: $(TEST_SRCDIR)/%.c $(API_OBJ) $(EXT_OBJ) | $(BINDIR)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Rule to compile an API source file into an object file
# Depends on the source file and potentially headers in API dir
$(OBJDIR)/%.o: $(API_SRCDIR)/%.c $(wildcard $(API_SRCDIR)/*.h) | $(OBJDIR)
	@echo "Compiling (API) $< -> $@..."
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile an EXT source file into an object file
# Depends on the source file and potentially headers in EXT dir
$(OBJDIR)/%.o: $(EXT_SRCDIR)/%.c $(wildcard $(EXT_SRCDIR)/*.h) | $(OBJDIR)
	@echo "Compiling (EXT) $< -> $@..."
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to create the object directory if it doesn't exist
$(OBJDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

# Rule to create the bin directory if it doesn't exist
$(BINDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

# Clean target: remove object files and executables
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Example of how to add a run target for a specific test
.PHONY: run-digits
run-digits: $(BINDIR)/digits
	@echo "Running digits..."
	./$(BINDIR)/digits

# Allow user to specify which binary to debug:
#   make debug BIN=digits
BIN        ?= 
# Extra GDB commands (e.g. -ex "print foo" -ex "print bar")
GDB_EXTRA  ?= --batch -q -ex "run" -ex "bt" -ex "info locals"

.PHONY: debug
debug:
ifeq ($(BIN),)
	$(error Usage: make debug BIN=<binary> [GDB_EXTRA="<extra -ex args>"])
endif
	@echo "→ Debugging $(BIN) in GDB…"
	./run-debug.sh $(BIN)