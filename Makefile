# Compiler and Flags
CC       := gcc
# CFLAGS: -Wall (warnings), -O2 (optimization), -Isrc/api (include path for headers), -g (debug symbols)
# CFLAGS   := -Wall -O3 -Isrc/api -fsanitize=address -g -Wextra
CFLAGS   := -Wall -O3 -Isrc/api -g -Wextra
# LDFLAGS: -lm (link math library for functions like sqrtf, powf, log2f, etc.)
LDFLAGS  := -lm

# Directories
API_SRCDIR  := src/api
TEST_SRCDIR := src/tests
OBJDIR      := obj
BINDIR      := build

# Find source files
API_SRC     := $(wildcard $(API_SRCDIR)/*.c)
TEST_SRC    := $(wildcard $(TEST_SRCDIR)/*.c)

# Generate object file names from API source files
API_OBJ     := $(patsubst $(API_SRCDIR)/%.c,$(OBJDIR)/%.o,$(API_SRC))

# Generate executable names from test source files (e.g., src/tests/foo.c -> bin/foo)
TEST_EXECS  := $(patsubst $(TEST_SRCDIR)/%.c,$(BINDIR)/%,$(TEST_SRC))

# Default target: build all test executables
.PHONY: all
all: $(TEST_EXECS)

# Rule to link a test executable
# Depends on the specific test source file and *all* API object files
# Uses order-only prerequisite for the bin directory
$(BINDIR)/%: $(TEST_SRCDIR)/%.c $(API_OBJ) | $(BINDIR)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Rule to compile an API source file into an object file
# Depends on the source file and any header in the api directory (simple dependency)
# Uses order-only prerequisite for the obj directory
$(OBJDIR)/%.o: $(API_SRCDIR)/%.c $(wildcard $(API_SRCDIR)/*.h) | $(OBJDIR)
	@echo "Compiling $<..."
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

# --- REMOVED THE PROBLEMATIC LINE ---
# $(TEST_EXECS): % : $(BINDIR)/%
#	@echo "$@ built."
# --- END REMOVAL ---

# Example of how to add a run target for a specific test
.PHONY: run-digits-test-run
run-digits-test-run: $(BINDIR)/digits-test-run
	@echo "Running digits-test-run..."
	./$(BINDIR)/digits-test-run

.PHONY: run-digits-test-run-old
run-digits-test-run-old: $(BINDIR)/digits-test-run-old
	@echo "Running digits-test-run-old..."
	./$(BINDIR)/digits-test-run-old

.PHONY: run-xor-test-run
run-xor-test-run: $(BINDIR)/xor-test-run
	@echo "Running xor-test-run..."
	./$(BINDIR)/xor-test-run