CC       := gcc
BASE_CFLAGS := -Wall -O3 -g -Wextra
LDFLAGS  := -lm

SRCDIR      := src
API_SRCDIR  := $(SRCDIR)/api
EXT_SRCDIR  := $(SRCDIR)/ext
NES_SRCDIR  := $(SRCDIR)/nes
TEST_SRCDIR := $(SRCDIR)/tests
OBJDIR      := obj
BINDIR      := build

API_OBJDIR     := $(OBJDIR)/api
API_NES_OBJDIR := $(OBJDIR)/api_nes
EXT_OBJDIR     := $(OBJDIR)/ext
NES_OBJDIR     := $(OBJDIR)/nes
EXT_BINDIR     := $(BINDIR)/ext
NES_BINDIR     := $(BINDIR)/nes

API_SRC     := $(wildcard $(API_SRCDIR)/*.c)
EXT_IMPL_SRC := $(wildcard $(EXT_SRCDIR)/*.c)
NES_IMPL_SRC := $(wildcard $(NES_SRCDIR)/*.c)
TEST_EXT_SRC := $(wildcard $(TEST_SRCDIR)/ext/*.c)
TEST_NES_SRC := $(wildcard $(TEST_SRCDIR)/nes/*.c)

API_OBJ     := $(patsubst $(API_SRCDIR)/%.c,$(API_OBJDIR)/%.o,$(API_SRC))
API_NES_OBJ := $(patsubst $(API_SRCDIR)/%.c,$(API_NES_OBJDIR)/%.o,$(API_SRC))
EXT_IMPL_OBJ := $(patsubst $(EXT_SRCDIR)/%.c,$(EXT_OBJDIR)/%.o,$(EXT_IMPL_SRC))
NES_IMPL_OBJ := $(patsubst $(NES_SRCDIR)/%.c,$(NES_OBJDIR)/%.o,$(NES_IMPL_SRC))

TEST_EXT_EXECS := $(patsubst $(TEST_SRCDIR)/ext/%.c,$(EXT_BINDIR)/%,$(TEST_EXT_SRC))
TEST_NES_EXECS := $(patsubst $(TEST_SRCDIR)/nes/%.c,$(NES_BINDIR)/%,$(TEST_NES_SRC))

.PHONY: all
all: $(TEST_EXT_EXECS) $(TEST_NES_EXECS)

$(EXT_BINDIR)/%: $(TEST_SRCDIR)/ext/%.c $(API_OBJ) $(EXT_IMPL_OBJ) | $(EXT_BINDIR)
	@echo "Linking (EXT) $@..."
	$(CC) $(BASE_CFLAGS) -I$(API_SRCDIR) -I$(EXT_SRCDIR) $^ $(LDFLAGS) -o $@

$(NES_BINDIR)/%: $(TEST_SRCDIR)/nes/%.c $(API_NES_OBJ) $(NES_IMPL_OBJ) | $(NES_BINDIR)
	@echo "Linking (NES) $@..."
	$(CC) $(BASE_CFLAGS) -DNES -I$(API_SRCDIR) -I$(NES_SRCDIR) $^ $(LDFLAGS) -o $@

$(API_OBJDIR)/%.o: $(API_SRCDIR)/%.c $(wildcard $(API_SRCDIR)/*.h) | $(API_OBJDIR)
	@echo "Compiling (API for EXT) $< -> $@..."
	$(CC) $(BASE_CFLAGS) -I$(API_SRCDIR) -c $< -o $@

$(API_NES_OBJDIR)/%.o: $(API_SRCDIR)/%.c $(wildcard $(API_SRCDIR)/*.h) | $(API_NES_OBJDIR)
	@echo "Compiling (API for NES) $< -> $@..."
	$(CC) $(BASE_CFLAGS) -DNES -I$(API_SRCDIR) -c $< -o $@

$(EXT_OBJDIR)/%.o: $(EXT_SRCDIR)/%.c $(wildcard $(EXT_SRCDIR)/*.h) $(wildcard $(API_SRCDIR)/*.h) | $(EXT_OBJDIR)
	@echo "Compiling (EXT Impl) $< -> $@..."
	$(CC) $(BASE_CFLAGS) -I$(API_SRCDIR) -I$(EXT_SRCDIR) -c $< -o $@

$(NES_OBJDIR)/%.o: $(NES_SRCDIR)/%.c $(wildcard $(NES_SRCDIR)/*.h) $(wildcard $(API_SRCDIR)/*.h) | $(NES_OBJDIR)
	@echo "Compiling (NES Impl) $< -> $@..."
	$(CC) $(BASE_CFLAGS) -DNES -I$(API_SRCDIR) -I$(NES_SRCDIR) -c $< -o $@

$(API_OBJDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

$(API_NES_OBJDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

$(EXT_OBJDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

$(NES_OBJDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

$(EXT_BINDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

$(NES_BINDIR):
	@echo "Creating directory $@..."
	@mkdir -p $@

.PHONY: clean
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJDIR) $(BINDIR)

.PHONY: run-ext-digits
run-ext-digits: $(EXT_BINDIR)/digits
	@echo "Running ext/digits..."
	./$(EXT_BINDIR)/digits

BIN        ?=
GDB_EXTRA  ?= --batch -q -ex "run" -ex "bt" -ex "info locals"

.PHONY: debug
debug:
ifeq ($(BIN),)
	$(error Usage: make debug BIN=<subdir>/<binary> [GDB_EXTRA="<extra -ex args>"])
endif
	@echo "→ Debugging $(BINDIR)/$(BIN) in GDB…"
	./run-debug.sh $(BIN)

