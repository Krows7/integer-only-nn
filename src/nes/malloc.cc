#include <stdlib.h>

#include <assert.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// --- Add these logging helpers ---
#define MALLOC_LOG_P401B ((volatile unsigned char*)0x401B)
#define MALLOC_LOG_P4018 ((volatile unsigned char*)0x4018)
#define MALLOC_LOG_P4019 ((volatile unsigned char*)0x4019)
#define MALLOC_LOG_P401A ((volatile unsigned char*)0x401A)

static void MLOG_STR(const char* s) {
    // Simple version: assumes s is null-terminated and contains newlines if desired by Lua
    // Or, send newline explicitly if Lua script requires it per string.
    // Your Lua cb flushes on '\n' or >80 chars. Let's ensure newline.
    // const char* p = s;
    // while (*p) {
    //     *MALLOC_LOG_P401B = *p++;
    // }
    // *MALLOC_LOG_P401B = '\n'; // Ensure Lua script logs this line
}

static void MLOG_U16(const char* label, uint16_t value) {
    // MLOG_STR(label); // Prints "label\n"
    // *MALLOC_LOG_P4018 = (unsigned char)(value & 0xFF);
    // *MALLOC_LOG_P4019 = (unsigned char)((value >> 8) & 0xFF);
    // *MALLOC_LOG_P401A = 1; // Triggers emu.log("Word: " .. value) in Lua
}

static void MLOG_PTR(const char* label, const void* ptr_val) {
    // MLOG_STR(label);
    // uint16_t value = (uint16_t)(uintptr_t)ptr_val;
    // *MALLOC_LOG_P4018 = (unsigned char)(value & 0xFF);
    // *MALLOC_LOG_P4019 = (unsigned char)((value >> 8) & 0xFF);
    // *MALLOC_LOG_P401A = 1;
}
// --- End of logging helpers ---


#define NDEBUG

#ifndef NDEBUG
#define TRACE(fmt, ...)                                                        \
  printf(fmt __VA_OPT__(, ) __VA_ARGS__)
#else
#define TRACE(fmt, ...) (void)0
#endif

extern char __heap_start;
extern char __heap_default_limit;

namespace {

class FreeChunk;

class Chunk {
public:
  char prev_free : 1;

private:
  size_t size_hi : 15;

public:
  size_t size() const { return size_hi << 1; }
  void set_size(size_t size) { size_hi = size >> 1; }

  void *end() const { return (char *)this + size(); }

  // Returns the previous chunk. Undefined behavior if the previous chunk is not
  // free. Legal to call on a pointer to heap end.
  FreeChunk *prev() const;

  Chunk *next() const;

  // Slow; prefer prev_free where possible.
  bool free() const;

  void set_free(bool free);
};

class FreeChunk : public Chunk {
public:
  struct FreeChunk *free_list_next;
  struct FreeChunk *free_list_prev;
  // char filler[...];
  // size_t trailing_size;

  // Initialize a region of memory as a free chunk, add it to the free list, and
  // return it.
  static FreeChunk *insert(void *begin, size_t size);

  size_t &trailing_size() {
    return *(size_t *)((char *)end() - sizeof(size_t));
  }

  size_t avail_size() const { return size() - sizeof(Chunk); }

  // Remove from the free list.
  void remove();
};

// Free chunks smaller than this cannot be allocated, and in-use chunks smaller
// than this cannot be freed if surrounded by in-use chunks. Accordingly, this
// acts as a global minimum chunk size.
constexpr size_t MIN_CHUNK_SIZE = sizeof(FreeChunk) + sizeof(size_t);

__attribute__((section(".data"))) volatile size_t heap_limit = (size_t)&__heap_default_limit;

Chunk *heap_end() {
  return reinterpret_cast<Chunk *>(&__heap_start + heap_limit);
}

// The sum total available size on the free list.
__attribute__((section(".bss"))) volatile size_t free_size;

// A circularly-linked list of free chunks ordered by decreasing age. nullptr if
// empty.
__attribute__((section(".bss"))) volatile FreeChunk *free_list;

// Free-ness is tracked by the next chunk's prev_free field, but the last chunk
// has no next chunk.
__attribute__((section(".bss"))) volatile bool last_free;

__attribute__((section(".bss"))) volatile bool initialized;

Chunk *Chunk::next() const {
  Chunk *next = reinterpret_cast<Chunk *>(end());
  return next != heap_end() ? next : nullptr;
}

FreeChunk *Chunk::prev() const {
  size_t prev_size = *reinterpret_cast<size_t *>((char *)this - sizeof(size_t));
  return reinterpret_cast<FreeChunk *>((char *)this - prev_size);
}

bool Chunk::free() const {
  Chunk *n = next();
  return n ? n->prev_free : last_free;
}

void Chunk::set_free(bool free) {
  Chunk *n = next();
  if (n)
    n->prev_free = free;
  else
    last_free = free;
}

void FreeChunk::remove() {
  MLOG_STR("--- FreeChunk::remove ---");
  MLOG_PTR("remove_this_addr", this);
  MLOG_U16("remove_this_hdr_size", this->size());     // From chunk's metadata
  MLOG_U16("remove_this_avail_size", this->avail_size()); // Calculated. Is this 0x8000?
  MLOG_U16("remove_free_size_b4_sub", free_size);    // Should be 0x13AC before corruption

  free_size -= avail_size(); // THE SUSPECTED LINE
  MLOG_U16("remove_free_size_after_sub", free_size); // Does this become 0x93AC?

  if (free_list_next == this) {
    TRACE("Free list emptied.\n");
    free_list = nullptr;
    return;
  }

  free_list_prev->free_list_next = free_list_next;
  free_list_next->free_list_prev = free_list_prev;
  if (free_list == this)
    free_list = free_list_next;

  // ... rest of the function with MLOG_STR for important path decisions ...
  MLOG_STR("--- FreeChunk::remove done ---");
}

FreeChunk *FreeChunk::insert(void *begin, size_t total_chunk_size) {
  MLOG_STR("--- FreeChunk::insert ---");
  MLOG_PTR("insert_begin_addr", begin);
  MLOG_U16("insert_total_chunk_size_arg", total_chunk_size);

  FreeChunk *new_free_chunk = (FreeChunk *)begin;
  new_free_chunk->set_size(total_chunk_size);

  // Check for potential OOB write if total_chunk_size is corrupt & large
  uintptr_t chunk_intended_end = (uintptr_t)begin + total_chunk_size;
  uintptr_t heap_actual_end = (uintptr_t)(&__heap_start) + heap_limit;
  if (chunk_intended_end > heap_actual_end) {
      MLOG_STR("!!! insert_trailing_size_WRITE_OOB_WARN !!!");
      MLOG_U16("insert_heap_actual_end_addr", (uint16_t)heap_actual_end);
      MLOG_U16("insert_chunk_computed_end_addr", (uint16_t)chunk_intended_end);
  }
  new_free_chunk->trailing_size() = total_chunk_size; // This write could be dangerous if total_chunk_size is huge

  MLOG_U16("insert_new_chunk_hdr_size_set", new_free_chunk->size());
  MLOG_U16("insert_new_chunk_avail_calc", new_free_chunk->avail_size());
  MLOG_U16("insert_free_size_b4_add", free_size);
  free_size += new_free_chunk->avail_size();
  MLOG_U16("insert_free_size_after_add", free_size);

  if (!free_list) {
    free_list = new_free_chunk->free_list_next = new_free_chunk->free_list_prev = new_free_chunk;
    return new_free_chunk;
  }

  // Insert to the end of the free list, so that allocations from the front
  // occur in FIFO order (first fit FIFO in Wilson et al).
  new_free_chunk->free_list_next = (FreeChunk*) free_list;
  new_free_chunk->free_list_prev = free_list->free_list_prev;
  new_free_chunk->free_list_prev->free_list_next = new_free_chunk;
  new_free_chunk->free_list_next->free_list_prev = new_free_chunk;

  // ... rest of the function with MLOG_STR for path decisions ...
  MLOG_STR("--- FreeChunk::insert done ---");
  return new_free_chunk;
}

// Find the first chunk in the free list that can successfully fit a new chunk
// of the given size.
FreeChunk *find_fit(size_t size_needed) { // size_needed is total chunk size
  MLOG_STR("--- find_fit ---");
  MLOG_U16("ff_looking_for_total_size", size_needed);

  if (!free_list) { return nullptr; }

  bool first = true; int loop_count = 0; const int MAX_LOOP = 20; // Safety for corrupted list
  for (FreeChunk *chunk = (FreeChunk*) free_list; 
       (first || chunk != (FreeChunk*) free_list) && loop_count < MAX_LOOP;
       chunk = chunk->free_list_next, first = false ) {
    MLOG_PTR("ff_considering_chunk_addr", chunk);
    MLOG_U16("ff_chunk_actual_hdr_size", chunk->size()); // VERY IMPORTANT: Is this already 0x8002?
    MLOG_U16("ff_chunk_actual_avail_size", chunk->avail_size());

    if (size_needed <= chunk->size()) { return chunk; }
    ++loop_count;
  }
  /* ... */
  return nullptr;
}

void *allocate_free_chunk(FreeChunk *free_chunk_to_carve_from, size_t new_busy_chunk_total_size) {
  MLOG_STR("--- allocate_free_chunk ---");
  MLOG_PTR("afc_carve_from_addr", free_chunk_to_carve_from);
  MLOG_U16("afc_carve_from_hdr_size", free_chunk_to_carve_from->size()); // From its metadata
  MLOG_U16("afc_carve_from_avail", free_chunk_to_carve_from->avail_size()); // Calculated from above
  MLOG_U16("afc_new_busy_total_size_req", new_busy_chunk_total_size);
  MLOG_U16("afc_free_size_b4_remove", free_size);

  free_chunk_to_carve_from->remove(); // remove() will log its internals
  MLOG_U16("afc_free_size_after_remove", free_size); // IMPORTANT: Does free_size become 0x93AC here?

  Chunk *busy_chunk_header = free_chunk_to_carve_from;
  size_t original_free_chunk_size = busy_chunk_header->size(); // This is still the size of free_chunk_to_carve_from
  size_t remainder_total_size = original_free_chunk_size - new_busy_chunk_total_size;
  MLOG_U16("afc_remainder_total_size_calc", remainder_total_size);

  if (remainder_total_size < MIN_CHUNK_SIZE) { /* ... */ } 
  else {
    MLOG_STR("afc_splitting_chunk");
    // ...
    FreeChunk* remainder_chunk = FreeChunk::insert(
        (char *)busy_chunk_header + new_busy_chunk_total_size,
        remainder_total_size); // insert() will log
    // ...
    busy_chunk_header->set_size(new_busy_chunk_total_size); // Resize the busy part
  }
  // ...
  MLOG_U16("afc_final_busy_chunk_hdr_size", busy_chunk_header->size());
  MLOG_STR("--- allocate_free_chunk done ---");
  busy_chunk_header->set_free(false);

  TRACE("Allocated size: %u\n", size);

  char *ptr = (char *)busy_chunk_header + sizeof(Chunk);
  TRACE("Allocated ptr: %p\n", ptr);
  return ptr;
}

void init() {
  MLOG_STR("--- Heap Init ---");
  MLOG_U16("sizeof(Chunk)", sizeof(Chunk)); // Crucial check!
  MLOG_U16("sizeof(FreeChunk)", sizeof(FreeChunk));
  MLOG_U16("MIN_CHUNK_SIZE", MIN_CHUNK_SIZE);
  MLOG_PTR("heap_start_addr", &__heap_start);
  MLOG_U16("heap_limit_val", heap_limit);
  MLOG_U16("free_size_b4_init_insert", free_size);

  FreeChunk::insert(&__heap_start, heap_limit)->prev_free = false; // insert() will also log
  MLOG_U16("free_size_after_init_insert", free_size);

  last_free = true;
  initialized = true;
  MLOG_STR("--- Heap Init Done ---");
}


} // namespace

extern "C" {

size_t __heap_limit() { return heap_limit; }

void __set_heap_limit(size_t new_limit) {
  TRACE("__set_heap_limit(%u)\n", new_limit);

  // Chunk sizes must be a multiple of two.
  if (new_limit & 1)
    --new_limit;

  if (!initialized) {
    heap_limit = (new_limit < MIN_CHUNK_SIZE) ? MIN_CHUNK_SIZE : new_limit;
    TRACE("Heap not yet initialized. Set limit to %u.\n", heap_limit);
    return;
  }

  // TODO: We can make this actually shrink the heap too...
  if (new_limit <= heap_limit) {
    TRACE("New limit %u smaller than current %u; returning.", new_limit,
          heap_limit);
    return;
  }

  size_t grow = new_limit - heap_limit;
  TRACE("Growing heap by %u\n", grow);
  if (last_free) {
    FreeChunk *last = heap_end()->prev();
    TRACE("Last chunk free; size %u\n", last->size());
    size_t new_size = last->size() + grow;
    last->set_size(new_size);
    last->trailing_size() = new_size;
    TRACE("Expanded to %u\n", new_size);
    free_size += grow;
  } else {
    TRACE("Last chunk not free.\n");
    if (grow < MIN_CHUNK_SIZE) {
      TRACE("Not enough new size for a chunk; returning.\n");
      return;
    }
    TRACE("Inserting new chunk.\n");
    FreeChunk::insert(heap_end(), grow);
    last_free = true;
  }

  heap_limit = new_limit;
}

size_t __heap_bytes_used() { return heap_limit - free_size; }

size_t __heap_bytes_free() { return free_size; }

// Return the size of chunk needed to hold a malloc request, or zero if
// impossible.
size_t chunk_size_for_malloc(size_t size) {
  if (size <= MIN_CHUNK_SIZE - sizeof(Chunk)) {
    TRACE("Increased size to minimum chunk size %u\n", MIN_CHUNK_SIZE);
    return MIN_CHUNK_SIZE;
  }

  char overhead = sizeof(Chunk);
  if (size & 1)
    overhead++;

  if (__builtin_add_overflow(size, overhead, &size))
    return 0;
  TRACE("Increased size to %u to account for overhead.\n", size);
  return size;
}

void *aligned_alloc(size_t alignment, size_t size) {
  TRACE("aligned_alloc(%u,%u)\n", alignment, size);

  if (alignment <= 2)
    return malloc(size);

  if (!size)
    return nullptr;

  // Only power of two alignments are valid.
  if (alignment & (alignment - 1))
    return nullptr;

  size = chunk_size_for_malloc(size);
  if (!size)
    return nullptr;

  // The region before the aligned chunk needs to be large enough to fit a freereallocated chunk should not be free");
  // chunk.
  if (__builtin_add_overflow(size, MIN_CHUNK_SIZE, &size))
    return nullptr;

  // Up to alignment-1 additional bytes may be needed to align the chunk start.
  if (__builtin_add_overflow(size, alignment - 1, &size))
    return nullptr;

  if (!initialized)
    init();

  FreeChunk *chunk = find_fit(size);
  if (!chunk)
    return nullptr;

  void *aligned_ptr = (char *)chunk + MIN_CHUNK_SIZE;
  TRACE("Initial alignment point: %p\n", aligned_ptr);

  // alignment is a power of two, so alignment-1 is a mask that selects the
  // misaligned bits.
  size_t past_alignment = (uintptr_t)aligned_ptr & (alignment - 1);
  if (past_alignment) {
    TRACE("%u bytes past aligned point.\n", past_alignment);
    aligned_ptr = (void *)((uintptr_t)aligned_ptr & ~(alignment - 1));
    TRACE("Moved pointer backwards to aligned point %p.\n", aligned_ptr);
    aligned_ptr = (char *)aligned_ptr + alignment;
    TRACE("Moved pointer one alignment unit forwards to %p.\n", aligned_ptr);
  }

  size_t chunk_size = chunk->size();

  auto *aligned_chunk_begin = (Chunk *)((char *)aligned_ptr - sizeof(Chunk));
  size_t prev_chunk_size = (char *)aligned_chunk_begin - (char *)chunk;

  TRACE("Inserting free chunk before aligned.\n");
  FreeChunk::insert(chunk, prev_chunk_size); // prev_free remains unchanged.

  TRACE("Temporarily inserting aligned free chunk.\n");
  FreeChunk *aligned_chunk =
      FreeChunk::insert(aligned_chunk_begin, chunk_size - prev_chunk_size);
  aligned_chunk->prev_free = true;

  TRACE("Allocating from aligned free chunk.\n");
  return allocate_free_chunk(aligned_chunk, size);
}

void *calloc(size_t num, size_t size) {
  const auto long_sz = (long)num * size;
  if (long_sz >> 16)
    return 0;
  const auto sz = (size_t)long_sz;
  const auto block = malloc(sz);

  if (!block)
    return nullptr;

  __memset(static_cast<char *>(block), 0, sz);
  return block;
}

void free(void *ptr) {
  TRACE("free(%p)\n", ptr);
  if (!ptr)
    return;

  Chunk *chunk = (Chunk *)((char *)ptr - sizeof(Chunk));
  size_t size = chunk->size();

  TRACE("Freeing chunk %p of size %u\n", chunk, size);

  // Coalesce with prev and next if possible, replacing chunk.
  Chunk *next = chunk->next();
  if (chunk->prev_free) {
    FreeChunk *prev = chunk->prev();
    size_t prev_size = prev->size();
    TRACE("Coalescing with previous free chunk %p of size %u\n", prev,
          prev_size);
    prev->remove();
    size += prev_size;
    chunk = prev;
    TRACE("New chunk %p of size %u\n", chunk, size);
  }
  if (next) {
    size_t next_size = next->size();
    TRACE("Next chunk: %p size %u\n", next, next_size);
    if (next->free()) {
      TRACE("Coalescing with next free chunk %p of size %u\n", next, next_size);
      static_cast<FreeChunk *>(next)->remove();
      size += next_size;
      TRACE("New chunk size %u\n", size);
    } else {
      TRACE("Next chunk not free.\n");
      next->prev_free = true;
    }
  } else {
    TRACE("No next chunk; last chunk now free.\n");
    last_free = true;
  }

  FreeChunk::insert(chunk, size);
}

void *malloc(size_t size) {
  MLOG_STR("--- malloc ---");
  MLOG_U16("malloc_req_payload_size", size);

  if (!size) { return nullptr; }

  size_t total_chunk_size_needed = chunk_size_for_malloc(size);
  if (!total_chunk_size_needed) { return nullptr; }
  MLOG_U16("malloc_adj_total_size", total_chunk_size_needed);

  if (!initialized) { MLOG_STR("malloc_calling_init"); init(); }

  MLOG_U16("malloc_free_size_b4_find", free_size);
  MLOG_PTR("malloc_free_list_b4_find", (const void*)free_list);

  FreeChunk *chunk_to_alloc_from = find_fit(total_chunk_size_needed); // find_fit will log
  
  if (!chunk_to_alloc_from) { return nullptr; }
  MLOG_PTR("malloc_found_chunk_addr", chunk_to_alloc_from);
  MLOG_U16("malloc_found_chunk_hdr_size", chunk_to_alloc_from->size()); // From chunk's metadata
  MLOG_U16("malloc_found_chunk_avail", chunk_to_alloc_from->avail_size());

  void* allocated_ptr = allocate_free_chunk(chunk_to_alloc_from, total_chunk_size_needed); // allocate_free_chunk will log
  MLOG_PTR("malloc_ret_ptr", allocated_ptr);
  MLOG_U16("malloc_free_size_final", free_size);
  MLOG_STR("--- malloc done ---");
  return allocated_ptr;
}


void *realloc(void *ptr, size_t size) {
  TRACE("realloc(%p, %u)\n", ptr, size);

  if (!size)
    return nullptr;
  if (!ptr)
    return malloc(size);

  // Keep original size around for malloc fallback.
  size_t malloc_size = size;

  size = chunk_size_for_malloc(size);
  if (!size)
    return nullptr;

  Chunk *chunk = (Chunk *)((char *)ptr - sizeof(Chunk));
  size_t old_size = chunk->size();
  TRACE("Old size: %u\n", old_size);

  if (size < old_size) {
    size_t shrink = size - old_size;
    TRACE("Shrinking by %u\n", shrink);
    Chunk *next = chunk->next();
    chunk->set_size(size);

    if (next && next->free()) {
      size_t next_size = next->size();
      TRACE("Next free chunk %p size %u\n", next, next_size);
      // Coalesce.
      static_cast<FreeChunk *>(next)->remove();
      FreeChunk::insert(chunk->end(), shrink + next_size)->prev_free = false;
      return ptr;
    }

    // Insert a new free chunk for the shrink if possible.
    if (shrink < MIN_CHUNK_SIZE) {
      TRACE("Remainder too small.");
      return ptr;
    }

    FreeChunk *after = FreeChunk::insert(chunk->end(), shrink);
    TRACE("Allocated remainder %p of size %u\n", after, after->size());
    after->prev_free = false;
    after->set_free(true);
    return ptr;
  }

  if (size == old_size) {
    TRACE("Destination size the same as current; done.\n");
    return ptr;
  }

  size_t grow = size - old_size;
  TRACE("Growing by %u\n", grow);
  Chunk *next = chunk->next();
  if (next) {
    size_t next_size = next->size();
    if (next->free() && grow <= next_size) {
      TRACE("Stealing from next chunk %p free w/ size %u\n", next, next_size);
      chunk->set_size(size);
      static_cast<FreeChunk *>(next)->remove();

      if (next_size - grow < MIN_CHUNK_SIZE) {
        chunk->set_size(old_size + next_size);
        TRACE("Not enough remainder, so size now %u\n", chunk->size());
      } else {
        TRACE("Inserting remainder of next chunk.\n");
        FreeChunk::insert(chunk->end(), next_size - grow);
      }
      chunk->set_free(false);
      return ptr;
    }
  }

  TRACE("Reallocating by copy.\n");
  void *new_ptr = malloc(malloc_size);
  if (!new_ptr)
    return nullptr;
  memcpy(new_ptr, ptr, old_size);
  free(ptr);
  return new_ptr;
}

} // extern "C"
