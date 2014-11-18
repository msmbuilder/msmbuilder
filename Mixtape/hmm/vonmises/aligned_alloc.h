#ifndef ALIGNED_ALLOC
#define ALIGNED_ALLOC
#include <stdlib.h>
#include <malloc.h>

void * malloc_simd(const size_t size) 
{
	const size_t alignment = 16;
#ifdef __linux__
	return memalign(alignment, size);
#endif
#ifdef WIN32
	return _aligned_malloc(size, alignment);
#endif
}

void free_simd(void * ptr) 
{
#ifdef __linux__
    free(ptr);
#endif
#ifdef WIN32
	_aligned_free(ptr);
#endif
}

#endif
