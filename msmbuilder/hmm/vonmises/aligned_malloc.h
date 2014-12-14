#ifndef ALIGNED_MALLOC_H
#define ALIGNED_MALLOC_H

#ifdef _MSC_VER
    #include <malloc.h>
#else
    #include <stdlib.h>
    static inline void *_aligned_malloc(size_t size, size_t alignment)
    {
        void *p;
        int ret = posix_memalign(&p, alignment, size);
        return (ret == 0) ? p : 0;
    }
    static inline void _aligned_free(void *memblock) {
        free(memblock);
    }
#endif

#endif