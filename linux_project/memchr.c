#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#define KB 1024
#define MB 1024 * KB

/* Nonzero if X is not aligned on a "long" boundary */
#ifdef UNALIGNED
#undef UNALIGNED
#endif

#define UNALIGNED(X) ((uintptr_t) X & (sizeof(long) - 1))

/* How many bytes are loaded each iteration of the word copy loop */
#define LBLOCKSIZE (sizeof(long))

/* Threshhold for punting to the bytewise iterator */
#define TOO_SMALL(LEN) ((LEN) < LBLOCKSIZE)

#if LONG_MAX == 2147483647L
#define DETECT_NULL(X) (((X) - (0x01010101)) & ~(X) & (0x80808080))
#else
#if LONG_MAX == 9223372036854775807L
/* Nonzero if X (a long int) contains a NULL byte. */
#define DETECT_NULL(X) \
    (((X) - (0x0101010101010101)) & ~(X) & (0x8080808080808080))
#else
#error long int is not a 32bit or 64bit type.
#endif
#endif

/* @return nonzero if (long)X contains the byte used to fill MASK. */
#define DETECT_CHAR(X, mask) DETECT_NULL((X) ^ (mask))

void printBits(size_t const size, void const * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

void *memchr_opt(const void *str, int c, size_t len)
{
    const unsigned char *src = (const unsigned char *) str;
    unsigned char d = c;

    while (UNALIGNED(src)) {
        if (!len--)
            return NULL;
        if (*src == d)
            return (void *) src;
        src++;
    }

    if (!TOO_SMALL(len)) {
        unsigned long *asrc = (unsigned long *) src;
        unsigned long mask = d << 8 | d;
        mask |= mask << 16;
        for (unsigned int i = 32; i < LBLOCKSIZE * 8; i <<= 1)
            mask |= mask << i;

        while (len >= LBLOCKSIZE) {
            if (DETECT_CHAR(*asrc, mask))
                break;
            asrc++;
            len -= LBLOCKSIZE;
        }

        src = (unsigned char *) asrc;
    }

    while (len--) {
        if (*src == d)
            return (void *) src;
        src++;
    }

    return NULL;
}

void *memchr(const void *s, int c, size_t n)
{
    const unsigned char *p = s;
    while (n-- != 0) {
        if ((unsigned char)c == *p++) {
            return (void *)(p - 1);
        }
    }
    return NULL;
}

void get_time(struct timespec *ts) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);

    ts->tv_sec = counter.QuadPart / frequency.QuadPart;
    ts->tv_nsec = (counter.QuadPart % frequency.QuadPart) * 1000000000 / frequency.QuadPart;
}

double measure_memchr(const char *str, int c, size_t len) {
    struct timespec start, end;
    get_time(&start);
    memchr(str, c, len);
    get_time( &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

double measure_memchr_opt(const char *str, int c, size_t len) {
    struct timespec start, end;
    get_time(&start);
    memchr_opt(str, c, len);
    get_time(&end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void run_experiment(size_t length) {
    char *str = malloc(length);
    if (!str) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // Initialize the string with some pattern, e.g., 'a'
    memset(str, 'a', length);

    // Positions to test: start, middle, end, not present
    size_t positions[] = {0, length / 2, length - 1, length + 1};

    for (int i = 0; i < 4; i++) {
        size_t pos = positions[i];
        char c = (pos < length) ? 'b' : 'z';
        if (pos < length) str[pos] = c;

        double memchr_time = 0, memchr_opt_time = 0;
        int iterations = 10;

        for (int j = 0; j < iterations; j++) {
            memchr_time += measure_memchr(str, c, length);
            memchr_opt_time += measure_memchr_opt(str, c, length);
        }

        memchr_time /= iterations;
        memchr_opt_time /= iterations;

        printf("Length: %zu, Position: %zu, memchr: %f, memchr_opt: %f\n",
               length, pos, memchr_time, memchr_opt_time);

        if (pos < length) str[pos] = 'a'; // Reset character
    }

    free(str);
}

int main() {
    size_t lengths[] = {1 * KB, 10 * KB, 100 * KB, 1 * MB, 10 * MB};

    for (int i = 0; i < 5; i++) {
        run_experiment(lengths[i]);
    }

    return 0;
}
