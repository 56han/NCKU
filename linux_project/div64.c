#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#define likely(x) __builtin_expect(!!(x), 1)

static inline bool is_power_of_2(unsigned long n)
{
    return (n != 0 && ((n & (n - 1)) == 0));
}

static inline int __ilog2_u32(uint32_t x)
{
    return x ? sizeof(x) * 8 - __builtin_clz(x) - 1 : 0;
}

static inline int __ilog2_u64(uint64_t x)
{
    uint32_t h = x >> 32;
    if (h)
        return sizeof(h) * 8 - __builtin_clz(h) + 31;
    return x ? sizeof(x) * 8 - __builtin_clz(x) - 1 : 0;
}

/* Calculates the logarithm base 2 of a 32-bit or 64-bit unsigned integer.
 * This function performs a constant-capable log base 2 computation, allowing
 * it to initialize global variables with constant data, which explains the
 * extensive use of ternary operators.
 * It automatically chooses the optimized version suitable for the size of 'n',
 * based on whether it is 32-bit or 64-bit.
 */
#define ilog2(n)                                                       \
    (__builtin_constant_p(n) ? ((n) < 2 ? 0 : 63 - __builtin_clzll(n)) \
     : (sizeof(n) <= 4)      ? __ilog2_u32(n)                          \
                             : __ilog2_u64(n))

/* In cases where the divisor is a constant, we compute its inverse during
 * compile time. This allows us to replace the division operation with several
 * inline multiplications, significantly improving performance.
 */
#define __div64_const32(n, ___b)                                            \
    ({                                                                      \
        /* This approach multiplies by the reciprocal of b: to divide n by  \
         * b, we multiply n by (p / b) and then divide by p. The            \
         * efficiency comes from the compiler's ability to optimize much    \
         * of this code during compile time through constant propagation,   \
         * leaving behind only a few multiplication operations. This is     \
         * why we use this large macro, as a static inline function         \
         * does not consistently achieve the same optimization.             \
         */                                                                 \
        uint64_t ___res, ___x, ___t, ___m, ___n = (n);                      \
        bool ___bias;                                                       \
                                                                            \
        /* determine MSB of b */                                            \
        uint32_t ___p = 1 << ilog2(___b);                                   \
                                                                            \
        /* compute m = ((p << 64) + b - 1) / b */                           \
        ___m = (~0ULL / ___b) * ___p;                                       \
        ___m += (((~0ULL % ___b + 1) * ___p) + ___b - 1) / ___b;            \
                                                                            \
        /* one less than the dividend with highest result */                \
        ___x = ~0ULL / ___b * ___b - 1;                                     \
                                                                            \
        /* test our ___m with res = m * x / (p << 64) */                    \
        ___res = ((___m & 0xffffffff) * (___x & 0xffffffff)) >> 32;         \
        ___t = ___res += (___m & 0xffffffff) * (___x >> 32);                \
        ___res += (___x & 0xffffffff) * (___m >> 32);                       \
        ___t = (___res < ___t) ? (1ULL << 32) : 0;                          \
        ___res = (___res >> 32) + ___t;                                     \
        ___res += (___m >> 32) * (___x >> 32);                              \
        ___res /= ___p;                                                     \
                                                                            \
        /* Next, we will clean up and enhance the efficiency of our current \
         * implementation.                                                  \
         */                                                                 \
        if (~0ULL % (___b / (___b & -___b)) == 0) {                         \
            /* special case, can be simplified to ... */                    \
            ___n /= (___b & -___b);                                         \
            ___m = ~0ULL / (___b / (___b & -___b));                         \
            ___p = 1;                                                       \
            ___bias = true;                                                 \
        } else if (___res != ___x / ___b) {                                 \
            /* It is necessary to introduce a bias to counteract errors     \
             * caused by bit truncation. Without this bias, an extra bit    \
             * would be required to accurately represent the variable 'm',  \
             * which would exceed the capacity of a 64-bit variable. To     \
             * manage this, the solution involves setting 'm' equal to 'p'  \
             * divided by 'b', and 'n' divided by 'b' as equal to           \
             * (n * m + m) / p. This approach is taken because avoiding the \
             * bias is not feasible due to the limitations imposed by bit   \
             * truncation errors and the 64-bit variable size constraint.   \
             */                                                             \
            ___bias = true;                                                 \
            /* Compute m = (p << 64) / b */                                 \
            ___m = (~0ULL / ___b) * ___p;                                   \
            ___m += ((~0ULL % ___b + 1) * ___p) / ___b;                  \
        } else {                                                            \
            /* Reduce m / p, and try to clear bit 31 of m when possible,    \
             * otherwise that will need extra overflow handling later.      \
             */                                                             \
            uint32_t ___bits = -(___m & -___m);                             \
            ___bits |= ___m >> 32;                                          \
            ___bits = (~___bits) << 1;                                      \
            /* If ___bits == 0 then setting bit 31 is  unavoidable.         \
             * Simply apply the maximum possible reduction in that          \
             * case. Otherwise the MSB of ___bits indicates the             \
             * best reduction we should apply.                              \
             */                                                             \
            if (!___bits) {                                                 \
                ___p /= (___m & -___m);                                     \
                ___m /= (___m & -___m);                                     \
            } else {                                                        \
                ___p >>= ilog2(___bits);                                    \
                ___m >>= ilog2(___bits);                                    \
            }                                                               \
            /* No bias needed. */                                           \
            ___bias = false;                                                \
        }                                                                   \
                                                                            \
        /* Now we have a combination of 2 conditions:                       \
         * 1. whether or not we need to apply a bias, and                   \
         * 2. whether or not there might be an overflow in the cross        \
         *    product determined by (___m & ((1 << 63) | (1 << 31))).       \
         *                                                                  \
         * Select the best way to do (m_bias + m * n) / (1 << 64).          \
         * From now on there will be actual runtime code generated.         \
         */                                                                 \
        ___res = __xprod_64(___m, ___n, ___bias);                           \
                                                                            \
        ___res /= ___p;                                                     \
    })

/*
 * Semantic:  retval = ((bias ? m : 0) + m * n) >> 64
 *
 * The product is a 128-bit value, scaled down to 64 bits.
 * Assuming constant propagation to optimize away unused conditional code.
 */
static inline uint64_t __xprod_64(const uint64_t m, uint64_t n, bool bias)
{
    uint32_t m_lo = m;
    uint32_t m_hi = m >> 32;
    uint32_t n_lo = n;
    uint32_t n_hi = n >> 32;
    uint64_t res;
    uint32_t res_lo, res_hi;

    if (!bias) {
        res = ((uint64_t) m_lo * n_lo) >> 32;
    } else if (!(m & ((1ULL << 63) | (1ULL << 31)))) {
        /* it is impossible for an overflow to occur in this case */
        res = (m + (uint64_t) m_lo * n_lo) >> 32;
    } else {
        res = m + (uint64_t) m_lo * n_lo;
        res_lo = res >> 32;
        res_hi = (res_lo < m_hi);
        res = res_lo | ((uint64_t) res_hi << 32);
    }

    if (!(m & ((1ULL << 63) | (1ULL << 31)))) {
        /* it is impossible for an overflow to occur in this case */
        res += (uint64_t) m_lo * n_hi;
        res += (uint64_t) m_hi * n_lo;
        res >>= 32;
    } else {
        res += (uint64_t) m_lo * n_hi;
        uint32_t tmp = res >> 32;
        res += (uint64_t) m_hi * n_lo;
        res_lo = res >> 32;
        res_hi = (res_lo < tmp);
        res = res_lo | ((uint64_t) res_hi << 32);
    }

    res += (uint64_t) m_hi * n_hi;

    return res;
}

static inline uint32_t __div64_32(uint64_t *n, uint32_t base)
{
    uint64_t rem = *n;
    uint64_t b = base;
    uint64_t res, d = 1;
    uint32_t high = rem >> 32;

    /* Reduce the thing a bit first */
    res = 0;
    if (high >= base) {
        high /= base;
        res = (uint64_t) high << 32;
        rem -= (uint64_t) (high * base) << 32;
    }

    while ((int64_t) b > 0 && b < rem) {
        b = b + b;
        d = d + d;
    }

    do {
        if (rem >= b) {
            rem -= b;
            res += d;
        }
        b >>= 1;
        d >>= 1;
    } while (d);

    *n = res;
    return rem;
}

/* The unnecessary pointer compare is there to check for type safety.
 * n must be 64bit.
 */
#define div64(n, base)                                               \
    ({                                                               \
        uint32_t __base = (base);                                    \
        uint32_t __rem;                                              \
        if (__builtin_constant_p(__base) && is_power_of_2(__base)) { \
            __rem = (n) & (__base - 1);                              \
            (n) >>= ilog2(__base);                                   \
        } else if (__builtin_constant_p(__base) && __base != 0) {    \
            uint32_t __res_lo, __n_lo = (n);                         \
            (n) = __div64_const32(n, __base);                        \
            /* the remainder can be computed with 32-bit regs */     \
            __res_lo = (n);                                          \
            __rem = __n_lo - __res_lo * __base;                      \
        } else if (likely(((n) >> 32) == 0)) {                       \
            __rem = (uint32_t) (n) % __base;                         \
            (n) = (uint32_t) (n) / __base;                           \
        } else {                                                     \
            __rem = __div64_32(&(n), __base);                        \
        }                                                            \
        __rem;                                                       \
    })

int main(int argc, char *argv[])
{
    uint64_t res;
    res = argc + 5;
    printf("%lu\n", res);
    div64(res, 3);
    printf("%lu\n", res);
}