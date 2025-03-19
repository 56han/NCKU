#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/timekeeping.h>
#include <linux/types.h>
#include <linux/delay.h>
#include <asm-generic/div64.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Yi-Han Wan");
MODULE_DESCRIPTION("A test module for do_div macro.");
MODULE_VERSION("0.1");

static uint64_t test_division(uint64_t n, uint32_t base) {
    uint64_t start, end;
    uint32_t remainder;

    start = ktime_get_ns();
    remainder = do_div(n, base);
    end = ktime_get_ns();

    printk(KERN_INFO "n: %llu, base: %u, remainder: %u, quotient: %llu, time: %llu ns\n", n, base, remainder, n, end - start);

    return end - start;
}

static int __init div_test_init(void) {
    printk(KERN_INFO "Loading do_div test module...\n");

    // Test case 1: __base is a constant and power of 2
    test_division(123456789012345ULL, 8); // 8 is a power of 2

    // Test case 2: __base is a constant and not power of 2
    test_division(123456789012345ULL, 10); // 10 is not a power of 2

    // Test case 3: n's high 32 bits are 0 and __base is not a constant
    uint32_t dynamic_base = jiffies % 100 + 1; // dynamic_base is not a compile-time constant
    test_division(12345ULL, dynamic_base); // n fits in 32 bits

    // Test case 4: General case where n's high 32 bits are non-zero and __base is not a constant
    dynamic_base = jiffies % 10000 + 1; // dynamic_base is not a compile-time constant
    test_division(98765432109876ULL, dynamic_base); // General case with high bits non-zero

    return 0;
}

static void __exit div_test_exit(void) {
    printk(KERN_INFO "Unloading do_div test module...\n");
}

module_init(div_test_init);
module_exit(div_test_exit);

