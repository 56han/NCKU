#include "list.h"
// struct list_head;

typedef int (*list_cmp_func_t)(void *,
                               const struct list_head *,
                               const struct list_head *);

static inline int q_size(struct list_head *head){
    if (!head)
        return 0;

    int len = 0;
    struct list_head *pos;

    list_for_each (pos, head)
        len++;
    return len;
    
}


void quick_sort(void *priv, struct list_head *head, list_cmp_func_t cmp);