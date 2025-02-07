#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "quick.h"
// #include "list.h"

typedef struct element{
    struct list_head list;
    int val;
    int seq;
} element_t;

void print_list(struct list_head *head) { // print the list to check

    element_t *entry, *safe;
    list_for_each_entry_safe (entry, safe, head, list) {
        printf("%d ", entry->val);
    }
    printf("\n");
}

int calculate_digits(int x) {
    int digits = 0;
    if (x == 0) {
        digits = 1;
    } else {
        while (x != 0) {
            x /= 10;
            digits++;
        }
    }
    return digits;
}

void copy_list(struct list_head *src, struct list_head *dest) {
    element_t *entry, *new_entry;
    list_for_each_entry(entry, src, list) {
        new_entry = malloc(sizeof(element_t));
        new_entry->val = entry->val;
        new_entry->seq = entry->seq;
        list_add_tail(&new_entry->list, dest);
    }
}

void quick_sort(void *priv, struct list_head *head, list_cmp_func_t cmp){
    if (list_empty(head) || list_is_singular(head))
        return;
    
    struct list_head *result = malloc(sizeof(struct list_head));
    struct list_head *left = malloc(sizeof(struct list_head));
    struct list_head *right = malloc(sizeof(struct list_head));
    struct list_head *pivot = malloc(sizeof(struct list_head));
    INIT_LIST_HEAD(result);
    INIT_LIST_HEAD(left);
    INIT_LIST_HEAD(right);
    INIT_LIST_HEAD(pivot);

    int size = q_size(head);
    int max_level = 10 * (calculate_digits(size)-1);
    struct list_head *begin[max_level];

    int i = 0;
    begin[0] = head;

    while(i >= 0){
        struct list_head *L = begin[i];
        if(list_is_singular(L) == 0 && list_empty(L) == 0){
            list_move(L->next, pivot);
            struct list_head *p = L->next;

            while(p){
                struct list_head *n = p;
                p = p->next;
                if(cmp(priv, pivot->next, n) > 0) // pivot >= n
                    list_move_tail(n, left);   
                else
                    list_move_tail(n, right); // pivot < n
                if(p == L)
                    break;
            }

            struct list_head *left_tmp = malloc(sizeof(struct list_head));
            struct list_head *right_tmp = malloc(sizeof(struct list_head));
            struct list_head *pivot_tmp = malloc(sizeof(struct list_head));
            INIT_LIST_HEAD(left_tmp);
            INIT_LIST_HEAD(right_tmp);
            INIT_LIST_HEAD(pivot_tmp);
            copy_list(left, left_tmp);
            copy_list(right, right_tmp);
            copy_list(pivot, pivot_tmp);

            begin[i] = left_tmp;
            begin[i + 1] = pivot_tmp;
            begin[i + 2] = right_tmp;

            INIT_LIST_HEAD(left);
            INIT_LIST_HEAD(right);
            INIT_LIST_HEAD(pivot);
            
            i += 2;
        }
        else{
            if(list_is_singular(L)){
                list_splice(L, result);
            }
            i--;
        }
    }
    INIT_LIST_HEAD(head);
    list_splice_tail_init(result, head);
}
