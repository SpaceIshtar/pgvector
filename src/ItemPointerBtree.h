#ifndef ItemPointerBtree_H
#define ItemPointerBtree_H

#include "postgres.h"
#include "math.h"

#include "access/genam.h"
#include "storage/itemptr.h"
#include "storage/bufpage.h"

typedef enum {
    IPTNODE_INTERNAL,
    IPTNODE_LEAF
} IPTNodeType;

#define TREE_ORDER (BLCKSZ-sizeof(IPTNodeType)-sizeof(int)-MAXALIGN(SizeOfPageHeaderData)-sizeof(ItemIdData)-sizeof(ItemPointerData))/(2*sizeof(ItemPointerData))-1 // TODO: Calculate the TREE_ORDER
#define MAX_KEYS (TREE_ORDER - 1)
#define MIN_KEYS (ceil((float)(TREE_ORDER+1) / 2) - 1)
#define MAX_CHILDREN TREE_ORDER

#define IsLeafNode(node) (node->type==IPTNODE_LEAF)

typedef struct IPTNode
{
    IPTNodeType type;
    int num_keys;
    ItemPointerData keys[TREE_ORDER]; /* Allocate TREE_ORDER fan-outs, but when the keys are full, need to split node, so the MAX_KEYS is TREE_ORDER - 1*/

    /* For internal node, values represents the position of child IPTNode
     * For leaf node, values represent the heaptid; However, leaf node should have at most (MAX_CHILDREN - 1) key-value pair
    */
    ItemPointerData values[TREE_ORDER + 1]; 
} IPTNode;

Buffer IPTNewBuffer(Relation index, ForkNumber fork_num);
void   IPTInitPage(Buffer buf, Page page);


ItemPointerData IPTSearch(Relation index, BlockNumber rootPage, ItemPointerData key);
void IPTInsert(Relation index, BlockNumber rootPage, ItemPointerData key, ItemPointerData value, BlockNumber* updatedRootPage);
void IPTDelete(Relation index, BlockNumber rootPage, ItemPointerData key);


#endif