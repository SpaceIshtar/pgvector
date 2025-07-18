#include "ItemPointerBtree.h"

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"

static IPTNode emptyNode;
/*
 * New buffer
 */
Buffer
IPTNewBuffer(Relation index, ForkNumber forkNum)
{
    Buffer      buf;
    LockRelationForExtension(index, ExclusiveLock);
	buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);
    UnlockRelationForExtension(index, ExclusiveLock);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

void
IPTInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), 0);
    if (PageAddItem(page, (Item) &emptyNode, sizeof(IPTNode), InvalidOffsetNumber, false, false) != FirstOffsetNumber)
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("insert iptnode failed")));
    }
}

static inline IPTNode* 
PageGetIPTNode(Page page)
{
    return ((IPTNode*) PageGetItem(page, PageGetItemId(page, FirstOffsetNumber)));
}

static uint32 IPTBinarySearch(IPTNode* node, ItemPointerData key)
{
    int low = 0, high = node->num_keys - 1;
    int mid;
    int    compare;
    ItemPointerData midData;
    if (node->num_keys == 0) return 0;
    while (low <= high)
    {
        mid = low + (high - low) / 2;
        midData = node->keys[mid];
        compare = ItemPointerCompare(&midData, &key);
        if (compare == 0)
            return mid;
        
        if (compare < 0)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return low;
}

ItemPointerData IPTSearch(Relation index, BlockNumber rootPage, ItemPointerData key)
{
    Buffer  buf;
    Page    page;
    IPTNode *node;
    ItemPointerData result;
    if (rootPage >= RelationGetNumberOfBlocks(index))
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("blk is not valid")));
    }
    buf = ReadBuffer(index, rootPage);
    LockBuffer(buf, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buf);
    node = PageGetIPTNode(page);
    if (IsLeafNode(node))
    {
        uint32 pos = IPTBinarySearch(node, key);
        result = node->values[pos];
        UnlockReleaseBuffer(buf);
        return result;
    }
    else
    {
        uint32 pos = IPTBinarySearch(node, key);
        BlockNumber nextPage = ItemPointerGetBlockNumber(&node->values[pos]);
        UnlockReleaseBuffer(buf);
        return IPTSearch(index, nextPage, key);
    }
}

static void IPTInsert_internal(Relation index, BlockNumber pageBlk, ItemPointerData key, ItemPointerData value, ItemPointer popKey, ItemPointer popValue, List** occupied)
{
    ItemPointerData nextLevelPopKey, nextLevelPopValue;
    Buffer buf;
    Page   page;
    IPTNode *node;
    uint32 pos;
    BlockNumber nextPage;
    GenericXLogState* state = GenericXLogStart(index);
    ItemPointerSetInvalid(&nextLevelPopKey);
    ItemPointerSetInvalid(&nextLevelPopValue);
    if (pageBlk >= RelationGetNumberOfBlocks(index))
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("blk is not valid")));
    }
    buf = ReadBuffer(index, pageBlk);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    *occupied = lappend_int(*occupied, buf);
    page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
    // page = BufferGetPage(buf);
    node = PageGetIPTNode(page);
    pos = IPTBinarySearch(node, key);
    if (IsLeafNode(node))
    {
        
        if (node->num_keys == 0)
        {
            ItemPointerSet(&node->keys[0], ItemPointerGetBlockNumber(&key), ItemPointerGetOffsetNumber(&key));
            // node->keys[0] = key;
            ItemPointerSet(&node->values[0], ItemPointerGetBlockNumber(&value), ItemPointerGetOffsetNumber(&value));
            // node->values[0] = value;
            node->num_keys++;
            GenericXLogFinish(state);
            // MarkBufferDirty(buf);
            UnlockReleaseBuffer(buf);
            return;
        }
        memmove(&node->keys[pos+1], &node->keys[pos], sizeof(ItemPointerData)*(node->num_keys - pos));
        memmove(&node->values[pos+1], &node->values[pos], sizeof(ItemPointerData)*(node->num_keys - pos));
        node->keys[pos] = key;
        node->values[pos] = value;
        node->num_keys++;

        if (node->num_keys == MAX_KEYS)
        {
            Buffer newBuf;
            Page   newPage;
            IPTNode *newNode;
            GenericXLogState* newState = GenericXLogStart(index);
            pos = (int) MIN_KEYS;
            newBuf = IPTNewBuffer(index, MAIN_FORKNUM);
            *occupied = lappend_int(*occupied, newBuf);
            // newPage = BufferGetPage(newBuf);
            newPage = GenericXLogRegisterBuffer(newState, newBuf, GENERIC_XLOG_FULL_IMAGE);
            IPTInitPage(newBuf, newPage);
            newNode = PageGetIPTNode(newPage);
            newNode->type = node->type;
            newNode->num_keys = MAX_KEYS - pos;
            memcpy(&newNode->keys[0], &node->keys[pos], sizeof(ItemPointerData)*(newNode->num_keys));
            memcpy(&newNode->values[0], &node->values[pos], sizeof(ItemPointerData)*(newNode->num_keys));
            node->num_keys = pos;
            ItemPointerSet(popKey, ItemPointerGetBlockNumber(&newNode->keys[0]), ItemPointerGetOffsetNumber(&newNode->keys[0]));
            ItemPointerSet(popValue, BufferGetBlockNumber(newBuf), FirstOffsetNumber);
            GenericXLogFinish(newState);
            // MarkBufferDirty(newBuf);
            UnlockReleaseBuffer(newBuf);
        }
    }
    else
    {
        nextPage = ItemPointerGetBlockNumber(&node->values[pos]);
        IPTInsert_internal(index, nextPage, key, value, &nextLevelPopKey, &nextLevelPopValue, occupied);
        if (!ItemPointerIsValid(&nextLevelPopKey))
        {
            GenericXLogAbort(state);
            UnlockReleaseBuffer(buf);
            return;
        }
        /* The key poped by next level must be at the pos*/
        memmove(&node->keys[pos+1], &node->keys[pos], sizeof(ItemPointerData)*(node->num_keys - pos));
        memmove(&node->values[pos+2], &node->values[pos+1], sizeof(ItemPointerData)*(node->num_keys - pos));
        node->keys[pos] = nextLevelPopKey;
        node->values[pos+1] = nextLevelPopValue;
        node->num_keys++;
        if (node->num_keys == MAX_KEYS)
        {
            Buffer newBuf;
            Page   newPage;
            IPTNode *newNode;
            GenericXLogState* newState = GenericXLogStart(index);
            pos = (int) MIN_KEYS;
            newBuf = IPTNewBuffer(index, MAIN_FORKNUM);
            *occupied = lappend_int(*occupied, newBuf);
            // newPage = BufferGetPage(newBuf);
            newPage = GenericXLogRegisterBuffer(newState, newBuf, GENERIC_XLOG_FULL_IMAGE);
            IPTInitPage(newBuf, newPage);
            newNode = PageGetIPTNode(newPage);
            newNode->type = node->type;
            newNode->num_keys = MAX_KEYS - pos - 1;
            ItemPointerSet(popKey, ItemPointerGetBlockNumber(&node->keys[pos]), ItemPointerGetOffsetNumber(&node->keys[pos]));
            ItemPointerSet(popValue, BufferGetBlockNumber(newBuf), FirstOffsetNumber);
            memcpy(&newNode->keys[0], &node->keys[pos+1], sizeof(ItemPointerData)*(newNode->num_keys));
            memcpy(&newNode->values[0], &node->values[pos+1], sizeof(ItemPointerData)*(newNode->num_keys+1));
            node->num_keys = pos;
            GenericXLogFinish(newState);
            // MarkBufferDirty(newBuf);
            UnlockReleaseBuffer(newBuf);
        }
    }
    GenericXLogFinish(state);
    // MarkBufferDirty(buf);
    UnlockReleaseBuffer(buf);
}

void IPTInsert(Relation index, BlockNumber rootPageBlk, ItemPointerData key, ItemPointerData value, BlockNumber* updatedRootPage)
{
    ItemPointerData nextLevelPopKey, nextLevelPopValue;
    GenericXLogState* state = NULL;
    List            * occupied = NIL;
    ListCell        * lc;
    ItemPointerSetInvalid(&nextLevelPopKey);
    ItemPointerSetInvalid(&nextLevelPopValue);
    IPTInsert_internal(index, rootPageBlk, key, value, &nextLevelPopKey, &nextLevelPopValue, &occupied);
    if (ItemPointerIsValid(&nextLevelPopKey))
    {
        /* Create new root page to store the popped value*/
        Buffer buf;
        Page   page;
        IPTNode *node;
        state = GenericXLogStart(index);
        buf = IPTNewBuffer(index, MAIN_FORKNUM);
        // page = BufferGetPage(buf);
        page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
        IPTInitPage(buf, page);
        node = PageGetIPTNode(page);
        node->type = IPTNODE_INTERNAL;
        node->num_keys = 1;
        node->keys[0] = nextLevelPopKey;
        ItemPointerSet(&node->values[0], rootPageBlk, FirstOffsetNumber);
        node->values[1] = nextLevelPopValue;
        *updatedRootPage = BufferGetBlockNumber(buf);
        // MarkBufferDirty(buf);
        // UnlockReleaseBuffer(buf);
        occupied = lappend_int(occupied, buf);
        GenericXLogFinish(state);
        UnlockReleaseBuffer(buf);
    }
    // GenericXLogFinish(state);
    // foreach(lc, occupied)
    // {
    //     UnlockReleaseBuffer(lc->int_value);
    // }
}

void IPTDelete(Relation index, BlockNumber rootPage, ItemPointerData key)
{
    elog(ERROR, "Unimplemented IPTDeleted");
}