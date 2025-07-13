#ifndef HOOKS_H
#define HOOKS_H

#include "postgres.h"

#include "access/genam.h"
#include "access/parallel.h"
#include "common/hashfn.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "nodes/extensible.h"
#include "port.h"				/* for random() */
#include "utils/relptr.h"
#include "utils/sampling.h"


#include "vector.h"
#include <optimizer/paths.h>

/* Create a hashmap that stores the bitmap result*/
typedef struct ItemPointerHashEntry
{
	ItemPointerData tid;
	char		status;
}			ItemPointerHashEntry;

#define SH_PREFIX itempointer
#define SH_ELEMENT_TYPE ItemPointerHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef bool (*ambitmapsearch)(itempointer_hash* bitmap, IndexScanDesc scan, ScanDirection direction);
typedef void (*hook_evaluateTID)(ItemPointer* tids, bool* results, int length, IndexScanDesc scan, ExprState  *qual, ExprContext  *econtext);
typedef bool (*ampushdownsearch)(IndexScanDesc scan, ScanDirection direction, hook_evaluateTID evaluate, ExprState *qual, ExprContext  *econtext);

typedef struct IndexHookInfo{
    Oid index_oid;
    ambitmapsearch bitmapsearch_func;
    ampushdownsearch pushdownsearch_func;
} IndexHookInfo;

typedef enum SelfDefinedNodeTag
{
    T_IndexWithBitmapPath = 1,
    T_IndexWithBitmapScan = 2,
} SelfDefinedNodeTag;

typedef struct IndexWithBitmapPath{
    SelfDefinedNodeTag type;
    SelfDefinedNodeTag pathtype;
    RelOptInfo *parent pg_node_attr(write_only_relids);
    PathTarget *pathtarget pg_node_attr(write_only_nondefault_pathtarget);
    ParamPathInfo *param_info pg_node_attr(write_only_req_outer);
    bool		parallel_aware;
	/* OK to use as part of parallel plan? */
	bool		parallel_safe;
	/* desired # of workers; 0 = not parallel */
	int			parallel_workers;

	/* estimated size/costs for path (see costsize.c for more info) */
	Cardinality rows;			/* estimated number of result tuples */
	Cost		startup_cost;	/* cost expended before fetching any tuples */
	Cost		total_cost;		/* total cost (assuming all tuples fetched) */


    IndexOptInfo *indexinfo;
    List    *orderByClauses; /* Must be vector clause and length = 1*/
    Cardinality  bitmap_rows;
} IndexWithBitmapPath;

typedef struct IndexWithBitmapScan{
    SelfDefinedNodeTag scantype;
    IndexHookInfo       *indexhookinfo;
    IndexOptInfo        *indexinfo;
    List                *orderByClauses;
    Cardinality        bitmap_rows;
} IndexWithBitmapScan;

typedef struct IndexWithBitmapScanState{
    CustomScanState customScanState;
    // BitmapIndexScanState *bitmapScanState; /* set when ExecInitScan*/
    ScanState     *bitmapScanState;
    struct itempointer_hash           *bitmapResult;
    IndexWithBitmapScan               *scan;
    bool                  first;
    IndexScanDesc         vectorScanDesc;
    Relation              vectorIndex;
    TupleTableSlot        *slot;
} IndexWithBitmapScanState;


typedef struct PushDownScanState
{
    CustomScanState customScanState;
    IndexScanState  *indexScanState;
    IndexHookInfo   *indexhookinfo;
} PushDownScanState;

void set_custom_rel_pathlist(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte);

Plan*   generateBitmapIndexScan(PlannerInfo *root, RelOptInfo *rel, CustomPath *best_path, List *tlist, List *clauses, List *custom_plans);
Node*   generateBitmapIndexScanState(CustomScan *cscan);

void    BeginIndexWithBitmapScan(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot* ExecIndexWithBitmapScan(CustomScanState *node);
void EndIndexWithBitmapScan (CustomScanState *node);
void ReScanIndexWithBitmapScan (CustomScanState *node);
void MarkPosIndexWithBitmapScan (CustomScanState *node);
void RestrPosIndexWithBitmapScan (CustomScanState *node);
Size EstimateDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt);
void InitializeDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate);
void ReInitializeDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate);
void InitializeWorkerIndexWithBitmapScan (CustomScanState *node, shm_toc *toc, void *coordinate);
void ShutdownIndexWithBitmapScan (CustomScanState *node);
void ExplainIndexWithBitmapScan (CustomScanState *node, List *ancestors, ExplainState *es);

void Evaluate_TID(ItemPointer* tids, bool* results, int length, IndexScanDesc scan, ExprState  *qual, ExprContext  *econtext);

Plan*   generatePushDownScan(PlannerInfo *root, RelOptInfo *rel, CustomPath *best_path, List *tlist, List *clauses, List *custom_plans);
Node*   generatePushDownScanState(CustomScan *cscan);

void    BeginPushDownScan(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot* ExecPushDownScan(CustomScanState *node);
void EndPushDownScan (CustomScanState *node);
void ReScanPushDownScan (CustomScanState *node);
void MarkPosPushDownScan (CustomScanState *node);
void RestrPosPushDownScan (CustomScanState *node);
Size EstimateDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt);
void InitializeDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate);
void ReInitializeDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate);
void InitializeWorkerPushDownScan (CustomScanState *node, shm_toc *toc, void *coordinate);
void ShutdownPushDownScan (CustomScanState *node);
void ExplainPushDownScan (CustomScanState *node, List *ancestors, ExplainState *es);

void register_hook(void);
void unregister_hook(void);

#endif