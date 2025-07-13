#include <time.h>
#include "postgres.h"

#include "hooks.h"
#include "hnsw.h"
#include "ivfflat.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/pg_am.h"
#include "catalog/namespace.h"
#include "executor/nodeBitmapIndexscan.h"
#include "executor/nodeBitmapOr.h"
#include "executor/nodeBitmapAnd.h"
#include "executor/nodeIndexscan.h"
#include "parser/parse_func.h"
#include "parser/parse_type.h"
#include "nodes/supportnodes.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pathnodes.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/restrictinfo.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM < 170000
static inline uint64
hook_murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

static inline uint32
hash_itempointer(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return hook_murmurhash64(x.i);
}

#define SH_PREFIX		itempointer
#define SH_ELEMENT_TYPE	ItemPointerHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_itempointer(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

static IndexHookInfo hnsw_hook_info = {InvalidOid, NULL, NULL};
static IndexHookInfo ivf_hook_info = {InvalidOid, NULL, NULL};
static Oid  vector_oid = InvalidOid;
static Oid  range_query_params_oid = InvalidOid;
static Oid  l2_distance_oid = InvalidOid;
static Oid  range_query_funcid = InvalidOid;

static set_rel_pathlist_hook_type next_set_pathlist_hook = NULL;

static CustomPathMethods bitmapIndexPathMethods = {"BitmapIndexPath", generateBitmapIndexScan, NULL};
static CustomScanMethods bitmapIndexScanMethods = {"BitmapIndexScan", generateBitmapIndexScanState};
static CustomExecMethods bitmapIndexExecMethods = {"BitmapIndexScanState", BeginIndexWithBitmapScan, ExecIndexWithBitmapScan, EndIndexWithBitmapScan, ReScanIndexWithBitmapScan, MarkPosIndexWithBitmapScan,
                                                    RestrPosIndexWithBitmapScan, EstimateDSMIndexWithBitmapScan, InitializeDSMIndexWithBitmapScan, ReInitializeDSMIndexWithBitmapScan, InitializeWorkerIndexWithBitmapScan,
                                                    ShutdownIndexWithBitmapScan, ExplainIndexWithBitmapScan};

static CustomPathMethods pushdownPathMethods = {"PushDownPath", generatePushDownScan, NULL};
static CustomScanMethods pushdownScanMethods = {"PushDownScan", generatePushDownScanState};
static CustomExecMethods pushdownExecMethods = {"PushDownScanState", BeginPushDownScan, ExecPushDownScan, EndPushDownScan, ReScanPushDownScan, MarkPosPushDownScan, 
                                                RestrPosPushDownScan, EstimateDSMPushDownScan, InitializeDSMPushDownScan, ReInitializeDSMPushDownScan, InitializeWorkerPushDownScan, ShutdownPushDownScan, ExplainPushDownScan};

static inline Oid
get_access_method_oid_by_name(const char *amname)
{
    return GetSysCacheOid1(AMNAME, Anum_pg_am_oid, CStringGetDatum(amname));
}

static void set_hook_info()
{
    if (!OidIsValid(hnsw_hook_info.index_oid))
    {
        hnsw_hook_info.index_oid = get_access_method_oid_by_name("hnsw");
        if (!OidIsValid(hnsw_hook_info.index_oid))
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Cannot find the oid of hnsw")));
        }
        hnsw_hook_info.bitmapsearch_func = hnswbitmapsearch;
        hnsw_hook_info.pushdownsearch_func = hnswpushdownsearch;
    }
    if (!OidIsValid(ivf_hook_info.index_oid))
    {
        ivf_hook_info.index_oid = get_access_method_oid_by_name("ivfflat");
        if (!OidIsValid(ivf_hook_info.index_oid))
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Cannot find the oid of ivf")));
        }
        ivf_hook_info.bitmapsearch_func = ivfflatbitmapsearch;
        ivf_hook_info.pushdownsearch_func = ivfflatpushdownsearch;
    }
}

static IndexHookInfo* getIndexHookInfo(Oid relam)
{
    if (hnsw_hook_info.index_oid == relam)
    {
        return &hnsw_hook_info;
    }
    else if (ivf_hook_info.index_oid == relam)
    {
        return &ivf_hook_info;
    }
    else{
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("The IndexOptInfo is not our index")));
    }
    return NULL;
}

/* find_orderby_vector_search
 * Two case is supported:
 * 1. ORDER BY embedding <-> "[1,2,3]"
 * 2. ORDER BY otherthings
 * (either order by vector distance, or order by other attributes)
 * DO NOT SUPPORT:
 * 1. ORDER BY embedding <-> "[1,2,3]" ASC, otherthings DESC
 * 2. ORDER BY multiple vector distance
*/
static bool find_orderby_vector_search(PlannerInfo *root, RelOptInfo *rel,  List **orderByVectorClauses, List **orderByOthers, List **vectorPathKeys)
{
    ListCell    *lc, *lc1;
    List        *orderby_clauses = NIL;
    List        *orderby_others = NIL;
    List        *pathkey_vector = NIL;
    foreach(lc, root->query_pathkeys)
    {
        PathKey *pathkey = (PathKey *)  lfirst(lc);
        int     em_member_length = list_length(pathkey->pk_eclass->ec_members);
        bool    found_vector_pathkey = false;

        if (pathkey->pk_strategy != BTLessStrategyNumber || pathkey->pk_nulls_first){
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 1")));
        }

        if (pathkey->pk_eclass->ec_has_volatile){
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 2")));
        }

        if (em_member_length != 1){
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 3")));
        }

        foreach(lc1, pathkey->pk_eclass->ec_members)
        {
            EquivalenceMember   *member = (EquivalenceMember *)  lfirst(lc1);
            Expr                *expr = member->em_expr;
            Oid                 op_oid = ((OpExpr*)expr)->opno;
            if (!IsA(expr, OpExpr))
            {
                orderby_others = lappend(orderby_others, expr);
                continue;
            }
            if (!bms_equal(member->em_relids, rel->relids)){
                continue;
            }

            if (op_oid == l2_distance_oid){
                orderby_clauses = lappend(orderby_clauses, expr);
                found_vector_pathkey = true;
            }
            else{
                orderby_others = lappend(orderby_others, expr);
            }
        }
        if (found_vector_pathkey)
        {
            pathkey_vector = lappend(pathkey_vector, pathkey);
        }
    }

    if (orderby_clauses != NIL && orderby_others != NIL){
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 4")));
    }

    *orderByVectorClauses = orderby_clauses;
    *orderByOthers = orderby_others;
    *vectorPathKeys = pathkey_vector;
    if (orderby_clauses != NIL){
        return true;
    }
    
    return false;
}

/* find_where_vector_search
 * 1. Check whether there exists Vector Range Query
 * 2. Check whether there exists other WHERE condition
 * return true if Vector Range Query exists
*/
static bool find_vector_clause(Expr* clause)
{
    if (IsA(clause, FuncExpr))
    {
        FuncExpr    *funcClause = (FuncExpr*) clause;
        Oid         funcid = funcClause->funcid;
        if (funcid == range_query_funcid) return true;
    }
    else if (IsA(clause, BoolExpr))
    {
        ListCell    *lc;
        BoolExpr    *boolExpr = (BoolExpr*) clause;
        bool        found = false;
        foreach(lc, boolExpr->args)
        {
            found = found | find_vector_clause((Expr*)lfirst(lc));
        }
        return found;
    }
    return false;
}
static bool find_where_vector_search(PlannerInfo *root, RelOptInfo *rel)
{
    ListCell    *lc;
    bool        found = false;
    foreach(lc, rel->baserestrictinfo)
    {
        RestrictInfo    *rinfo = (RestrictInfo*) lfirst(lc);
        Expr     *clause = (Expr*) rinfo->clause;
        found = found | find_vector_clause(clause);
        if (found) break;
    }
    return found;
}

/* bitmap + indexscan is considered if one of the following cases exists:
 * 1. WHERE conditition bitmap scan + ORDER BY vector search
 * 2. WHERE condition vector search + ORDER BY vector search (two vector search are on different columns)
 * We do not consider the following case because directly indexscan instead of generating bitmap would be better:
 * 1. vector search in WHERE and ORDER BY are on the same column
 * 2. WHERE condition vector search + ORDER BY otherthings 
 * 
 * 
*/
static Path* create_bitmappath_recursive(PlannerInfo *root, RelOptInfo *rel, Expr *clause, RestrictInfo* rinfo)
{
    ListCell    *lc;
    /* If the clause is bool, create a bitmapAND or bitmapOR path, and corresponding subpaths*/
    if (IsA(clause, BoolExpr))
    {
        BoolExpr    *boolClause = (BoolExpr*)clause;
        if (is_orclause(clause)){
            List    *subpaths = NIL;
            foreach(lc, boolClause->args)
            {
                Expr    *arg = (Expr*) lfirst(lc);
                Path    *subpath = create_bitmappath_recursive(root, rel, arg, rinfo);
                if (subpath == NULL){
                    return NULL;
                }
                subpaths = lappend(subpaths, ((BitmapHeapPath*)subpath)->bitmapqual);
            }
            return (Path*) create_bitmap_or_path(root, rel, subpaths);
        }
        else if (is_andclause(clause)){
            List    *subpaths = NIL;
            foreach(lc, boolClause->args)
            {
                Expr    *arg = (Expr*) lfirst(lc);
                Path    *subpath = create_bitmappath_recursive(root, rel, arg, rinfo);
                if (subpath == NULL){
                    return NULL;
                }
                subpaths = lappend(subpaths, ((BitmapHeapPath*)subpath)->bitmapqual);
            }
            return (Path*) create_bitmap_and_path(root, rel, subpaths);
        }
        else{
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 5")));
        }
    }
    /* If clause is operator, find corresponding index that deal with the clause, and then create indexscan to */
    else if (IsA(clause, OpExpr)){
        int             indexcol;
        OpExpr          *operatorExpr = (OpExpr*) clause;
        IndexOptInfo    *final_index = NULL;
        IndexPath       *pathnode = NULL;
        IndexClause     *iclause = NULL;
        Node            *leftop = (Node*) linitial(operatorExpr->args), *rightop = (Node*) lsecond(operatorExpr->args);
        // Oid             expr_op = operatorExpr->opno;
        // Oid             expr_coll = operatorExpr->inputcollid;
        foreach(lc, rel->indexlist)
        {
            IndexOptInfo    *index = (IndexOptInfo*) lfirst(lc);
            // Index           index_relid = index->rel->relid;
            if (index->indpred != NIL && !index->predOK) continue;
            for (indexcol = 0; indexcol < index->nkeycolumns; indexcol++)
            {
                // Oid     opfamily = index->opfamily[indexcol];
                int     indkey = index->indexkeys[indexcol];
                if (IsA(leftop, Var) && index->rel->relid == ((Var*)leftop)->varno && indkey == ((Var*)leftop)->varattno && ((Var *) leftop)->varnullingrels == NULL)
                {
                    if (!IsA(rightop, Const)){
                        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 7")));
                    }
                    if (index->amhasgetbitmap)
                    {
                        final_index = index;
                        break;
                    }
                    else
                    {
                        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 8: bitmap found corresonding index but index does not have amhasgetbitmap")));
                    }
                }
                else if (IsA(rightop, Var) && index->rel->relid == ((Var*)rightop)->varno && indkey == ((Var*)rightop)->varattno && ((Var *) rightop)->varnullingrels == NULL)
                {
                    if (!IsA(leftop, Const)){
                        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 7")));
                    }
                    if (index->amhasgetbitmap)
                    {
                        final_index = index;
                        operatorExpr->args = list_make2(rightop, leftop);
                        break;
                    }
                    else
                    {
                        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 8: bitmap found corresonding index but index does not have amhasgetbitmap")));
                    }
                }
            }
            if (final_index != NULL){
                break;
            }
        }
        
        /* Build IndexPath*/
        if (final_index == NULL){
            return NULL;
        }
        
        iclause = makeNode(IndexClause);
        iclause->rinfo = rinfo;
        iclause->indexquals = list_make1(make_simple_restrictinfo(root, clause)); //TODO: not sure whether correct
        iclause->lossy = false;
        iclause->indexcol = indexcol;
        iclause->indexcols = NIL;
        pathnode = makeNode(IndexPath);
        pathnode->path.pathtype = T_IndexScan;
        pathnode->path.parent = rel;
        pathnode->path.pathtarget = rel->reltarget;
        pathnode->path.param_info = get_baserel_parampathinfo(root, rel, rel->lateral_relids); // TODO
        pathnode->path.parallel_aware = false;
        pathnode->path.parallel_safe = rel->consider_parallel;
        pathnode->path.parallel_workers = 0;
        pathnode->path.pathkeys = NIL;

        pathnode->indexinfo = final_index;
        pathnode->indexclauses = list_make1(iclause);
        pathnode->indexorderbys = NIL;
        pathnode->indexorderbycols = NIL;
        pathnode->indexscandir = ForwardScanDirection;
        
        return (Path*) create_bitmap_heap_path(root, rel, (Path*) pathnode, rel->lateral_relids, 1, 0);
    }
    else if (IsA(clause, FuncExpr))
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 9: TODO NEXT")));
        return NULL;
    }
    else
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 10")));
        return NULL;
    }
}

static CustomPath* create_bitmapIndexPath(PlannerInfo *root, RelOptInfo *rel, List *vectorOrderByClauses, List  *vectorOrderByPathKeys)
{
    IndexOptInfo    *index;
    ListCell    *lc;
    List        *and_list = NIL;
    bool        found_orderby_index = false;
    Path  *bitmappath = NULL;
    IndexWithBitmapPath       *orderByPath = NULL;
    CustomPath  *pathnode = makeNode(CustomPath);
    /* Set Basic Info for pathnode->path*/
    pathnode->path.pathtype = T_CustomScan;
    pathnode->path.parent = rel;
    pathnode->path.pathtarget = rel->reltarget;
    pathnode->path.param_info = get_baserel_parampathinfo(root, rel, rel->lateral_relids); // TODO
    pathnode->path.parallel_aware = false;
    pathnode->path.parallel_safe = rel->consider_parallel;
    pathnode->path.parallel_workers = 0;
    pathnode->path.pathkeys = vectorOrderByPathKeys;
    pathnode->flags = CUSTOMPATH_SUPPORT_PROJECTION;


    /* Deal with WHERE condition to generate subplans, and stored in custom_plans*/
    foreach(lc, rel->baserestrictinfo)
    {
        RestrictInfo    *rinfo = (RestrictInfo*) lfirst(lc);
        Expr    *iclause = (Expr*) rinfo->clause;
        Path    *subpath = create_bitmappath_recursive(root, rel, iclause, rinfo);
        if (subpath == NULL){
            return NULL;
        }
        and_list = lappend(and_list, subpath);
    }
    if (list_length(and_list) > 1)
    {
        BitmapAndPath *and_path = create_bitmap_and_path(root, rel, and_list);
        bitmappath = (Path*) create_bitmap_heap_path(root, rel, (Path*) and_path, rel->lateral_relids, 1, 0);
    }
    else if (list_length(and_list) == 0)
    {
        return NULL;
    }
    else{
        Path    *subpath = linitial(and_list);
        if (!IsA(subpath, BitmapHeapPath) && !IsA(subpath, BitmapOrPath) && !IsA(subpath, BitmapAndPath)){
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("subpath here must be one of BitmapHeapPath, BitmapOrPath and BitmapAndPath")));
        }
        if (!IsA(subpath, BitmapHeapPath))
        {
            bitmappath = (Path*) create_bitmap_heap_path(root, rel, subpath, rel->lateral_relids, 1, 0);
        }
        else
        {
            bitmappath = subpath;
        }
    }

    pathnode->custom_paths = lappend(pathnode->custom_paths, bitmappath);

    /* Deal with ORDER BY clause: must be vector clause*/
    foreach(lc, rel->indexlist)
    {
        OpExpr    *expr = linitial(vectorOrderByClauses);
        Node            *leftop = (Node*) linitial(((OpExpr*)expr)->args), *rightop = (Node*) lsecond(((OpExpr*)expr)->args);
        Oid     opoid = ((OpExpr*)expr)->opno;
        int     indkey;
        index = (IndexOptInfo*) lfirst(lc);
        indkey = index->indexkeys[0];
        if (index->nkeycolumns > 1) continue; /* Vector index is built on one column*/
        if (!IsA(expr, OpExpr))
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("vector order by clause is not OpExpr")));
        }
        if (opoid != l2_distance_oid)
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Operator should be <->")));
        }
        if (IsA(leftop, Var) && index->rel->relid == ((Var*)leftop)->varno && ((Var*)leftop)->varattno == indkey && ((Var*)leftop)->varnullingrels == NULL)
        {
            if (IsA(rightop, Const))
            {
                found_orderby_index = true;
                break;
            }
        }
        if (IsA(rightop, Var) && index->rel->relid == ((Var*)rightop)->varno && ((Var*)rightop)->varattno == indkey && ((Var*)rightop)->varnullingrels == NULL)
        {
            if (IsA(leftop, Const))
            {
                found_orderby_index = true;
                break;
            }
        }
        if (found_orderby_index) break;
    }

    orderByPath = palloc0fast(sizeof(IndexWithBitmapPath));
    orderByPath->type = T_IndexWithBitmapPath;
    orderByPath->pathtype = T_IndexWithBitmapScan;
    orderByPath->parent = rel;
    orderByPath->pathtarget = rel->reltarget;
    orderByPath->param_info = get_baserel_parampathinfo(root, rel, rel->lateral_relids);
    orderByPath->parallel_aware = false;
    orderByPath->parallel_safe = rel->consider_parallel;
    orderByPath->parallel_workers = 0;
    orderByPath->bitmap_rows = bitmappath->rows;

    orderByPath->indexinfo = index;
    orderByPath->orderByClauses = vectorOrderByClauses;

    pathnode->custom_private = lappend(pathnode->custom_private, orderByPath);
    
    pathnode->methods = &bitmapIndexPathMethods;
    return pathnode;
}


static IndexPath* generate_index_path(PlannerInfo *root, RelOptInfo *rel, List  *orderByVectorClauses, List *vectorPathkeys)
{
    IndexOptInfo    *index;
    ListCell    *lc;
    IndexPath   *ipath = makeNode(IndexPath);
    int         indexcol = 0;
    bool        found_orderby_index = false;
    List        *orderByCols = NIL;

    if (list_length(orderByVectorClauses) > 1)
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("too many vector orderby clauses")));
    }

    orderByCols = lappend_int(orderByCols, indexcol);

    foreach(lc, rel->indexlist)
    {
        OpExpr    *expr = linitial(orderByVectorClauses);
        Node            *leftop = (Node*) linitial(((OpExpr*)expr)->args), *rightop = (Node*) lsecond(((OpExpr*)expr)->args);
        Oid     opoid = ((OpExpr*)expr)->opno;
        int     indkey;
        index = (IndexOptInfo*) lfirst(lc);
        indkey = index->indexkeys[0];
        if (index->nkeycolumns > 1) continue; /* Vector index is built on one column*/
        if (!IsA(expr, OpExpr))
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("vector order by clause is not OpExpr")));
        }
        if (opoid != l2_distance_oid)
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Operator should be <->")));
        }
        if (IsA(leftop, Var) && index->rel->relid == ((Var*)leftop)->varno && ((Var*)leftop)->varattno == indkey && ((Var*)leftop)->varnullingrels == NULL)
        {
            if (IsA(rightop, Const))
            {
                found_orderby_index = true;
                break;
            }
        }
        if (IsA(rightop, Var) && index->rel->relid == ((Var*)rightop)->varno && ((Var*)rightop)->varattno == indkey && ((Var*)rightop)->varnullingrels == NULL)
        {
            if (IsA(leftop, Const))
            {
                found_orderby_index = true;
                break;
            }
        }
        if (found_orderby_index) break;
    }


    ipath->path.pathtype = T_IndexScan;
    ipath->path.parent = rel;
    ipath->path.pathtarget = rel->reltarget;
    ipath->path.param_info = get_baserel_parampathinfo(root, rel, rel->lateral_relids);
    ipath->path.parallel_aware = false;
    ipath->path.parallel_safe = rel->consider_parallel;
    ipath->path.parallel_workers = 0;
    ipath->path.pathkeys = vectorPathkeys;
    
    ipath->indexinfo = index;
    ipath->indexclauses = NIL;
    ipath->indexorderbys = orderByVectorClauses;
    ipath->indexorderbycols = orderByCols;
    ipath->indexscandir = ForwardScanDirection;
    return ipath;
}

static CustomPath* generate_push_down_path(PlannerInfo  *root, RelOptInfo   *rel, IndexPath *index_path, List *vectorOrderByPathKeys)
{
    CustomPath  *pathnode;
    if (!index_path)
    {
        return NULL;
    }
    pathnode = makeNode(CustomPath);
    pathnode->path.pathtype = T_CustomScan;
    pathnode->path.parent = rel;
    pathnode->path.pathtarget = rel->reltarget;
    pathnode->path.param_info = get_baserel_parampathinfo(root, rel, rel->lateral_relids); // TODO
    pathnode->path.parallel_aware = false;
    pathnode->path.parallel_safe = rel->consider_parallel;
    pathnode->path.parallel_workers = 0;
    pathnode->path.pathkeys = vectorOrderByPathKeys;
    pathnode->custom_paths = NIL;
    pathnode->custom_private = NIL;
    pathnode->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    pathnode->custom_paths = lappend(pathnode->custom_paths, index_path);
    pathnode->methods = &pushdownPathMethods;

    return pathnode;
}

static RelOptInfo* copy_rel(RelOptInfo *origin)
{
    RelOptInfo *rel = makeNode(RelOptInfo);
    memcpy(rel, origin, sizeof(RelOptInfo));
    rel->pathlist = NIL;
    return rel;
}

static List* generate_prefilter_paths(PlannerInfo *root, RelOptInfo *rel)
{
    List* result_list = NIL, *temp_list = NIL;
    ListCell *p1;
    RelOptInfo *newrel = copy_rel(rel);
    create_index_paths(root, newrel);
    temp_list = newrel->pathlist;
    foreach(p1, temp_list)
    {
        Path *path = (Path *) lfirst(p1);
        if (!IsA(path, IndexPath))
        {
            result_list = lappend(result_list, path);
        }
        else
        {
            IndexPath* ipath = (IndexPath*)path;
            if (ipath->indexclauses && !ipath->indexorderbys){
                result_list = lappend(result_list, path);
            }
        }
    }
    return result_list;
}

/* Core Function to generate a CustomPath*/
void set_custom_rel_pathlist(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{   
    return;
    bool    hasOrderByVector, hasWhereVector;
    ListCell    *lc;
    Path    *seqPath = NULL, *indexPath = NULL, *push_down_path = NULL;
    List    *prefilterPaths = NIL;
    CustomPath  *bitmapIndexPath = NULL;
    List    *orderByVectorClauses = NIL, *orderByOtherClauses = NIL, *vectorPathkeys = NIL;
    Relids  required_outer = rel->lateral_relids;
    
    set_hook_info();

    /* In the original code, there are branches that deal with forgien table or sampled relation.
    *  For simplicity, here we assume only plain rel path list exists
    */
    if (rel->rtekind != RTE_RELATION)
        return;
    
    /* If the query does not contain vector search, quit custom scan*/
    hasOrderByVector = find_orderby_vector_search(root, rel, &orderByVectorClauses, &orderByOtherClauses, &vectorPathkeys);
    hasWhereVector = find_where_vector_search(root, rel);
    if (!hasOrderByVector && !hasWhereVector){
        return;
    }
    if (list_length(orderByVectorClauses)>1){
        return;
    }
    
    /* Case 0: seq scan */
    seqPath = create_seqscan_path(root, rel, required_outer, 0);
    
    /* Case 1: Pre-filtering*/
    prefilterPaths = generate_prefilter_paths(root, rel);
    
    /* Case 2: IndexScan + post-filtering*/
    indexPath = (Path*) generate_index_path(root, rel, orderByVectorClauses, vectorPathkeys);
    
    /* Case 3: bitmap + index scan*/
    if (hasOrderByVector)
        bitmapIndexPath = create_bitmapIndexPath(root, rel, orderByVectorClauses, vectorPathkeys);

    /* Case 4: push down filter during index scan*/
    /* 1. 在hook里定义并实现一个能根据evaluate某个（些）itempointer是否符合filter的函数
     * 2. 对于index而言，要实现一个amgettuple_push_down_filter，这个接口和amgettuple是一样的，只不过在这个函数里index可以调用刚才说的这个定义好的函数来evaluate搜索过程中的点
     * 2. 实现一个custompath，在custom_paths里创建一个indexpath
     * 3. initexec的时候，有IndexScanDesc scan，将scan->indexRelation->rd_indam->amgettuple换成amgettuple_push_down_filter，其他函数就直接调用nodeIndexscan.h里的应该就好
    */
    push_down_path = (Path*) generate_push_down_path(root, rel, (IndexPath*) indexPath, vectorPathkeys);
    
    /* Estimate cost for these plans and select the best and add it to rel->pathlist*/

    /* The following part is for TEST:
     * set the cost of the path we want to test to be small.
     */

    // if (seqPath)
    // {
    //     seqPath->rows = 10;
    //     seqPath->startup_cost = 0.1;
    //     seqPath->total_cost = 1;
    //     add_path(rel, seqPath);
    // }

    // foreach(lc, prefilterPaths)
    // {
    //     Path *path = (Path *) lfirst(lc);
    //     path->rows = 10;
    //     path->startup_cost = 0.1;
    //     path->total_cost = 1;
    //     add_path(rel, path);
    // }

    // if (indexPath)
    // {
    //     indexPath->rows = 10;
    //     indexPath->startup_cost = 0.1;
    //     indexPath->total_cost = 1;
    //     add_path(rel, indexPath);
    // }

    // if (bitmapIndexPath){
    //     bitmapIndexPath->path.rows = 10;
    //     bitmapIndexPath->path.startup_cost = 0.1;
    //     bitmapIndexPath->path.total_cost = 1;
    //     add_path(rel, (Path*) bitmapIndexPath);
    // }

    if (push_down_path)
    {
        push_down_path->rows = 10;
        push_down_path->startup_cost = 0.1;
        push_down_path->total_cost = 1;
        add_path(rel, push_down_path);
    }
}

/* Functions for Bitmap+IndexScan*/

static List* fix_orderby_clauses(PlannerInfo *root, RelOptInfo *rel, IndexWithBitmapPath *indexbitmappath, List *origin_clauses)
{
    IndexOptInfo    *index = indexbitmappath->indexinfo;
    List        *result = NIL;
    ListCell    *lc;
    foreach(lc, origin_clauses)
    {
        Expr    *expr = lfirst(lc);
        if (IsA(expr, OpExpr))
        {
            OpExpr  *opExpr = (OpExpr*) expr;
            Node    *leftop = (Node*) linitial(opExpr->args);
            if (IsA(leftop, Var))
            {
                if (((Var*)leftop)->varno == index->rel->relid)
                {
                    int indexcol = 0;
                    for (indexcol = 0; indexcol < index->nkeycolumns; indexcol++)
                    {
                        /* match leftop to the column of index*/
                        int indkey = index->indexkeys[indexcol];
                        if (indkey != 0)
                        {
                            if (indkey == ((Var*)leftop)->varattno && ((Var*)leftop)->varnullingrels == NULL)
                            {
                                /* find match */
                                Var *result = (Var*) copyObject(leftop);
                                result->varno = INDEX_VAR;
                                result->varattno =  indexcol + 1;
                                linitial(opExpr->args) = result;
                            }
                        }
                        else
                        {
                            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("indkey should be non-zero for vector index")));
                        }
                    }
                }
                else{
                    ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("leftop Var doesnot match the index")));
                }
            }
            else
            {
                ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("only support leftop should be Var now")));
            }
        }
        
        result = lappend(result, expr);
        
    }
    return result;
}

/* Callback function to create BitmapIndexScan*/
Plan*   generateBitmapIndexScan(PlannerInfo *root, RelOptInfo *rel, CustomPath *best_path, List *tlist, List *clauses, List *custom_plans)
{
    CustomScan  *result = makeNode(CustomScan);
    IndexWithBitmapScan* indexbitmapscan = palloc0(sizeof(IndexWithBitmapScan));
    Plan        *plan = &result->scan.plan;
    IndexWithBitmapPath *indexbitmappath = (IndexWithBitmapPath*) linitial(best_path->custom_private);
    if (list_length(best_path->custom_private) != 1)
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Length of best_path->custom_private should be 1")));
    }
    plan->targetlist = tlist;
    plan->qual = NIL; // 不需要post-filter处理
    plan->lefttree = NULL;
    plan->righttree = NULL;
    
    result->scan.scanrelid = best_path->path.parent->relid;
    result->flags = best_path->flags;
    result->custom_exprs = NIL;
    result->custom_plans = custom_plans;
    result->custom_private = NIL;
    result->custom_scan_tlist = tlist;
    result->custom_relids = rel->relids; // TODO: not sure
    result->methods = &bitmapIndexScanMethods;
    
    /* According to the IndexWithBitmapPath in best_path->custom_private, 
     * 1. generate IndexWithBitmapScan and stores in result->custom_private
    */
    indexbitmapscan->scantype = T_IndexWithBitmapScan;
    indexbitmapscan->indexinfo = indexbitmappath->indexinfo;
    indexbitmapscan->indexhookinfo = getIndexHookInfo(indexbitmappath->indexinfo->relam);
    /* TODO: fix orderByClauses, need to correct varno and varattno of leftop*/
    indexbitmapscan->orderByClauses = fix_orderby_clauses(root, rel, indexbitmappath, indexbitmappath->orderByClauses);
    indexbitmapscan->bitmap_rows = indexbitmappath->bitmap_rows;
    
    result->custom_private = lappend(result->custom_private, indexbitmapscan);
    
    return (Plan*) result;
}

Node* generateBitmapIndexScanState(CustomScan *scan)
{
    IndexWithBitmapScanState    *result = NULL;
    CustomScanState             *customScanState = NULL;
    if (list_length(scan->custom_private) != 1 || list_length(scan->custom_plans) != 1)
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Both scan->custom_plans and scan->custom_private should only have 1 element")));
    }
    result = palloc0(sizeof(IndexWithBitmapScanState));
    customScanState = &result->customScanState;
    customScanState->ss.ps.type = T_CustomScanState;
    customScanState->flags = scan->flags;
    customScanState->methods = &bitmapIndexExecMethods;
    result->bitmapScanState = NULL;
    result->vectorScanDesc = NULL;
    return (Node*) result;
}

void BeginIndexWithBitmapScan(CustomScanState *node, EState *estate, int eflags)
{
    CustomScan  *scan = (CustomScan*) (node->ss.ps.plan);
    Scan *heapScan = (Scan*)linitial(scan->custom_plans);
    IndexWithBitmapScan *vectorScan = (IndexWithBitmapScan*)linitial(scan->custom_private);
    Plan *bitmapScan = heapScan->plan.lefttree;
    // BitmapIndexScan *bitmapIndexScan = (BitmapIndexScan*) (heapScan->scan.plan.lefttree);
    // BitmapIndexScanState *bitmapIndexScanState = ExecInitBitmapIndexScan(bitmapIndexScan, estate, eflags);
    IndexWithBitmapScanState *myscanstate = (IndexWithBitmapScanState*) node;
    Oid             vectorIndexOid = vectorScan->indexinfo->indexoid;
    Relation        heapRelation = node->ss.ss_currentRelation;
    Relation        vectorIndexRelation = index_open(vectorIndexOid, AccessShareLock);
    // myscanstate->bitmapScanState = bitmapIndexScanState;
    if (IsA(bitmapScan, BitmapOr))
    {
        myscanstate->bitmapScanState = (ScanState*)ExecInitBitmapOr((BitmapOr*)bitmapScan, estate, eflags);
    }
    else if (IsA(bitmapScan, BitmapAnd))
    {
        myscanstate->bitmapScanState = (ScanState*)ExecInitBitmapAnd((BitmapAnd*)bitmapScan, estate, eflags);
    }
    else if (IsA(bitmapScan, BitmapIndexScan))
    {
        myscanstate->bitmapScanState = (ScanState*)ExecInitBitmapIndexScan((BitmapIndexScan*)bitmapScan, estate, eflags);
    }
    else
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not supported yet: bitmap scan should be BitmapOr, BitmapAnd or BitmapIndexScan")));
    }
    myscanstate->bitmapResult = NULL;
    myscanstate->scan = vectorScan;
    myscanstate->first = true;
    myscanstate->vectorIndex = vectorIndexRelation;
    myscanstate->vectorScanDesc = index_beginscan(heapRelation, vectorIndexRelation, estate->es_snapshot, 0, 1);
    myscanstate->slot = table_slot_create(heapRelation, NULL);
}

TupleTableSlot* ExecIndexWithBitmapScan(CustomScanState *node)
{
    IndexWithBitmapScanState    *myscanstate = (IndexWithBitmapScanState*)node;
    IndexWithBitmapScan         *myscan = (IndexWithBitmapScan*)myscanstate->customScanState.ss.ps.plan;
    ambitmapsearch              vectorSearchMethod = myscanstate->scan->indexhookinfo->bitmapsearch_func;
    TupleTableSlot              *slot = myscanstate->slot;
    IndexScanDesc               inputIndexScanDesc = myscanstate->vectorScanDesc;
    double                      hash_table_size = myscan->bitmap_rows * 1.2;
    if (myscanstate->first){
        /* Search Bitmap Index and get bitmap*/
        // struct timespec start, end;
        // double elapsed;
        TIDBitmap* bitmapResult;
        TBMIterator *iterator;
        // clock_gettime(CLOCK_MONOTONIC, &start);
        myscanstate->bitmapResult = itempointer_create(CurrentMemoryContext, hash_table_size, NULL);
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // elapsed = (end.tv_sec - start.tv_sec) + 
            //   (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // printf("create bitmapResult time: %.5f seconds\n", elapsed);
        // clock_gettime(CLOCK_MONOTONIC, &start);
        if (IsA(myscanstate->bitmapScanState, BitmapOrState))
        {
            bitmapResult = (TIDBitmap*) MultiExecBitmapOr((BitmapOrState*)myscanstate->bitmapScanState);
        }
        else if (IsA(myscanstate->bitmapScanState, BitmapAndState))
        {
            bitmapResult = (TIDBitmap*) MultiExecBitmapAnd((BitmapAndState*)myscanstate->bitmapScanState);
        }
        else if (IsA(myscanstate->bitmapScanState, BitmapIndexScanState))
        {
            bitmapResult = (TIDBitmap*) MultiExecBitmapIndexScan((BitmapIndexScanState*)myscanstate->bitmapScanState);
        }
        else
        {
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not supported yet: bitmap scan state should be BitmapOrState, BitmapAndState or BitmapIndexScanState")));
        }

        // clock_gettime(CLOCK_MONOTONIC, &end);
        // elapsed = (end.tv_sec - start.tv_sec) + 
        //       (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // printf("BitmapIndexScan time: %.5f seconds\n", elapsed);
        // clock_gettime(CLOCK_MONOTONIC, &start);
        iterator = tbm_begin_iterate(bitmapResult);
        
        while (true)
        {
            TBMIterateResult *tbmResult = tbm_iterate(iterator);
            int ituple;
            if (tbmResult == NULL){
                break;
            }
            for (ituple = 0; ituple < tbmResult->ntuples; ituple++)
            {
                ItemPointerData itemPointer;
                bool            found;
                ItemPointerSet(&itemPointer, tbmResult->blockno, tbmResult->offsets[ituple]);
                itempointer_insert(myscanstate->bitmapResult, itemPointer, &found);
            }
        }
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // elapsed = (end.tv_sec - start.tv_sec) + 
        //       (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // printf("Calculate bitmap time: %.5f seconds\n", elapsed);

        // clock_gettime(CLOCK_MONOTONIC, &start);
        /* Initialize vector search*/
        {
            IndexWithBitmapScan *scan = myscanstate->scan;
            List                *orderByClauses = scan->orderByClauses;
            int                 nkeysOrderBy = list_length(orderByClauses);
            ScanKey             scanKeysOrderBy = (ScanKey) palloc(nkeysOrderBy * sizeof(ScanKeyData));
            int                 j = 0;
            ListCell            *lc;

            foreach(lc, orderByClauses)
            {
                Expr            *clause = (Expr*) lfirst(lc);
                ScanKey         scan_key = &scanKeysOrderBy[j++];
                if (IsA(clause, OpExpr))
                {
                    int         flags = 0;
                    Datum       scanvalue;
                    Oid         opno = ((OpExpr *)clause)->opno;
                    RegProcedure opfuncid = ((OpExpr *)clause)->opfuncid;
                    Expr        *leftop = (Expr *) linitial(((OpExpr*)clause)->args);
                    Expr        *rightop = (Expr *) lsecond(((OpExpr*)clause)->args);
                    Relation    indexRelation = myscanstate->vectorScanDesc->indexRelation;
                    Oid         opfamily;
                    int         op_strategy;
                    Oid         op_lefttype;
                    Oid         op_righttype;
                    AttrNumber  varattno;
                    
                    if (!IsA(leftop, Var))
                    {
                           ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 14")));
                    }
                    if (!IsA(rightop, Const))
                    {
                        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 15")));
                    }
                    varattno = ((Var *)leftop)->varattno;
                    opfamily = indexRelation->rd_opfamily[varattno - 1];
                    get_op_opfamily_properties(opno, opfamily, true, &op_strategy, &op_lefttype, &op_righttype);
                    flags |= SK_ORDER_BY;
                    scanvalue = ((Const *)rightop)->constvalue;

                    scan_key->sk_flags = flags;
                    scan_key->sk_attno = varattno;
                    scan_key->sk_strategy = op_strategy;
                    scan_key->sk_subtype = op_righttype;
                    scan_key->sk_collation = ((OpExpr *) clause)->inputcollid;
                    if (RegProcedureIsValid(opfuncid))
                    {
                        fmgr_info(opfuncid, &scan_key->sk_func);
                    }
                    else
                    {
                        Assert(flags & (SK_SEARCHNULL | SK_SEARCHNOTNULL));
                        MemSet(&scan_key->sk_func, 0, sizeof(scan_key->sk_func));
                    }
                    scan_key->sk_argument = scanvalue;
                }
                else
                {
                    ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not Supported Yet 13")));
                }
            }

            index_rescan(myscanstate->vectorScanDesc, NULL, 0, scanKeysOrderBy, nkeysOrderBy);
        }
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // elapsed = (end.tv_sec - start.tv_sec) + 
        //       (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // printf("init vector search time: %.5f seconds\n", elapsed);

        myscanstate->first = false;
    }

    while (vectorSearchMethod(myscanstate->bitmapResult ,inputIndexScanDesc, ForwardScanDirection))
    {
        // ItemPointer tid = &inputIndexScanDesc->xs_heaptid;
        if (!index_fetch_heap(inputIndexScanDesc, slot)){
            ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Cannot fetch tuple according to tid")));
        }
        return slot;
    }
    
    return ExecClearTuple(slot);
}

void EndIndexWithBitmapScan(CustomScanState *node)
{
    IndexWithBitmapScanState    *myscanstate = (IndexWithBitmapScanState*)node;
    IndexScanDesc               indexScanDesc = myscanstate->vectorScanDesc;
    if (IsA(myscanstate->bitmapScanState, BitmapOrState))
    {
        ExecEndBitmapOr((BitmapOrState*)myscanstate->bitmapScanState);
    }
    else if (IsA(myscanstate->bitmapScanState, BitmapAndState))
    {
        ExecEndBitmapAnd((BitmapAndState*)myscanstate->bitmapScanState);
    }
    else if (IsA(myscanstate->bitmapScanState, BitmapIndexScanState))
    {
        ExecEndBitmapIndexScan((BitmapIndexScanState*)myscanstate->bitmapScanState);
    }
    else
    {
        ereport(ERROR,(errcode(ERRCODE_DATA_EXCEPTION),errmsg("Not supported yet: bitmap scan state should be BitmapOrState, BitmapAndState or BitmapIndexScanState")));
    }
    if (node->ss.ps.ps_ResultTupleSlot)
        ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);
    ExecClearTuple(node->ss.ss_ScanTupleSlot);
    index_endscan(indexScanDesc);
    index_close(myscanstate->vectorIndex, NoLock);
}

void ReScanIndexWithBitmapScan (CustomScanState *node)
{

}

void MarkPosIndexWithBitmapScan (CustomScanState *node)
{

}

void RestrPosIndexWithBitmapScan (CustomScanState *node){

}

Size EstimateDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt)
{
    return 0;
}

void InitializeDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate)
{

}

void ReInitializeDSMIndexWithBitmapScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate)
{

}

void InitializeWorkerIndexWithBitmapScan (CustomScanState *node, shm_toc *toc, void *coordinate)
{

}

void ShutdownIndexWithBitmapScan (CustomScanState *node)
{

}

void ExplainIndexWithBitmapScan (CustomScanState *node, List *ancestors, ExplainState *es)
{

}



/* Functions for Pushdown scan*/
Plan*   generatePushDownScan(PlannerInfo *root, RelOptInfo *rel, CustomPath *best_path, List *tlist, List *clauses, List *custom_plans)
{
    CustomScan  *result = makeNode(CustomScan);
    Plan        *plan = &result->scan.plan;
    IndexPath   *ipath = (IndexPath*)linitial(best_path->custom_paths);
    plan->targetlist = tlist;
    plan->qual = NIL; // 不需要post-filter处理
    plan->lefttree = NULL;
    plan->righttree = NULL;
    
    result->scan.scanrelid = best_path->path.parent->relid;
    result->flags = best_path->flags;
    result->custom_exprs = NIL;
    result->custom_plans = custom_plans;
    result->custom_private = NIL;
    result->custom_scan_tlist = tlist;
    result->custom_relids = rel->relids; // TODO: not sure
    result->methods = &pushdownScanMethods;
    result->custom_private = lappend(result->custom_private, ipath->indexinfo);
    return (Plan*) result;
}

Node*   generatePushDownScanState(CustomScan *cscan)
{
    PushDownScanState   *result = NULL;
    CustomScanState     *customScanState = NULL;
    // IndexScan           *indexScan = (IndexScan*)linitial(cscan->custom_plans);
    IndexOptInfo        *indexInfo = (IndexOptInfo*)linitial(cscan->custom_private);
    result = palloc0(sizeof(PushDownScanState));
    customScanState = &result->customScanState;
    customScanState->ss.ps.type = T_CustomScanState;
    customScanState->flags = cscan->flags;
    customScanState->methods = &pushdownExecMethods;
    result->indexScanState = NULL;
    result->indexhookinfo = getIndexHookInfo(indexInfo->relam);
    return (Node*) result;
}

void Evaluate_TID(ItemPointer* tids, bool* results, int length, IndexScanDesc scan, ExprState  *qual, ExprContext  *econtext)
{
    bool all_dead = false;
    for (int i = 0; i < length; i++)
    {
        if (qual == NULL){
            results[i] = true;
            continue;
        }
        if (table_index_fetch_tuple(scan->xs_heapfetch, tids[i], scan->xs_snapshot, econtext->ecxt_scantuple, &scan->xs_heap_continue, &all_dead))
        {
            if (ExecQual(qual, econtext))
            {
                results[i] = true;
            }
            else
            {
                results[i] = false;
            }
        }
        else
        {
            results[i] = false;
        }
    }
}

static ItemPointer push_down_getnext_tid(PushDownScanState *scanState, ScanDirection direction)
{
    IndexScanState  *indexScanState = scanState->indexScanState;
    ExprState   *qual = indexScanState->ss.ps.qual;
    ExprContext *econtext = indexScanState->ss.ps.ps_ExprContext;
    IndexScanDesc   scan = indexScanState->iss_ScanDesc;
    bool found = false;
    // scanState->customScanState.ss.ss_ScanTupleSlot = table_slot_create(scan->heapRelation, NULL);
    econtext->ecxt_scantuple = scanState->customScanState.ss.ss_ScanTupleSlot;
    found = scanState->indexhookinfo->pushdownsearch_func(scan, direction, Evaluate_TID, qual, econtext);
    scan->kill_prior_tuple = false;
    scan->xs_heap_continue = false;
    if (!found)
    {
        if (scan->xs_heapfetch)
        {
            table_index_fetch_reset(scan->xs_heapfetch);
        }
        return NULL;
    }

    return &scan->xs_heaptid;
}

static bool push_down_scan_getnext_slot(PushDownScanState *scanState, TupleTableSlot  *slot)
{
    IndexScanState  *indexScanState = scanState->indexScanState;
    IndexScanDesc scan = indexScanState->iss_ScanDesc;
    for (;;)
	{
		if (!scan->xs_heap_continue)
		{
			ItemPointer tid;

			/* Time to fetch the next TID from the index */
			tid = push_down_getnext_tid(scanState, ForwardScanDirection);

			/* If we're out of index entries, we're done */
			if (tid == NULL)
				break;

			Assert(ItemPointerEquals(tid, &scan->xs_heaptid));
		}

		/*
		 * Fetch the next (or only) visible heap tuple for this index entry.
		 * If we don't find anything, loop around and grab the next TID from
		 * the index.
		 */
		Assert(ItemPointerIsValid(&scan->xs_heaptid));
		if (index_fetch_heap(scan, slot))
			return true;
	}

	return false;
}


void BeginPushDownScan(CustomScanState *node, EState *estate, int eflags)
{
    CustomScan  *customScan = (CustomScan*) node->ss.ps.plan;
    IndexScan   *indexScan = (IndexScan*) linitial(customScan->custom_plans);
    PushDownScanState *myScan = (PushDownScanState*) node;
    myScan->indexScanState = ExecInitIndexScan(indexScan, estate, eflags);
}

TupleTableSlot* ExecPushDownScan(CustomScanState *node)
{
    PushDownScanState *myScan = (PushDownScanState*) node;
    IndexScanState    *indexScanState = myScan->indexScanState;
    TupleTableSlot    *slot = NULL;

    if (indexScanState->iss_ScanDesc == NULL)
    {
        indexScanState->iss_ScanDesc = index_beginscan(indexScanState->ss.ss_currentRelation, indexScanState->iss_RelationDesc, indexScanState->ss.ps.state->es_snapshot, indexScanState->iss_NumScanKeys, indexScanState->iss_NumOrderByKeys);
        myScan->customScanState.ss.ss_ScanTupleSlot = table_slot_create(indexScanState->ss.ss_currentRelation, NULL);
        if (indexScanState->iss_NumRuntimeKeys == 0 || indexScanState->iss_RuntimeKeysReady)
        {
            index_rescan(indexScanState->iss_ScanDesc, indexScanState->iss_ScanKeys, indexScanState->iss_NumScanKeys, indexScanState->iss_OrderByKeys, indexScanState->iss_NumOrderByKeys);
        }
    }
    slot = myScan->customScanState.ss.ss_ScanTupleSlot;
    while (push_down_scan_getnext_slot(myScan, slot))
    {
        return slot;
    }

    return NULL;
}

void EndPushDownScan(CustomScanState *node)
{
    PushDownScanState *myScanState = (PushDownScanState*) node;
    ExecEndIndexScan(myScanState->indexScanState);
    if (node->ss.ps.ps_ResultTupleSlot)
        ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);
    ExecClearTuple(node->ss.ss_ScanTupleSlot);
}

void ReScanPushDownScan (CustomScanState *node)
{

}

void MarkPosPushDownScan (CustomScanState *node)
{

}

void RestrPosPushDownScan (CustomScanState *node){

}

Size EstimateDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt)
{
    return 0;
}

void InitializeDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate)
{

}

void ReInitializeDSMPushDownScan (CustomScanState *node, ParallelContext *pcxt, void *coordinate)
{

}

void InitializeWorkerPushDownScan (CustomScanState *node, shm_toc *toc, void *coordinate)
{

}

void ShutdownPushDownScan (CustomScanState *node)
{

}

void ExplainPushDownScan (CustomScanState *node, List *ancestors, ExplainState *es)
{

}


static void set_oids(void)
{
    TypeName    *tname = makeTypeNameFromNameList(list_make1(makeString("vector")));
    TypeName    *range_param_name = makeTypeNameFromNameList(list_make1(makeString("range_query_params")));
    Oid   ann_dwithin_params[3] = {InvalidOid, InvalidOid, FLOAT4OID};
    vector_oid = typenameTypeId(NULL, tname);
    ann_dwithin_params[0] = vector_oid;
    ann_dwithin_params[1] = vector_oid;
    range_query_params_oid = typenameTypeId(NULL, range_param_name);
    l2_distance_oid = OpernameGetOprid(list_make1(makeString("<->")), vector_oid, vector_oid);
    // range_query_funcid = LookupFuncName(list_make1(makeString("ANN_DWithin")), 3, ann_dwithin_params, false);
}

void register_hook(void){
    return;
    next_set_pathlist_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = set_custom_rel_pathlist;
    set_oids();
}

void unregister_hook(void){
    return;
    set_rel_pathlist_hook = next_set_pathlist_hook;
    next_set_pathlist_hook = NULL;
}