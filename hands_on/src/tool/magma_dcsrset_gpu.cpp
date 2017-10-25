#include <string>
#include <iostream>
#include "magma_v2.h"
#include "magmasparse.h"
#include "tool.hpp"
using namespace std;
magma_int_t magma_dcsrset_gpu(
    magma_int_t m,
    magma_int_t n,
    magmaIndex_ptr row,
    magmaIndex_ptr col,
    magmaDouble_ptr val,
    magma_d_matrix *A,
    magma_queue_t queue )
{
    A->num_rows = m;
    A->num_cols = n;
    magma_index_t nnz;
    magma_index_getvector( 1, row+m, 1, &nnz, 1, queue );
    A->nnz = (magma_int_t) nnz;
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_DEV;
    A->dval = val;
    A->dcol = col;
    A->drow = row;

    return MAGMA_SUCCESS;
}
