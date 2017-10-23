////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    tool.hpp
/// @brief   The tool header.
///
/// @author  Yuhsiang Mike Tsai

#ifndef SCSC_TOOL_HPP
#define SCSC_TOOL_HPP

#include <string>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Change string to argc/argv.
///
///
void string2arg(std::string str, int *argc, char ***argv);

magma_int_t magma_dcsrset_gpu(
    magma_int_t m,
    magma_int_t n,
    magmaIndex_ptr row,
    magmaIndex_ptr col,
    magmaDouble_ptr val,
    magma_d_matrix *A,
    magma_queue_t queue );

#endif  // SCSC_TOOL_HPP
