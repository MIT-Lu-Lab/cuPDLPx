#ifndef PRESOLVE_H
#define PRESOLVE_H

#include "PSLP_API.h"
#include "PSLP_stats.h"
#include "cupdlpx.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        Presolver *presolver;
        Settings *settings;
        lp_problem_t *reduced_problem;
        bool problem_solved_during_presolve;
        double presolve_time;
        char presolve_status;
    } cupdlpx_presolve_info_t;

    cupdlpx_presolve_info_t *pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params);

    cupdlpx_result_t *create_result_from_presolve(const cupdlpx_presolve_info_t *info, const lp_problem_t *original_prob);

    const char *get_presolve_status_str(enum PresolveStatus_ status);

    void pslp_postsolve(const cupdlpx_presolve_info_t *info,
                        cupdlpx_result_t *reduced_result,
                        const lp_problem_t *original_prob);

    void cupdlpx_presolve_info_free(cupdlpx_presolve_info_t *info);

#ifdef __cplusplus
}
#endif

#endif // PRESOLVE_H