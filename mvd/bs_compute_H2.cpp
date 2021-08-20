#include <math.h>
#include "mex.h"
#include <omp.h>

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(!mxIsSingle(prhs[0])||mxIsComplex(prhs[0])||mxGetNumberOfDimensions(prhs[0])!=2) //A
    {
        mexErrMsgTxt("The first input must be single, real, two dimensional matrix...");
    }
    ///////////////////////////////////
    if(!mxIsInt32(prhs[1])||mxIsComplex(prhs[1])) //x1
    {
        mexErrMsgTxt("The second input must be int32, real, one column vector......");
    }
    if(!mxIsInt32(prhs[2])||mxIsComplex(prhs[2]))//y1
    {
        mexErrMsgTxt("The third input must be int32, real, one column vector...");
    }
    if(!mxIsSingle(prhs[3])||mxIsComplex(prhs[3])) //v1
    {
        mexErrMsgTxt("The third input must be single, real, one column vector...");
    }
    if(!mxIsInt32(prhs[4])||mxIsComplex(prhs[4])) //x2
    {
        mexErrMsgTxt("The fifth input must be int32, real, one column vector......");
    }
    if(!mxIsInt32(prhs[5])||mxIsComplex(prhs[5]))//y2
    {
        mexErrMsgTxt("The sixth input must be int32, real, one column vector...");
    }
    if(!mxIsSingle(prhs[6])||mxIsComplex(prhs[6])) //v2
    {
        mexErrMsgTxt("The seventh input must be single, real, one column vector...");
    }
    ///////////////////////////////////
    if(nrhs!=7)
    {
        mexErrMsgTxt("The number of input arguments is wrong...");
    }
    if(nlhs!=1)
    {
        mexErrMsgTxt("The number of output arguments is wrong...");
    }
    float *A = (float*) mxGetPr(prhs[0]);
    
    int *x1 = (int*) mxGetPr(prhs[1]);
    int *y1 = (int*) mxGetPr(prhs[2]);
    float *v1 = (float*) mxGetPr(prhs[3]);
    int *x2 = (int*) mxGetPr(prhs[4]);
    int *y2 = (int*) mxGetPr(prhs[5]);
    float *v2 = (float*) mxGetPr(prhs[6]);
    
    int M = mxGetM(prhs[0]);
    int N = mxGetN(prhs[0]);
    int non_zero_num1 = mxGetM(prhs[1]);
    int non_zero_num2 = mxGetM(prhs[4]);
    
    plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float *J = (float*) mxGetPr(plhs[0]);
    float tmp_J = 0;
    
#pragma omp parallel for shared(A, x1, y1, v1, x2, y2, v2, M, N) reduction(+:tmp_J) schedule(static)
    for(int iter_a = 0; iter_a < non_zero_num1; iter_a++)
    {
        for(int iter_b = 0; iter_b < non_zero_num2; iter_b++)
        {
            float tmp1 = A[x2[iter_b] + x1[iter_a]*M];
            float tmp2 = A[y2[iter_b] + y1[iter_a]*M];
            tmp_J += v1[iter_a]*v2[iter_b]*(tmp1-tmp2)*(tmp1-tmp2);
        }
    }
    J[0] = tmp_J;
}