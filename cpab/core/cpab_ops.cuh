#ifndef CPAB_OPS_GPU
#define CPAB_OPS_GPU


__device__ int get_cell(float x, const float xmin, const float xmax, const int nc);
__device__ float get_velocity(float x, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc);


__global__ void kernel_get_cell(
    const int n_points, const float* points, 
    const float xmin, const float xmax, const int nc, 
    int* newpoints);

__global__ void kernel_get_velocity(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints);

__global__ void kernel_integrate_numeric(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, 
    float* newpoints);

__global__ void kernel_integrate_closed_form(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc,  
    float* newpoints);

__global__ void kernel_derivative_closed_form(
    const int n_points, const int n_batch, const int d,
    const float* x, const float* A, const float* B, 
    const int xmin, const int xmax, const int nc, 
    float* gradpoints);


__global__ void kernel_integrate_closed_form_trace(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc,  
    float* newpoints);

__global__ void kernel_derivative_closed_form_trace(
    const int n_points, const int n_batch, const int d,
    const float* newpoints, const float* x, const float* A, const float* B, 
    const float xmin, const float xmax, const int nc, 
    float* gradpoints);
  
#endif