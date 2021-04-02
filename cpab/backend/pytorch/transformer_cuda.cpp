#include <torch/extension.h>

// Cuda forward declaration
// at::Tensor cpab_cuda_forward(at::Tensor points_in, at::Tensor trels_in,  
//                              at::Tensor nstepsolver_in, at::Tensor nc_in, 
// 							 const int broadcast, at::Tensor output);
// at::Tensor cpab_cuda_backward(at::Tensor points_in, at::Tensor As_in, 
//                               at::Tensor Bs_in, at::Tensor nstepsolver_in,
//                               at::Tensor nc, const int broadcast, at::Tensor output);
at::Tensor cuda_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, at::Tensor output);
at::Tensor cuda_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient);
at::Tensor cuda_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient);

// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// FUNCTIONS

at::Tensor torch_get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, at::transpose(theta, 0, 1));//.reshape({-1,2});
}

at::Tensor torch_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    
    // Problem size
    const int n_points = points.size(0);

    // Allocate output
    auto output = torch::zeros({n_points}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Call kernel launcher
    return cuda_get_cell(points, xmin, xmax, nc, output);
}

at::Tensor torch_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);
    
    // Call kernel launcher
    return cuda_get_velocity(points, theta, At, xmin, xmax, nc, output);
}


// INTEGRATION

at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_numeric(points, theta, At, xmin, xmax, nc, nSteps1, nSteps2, output);
}

at::Tensor torch_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_closed_form(points, theta, At, xmin, xmax, nc, output);
}

// DERIVATIVE

at::Tensor torch_derivative_numeric(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

    at::Tensor phi_1 =  torch_integrate_numeric(points, theta, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

at::Tensor torch_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_closed_form(points, theta, At, Bt, xmin, xmax, nc, gradient);
}


// TRANSFORMATION

at::Tensor torch_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int e = 3;

    // Allocate output
    auto output = torch::zeros({n_batch, n_points, e}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_closed_form_trace(points, theta, At, xmin, xmax, nc, output);
}

at::Tensor torch_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(output);
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_closed_form_trace(output, points, theta, At, Bt, xmin, xmax, nc, gradient);
}


at::Tensor torch_derivative_numeric_trace(at::Tensor phi_1, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Do input checking
    CHECK_INPUT(phi_1);
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

    // at::Tensor phi_1 =  torch_integrate_numeric(points, theta, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta_2.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}



// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_cell", &torch_get_cell, "Get cell");
    m.def("get_velocity", &torch_get_velocity, "Get Velocity");
    m.def("integrate_closed_form", &torch_integrate_closed_form, "Integrate closed form");
    m.def("integrate_numeric", &torch_integrate_numeric, "Integrate numeric");
    m.def("derivative_closed_form", &torch_derivative_closed_form, "Derivative closed form");
    m.def("derivative_numeric", &torch_derivative_numeric, "Derivative numeric");
    m.def("integrate_closed_form_trace", &torch_integrate_closed_form_trace, "Integrate closed form trace");
    m.def("derivative_closed_form_trace", &torch_derivative_closed_form_trace, "Derivative closed form trace");
    m.def("derivative_numeric_trace", &torch_derivative_numeric_trace, "Derivative numeric trace");
}