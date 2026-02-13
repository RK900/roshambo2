// MIT License
// 
// Copyright (c) 2025 molecularinformatics  
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <pybind11/stl.h>
#include <string>
#include <cassert>
#include <cuda_runtime_api.h>
#include "cuda_functions.cuh"
#include <omp.h>
#include <vector>

namespace py = pybind11;

enum loglevel {
  OFF,
  INFO,
  DEBUG
};

// Keep the original function for backward compatibility
void optimize_overlap_color(py::array_t<float> A, py::array_t<int> AT, py::array_t<int> AN,
                            py::array_t<float> B, py::array_t<int> BT, py::array_t<int> BN, 
                            py::array_t<float> RMAT, py::array_t<float> PMAT, py::array_t<float> V,
                            bool optim_color, float mixing_param, float lr_q, float lr_t, int nsteps, 
                            int start_mode_method, int requested_gpus, int loglevel);

// New Persistent Context Class
class CudaOverlapContext {
public:
    int n_gpus;
    int max_mols;
    int max_atoms; // Max atoms per molecule (padded size)
    int n_features;
    int scores_dim; // usually 20
    
    // Per-device resources
    struct DeviceResources {
        float * molBs_device;
        int * molB_type_device;
        int * molB_num_atoms_device;
        
        float * molA_device;
        int * molA_type_device;
        
        float * rmat_device;
        float * pmat_device;
        float * scores_device;
        
        long molBs_size;
        long molB_type_size;
        long molB_num_atoms_size;
        long molA_size;
        long molA_type_size;
        long rmat_size;
        long pmat_size;
        long scores_size;
    };
    
    std::vector<DeviceResources> resources;
    std::vector<int> device_ids;

    CudaOverlapContext(int requested_gpus, int max_mols, int max_atoms, int n_features, int scores_dim, int device_offset=0)
        : n_gpus(requested_gpus), max_mols(max_mols), max_atoms(max_atoms), n_features(n_features), scores_dim(scores_dim) {

        int deviceCount = 0;
        CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));

        if (deviceCount < requested_gpus + device_offset) {
             throw std::runtime_error("roshambo2.cuda: Error: "+std::to_string(requested_gpus)+" GPUs requested with offset "+std::to_string(device_offset)+" but only "+std::to_string(deviceCount)+" GPUs found on machine!.");
        }

        resources.resize(n_gpus);
        device_ids.resize(n_gpus);

        // Calculate max sizes per GPU (assuming we might split max_mols across GPUs if needed,
        // but for now we allocate full capacity buffers or split capacity?
        // To be safe and simple: let's assume we split the BATCH across GPUs.
        // So each GPU needs to handle at most ceil(max_mols / n_gpus) molecules.

        int mols_per_gpu = (max_mols + n_gpus - 1) / n_gpus;

        #pragma omp parallel for num_threads(n_gpus)
        for(int k=0; k<n_gpus; ++k) {
            int device_id = (k + device_offset) % deviceCount;
            device_ids[k] = device_id;
            
            CUDA_CHECK_ERROR(cudaSetDevice(device_id));
            
            DeviceResources &res = resources[k];
            
            // Allocate Mol B buffers (Data Mols)
            // f_x: (mols_per_gpu, max_atoms, 4)
            res.molBs_size = (long)mols_per_gpu * max_atoms * 4 * sizeof(float);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.molBs_device, res.molBs_size));
            
            // f_types: (mols_per_gpu, max_atoms)
            res.molB_type_size = (long)mols_per_gpu * max_atoms * sizeof(int);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.molB_type_device, res.molB_type_size));

            // f_n_real: (mols_per_gpu)
            res.molB_num_atoms_size = (long)mols_per_gpu * sizeof(int);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.molB_num_atoms_device, res.molB_num_atoms_size));
            
            // Allocate Mol A buffers (Query Mol) - assuming 1 query at a time for simplicity in this context?
            // Or batch query? The original code loops over queries. 
            // In training loop, we usually have 1 query vs N data.
            // Let's allocate for 1 query molecule with max_atoms
            res.molA_size = (long)1 * max_atoms * 4 * sizeof(float); // 1 query, max atoms, 4 dims
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.molA_device, res.molA_size));

            res.molA_type_size = (long)1 * max_atoms * sizeof(int);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.molA_type_device, res.molA_type_size));
            
            // Interaction Matrices
            res.rmat_size = (long)n_features * n_features * sizeof(float);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.rmat_device, res.rmat_size));
            
            res.pmat_size = (long)n_features * n_features * sizeof(float);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.pmat_device, res.pmat_size));
            
            // Scores
            res.scores_size = (long)mols_per_gpu * scores_dim * sizeof(float);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&res.scores_device, res.scores_size));
        }
    }
    
    ~CudaOverlapContext() {
        for(int k=0; k<n_gpus; ++k) {
            CUDA_CHECK_ERROR(cudaSetDevice(device_ids[k]));
            DeviceResources &res = resources[k];
            cudaFree(res.molBs_device);
            cudaFree(res.molB_type_device);
            cudaFree(res.molB_num_atoms_device);
            cudaFree(res.molA_device);
            cudaFree(res.molA_type_device);
            cudaFree(res.rmat_device);
            cudaFree(res.pmat_device);
            cudaFree(res.scores_device);
        }
    }
    
    // optimize_overlap method re-using buffers
    void optimize(py::array_t<float> A, py::array_t<int> AT, py::array_t<int> AN,
                 py::array_t<float> B, py::array_t<int> BT, py::array_t<int> BN, 
                 py::array_t<float> RMAT, py::array_t<float> PMAT, py::array_t<float> V,
                 bool optim_color, float mixing_param, float lr_q, float lr_t, int nsteps, 
                 int start_mode_method, int loglevel) {
                 
        // Check input sizes fit in pre-allocated buffers
        auto molBs = B.unchecked<3>();
        long current_n_mols = molBs.shape(0);
        long current_mol_atoms = molBs.shape(1);
        
        if (current_n_mols > max_mols) {
            throw std::runtime_error("Input batch size exceeds max_mols allocated in context");
        }
        if (current_mol_atoms > max_atoms) {
            throw std::runtime_error("Input molecule size exceeds max_atoms allocated in context");
        }
        
        // Split batch across GPUs
        int base_len = current_n_mols / n_gpus;
        int remainder = current_n_mols % n_gpus;
        
        int start_index[n_gpus];
        int chunk_size[n_gpus];
        
        int l=0;
        for(int k=0; k<n_gpus; ++k){
            start_index[k] = l;
            chunk_size[k] = base_len + (k==0 ? remainder : 0);
            l += chunk_size[k];
        }
        
        auto molA = A.unchecked<3>();
        auto molA_type = AT.unchecked<2>();
        auto molA_num_atoms = AN.unchecked<1>();
        
        // Interaction matrices pointers
        auto rmat = RMAT.unchecked<2>();
        auto pmat = PMAT.unchecked<2>();
        const float * ptr_rmat = rmat.data(0,0);
        const float * ptr_pmat = pmat.data(0,0);
        
        // Loop over queries (usually 1)
        for(py::ssize_t i=0; i < molA.shape(0); ++i) {
            
             #pragma omp parallel for num_threads(n_gpus)
             for(int k=0; k<n_gpus; ++k) {
                 if (chunk_size[k] == 0) continue;
                 
                 int device_id = device_ids[k];
                 CUDA_CHECK_ERROR(cudaSetDevice(device_id));
                 DeviceResources &res = resources[k];
                 
                 // Copy Query Mol A
                 long molA_copy_size = molA.shape(1) * molA.shape(2) * sizeof(float);
                 CUDA_CHECK_ERROR(cudaMemcpy(res.molA_device, molA.data(i,0,0), molA_copy_size, cudaMemcpyHostToDevice));
                 
                 long molA_type_copy_size = molA_type.shape(1) * sizeof(int);
                 CUDA_CHECK_ERROR(cudaMemcpy(res.molA_type_device, molA_type.data(i,0), molA_type_copy_size, cudaMemcpyHostToDevice));
                 
                 int molA_atoms_i = molA_num_atoms(i);

                 // Copy Reference Matrices (only if changed? for now copy every time to be safe/simple)
                 // Or we could have an update_matrices method. optimize_overlap usually implies potentially different matrices.
                 CUDA_CHECK_ERROR(cudaMemcpy(res.rmat_device, ptr_rmat, res.rmat_size, cudaMemcpyHostToDevice));
                 CUDA_CHECK_ERROR(cudaMemcpy(res.pmat_device, ptr_pmat, res.pmat_size, cudaMemcpyHostToDevice));
                 
                 // Copy Batch Data Mol B chunk
                 auto molB = B.unchecked<3>();
                 auto molB_type = BT.unchecked<2>();
                 auto molB_num_atoms = BN.unchecked<1>();
                 
                 int s_idx = start_index[k];
                 int c_size = chunk_size[k];
                 
                 const float * ptr_molB = molB.data(s_idx, 0, 0);
                 long molB_copy_size = c_size * molB.shape(1) * molB.shape(2) * sizeof(float);
                 CUDA_CHECK_ERROR(cudaMemcpy(res.molBs_device, ptr_molB, molB_copy_size, cudaMemcpyHostToDevice));
                 
                 const int * ptr_molB_type = molB_type.data(s_idx, 0);
                 long molB_type_copy_size = c_size * molB_type.shape(1) * sizeof(int);
                 CUDA_CHECK_ERROR(cudaMemcpy(res.molB_type_device, ptr_molB_type, molB_type_copy_size, cudaMemcpyHostToDevice));
                 
                 const int * ptr_molB_num = molB_num_atoms.data(s_idx);
                 long molB_num_copy_size = c_size * sizeof(int);
                 CUDA_CHECK_ERROR(cudaMemcpy(res.molB_num_atoms_device, ptr_molB_num, molB_num_copy_size, cudaMemcpyHostToDevice));
                 
                 // Run Kernel
                 // Note: we need to pass the ACTUAL dimensions of the data we just copied, not the max buffers
                 optimize_overlap_gpu(
                     res.molA_device, res.molA_type_device, molA_atoms_i, current_mol_atoms,
                     res.molBs_device, res.molB_type_device, res.molB_num_atoms_device, current_mol_atoms, c_size,
                     res.rmat_device, res.pmat_device, n_features, res.scores_device,
                     optim_color, lr_q, lr_t, nsteps, mixing_param, start_mode_method, device_id
                 );
                 
                 // Copy Scores Back
                 float * scores_host = V.mutable_data(i, s_idx, 0);
                 long scores_copy_size = c_size * scores_dim * sizeof(float);
                 CUDA_CHECK_ERROR(cudaMemcpy(scores_host, res.scores_device, scores_copy_size, cudaMemcpyDeviceToHost));
             }
        }
    }

    // Zero-copy torch::Tensor path - avoids CPU<->GPU round-trips entirely
    // When all tensors are already on the correct CUDA device, passes data_ptrs
    // directly to the kernel with no memory copies for input data.
    // Falls back to device-to-device copies for multi-GPU or cross-device cases.
    void optimize_torch(torch::Tensor A, torch::Tensor AT, torch::Tensor AN,
                        torch::Tensor B, torch::Tensor BT, torch::Tensor BN,
                        torch::Tensor RMAT, torch::Tensor PMAT, torch::Tensor V,
                        bool optim_color, float mixing_param, float lr_q, float lr_t, int nsteps,
                        int start_mode_method, int loglevel) {

        TORCH_CHECK(A.dim() == 3, "A must be 3D (n_queries, max_atoms, 4)");
        TORCH_CHECK(B.dim() == 3, "B must be 3D (n_mols, max_atoms, 4)");
        TORCH_CHECK(V.dim() == 3, "V must be 3D (n_queries, n_mols, scores_dim)");

        A = A.contiguous().to(torch::kFloat32);
        B = B.contiguous().to(torch::kFloat32);
        AT = AT.contiguous().to(torch::kInt32);
        BT = BT.contiguous().to(torch::kInt32);
        RMAT = RMAT.contiguous().to(torch::kFloat32);
        PMAT = PMAT.contiguous().to(torch::kFloat32);

        // AN/BN are small (one int per molecule) - move to CPU for host-side access
        auto AN_cpu = AN.to(torch::kCPU).contiguous().to(torch::kInt32);
        auto BN_cpu = BN.to(torch::kCPU).contiguous().to(torch::kInt32);

        long current_n_mols = B.size(0);
        long current_mol_atoms = B.size(1);
        long query_mol_atoms = A.size(1);

        if (current_n_mols > max_mols) {
            throw std::runtime_error("Batch size " + std::to_string(current_n_mols) +
                                     " exceeds max_mols " + std::to_string(max_mols));
        }

        // Single-GPU zero-copy fast path (common case in DDP training)
        if (n_gpus == 1) {
            int device_id = device_ids[0];
            CUDA_CHECK_ERROR(cudaSetDevice(device_id));

            auto target_device = torch::Device(torch::kCUDA, device_id);
            bool all_on_device = A.is_cuda() && A.device().index() == device_id &&
                                 B.is_cuda() && B.device().index() == device_id;

            if (all_on_device) {
                // Ensure interaction matrices and BN are on device
                auto rmat_d = RMAT.to(target_device).contiguous();
                auto pmat_d = PMAT.to(target_device).contiguous();
                auto BN_d = BN.to(target_device).contiguous().to(torch::kInt32);

                // Ensure output tensor is on device
                TORCH_CHECK(V.is_cuda() && V.device().index() == device_id,
                            "Output tensor V must be on CUDA device " + std::to_string(device_id));

                // Loop over queries - pass tensor data_ptrs directly to kernel
                for (int64_t i = 0; i < A.size(0); ++i) {
                    int molA_atoms_i = AN_cpu.data_ptr<int>()[i];

                    optimize_overlap_gpu(
                        A.data_ptr<float>() + i * query_mol_atoms * 4,
                        AT.data_ptr<int>() + i * query_mol_atoms,
                        molA_atoms_i,
                        (int)query_mol_atoms,
                        B.data_ptr<float>(),
                        BT.data_ptr<int>(),
                        BN_d.data_ptr<int>(),
                        (int)current_mol_atoms,
                        current_n_mols,
                        rmat_d.data_ptr<float>(),
                        pmat_d.data_ptr<float>(),
                        n_features,
                        V.data_ptr<float>() + i * V.size(1) * V.size(2),
                        optim_color, lr_q, lr_t, nsteps, mixing_param,
                        start_mode_method, device_id
                    );
                }
                return;
            }
        }

        // Fallback: multi-GPU or cross-device — copy to pre-allocated buffers
        int base_len = current_n_mols / n_gpus;
        int remainder = current_n_mols % n_gpus;

        std::vector<int> start_index(n_gpus), chunk_size(n_gpus);
        int l = 0;
        for (int k = 0; k < n_gpus; ++k) {
            start_index[k] = l;
            chunk_size[k] = base_len + (k == 0 ? remainder : 0);
            l += chunk_size[k];
        }

        for (int64_t i = 0; i < A.size(0); ++i) {
            #pragma omp parallel for num_threads(n_gpus)
            for (int k = 0; k < n_gpus; ++k) {
                if (chunk_size[k] == 0) continue;

                int device_id = device_ids[k];
                CUDA_CHECK_ERROR(cudaSetDevice(device_id));
                DeviceResources &res = resources[k];

                // Determine copy kind based on source tensor location
                cudaMemcpyKind input_copy = A.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                cudaMemcpyKind rmat_copy = RMAT.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

                // Copy Query Mol A
                long molA_copy_size = query_mol_atoms * 4 * sizeof(float);
                CUDA_CHECK_ERROR(cudaMemcpy(res.molA_device,
                    A.data_ptr<float>() + i * query_mol_atoms * 4,
                    molA_copy_size, input_copy));

                long molA_type_copy_size = query_mol_atoms * sizeof(int);
                CUDA_CHECK_ERROR(cudaMemcpy(res.molA_type_device,
                    AT.data_ptr<int>() + i * query_mol_atoms,
                    molA_type_copy_size, input_copy));

                int molA_atoms_i = AN_cpu.data_ptr<int>()[i];

                // Copy Interaction Matrices
                CUDA_CHECK_ERROR(cudaMemcpy(res.rmat_device, RMAT.data_ptr<float>(), res.rmat_size, rmat_copy));
                CUDA_CHECK_ERROR(cudaMemcpy(res.pmat_device, PMAT.data_ptr<float>(), res.pmat_size, rmat_copy));

                // Copy Batch Data Mol B chunk
                int s_idx = start_index[k];
                int c_size = chunk_size[k];

                const float *ptr_molB = B.data_ptr<float>() + (long)s_idx * current_mol_atoms * 4;
                long molB_copy_size = (long)c_size * current_mol_atoms * 4 * sizeof(float);
                CUDA_CHECK_ERROR(cudaMemcpy(res.molBs_device, ptr_molB, molB_copy_size, input_copy));

                const int *ptr_molB_type = BT.data_ptr<int>() + (long)s_idx * current_mol_atoms;
                long molB_type_copy_size = (long)c_size * current_mol_atoms * sizeof(int);
                CUDA_CHECK_ERROR(cudaMemcpy(res.molB_type_device, ptr_molB_type, molB_type_copy_size, input_copy));

                const int *ptr_molB_num = BN_cpu.data_ptr<int>() + s_idx;
                long molB_num_copy_size = c_size * sizeof(int);
                CUDA_CHECK_ERROR(cudaMemcpy(res.molB_num_atoms_device, ptr_molB_num, molB_num_copy_size, cudaMemcpyHostToDevice));

                // Run Kernel
                optimize_overlap_gpu(
                    res.molA_device, res.molA_type_device, molA_atoms_i, (int)current_mol_atoms,
                    res.molBs_device, res.molB_type_device, res.molB_num_atoms_device, (int)current_mol_atoms, c_size,
                    res.rmat_device, res.pmat_device, n_features, res.scores_device,
                    optim_color, lr_q, lr_t, nsteps, mixing_param, start_mode_method, device_id
                );

                // Copy Scores to output tensor
                cudaMemcpyKind scores_copy = V.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
                float *scores_out = V.data_ptr<float>() + i * V.size(1) * V.size(2) + (long)s_idx * scores_dim;
                long scores_copy_size = (long)c_size * scores_dim * sizeof(float);
                CUDA_CHECK_ERROR(cudaMemcpy(scores_out, res.scores_device, scores_copy_size, scores_copy));
            }
        }
    }

};


// Original Function Implementation (Copy-Pasted from before but cleaned up/maintained for backwards compat)
void optimize_overlap_color(py::array_t<float> A, py::array_t<int> AT, py::array_t<int> AN,
                            py::array_t<float> B, py::array_t<int> BT, py::array_t<int> BN, 
                            py::array_t<float> RMAT, py::array_t<float> PMAT, py::array_t<float> V,
                            bool optim_color, float mixing_param, float lr_q, float lr_t, int nsteps, 
                            int start_mode_method, int requested_gpus, int loglevel){
    
    // Use the Context class internally for simplicity if we wanted, or keep the old implementation.
    // Given the old implementation was massive and we want to replace it effectively, 
    // let's just create a temporary context and run it.
    // This is less efficient than a persistent context but ensures code reuse and correctness.
    
    // Determine max sizes from input
    auto molBs = B.unchecked<3>();
    auto molAs = A.unchecked<3>();
    int max_mols = molBs.shape(0);
    int max_atoms = std::max(molBs.shape(1), molAs.shape(1));
    auto rmat = RMAT.unchecked<2>();
    int n_features = rmat.shape(0);
    
    CudaOverlapContext ctx(requested_gpus, max_mols, max_atoms, n_features, 20);
    ctx.optimize(A, AT, AN, B, BT, BN, RMAT, PMAT, V, optim_color, mixing_param, lr_q, lr_t, nsteps, start_mode_method, loglevel);
}


PYBIND11_MODULE(_roshambo2_cuda, m) {
    m.def("optimize_overlap_color", &optimize_overlap_color, "computes overlap of ref mol A with fit mols B");

    py::class_<CudaOverlapContext>(m, "CudaOverlapContext")
        .def(py::init<int, int, int, int, int, int>(), py::arg("requested_gpus"), py::arg("max_mols"), py::arg("max_atoms"), py::arg("n_features"), py::arg("scores_dim"), py::arg("device_offset")=0)
        .def("optimize", &CudaOverlapContext::optimize, "Run optimization with persistent context")
        .def("optimize_torch", &CudaOverlapContext::optimize_torch, "Run optimization with torch::Tensor zero-copy path");
}