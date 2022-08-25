// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>

#include "defines.inc"
#include "torch_proxy.h"

#ifdef __TORCH_FTN_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifndef NDEBUG
#define __TORCH_FTN_DEBUG 1
#endif

#ifdef __TORCH_FTN_DEBUG
#include <cstdarg>
#endif


using ftn_shape_type = int32_t;

/*
 * Private functions
 */
namespace {

#ifdef __TORCH_FTN_DEBUG
#define debug_print printf
#else
void debug_print(const char* format, ...) {}
#endif

bool is_present_flag(int flags, int probe) {
    return (flags & probe) != 0;
}
    
bool is_device_ptr(void* ptr) {
#ifdef __TORCH_FTN_USE_CUDA
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return attributes.devicePointer != NULL;
#else
    return false;
#endif
}

template<typename T>
torch::Tensor tensor_from_array(const std::vector<ftn_shape_type>& f_shape, T* data) {
    torch::Device            device = torch::kCPU;
    if (is_device_ptr(data)) device = torch::kCUDA;

    std::vector<int64_t> shape(f_shape.size());
    for (size_t i=0; i<f_shape.size(); i++) {
        shape[i] = f_shape[ f_shape.size() - (i+1) ];
    }

    debug_print("Creating %s tensor with shape [", device == torch::kCPU ? "CPU" : "GPU");
    for (auto s : shape) {
        debug_print("%lu ", s);
    }
    debug_print("]\n");

    return torch::from_blob(data, shape, [] (void *) {}, device);
}

std::unordered_map<void*, void*> device_host_map;

}

/*
 * Module
 */
void torch_module_load_cpp(void** h_module, const char* file_name, int flags) {
    auto module = new torch::jit::Module;
    debug_print("Loading module from file %s... ", file_name);
    *module = torch::jit::load(file_name);
    if (is_present_flag(flags, TORCH_FTN_MODULE_USE_DEVICE)) {
         module->to(at::kCUDA);
    }
    *h_module = static_cast<void*>(module);
    debug_print("done. Handle %p, %s backend\n", *h_module,
        is_present_flag(flags, TORCH_FTN_MODULE_USE_DEVICE) ? "GPU" : "CPU");
}

void torch_module_forward_cpp(void* h_module, void* h_input, void** h_output, int flags) {
    c10::InferenceMode mode( is_present_flag(flags, TORCH_FTN_MODULE_USE_INFERENCE_MODE) );
    auto module = static_cast<torch::jit::Module*>(h_module);
    auto input  = static_cast<torch::Tensor*>(h_input);
    auto output = new torch::Tensor;

    debug_print("Module %p :: forward\n", h_module);
    std::vector<torch::jit::IValue> inputs{*input};
    *output = module->forward(inputs).toTensor().contiguous();
    *h_output = static_cast<void*>(output);
}

void torch_module_train_cpp(void* h_module, void* h_input, void* h_target, void* h_optimizer, float* loss) {
    auto module = static_cast<torch::jit::Module*>(h_module);
    auto input  = static_cast<torch::Tensor*>(h_input);
    auto target = static_cast<torch::Tensor*>(h_target);
    auto optim  = static_cast<torch::optim::Optimizer*>(h_optimizer);

    optim->zero_grad();
    std::vector<torch::jit::IValue> inputs{*input};
    auto prediction = module->forward(inputs).toTensor();
    auto loss_tensor = torch::nn::functional::mse_loss(prediction, *target);
    loss_tensor.backward();
    optim->step();
    *loss = loss_tensor.template item<float>();
}

void torch_module_save_cpp(void* h_module, char* filename) {
    auto module = static_cast<torch::jit::Module*>(h_module);
    module->save(filename);
}

void torch_module_free_cpp(void* h_module) {
    delete static_cast<torch::jit::Module*>(h_module);
}

/*
 * Miscellaneous
 */
void torch_optimizer_create_sgd_cpp(void** handle, void* h_module, float lr) {
    auto module = static_cast<torch::jit::Module*>(h_module);

    // https://github.com/pytorch/pytorch/issues/28478
    std::vector<at::Tensor> parameters;
    for (const auto& param : module->parameters()) {
        parameters.push_back(param);
    }
    torch::optim::Optimizer* optimizer = new torch::optim::SGD(parameters, lr);
    *handle = static_cast<void*>(optimizer);
}

void torch_optimizer_free_cpp(void* handle) {
    delete static_cast<torch::optim::Optimizer*>(handle);
}

/*
 * Tensor
 */
void torch_tensor_from_array_float_cpp(void** handle, void* array, int arr_rank, ftn_shape_type* arr_shape, int elem_type, int elem_size) {
    auto tensor = new torch::Tensor;

    if        (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 4) {
        *tensor = tensor_from_array(std::vector<ftn_shape_type>(arr_shape, arr_shape+arr_rank), (float*)  array);
    } else if (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 8) {
        *tensor = tensor_from_array(std::vector<ftn_shape_type>(arr_shape, arr_shape+arr_rank), (double*) array);
    } else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 4) {
        *tensor = tensor_from_array(std::vector<ftn_shape_type>(arr_shape, arr_shape+arr_rank), (int32_t*)array);
    } else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 8) {
        *tensor = tensor_from_array(std::vector<ftn_shape_type>(arr_shape, arr_shape+arr_rank), (int64_t*)array);
    } else {
        throw std::runtime_error("No known conversion to tensor from Fortran array with element size "+std::to_string(elem_size));
    }

    *handle = static_cast<void*>(tensor);
}

void torch_tensor_to_array_cpp(void* handle,
        void** host_ptr, void** device_ptr, int arr_rank, ftn_shape_type* arr_shape, int elem_size) {
    auto tensor  = static_cast<torch::Tensor*>(handle);
    
    if (arr_rank != tensor->dim()) {
        throw std::runtime_error("Ranks mismatch passing Tensor to Fortran, expected "
            +std::to_string(tensor->dim())+" got "+std::to_string(arr_rank));
    }
    
    // Reverse the axes for Fortran
    size_t size_bytes = 0;
    for (int d=0; d<arr_rank; d++) {
        arr_shape[arr_rank - (d+1)] = tensor->size(d);
        size_bytes += tensor->size(d);
    }
    size_bytes *= elem_size;

    auto ptr = static_cast<void*>(tensor->data_ptr());
    if (is_device_ptr(ptr)) {
        debug_print("Passing GPU tensor %p to Fortran array\n", handle);

        if (device_host_map.find(ptr) == device_host_map.end()) {
            device_host_map[ptr] = new int8_t[size_bytes];
        }
        *host_ptr = device_host_map[ptr];
        *device_ptr = ptr;
    } else {
        debug_print("Passing CPU %p to Fortran array\n", handle);

        *host_ptr = ptr;
        *device_ptr = nullptr;
    }
}

void torch_tensor_backward(void* handle) {
    auto tensor  = static_cast<torch::Tensor*>(handle);
    tensor->backward();
}

void torch_tensor_free_cpp(void* handle, void* host_ptr, void* device_ptr) {
    if (host_ptr != 0 && device_ptr != 0) {
        delete[] (int8_t*)device_host_map[device_ptr];
        device_host_map.erase(device_ptr);
    }
    delete static_cast<torch::Tensor*>(handle);
}

void* torch_helper_ptr_to_devptr_cpp(void* ptr) {
    // Well, that's that
    return ptr;
}

