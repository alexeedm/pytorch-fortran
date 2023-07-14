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

#include <filesystem>
#include <mutex>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <torch/csrc/jit/python/pybind.h>
#include <pybind11/embed.h>

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

/*
 * Defines and shortcuts
 */
namespace py = pybind11;
using IWrap  = std::vector<torch::IValue>;
using Scratch = std::vector<char>;

/*
 * Private functions
 */
namespace {

#ifdef __TORCH_FTN_DEBUG
#define debug_print printf
#else
void debug_print(const char* format, ...) {}
#endif

template<typename Lambda>
inline auto wrap_exceptions_lambda(
    Lambda code, std::string prefix = "Caught PyTorch exception") -> decltype(code()) {

    try {
        return code();
    } catch(const std::exception &e){
        printf("%s: %s\n", prefix.c_str(), e.what());
        throw;
    }
}

#define WRAP_EXCEPTIONS(code) wrap_exceptions_lambda( [&] () { return code; } )

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

/*
 * Device to host scratch map
 */
using D2HScratchMap = std::unordered_map<void*, std::shared_ptr<Scratch>>;
D2HScratchMap device_scratch_map;
std::mutex h2s_mutex;

void update_map(void* ptr, std::shared_ptr<Scratch>& p_scratch) {
    if (device_scratch_map.find(ptr) == device_scratch_map.end() ||
        // Use const .at accessor for thread safety
        const_cast<D2HScratchMap&>(device_scratch_map).at(ptr) != p_scratch)
    {
        std::unique_lock<std::mutex> lock(h2s_mutex);
        device_scratch_map[ptr] = p_scratch;
    }
}

torch::Tensor tensor_from_array(void* array,
    int arr_rank, FtnShapeType* ftn_shape, int elem_type, int elem_size) {

    // Reverse axes fortran -> c
    std::vector<int64_t> torch_shape(arr_rank);
    for (int i=0; i<arr_rank; i++) {
        torch_shape[i] = ftn_shape[ arr_rank - (i+1) ];
    }

    // Select device
    bool is_on_gpu = is_device_ptr(array);
    torch::TensorOptions  options = torch::TensorOptions().device(torch::kCPU);
    if (is_on_gpu)        options = torch::TensorOptions().device(torch::kCUDA);

    // Select data type
    if      (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 4) { options = options.dtype(torch::kFloat32); }
    else if (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 8) { options = options.dtype(torch::kFloat64); }
    else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 4) { options = options.dtype(torch::kInt32);   }
    else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 8) { options = options.dtype(torch::kInt64);   }
    else {
        throw std::runtime_error("No known conversion to tensor from Fortran array with element size "+std::to_string(elem_size));
    }

    debug_print("Creating %s tensor with shape [", is_on_gpu ? "GPU" : "CPU");
    for (auto s : torch_shape) {
        debug_print("%lu ", s);
    }
    debug_print("]\n");

    return WRAP_EXCEPTIONS(torch::from_blob(array, torch_shape, [] (void *) {}, options));
}

/*
 * Class definitions
 */

struct JITModule {
public:
    torch::jit::Module jit_module;
    std::shared_ptr<Scratch> host_cache;
};

class PyModule {
public:
    py::module py_module;
    std::shared_ptr<Scratch> host_cache;

    PyModule(const std::string& path, const std::string& module_name) {
        std::lock_guard lock(mutex);
        if (initialized) {
            throw std::runtime_error("Only one PyModule can be opened at a time, trying to open " + module_name);
        }
        initialized = true;
        guard = std::make_unique<py::scoped_interpreter>();
        auto sys = py::module::import("sys");
        sys.attr("path").attr("insert")(1, path.c_str());
        py_module = py::module::import(module_name.c_str());
    }

    ~PyModule() = default;

private:
    std::unique_ptr<py::scoped_interpreter> guard;
    static std::mutex mutex;
    static bool initialized;
};

}

/*
 * Exceptions initiated from Fortran
 */
void torch_throw_cpp(const char* message) {
    throw std::runtime_error(message);
}

/*
 * Module
 */
void torch_module_load_cpp(void** h_module, const char* file_name, int flags) {
    // Free memory if we're loading a new module
    torch_module_free_cpp(*h_module);
    auto mod = new JITModule;
    mod->host_cache = std::make_shared<Scratch>(0);

    debug_print("Loading module from file '%s'... ", file_name);
    wrap_exceptions_lambda(
        [&] () { mod->jit_module = torch::jit::load(file_name); },
        std::string("Caugth PyTorch error opening file ") + file_name
    );

    if (is_present_flag(flags, TORCH_FTN_MODULE_USE_DEVICE)) {
        WRAP_EXCEPTIONS( mod->jit_module.to(torch::kCUDA) );
    }
    *h_module = static_cast<void*>(mod);
    debug_print("done. Handle %p, %s backend\n", *h_module,
        is_present_flag(flags, TORCH_FTN_MODULE_USE_DEVICE) ? "GPU" : "CPU");
}

void torch_module_forward_cpp(void* h_module, void* h_inputs, void** h_output, int flags) {
    c10::InferenceMode mode( is_present_flag(flags, TORCH_FTN_MODULE_USE_INFERENCE_MODE) );
    auto mod    = static_cast<JITModule*>(h_module);
    auto inputs = static_cast<IWrap*>(h_inputs);

    debug_print("Module %p :: forward(in: %p * %ld)\n", h_module, h_inputs, inputs->size());

    if (*h_output == nullptr) { *h_output = static_cast<void*>(new torch::Tensor); }
    auto p_tensor = static_cast<torch::Tensor*>(*h_output);
    *p_tensor = WRAP_EXCEPTIONS( mod->jit_module.forward(*inputs).toTensor().contiguous() );
    update_map(p_tensor->data_ptr(), mod->host_cache);
}

void torch_module_train_cpp(void* h_module, void* h_inputs, void* h_target, void* h_optimizer, float* loss) {
    auto mod = static_cast<JITModule*>(h_module);
    auto inputs = static_cast<IWrap*>(h_inputs);
    auto target = static_cast<torch::Tensor*>(h_target);
    auto optim  = static_cast<torch::optim::Optimizer*>(h_optimizer);

    debug_print("Module %p :: train(in: %p * %ld, target: %p)\n", h_module, h_inputs, inputs->size(), h_target);
    wrap_exceptions_lambda( [&] () {
        optim->zero_grad();
        auto prediction = mod->jit_module.forward(*inputs).toTensor();
        auto loss_tensor = torch::nn::functional::mse_loss(prediction, *target);
        loss_tensor.backward();
        optim->step();
        *loss = loss_tensor.template item<float>();
    });
}

void torch_module_save_cpp(void* h_module, char* filename) {
    auto mod = static_cast<JITModule*>(h_module);
    WRAP_EXCEPTIONS( mod->jit_module.save(filename) );
}

void torch_module_free_cpp(void* h_module) {
    delete static_cast<JITModule*>(h_module);
}

/*
 * Python Module
 */

bool PyModule::initialized = false;
std::mutex PyModule::mutex;

void torch_pymodule_load_cpp(void** h_module, const char* file_name) {
    std::filesystem::path path(file_name);
    auto folder = std::filesystem::absolute(path).parent_path().string();
    auto module_name = path.stem();

    debug_print("Loading Python module from file '%s': folder '%s' module name '%s' ",
        file_name, folder.c_str(), module_name.c_str());

    // Free memory if we're loading a new module
    torch_pymodule_free_cpp(*h_module);
    auto mod = WRAP_EXCEPTIONS( new PyModule(folder, module_name) );
    mod->host_cache = std::make_shared<Scratch>(0);

    *h_module = static_cast<void*>(mod);
    debug_print("done. Handle %p\n", h_module);
}

void torch_pymodule_forward_cpp(void* h_module, void* h_inputs, void** h_output) {
    auto mod    = static_cast<PyModule*>(h_module);
    auto inputs = static_cast<IWrap*>   (h_inputs);

    debug_print("PyModule %p :: forward(in: %p)\n", h_module, h_inputs);
    
    if (*h_output == nullptr) { *h_output = static_cast<void*>(new torch::Tensor); }
    auto p_tensor = static_cast<torch::Tensor*>(*h_output);
    *p_tensor = WRAP_EXCEPTIONS( mod->py_module.attr("ftn_pytorch_forward")(*inputs).cast<torch::Tensor>() );
    update_map(p_tensor->data_ptr(), mod->host_cache);
}

void torch_pymodule_train_cpp(void* h_module, void* h_inputs, void* h_target, bool* is_completed, float* loss) {
    auto mod     = static_cast<PyModule*>(h_module);
    auto inputs  = static_cast<IWrap*>(h_inputs);
    auto target  = static_cast<torch::Tensor*>(h_target);

    debug_print("PyModule %p :: train(in: %p, target: %p)\n", h_module, h_inputs, h_target);
    auto res = WRAP_EXCEPTIONS( mod->py_module.attr("ftn_pytorch_train")(*inputs, *target) );
    auto res_tuple = res.cast<std::tuple<bool, float>>();

    *is_completed = std::get<0>(res_tuple);
    *loss         = std::get<1>(res_tuple);
}

void torch_pymodule_save_cpp(void* h_module, char* filename) {
    auto mod = static_cast<PyModule*>(h_module);
    debug_print("PyModule %p :: save\n", h_module);
    WRAP_EXCEPTIONS( mod->py_module.attr("ftn_pytorch_save")(filename) );
}

void torch_pymodule_free_cpp(void* h_module) {
    auto mod = static_cast<PyModule*>(h_module);
    debug_print("Destroying PyModule %p\n", h_module);
    delete mod;
}

/*
 * Miscellaneous
 */
void torch_optimizer_create_sgd_cpp(void** handle, void* h_module, float lr) {
    auto mod = static_cast<torch::jit::Module*>(h_module);

    // https://github.com/pytorch/pytorch/issues/28478
    std::vector<torch::Tensor> parameters;
    for (const auto& param : mod->parameters()) {
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
void torch_tensor_from_array_cpp(void** handle,
        void* array, int arr_rank, FtnShapeType* ftn_shape, int elem_type, int elem_size) {
    
    // Free memory if we're reusing the tensor handle
    if (*handle) { torch_tensor_free_cpp(*handle); }
    auto tensor = new torch::Tensor;
    *tensor = tensor_from_array(array, arr_rank, ftn_shape, elem_type, elem_size);
    *handle = static_cast<void*>(tensor);
}

void torch_tensor_to_array_cpp(void* handle,
        void** host_ptr, void** device_ptr, int arr_rank, FtnShapeType* arr_shape, int elem_size) {
    auto tensor  = static_cast<torch::Tensor*>(handle);
    
    if (arr_rank != tensor->dim()) {
        throw std::runtime_error("Ranks mismatch passing Tensor to Fortran, expected "
            +std::to_string(tensor->dim())+" got "+std::to_string(arr_rank));
    }
    
    // Reverse the axes for Fortran
    size_t size_bytes = 1;
    for (int d=0; d<arr_rank; d++) {
        arr_shape[arr_rank - (d+1)] = tensor->size(d);
        size_bytes *= tensor->size(d);
    }
    size_bytes *= elem_size;

    auto ptr = static_cast<void*>(tensor->data_ptr());
    if (is_device_ptr(ptr)) {
        debug_print("Passing GPU tensor %p to Fortran array\n", handle);

        if (device_scratch_map.find(ptr) == device_scratch_map.end()) {
            // The tensor is not from the module output, create its own host scratch space
            std::unique_lock<std::mutex> lock(h2s_mutex);
            device_scratch_map[ptr] = std::make_shared<Scratch>(size_bytes);
        }
        // Const .at function is thread safe
        auto& scratch = const_cast<D2HScratchMap&>(device_scratch_map).at(ptr);
        if (scratch->size() < size_bytes) {
            std::unique_lock<std::mutex> lock(h2s_mutex);
            scratch->resize(size_bytes);
        }
        *host_ptr = scratch->data();
        *device_ptr = ptr;
    } else {
        debug_print("Passing CPU %p to Fortran array\n", handle);

        *host_ptr = ptr;
        *device_ptr = nullptr;
    }
}

void torch_tensor_backward(void* handle) {
    auto tensor  = static_cast<torch::Tensor*>(handle);
    WRAP_EXCEPTIONS( tensor->backward() );
}

void torch_tensor_free_cpp(void* handle) {
    auto tensor = static_cast<torch::Tensor*>(handle);
    void* ptr = tensor->data_ptr();
    
    if (device_scratch_map.find(ptr) != device_scratch_map.end()) {
        std::unique_lock<std::mutex> lock(h2s_mutex);
        device_scratch_map.erase(ptr);
    }
    delete tensor;
}

void* torch_helper_ptr_to_devptr_cpp(void* ptr) {
    // Well, that's that
    return ptr;
}

/*
 * IValue vector wrap
 */
void torch_tensor_wrap_create_cpp(void** handle) {
    // Free memory if we're loading a new module
    torch_tensor_wrap_free_cpp(*handle);

    auto iwrap = new IWrap;
    debug_print("Created tensor wrap %p\n", iwrap);
    *handle = static_cast<void*>(iwrap);
}

void torch_tensor_wrap_add_tensor_cpp(void* handle, void* h_tensor) {
    auto iwrap  = static_cast<IWrap*>(handle);
    auto tensor = static_cast<torch::Tensor*>(h_tensor);
    
    iwrap->push_back(*tensor);
}

void torch_tensor_wrap_add_array_cpp(void* handle, void* array, int arr_rank, FtnShapeType* arr_shape, int elem_type, int elem_size) {
    auto iwrap = static_cast<IWrap*>(handle);
    debug_print("Populating tensor wrap %p\n", iwrap);
    iwrap->push_back(tensor_from_array(array, arr_rank, arr_shape, elem_type, elem_size));
}

void torch_tensor_wrap_add_scalar_cpp(void* handle, void* value, int elem_type, int elem_size) {
    auto iwrap = static_cast<IWrap*>(handle);

    if        (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 4) {
        iwrap->emplace_back( *static_cast<float*>  (value) );
    } else if (elem_type == TORCH_FTN_TYPE_FP  && elem_size == 8) {
        iwrap->emplace_back( *static_cast<double*> (value) );
    } else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 4) {
        iwrap->emplace_back( *static_cast<int32_t*>(value) );
    } else if (elem_type == TORCH_FTN_TYPE_INT && elem_size == 8) {
        iwrap->emplace_back( *static_cast<int64_t*>(value) );
    } else {
        throw std::runtime_error("No known conversion from Fortran scalar with element size "+std::to_string(elem_size));
    }
}

void torch_tensor_wrap_clear_cpp(void* handle) {
    static_cast<IWrap*>(handle)->clear();
}

void torch_tensor_wrap_free_cpp(void* handle) {
    delete static_cast<IWrap*>(handle);
}
