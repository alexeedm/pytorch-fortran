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

#pragma once

extern "C" {
    void torch_module_load_cpp(void** h_module, const char* file_name, int flags);
    void torch_module_forward_cpp(void* h_module, void* h_input, void** h_output, int flags);
    void torch_module_train_cpp(void* h_module, void* h_input, void* h_target, void* h_optimizer, float* loss);
    void torch_optimizer_create_sgd_cpp(void** handle, void* h_module, float lr);
    void torch_module_save_cpp(void* h_module, char* filename);
    void torch_module_free_cpp(void* h_module);

    void torch_tensor_from_array_float_cpp(void** handle,
        void*  array, int arr_rank, int* arr_shape, int elem_type, int elem_size);
    void torch_tensor_to_array_cpp  (void* handle,
        void** host_ptr, void** device_ptr, int arr_rank, int* arr_shape, int elem_size);
    void torch_tensor_free_cpp(void* handle, void* host_ptr, void* device_ptr);
    void* torch_helper_ptr_to_devptr_cpp(void* ptr);
}