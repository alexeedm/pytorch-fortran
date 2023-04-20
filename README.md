# Pytorch Fortran bindings

The goal of this code is to provide Fortran HPC codes with a simple way to use Pytorch deep learning framework.
We want Fortran developers to take advantage of rich and optimized Torch ecosystem from within their existing codes.
The code is very much work-in-progress right now and any feedback or bug reports are welcome.

## Features

*  Define the model conveniently in Python, save it and open in Fortran
*  Pass Fortran arrays into the model, run inference and get output as a native Fortran array
*  Train the model from inside Fortran and save it
*  Run the model on the CPU or the GPU with the data also coming from the CPU or GPU
*  Use OpenACC to achieve zero-copy data transfer for the GPU models
*  Focus on achieving negligible performance overhead

## Building

To assist with the build, we provide the Docker and [HPCCM](https://github.com/NVIDIA/hpc-container-maker) recipe for the container with all the necessary dependencies installed, see [container](container/)

You'll need to mount a folder with the cloned repository into the container, cd into this folder from the running container and execute `./make_nvhpc.sh`, `./make_gcc.sh` or `./make_intel.sh` depending on the compiler you want to use.

To enable the GPU support, you'll need the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) build. GNU compiler is ramping up its OpenACC implementation, and soon may also be supported.
Changing the compiler is possible by modifying `CMAKE_Fortran_COMPILER` cmake flag. Note that we are still working on testing different compilers, so issues are possible.

## Examples

[examples](examples/) folder contains three samples:
   * inference with the pre-trained ResNet;
   * end-to-end training on a simple NN predicting a polynomial;
   * training and inference through directly running Python (as opposed to pre-compiled Torch scripts), this example is work-in-progress.
The polynomial case will run on the GPU if both the bindings and the example are compiled with the OpenACC support.
Before running the examples, you'll need to execute `setup-model.py` scripts in the corresponding example folder that would define the model and store in on the disk.
With the models saved and ready, run the following:
```
cd /path/to/repository/
install/bin/resnet_forward ../examples/resnet_forward/traced_model.pt
install/bin/polynomial ../examples/polynomial/traced_model.pt ../examples/polynomial/your_new_trained_model.pt
install/bin/python_training  ../examples/python_training/model.py
```

## API

We are working on documenting the full API. Please refer to the examples for more details.
The bindings are provided through the following Fortran classes:

### Class `torch_tensor`
This class represents a light-weight Pytorch representation of a Fortran array. It does not own the data and only keeps the respective pointer.
Supported arrays of ranks up to 7 and datatypes `real32`, `real64`, `int32`, `int64`.
Members:
* `from_array(Fortran array or pointer :: array)` : create the tensor representation of a Fortran array.
* `to_array(pointer :: array)` : create a Fortran pointer from the tensor. This API should be used to convert the returning data of a Pytorch model to the Fortran array.

### Class `torch_tensor_wrap`
This class wraps a few tensors or scalars that can be passed as input into Pytorch models.
Arrays and scalars must be of types `real32`, `real64`, `int32` or `int64`.
Members:
* `add_scalar(scalar)` : add the scalar value into the wrapper.
* `add_tensor(torch_tensor :: tensor)` : add the tensor into the wrapper.
* `add_array(Fortran array or pointe :: array)` : create the tensor representation of a Fortran array and add it into the wrapper.


### Class `torch_module`
This class represents the traced Pytorch model, typically a result of `torch.jit.trace` or `torch.jit.script` call from your Python script. This class in **not thread-safe**. For multi-threaded inference either create a threaded Pytorch model, or use a `torch_module` instance per thread (the latter could be less efficient).
Members:
* `load( character(*) :: filename, integer :: flags)` : load the module from a file. Flag can be set to `module_use_device` to enable the GPU processing.
* `forward(torch_tensor_wrap :: inputs, torch_tensor :: output, integer :: flags)` : run the inference with Pytorch. The tensors and scalars from the `inputs` will be passed into Pytorch and the `output` will contain the result. `flags` is unused now
* `create_optimizer_sgd(real :: learning_rate)` : create an SGD optimizer to use in the following training 
* `train(torch_tensor_wrap :: inputs, torch_tensor :: target, real :: loss)` : perform a single training step where `target` is the target result and `loss` is the L2 squared loss returned by the optimizer
* `save(character(*) :: filename)` : save the trained model

### Class `torch_pymodule`
This class represents the Pytorch Python script and required the interpreter to be called. Only one `torch_pymodule` can be opened at a time due to the Python interpreter limitation. Overheads calling this class are higher than with `torch_module`, but contrary to the `torch_module%train` one can now train their Pytorch model with any optimizer, dropouts, etc. The intended usage of this class is to run online training with a complex pipeline that cannot be expressed as TorchScript.
Members:
* `load( character(*) :: filename)` : load the module from a Python script
* `forward(torch_tensor_wrap :: inputs, torch_tensor :: output)` : execute `ftn_pytorch_forward` function from the Python script. The function is expected to accept tensors and scalars and returns one tensor. The tensors and scalars from the `inputs` will be passed as argument and the `output` will contain the result.
* `train(torch_tensor_wrap :: inputs, torch_tensor :: target, real :: loss)` : execute `ftn_pytorch_train` function from the Python script. The function is expected to accept tensors and scalars (with the last argument required to be the target tensor) and returns a tuple of bool `is_completed` and float `loss`. `is_completed` is returned as a result of the `train` function, and `loss` is set accordingly to the Python output. `is_completed` is meant to signify that the training is completed due to any stopping criterion 
* `save(character(*) :: filename)` : save the trained model

## Changelog

### v0.4
* Fixed issues with `target` attribute and `resnet_forward` crash with GNU
* Updated `container.py` to work with more recent compilers
* PyTorch 1.12 may suffer from an issue described here: https://github.com/pytorch/pytorch/issues/68876 You should update to 1.13 if you see a compilation error similar to this one 
  ```
  fatal error: torch/csrc/generic/Storage.h: No such file or directory
  ```
* Now the commits will go directly to the `main` branch instead of `vX.X` and we will use tags instead

### v0.3
* Changed interface: `forward` and `train` routines now accept `torch_tensor_wrap` instead of just `torch_tensor`. This allows a user to add multiple inputs consisting of tensors of different size and scalar values
* Fixed possible small memory leaks due to tensor handles
* Fixed build targets in the scripts, they now properly build Release versions by default
* Added a short API help