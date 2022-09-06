# Pytorch Fortran bindings

The goal of this code is provide Fortran HPC codes with a simple way to use Pytorch deep learning framework.
We want Fortran developers to take advantage of rich and optimized Torch ecosystem from within their existing codes.
The code is very much work-in-progress right now and any feedback or bug reports are welcome.

## Features

*  Define the model convinently in Python, save it and open in Fortran
*  Pass Fortran arrays into the model, run inference and get output as a native Fortran array
*  Train the model from inside Fortran (limit support for now) and save it
*  Run the model on the CPU or the GPU with the data also coming from the CPU or GPU
*  Use OpenACC to achieve zero-copy data transfer for the GPU models
*  Focus on achieving negligible performance overhead

## Building

To assist with the build, we provide the Docker and [HPCCM](https://github.com/NVIDIA/hpc-container-maker) recipe for the container with all the necessary dependancies installed, see [container](container/)

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

We are working on documenting the API, for now please refer to the examples.