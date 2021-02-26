# Pytorch Fortran bindings

The goal of this code is provide Fortran HPC codes with a simple way to use Pytorch deep learning framework.
We want Fortran developers to take advantage of rich and optimized Torch ecosystem from within their existing codes.
The code is very much work-in-progress right now and any feedback or bug reports are welcome.

## Features

*  Define the model convinently in Python, save it and open in Fortran
*  Pass Fortran arrays into the model, run inference and get output as a native Fortran array
*  Train the model from inside Fortran (limit support for now) and save it
*  Run the model on the CPU or the GPU with the data also coming from the CPU or GPU
*  Focus on achieving negligible performance overhead

## Building

To assist with the build, we provide the Docker and [HPCCM](https://github.com/NVIDIA/hpc-container-maker) recipe for the container with all the necessary dependancies installed, see [container](container/)

You'll need to mount a folder with the cloned repository into the container, cd into this folder from the running container and execute
```
./make_all.sh
```

By default, we build the code with [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) Fortran compiler without GPU support.
To enable the GPU support, change `OPENACC` parameter in `make_all.sh` to 1.
Changing the compiler is possible by modifying `CMAKE_Fortran_COMPILER` cmake flag. Note that we are still working on testing different compilers, so issues are possible.

## Examples

[examples](examples/) folder contains two samples: inference with pre-trained ResNet and end-to-end training on a simple NN predicting a polynomial.
Before running the examples, you'll need to execute `setup-model.py` scripts in the corresponding example folder that would define the model and store in to the disk.
With the saved models, run the following:
```
cd /path/to/repository/
install/bin/resnet_forward examples/resnet_forward/traced_model.pt
install/bin/polynomial     examples/polynomial/traced_model.pt     examples/polynomial/your_new_trained_model.pt
```

## API

We are working on documenting the API, for now please refer to the examples.