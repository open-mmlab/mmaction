# Known Issues

1. Error on RTX cards with CUDA 10.0

Description:

THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument #15797

Solution: Install pytorch wheel with cuda 10.0 via

```bash
pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
```
