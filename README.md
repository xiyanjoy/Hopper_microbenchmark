# How to test
```
nvcc tma_bandwidth.cu -maxrregcount=255 --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills -arch=sm_90a -I/path/to/cutlass/include -o output && ./output
```