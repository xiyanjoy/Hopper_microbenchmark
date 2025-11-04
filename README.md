# How to test
```
nvcc tma_bandwidth.cu -maxrregcount=255 -lineinfo --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills -arch=sm_90a -I/path/to/cutlass/include -o output && ./output
nvcc ldg_bandwidth.cu -maxrregcount=255 -lineinfo --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills -arch=sm_90a -o output && ./output
```