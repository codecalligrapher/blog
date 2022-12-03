---
title: "Prefetching Memory in CUDA"
date: 2022-08-06:00:00-00:00
math: true
---
{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ]
        });
    });
</script>

{{ end }}
{{</ math.inline >}}


## Threads, Blocks and Grids
A thread is a single "process" on GPU. Any given GPU kernel can use *blocks* of threads, grouped into a *grid* of blocks. A kernel is executed as a grid of blocks of threads. Each block is run by a single **Streaming Multiprocessor** (SM) and in most usual, single-node cases can't be migrated to other SMs. One SM may execute several CUDA blocks concurrently.

## Paging
Paging is a memory-management technique which allows a process's physical address space to be non-contiguous. Paging prevents two main problems:
- External memory fragmentation and
- the associate need for contraction

Paging is usually accomplished by breaking physical memory into fixed-sized blocks called *frames*, and breaking logical memory into blocks of the same size called **pages**. When a process is run, its pages are loaded from secondary memory (file system or backing store) into the memory page. The most interesting aspect of this is that it allows a process to have a logical 64-bit address space, although the system has less than $2^{64}$ bytes of physical memory.

## Page Faults
A page *fault* occurs when a process requests tries to access a page that wasn't brought into memory (whether it be device or host). The paging hardward will notice that an invalid bit is set, and goes on to execute a straightforward procedure to handle this:
1. Check an internal table for the process to determine whether the reference itself was valid
2. If the reference was invalid, terminate the process, else we page in the data
3. Find a free frame in physical memory (a frame is a fixed-size collection of blocks that is indexed in physical memory)
4. Schedule a secondary storage operation to load the needed page into the newly allocated frame 
5. When the read is complete, the internal table kept with the process and the page table is modified to indicate that the page is now in memory
6. The instruction is restarted

There are two main advantages of GPU page-faulting:
- the CUDA system doesn't need to sync all managed memory allocations to GPU before each kernel since faulting causes automatic migration
- page mapped to GPU addess space

The above process takes non-zero time, and a series of page-faults can result in significant memory overhead for any CUDA kernel. 

## Unified Memory
The CUDA programming model streamlines kernel development by implementing Unified Memory (UM) access, eliminating the need for explicit data movement via `cudaMemcp*()`. This is since the UM model enables all processes to see a coherent memory image with a common address address, where explicit calls to memory movement is handled by CUDA.

UM is for writing streamlined code, and does not necessarily result in a speed increase.

More significantly, non-explicit allocation of memory resources may result in a large amout of page-faulting procedures which go on during the kernel execution. However, non-explicit allocation of memory resources may result in a large amout of page-faulting procedures which go on during the kernel execution.

According to the [CUDA Performance tuning guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-performance-tuning):
- Faults should be avoided: fault handling takes a while since it may include TLB invalidates, data migrations and page table updates
- Data should be local to the access processor to minimize memory access latency and maximize bandwidth
- Overhead of migration may exceed the benefits of locality if data is constantly migrated

Hence we can _not_ use UM, since the UM drives can't detect common access patterns and optimize around it. WHen access patterns are non-obvious, it needs some guidance

## Prefetching
Data prefetching is moving data to a processor's main memory and creating the mapping the page tables BEFORE data processing begins with the aim to avoid faults and establish locality.

`cudaMemPrefetchhAsync` prefetches memory to the specified destination device

```cpp
    cudaError_t cudaMemPrefetchAsync(
        const void *devPtr,     // memory region
        size_t count,           // number of bytes
        inst dstDevice,         // device ID
        cudaStream_t stream
    );
```

## Profiling Prefetches
The following shows profile statistics using `nsys` of two kernel which do exactly the same operation (squaring a series of float values on GPU). 

### With prefetching

```

CUDA Memory Operation Statistics (by time):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)              Operation
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ---------------------------------
    100.0       10,592,835     64  165,513.0  165,313.0   165,249   178,465      1,645.6  [CUDA Unified Memory memcpy DtoH]


CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation
 ----------  -----  --------  --------  --------  --------  -----------  ---------------------------------
    134.218     64     2.097     2.097     2.097     2.097        0.000  [CUDA Unified Memory memcpy DtoH]

```

### Without prefetching
```
CUDA Memory Operation Statistics (by time):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     67.0       22,186,252  1,536  14,444.2   3,935.0     1,278   120,226     23,567.2  [CUDA Unified Memory memcpy HtoD]
     33.0       10,937,857    768  14,242.0   3,359.0       895   102,529     23,680.5  [CUDA Unified Memory memcpy DtoH]

[9/9] Executing 'gpumemsizesum' stats report

CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation
 ----------  -----  --------  --------  --------  --------  -----------  ---------------------------------
    268.435  1,536     0.175     0.033     0.004     1.044        0.301  [CUDA Unified Memory memcpy HtoD]
    134.218    768     0.175     0.033     0.004     1.044        0.301  [CUDA Unified Memory memcpy DtoH]
```
<!-- `void* malloc (size_t size);` allocates a block of `size` memory, returning a pointer to the beginning of this block. `cudaMalloc()` does the same for linear memory, typically copy from host to device using `cudaMemcpy()`.
 -->

Of note are two things: the time taken for memory operations without prefetching is nearly triple that of the prefetch, and the size of memory operations is much smaller. The only difference between both kernels is a call to `cudaMemPrefetchAsync` for any data structure that was to be copied to device (GPU).

```cpp
int deviceId;

const int N = 2<<24;
size_t size = N * sizeof(float);
// Declare a float pointer
float *a;
// Set up unified memory
cudaMallocManaged(&a, size);

// Up till this point is usual, the only difference is this call to the prefetch
cudaMemPrefetchAsync(a, size, deviceId);

// Go on to specify the number of threads, blocks etc.
```
