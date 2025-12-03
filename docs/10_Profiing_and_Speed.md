## On Topics of Code Performance

> *I feel the need—the need* for *speed*!

*My code just runs fine, why should I care about speed?* By the end of the day, most computational research is limited by the size of compute. That is how many inputs you can process, how many setups you can afford, the amount of predictions you can make, and the size of models you can train and evaluate. Most of the time, the larger the scale, the higher chance to get more impressive results. Thus, higher chance to have higher impact publications. At the scale of performing the same type of computation thousands to millions of times, any efficiency improvement will greatly reduce total computational time. This is an aspect often overlooked by both computing researchers and domain experts. Here is a list of reasons why this is important and we as researchers should care:

* Faster code conserves compute resources, enabling more extensive experiments within the same infrastructure, so you can increase the sample size, try more parameter settings, or use more complex models without extra hardware.
* Optimizing code speed can dramatically increase the rate of iteration. Faster experiments mean you can test, tweak, and refine models or setups more frequently, leading to quicker insights and higher-quality results.
* While code that runs on a small dataset may seem fine, this doesn't guarantee it will perform similarly on larger datasets or models. Optimized code scales better, avoiding bottlenecks that might arise at larger scales.
* Many computational resources are charged by usage time. Speeding up code reduces runtime, which can cut costs and free up funding or resources for other research needs.
* Speedier, efficient code can make it easier for others to replicate your work without needing access to high-end resources, thus fostering collaboration and openness.

### Things About Python Performance

> Life is short, I use Python.

Python is convenient. It is easy to use, but it is very slow. Python's nature as an interpreted language affects its speed compared to compiled languages like C/C++ and Java. Without optimized libraries, Python's execution is typically slower because each line is translated to machine code in real time, rather than ahead of time as with compiled languages. In many projects, leveraging the right libraries like `numpy`, `pandas`, or `scipy` is essential because these libraries wrap optimized C/C++ code. By offloading performance-critical tasks to these libraries, you can achieve significant speed-ups.

Here is a speed comparison between various programming languages (<https://github.com/niklas-heer/speed-comparison?tab=readme-ov-file>), **showing that Python is about 2x slower than Java and up to 10x slower than C/C++ in benchmarks that do not use these optimized libraries**. This difference highlights the importance of choosing the right tools in Python, as using libraries that are optimized with compiled C/C++ code can narrow this performance gap. Using such libraries effectively turns Python into a high-level "glue language" that allows you to benefit from both ease of use and execution speed. This approach allows you to handle various project demands efficiently without sacrificing performance where it matters most.

In addition to raw Python speed, Python's **Global Interpreter Lock (GIL)** could also have a huge negative effect on code performance. GIL is a lock (a mutex in CS terms) that allows only one thread to take control of the Python interpreter. This is a solution to ensure memory safety. However, GIL can create a bottleneck in CPU-bound and multi-threaded code. Use multi-processing (<https://docs.python.org/3/library/multiprocessing.html>) instead of multi-threading in Python implementation to bypass this limitation. However, keep in mind that multi-processing needs more resources and is harder to share data between each process.

#### Other Things About Code Performance

> There is no silver bullet.

First and foremost, there is no silver bullet to solve all performance issues. Understand tasks and find bottlenecks first, then find solutions for specific tasks.

#### Algorithmic Efficiency

#### Hardware Efficiency

It is very important to keep in mind that your program will always be bounded or bottlenecked by some hardware limitation. Fully using all the hardware does not necessarily mean your method is optimal.

1. **CPU-Bound:**

   - **Definition:** A process is said to be CPU-bound when its performance is limited primarily by the speed of the CPU. The code spends most of its time processing data rather than waiting for other resources, like I/O or memory.
   - **Symptoms:** High CPU usage and low idle times, with minimal usage of other resources.
   - **Optimization Focus:**
     - Parallelize tasks to utilize multiple cores or processors (e.g., using multi-threading or multi-processing).
     - Optimize algorithms and data structures to reduce the number of operations.
     - Use optimized libraries (e.g., NumPy, SciPy for numerical computations).
     - Profile to identify hotspots and optimize those specific sections of the code.
2. **GPU-Bound:**

   - **Definition:** A process is GPU-bound when its performance is limited by the GPU's ability to process data, especially in tasks like machine learning, deep learning, or data-intensive computations.
   - **Symptoms:** High GPU usage and long waiting times for GPU tasks to complete. You may see a heavy workload on the GPU but the CPU is underutilized.
   - **Optimization Focus:**
     - Optimize GPU utilization by maximizing parallelism and using batch processing for tasks like training neural networks.
     - Ensure that the GPU is being used effectively by offloading computations that can benefit from parallel processing.
     - Use specialized libraries like TensorFlow, PyTorch, or CuPy, which leverage GPU acceleration.
     - Profile GPU usage (e.g., with `nvidia-smi` or libraries like `GPUtil`) to check if the GPU is bottlenecking any part of the computation.
3. **Memory-Bound:**

   - **Definition:** A process is memory-bound when its performance is limited by the speed of memory, including RAM, cache, or GPU memory. The system is spending too much time waiting for data to be read from or written to memory.
   - **Symptoms:** High memory usage, frequent paging (moving data between RAM and disk), and slow performance due to memory constraints. This often happens when large datasets or high-resolution data are involved.
   - **Optimization Focus:**
     - Optimize data structures to minimize memory usage.
     - Use memory-mapped files or streams to process large datasets without loading everything into memory.
     - Offload computations to external storage or distributed systems (e.g., using Dask or Hadoop for parallelized processing).
     - Use algorithms that process data in smaller chunks (batch processing or windowing techniques).
     - Profile memory usage (e.g., using Python's `memory_profiler` or `psutil` libraries) to identify areas where memory use can be reduced.

Profiling Tools:
To assess whether your code is CPU-bound, GPU-bound, or memory-bound, you can use various profiling tools:

- **CPU Profiling:** Use `cProfile`, `line_profiler`, or `Py-Spy` to profile CPU performance.
- **GPU Profiling:** For GPU-related tasks, use `nvidia-smi`, `nvprof`, or TensorFlow/PyTorch profiling tools.
- **Memory Profiling:** Use `memory_profiler`, `objgraph`, or `psutil` to analyze memory consumption.

### Extra Things for Machine Learning Projects

1. **Hardware Usage**: Fast hardware is only useful if programs can fully use it. Using a fancy GPU does not directly translate to performance boosts. Always check your GPU and CPU usage during training and evaluation if needed.
2. **Dataloader**: One of the more common bottlenecks in GPU-accelerated machine learning tasks is in fact the dataloader, the part of the program that reads data from hard drives and loads them into system memory and GPU memory. A slow dataloader often results in lower GPU usage, since the GPU is waiting for data to be transferred from hard drive/system RAM to VRAM.
3. **Know Your Hardware**: Understanding what the best hardware for your task is always important. Not all GPUs and CPUs are equal. In terms of CPUs, keep in mind some CPUs have faster AVX2 and/or AVX512 support. In terms of GPUs, some later models will support TF32 and BF16. If your program can take advantage of them, it will lead to 2x to 100x speed increases on the correct hardware.
4. **Evaluation Code**: While not very common, evaluation step code can also be a bottleneck. Keep in mind that evaluation is performed after each training epoch and would take a significant amount of time. Thus, pick only necessary metrics to monitor during the training process, while performing full-scale evaluation after training is finished.
