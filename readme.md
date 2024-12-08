# Fast softmax

When doing some profiling of triton and cuda kernels. I wanted to do a comparison of softmax speeds
since softmax is one of the kernels that triton gives as an example in their documentation.

As it turned out writing a fast softmax kernel was not as straightforward as I thought it would be.

This repo contains all of the kernels that I've tried as well as a detailed explanation on how they work and 
how I got to them.

It's also available in an [animated video form](https://youtu.be/IpHjDoW4ffw)

## Background

Softmax is a function that takes in a vector of real numbers and returns a probability distribution

The usual way of calculating it is by replacing each element with an exponent, raised to the power of said element
divided by the sum of exponents of all elements in our vector


$$\Large\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

Although there is one problem with this approach, since it uses an exponential function, that grows - well exponentially
if our input vector will contain multiplepositive values, it can overflow as we will add a lot of big numbers together
in our divisor

We can mitigate this by subtracting the maximum of our vector from the exponent. 
That way - the powers will always be negative, and our values will remain in range of 0 to 1

$$\Large\text{softmax}(x_i) = \frac{e^{x_i - max(x)}}{\sum_{j=1}^{K} e^{x_j - max(x)}}$$


## Theoretical Performance Limit

To have an estimate for how fast our kernel can theoretically be, we need to calculate how many floating-point operations we are calculating and how much memory we are accessing.

For bytes loaded, it's quite simple. We load the whole vector once and save it once, so we get 2 times our vector size memory accesses of floating-point values that are 4 bytes each.

For the FLOPs, we have to split our function into suboperations:

$$m = max(x)$$

$$x = x - m$$

$$exp = e^x$$

$$s = sum(exp)$$

$$out = \\frac{exp_i}{s}$$

This leaves us with 5 FLOPs per 8 bytes loaded.

With this info, we can calculate a theoretical maximum of performance that we can get out of this kernel. With 5 floating-point operations per 8 loaded bytes, we are bottlenecked by memory bandwidth, which is 1 TB/s on my GPU. That gives us 

$$\\frac{5}{8}*1\\frac{TB}{s} = 625\\,GFLOPs$$

In practice the threashold is higher as long as our value fit in the cache since CUDA uses something called [write back cache](https://en.wikipedia.org/wiki/Cache_(computing)#Writing_policies)

## Initial Kernel

When I did and MNIST solver I wrote this naive softmax kernel

```C
__global__ void softmax(int w, int h, float* input, float* output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = input[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = max(maxval, input[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += exp(input[row*w + i] - maxval);
    }
    output[row*w + col] = exp(input[row*w + col]-maxval)/(divisor);
  }
}
```

There is one major bottleneck in this kernel: each thread in the row recalculates the maximum value and the divisor.
While this wasn't really a big problem in the MNIST solver, where the height of the input was much bigger than the width, in recent trends, the amount of classes that we are predicting is much bigger than the batch size we are feeding the model(think llama3 with a 128000 vocab size)

For just 1024 elements, it achieves a magnificent **8.9 GFLOPS**.

## Optimization Strategies

### 1. Distributing work between threads in a block

The key to making a fast softmax algorithm is understanding how to perform a fast reduction algorithm. A reduction algorithm is a type of algorithm where we need to perform an operation on every input element where the input to the operation is a result of the previous input.

In order for this to parallelize nicely, the operator needs to be associative. That means that no matter the order of the operations, the result will be the same. This gives us a wonderful property: we don't need to calculate sequentially, but we can do it in a tree-like manner.

![image](https://github.com/user-attachments/assets/02231162-8fa1-45a1-a663-f2259bcd0bc7)

In the case of our softmax, we perform two associative reductions:

1. One for finding the maximum.
2. The second for summing all elements to calculate our divisor.

To distribute work between threads, we first divide the input equally among threads. Each thread independently performs a reduction on its subset of the data. Then, we transmit the reduced values between threads and finalize the reduction.

![image](https://github.com/user-attachments/assets/5913e224-40d3-4195-b585-18338ae0234b)

This is the code we can use for our faster reduction:

```C
__shared__ float reduction[BLOCK_DIM_Y]; 
float maxval = FLOAT_MIN;
for (int i = ty*BLOCK_DIM_Y; i<min(w, (ty+1)*BLOCK_DIM_Y); i++)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}

reduction[ty] = maxval;
for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
{
  __syncthreads();
  if (ty < stride)
  {
    reduction[ty] = fmaxf(reduction[ty], reduction[ty+stride]);
  }
}

__syncthreads();
maxval = reduction[0];
```


### 2. Memory Coalescing

We can see that we are accessing our data with a stride of `BLOCK_SIZE`. If you watched the [video on DRAM and memory coalescing](https://www.youtube.com/watch?v=huhg3V4ZRW0), you know that this is a very bad access pattern. To improve, we need to change our access pattern so that each thread accesses values that are adjacent in memory.

And the change is very simple, we just change our initial for loop from 
```C
for (int i = ty*BLOCK_DIM_Y; i<min(w, (ty+1)*BLOCK_DIM_Y); i++)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}
```

to

```C
for (int i = ty; i<w; i+=BLOCK_DIM_Y)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}
```


### 3. Register-Level Reductions

Threads in a processing block have a shared register file, so there is nothing stopping us from using this fact to share data between the threads faster. We can use warp intrinsics like 

```C
#define MASK 0xffffffff
__shfl_xor_sync(MASK, variable, offset, warp_size)
```

It is used for retrieving the value of our variable from another thread within the same warp, here is how it works in pseudocode:

```C
laneId = threadId%32
if laneid & MASK and laneId < warp_size:
    target_lane_id = laneId ^ offset
    return get_variable_from_other(variable, target_lane_id)
```

Right now instead of just doing a reduction in shared memory we start in registers, then move all of the results to a single warp using shared memory. Then we can do another reduction in registers to get a final value:

![image](https://github.com/user-attachments/assets/c0889280-28a5-4501-a1c8-ca66bd91a3b5)


```C
#define MASK 0xffffffff
float maxval = FLOAT_MIN;
for (int i = ty; i<w; i+=BLOCK_DIM_Y)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}
for (int i = 16; i>0; i/=2)
{
  maxval = fmaxf(maxval, __shfl_xor_sync(MASK, maxval, i, 32));
}

if (ty%32 == 0)
{
  reduction[warp_id] = maxval;
}
__syncthreads();
if (warp_id == 0)
{
    maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;

    for (int i = 16; i>0; i/=2)
    {
      maxval = fmaxf(maxval, __shfl_xor_sync(MASK, maxval, i, 32));
    }
}
```
### 4. Float4 Loading

Our next step is to utilize loading in `float4`. This holds multiple very low-level benefits: we issue one instruction for four memory loads, reducing the amount of instructions issued and the number of index calculations for memory access.

This is how we have to change all memory accesses, instead of doing:
```C
float maxval = FLOAT_MIN;
for (int i = ty; i<w; i+=BLOCK_DIM_Y)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}
```

We have to do 

```C
float maxval = FLOAT_MIN;
for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
{
  float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
  maxval = fmaxf(maxval, val.x);
  maxval = fmaxf(maxval, val.y);
  maxval = fmaxf(maxval, val.z);
  maxval = fmaxf(maxval, val.w);
}
```

### 5. Loop unrolling

The next step is to unroll our loops, the compiler usually does that for you, but you can control this behaviour by using a pragma directive
that takes in the amout of unrolls we want the compiler to do

```C
#pragma unroll UNROLL_FACTOR
```

With this we can do a search for the best parameters of `UNROLL_FACTOR` and `BLOCK_DIM`


### 6. **Online Normalization**

The problem with our existing kernel is that we load our input twice: once when finding the maximum value and another time when calculating the divisor. Problem is that the divisor calculation requires a maxval calculation.
The solution proposed by NVIDIA in their paper, [Online Normalizer Calculation for Softmax](https://arxiv.org/pdf/1805.02867) 
addresses this issue by calculating the divisor in the same loop as finding the maximum.

What we need to ask is to get to the solution is how does the previous value of the divisor change when we find a new maxval. Let's say we calculated the first estimate


The key insight is that the contribution of an element \(x_i\) to the divisor changes as new maximum values are discovered. By maintaining a running divisor and adjusting it dynamically, we eliminate the need for a second loop, reducing memory accesses.
 
$$d_1 = e^{x_1- max_1}$$

Then in the next iteration we find a new maxval, now our previous estimate of the divisor changes to

$$$d_2 = e^{x_1- max_2}$$

We can wite it out like this and simplify to see how much we need to fix our previous estimate by 

$$\begin{align*}
d_2 &= d_1 \frac{d_2}{d_1} \\
&= d_1 \frac{e^{x_1 - \text{max}_2}}{e^{x_1 - \text{max}_1}} \\
&= d_1 e^{(x_1 - \text{max}_2) - (x_1 - \text{max}_1)} \\
&= d_1 e^{\text{max}_1 - \text{max}_2}
\end{align*}$$

Tada, we just found a correction value that only depends on our previous and new maxval

Now we can incorporate it into our code, our initial reduction in one thread changes to 

```C
for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
{
    float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
    maxval = fmaxf(maxval, val.x);
    maxval = fmaxf(maxval, val.y);
    maxval = fmaxf(maxval, val.z);
    maxval = fmaxf(maxval, val.w);
    if (maxval > old_maxval)
    {
      divisor *= __expf(old_maxval - maxval);
      old_maxval = maxval;
    }
    divisor += __expf(val.x - maxval);
    divisor += __expf(val.y - maxval);
    divisor += __expf(val.z - maxval);
    divisor += __expf(val.w - maxval);
}
```

Theoretically we don't need to do the if statement, but exponents are expensive so it's best to avoid doing them if we don't need to

We also need to change how we do the reduction on the warp level 

```C
float incoming_divisor;
float incoming_maxval;
for (int i = 16; i>0; i/=2)
{
  incoming_maxval = __shfl_xor_sync(0xffffffff, maxval, i, 32);
  incoming_divisor = __shfl_xor_sync(0xffffffff, divisor, i, 32);
  if (incoming_maxval > maxval)
  {
    divisor *= __expf(maxval - incoming_maxval);
    maxval = incoming_maxval;
  }
  else 
  {
    incoming_divisor *= __expf(incoming_maxval - maxval);
  }
  divisor += incoming_divisor;
}

if (ty%32 == 0)
{
  reduction_max[warp_id] = maxval;
  reduction_div[warp_id] = divisor;
}
```

The important change here is that we need to determine which value to fix, if the incoming maximum is bigger than our maximum, we need to fix our divisor, and else we need to fix the incoming divisor

## Final Results

With all the optimizations applied, here's the performance comparison:

![image](https://github.com/user-attachments/assets/50f882de-ffca-4bac-9e44-d4780e4acab7)


## Support and Shoutouts

I'm hosting a [Buy Me a Coffee](https://buymeacoffee.com/simonoz) for those who want to support my work. A shoutout to:

- Alex
- Udit Ransaria
- Stuart McVicar
- Ilgwon Ha

and three anonymous donors who supported me so far.

## References

https://arxiv.org/pdf/1805.02867

https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/dev/cuda/softmax_forward.cu#L4

https://siboehm.com/articles/22/CUDA-MMM

https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F

Programming Masively Parallel Processors book
