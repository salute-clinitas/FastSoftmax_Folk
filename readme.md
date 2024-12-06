# Fast softmax

When doing some profiling of triton and cuda kernels. I wanted to do a comparison of softmax speeds
since softmax is one of the kernels that triton gives as an example in their documentation.

As it turned out writing a fast softmax kernel was not as straightforward as I thought it would be.

This repo contains all of the kernels that I've tried as well as a detailed explanation on how they work and 
how I got to them.

## Background

Softmax is a function that takes in a vector of real numbers and returns a probability distribution

The usual way of calculating it is by replacing each element with an exponent, raised to the power of said element
divided by the sum of exponents of all elements in our vector

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$
