---
title: "Unravelling `tf.einsum`"
date: 2022-07-18T23:00:42-04:00
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

## Origin Story
Recently, I was trying to disect the original [DCNN Paper](https://arxiv.org/abs/1511.02136v6) which utilized a *diffusion* kernel to more readily make use of implicit graph-structure in common tasks such as node, edge and graph classification. However, an existing implementation I fonund had a curious piece of notation which led me down the rabbithole of Tensor calculus.

Coordinates are maps used to solve a given problem. A coordinate transform allows mapping from one frame of reference to another (converting from a map of your high school, to the location of your high school in reference to where it is in the city, compared to a country-wide map).

To say a number is a sclar means that the value does no change when transformed from one coordinate system to another (e.g. the distance between two points on a flat plain is irrespective of where true north is).

A vector is directional, and can be formed on the basis of the reference set of coordinates. For example, a vector between your home and the nearest fire-station can be broken down into a sum of north- and east-facing vectors.

## Tensors
A tensor describes the superset of transformations which include scalars and vectors:  
- $0$-tensors are constant functions, which we identify as scalars  
- $1$-tensors are linear functions, which we call vectors  
- $2$-tensors are bilinear functions, which we call matrices 

A **Tensor** describes any general transformation, independent of any basis function between sets of algebraic objects related to a vector space

---

Back to the paper, there was a particular function which claimed to do batch matrix multiplication:


```python
tf.einsum('ijk,kl->ijl', A, B)
```

where $A$ was the diffusion kernel and $B$ was a feature vector (and `tf` was `tensorflow`). So $A$ would have dimensions (batch_size, m, n) and $B$ would have dimensions (n, k), where:
- batch_size: number of nodes to process in a given batch (for model trainining)
- n: number of features 
- m: number of nodes 
- k: number of "hops"

Ignoring the technicalities of the paper and the actual definitions above, I wanted to know what the actual heck this strange `einsum` function was trying to do

## Einstein Summation

Enter *Einstein* summation: 
In "Einstein" summation, the repeated index defines what we sum by, the expression must have a repeated index, so:
$$
\sum_{i=1}^n = a_1x_1 + a_2x_2 + ... + a_nx_n \equiv a_ix_i
$$
is valid. But $a_{ij}x_k$ is not, whilst $a_{ij}x_j$ is:
$$
a_{ij}x_j \equiv a_{i1}x_1 + a_{i2}x_2 + ... + a_{in}x_n
$$

Double sums are handled as follows, for example summation on both $i$ and $j$:
$$
a_{ij}x_iy_j
$$

In the `einsum` function, the first argument `ijk,kl->ijl` signified summation on the $k^{th}$ dimension

---

Now that I understood what the notation meant, I wanted a better grasp on the actual mechanics behind the function. Using synthetic Tensors as follows:


```python
k = 2
batch_size, m, n = None, 4, 2
init = tf.random.uniform(shape=(m, n), minval=0, maxval=16, dtype=tf.int32)
A = tf.Variable(init)
A = tf.expand_dims(A, 0)
A
```




    <tf.Tensor: shape=(1, 4, 2), dtype=int32, numpy=
    array([[[14,  4],
            [ 4, 12],
            [ 9, 13],
            [ 0, 13]]], dtype=int32)>




```python
init = tf.random.uniform(shape=(n, k), minval=0, maxval=16, dtype=tf.int32)
B = tf.Variable(init)
B
```




    <tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
    array([[3, 9],
           [5, 1]], dtype=int32)>



### `tf.matmul`  
Here is where I used the two prior defined Tensors to basically see what would happen. It was also at this point I realised that TensorFlow 2 now included a function `matmul` which essentially did the same!


```python
C = tf.einsum('ijk,kl->ijl', A, B)
C
```




    <tf.Tensor: shape=(1, 4, 2), dtype=int32, numpy=
    array([[[ 62, 130],
            [ 72,  48],
            [ 92,  94],
            [ 65,  13]]], dtype=int32)>




```python
tf.matmul(A, B)
```




    <tf.Tensor: shape=(1, 4, 2), dtype=int32, numpy=
    array([[[ 62, 130],
            [ 72,  48],
            [ 92,  94],
            [ 65,  13]]], dtype=int32)>



## Minimum-Viable Example

Okay, now simplifying even further; firstly by creating a rank-2 tensor (i.e. a matrix) using numpy and then finding the matrix product


```python
import numpy as np

A = np.matrix('''
    1 4;
    2 3
''')

B = np.matrix('''
    5 7;
    6 8
''')

C = A @ B
C
```




    matrix([[29, 39],
            [28, 38]])



Every element in $C$, $C_{ik}$ is:
$$
C_{ik} = \sum_jA_{ij}B_{jk}
$$

$C_{01} = 39$ so

$$
C_{01} = \sum_j A_{0j} B_{j1} (1\times 7)_ {j=0} + (4\times 8)_{j=1} 
$$

Followed by converting the above matrices to TensorFlow objects and repeating the operation to somehow test that I grasped the notation:


```python
A = tf.convert_to_tensor(A)
B = tf.convert_to_tensor(B)

A, B
```




    (<tf.Tensor: shape=(2, 2), dtype=int64, numpy=
     array([[1, 4],
            [2, 3]])>,
     <tf.Tensor: shape=(2, 2), dtype=int64, numpy=
     array([[5, 7],
            [6, 8]])>)



It worked! The output of `einsum` below is consistent with `matmul` above


```python
# equivalent to A @ B or tf.matmul(A, B)
tf.einsum('ij,jk->ik', A, B)
```




    <tf.Tensor: shape=(2, 2), dtype=int64, numpy=
    array([[29, 39],
           [28, 38]])>



## Slightly-Less Minimum Example  
Now on to a slightly more complex example, I created a rank-2 Tensor and a rank-1 Tensor for multiplication against


```python
# applying to batch case
A = tf.Variable([
    [[1,2],
    [3,4]],
    [[3, 5], 
    [2, 9]]
])

B = tf.Variable(
    [[2], [1]]
)
A.shape, B.shape
```




    (TensorShape([2, 2, 2]), TensorShape([2, 1]))




```python
tf.matmul(A, B)
```




    <tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
    array([[[ 4],
            [10]],
    
           [[11],
            [13]]], dtype=int32)>



For the $ijl^{th}$ element in $C$, sum across the $k^{th}$ dimension in A and B

```
output[i,j,l] = sum_k A[i,j,k] * B[k, l]
```


```python
# for the ijl-th element in C, 
C = tf.einsum('ijk,kl->ijl', A, B)
C
```




    <tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
    array([[[ 4],
            [10]],
    
           [[11],
            [13]]], dtype=int32)>



and success! I think I have a fair grasp on how `einsum` and Einstein summation works, and how/why it can be sometimes simpler just to use the built-in `matmul` function, but also where batch dimensions may mess with the built-in functions and we would want to define it in finer detail
