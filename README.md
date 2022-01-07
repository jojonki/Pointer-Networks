# Pointer-Networks (unofficial)

Pointer Networks
Oriol Vinyals, Meire Fortunato, Navdeep Jaitly.  
https://arxiv.org/abs/1506.03134

**Pointer Networks** is a new neural architecture to learn the conditional probability of an output sequence with elements that are discrete tokens corresponding to positions in an input sequence.

In this repo, I put two examples of **Pointer Networks** models.


## The Sequence Model

In the sequence model, the length of output is the same as the length of input. I put a toy task of **sorting** task. The output is the sorted indices of the input. See the following example.
```
// An example
// Input :  [0, 3, 1, 2]
// Output:  [0, 2, 3, 1]

$ python sequence_train.py
epoch: 0, Loss: 0.99817
Acc: 0.57% (51/9000)
epoch: 2, Loss: 0.00077
Acc: 100.00% (9000/9000)
epoch: 4, Loss: 0.00032
Acc: 99.99% (8999/9000)
----Test result---
Acc: 100.00% (1000/1000)
```

## The Boundary Model

In the boundary model, the output is a tuple like `(start_index, end_index)`. I took up the following boundary toy task. See [this site](https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264).

>Let’s try out some code on a toy problem. Pointer networks are really most relevant for recurrency-sensitive data sequences, so we’ll create one. Suppose we assume our input data is a sequence of integers between 0 and 10 (with possible duplicates) of unknown length. Each sequence always begins with low integers (random values between 1 to 5), has a run of high integers (random values between 6 to 10), then turns low again to finish (1 to 5).
>For example, a sequence might be “4,1,2,3,1,1,6,9,10,8,6,3,1,1”, with the run of high integers in bold, surrounded by runs of low integers. We want to train a network that can point to these two change points — the beginning and end of the run of highs in the middle, regardless of the sequence length.

```
// An example
// Input : [4,1,2,3,1,1,6,9,10,8,6,3,1,1]
// Output: [6, 10]

$ python boundary_train.py
epoch: 0, Loss: 0.28288
acc
Acc: 98.79% (8891/9000)
epoch: 2, Loss: 0.00291
acc
Acc: 99.96% (8996/9000)
epoch: 4, Loss: 0.00091
acc
Acc: 100.00% (9000/9000)
----Test result---
Acc: 100.00% (1000/1000)
```

