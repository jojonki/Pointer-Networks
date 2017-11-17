# Pointer-Networks


## The Sequence Model

## The Boundary Model

The task is a toy task of [this site](https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264).

>Let’s try out some code on a toy problem. Pointer networks are really most relevant for recurrency-sensitive data sequences, so we’ll create one. Suppose we assume our input data is a sequence of integers between 0 and 10 (with possible duplicates) of unknown length. Each sequence always begins with low integers (random values between 1 to 5), has a run of high integers (random values between 6 to 10), then turns low again to finish (1 to 5).
>For example, a sequence might be “4,1,2,3,1,1,6,9,10,8,6,3,1,1”, with the run of high integers in bold, surrounded by runs of low integers. We want to train a network that can point to these two change points — the beginning and end of the run of highs in the middle, regardless of the sequence length.

```
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

