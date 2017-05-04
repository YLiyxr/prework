# Prework assignment for Applied AI (iX 2017)

## Part I: Reading response questions

Create a plain text file called `reading.txt` with response to the following
questions about the readings from the assignment sheet. All answers should be
three sentences or less (or a small code chunk), and none require
equations (maybe some multiplication).

#### Python

*Nothing required in `readings.txt` for these.*

1. Write PEP 8-compliant code for Part II.
2. Use (at least one) dictionary or list comprehension in Part II.

#### NumPy

3. Write a vectorized version of the code snippet below.

```python
import numpy as np

X = []
for _ in xrange(10):
    row = np.zeros(5)
    if np.random.rand() > 0.5:
        for i in xrange(len(row)):
            row[i] = 1
    X.append(row)
X = np.vstack(X)
```

#### Basics of machine learning

[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

4. Gradient descent - more specifically, stochastic gradient descent - is 
the primary algorithm used to train machine learning models.
At the end of Section I.1, the SGD algorithm listing says to loop *m* times.
But how do we choose *m*? One idea is to keep looping until none of the model
parameters change by more than, say, 0.000001. But this isn't always a good
idea. Look up **overfitting** on the world wide web. This is an essential
concept in machine learning. Give one way to prevent overfitting involving the
number of training epochs *m*.

5. Another way to prevent overfitting is a technique called *regularization*.
Look up **ridge regression** on the world wide web. Give a description, in
words, of how the linear regression loss function *J* is altered when we use
a ridge penalty. Give an intuitive explanation as to how the penalty affects the
model parameters.

6. Part II of the notes discusses classification with logistic regression.
I just had a great idea! Instead of using this weird logistic loss function,
let's just use 0-1 loss (given below) to train our model. Afterall, that's the exact
metric I want my classifier to be optimized for. Spoiler: this isn't a great idea. Why not?

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\ell(\theta)&space;=&space;\left\lbrace\begin{array}{ll}&space;0&space;&&space;\text{sign}(\theta^Tx^{(i)})&space;=&space;y^{(i)}&space;\\&space;1&space;&&space;\text{o.w.}&space;\end{array}&space;\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\ell(\theta)&space;=&space;\left\lbrace\begin{array}{ll}&space;0&space;&&space;\text{sign}(\theta^Tx^{(i)})&space;=&space;y^{(i)}&space;\\&space;1&space;&&space;\text{o.w.}&space;\end{array}&space;\right." title="\ell(\theta) = \left\lbrace\begin{array}{ll} 0 & \text{sign}(\theta^Tx^{(i)}) = y^{(i)} \\ 1 & \text{o.w.} \end{array} \right." /></a>

(Here, we'll assume our labels are -1's and +1's, instead of 0's and 1's).

#### Linear algebra for machine learning

[CS229 linear algebra notes](http://cs229.stanford.edu/section/cs229-linalg.pdf)

Say I have a matrix *A* which is *n* x *p* and a matrix *B* which is *p* x *d*.

7. Suppose I also have a *d*-dimensional vector *y*, and I want to compute the
product *z = ABy*. Under what circumstances should I compute *AB* first?
When should I compute *By* first? Big-*O* notation might be useful.

8. Now suppose I have *k* different *d*-dimensional vectors
*y*<sub>1</sub>,..., *y*<sub>k</sub> and I want to compute all of the products
*z*<sub>i</sub> = *ABy*<sub>i</sub> for *i*=1,...,*k*.
Under what circumstances should I precompute *AB* before computing the
products *z*<sub>i</sub>?

#### TensorFlow

9. The **softmax** function is given in equation (8) of Section III.9.3 in the
[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
It's an important function in machine learning, so TensorFlow implements it in
`tf.nn.softmax(...)`. But unfortunately someone pushed a bad change to 
TensorFlow's C implementation of the softmax function, and now it's broken.
Implement the `softmax(...)` function in the snippet below using base
TensorFlow functions, like `tf.exp(...)` and `tf.reduce_sum(...)`, so that
the code chunk works.

```python
import tensorflow as tf

d = 10
n_classes = 5
X = tf.placeholder(tf.float32, (None, d))
theta = tf.get_variable('theta', (d, n_classes), dtype=tf.float32)
h = tf.matmul(X, theta)

def softmax(h):
    pass

predictions = softmax(h)
```

10. TensorFlow uses **automatic differentiation** (autodiff) in order to
train complex machine learning models with gradient descent using an algorithm
called **backpropagation** (which we'll learn more about later).
TensorFlow uses a type of autodiff called **symbolic differentiation** in which
every operator knows its own gradient. There's another type of autodiff called
**numerical differentiation**. Look up symbolic and numerical differentiation
on the world wide web. Describe one advantage and one disadvantage for each.
Why did TensorFlow go with symbolic differentiation?

## Part II: Sentiment analysis warmup

Let's get our hands dirty. We're going to train a model
to predict whether movie reviews are positive or negative. For example,
given the sentence

```
a trashy , exploitative , thoroughly unpleasant experience .
```

we would want to predict that it has a low probability of being positive.

Luckily, we have access to a bunch of movie reviews and someone has taken the
time to mark whether or not they were positive. Checkout `data/train.text`.
We're going to train a logistic regression model (which you read about)
over a bag-of-words representation of the sentences.
This is a very simple vector representation of words and sentences. We'll take
a fixed vocabulary of size *d* where every word has an integer index.

* To represent a word, we simply take a *d* dimensional vector of zeros and put
a one at the word's index
* To represent a sentence, we sum the *d* dimensional vectors of all the words
in the sentence

These sentence representations lose all word order information, but we can
generalize the notion of "word" to an n-gram, which is a (short) sequence of
words. So, we would count `not good` as a single entry in the vocabulary,
as well as both `not` and `good`. We'll assemble all of these sentence vectors
into a matrix so that we can train a logistic regression model as normal,
using the provided labels.

We provide expected line counts in the functions you have to implement. These
are just to give you an idea of how many operations you need. Don't stress if 
you're over (or under) the counts.

#### Setup

If you already know what you're doing with Git, clone this repository.
If not, download it as a zip folder using the green "Clone or download" button
near the top of the page.

#### Creating word features

First up, let's create our n-gram bag-of-words featurizers. Do the following.

1. Implement `words_to_ngrams(...)` in `features.py` which converts a list of
words to a list of n-grams that pass a filter.
2. Implement `sentence_to_ngram_counts(...)` in `features.py` which converts
a raw sentence string to a list to a [Counter](https://docs.python.org/2/library/collections.html#collections.Counter) of the unique n-grams in it.
3. Look at the default filter in `util.py` and understand what it's doing.
Feel free to implement another one!
4. Look at SymbolTable in `util.py` and understand how it converts n-gram
counts to a data matrix.

#### Training our first model with NumPy

Next, we're going to complete a logistic regression model that we train with
NumPy operations. First, implement `sigmoid(...)` in `logistic_regression.py`.
Then, do the following in the `LogisticRegression` class.

1. Implement `_grad(...)` which computes the gradients of the log loss with
respect to the model parameters. The gradients are covered in the
[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
2. Implement `_update_weights(...)` which updates the parameters of the model
by calling `_grad(...)`.
3. Implement `predict(...)` to get probabilistic predictions for test data.
4. Look at `LogisticRegressionBase.train(...)` and understand the SGD
procedure it's following.

You're ready to train your model and get some results! Read `train.py` to
understand what it's doing. Run

```bash
python train.py
```

to train your model. You should get a test accuracy of at least 71%. Feel free
to change the hyperparameters (such as the learning rate or number of epochs).
Can you get the model to overfit (very high training accuracy, low test
accuracy)?

#### Level up: machine learning in TensorFlow

NumPy is great, but it isn't a good fit for large-scale models and deep
neural networks. That's where TensorFlow comes in. TensorFlow is overkill for
such a simple model and such a small dataset, but let's get some practice.
Do the following in the `TFLogisticRegression` class.

1. We don't have to compute any derivatives here by hand. TensorFlow does that
for us if we just tell it the loss. Implement `_loss(...)` to compute the
logistic loss (equivalent to cross entropy loss here) of the un-sigmoided
score (also called "logits").
2. Implement `_update_weights(...)` which runs a batch of data through the model
graph and updates the weights with gradient descent.
3. Implement `predict(...)` to get probabilistic predictions for test data.

Run

```bash
python train.py tf
```
to train the logistic regression model with TensorFlow. You should get similar
results.

## Submitting your work

That's it! Congrats, you're done with your first assignment. Create a .zip
file with `readings.txt`, `features.py`, and `logistic_regression.py`.
**ADD CANVAS UPLOAD INSTRUCTIONS.**
