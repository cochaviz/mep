# Measuring Task Difficulty

I'm looking for methods to determine whether one task is more difficult than
another. Very simply, we could look at the improvement after fine-tuning in
terms of accuracy, precision, F1 or what have you. While this is very direct and
should work, I am curious to different methods of establishing task difficulty.

## Using Loss as Difficulty

The authors @arnoldUniformSamplingEpisode2022 use the negative log-likelihood
(or cross-entropy) at training time to determine the difficulty of a task.
Effectively, their difficulty is more akin to 'difficulty to train on', rather
than 'difficulty to successfully complete' which is what I would normally
associate with the difficulty of a task. Just quickly remembering the
log-likelihood:

$$
\mathrm{Loss}(\mathbf{x}) = - \log\mathbb{P}_\theta(y | \mathbf{x})
$$

Where, if we take the likelihood of attributing the $k$th label to example $y$ as
$\hat{y}_k$ and $y_k$ the one-hot encoding of the class of label $y$ (i.e. $0$
where class is incorrect and $1$ where label is correct),

$$
\mathrm{CrossEntropy}(y_k, \hat{y}_k) = - \sum_k y_k \log \hat{y}_k \\
$$

we can see that log-likelihood and cross-entropy are equivalent if we use
one-hot encoding of the ground-truth label. For each episode (i.e. each few-shot
subset of the training data), we then track this loss as the quantifier for the
difficulty of this particular episode.

One advantage of this method is that it is very easy to obtain, and does very
well reflect how likely a model is to be right about a particular set of
examples. Still, this only evaluates the model on a limited number of training
samples. We might otherwise track the loss over multiple episodes and determine
the difficulty of the over-all task based on the final loss.
