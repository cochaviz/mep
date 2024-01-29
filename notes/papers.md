# Notes on Papers

## On Measuring the Intrinsic Few-Shot Hardness of Datasets

Advances **few-shot learnability** in two ways:

1. Given a dataset, few-shot performance of various adaptation methods is highly
   correlated.

   - ==What is an adaptation method?==
   - ==What constitutes few-shot performance?==

   This is saying that the extent to which a method improves the few-shot
   learning ability of a model on a particular dataset is more dependent on the
   data rather than the method.

   - ==Does this hold across different models?==

2. A metric that measures the intrinsic hardness of a dataset according to some
   distance between the train and test set. The so-called **Spread**.

   This is justified by mentioning a higher Spearman correlation (0.467 vs
   0.356, where the latter is probably the best of previous methods).

### Method-Specific Few-Shot Hardness

- **MFH**: method-specific few-shot hardness is a function that takes a dataset
  $D$, an adaptation method $m$, a pre-trained model $f$, and returns the
  few-shot hardness of $D$ with respect to $m$ and $f$. Concretely, it is the
  accuracy of the adapted model, $f_m$, normalized over the accuracy prior to fine-tuning.

  $$ \mathbf{MFH}(D, f, m) = \frac{\mathrm{acc}(m(f, D_{tr}), D_{ts})}{\mathrm{acc}(f, D_{ts})} $$

They used a couple of datasets to determine how their measure, MFH, compared to
others. Firstly **GLUE** and **SuperGLUE**, where they considered 11 tasks that
cover different task formats and objectives an called it **FS-GLUE**. These also
use various performance metrics.

Next, they also curated a dataset called **FS-NLI** by extracting tasks from two
other datasets and standardizing the performance metric to accuracy. This
resulted in "28 datasets, covering a wide range of tasks such as sentiment
analysis, negation comprehension, name entity classification (NEC), paraphrase
detection, event classification, logical entailment et".

### Intrinsic Few-Shot Hardness

- **IFH**: Given that the correlations between were fairly evenly distributed
  among datasets and that their average correlations differ significantly (do
  they?), we can define the hardness according to the average of the
  correlations for each database.

  $$ \mathbf{IFH(f, D)} = \frac{\sum_{m \in M} \mathbf{MFH}(D, f, m)}{|M|} $$

The problem is that this is very computationally expensive because it requires
determining MFH over multiple fine-tuning methods in $M$ (==although I don't
directly see the problem as we've done exactly this in the previous
paragraph.==). Therefore, the authors propose the **Spread** metric which
approximates IFH based solely on the databases and how the models interpret
these.

### Experiments

The authors performed various experiments of which some are more interesting
than others. Let's rank them from most to least interesting:

1. _Measuring IFH across pre-trained models_: While, before, we expected
   relatively uniform IFH across different methods, the result here seems less
   so. There is a fairly large disparity between models in terms of their
   correlation (ranging from .58 to .72). This correlation describes strength of
   the linear relation between the IFH metric and the performance disparity
   before and after applying the fine-tuning method on different models.

   I guess the conclusion that we can draw is that we can expect the hardness
   measurement to be fairly predictive of the ability of the model to improve,
   regardless of the underlying pre-trained model. Some datasets are
   intrinsically harder than others, and this hardness might be more predictive
   of a model's performance than the ability of the model itself.

   It's problematic that this conclusion is drawn based on just two
   pre-trained models. It might be necessary to determine whether this holds
   for a variety of pre-trained models. If, regardless of the pre-trained
   model, this conclusion still holds, there is no sense in trying to find a
   metric for determining the few-shot learning ability of a particular model
   _in general_. It might still be sensible to determine how well a model
   generalizes from a certain task, comparing the ability of models to
   generalize on a certain tasks. This, however, does rest on the assumption
   that the authors made a mistake in discussing _datasets_ over _tasks_.

   ==If it is true that we can determine the hardness of a dataset, we might be
   able to argue that certain safety datasets are inadequate for training or
   testing. Therefore, instead of scrutinizing the model for its ability to
   withstand attacks now and in the future, we scrutinize the dataset on which
   we draw these conclusions. _Should we separate the task from the data? Or is
   the data the task itself?_ Since we are talking about edge-cases (even in
   only some cases not necessarily all), we should not consider the task
   'following the policy', but 'does this resemble a previous violation of the
   policy?'. Therefore, in this particular case, we _should_ consider the
   intrinsic hardness of the dataset as an indication of whether or not is
   constitutes a reasonable example of the violation of the policy. We could
   make the requirement that these examples are publicly available to ensure
   that the model can be expected to follow its guidelines.==

2. _Measuring Intrinsic Dataset Hardness_: The authors compared their method for
   dataset hardness with existing metrics in the field. These results show
   promising, as their correlation with IFH and the dataset hardness features
   are not only the strongest but also the fastest. Since their metric is
   (essentially) solely based on database properties, they proposed a method for
   performing the train/test split based on their metric to improve model
   performance.

   - ==How does this influence the distributions of the data? Is it legit?==
   - ==They, again, only use few models.==

## Making Pre-Trained Language Models Better Few-Shot Learners

This was a paper referenced in the previous paper, and one that I discovered
independently and wanted to read anyway. This probably means it should be a
useful one to understand?

## Large Language Models are Few-Shot Learners

In this paper, researchers at OpenAI introduce GPT-3. Here, they specifically
consider the task of zero-, one-, and few-shot _in-context_ learning and its
impact on accuracy. In context-learning ability refers to the improved task
performance when presented with examples of the task in the current text
context. By showing GPT-3 examples in natural language before asking it to
perform a task, it shows improved capabilities. In the paper, this is referred
to as few-shot learning, which is different from few-shot learning in the
traditional sense which refers to classification tasks where the number of
examples in a particular class is small.

They did this specifically to evaluate the generalization performance of GPT-3,
I guess coming from the perspective of a non-expert that could simply show a
couple of examples for a task instead of needing technical knowledge on
fine-tuning. However, this does beg the question: is this a 'more powerful
method' than traditional fine-tuning or vice-versa? This is a particularly
interesting question with regard to the idea of adaptability as an integral part
of security.

## GPTFuzzer: Red-Teaming Large Language Models with Aut-Generated Jailbreak Prompts - Yu et al

Reproducing the paper seems super easy given that all the code is [available on
github](https://github.com/sherdencooper/GPTFuzz) and it seems really easy to
work with.

## Jargon

This paragraph contains a list of unknown terms and their definitions collected
over the course of noting the papers described above.

- _Light-weight / Parameter Efficient Tuning (PET)_: Parameter-efficient tuning
  enables fine-tuning an LLM on a new task without retraining all its
  parameters, often counted in billions. Instead, a small subset of the model’s
  parameters or additional parameters are fine-tuned while the rest remain
  frozen. This “delta tuning” approach can be seen as a refined version of
  retraining specific layers or appending a classifier to a pre-trained model,
  aiming for comparable performance as fine-tuning the entire model.
  [src](https://medium.com/data-from-the-trenches/parameter-efficient-llm-fine-tuning-2577d93cd9e0)

- _Natural Language Inference (NLI)_: Natural language inference (NLI) is the
  task of determining whether a "hypothesis" is true (entailment), false
  (contradiction), or undetermined (neutral) given a "premise".
  [src](https://paperswithcode.com/task/natural-language-inference/latest)

- _Prompt-based Fine-tuning_: In prompt-based fine-tuning, we train a generative
  model on specific tasks by using prompts and their respective responses.
  Consider the following example:

  > Assume the template is “`<text>` It is `<mask>`”, where the token `<text>`
  > stands for the original text, and the verbalizer is {“positive”:“great”,
  > “negative”:“terrible”}. The sentence “Albert Einstein was one of the
  > greatest intellects of his time.” will first be wrapped by the pre-defined
  > template as “Albert Einstein was one of the greatest intellects of his
  > time. It is `<mask>`”. The wrapped sentence is then tokenized and fed into a
  > PLM to predict the distribution over vocabulary on the `<mask>` token
  > position. It is expected that the word great should have a larger
  > probability than terrible.

  In short, we're translating specific tasks into a generative context.

- _Pearson Correlation Coefficient_: In statistics, the Pearson correlation
  coefficient [PCC](a) is a correlation coefficient that measures linear
  correlation between two sets of data. It is the ratio between the covariance
  of two variables and the product of their standard deviations; thus, it is
  essentially a normalized measurement of the covariance, such that the result
  always has a value between −1 and 1.
  [src](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
  Importantly, it only measures _linear_ correlation.
