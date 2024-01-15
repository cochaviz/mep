---
bibliography: ../mep.bib
---

# Notes on adaptability

- My first idea was to just use some custom metric of adaptability, but I don't
  think that is very convincing given that I have no credibility. I should ask
  some people in the field.

- **Few-shot learning might be a good candidate?**

  > Definition 2.2. Few-shot Learning (FSL) is a type of machine learning problems
  > (specified by E, T , and P), where E contains only a limited number of
  > examples with supervised information for the target T.
  >
  > - @wangGeneralizingFewExamples2021

  It can be well-argued our problem fits within the general 'vibe' of few-shot
  learning problems. The authors mention three typical scenarios of FSL:

  1. Acting as a test bed for learning like human.
  2. Learning for rare cases.
  3. Reducing data gathering effort and computational cost.

  Clearly, point 2 fits pretty well within our context. Attacks are rare cases
  after all. There is also one-shot learning, but it's generally pretty easy to
  extrapolate and create multiple similar cases of attacks. Thus, even if we
  regularly find new types of attacks, we say that it's relatively easy to
  (manually) generate new samples from them. We consider this hard to do on a
  large scale, however. Therefore it's not many-shot learning.

  On the other hand, few-shot learning focuses on the idea of small number of
  samples for a specific class. **We would somehow have to bridge this gap to
  include rare cases of common classes?** Although point two seems to include
  this, and a paper by @brownLanguageModelsAre2020 also seems to support this.

- **Is there a way to quantify the few-shot learning ability of a model?**

  The authors @basuHARDMETADATASETUNDERSTANDING2023, propose a model for
  identifying hard tasks and showing how models generally show reduced
  performance on those types of tasks. The disparity between the 'normal' and
  hard-extracted set could be indicative of few-shot learning performance.

  While this doesn't exactly indicate what I would like, **it does display the
  ability of a model to work with small numbers of samples in a relatively
  general manner**. Furthermore, it might show how some attacks could be
  inherently more difficult to attack against.

  - The procedure introduced by @basuHARDMETADATASETUNDERSTANDING2023 is made
    for sets of images. Could we also use this for text and apply it to attack
    sets?

    It seems so, but there is also the issue of classes. This will probably
    require a taxonomy on attacks which might make the metric very dependent on
    where we draw boundaries between attacks.

  - Another paper proposed by @zhaoMeasuringIntrinsicFewShot2022a performs a
    similar task, but now specifically for NLP tasks. In this paper, they
    propose a method for determining the _Intrinsic Few-shot Hardness_ for a
    specific dataset respective to a model, i.e. a measure of how hard this
    particular model will find it to train on this dataset.

    Maybe we can reproduce their findings in the context of adversarial
    examples? If it holds in that case too, there might be something to say for
    the use of such examples in measuring few-shot learning performance.

- The question I was wondering about before[^question] is kind of hard to
  answer. It's very general and dependent on model complexity. Models also
  generally tend to forget more than they stop learning necessarily.

  We can, however, look into whether they tend to forget defenses against other
  attacks. **Can be somehow proof or disprove that defending against one attack
  reduces the defenses against another?**

[^question]: "Is there a point at which learning becomes saturated?"

- The fact that "LLMs are trained using Reinforcement Learning through Human
  Feedback (RLHF)" [@yuGPTFUZZERRedTeaming2023] gives more support to the idea
  that the number of samples during alignment is fundamentally limited.

## How Intrinsic Few-Shot Hardness relates to formal Few-shot Learning

According to @wangGeneralizingFewExamples2021, the difficulty of few-shot
learning originates from two disparities between the realistic, or empirical,
best hypothesis, $h_I$, and the optimal hypothesis, $\hat{h}$. Firstly, we
are limited by what can be achieved using empirical methods. Secondly, we are
limited by what can be achieved within our model.

Considering the latter first, we have to imagine there is a limited pool of
hypotheses, $\mathcal{H}$, that can be produced using the model and data
available. The disparity between the optimal hypothesis and the optimal
hypothesis within $\mathcal{H}$, $h^*$, can be described by @eq:app_error, also
called the _approximation error_.

$$
\mathcal{E}_{app}(\mathcal{H}) = \mathbb{E}[R(\hat{h}) - R(h^*)]
$${#eq:app_error}

As mentioned before, the former is based on the limitations of empirical
methods. Specifically to approach the optimal hypothesis in $\mathcal{H}$. This
in turn creates another disparity called the _estimation error_ as expressed by
[@eq:est_error]

$$
\mathcal{E}_{est}(\mathcal{H}, I) = \mathbb{E}[R(h^*) - R(h_I)]
$${#eq:est_error}

The total error caused by the disparity between $\hat{h}$ and $h_I$ is then
described by their sum per @eq:tot_error.

$$
\mathbb{E}[R(h^*) - R(h_I)] = \mathcal{E}_{app}(\mathcal{H}) +
\mathcal{E}_{est}(\mathcal{H}, I)
$${#eq:tot_error}

Compare this with the method introduced by @zhaoMeasuringIntrinsicFewShot2022a
the so-called, _spread_, which is defined as follows:

## Candidates and Applicability

Perhaps trying to measure the few-shot learning performance is a futile effort.
However, we won't know unless we try! The following are potential methods for
determining the few-shot ability, i.e. adaptability, in various ways.

- _Method-specific Few-shot Hardness_ (MFH): Method-specific few-shot hardness
  is a function that takes a dataset $D$, an adaptation method $m$, a
  pre-trained model $f$, and returns the few-shot hardness of $D$ with respect
  to $m$ and $f$. Concretely, it is the accuracy of the adapted model, $f_m$,
  normalized over the accuracy prior to fine-tuning.

  $$
  \mathbf{MFH}(D, f, m) = \frac{
      \mathrm{acc}(m(f, D_{tr}), D_{ts})
    }{
      \mathrm{acc}(f, D_{ts})
    }
  $$

  - Fairly inexpensive to compute
  - Only describes the performance of a model respective to a certain
    fine-tuning method and dataset.

- _Intrinsic Few-shot Hardness_ (IFH): Given that the correlations between were
  fairly evenly distributed among datasets and that their average correlations
  differ significantly (do they?), we can define the hardness according to the
  average of the correlations for each database.

  $$
  \mathbf{IFH}(f, D) = \frac{\sum_{m \in M} \mathbf{MFH}(D, f, m)}{|M|}
  $$

  - Based on the assumption that hardness is intrinsic to certain datasets and
    that there is not a large disparity between the MFH of different fine-tuning
    methods.
  - Quite an expensive experiment to run as multiple fine-tuning methods are
    necessary for a single measurement.

- _Spread_:

- _Hard-Meta-Dataset++ Disparity_ (HMDD): By extracting the explicitly hard
  tasks from a dataset, $D$, using this method, resulting in $D'$, we can
  compare the performance before and after the extraction and use that as the
  performance metric:

  $$
    \mathbf{HMDD}(f, D, perf) = \frac{
      \mathrm{perf}(f_{D'_{tr}}, D'_{ts})
    }{
      \mathrm{perf}(f_{D_{tr}}, D_{ts})
    }
  $$

  As we expect the performance, where $\mathrm{perf}$ is a metric according to
  'higher is better', on the extracted dataset to be lower than the
  non-extracted one, we expect $0 \le \mathbf{HMDD} \le 1$.
  
## References
