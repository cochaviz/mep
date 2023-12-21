# Notes on Adaptability

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
