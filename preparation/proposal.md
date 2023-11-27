---
title: Research Proposal
author: Zohar Cochavi
date: 16-10-2023
---

## Introduction

Large-language models are used prevalently and are only expected to play a
larger role as time goes on. The safety of these systems is therefore of upmost
importance, as this impacts the harm done, directly or indirectly, as a
consequence of their availability. Most influential in this case are models
deployed as _Machine Learning as a Service_ (MLaaS) systems. Most notably
ChatGTP, Google Bard, and Microsoft Bing which, besides only being accessible
via an API, are closed-source [@]. The verifiability of that safety is therefore
hard to establish which is concerning, especially in conjunction with the
aforementioned influence they have.

The safety of publicly available closed-source systems should therefore be
scrutinized. Currently, these efforts come in the form of _jailbreaking_ in
which adversaries engineer prompts that circumvent safety mechanisms. The aim of
jailbreaking is to have a model behave as if it has no (or limited?) safety
mechanisms.

Currently, LLM safety is enforced by fine-tuning and post-processing filters
akin to spam filters [@]. While these measures are successful in some manner,
they do not fundamentally tackle the problem of safety. They reflect the idea of
safety as an afterthought, instead of safety as a fundamental property of the
system. My aim is to determine whether this particular idea can be exploited in
a closed-source setting. This requires identifying the reasons  why current
methods cannot reliably exploit all such systems, and whether this approach
relieves some of these problems.

## Research Question

_**Can current methods that provide safe LLMs be shown to be inherently
insecure? That is, can we show that safety training as a fine-tuning step is
only effective within some bounds?**_

- What are the current state-of-the-art methods used to jailbreak open-source
   and closed-source LLM systems.
- Is there a disparity between the performance of jailbreaking methods on
   open-source and closed-source LLM systems. (Very probably yes, you can probe
   an open-source system more).
- Can we show that these models are performing inference attacks with the goal
  of inferring the fine-tuning data specifically? Does that mean that the safety
  of a LLM is directly correlated with the amount of safety training? Does this
  allow for a quantitative measure of the safety of an LLM?

<!-- An important distinction to make is between white- and black-box attacks. In
practice, I believe this distinction appears in the form of open- and
closed-source models respectively. In the context of this text, I will use the
terms open- and closed-source as follows: _Open-source_ models have publicly
available source code and pre-trained models, while _Closed-source_ models can
only be probed through APIs in a black-box setting. Closed-source then also
implies that models can only be probed in a limited manner. While open-source
models allow for near unlimited probing and fine-tuning of the attack method,
closed-source models require the adversary to deploy an effective attack with
limited samples. In this text, I will the _source_ and _box_ models
interchangeably, adhering to the definitions as given before. -->

**Hypothesis**: Yes, because there open-source models are often
provided with pre-trained versions on which large amounts of probing can be
performed.

**Claim**: An attack developed for an open-source (pre-trained) model should
therefore not be feasible on closed-source systems.

## Plan of Attack

## Notes

- Can we maybe figure out why current methods work? Is there some underlying
  mechanism that makes them able to break these systems?

- Does fine-tuning actually make these models safer? It feels like there is some
  limit, but can we show where that limit is independent of current attack
  models?  I guess we could show how attack success rate trends with the amount of
  fine-tuning that's happening. Then we would be able to say whether current
  safety training is currently adequate and what that might mean in the future.

- Each paper I'm reading just appears to do the same thing in a slightly
  different way: train some other LLM to perform an adversarial attack on
  open-source models. The only one that felt really different was the one about
  **time-based attacks**. There is something fundamentally different about that
  paper than the others, but I'm not quite sure what...

  I think it's because they were determined to _understand_ something about the
  model or the way that it ensures safety. Others just try to optimize a model
  to attack, there is no new knowledge.

- Can we figure out why these attacks work? That is, the attacks based on
  fine-tuning to ensure safety? The victim model clearly doesn't understand
  everything about avoiding harm. Is that what we're exploiting, or is it simply
  the case that there is not enough training done? What could be the different
  reasons for these models being vulnerable to jailbreaking?

  1. _Not enough training_: There is no upper bound, training and safety are
     ~linear or better.
  2. _Wrong type of training_: There is an upper bound, and training will have
     diminishing returns. Given enough (reasonable) time, one will always be
     able to crack an LLM.

  The paper by @weiJailbrokenHowDoes2023, shows exactly this. Essentially, it
  displays how training by fine-tuning will always be exploitable. While
  incredibly valuable as a first step, contemporary understanding of systems
  tells us that there is no such thing as a perfectly secure or safe system.
  Security and safety are easily confused, regardless both only work within
  certain bounds.
  
  As determining whether a system is secure or safe enough requires us to
  quantitatively determine whether it fulfills certain conditions, we need a
  quantitative measure of the system under scrutiny. While the paper by
  @weiJailbrokenHowDoes2023 contributes significantly towards such an
  understanding, it only does so in a qualitative sense. I therefore want to
  have a qualitative measure of the safety of a system based, or at least to
  determine whether such a measure is possible.

  Explicitly, what I would like to investigate is how the amount of safety
  fine-tuning is correlated with the effectiveness of state-of-the-art attacks.
  Because there are many different LLM-type systems, I would like to investigate
  whether these all display similar responses to such fine-tuning.

- AI safety measures are put in place to prevent malicious actors from abusing
  the system. This is therefore not a case of safety, but security as we are
  preventing an external actor from causing harm. Furthermore, "alignment with
  human values" is often called as source of the problem, but alignment is a
  matter of perspective. In this case, the values are not aligned with the
  values of the _creator_, not that of the user. Even if we, in most of these
  black-white-scenarios, consider users malicious.
