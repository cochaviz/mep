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
a closed-source setting.

## Research Question

_**What is necessary to improve current state-of-the-art methods to work on
limited training data such that they can reliably attack MLaaS systems?**_

* What are the current state-of-the-art methods used to jailbreak open-source
   and closed-source LLM systems.
* Is there a disparity between the performance of jailbreaking methods on
   open-source and closed-source LLM systems.

> An important distinction to make is between white- and black-box attacks. In
> practice, I believe this distinction appears in the form of open- and
> closed-source models respectively. In the context of this text, I will use the
> terms open- and closed-source as follows: _Open-source_ models have publicly
> available source code and pre-trained models, while _Closed-source_ models can
> only be probed through APIs in a black-box setting. Closed-source then also
> implies that models can only be probed in a limited manner. While open-source
> models allow for near unlimited probing and fine-tuning of the attack method,
> closed-source models require the adversary to deploy an effective attack with
> limited samples. In this text, I will the _source_ and _box_ models
> interchangeably, adhering to the definitions as given before.

**Hypothesis**: Yes, because there open-source models are often
provided with pre-trained versions on which large amounts of probing can be
performed.

**Claim**: An attack developed for an open-source (pre-trained) model should
therefore not be feasible on closed-source systems.

* Does this disparity

## Plan of Attack
