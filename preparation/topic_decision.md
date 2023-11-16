---
bibliography: bibliography.bib
---

# Topic Decision

## Ideas from Stjepan

- Pair-wise comparison of trustworthy AI pillars.
- Inference attacks on text-to-image models [@wuMembershipInferenceAttacks2022].

## Investigation

Firstly, considering the trade-offs between pillars of trustworthy AI, I find it
interesting but rather abstract. I would feel more confident jumping into
something like membership inference just because the goal is very clear.

It seems like several papers have attempted _membership inference_ attacks on
text-to-image models. Most of these attacks are performed on GANs and/or
diffusion models, but the most recent paper mentions that none are practical
in real-life scenarios:

> The proposed attacks outperform the baseline from the literature, but their
> performance is far from rendering them useful in real-world applications
> [@dubinskiMoreRealisticMembership2023]

## Ideas

I have found about 4-5 papers in 2023 (and 1 or 2 in 2022) that talk about
membership inference in text-to-image models. They cover an extensive range of
models, but the most recent paper does mention no attacks are relevant in
real-life scenarios. Perhaps it would be interesting to see how these attacks
fare against Large Multi-modal Models (<http://arxiv.org/abs/2309.17421>), but as
such models are not publicly accessible I don't think that is realistic.
Nonetheless, the question of what would it take to make text-to-image membership
inference attacks a realistic threat seems to me an interesting one.

These text-to-image attacks also reminded me of the text-based attacks that are
used against LLMs, specifically to circumvent their safety features. There is
definitely a lot more research in this field, but it doesn't seem like there are
any 'mulit-stage' attacks that execute some inference as recon and then apply
this information to the attack. Most assume you have a model that you can probe
without any limit to the number of requests, or have very specific prompts that
make the LLM break its guidelines (<https://github.com/QData/TextAttack>,
<https://arxiv.org/abs/2307.15043>, <https://browse.arxiv.org/pdf/2307.02483.pdf>).
I might very well have missed something when reading the papers, but before I go
into the weeds of determining whether it's feasible or even relevant, I wanted
your opinion on the idea, which concretely would be: can we use existing
inference attacks to augment text-based attacks on LLMs to improve their
reliability. It seems like the safety of a model is only really a 'fine-tuning'
step, not a fundamental part of the model. Therefore, I think using a recon step
on a deployed model might allow an attack to be more effective as it would only
have to infer information about the fine-tuning step (assuming the model is
known and the training data is publicly available).

The last idea was to investigate what kind of impact these safety measures have
on the performance of black-box inference attacks. I don't know much about how
these safety measures are implemented or enforced, but i can imagine they have
an impact on the performance black-box inference attacks. This question very
much lacks specifics, but since we're just shooting ideas I thought I'd include
it anyway.
