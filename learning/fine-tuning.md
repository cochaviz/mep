# Fine-Tuning

Many different methods of fine-tuning exist. However, for determining whether a
certain response is preferable with regard to (general) human values, RLHF is a
popular candidate. ==Currently my understanding of fine-tuning is lacking to a
significant degree when considering that my whole thesis is about evaluating
fine-tuning.== Since RLHF is the most influential in terms of the ethics of
LLMs, this is what we will consider for now.

## Reinforcement Learning through Human Feedback

_Reinforcement Learning through Human Feedback_ is the process of fine-tuning a
model by reinforcement learning where the policy is based on human preferences.
Generally, assuming we have a pre-trained and unaligned LLM-based chatbot, we
take the following steps:

1. Collect human feedback.
2. Fit a reward model on the human feedback.
3. Optimizing the LLM behavior using the reward model.

One 'fundamental' open problem highlighted in
@casperOpenProblemsFundamental2023a concerns the issue of disagreement: _A
single reward function cannot represent a diverse society of humans_. My
response to this issue is as follows:

> This is true, but misses the point. It implies some dependency on RLHF,
> however, this is a problem with literally every type of system where humans
> attempt to align a system (take a government for instance, or even a family
> vacation). This problem cannot be overcome by any system since humans will
> always disagree on some aspects.
> 
> Indeed, when disagreements are sufficient, this should somehow be highlighted
> by the system - or should it? Perhaps the system should be transparent enough
> to ensure that we can always identify particular 'rules' as contested. Perhaps
> we should let governments or citizen-bodies determine whether certain 'topics'
> are contested, after which experts should determine whether certain policy
> points are contested. There are no right answers.
> 
> One insinuation I do find definitively wrong however, is that we can overcome
> this issue by using a different system than RLHF. We should accept that
> disagreement never be overcome.