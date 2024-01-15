# Research Question

The general goal of this research is to establish a quantifiable metric for the
security of AI systems, specifically generative large language models (LLMs)
such as ChatGPT, Bard, etc. As we can already use Red-Teaming tools to quantify
the ability of a model to resist a set of adversarial examples, I would like to
propose alternatives to this approach. My desire for this is based on the
premise that security of an LLM is more than just the capacity of the model in
its current state. To this end, answering the following questions should
diversify the perspective of LLM security and advance the field as a
consequence:

1. Should the dataset on which the model is trained be scrutinized on the basis
   of security? More concretely: should the dataset used to enforce the model
   policy be considered when determining the security of a model?

2. While we can establish the current ability of a model to resist adversarial
   examples, should the adaptability of a model also be considered when
   determining its security? Is the capacity of a model to adapt to new
   adversarial examples indicative of its security?

3. While there is a distinction between safety and security, they are very
   closely related. Does, in the case of LLMs, security have some predictive
   value with regard to safety?