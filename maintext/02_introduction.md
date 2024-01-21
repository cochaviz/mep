# Introduction

==Why are LLMs relevant==

As indicated by @casperOpenProblemsFundamental2023a, the issues pertaining to
LLM alignment, and by extension its safety, are currently solved by
_Reinforcement Learning through Human Feedback_ which, although powerful, has
limitations that are severely unadressed. 

- ==These methods are all aimed toward generating data that could improve the
  behavior of the model in a pretty straightforward manner. However, it doesn't
  seem like anyone is trying to _scrutinize the quality of the data when
  considering the behavior of the model_.==

- ==I thoroughly believe there is no way of quantifying a certain policy when it
  is contested. From this, we can deduce that people will always discuss whether
  some particular action followed the policy. If we restrict this discussion,
  we're projecting our values onto the user which, when considering ethical
  discussions, is undesirable; an external policymaker should be able to
  influence the decisionmaking of the policy used to evaluate the model. This
  means that humans should always be able to be kept in the loop when
  costructing or evaluating the policy the model should follow. This, in turn,
  means that some sort of human feedback will always be necessary if the point
  of the policy is contested enough.==

- ==Given the above, we can expect policy definition to devaite depending on who
  we ask. This means that certain undesirable behavior can occur if we
  incorporate _all_ feeback with equal weight; this might occur in a situation
  where no-one is happy with the outcome. The quality of the data can therefore
  be indicative of the user satisfaction (and therefore safety) with regard to
  the model. Long story short: even though we are dealing with a contested
  topic, we should maintain enough clarity in the data to avoid undesirable
  behavior.==