# Meeting Notes

## 23 Nov 2023

_Agenda:_

- My plan
- Kick-off meeting as soon as I have all my credits
- Two courses during the thesis => together with the last point probably later
  than June.
- PhD student help?

_Conclusion:_

- Write proposal asap
- Some PhD or Master's student (maybe from Radboud) will help

## 5 Dec 2023

_Conclusion:_

- A couple of things are still too vague to answer concretely:
  - Can we have a more concrete qualitative definition of adaptability?
  - What is a successful attack in this context, and why would it be useful to
    use that definition?
  - How do we define an adaptability characteristic and how can we distinguish
    between them?

- Therefore, the following plan:
  1. Use research from fields other than LLM security to determine how we could
     look at adaptability.
  2. Try and get some preliminary results with red-teaming tools. See how they
     work and why they constitute successful attacks, and whether those
     definitions are appropriate within our context.

## 21 Dec 2023

_Conclusion_:

- Get your ass to work

## Discussions with People

- Use different iterations of a model to verify the results from correlating
  number of iterations and spread. The architecture and pretraining haven't
  changed, therefore we don't expect to see any changes in the metric. (Felix)

- If the few-shot learning ability of a model is the same for training time as
  for testing time, we can imagine a situation in which a user-induced one-shot
  example is sufficient to revert the safety training. This leads to two
  questions: (i) Are training-time and inference time fine-tuning equivalent,
  and (ii) does this actually imply that improved few-shot learning ability
  directly counteracts safety?