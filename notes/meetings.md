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

<<<<<<< Updated upstream
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
=======
## 31 Jan 2024

_Agenda:_

- Finally, I have all my credits
- Started working on the Fuzzer, but I realised after it might be better to work
  on reproducing the paper by Zhao et al. (IFH)
- Deliberated heavily on the definitions of safety vs. security. Why? Everyone
  seems to use them as interchangeable and, to me, it seems like the goal of
  security is a lot more well-defined than safety. This would make security more
  'solvable' than safety. Using the terms wrong can lead to confusion and a
  misdirection of efforts.
- On a similar threat, I was wondering about a counter-argument to the issue of
  'adaptability' as an attribute to safety: if this means that prompt-based
  fine-tuning is also more effective, does that imply that jailbreaks that use
  _competing objectives_ as a failure more are also more effective?
- In essence, what I would like to do is to determine whether security and
  safety somehow differ in how-well defined their policy is. Maybe we can even
  somehow show that the issue of security is indeed 'well-defined' (whatever
  that may mean). If this is the case, there is a strong argument to be made for
  the prospect of separate systems that identify malicious prompts.

_Conclusion:_

- None of my premises were refuted, so that's nice I guess.
- Get to work, but hopefully there will be someone that could help me out.
>>>>>>> Stashed changes
