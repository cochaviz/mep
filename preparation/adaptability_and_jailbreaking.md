# On Adaptability and Jailbreaking

An approach to determining the security of a LLM could be to not only determine
how resistant the model _currently_ is to attacks, but how well it might resist
them in the _future_. Up until now, the resistiveness against jailbreaking is
generally approached from a strict red-teaming perspective. Models, however, are
constantly being adapted to new use-cases, fine-tuned to better perform using
human feedback, and, along those same lines, improved to resist new attack
examples. The ability for a model to adapt to new attacks quickly and in some
generalizable manner is therefore important for its ability to resist new
attacks. This principle, which we will henceforth refer to as **adaptability**,
when quantified, could therefore be an alternative measure of a model's
restiveness against (jailbreaking) attacks.

## Adaptability as Few-Shot Learning Ability

Looking for such a princple, one quickly arrives at the idea of so-called
**few-shot learning**. According to @wangGeneralizingFewExamples2021, the
principle of few-shot learning can be used to describe a situation where a
model has to perform "learning for rare cases". This seems particularly
applicable in the context of jailbreaking or fine-tuning with human feedback in
general [@yuGPTFUZZERRedTeaming2023]. A more general sentiment of the idea of large
language models as few-shot learners is shared by @brownLanguageModelsAre2020 in
the famous and aptly-named "Language Models are Few-Shot Learners".

Therefore, we can express the adaptability of a model as a quantification of its
few-shot learning ability. Various such measures exist, such as the **Inherent
Few-shot Hardness** (IFH) as proposed by @zhaoMeasuringIntrinsicFewShot2022a.
The authors define some distance between the training and test set of a
particular dataset with respect to some model. This distance is then the
so-called IFH. Their paper uses examples, models and datasets in the context of
NLP which makes it particularly applicable for our use-case, although they have
discovered some hints to the independence of the hardness of a dataset and the
model which learns on the dataset. Another example is that of
**HardMetaDataset++** as proposed by @basuHARDMETADATASETUNDERSTANDING2023. They
similarly identify the hardness of a task relative to a model, but, instead of
establishing a metric, they extract the difficult tasks from the dataset. This
creates a disparity between the normal and hard dataset that could be used to
identify the few-shot performance of a model. While their method was concerned
with image classification, their approach seems to work with any neural net.

It is important to note that the type of task need not be related to adversarial
examples. It is the general capability of an LLM to adapt to new examples that
we determine to be important for security. I claim that the faster a model can
adapt the safer it is because of the changing (maybe even volatile) nature of
cyber security and policy. The metric is aimed to determine the general
_adaptability_ of a model which happens to be expressed in few-shot learning
performance.

## Verification of Security by Adaptability

Now, there are two sides to the claim that these metrics for few-shot learning
are valid measures for security. Firstly, (i) models with higher scores should
perform better on tasks with few examples. This is simply per the definition of
few-shot learning @wangGeneralizingFewExamples2021. Secondly, models with higher
scores should perform better on security-related tasks. This is because I claim
that (ii) LLMs learn as few-shot learners, and/or because security-related tasks
are few and far between in 'normal' use-cases, which, given (i), should mean
that these LLMs should perform better on security-related tasks.

Ideally, we would first verify statement (i), since (ii) is necessary for (i).
However, the first is rather hard to verify since measuring few-shot learning
performance is still an open research problem. Still, we could verify the claims
the authors made in the respective papers which, in case of the paper by
@zhaoMeasuringIntrinsicFewShot2022a is feasible, but seems very challenging in
the case of the paper written by @basuHARDMETADATASETUNDERSTANDING2023. Again,
because their methodology and results are in the context of image processing.
My preferred approach, however, would be at least determine these approaches are
consistent. That is, ranking models by their performance using each metric
should result in the same list.

The second statement can be verified by deploying various red-teaming tools
available for use on LLMs[^red-teaming]. These deploy attacks on LLMs using an automated
approach, and then determine whether such an attack was successful. Firstly, we
expect models with higher scores in the adaptability test to perform better on
these tests given they have been trained in an equal manner. While this is not a
problem in principle, this does limit the applicability of this test on a
real-world setting. Often, pre-trained models are distributed which are then
fine-tuning for specific tasks. Ideally, the test should be independent of such
pre-training. Therefore, an alternative method would be perform fine-tuning on
some fairly general security suite. Then, we determine how much the models have
improved using the same red-teaming tools. This more directly tests their
adaptability in a security setting.

[^red-teaming]: Various red-teaming tools exist which evaluate the model on its
    resistance on a range of jailbreaking attacks. These vary from simple
    rule-based tests, to genetic algorithms that modify existing prompts to
    break the target model [@perezRedTeamingLanguage2022,
    @yuGPTFUZZERRedTeaming2023].

## Conclusion

This brings us to the final set of research questions that have arisen in this
discussion, the main question being:

_Can we quantify the few-shot learning ability of an LLM, and use this as a
valid security measure?_

Here, we use methods as proposed by @zhaoMeasuringIntrinsicFewShot2022a and
@basuHARDMETADATASETUNDERSTANDING2023 to quantify the few-shot learning ability
(as explained in [Verification of Security by Adaptability]). Verifying whether
these actually measure few-shot learning ability is hard, but if they do, we
expect both to be consistent with one-another.

_Are the proposed measures for few-shot learning ability consistent with
one-another?_

If they are not, we should determine why this is the case. Otherwise, the
following question should also be confirmed.

_Are the proposed measures for few-shot learning ability consistent with the
security evaluation according to red-teaming tools given equal training?_

If so, we ask the following question:

_Are the proposed measures for few-shot learning ability predictive of how well
models adapt to new security-specific training?_

If so, this would then confirm the hypothesis.

## References
