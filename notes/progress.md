# Progress Notes

These notes represent the progress made during a two-week stretch. The idea is
that I can show my progress from beginning to end at each bi-weekly meeting. I
have started this document on the 19th of February which is a little over a
month since the (effective) start of my thesis. Therefore, I will write a short
recap up until the last progress meeting which took place on the 14th of
February.

## Recap

I've established a research question: _Does the few-shot learning ability of a
an LLM contribute to the it security, i.e. its ability to resist jailbreaking
attacks?_ To this end, I've looked into various research papers that discuss
few-shot learning in the context of LLMs as well as machine learning in general.

## The Strategy

One of these few-shot learning oriented papers, by
@zhaoMeasuringIntrinsicFewShot2022a, talks about some 'intrinsic' hardness of
datasets. They arrive to this possibility by measuring the improvement that a
model sees after fine-tuning on certain tasks in a few-shot learning regime.
They consider the Spearman's correlation between different methods based on the,
as shown in

![Fine-tuning methods and their Spearman's correlation which indicates the
strength of a monotonic relation. If there is a: strong correlation, their
'ranking' of how much they improved at each task is similar, suggesting that
certain tasks are 'intrinsically harder' than
others.](assets/mfh_spearman.png){#fig:mfh}

It does make intuitive sense that some tasks are harder than others, which I am
trying to apply to the idea of _failure modes_ in jailbreaking as introduced by
@weiJailbrokenHowDoes2023. They identify two failure modes:

1. _Competing Objectives_ (CO):

   > Competing objectives stems from the observation that safety-trained LLMs
   > are typically trained against multiple objectives that can conflict with
   > each other. Specifically, state-of-the-art LLMs are trained for language
   > modeling [12], instruction following [41, 7], and safety [38, 7]. This
   > training can be exploited by crafting prompts that force a choice between
   > either a restricted behavior or a response that is heavily penalized by the
   > pretraining and instruction following objectives.

2. _Mismatched Generalization_ (MG):

   > Our second failure mode comes from observing that pretraining is done on a
   > larger and more diverse dataset than safety training, and thus the model
   > has many capabilities not covered by safety training. This mismatch can be
   > exploited for jailbreaks by constructing prompts on which pretraining and
   > instruction following generalize, but the model’s safety training does not.
   > For such prompts, the model responds, but without safety considerations.

To me, these failure modes seem to strongly relate to the outer and inner
alignment problems respectively. To this end, I would expect the former to be
more difficult to train on for pre-trained chat-like LLMs as this competes with
their original training policy. The latter, however, is a matter of a lack of
context, meaning that if we would provide safety training within that context,
the problem would essentially be solved. Moreover, if the model is more
adaptable, i.e. it has a greater capacity of learning in a few-shot learning
context compared to another model, it should be better equipped against such
attacks. ==This rests on the assumption that, somehow, the policy is 'deeply
embedded' into the model, and is more difficult to overcome than a lack of
examples. While this makes intuitive sense, I'm not sure whether this is fully
correct and my method is appropriate for determining whether this is actually a
inner- vs outer alignment problem.==

To show this, we need a dataset that contains jailbreaking examples where each
of the examples is classified by its failure mode. This way we can fine-tune the
model on both tasks (i.e. train the model to respond to the adversarial examples
appropriately) and determine which task has seen the most improvement. We can
then compare the increase in accuracy in these two tasks with that of the ones
mentioned by @zhaoMeasuringIntrinsicFewShot2022a. We expect the MG failure mode
to be similar to theirs as all these are few-shot learning tasks. Since the CO
failure modes is actively inhibiting the policy of the model, we would expect
that to be a harder task to train on, and therefore see less improvement on that
task under similar circumstances. Furthermore, training on the latter should
also have a greater inhibiting factor on general model performance as, again,
the model policy will be influenced.

### The Execution

During this time, I have primarily read and thought about how to structure my
research. It was during the last two weeks of this process that I have actually
been working toward a specific implementation-oriented goal. Therefore, besides
the plan I've made above and literature the necessary study, I have completed the
following tasks:

- Created code to interact with the fine-tuners used in
  @zhaoMeasuringIntrinsicFewShot2022a (2/8 are fully set up).
- Recreated the analysis presented in @zhaoMeasuringIntrinsicFewShot2022a.
- Set up the code so it also works in Google Colab. (While not hard, I had never
  worked with this platform before, and needed to figure out how to get around
  the lack of persistent environments.)

I had also worked on reproducing another paper written by
@yuGPTFUZZERRedTeaming2023b, which turned out to be a rabbit-hole that took a
lot of time, but didn't get me anywhere.

## Feb 14 - Feb 28

## References
