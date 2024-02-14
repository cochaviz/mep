# Experiments - Notes and Ideas

## Messing About

- Simply trying to run some LLMs through huggingface transformers shows that
  inference on models ~500 million parameters takes about 5 seconds in moderate
  context size (i.e. about 5 x 64 tokens in a conversation). Any larger models
  just take too long: _ModalGPT_ took about 5 minutes per response, and
  _TinyGPT_ about 1-2 minutes.

## Replicating IFH

- So after working for about 1.5 weeks, I have come to a point where I _could_
  technically reproduce the paper. The problem is, however, that I would need to
  adjust each of the ~15 tasks to each of the ~5 fine-tuning methods. This would
  just take too much time, therefore I will simply use the results included in
  their repo and use my own 'jailbreaking task' on the setup I have now. Then
  the plan is to determine how similar jailbreaking is compared to the other
  tasks in terms of fine-tuning.
- Determining 'similarity' is a bit tricky here.
