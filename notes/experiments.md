# Experiments - Notes and Ideas

## Messing About

- Simply trying to run some LLMs through huggingface transformers shows that
  inference on models ~500 million parameters takes about 5 seconds in moderate
  context size (i.e. about 5 x 64 tokens in a conversation). Any larger models
  just take too long: _ModalGPT_ took about 5 minutes per response, and
  _TinyGPT_ about 1-2 minutes.
