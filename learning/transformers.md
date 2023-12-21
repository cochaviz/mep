# Transformers

The most important thing to understand about transformers conceptually, is that
they have an, effectively, infinite attention span. Meaning that they can
reference any part of a generated piece of text, allowing them to stay coherent
regardless of the input size.

The transformer is modeled according to an encoder-decoder network. Here, the
encoder carries the task of converting the input text to its **embedding** (some
many-dimensional encoding of a word where the distance between words expresses
dissimilarity) and determining the relation of each word to the others in what
is called **self-attention**. The decoder effectively has two parts: (i) a
structure which is essentially an encoder for the generated output sequence, and
(ii) a structure which takes the encodings of the input and output and predicts
the most likely word by some linear classifier.

## Encoder

## Decoder
