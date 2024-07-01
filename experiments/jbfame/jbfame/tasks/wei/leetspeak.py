import base64

import pandas as pd

from jbfame.tasks.base import PrepareOnlyTask, TaskDict
from jbfame.tasks.null import Null


class Leetspeak(PrepareOnlyTask):
    """
    An obfuscation attack that presents the prompt in leetspeak, in which “a” is
    replaced with “@”, “e” is replaced with “3”, “i” is replaced with “!”, “o” is
    replaced with “0”, “are” is replaced with “r”, “be” is replaced with “b”, etc.
    """

    name = "leetspeak"
    alphabet = {
        "a": "@",
        "e": "3",
        "i": "!",
        "o": "0",
        "are": "r",
        "be": "b",
    }

    def _prepare(self, prior_tasks: TaskDict) -> str:
        def apply_transform(prompt: str) -> str:
            return "".join((map(lambda character: self.alphabet.get(character, character), prompt)))

        null_df = pd.read_parquet(prior_tasks[Null.name].prepared)

        # attach question id to prompt
        enc_df = pd.DataFrame({
            "prompt": null_df["prompt"], 
            "q_id": null_df.index.values
        })
        # encode prompt
        enc_df["prompt"] = enc_df["prompt"].apply(
            apply_transform
        )
        # save to parquet
        self.prepared = self.downloaded.replace(".dummy", ".parquet")
        enc_df.to_parquet(self.prepared, index=False)

        return self.prepared

    