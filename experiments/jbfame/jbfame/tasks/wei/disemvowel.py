from jbfame.tasks.base import PrepareOnlyTask, TaskDict
from jbfame.tasks.null import Null
import pandas as pd


class Disemvowel(PrepareOnlyTask):
    """
    An obfuscation attack that presents the prompt with all vowels removed.
    """

    name = "disemvowel"
    vowels = [ "a", "e", "i", "o", "u" ]

    def _apply_transform(self, prompt: str) -> str:
        return "".join(filter(lambda x: x not in self.vowels, prompt))

    def _prepare(self, prior_tasks: TaskDict) -> str:
        null_df = pd.read_parquet(prior_tasks[Null.name].prepared)

        # attach question id to prompt
        enc_df = pd.DataFrame({
            "prompt": null_df["prompt"], 
            "q_id": null_df.index.values
        })
        # encode prompt
        enc_df["prompt"] = enc_df["prompt"].apply(
            self._apply_transform
        )
        # save to parquet
        self.prepared = self.downloaded.replace(".dummy", ".parquet")
        enc_df.to_parquet(self.prepared, index=False)

        return self.prepared

    