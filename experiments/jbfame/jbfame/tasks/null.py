from dataclasses import dataclass
import os
import subprocess

import pandas as pd

from jbfame.tasks.base import Task, TaskDict

class Null(Task):
    """
    Null task. This task is used to supply the other tasks with questions.
    """

    name: str = "null"

    def _download(self, output_dir: str) -> str:
        null_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv"
        # null_link = "https://raw.githubusercontent.com/SheltonLiu-N/AutoDAN/main/data/advbench/harmful_behaviors.csv"
        null_path = f"{self.name}.csv"

        subprocess.run(["wget", "-qO", null_path, null_link], cwd=output_dir).check_returncode()

        self.downloaded = os.path.join(output_dir, null_path)
        return self.downloaded

    def _prepare(self, prior_tasks: TaskDict) -> str: 
        """
        Extract question and question id from the dataset. This dataset is used to
        supply the other datasets with questions.
        """
        null_df = pd.read_csv(self.downloaded)

        try:
            null_df.drop(columns=["target"], inplace=True) 
            null_df.rename(columns={"goal": "question"}, inplace=True)
        except KeyError:
            print("Assuming null dataset is default and in the right format.")

        null_df = pd.DataFrame({ 
                "prompt": null_df["question"].to_list(),
        })

        self.prepared = self.downloaded.replace(".csv", ".parquet")
        null_df.to_parquet(self.prepared, index=False)

        return self.prepared
