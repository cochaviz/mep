from dataclasses import dataclass
from itertools import product
import subprocess
import os

import pandas as pd

from jbfame.tasks.null import Null
from jbfame.tasks.base import Task, TaskDict

@dataclass
class Dan(Task):
    name: str = "dan"

    def _download(self, output_dir: str) -> str:
        dan_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv"
        dan_path = "dan.csv"

        subprocess.run(["wget", "-qO", dan_path, dan_link], cwd=output_dir, check=True)

        self.downloaded = os.path.join(output_dir, dan_path)
        return self.downloaded

    def _prepare(self, prior_tasks: TaskDict) -> str: 
        dan_df = pd.read_csv(self.downloaded)
        null_df = pd.read_parquet(prior_tasks[Null.name].prepared) # type: ignore

        # df where all questions (and their ids) are combined with all prompts
        dan_df = pd.DataFrame(
            product(dan_df["prompt"], zip(null_df["prompt"], null_df.index.to_list())) , columns=["prompt", "question"]
        )
        # unpack question and question id
        dan_df[["question", "q_id"]] = pd.DataFrame(dan_df["question"].to_list(), index=dan_df.index)
        dan_df["prompt"] = dan_df[["prompt", "question"]].agg("\n".join, axis="columns")

        drop_columns = list(set(dan_df.columns.to_list()) - set(["prompt", "q_id"]))
        dan_df.drop(columns=drop_columns, inplace=True)

        self.prepared = self.downloaded.replace(".csv", ".parquet")
        dan_df.to_parquet(self.prepared, index=False)

        return self.prepared
