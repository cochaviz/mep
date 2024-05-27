#!/usr/bin/env python3

from typing import Optional, Any

import pandas as pd
import os
import typer

prompt_a_text = "To which category does the following prompt belong to?"

options_dict = {
    "failure_mode": ["Competing Objectives", "Mismatched Generalization", "Both", "Neither"],
    "jailbreak" : ["Jailbroken", "Not Jailbroken", "Not Sure"]
}
options = options_dict["jailbreak"]

options_text = "\n".join(map(lambda option: f"{options.index(option)}: {option}", options))
prompt_b_text = f"{options_text}\nEnter the number of the category"

try:
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.console import Console

    rich = Console()
except ImportError:
    print("Not using rich for console output")
    rich = None

def default_input(data: pd.Series, remaining: Optional[int] = None) -> str:
    remaining_text = f" ({remaining} samples remaining)" if remaining else ""
    return input(prompt_a_text + f"\n\n===BEGIN===\n{data.values[0]}\n===END===\n\n" + prompt_b_text + f"{remaining_text}: ")

def rich_input(data: pd.Series, remaining: Optional[int] = None) -> str:
    prompt_a = Text(prompt_a_text)
    prompt_a.stylize("yellow bold")

    template_prompt = Text(f"\n\n{data.values[0]}\n\n" )

    prompt_b = Text(prompt_b_text)
    prompt_b.stylize("magenta bold")

    remaining_text = Text(f"\n{remaining} remaining\n\n", style="dim")

    rich.clear()

    return Prompt.ask(
        Text.assemble(prompt_a, template_prompt, remaining_text if remaining else "", prompt_b), 
        choices=list(map(lambda x: str(x), range(len(options))))
    )

def ask_label(unlabeled) -> Optional[tuple[Any, int]]:
    if len(unlabeled) == 0:
        return None
    
    sample = unlabeled.sample()
    label = rich_input(sample, len(unlabeled) - 1) if rich else default_input(sample, len(unlabeled) - 1)
    return sample, int(label)

def label_templates(
    source: pd.Series, 
    labels: pd.Series,
) -> pd.Series:
    unlabeled = source[labels.isna()]

    if len(unlabeled) == 0:
        print("All values have been labeled!")
        return labels
   
    try: 
        while (value_label := ask_label(unlabeled)):
            value, label = value_label
            labels[value.index] = label
            unlabeled = source[labels.isna()]

    except KeyboardInterrupt:
        print("\n\nInterrupted by user... Saving progress!")

    return labels


def write_df_as_file(df: pd.DataFrame, filename: str) -> None:
    if filename.endswith(".csv"):
        return df.to_csv(filename, index=False)
    if filename.endswith(".parquet"):
        return df.to_parquet(filename, index=False)

    raise NotImplementedError("File format not supported")

def open_file_as_df(file: str) -> pd.DataFrame:
    if file.endswith(".csv"):
        return pd.read_csv(file)
    if file.endswith(".parquet"):
        return pd.read_parquet(file)

    raise NotImplementedError("File format not supported")

def main(
    source_file: str,
    source_column: str,
    target_file: Optional[str] = None,
    target_column: str = "manual_label",
):
    """
    CLI for labeling the failure modes of the prompts. Use the package `rich`
    for a better experience. If not installed, the CLI will use the default
    input method. To exit the labeling process, press `Ctrl + C`.
    """
    if not target_file:
        name_tags_extensions = os.path.basename(source_file).split('.')
        basename, tags, extension = name_tags_extensions[0], ".".join(name_tags_extensions[1:-1]), name_tags_extensions[-1]

        target_file = os.path.join(
            os.path.dirname(source_file), 
            f"{basename}.{tags}.labeled.{extension}"
        )

    df = open_file_as_df(source_file)

    if target_column in df.columns:
        labeled = df
    else:
        try:
            labeled = open_file_as_df(target_file)
        except FileNotFoundError:
            print("Creating new file for labeled data...")
            labeled = df.copy()
            labeled.insert(0, target_column, None)

    labeled[target_column] = label_templates(
        df[source_column], 
        labeled[target_column],
    )

    if rich:
        rich.clear()
        rich.print("Saving progress...", style="bold green")
    else:
        print("Saving progress...")

    write_df_as_file(labeled, target_file)

    labeled_proportion = labeled[target_column].count() / len(labeled)

    if rich:
        rich.print(f"Labeled {labeled_proportion} of values in {source_file}", style="green")
    else:
        print(f"Labeled {labeled_proportion} of values in {source_file}")

    if rich:
        rich.print("Exiting...", style="bold green")
    else:
        print("Exiting...")

if __name__=="__main__":
    typer.run(main)