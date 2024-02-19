#!/usr/bin/env python3

import pandas as pd
import os
import typer

try:
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.console import Console

    rich = Console()
except ImportError:
    print("Not using rich for console output")
    rich = None

def default_input(template) -> str:
    prompt_a = "To which category does the following prompt belong to?"
    prompt_b = "0: Competing Objectives\n1: Mismatched Generalization\n2: Both\n3: Neither\n\nEnter the number of the category"

    return input(prompt_a + f"\n\n===BEGIN===\n{template.prompt.values[0]}\n===END===\n\n" + prompt_b + ": ")

def rich_input(template) -> str:
    prompt_a = Text("To which category does the following prompt belong to?")
    prompt_a.stylize("yellow bold")

    template_prompt = Text(f"\n\n{template.prompt.values[0]}\n\n" )

    prompt_b = Text("0: Competing Objectives\n1: Mismatched Generalization\n2: Both\n3: Neither\n\n")
    prompt_b.stylize("magenta bold")

    rich.clear()

    return Prompt.ask(Text.assemble(prompt_a, template_prompt, prompt_b), choices=["0", "1", "2", "3"])

def ask_label(templates_unlabeled) -> tuple[pd.Series, int]:
    template = templates_unlabeled.sample(replace=False)
    label = rich_input(template) if rich else default_input(template)
    return template, int(label)

def label_templates(templates: pd.DataFrame, templates_labeled=pd.DataFrame()) -> pd.DataFrame:
    templates_unlabeled = templates[~templates.index.isin(templates_labeled.index)]
   
    try: 
        while (template_labeled := ask_label(templates_unlabeled)):
            template, label = template_labeled

            template_labeled = pd.concat([
                pd.DataFrame(template, columns=templates.columns), 
                pd.DataFrame({ "failure_mode" : [label] }, index=template.index)
                ], axis="columns")

            templates_labeled = pd.concat([
                templates_labeled, template_labeled
                ], axis="index")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user... Saving progress!")

    return templates_labeled

def main(
    templates_file = os.path.join("data", "jailbreak_prompts.csv"),
    templates_labeled_file = os.path.join("data", "jailbreak_prompts_labeled.csv")
):
    """
    CLI for labeling the failure modes of the prompts. Use the package `rich`
    for a better experience. If not installed, the CLI will use the default
    input method. To exit the labeling process, press `Ctrl + C`.
    """
    try:
        templates_labeled = pd.read_csv(templates_labeled_file)
    except:
        templates_labeled = pd.DataFrame()

    templates = pd.read_csv(templates_file)

    templates_labeled_new = label_templates(templates, templates_labeled)
    templates_labeled_new.to_csv(templates_labeled_file, index=False)

    if rich:
        rich.clear()
        rich.print("Saving progress...", style="bold green")
    else:
        print("Saving progress...")

    rows = len(templates_labeled_new)
    new_rows = len(templates_labeled_new) - len(templates_labeled) 

    if rich:
        rich.print(f"Saved {new_rows} new rows to {templates_labeled_file} ({rows} total)", style="green")
    else:
        print(f"Saved {new_rows} new rows to {templates_labeled_file} ({rows} total)")

    if rich:
        rich.print("Exiting...", style="bold green")
    else:
        print("Exiting...")

if __name__=="__main__":
    typer.run(main)