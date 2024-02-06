#!/usr/bin/env python3

import subprocess
import typer
import os

from enum import Enum

base_dir = "fine_tuners"
remote_location = f"https://raw.githubusercontent.com/cochaviz/mep/experiments/src/replicating_ifh/{base_dir}"

class AvailableFineTuner(str, Enum):
    none = "none"
    lmbff = "lmbff"
    adapet = "adapet"
    all = "all"

def setup_local(method: AvailableFineTuner):
    return subprocess.run(["bash", f"fine_tuners/{method}.sh"], 
                          stdout=open(f"{base_dir}/{method}.log", "w"), 
                          stderr=subprocess.STDOUT, 
                          cwd=base_dir)
    
def setup_remote(method: AvailableFineTuner):
    remote_file = subprocess.Popen(
        ["wget", "-qO-", f"{remote_location}/{method}.sh"]
        , stdout=subprocess.PIPE)
    subprocess.run(["bash"], 
                   stdout=open(f"{base_dir}/{method}.log", "w"), 
                   stdin=remote_file.stdout, 
                   cwd=base_dir)
    remote_file.wait()

def download_and_prepare(
    fine_tuner: AvailableFineTuner,
    remote: bool = False,
    verbose: bool = False
    ):
    methods = [ fine_tuner ]

    if "none" in fine_tuner:
        print("No fine-tuners selected.")
        return
    if "all" in fine_tuner:
        methods = [ e.value for e in AvailableFineTuner ]

    try:
        os.mkdir("fine_tuners")
    except FileExistsError:
        print("fine_tuners directory already exists.") 

    for method in methods:
        print(f"Setting up: {method}...")

        process = setup_remote(method) if remote else setup_local(method)

        if verbose:
            print("==[BEGIN: {method}.log]==")
            print(open(f"{base_dir}/{method}.log").read())
            print("==[END: {method}.log]==")

        if process.returncode == 0:
            print(f"Successfully set up: {method}!")
        elif process.returncode == 2:
            print(f"Setup already exists: {method}... Check logs.")
        else:
            print(f"Failed to set up: {method}... Check logs.")
            
if __name__ == "__main__":
    typer.run(download_and_prepare)
