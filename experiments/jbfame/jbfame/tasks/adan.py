
import os
import subprocess
import traceback

from jbfame.tasks.base import Task, TaskDict

class ADan(Task):
    name = "adan"

    def _download(self, output_dir: str) -> str:
        try:
            # download source
            subprocess.run("test -d AutoDAN || git clone https://github.com/SheltonLiu-N/AutoDAN.git AutoDAN", cwd=output_dir, shell=True).check_returncode()
            
            # prepare environment 
            if subprocess.run("conda env list | grep AutoDAN", shell=True).returncode != 0:     
                subprocess.run("conda create -y -n AutoDAN python=3.9 && conda run -n AutoDAN pip install -r AutoDAN/requirements.txt", cwd=output_dir, shell=True).check_returncode()

            # download models 
            try:
                subprocess.run("cd AutoDAN/models && conda run -n AutoDAN python download_models.py", cwd=output_dir, shell=True).check_returncode()
            except subprocess.CalledProcessError as e:
                print("Inability to download models is most probably because of Llama2 being a private model. Please login with HuggingFace using credentials that have access to the model.")
                raise e
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            traceback.print_exc()

        # download nltk dependencies in autodan environment
        try:
            subprocess.run(
                "conda run -n AutoDAN python -m ntlk.download all",
                shell=True,
            ).check_returncode()
        except subprocess.CalledProcessError as e:
            print("Something went wrong with downloading nltk dependencies. Please check the error message.")
            print(e.stderr)
            traceback.print_exc()

        self.downloaded = os.path.join(output_dir, "AutoDAN")
        return self.downloaded
        

    def _prepare(self, prior_tasks: TaskDict) -> str:
        raise NotImplementedError("This function is not implemented yet.")

        subprocess.call("conda run -n AutoDAN python autodan_hga_eval.py".split(), cwd=downloaded_task["adan"]) 
    
        return ":)"