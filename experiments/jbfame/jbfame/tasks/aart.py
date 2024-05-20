import pandas as pd
import art

from jbfame.tasks.base import PrepareOnlyTask, TaskDict
from jbfame.tasks.null import Null

class AArt(PrepareOnlyTask):
    name = "aart"

    def prepare(self, prior_tasks: TaskDict) -> str:
        def remove_words(text, words):
            return filter(lambda token: token.lower() not in words, text.split())

        def substitute(text, word_map):
            return [ text.replace(word, mapped_word) for word, mapped_word in word_map.items() ]

        def find_keywords_spacy(sentences: pd.Series):
            try:
                import spacy
                en = spacy.load('en_core_web_sm')
            except ImportError:
                raise ImportError("Cannot find scapy, please install with pip install scapy to use this task.")
            except OSError:
                raise OSError("Cannot find the model, please download with python -m spacy download en_core_web_sm")

            return sentences.apply(lambda sentence: remove_words(sentence, en.Defaults.stop_words))
            
        null_df = pd.read_parquet(prior_tasks[Null.name].prepared)

        # find keywords and transform into ascii_art
        null_df["ascii_map"] = find_keywords_spacy(null_df["prompt"]).apply(lambda keywords: { 
            keyword : "\n" + art.text2art(keyword) for keyword in keywords 
        })

        # create new dataframe where questions have their keywords replaced by ascii art
        aart_df = pd.DataFrame(
            null_df.apply(lambda row: substitute(row["prompt"], row["ascii_map"]), axis="columns").explode(),
            columns=["prompt"],
        ).reset_index(names=["q_id"])

        self.prepared = self.downloaded.replace(".dummy", ".parquet")
        aart_df.to_parquet(self.prepared, index=False)

        return self.prepared