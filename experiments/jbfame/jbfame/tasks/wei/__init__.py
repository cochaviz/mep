from jbfame.tasks.base import Task

from jbfame.tasks.wei.prefix_injection import PrefixInjection
from jbfame.tasks.wei.prefix_injection_hello import PrefixInjectionHello
from jbfame.tasks.wei.refusal_suppression import RefusalSuppression
from jbfame.tasks.wei.base64_input_only import Base64InputOnly
from jbfame.tasks.wei.style_injection_short import StyleInjectionShort
from jbfame.tasks.wei.style_injection_json import StyleInjectionJson
from jbfame.tasks.wei.distractors import Distractors
from jbfame.tasks.wei.distractors_negated import DistractorsNegated
from jbfame.tasks.wei.wikipedia import Wikipedia
from jbfame.tasks.wei.wikipedia_with_title import WikipediaWithTitle
from jbfame.tasks.wei.disemvowel import Disemvowel
from jbfame.tasks.wei.leetspeak import Leetspeak


all_tasks: dict[str, type[Task]] = { 
    PrefixInjection.name: PrefixInjection,
    PrefixInjectionHello.name: PrefixInjectionHello,
    RefusalSuppression.name: RefusalSuppression,
    Base64InputOnly.name: Base64InputOnly,
    StyleInjectionShort.name: StyleInjectionShort,
    StyleInjectionJson.name: StyleInjectionJson,
    Distractors.name: Distractors,
    DistractorsNegated.name: DistractorsNegated,
    Wikipedia.name: Wikipedia,
    WikipediaWithTitle.name: WikipediaWithTitle,
    Disemvowel.name: Disemvowel,
    Leetspeak.name: Leetspeak,
}

__all__ = [
    "Base64InputOnly",
    "PrefixInjection",
    "PrefixInjectionHello",
    "RefusalSuppression",
    "StyleInjectionShort",
    "StyleInjectionJson",
    "Distractors",
    "DistractorsNegated",
    "Wikipedia",
    "WikipediaWithTitle",
    "Disemvowel",
    "Leetspeak",
]