from dataclasses import dataclass, field, InitVar

from bs4 import Tag


@dataclass
class DejargonizerData:
    common_words_amount: int = field(init=False)
    common_words_percentage: float = field(init=False)
    mid_frequency_words_amount: int = field(init=False)
    mid_frequency_words_percentage: float = field(init=False)
    rare_words_amount: int = field(init=False)
    rare_words_percentage: float = field(init=False)
    suitability_for_general_audience_score: float = field(init=False)
    number_of_words: int = field(init=False)
    stat_div_html: InitVar[Tag]

    def __post_init__(self, stat_div_html: Tag):
        pass