from custom_metrics.metrics.explanation_quality_metrics import (
    explanation_type_metric_explicit,
    explanation_type_metric_explicit_v2,
    connection_to_everyday_life_metric_explicit,
    connection_to_everyday_life_metric_explicit_v2,
    connection_to_everyday_life_metric_explicit_v3,
    connection_to_everyday_life_metric_explicit_v4,
    metaphor_metric_explicit,
    metaphor_metric_explicit_v2,
    metaphor_metric_explicit_v3,
    metaphor_metric_explicit_v4,
    metaphor_metric_explicit_v5,
    metaphor_metric_explicit_v6,
    metaphor_metric_explicit_v7,
    metaphor_metric_explicit_v8,
    metaphor_metric_explicit_v8_deepseek,
    metaphor_metric_explicit_v8_deepseek_chat,
    metaphor_metric_explicit_v8_gpt4o_mini,
    metaphor_metric_explicit_v9,
    metaphor_metric_explicit_v10,
    metaphor_metric_explicit_v11,
    metaphor_metric_explicit_v12,
    metaphor_metric_explicit_v12_deepseek,
    metaphor_metric_explicit_v12_deepseek_chat,
    metaphor_metric_explicit_v12_gpt4o_mini,
    metaphor_metric_explicit_v12_1,
    content_units_metric_explicit,
    content_units_metric_explicit_v2,
    humor_metric_explicit,
    humor_metric_explicit_v2,
    humor_metric_explicit_v3,
    humor_metric_explicit_v4,
    humor_metric_explicit_v5,
    analogy_metric_explicit,
    analogy_metric_explicit_v2,
    correctness_metric_explicit,
    # New metrics from Baram-Tsabari & Lewenstein (2012)
    scaffolding_metric,
)
from custom_metrics.metrics.zemla_metrics import (
    internal_coherence_metric_explicit,
    completeness_metric_explicit,
    alternatives_metric_explicit,
    articulation_metric_explicit,
    perceived_truth_metric_explicit
)
from custom_metrics.metrics.jargon_metric import (
    jargon_metric
)
from custom_metrics.metrics.readablity_metrics import (
    flesch_kincaid,
    flesch_reading_ease,
    dale_chall,
    ari
)