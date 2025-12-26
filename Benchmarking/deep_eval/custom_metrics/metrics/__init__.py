from custom_metrics.metrics.explanation_quality_metrics import (
    explanation_type_metric_explicit,
    explanation_type_metric_explicit_v2,
    connection_to_everyday_life_metric_explicit,
    connection_to_everyday_life_metric_explicit_v2,
    metaphor_metric_explicit,
    metaphor_metric_explicit_v2,
    metaphor_metric_explicit_v3,
    content_units_metric_explicit,
    content_units_metric_explicit_v2,
    humor_metric_explicit,
    humor_metric_explicit_v2,
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