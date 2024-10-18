from deepeval.test_case import LLMTestCaseParams

from readability_metrics.readability import ReadabilityMetric

# r = Readability(text)
#
# r.flesch_kincaid()
# r.flesch()
# r.gunning_fog()
# r.coleman_liau()
# r.dale_chall()
# r.ari()
# r.linsear_write()
# r.smog()
# r.spache()
#

flesch_kincaid = ReadabilityMetric(
    name="Flesch-Kincaid Grade Level",
    metric="flesch_kincaid"
)

flesch_reading_ease = ReadabilityMetric(
    name="Flesch Reading Ease",
    metric="flesch"
)

dale_chall = ReadabilityMetric(
    name="Dale Chall",
    metric="dale_chall"
)

ari = ReadabilityMetric(
    name="ARI",
    metric='ari'
)