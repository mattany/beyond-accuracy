from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from custom_metrics.metrics.constants import g_eval_default_params

explanation_type_metric_explicit = GEval(
    name="Explanation Type",
    evaluation_steps=[
        """1.Given the below 5 numbered categories of explanation, assign a score matching the most advanced explanation type present in the answer.
{
    "explanation_types": {
        {
            "score": 0,
            "type": "Absent",
            "description": "No explanation provided.",
            "example": ""
        },
        {
            "score": 2.5,
            "type": "Definition",
            "description": "A short definition of a certain entity is present, without further explanation. Look for explanations that raise more questions rather than providing a sufficient explanation."
            "example": "The internet is a virtual network."
        },
        {
            "score": 5,
            "type": "Elucidating",
            "description": "A definition with an example/nonexample. Focus on providing clear, direct information and examples.",
            "example": "Antibiotics only work on bacteria, which means that they can only be used for diseases caused by microbes belonging to the bacteria family. Flu, on the other hand, is caused by viruses."
        },
        {
            "score": 7.5,
            "type": "Quasiscientific",
            "description": "An explanation that creates an image in the mind, often by using an analogy. Look for language that draws a visual or conceptual parallel. Phrases like 'consider as,' 'similar to,' or 'like a' indicate analogies.",
            "example": "Consider each computer as a node and the Internet as a web."
        },
        {
            "score": 10,
            "type": "Transformative",
            "description": "Any explanation whose starting point is what the audience might think, that points to problems with the existing conceptions, or that explains why the scientifically accepted theory is more plausible or fruitful. Look for statements that challenge common misconceptions or preconceived notions. Phrases like 'it may seem counterintuitive,' 'most people think,' or 'common belief' indicate challenges to existing views.",
            "example": "I believe that the Bible must be interpreted in the context in which it was written. When the original text was written, people did not have our understanding of the natural world. They needed an explanation for their existence in terms that they could understand. That took the form of God creating them. Today we have proof that species evolve from one another and there is no reason to think that we are so special that we should not follow the same rules as the rest of nature."
        }
    }
}
""",
        "2. When scoring, do not consider correctness. Instead, follow the descriptions in step 1 to determine the score.",
        "3. If an answer contains multiple types of explanations, assign the score based on the best explanation type in the answer.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

explanation_type_metric_explicit_v2 = GEval(
    name="Explanation Type",
    evaluation_steps=[
        """1. Given the below 5 categories of explanation, assign a score matching the most advanced explanation type present in the answer.

SCORE 0 - Absent:
  No explanation provided. The response does not attempt to explain the concept.
  Example: (empty or off-topic response)

SCORE 2.5 - Definition:
  A short definition is present, but without elaboration or examples. The explanation names or describes what something is, but leaves the reader with unanswered questions.
  Example: "The internet is a virtual network."

SCORE 5 - Elucidating:
  A definition accompanied by concrete examples or non-examples that clarify the concept. The explanation provides specific instances that help the reader understand.
  Example: "Antibiotics only work on bacteria, which means they can only be used for diseases caused by microbes belonging to the bacteria family. Flu, on the other hand, is caused by viruses."

SCORE 7.5 - Quasiscientific:
  An explanation that creates a mental image, typically through an analogy. Look for explicit comparisons between the concept and something familiar. Phrases like "think of it as," "similar to," "like a," or "imagine" indicate analogies.
  Example: "Think of each computer as a node and the Internet as a web connecting them all together."

SCORE 10 - Transformative:
  An explanation that explicitly addresses what the audience might incorrectly believe, identifies problems with that misconception, and explains why the correct view is more accurate. Look for phrases like "you might think," "it seems like," "contrary to popular belief," or "a common misconception is."
  Example: "You might think that heavier objects fall faster than lighter ones—after all, a bowling ball seems to hit the ground before a feather. But in a vacuum, they fall at exactly the same rate. Air resistance is what makes the feather float, not its weight. Galileo demonstrated this centuries ago, overturning the intuition people had held since Aristotle."
""",
        "2. When scoring, do not consider correctness. Instead, follow the descriptions in step 1 to determine the score.",
        "3. If an answer contains multiple types of explanations, assign the score based on the best (highest-scoring) explanation type present.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

content_units_metric_explicit = GEval(
    name="Content Units Explicit",
    evaluation_steps=[
        '1. A standalone fact is a fact that does not depend on other facts. Identify and extract all standalone facts from the Actual Output.',
        '2. Count each standalone fact as a separate content unit.',
        '3. Pay no attention to other dimensions such as factual correctness.',
        '4. Return the amount of content units present in the Actual Output.',
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

content_units_metric_explicit_v2 = GEval(
    name="Content Units Explicit (v2)",
    evaluation_steps=[
        """1. A CONTENT UNIT is a single, independent piece of information that could stand alone as a fact.
   Each content unit should convey one distinct idea, claim, or piece of data.
   
   Examples of how to count content units:
   - "Water boils at 100°C" = 1 content unit
   - "Water boils at 100°C and freezes at 0°C" = 2 content units (two separate facts joined by 'and')
   - "The heart pumps blood through the body" = 1 content unit
   - "The heart has four chambers: two atria and two ventricles" = 2 content units (the number of chambers + their names)
   
   Example passage with 2 content units:
   "Two facts motivate my research—first, diverse systems are healthier systems, and second, humans are rapidly altering diversity around the globe."
   → Content unit 1: diverse systems are healthier
   → Content unit 2: humans are altering diversity""",
        """2. Counting rules:
   - Count each distinct fact separately, even if in the same sentence
   - Definitions count as 1 content unit
   - Examples, analogies, and metaphors do NOT count as separate units—they illustrate or explain facts, not add new ones
   - Restating the same fact in different words = still 1 unit""",
        "3. Do NOT evaluate correctness—count all stated facts regardless of accuracy.",
        "4. Return the total count of content units as the score.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

connection_to_everyday_life_metric_explicit = GEval(
    name="Connection to everyday life",
    evaluation_steps=[
        """1.Check the output contains an explicit connection to common knowledge, a previous event, or a news
story that was not already embedded in the question.""",
    "2. Return a score of 10 if the above holds, and a score of 0 otherwise."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

connection_to_everyday_life_metric_explicit_v2 = GEval(
    name="Connection to everyday life (v2)",
    evaluation_steps=[
        """1. Check if the answer contains an EXPLICIT reference to a NAMED external entity:
   - A specific news story or current event (e.g., "the H1N1 virus outbreak", "the 2008 financial crisis")
   - A named historical event (e.g., "World War II", "the Chernobyl disaster")
   - A named public figure, brand, or cultural reference (e.g., "Einstein", "Tesla", "Star Wars")""",
        """2. The following do NOT count as connections:
   - Generic everyday examples without named entities ("like boiling water", "similar to driving a car")
   - Vague references ("as we all know", "in everyday life", "people often")
   - Scientific analogies or metaphors (these are captured by other metrics)
   - References already embedded in the original question""",
        """3. Return a score of 10 ONLY if there is at least one named external entity as defined in step 1.
   Return 0 otherwise. Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)


humor_metric_explicit = GEval(
    name="Humor Explicit",
    evaluation_steps=["1. Determine if the explanation includes explicit jokes or ironic language.",
                      "2. Return a score of 10 if jokes or ironic language are present in the answer, and 0 otherwise.",
                      "3. If you aren't sure whether the answer contains jokes or ironic language, return a score of 5."],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v2: Strictly aligned with Baram-Tsabari (2012) definition
# "Humor included both explicit jokes and ironic language"
humor_metric_explicit_v2 = GEval(
    name="Humor Explicit (v2)",
    evaluation_steps=[
        """1. HUMOR includes explicit jokes AND ironic language.
   
   Look for ANY of the following:
   - Explicit jokes (with or without a formal punchline)
   - Puns or wordplay meant to amuse
   - Ironic language: understatement, tongue-in-cheek remarks, or wry observations
   
   Examples of humor PRESENT:
   - "...killing our own cells, which wouldn't be very wise" (ironic understatement)
   - "Atoms are like tiny LEGO blocks, except you can't step on them at 3 AM" (joke with punchline)
   - "Why did the electron leave the atom? Because it had no potential" (pun)
   - "Evolution doesn't plan ahead – if it did, it would have given us better knees" (ironic observation)""",
        """2. The following are NOT humor:
   - Creative or vivid analogies/metaphors without irony (e.g., "DNA is like a blueprint", "bacteria are like ninjas")
   - Engaging or enthusiastic tone without jokes or irony
   - Personification without irony (e.g., "the virus wants to replicate", "chemicals are villains")
   - Vivid or dramatic descriptions
   - Playful language that lacks actual jokes or ironic statements""",
        """3. The key question: Is there an explicit JOKE or IRONIC statement?
   If yes → return 10.
   If no (even if creative, vivid, or playful) → return 0.
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)



analogy_metric_explicit = GEval(
    name="Analogy Explicit",
    evaluation_steps=[
      """1. Consider the following definition of analogies: Analogies are defined as a systematic mapping between two situations:
the source (familiar situation) and the target (novel situation).
      """,
    "2. Based on the above definition, determine whether the explanation includes analogies or not. Do not take correctness into account.",
    "3. Return a score of 10 if at least one analogy is present in the answer, and 0 if no analogies are present in the answer.",
    "4. If you aren't sure whether the answer contains a analogy or not, return a score of 5.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

analogy_metric_explicit_v2 = GEval(
    name="Analogy Explicit (v2)",
    evaluation_steps=[
        """1. An ANALOGY explicitly compares TWO DIFFERENT DOMAINS to explain a concept. It must have:
   - A SOURCE: a familiar, concrete situation (e.g., "a library", "traffic on a highway", "a factory assembly line")
   - A TARGET: the scientific/abstract concept being explained
   - EXPLICIT comparison language such as: "like", "similar to", "just as", "think of X as Y", "imagine", "consider", "analogous to", "comparable to"
   
   Examples of analogies:
   - "Think of DNA like a recipe book" (source: recipe book, target: DNA)
   - "The immune system works like an army defending a castle" (source: army/castle, target: immune system)
   - "Electrons orbit the nucleus similar to how planets orbit the sun" (source: solar system, target: atom)""",
        """2. The following are NOT analogies:
   - Simple examples without cross-domain comparison ("For example, water boils at 100°C")
   - Definitions ("Photosynthesis is the process by which plants convert sunlight to energy")
   - Within-domain comparisons ("Viruses are smaller than bacteria")
   - Metaphors without explicit comparison markers ("The brain is a computer" - this is a metaphor, not an analogy)""",
        """3. Return a score of 10 if at least one clear analogy (as defined above) is present.
   Return 0 otherwise. Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

metaphor_metric_explicit = GEval(
    name="Metaphor Explicit",
    evaluation_steps=[
      """1. Consider the following definition of metaphors: Metaphors structure one concept in terms of another. Unlike
analogies, metaphors do not necessarily map directly between source and
target; similarities can be associative.
      """,
    "2. Based on the above definition, determine whether the explanation includes metaphors or not. Do not take correctness into account.",
    "3. Return a score of 10 if at least one metaphor is present in the answer, and 0 if no metaphors are present in the answer.",
    "4. If you aren't sure whether the answer contains a metaphor or not, return a score of 5.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

metaphor_metric_explicit_v2 = GEval(
    name="Metaphor Explicit (v2)",
    evaluation_steps=[
        """1. A METAPHOR describes one thing AS IF it were another, WITHOUT explicit comparison words like "like" or "similar to".
   Metaphors use figurative language that would be literally false but conveys meaning through imagery.
   
   Types of metaphors to look for:
   - Direct identity statements: "The cell is a factory", "Genes are blueprints", "The brain is a computer"
   - Personification: "The virus wants to spread", "Evolution selects for traits", "Nature designed this mechanism"
   - Using one domain's vocabulary for another: "genetic code", "cellular machinery", "information flows through neurons"
   
   Examples of metaphors:
   - "The heart is a pump" (direct metaphor - heart literally described as pump)
   - "Cancer cells are invaders" (personification/role assignment)
   - "The genetic code contains instructions" (code/instructions vocabulary applied to biology)""",
        """2. The following are NOT metaphors:
   - Explicit comparisons using "like", "similar to", "as if" (these are similes or analogies)
   - Technical terms that have become standard ("genetic code" is borderline - only count if used with clear figurative intent)
   - Literal descriptions ("The heart pumps blood" - this is literal, not metaphorical)
   - Simple examples or definitions""",
        """3. Return a score of 10 if at least one clear metaphor (as defined above) is present.
   Return 0 otherwise. Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v3: Improved version addressing edge cases identified in consistency analysis
metaphor_metric_explicit_v3 = GEval(
    name="Metaphor Explicit (v3)",
    evaluation_steps=[
        """1. A METAPHOR describes one thing AS IF it were another, WITHOUT explicit comparison words.
   Metaphors use figurative language that would be literally false but conveys meaning through imagery.
   
   CLEAR METAPHORS (score 10):
   - Direct identity statements: "The cell is a factory", "Genes are blueprints", "The brain is a computer"
   - Strong personification with IMPOSSIBLE actions: "The virus wants to spread", "Evolution designed this", "Nature selected for traits"
     (These attribute intention/desire to entities that cannot literally have them)
   - Cross-domain vocabulary used figuratively: "genetic code contains instructions", "cellular machinery", "molecular scissors"
   
   Examples:
   - "The heart is a pump" → METAPHOR (heart literally called a pump)
   - "Cancer cells are invaders" → METAPHOR (cells assigned military role)
   - "Beyond that lie dragons" → METAPHOR (figurative danger/unknown)
   - "Zombie-proteins that reanimate" → METAPHOR (proteins described as zombies)""",
        """2. NOT metaphors (exclude all of these):
   
   - SIMILES/ANALOGIES: comparisons with "like", "similar to", "imagine", "think of it as"
   - IDIOMS: frozen phrases with conventional figurative meanings that native speakers would recognize as common sayings (e.g., "play mind tricks", "tip of the iceberg", "here be dragons")
   - METONYMY: organizations standing for their people ("The USA wants" = government wants, "The company decided")
   - WEAK PERSONIFICATION: animals, systems, or abstractions with common traits ("cats figured out", "server checks", "the market fears")
   - SHAPE DESCRIPTIONS: literal visual resemblance ("potato-shaped asteroid", "mushroom cloud")
   - TECHNICAL TERMS & DEAD METAPHORS: standardized vocabulary, including terms used literally in their original domain ("shock wave", "chain reaction" in physics) or so conventionalized they no longer feel figurative ("genetic code", "viral marketing")
   - LITERAL STATEMENTS: "The heart pumps blood", "Fields attract and repel" """,
        """3. KEY DECISION RULE:
   Ask: "Is this phrase NOVEL to this context, or would a native speaker recognize it as a common expression?"
   
   - If a native speaker would say "oh, that's a common saying" → NOT a metaphor (it's an idiom)
   - If the figurative language is FRESH and INVENTED for this explanation → METAPHOR
   
   Only score 10 for ACTIVE metaphors that create fresh figurative meaning.
   Return 0 for idioms, explicit analogies, weak personification, or technical terms. 
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v4: Further refined based on edge case analysis - stricter novelty focus
metaphor_metric_explicit_v4 = GEval(
    name="Metaphor Explicit (v4)",
    evaluation_steps=[
        """1. A METAPHOR is NOVEL figurative language that creates fresh imagery, NOT standard expressions.
   
   CLEAR METAPHORS (score 10) - creative/unexpected comparisons:
   - "Zombie-proteins that reanimate and need a head-shot" (creative extended metaphor)
   - "Your DNA is a recipe book written in a 4-letter alphabet" (fresh framing, not textbook)
   - "Neurons gossiping with each other" (unexpected personification)
   - "The star is a cosmic pressure cooker slowly boiling itself" (vivid novel imagery)""",
        """2. NOT metaphors (exclude all of these):
   
   - SIMILES/ANALOGIES: comparisons with "like", "similar to", "imagine", "think of it as"
   - IDIOMS: frozen phrases with conventional figurative meanings that native speakers would recognize as common sayings (e.g., "play mind tricks", "tip of the iceberg", "here be dragons")
   - METONYMY: organizations/institutions standing for their people ("The USA wants", "The company decided", "Science says")
   - WEAK PERSONIFICATION: everyday humanizing language for animals, systems, or abstractions ("cats figured out", "the algorithm decides", "the market fears")
   - SHAPE DESCRIPTIONS: literal visual resemblance ("potato-shaped asteroid", "mushroom cloud")
   - TECHNICAL TERMS & DEAD METAPHORS: standardized vocabulary or textbook analogies ("the cell is a factory", "DNA is a blueprint", "the heart is a pump", "genetic code", "viral marketing")
   - LITERAL STATEMENTS: "The heart pumps blood", "Fields attract and repel" """,
        """3. KEY DECISION RULE:
   Ask: "Is this phrase NOVEL to this context, or would a native speaker recognize it as a common expression?"
   
   - If a native speaker would say "oh, that's a common saying" → NOT a metaphor (it's an idiom)
   - If the figurative language is FRESH and INVENTED for this explanation → METAPHOR
   
   Only score 10 for ACTIVE metaphors that create fresh figurative meaning.
   Return 0 for idioms, explicit analogies, weak personification, or technical terms. 
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v5: Better handling of creative personification
metaphor_metric_explicit_v5 = GEval(
    name="Metaphor Explicit (v5)",
    evaluation_steps=[
        """1. A METAPHOR is NOVEL figurative language that creates fresh imagery, NOT standard expressions.
   
   CLEAR METAPHORS (score 10) - creative/unexpected language:
   - Extended creative metaphors: "rogue proteins throwing a tantrum inside your cells"
   - Fresh framing: "Your DNA is a recipe book written in a 4-letter alphabet"
   - CREATIVE PERSONIFICATION (vivid, unexpected actions): "electrons waltzing around the nucleus", "molecules whispering secrets", "antibodies going to war", "a tsunami of medication swept through"
   - Novel imagery: "The star is a cosmic pressure cooker slowly boiling itself"
   
   Creative personification IS a metaphor when the action/description is vivid, humorous, or unexpected.""",
        """2. NOT metaphors (exclude all of these):
   
   - SIMILES/ANALOGIES: comparisons with "like", "similar to", "imagine", "think of it as"
   - IDIOMS: frozen phrases recognized as common sayings ("tip of the iceberg", "here be dragons", "play mind tricks")
   - METONYMY: organizations standing for people ("The USA wants", "Science says")
   - STANDARD PERSONIFICATION: common/unremarkable humanizing phrases ("cats figured out", "the algorithm decides", "the market fears", "evolution favors")
   - SHAPE DESCRIPTIONS: literal visual resemblance ("potato-shaped asteroid", "mushroom cloud")
   - TECHNICAL TERMS & DEAD METAPHORS: textbook analogies or standardized vocabulary ("the cell is a factory", "DNA is a blueprint", "genetic code")
   - LITERAL STATEMENTS: "The heart pumps blood" """,
        """3. KEY DECISION RULE - PERSONIFICATION TEST:
   
   For personification specifically, ask: "Is this VIVID/CREATIVE or STANDARD/UNREMARKABLE?"
   - "The market fears recession" → STANDARD (everyone says this) → NOT metaphor
   - "White blood cells throwing punches at invaders" → CREATIVE/VIVID → METAPHOR
   
   General rule: Is the figurative language FRESH and INVENTED, or would a native speaker recognize it as everyday language?
   
   Score 10 for active metaphors with fresh figurative meaning. Return 0 otherwise.
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)


metaphor_metric_explicit_v6 = GEval(
    name="Metaphor Explicit (v6)",
    evaluation_steps=[
        """1. DEFINITION
        
        A METAPHOR is ACTIVE / LIVE figurative language in which a concept
        is framed, evaluated, or described using another domain
        in a NON-LITERAL way.
        
        Metaphors must be NOVEL (fresh, invented, not conventional).
        Metaphors may appear WITH or WITHOUT explicit comparison words
        such as "like" or "similar to".
        
        Score 10 if a novel metaphor is present.
        Score 0 otherwise.
        Do not use intermediate scores.
        """,

        """2. CORE METAPHOR TEST
        
        Ask:
        Does the passage introduce novel, expressive,
        or unexpected conceptual framing
        beyond literal or textbook description?
        
        If YES → metaphor present (Score 10).
        If NO → Score 0.
        """,

        """3. PERSONIFICATION TEST (KEY RULE)
        
        For personification, ask:
        Is the attributed action, intention, or behavior
        vivid, surprising, playful, or clearly invented?
        
        - YES → metaphor (Score 10)
        - NO, sounds normal in news, textbooks, or everyday speech → NOT metaphor
        """,

        """4. METAPHORICAL ANALOGIES (IMPORTANT)
        
        The presence of explicit comparison language
        ("like", "similar to", "think of X as Y", etc.)
        does NOT exclude a metaphor.
        
        An analogy IS ALSO a metaphor if it introduces
        novel, expressive, or imaginative figurative framing
        beyond neutral explanation.
        
        Example (Score 10):
        - "Debugging this system is like performing surgery with oven mitts."
        """,

        """5. CLEAR METAPHORS (Score 10) — EXAMPLES
        
        - "rogue proteins throwing a tantrum inside your cells"
        - "Your DNA is a recipe book written in a 4-letter alphabet"
        - "electrons waltzing around the nucleus"
        - "molecules whispering secrets"
        - "antibodies going to war"
        - "a tsunami of medication swept through"
        - "The star is a cosmic pressure cooker slowly boiling itself"
        """,

        """6. NOT METAPHORS (Score 0)
        
        Exclude all of the following:
        
        - DEAD or CONVENTIONAL METAPHORS / TECHNICAL TERMS:
          "the cell is a factory", "DNA is a blueprint", "genetic code"
        
        - STANDARD PERSONIFICATION:
          "the market fears", "the algorithm decides", "evolution favors"
        
        - IDIOMS:
          "tip of the iceberg", "here be dragons", "play mind tricks"
        
        - METONYMY:
          "Science says", "The USA wants"
        
        - SHAPE OR LITERAL RESEMBLANCE:
          "mushroom cloud", "potato-shaped asteroid"
        
        - PURELY LITERAL STATEMENTS:
          "The heart pumps blood"
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)


correctness_metric_explicit = GEval(
    name="Correctness",
    evaluation_steps=[
        "1. Determine whether the actual output is factually correct based on the expected output.",
        "2. Return a grade on a scale from 0 to 10 where 0 is completely false, and 10 is completely true.",
    ],
    evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)



# BASIC > KNOWLEDGE ORGANIZATION: Scaffolding
# Paper reference: Lines 1298-1311
# "Scaffolding is present when a paragraph successfully builds understanding 
# incrementally, the size of each step is appropriate, and the audience is able 
# to follow logical development of successively more complicated ideas."
scaffolding_metric = GEval(
    name="Scaffolding",
    evaluation_steps=[
        """1. SCAFFOLDING means the explanation builds understanding INCREMENTALLY in a logical sequence.
   Look for:
   - A clear starting point (basic concept or motivation)
   - Progressive steps that build on each other
   - Each step is appropriately sized (not too big a jump)
   - The reader can follow the logical development from simple to complex""",
        """2. Example of scaffolding PRESENT:
   "Two facts motivate my research—first, diverse systems are healthier systems and second, 
   humans are rapidly altering diversity around the globe... My research asks if it matters 
   that species are being gained and lost rapidly from these communities... By manipulating 
   plant number and type (i.e., native plants versus exotic plants) in each community, I can 
   begin to uncover the mechanisms underlying the superior performance of diverse systems."
   → Starts with motivation (2 facts) → poses research question → describes method
   
   Example of scaffolding ABSENT:
   "My research uses plant manipulation experiments to study biodiversity mechanisms in communities 
   where species turnover affects ecosystem health through alterations in system diversity."
   → All concepts presented at once, no progressive building""",
        """3. Return a score of 10 if the explanation demonstrates clear scaffolding (incremental building).
   Return 0 if ideas are presented all at once or in a disorganized manner.
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

### Deprecated Metrics ###

# analogy_metric = GEval(
#     name="Analogy",
#     criteria="""Analogies are defined as a systematic mapping between two situations:
# the source (familiar situation) and the target (novel situation). Determine whether the explanation includes analogies.""",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
# )
#
#
# metaphor_metric = GEval(
#     name="Metaphor",
#     criteria="""Metaphors structure one concept in terms of another. Unlike
# analogies, metaphors do not necessarily map directly between source and
# target; similarities can be associative. Determine whether the explanation includes metaphors or not. Do not take correctness into account.""",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
# )
#
# correctness_metric = GEval(
#     name="Correctness",
#     criteria="Determine whether the actual output is factually correct based on the expected output.",
#     evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
# )
#
# content_units_metric = GEval(
#     name="Content Units",
#     criteria="""A "content unit" is defined as any standalone
# fact. For example, the sentence "Two facts motivate my research—first, diverse systems are healthier
# systems, and second, humans are rapidly altering diversity around the globe"
# would be coded as having two content units. Return the amount of content units in the answer.""",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
# )
#
# humor_metric = GEval(
#     name="Humor",
#     criteria="The explanation includes explicit jokes or ironic language.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
# )