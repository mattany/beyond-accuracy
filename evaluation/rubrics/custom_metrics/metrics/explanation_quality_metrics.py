from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evaluation.rubrics.custom_metrics.metrics.constants import g_eval_default_params, g_eval_deepseek_params

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

# v3: Focus on NAMED cultural references, exclude generic analogies
# Key insight: Analogies using generic concepts are NOT connections to everyday life
# But NAMED references to specific cultural entities ARE connections
connection_to_everyday_life_metric_explicit_v3 = GEval(
    name="Connection to everyday life (v3)",
    evaluation_steps=[
        """1. Check if the answer contains a NAMED reference from popular culture.
   
   SCORE 10 - Any of the following:
   - Named TV shows, movies, books, documentaries (e.g., "Friends", "Game of Thrones", "The Matrix")
   - Named magazines, newspapers, publications (e.g., "Time Magazine", "Vogue", "BBC")
   - Named historical figures commonly known (e.g., "Napoleon", "Julius Caesar", "Cleopatra")
   - Named historical events (e.g., "the French Revolution", "the Renaissance")
   - Named brands, products, companies (e.g., "Google", "Netflix", "Amazon")
   - Named games (e.g., "Monopoly", "Scrabble", "basketball")
   - Pop culture jokes or memes (e.g., references to viral internet moments)
   - Famous landmarks (e.g., "crowded like Times Square", "tall as the Eiffel Tower")""",

        """2. ALSO SCORE 10: Concrete everyday activities that people commonly do.
   
   These are relatable human experiences that ground abstract concepts in daily life.
   The activity must be something a general audience does or understands from personal experience.
   
   SCORE 10 EXAMPLES:
   - Household tasks: "like doing laundry", "similar to washing dishes", "like vacuuming"
   - Errands: "like mailing a letter", "similar to shopping for groceries", "like waiting at the DMV"
   - Food & cooking: "like mixing ingredients for a cake", "similar to marinating meat", "like seasoning a soup"
   - Social activities: "like hosting a dinner party", "similar to playing board games with friends"
   - Common experiences: "like getting stuck in traffic", "similar to waiting for your coffee to brew"
   - Work/school: "like organizing your desk", "similar to studying for an exam"
   
   KEY: The activity must be EXTERNAL to the scientific topic being explained.
   It should make the reader think "Oh, I know what that's like!" """,
        
        """3. SCORE 0 - The following do NOT count:
   
   ABSTRACT ANALOGIES (too generic):
   - Natural phenomena: "like a river flowing", "like waves in the ocean", "like a pebble dropping"
   - Abstract physics: "like a ball rolling downhill", "like a spring bouncing"
   - Generic objects: "like a sponge absorbing", "like a balloon inflating"
   
   ALSO SCORE 0:
   - Scientific terminology (e.g., "mitochondria", "quantum entanglement", "oxidation")
   - Technical concepts without cultural grounding (e.g., "the process involves hydrolysis")
   - Vague references (e.g., "as we all know", "in everyday life", "people often")
   - References already embedded in the original question
   - Geographic/astronomical names used technically (e.g., "the Pacific plate", "the Andromeda galaxy")
   - Answers with no cultural or everyday life references at all""",

        """4. IMPORTANT: Everyday content directly relevant to the question topic does NOT count as a connection.
   
   A "connection to everyday life" must bring in something EXTERNAL to the topic being discussed.
   If the everyday reference is just part of answering the question, it's not a connection.
   
   SCORE 0 EXAMPLES (content is directly relevant to question):
   - Question about babies → mentioning "breastfeeding" or "diapers" (directly relevant, not a connection)
   - Question about cooking → mentioning "your kitchen stove" (directly relevant, not a connection)
   - Question about cars → mentioning "your gas tank" (directly relevant, not a connection)
   - Question about sleep → mentioning "your bed" or "pajamas" (directly relevant, not a connection)
   
   SCORE 10 EXAMPLES (content brings in external reference):
   - Question about physics → mentioning "like a scene from Interstellar" (external movie reference)
   - Question about biology → mentioning "like that episode of House M.D." (external TV reference)
   - Question about chemistry → mentioning "like mixing ingredients for a cake" (external everyday activity)""",
        
        """5. Return score 10 if ANY item from steps 1-2 is present AND it passes the test in step 4.
   Return score 0 otherwise. Do not use intermediate scores.
   
   FINAL EXAMPLES:
   - "like a factory assembly line" → Score 0 (generic analogy)
   - "like a scene from Jurassic Park" → Score 10 (named movie, external to topic)
   - "like mailing a package" → Score 10 (concrete everyday activity, external to topic)
   - "like water flowing downhill" → Score 0 (abstract nature analogy)"""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)


# v4: More permissive on named references - historical figures, organizations, publications all count
# Key fixes:
# 1. Historical figures (Galileo, Napoleon) ARE valid named references
# 2. Organizations (NASA, FCC) ARE valid named references  
# 3. Publications (Psychology Today, The Economist) ARE valid named references
# 4. Famous weapons/objects (Tsar Bomba) ARE valid named references
# 5. BUT: Reference in a question ABOUT that topic doesn't count (e.g., Jurassic Park in a dinosaur question that asks about JP)
connection_to_everyday_life_metric_explicit_v4 = GEval(
    name="Connection to everyday life (v4)",
    evaluation_steps=[
        """STEP 1: Check for ANY named reference from the following categories.
   
   SCORE 10 for ANY of these:
   
   A) POPULAR CULTURE:
   - Movies, TV shows, books, documentaries (e.g., "Jurassic Park", "Game of Thrones", "The Matrix")
   - Music, bands, songs (e.g., "like a Beatles song", "Beethoven's 5th")
   - Video games, board games (e.g., "Monopoly", "Minecraft", "chess")
   - Internet culture, memes, viral moments
   - Fictional characters (e.g., "Batman", "Sherlock Holmes")
   
   B) HISTORICAL FIGURES AND EVENTS:
   - Famous scientists, leaders, artists (e.g., "Galileo", "Einstein", "Napoleon", "Carl Sagan")
   - Historical events (e.g., "the French Revolution", "World War II", "the Cold War")
   - Famous inventions or discoveries (e.g., "the Tsar Bomba", "the Manhattan Project")
   
   C) ORGANIZATIONS AND BRANDS:
   - Companies, brands, products (e.g., "Google", "Netflix", "American Apparel", "Teflon")
   - Government agencies (e.g., "NASA", "FBI", "FCC")
   - Well-known institutions (e.g., "Harvard", "the Smithsonian")
   
   D) PUBLICATIONS AND MEDIA:
   - Magazines, newspapers (e.g., "The Economist", "Psychology Today", "Time Magazine")
   - Websites, platforms (e.g., "Yahoo Answers", "Wikipedia", "Reddit")
   
   E) FAMOUS PLACES:
   - Landmarks (e.g., "Times Square", "the Eiffel Tower", "Chernobyl")
   - Well-known locations used as reference points""",

        """STEP 2: ALSO check for concrete everyday activities.
   
   SCORE 10 for relatable human experiences that ground abstract concepts:
   - Household tasks: "like doing laundry", "like vacuuming"
   - Food & cooking: "like mixing cake batter", "like grocery shopping"  
   - Common experiences: "like getting stuck in traffic", "like waiting for coffee"
   - Childhood activities: "like a child spinning in a circle with arms outstretched"
   - Social activities: "like hosting a dinner party"
   
   KEY: The activity must be something a general audience experiences personally.""",

        """STEP 3: CHECK THE "EXTERNAL TO TOPIC" RULE.
   
   IMPORTANT: The reference must be EXTERNAL to the question topic.
   
   SCORE 0 if the reference is ALREADY PART of the question:
   - Question asks "how has our knowledge of dinosaurs changed since Jurassic Park" 
     → Mentioning "Jurassic Park" in the answer does NOT count (it's part of the question)
   - Question asks about NASA missions → Mentioning "NASA" does NOT count
   - Question asks about Einstein's theories → Mentioning "Einstein" does NOT count
   
   SCORE 10 if the reference is EXTERNAL:
   - Question about physics → Mentioning "Galileo" DOES count (external historical figure)
   - Question about nuclear reactors → Mentioning "the Cold War" or "Tsar Bomba" DOES count
   - Question about space → Mentioning "NASA" DOES count (if NASA wasn't in the question)""",
   
        """STEP 4: What does NOT count (Score 0).
   
   - Generic physics analogies: "like a ball rolling downhill", "like waves in water"
   - Nature analogies: "like a river flowing", "like clouds forming"
   - Abstract comparisons: "like a sponge absorbing", "like a balloon"
   - Vague references: "as we all know", "in everyday life"
   - Technical jargon without cultural grounding
   - References already embedded in the original question (see Step 3)""",

        """STEP 5: FINAL DECISION.
   
   Score 10 if:
   - You found ANY named reference from Step 1 categories, AND
   - The reference is EXTERNAL to the question topic (passes Step 3)
   
   OR:
   - You found a concrete everyday activity from Step 2, AND
   - The activity is EXTERNAL to the question topic
   
   Score 0 otherwise.
   
   EXAMPLES:
   - "Galileo discovered the moons of Jupiter" in a question about CERN → Score 10 (Galileo is external)
   - "the Tsar Bomba" in a question about killing stars → Score 10 (famous Cold War bomb, external)
   - "Psychology Today" in any answer → Score 10 (named publication)
   - "NASA" in a question NOT about NASA → Score 10 (famous organization)
   - "like a child spinning" in a physics question → Score 10 (relatable everyday activity)
   - "Jurassic Park" in a question asking about Jurassic Park → Score 0 (NOT external)
   
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
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

# v3: Calibrated version with borderline examples to reduce ambiguity
# Addresses issues identified in human-model disagreement analysis:
# - Absurdist analogies that ARE humorous
# - Parenthetical snark/wry asides
# - Ironic understatement without explicit markers
# - Dark humor through violent/extreme imagery
humor_metric_explicit_v3 = GEval(
    name="Humor Explicit (v3)",
    evaluation_steps=[
        """1. HUMOR includes explicit jokes AND ironic language.
   
   Look for ANY of the following:
   - Explicit jokes (with or without a formal punchline)
   - Puns or wordplay meant to amuse
   - Ironic language: understatement, tongue-in-cheek remarks, or wry observations
   - Absurdist comparisons that highlight ridiculousness through exaggeration or juxtaposition
   - Parenthetical snark or dismissive/wry asides
   - Dark humor using violent or extreme imagery for comic effect
   
   CALIBRATION EXAMPLES - These ARE humor (Score 10):
   - "It would be like saying you've falsified the round earth hypothesis by driving on a flat road" 
     (absurdist analogy - the ridiculousness IS the joke)
   - "U-238 (mostly useless)" 
     (parenthetical snark - dismissive aside meant to amuse)
   - "That's it, really. The whole process is about five undergrad courses worth of material" 
     (ironic understatement - trivializing complexity)
   - "...it can be used to build something or to cave someone's head in" 
     (dark humor - violent imagery for ironic effect)
   - "I hope it tastes and smells like bacon" (about nuclear annihilation) 
     (dark humor - absurd juxtaposition)
   - "zombie-proteins, you need to go for the head-shot, and remember to double tap" 
     (pop culture joke reference)
   - "I'm guessing your crowd has read too much Twilight" 
     (sarcastic/ironic jab)""",
        """2. The following are NOT humor:
   - Creative or vivid analogies/metaphors without irony or absurdity 
     (e.g., "DNA is like a blueprint", "electrons flow like water")
   - Engaging or enthusiastic tone without jokes or irony
   - Personification without irony (e.g., "the virus wants to replicate")
   - Vivid or dramatic descriptions that lack comedic intent
   - Playful language that lacks actual jokes or ironic statements
   
   CALIBRATION EXAMPLES - These are NOT humor (Score 0):
   - "like a pebble rolling down a hill" (straightforward analogy, no absurdity)
   - "bacteria are like tiny factories" (vivid but not ironic)
   - "the immune system is an army defending your body" (metaphor without comedic intent)""",
        """3. KEY QUESTION: Would a reasonable reader find this amusing or witty?
   
   Consider: Does the text contain intentional wit, irony, or comedic effect?
   - Absurdist comparisons that make you think "that's ridiculous" = humor
   - Parenthetical asides that seem snarky or dismissive = humor  
   - Understatement after buildup that seems intentionally anticlimactic = humor
   - Dark/violent imagery used for shock-humor effect = humor
   
   If yes → return 10.
   If no (even if creative, vivid, or playful) → return 0.
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v4: Simplified vocabulary version of v3
# - Replaced sophisticated words (wry, parenthetical, juxtaposition, absurdist)
# - Split into clearer numbered steps
# - Uses examples NOT from the evaluation dataset to avoid data leakage
humor_metric_explicit_v4 = GEval(
    name="Humor Explicit (v4)",
    evaluation_steps=[
        """STEP 1: Look for JOKES.

Does the text contain any of these?
- A joke with a punchline
- A pun or wordplay  
- A reference to movies, TV, or pop culture used to be funny
- A sarcastic or mocking comment

EXAMPLES of jokes (Score 10):
- "Schrodinger's cat walks into a bar... and doesn't" (joke about superposition in quantum physics)
- "Pluto was demoted to dwarf planet - talk about a mid-life crisis" (personification joke)
- "If you think nobody cares about you, try missing a few tax payments" (sarcastic observation)

If you found a joke → Score 10. Otherwise continue to Step 2.""",

        """STEP 2: Look for IRONY or SARCASM.

Does the text contain any of these?
- Saying the opposite of what you mean to be funny
- Making something serious sound unimportant on purpose (understatement)
- A side comment in brackets or parentheses that is dismissive or mocking
- Colorful or silly imagery used in a serious context (the contrast is funny)
- Dark humor: violent, morbid, or extreme language used casually for shock value

EXAMPLES of irony (Score 10):
- "The surgery went well, apart from the patient dying" (dark understatement)
- "It's perfectly safe, unless you count the possibility of horrible death" (dark humor - casual mention of death)
- "Nature is beautiful, if you ignore all the things trying to eat you" (dark humor - unexpected violent twist)
- "String theory (which nobody actually understands)" (dismissive side comment)
- "You'd basically become a very expensive puddle" about extreme pressure (silly image + serious topic)

If you found irony or sarcasm → Score 10. Otherwise continue to Step 3.""",

        """STEP 3: Look for RIDICULOUS comparisons.

Does the text make a comparison that is SO silly or extreme that it's meant to be funny?
The humor comes from the absurdity - the comparison is intentionally too extreme or too silly.

EXAMPLES of ridiculous comparisons (Score 10):
- "Saying magnets work by magic is like saying cars run on wishes" (exaggerated comparison)
- "Expecting your immune system to fight off everything is like expecting one security guard to defend a shopping mall from an army" (absurd scale mismatch)
- "The odds are about the same as finding a specific grain of sand on every beach on Earth" (comically extreme)

If you found a ridiculous comparison meant to amuse → Score 10. Otherwise continue to Step 4.""",

        """STEP 4: Check it's NOT just vivid language.

The following are NOT humor - they are just good explanations:
- Normal comparisons: "DNA is like a blueprint", "electricity flows like water"
- Lively descriptions: "neurons firing like a symphony" 
- Personification: "the virus hijacks the cell"
- Enthusiastic tone without actual jokes

FINAL DECISION:
- If you found a joke, irony, sarcasm, or ridiculous comparison in Steps 1-3 → Score 10
- If you only found vivid language or good explanations → Score 0

Do not use scores between 0 and 10."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)


# v5: More conservative - distinguishes intentional humor from casual/informal tone
# Key fix: Casual language, candid remarks, and informal tone are NOT humor
# Also catches absurdist/deadpan humor (unexpected non-sequiturs)
humor_metric_explicit_v5 = GEval(
    name="Humor Explicit (v5)",
    evaluation_steps=[
        """STEP 1: Look for INTENTIONAL JOKES.

Does the text contain any of these INTENTIONAL comedic elements?
- A joke with a clear punchline or comedic setup
- A pun or deliberate wordplay
- A pop culture reference used for comedic effect (not just as an example)
- A clearly sarcastic or mocking statement that subverts expectations

EXAMPLES of intentional jokes (Score 10):
- "Schrodinger's cat walks into a bar... and doesn't" (punchline joke)
- "holy proofreading batman" (pop culture catchphrase used for comedic effect)
- "you would not pass go, nor would you collect $200, you would go directly to nobel prize" (Monopoly reference twisted for humor)

If you found an intentional joke → Score 10. Otherwise continue to Step 2.""",

        """STEP 2: Look for ABSURDIST or DEADPAN humor.

Does the text contain any of these?
- An unexpected non-sequitur at the end that breaks the serious tone
- A brand name, product, or random word dropped without explanation for comedic effect
- A jarring tonal shift that seems intentionally funny
- Self-deprecating or self-aware comments that acknowledge absurdity

EXAMPLES of absurdist/deadpan humor (Score 10):
- A serious scientific answer that ends with "...and partially because **american apparel**" (non-sequitur brand drop)
- "I'd be lying if I said I've seen it" in a technical explanation (self-aware honesty)
- "at which point you have big problems^(tm)" (fake trademark = comedic aside)
- "now, i must go masturbate before the shakes set in" (absurd self-deprecating callback)

If you found absurdist or deadpan humor → Score 10. Otherwise continue to Step 3.""",

        """STEP 3: Look for IRONY, SARCASM, or DARK HUMOR.

Does the text contain any of these?
- Saying the opposite of what you mean with clear comedic intent
- Dark understatement: making something serious sound trivially unimportant
- Violent, morbid, or extreme language used casually for shock-comedy value

EXAMPLES of irony/dark humor (Score 10):
- "The surgery went well, apart from the patient dying" (dark understatement)
- "It's perfectly safe, unless you count the possibility of horrible death" (dark humor)
- "telescopes were played out" (ironic dismissal of something obviously important)

If you found clear irony or dark humor → Score 10. Otherwise continue to Step 4.""",

        """STEP 4: CRITICAL - Distinguish HUMOR from CASUAL TONE.

The following are NOT humor - they are just informal/casual writing:
- Candid admissions: "I would recommend staying away from Yahoo Answers" (just honest advice)
- Casual language: "[pretty good summary here]" (informal tone, not sarcasm)
- Enthusiastic phrases: "great question!" (friendly, not funny)
- Mild exaggeration without comedic intent: "scrounge up enough firepower" (vivid language)
- Informal asides that don't subvert expectations: "which is why it's not used much on earth"
- Rhetorical questions: "good question" (engaging, not joking)

KEY TEST: Would a reader actually laugh or smile? 
- Casual/informal tone = reader thinks "that's friendly" → Score 0
- Intentional humor = reader thinks "that's funny" → Score 10

FINAL DECISION:
- If you found intentional humor in Steps 1-3 that passes the laugh test → Score 10
- If you only found casual tone, informal language, or vivid explanations → Score 0

Do not use scores between 0 and 10."""
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

# **v6 clarifies and formalizes the overlap between metaphors and analogies**, allowing explicit comparisons
# (e.g., “like”) to be metaphorical when they introduce novel, expressive framing, while keeping strict binary scoring.
# It tightens alignment with standard definitions while making the operational boundaries clearer and more
# robust for annotation.
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

# v7 tightens metaphor detection by explicitly excluding idioms and other frozen expressions while preserving
# human-accepted abstract metaphors, reducing both false positives and false negatives. It also adds clearer decision
# tests and a conservative tie-breaking rule to better align LLM judgments with human annotation behavior.
metaphor_metric_explicit_v7 = GEval(
    name="Metaphor Explicit (v7)",
    evaluation_steps=[
        """1. DEFINITION
        
        A METAPHOR is ACTIVE / LIVE figurative language in which a concept
        is framed, evaluated, or described using another domain
        in a NON-LITERAL way.
        
        For this task, ONLY score ACTIVE (NON-FROZEN) metaphors.
        Conventional, idiomatic, or dead metaphors must be scored 0.
        
        Metaphors may be imagistic OR abstract.
        Metaphors may appear WITH or WITHOUT explicit comparison words
        such as "like", "as", or "similar to".
        
        Score 10 if an active metaphor is present.
        Score 0 otherwise.
        Do not use intermediate scores.
        """,

        """2. CORE METAPHOR TEST
        
        Ask:
        Does the passage introduce a NON-LITERAL conceptual framing
        that changes how the concept is understood, evaluated,
        or interpreted?
        
        If YES → metaphor present (Score 10).
        If NO → Score 0.
        """,

        """3. CLEAR METAPHORS (Score 10) — EXAMPLES
        
        These are examples of ACTIVE / LIVE metaphors:
        
        - "rogue proteins throwing a tantrum inside your cells"
        - "Your DNA is a recipe book written in a 4-letter alphabet"
        - "electrons waltzing around the nucleus"
        - "molecules whispering secrets"
        - "antibodies going to war"
        - "a tsunami of medication swept through"
        - "The star is a cosmic pressure cooker slowly boiling itself"
        
        These examples show novel, non-literal conceptual framing.
        """,

        """4. NOVELTY & FROZENNESS TEST (CRITICAL)
        
        Before scoring a metaphor, ask:
        
        “Would this phrasing be recognized by most native speakers
        as a common, idiomatic, or stock expression?”
        
        If YES → NOT a metaphor (Score 0),
        even if the phrase is figurative in origin.
        
        Examples of expressions that should be treated as FROZEN:
        - common idioms ("tip of the iceberg")
        - clichés and high-frequency figurative phrases
        - textbook metaphors treated as terminology
        """,

        """5. LITERAL FALLBACK TEST
        
        Ask:
        
        “Can this sentence be interpreted fully literally,
        with no loss of meaning or intent?”
        
        If YES → NOT a metaphor (Score 0).
        
        Metaphors require meaning that DEPENDS on
        non-literal conceptual transfer.
        """,

        """6. ABSTRACT METAPHORS (ALLOWED, WITH EXAMPLES)
        
        Metaphors do NOT need to be visual or imagistic.
        
        Abstract metaphors SHOULD be scored 10 if:
        - a non-literal domain is clearly imported
        - the framing alters how the concept is understood
        - the meaning cannot be preserved under a literal reading
        
        Examples (Score 10):
        - "This argument collapses under its own ambition."
        - "Ideas gain momentum as they spread."
        - "The theory rests on a fragile foundation."
        """,

        """7. PERSONIFICATION TEST
        
        For personification, ask:
        
        “Is the attributed action, intention, or behavior
        clearly impossible or unexpected for the domain?”
        
        - YES → metaphor (Score 10)
        - NO, sounds like standard descriptive language → NOT metaphor
        
        Examples:
        - "White blood cells throwing punches at invaders" → metaphor
        - "The market fears inflation" → NOT metaphor
        """,

        """8. METAPHORICAL ANALOGIES
        
        The presence of explicit comparison language
        ("like", "similar to", "think of X as Y", etc.)
        does NOT exclude a metaphor.
        
        An analogy is ALSO a metaphor if it introduces
        novel, expressive, or non-standard figurative framing,
        rather than neutral explanation.
        
        Example (Score 10):
        - "Debugging this problem is like trying to untangle headphones in the dark."
        """,

        """9. NOT METAPHORS (ALWAYS SCORE 0)
        
        Exclude all of the following:
        
        - DEAD or CONVENTIONAL METAPHORS / TECHNICAL TERMS:
          "the cell is a factory", "genetic code"
        
        - IDIOMS and FROZEN PHRASES:
          "tip of the iceberg", "here be dragons"
        
        - STANDARD PERSONIFICATION:
          "the algorithm decides", "evolution favors traits"
        
        - METONYMY:
          "Science says", "The White House announced"
        
        - SHAPE OR LITERAL RESEMBLANCE:
          "mushroom cloud", "potato-shaped asteroid"
        
        - PURELY LITERAL STATEMENTS:
          "The heart pumps blood"
        """,

        """10. CONSERVATIVE TIE-BREAKING RULE
        
        When the case is borderline or ambiguous:
        
        - Prefer Score 0 over Score 10.
        - Humans apply a conservative standard;
          match that standard here.
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v8 further refines domain-shift logic, clarifies frozen idiom handling,
# and adds richer examples throughout to reduce ambiguity in borderline cases.
metaphor_metric_explicit_v8 = GEval(
    name="Metaphor Explicit (v8)",
    evaluation_steps=[
        """1. DEFINITION
        
        A METAPHOR is ACTIVE / LIVE figurative language in which a concept
        is framed, evaluated, or described using another domain
        in a NON-LITERAL way.
        
        For this task, ONLY score ACTIVE metaphors.
        Conventional, idiomatic, or dead metaphors must be scored 0,
        UNLESS they are reactivated via a clear domain shift.
        
        Metaphors may be imagistic OR abstract.
        Metaphors may appear WITH or WITHOUT explicit comparison words
        such as "like", "as", or "similar to".
        
        Score 10 if an active metaphor is present.
        Score 0 otherwise.
        Do not use intermediate scores.
        """,

        """2. CORE METAPHOR TEST (WITH EXAMPLES)
        
        Ask:
        Does the passage DEPEND on a NON-LITERAL conceptual framing
        that changes how the target concept is understood,
        explained, or evaluated?
        
        If YES → continue evaluation.
        If NO → NOT a metaphor (Score 0).
        
        Examples:
        - "The theory cannibalizes its own assumptions." → metaphor
        - "The theory has internal inconsistencies." → NOT metaphor
        """,

        """3. NOVELTY & DOMAIN-SHIFT TEST (CRITICAL, WITH EXAMPLES)
        
        Novelty does NOT require original wording.
        
        A metaphor may be ACTIVE if novelty arises from:
        - the phrasing itself, OR
        - a DOMAIN SHIFT, where a familiar expression is applied
          to a domain where it is not conventionally used.
        
        HOWEVER:
        - If an expression is idiomatic and used freely across domains,
          it should be treated as FROZEN and scored 0.
        
        Positive examples (Score 10):
        - "She put him in the friend zone so hard that a force field
           seemed to form around him."
          (physics → social interaction)
        
        Negative examples (Score 0):
        - "That policy change was the final nail in the coffin."
          (domain-agnostic idiom)
        """,

        """4. FROZEN / IDIOMATIC METAPHORS (WITH EXAMPLES)
        
        Treat expressions as FROZEN (Score 0) if they are:
        - idiomatic and widely recognized
        - processed without invoking their source domain
        - commonly used across many topics
        
        Examples (Score 0):
        - "tip of the iceberg"
        - "double-edged sword"
        - "final nail in the coffin"
        
        These do NOT become active merely by appearing in
        technical, scientific, or abstract contexts.
        """,

        """5. LITERAL FALLBACK TEST (WITH EXAMPLES)
        
        Ask:
        Can the sentence be interpreted fully literally,
        with no loss of meaning or intent?
        
        If YES → NOT a metaphor (Score 0).
        
        Examples:
        - "The committee rejected the proposal." → NOT metaphor
        - "The proposal was smothered before it could take shape." → metaphor
        """,

        """6. ABSTRACT METAPHORS (ALLOWED, WITH EXAMPLES)
        
        Metaphors do NOT need to be visual or imagistic.
        
        Abstract metaphors SHOULD be scored 10 if:
        - a non-literal source domain is imported
        - the framing alters how the concept is understood
        - the meaning cannot be preserved under a literal reading
        
        Examples (Score 10):
        - "The policy devours the resources it was meant to protect."
        - "The algorithm starves certain outcomes of attention."
        
        Abstract, academic, or technical language alone
        does NOT qualify as metaphor.
        """,

        """7. PERSONIFICATION TEST (WITH EXAMPLES)
        
        For personification, ask:
        Is the attributed action, intention, or behavior
        clearly impossible or unexpected for the domain?
        
        If YES → metaphor (Score 10).
        If NO → NOT metaphor (Score 0).
        
        Examples:
        - "White blood cells ambushing invaders." → metaphor
        - "The market responded to new data." → NOT metaphor
        """,

        """8. METAPHORICAL ANALOGIES (WITH EXAMPLES)
        
        Explicit comparison language ("like", "similar to",
        "think of X as Y", etc.) does NOT exclude a metaphor.
        
        An analogy is ALSO a metaphor if it introduces
        novel, expressive, or domain-shifting figurative framing,
        rather than neutral explanation.
        
        Examples:
        - "Debugging this problem is like trying to untangle
           headphones while wearing gloves." → metaphor
        - "DNA is like a recipe book." → NOT metaphor
        """,

        """9. NOT METAPHORS (SUMMARY WITH EXAMPLES)
        
        Always score 0 for:
        - purely literal descriptions
        - standard technical terminology
        - metonymy (institutions standing for people)
        - shape or visual resemblance only
        - routine evaluative or descriptive verbs
        
        Examples (Score 0):
        - "The heart pumps blood."
        - "The White House announced new measures."
        - "a mushroom-shaped cloud"
        """,

        """10. CONSERVATIVE TIE-BREAKING RULE
        
        When the case is borderline or ambiguous:
        
        - Prefer Score 0 over Score 10.
        - Humans apply a conservative standard;
          match that standard here.
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v8_deepseek: Same as v8 but using deepseek-reasoner model (slow, chain-of-thought)
# Requires DEEPSEEK_API_KEY environment variable
from evaluation.rubrics.custom_metrics.metrics.deepseek_model import deepseek_reasoner, deepseek_chat
metaphor_metric_explicit_v8_deepseek = GEval(
    name="Metaphor Explicit (v8-deepseek)",
    evaluation_steps=metaphor_metric_explicit_v8.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_reasoner
)

# v8_deepseek_chat: Same as v8 but using deepseek-chat model (fast)
metaphor_metric_explicit_v8_deepseek_chat = GEval(
    name="Metaphor Explicit (v8-deepseek-chat)",
    evaluation_steps=metaphor_metric_explicit_v8.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_chat
)

# v8_gpt4o_mini: Same as v8 but using gpt-4o-mini (cheaper, faster)
metaphor_metric_explicit_v8_gpt4o_mini = GEval(
    name="Metaphor Explicit (v8-gpt4o-mini)",
    evaluation_steps=metaphor_metric_explicit_v8.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini"
)

# v9 is a streamlined version focusing on cognitive accessibility of source domains,
# clearer frozen/active distinction, and reduced prompt length for better consistency.
metaphor_metric_explicit_v9 = GEval(
    name="Metaphor Explicit (v9)",
    evaluation_steps=[
        """1. DEFINITION
        
        A METAPHOR is ACTIVE / LIVE figurative language in which a concept
        is framed using another domain in a NON-LITERAL way.
        
        Score 10 only for ACTIVE metaphors.
        Score 0 for frozen, idiomatic, or literal language.
        Do not use intermediate scores.
        """,

        """2. CORE TEST
        
        Ask:
        Does the sentence REQUIRE a non-literal source domain
        to convey its meaning?
        
        If NO → Score 0.
        If YES → continue.
        
        Examples:
        - "The proposal was smothered before it could take shape." → 10
        - "The proposal was rejected." → 0
        """,

        """3. NOVELTY & DOMAIN USE
        
        A metaphor may be ACTIVE if novelty comes from:
        - the phrasing itself, OR
        - applying a source domain to a target where it is not normally used.
        
        Novelty is about CONCEPTUAL REFRAMING,
        not surface creativity or topic change alone.
        """,

        """4. FROZEN METAPHORS (CRITICAL EXCLUSION)
        
        Treat an expression as FROZEN (Score 0) if:
        - it is widely used as a fixed label for an abstract property, AND
        - speakers do NOT cognitively access the source domain in normal use,
          even if the source can be reconstructed.
        
        This includes domain-agnostic idioms and fully lexicalized metaphors.
        
        Example (Score 0):
        - "That policy change was the final nail in the coffin."
        """,

        """5. DOMAIN-SHIFT REACTIVATION (ALLOWED)
        
        A familiar expression MAY be scored 10 if:
        - its source domain is still cognitively accessible, AND
        - it is applied to a target where that domain is atypical,
          creating clear ontological tension.
        
        Example (Score 10):
        - "She put him in the friend zone so hard
           that a force field seemed to form around him."
        """,

        """6. ABSTRACT & PERSONIFICATION CASES
        
        Metaphors need not be visual.
        Abstract or technical language may be metaphorical
        if it imports a non-literal source domain that does real work.
        
        Personification counts ONLY if the action would be impossible
        or clearly unexpected for the domain.
        
        Examples:
        - "The algorithm starves certain outcomes of attention." → 10
        - "The market reacted to the news." → 0
        """,

        """7. ANALOGIES
        
        Explicit comparisons ("like", "as", etc.) do NOT exclude metaphors.
        
        An analogy is ALSO a metaphor if it introduces
        expressive or domain-shifting framing,
        not just neutral explanation.
        
        Examples:
        - "Debugging this is like untangling headphones in the dark." → 10
        - "DNA is like a recipe book." → 0
        """,

        """8. TIE-BREAKING RULE
        
        If the case is borderline or uncertain:
        prefer Score 0.
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v10 is a minimal, highly condensed prompt focusing on the core active/frozen distinction
# with clear examples and conservative tie-breaking.
metaphor_metric_explicit_v10 = GEval(
    name="Metaphor Explicit (v10)",
    evaluation_steps=[
        """1. METAPHOR DEFINITION
        
        A METAPHOR is present ONLY if the sentence REQUIRES
        a NON-LITERAL source domain to convey its meaning.
        
        If the meaning is fully clear without invoking another domain,
        score 0.
        Otherwise, continue.
        """,

        """2. ACTIVE VS FROZEN TEST (CRITICAL)
        
        Score 10 ONLY if:
        - the source domain is cognitively accessed by speakers, AND
        - it creates clear conceptual tension with the target domain.
        
        Score 0 if the expression is a fixed, idiomatic label
        whose source domain is no longer accessed,
        even if it can be reconstructed.
        
        Domain change alone is NOT enough.
        """,

        """3. FINAL DECISION RULE (WITH EXAMPLES)
        
        Score 10 (metaphor):
        - "The policy devours the resources it was meant to protect."
        - "A force field seemed to form around him socially."
        
        Score 0 (not metaphor):
        - "That decision was the final nail in the coffin."
        - "The market reacted to the news."
        
        If uncertain, score 0.
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v11 balances detail and structure: includes necessity test, frozenness override rule,
# and clearer examples while maintaining moderate prompt length.
metaphor_metric_explicit_v11 = GEval(
    name="Metaphor Explicit (v11)",
    evaluation_steps=[
        """1. DEFINITION
        
        A METAPHOR is ACTIVE / LIVE figurative language in which a concept
        is framed, evaluated, or described using another domain
        in a NON-LITERAL way.
        
        Score 10 ONLY for ACTIVE metaphors.
        Score 0 for literal, idiomatic, or dead metaphors.
        
        Metaphors may be imagistic OR abstract.
        Metaphors may appear WITH or WITHOUT explicit comparison words
        such as "like", "as", or "similar to".
        
        Do not use intermediate scores.
        """,

        """2. NECESSITY TEST (PRIMARY GATE)
        
        Ask:
        Does the sentence REQUIRE importing a source domain
        (beyond literal meaning) to convey its point?
        
        If the meaning is fully preserved under a literal paraphrase,
        score 0 and STOP.
        
        Examples:
        - "The theory cannibalizes its own assumptions." → 10 (domain import)
        - "The theory has a fatal flaw." → 0 (fixed label)
        """,

        """3. FROZENNESS TEST (DECISIVE)
        
        Even if figurative in origin, score 0 if the expression:
        - functions as a fixed label for an abstract property, AND
        - is normally processed without accessing its source domain.
        
        These expressions do NOT become active merely by being applied
        to a new topic or technical domain.
        
        Examples (Score 0):
        - "final nail in the coffin" (idiomatic)
        - "double-edged sword" (lexicalized)
        - "tip of the iceberg" (semantic bleaching)
        """,

        """4. DOMAIN-SHIFT ACTIVATION (ALLOWED BUT RESTRICTED)
        
        A metaphor may be ACTIVE if:
        - the source domain is still cognitively accessed, AND
        - applying it to the target creates clear ontological tension
          (the target is not normally described this way).
        
        Domain change alone is NOT sufficient;
        the source domain must do conceptual work.
        
        Example (Score 10):
        - "She put him in the friend zone so hard that a force field
           seemed to form around him." (physics → social)
        """,

        """5. ABSTRACT & PERSONIFICATION CASES
        
        Metaphors need not be visual.
        
        Abstract metaphors count ONLY when the imported domain
        changes how the concept is reasoned about,
        not merely how it is described.
        
        Personification counts ONLY if the action would be
        impossible or clearly unexpected for the domain.
        
        Examples:
        - "The policy devours the resources it was meant to protect." → 10 (agent transfer)
        - "The algorithm starves certain outcomes of attention." → 10 (biological framing)
        - "The market reacted to the news." → 0 (standard usage)
        """,

        """6. ANALOGIES
        
        Explicit comparison language ("like", "as", etc.)
        does NOT exclude metaphors.
        
        An analogy is ALSO a metaphor ONLY if it adds expressive
        or domain-shifting framing beyond neutral explanation.
        
        Examples:
        - "Debugging this is like untangling headphones
           while wearing gloves." → 10 (expressive mapping)
        - "DNA is like a recipe book." → 0 (didactic analogy)
        """,

        """7. FINAL RULE
        
        If multiple rules conflict, the FROZENNESS TEST overrides
        all others.
        
        When uncertain, score 0.
        """
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

#
# v12: tighten v8 to reduce ambiguity across different evaluator models.
# Key changes vs v8:
# - Force a binary 0/1 score (not 0/10, not intermediate decimals).
# - Treat stock idioms + technicalized metaphor terms as Score 0.
# - Tighten analogy/personification rules: require mapped structure or vivid, central framing.
#
metaphor_metric_explicit_v12 = GEval(
    name="Metaphor Explicit (v12)",
    evaluation_steps=[
        """SCORING OUTPUT (MANDATORY)

Return ONLY a single number: 10 or 0.
- 10 = an ACTIVE metaphor is present
- 0  = no active metaphor is present

Do NOT output any other text.
Do NOT use intermediate scores.
""",

        """1. DEFINITION (WHAT COUNTS)

Score 10 ONLY if the text contains an ACTIVE metaphorical framing:
- A NON-LITERAL source domain is imported, AND
- it does real explanatory or evaluative work (not just a throwaway label), AND
- the meaning would change if rewritten literally.

Otherwise score 0.
""",

        """2. STRONG POSITIVE SIGNALS (Score 10)

Score 10 when at least one of the following is present:
- An explicit cross-domain scenario used to explain the target
  (e.g., comparing a mental process to a traffic jam caused by a poorly-timed lane closure)
- A vivid, specific simile/analogy with a concrete mechanism/scene
  (e.g., "like trying to push a rope uphill" or "like sandpaper on glass")
- Reactivated figurative language where the source domain is clearly invoked
  with extra source-domain detail (not just a fixed phrase).

Additional positive examples (Score 10):
- "Your experience of 'now' is like standing on a moving walkway that keeps carrying you forward." (experience domain → time)
- "The usual rules briefly take a coffee break here." (personification of abstractions)
""",

        """3. AUTOMATIC NEGATIVES (Score 0) — COMMON AMBIGUITIES

Score 0 if the only figurative language is any of the following:

- Stock idioms / clichés used as fixed labels:
  Examples (Score 0):
  - "a whole new ballgame"
  - "a different kettle of fish"
  - "the last straw"
  (Clichés used as fixed labels, not active framing.)

- Technicalized metaphor terms / domain jargon that function as labels:
  Examples (Score 0):
  - "bottleneck" used as a standard technical label with no vivid mapping
  - "pipeline" used as a standard engineering label

- Generic imagery that does not introduce mapped structure:
  Example (Score 0):
  - "The explanation was clear as day." (a generic vivid phrase; no cross-domain mapped structure)

- Historical models reported as history (not used as live framing):
  Example (Score 0):
  - "People once modeled heat as a weightless substance called 'caloric'."
    (If the text is merely reporting this historical view, not using it as live framing.)
""",

        """4. ANALOGY VS METAPHOR (STRICT RULE)

Score 0 for purely didactic label-mappings with no mapped structure:
- "Memory is like a hard drive." (generic teaching analogy)
- Calling someone “a robot” purely as a label (meaning emotionally flat/automatic), with no mapped scenario or source-domain structure.

Score 10 if the analogy includes specific source-domain roles/actions/constraints
that are mapped onto the target. Clear examples (Score 10):
- "Trying to do X while monitoring yourself is like driving with the parking brake slightly on: you can move, but every action fights resistance."
- "The system behaves like an assembly line where one stuck station backs up the entire belt, slowing everything downstream."
- "The components act like teammates pulling a rope in opposite directions, so progress stalls until one side yields."
""",

        """5. PERSONIFICATION (STRICT RULE)

Score 10 only if the personification is vivid AND central (it changes understanding),
typically with specific agentive actions that are impossible for the domain.

Score 0 for routine, conventional scientific shorthand
(e.g., \"cells attack\", \"the market responded\") unless extended into a concrete scene.
""",

        """6. TIE-BREAKER

If borderline or uncertain, score 0.
"""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    **g_eval_default_params
)

# v12 variants for other evaluator models (same prompt)
metaphor_metric_explicit_v12_deepseek = GEval(
    name="Metaphor Explicit (v12-deepseek)",
    evaluation_steps=metaphor_metric_explicit_v12.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_reasoner
)

metaphor_metric_explicit_v12_deepseek_chat = GEval(
    name="Metaphor Explicit (v12-deepseek-chat)",
    evaluation_steps=metaphor_metric_explicit_v12.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_chat
)

metaphor_metric_explicit_v12_gpt4o_mini = GEval(
    name="Metaphor Explicit (v12-gpt4o-mini)",
    evaluation_steps=metaphor_metric_explicit_v12.evaluation_steps,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini"
)

# v12.1: targeted tightening to remove the two biggest remaining instabilities in v12:
# - Single-clause “vibe” similes used as generic descriptors → Score 0 unless extended mapping.
# - Colloquial quantity/speed intensifiers → Score 0 unless extended mapping.
metaphor_metric_explicit_v12_1 = GEval(
    name="Metaphor Explicit (v12.1)",
    evaluation_steps=[
        """SCORING OUTPUT (MANDATORY)

Return ONLY a single number: 10 or 0.
- 10 = an ACTIVE metaphor is present
- 0  = no active metaphor is present

Do NOT output any other text.
Do NOT use intermediate scores.
""",

        """1. DEFINITION (WHAT COUNTS)

Score 10 ONLY if the text contains an ACTIVE metaphorical framing:
- A NON-LITERAL source domain is imported, AND
- it does real explanatory or evaluative work (not just a throwaway label), AND
- the meaning would change if rewritten literally.

Otherwise score 0.
""",

        """2. STRONG POSITIVE SIGNALS (Score 10)

Score 10 when at least one of the following is present:
- A cross-domain scenario with specific roles/actions/constraints mapped onto the target
  (e.g., an assembly line jam, a parking brake adding drag, etc.).
- A vivid simile/analogy that explains mechanism, not just vibes.
- Reactivated figurative language with extra source-domain detail beyond a fixed phrase.

Examples (Score 10):
- "Trying to do X while monitoring yourself is like driving with the parking brake slightly on: you can move, but every action fights resistance."
- "One stuck station on the assembly line backs up the whole belt, slowing everything downstream."
""",

        """3. AUTOMATIC NEGATIVES (Score 0) — KEY EDGE CASES

Score 0 if the only figurative language is any of the following:

3A) SINGLE-CLAUSE “VIBE” SIMILES (Score 0)
- One-off comparisons that only communicate a generic property (mysterious / intangible / effortless / fast)
  without adding mapped structure.
Examples (Score 0):
- "It was like a mirage." (just “hard to pin down”)
- "It moved like a whisper." (just “quiet/subtle”)

3B) COLLOQUIAL QUANTITY/SPEED INTENSIFIERS (Score 0)
- Casual hyperbole used for emphasis, not a sustained source-domain frame.
Examples (Score 0):
- "They crank out updates nonstop."
- "The bacteria multiply like crazy."

If and ONLY if the text EXTENDS these into a concrete mapped scene (roles/actions/constraints),
you may score 10.
""",

        """4. FROZEN / IDIOMATIC EXPRESSIONS (Score 0)

Score 0 for stock idioms / clichés used as fixed labels (no mapped structure).
Examples (Score 0):
- "a whole new ballgame"
- "a different kettle of fish"
- "the last straw"
""",

        """5. ANALOGY VS METAPHOR (STRICT RULE)

DECISION RULE (YES/NO)

Does the text explicitly state at least ONE role/action/constraint correspondence
between the source and target AND use it to infer something about the target (a consequence)?

YES → Score 10
NO  → Score 0

Clear Score 0 examples (label-only / didactic):
- "Memory is like a hard drive." (no role/action/constraint correspondence; no inference)
- Calling someone “a robot” purely as a label. (no role/action/constraint correspondence; no inference)

Clear Score 10 examples (mapping USED to explain):
- "Treat the cache like a pantry: if the shelf is empty you must go shopping (slow), but if it’s stocked you can cook immediately (fast)."
  (role/action/constraint correspondence: cache state ↔ pantry stock level; inference: empty ↔ slow, stocked ↔ fast)
- "It’s like a relay race: one runner can’t finish until the baton is passed, so delays propagate to the whole team."
  (role/action/constraint correspondence: dependency ↔ baton handoff constraint; inference: delay propagates)
- "The process behaves like a bouncer at a club: it lets some requests in and turns others away based on a rule."
  (role/action/constraint correspondence: gatekeeper admits/rejects ↔ bouncer admits/rejects; inference: some requests allowed, others denied)
""",

        """6. PERSONIFICATION (STRICT RULE)

Score 10 only if personification is vivid AND central (it changes understanding),
with specific agentive actions that are impossible for the domain.

Score 0 for routine, conventional shorthand (e.g., \"cells attack\", \"the market responded\")
unless extended into a concrete scene.
""",

        """7. TIE-BREAKER

If borderline or uncertain, score 0.
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


# Scaffolding v2: Clearer criteria with explicit handling of short answers
# Key improvements:
# - Defines minimum requirement: at least 2 distinct ideas with logical order
# - Short answers CAN have scaffolding if they order ideas simple→complex
# - Explicit list of what IS and IS NOT scaffolding
# - Clarifies that length alone doesn't determine scaffolding
scaffolding_metric_v2 = GEval(
    name="Scaffolding (v2)",
    evaluation_steps=[
        """STEP 1: DEFINITION - What is scaffolding?

Scaffolding means the explanation BUILDS UNDERSTANDING by introducing ideas in order of increasing complexity.

WHAT MAKES AN IDEA "SIMPLER" vs "MORE COMPLEX"?

1. STANDALONE vs DEPENDENT:
   - Simple: "Atoms are tiny particles" (understood alone)
   - Complex: "Electrons orbit the nucleus" (requires knowing what atoms are)

2. FEWER vs MORE PREREQUISITES:
   - Simple: "Plants need sunlight" (common knowledge)
   - Complex: "Chlorophyll absorbs photons to split water molecules" (requires chemistry background)

3. CONCRETE vs ABSTRACT:
   - Simple: "Ice melts when heated" (observable)
   - Complex: "Phase transitions occur when thermal energy exceeds intermolecular bonds" (abstract principle)

4. SINGLE vs COMBINED CONCEPTS:
   - Simple: "DNA stores genetic information" (one idea)
   - Complex: "DNA replication errors during cell division cause mutations that may lead to cancer" (combines 4 concepts)

MINIMUM REQUIREMENT: At least 2 ideas where the second DEPENDS ON or DEEPENS the first.

A SINGLE fact or definition is NOT scaffolding (nothing to build on).""",

        """STEP 2: Examples of SCAFFOLDING PRESENT (Score 10):

SHORT ANSWER examples:
- "Cells are the basic building blocks of life. DNA inside cells contains instructions for making proteins."
  → First establishes cells, then zooms into DNA - builds from whole to part
  
- "When you heat water, molecules move faster. This is why boiling water evaporates."
  → First explains mechanism (molecules), then applies to observable phenomenon
  
- "Gravity pulls things toward Earth. That's why we don't float away."
  → First states rule, then connects to everyday experience

LONGER ANSWER example:
- "Two facts motivate my research—first, diverse systems are healthier... My research asks if it matters that species are being gained and lost... By manipulating plant number and type, I can begin to uncover the mechanisms..."
  → Starts with motivation → poses question → describes method (clear progression)""",

        """STEP 3: Examples of SCAFFOLDING ABSENT (Score 0):

SINGLE FACT (nothing to build on):
- "Photosynthesis converts sunlight into energy."
  → Just one idea, no building possible
  
- "The mitochondria is the powerhouse of the cell."
  → Single definition, no progression

UNORDERED LIST of facts:
- "DNA contains genes. Proteins are made from amino acids. Cells divide through mitosis."
  → Multiple facts but no logical order or building between them

ALL AT ONCE (no progression):
- "Quantum entanglement occurs when particles become correlated such that measuring one instantaneously affects the other regardless of distance due to non-local correlations established at particle creation."
  → All concepts crammed together with no build-up

COMPLEX BEFORE SIMPLE:
- "The non-local correlations from entanglement happen because... basically, particles can be connected."
  → Starts with the hard part, then gives the simple explanation

LOGICAL FLOW WITHOUT BUILDING (NOT scaffolding):
- "Scientists collected samples. They analyzed the samples. They published their findings."
  → Ideas flow logically but stay at same complexity level - no building
- "The experiment tested temperature. Results showed higher temps increased reaction speed. This confirmed the hypothesis."
  → Coherent narrative, but each sentence is equally complex - no scaffolding
- "Photosynthesis happens in leaves. It uses sunlight. Plants need it to survive."
  → Related facts that flow, but none requires understanding the previous one

KEY DISTINCTION: Scaffolding requires INCREASING DEPTH, not just logical coherence.""",

        """STEP 4: Common edge cases:

SHORT ≠ NO SCAFFOLDING:
- A 2-sentence answer CAN have scaffolding if idea #1 prepares for idea #2
- Length doesn't matter; logical ordering does

TRANSITION WORDS DON'T GUARANTEE SCAFFOLDING:
- "because", "therefore", "this means" can indicate building, but not always

WITH transitions, IS scaffolding:
- "Gravity pulls objects toward Earth. Because of this, dropped objects fall."
  → Simple rule first, then applies it (builds)

WITH transitions, NOT scaffolding:
- "Entropy increases because systems tend toward disorder because energy spreads out."
  → "because" used twice but all concepts at same complexity level (no building)
- "Quantum tunneling occurs because wavefunctions extend beyond barriers."
  → Single complex statement, transition word doesn't help

WITHOUT transitions, IS scaffolding:
- "Atoms have electrons. Electrons carry charge. Charge creates electricity."
  → Clear simple→complex progression despite no explicit connectors

CHRONOLOGICAL ORDER ≠ SCAFFOLDING:
- "First I added water, then I heated it, then it boiled" is just a sequence
- Scaffolding requires conceptual building, not just time order

LOGICAL FLOW ≠ SCAFFOLDING:
- A well-organized answer can flow smoothly without scaffolding
- Scaffolding requires each idea to be MORE COMPLEX than the previous
- If all ideas are at the same complexity level, it's NOT scaffolding (even if coherent)

FINAL DECISION:
- Score 10: Ideas are ordered from simple→complex, with building/dependence
- Score 0: Single idea, random order, or complex-before-simple
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