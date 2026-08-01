# Review: Hardest Question Guidelines (Google Doc)

**Doc:** "Realm Static Image Q — Hardest Charts"
**Date:** 2026-07-29
**Reviewer:** opencode

---

## 1. Overall Assessment

The guidelines are well-structured and the 7 examples are genuinely high-quality "Hardest" questions. They demonstrate multi-step reasoning that goes well beyond Challenging. However, there are **5 structural issues** that need addressing before merging these into the production prompt, and **3 opportunities** to strengthen the guidelines further.

**Score: 7/10** — Excellent examples, needs prompt-level refinement.

---

## 2. Example-by-Example Review

### Example 1 — Stacked Area / Line Plot (renewable energy)

**Question:** "For the 10-year period from 2006 to 2015, windpower's share of electricity in the EU increased by how many percentage points?"

**Why It's Good:** Correct — requires locating two data points (2006 and 2015), reading approximate values, computing the difference. Answer is a small number (10.0%).

**Issue:** The calculation `10.0%` could be ambiguous — is it 10 percentage points or a factor of 10x? The answer should be the raw number (10) not `10.0%`. The `answer_format` should be `number`, and the question should say "by how many percentage points" (which it does, correctly).

**Rating: 9/10**

---

### Example 2 — Multi-panel Line Plot (biodiversity indices)

**Question:** "Which index experiences the largest absolute percentage decline from 1970 to 2015, and what is the rounded percentage change to the nearest whole number?"

**Why It's Good:** Requires computing percentage changes for ALL indices across 4 panels, then comparing.

**Issue:** This question asks for TWO things (which index + what is the percentage). Per the existing QA handbook rules, a question should have ONE clear operation. Asking for both makes the answer format ambiguous — is the answer the index name or the number? The current validator would flag this as "Multiple questions detected" (rule in `_rule_groups.py` line 137). The question should pick one: either "Which index has the largest percentage decline?" (answer: index name) OR "What is the percentage decline of [specific index]?" (answer: number).

**Rating: 6/10** — Needs restructuring to be a single-answer question.

---

### Example 3 — Map (climate zones + forest types)

**Question:** "Where does the coniferous forest from the left map match with 'very wet' and 'not too warm' from the climate map?"

**Why It's Good:** Cross-map comparison requires overlaying two panels mentally.

**Issue:** The answer `NW Costa Rica` is somewhat ambiguous — "NW" is a spatial direction and "Costa Rica" is a label. The validator would flag this because the answer could be interpreted as "NW" or "Costa Rica" or "NW Costa Rica". The question should pin the format: "Give the country name only" or "Give the cardinal direction and country." Also, "match with" is vague — "overlaps with" or "coincides with" would be clearer.

**Rating: 7/10**

---

### Example 4 — Infographic (Aeropuertos)

**Question:** "Which province shows the highest air connectivity, as measured by the number of direct flights to the business axis?"

**Why It's Good:** Requires finding the province with the longest bar or highest number.

**Issue:** This is borderline "single-criterion matchmaking" (one of our anti-patterns). The question asks "which has the highest X" — which is the exact anti-pattern we added. However, the infographic has 30+ provinces with complex visual encoding (color + bars + labels), so finding the maximum IS genuinely hard for Qwen. The question is acceptable IF the infographic is complex enough that simple OCR won't work.

**Rating: 7/10** — Acceptable for complex infographics, but borderline for simpler bar charts.

---

### Example 5 — Scatter Plot (GDP vs CO2)

**Question:** "Starting from the overall point of all the countries, which country is in the extreme upper right corner, being the world's biggest CO2 emitter, and which is in the extreme lower left, being the country with the smallest CO2 emissions?"

**Why It's Good:** Multi-panel reasoning + visual location + domain knowledge.

**Issue:** This asks for TWO answers again (biggest emitter + smallest emitter). The validator would flag this. Should be split into two separate questions or restructured: "Which country appears in the extreme lower-left corner of the scatter plot?" (single answer).

Also, the question says "Starting from the overall point of all the countries" — this is filler text that doesn't add reasoning value. It should be removed.

**Rating: 5/10** — Two-answer format violates the single-answer rule.

---

### Example 6 — Map (Iran historical)

**Question:** "In geographical order, which nation has the highest value of the indicator between the years 1500 to 2000?"

**Why It's Good:** Cross-temporal comparison with spatial reasoning.

**Issue:** This question is somewhat vague. "In geographical order" —什么意思? Does it mean sorting countries by their position? "Highest value of the indicator" — which indicator? The map seems to show a single indicator, but the question doesn't specify. Also, "between the years 1500 to 2000" is the full range of the map — there's no temporal comparison. This feels like "find the maximum value on the map" which is a simple task.

**Rating: 4/10** — Too vague, doesn't require genuine multi-step reasoning.

---

### Example 7 — Old Document (table with handwritten data)

**Question:** "Based on the data shown, which year saw the lowest reported value in the 'Schüler' column, and what was the exact value recorded for that year?"

**Why It's Good:** Requires reading a complex historical table with mixed handwriting and typed text.

**Issue:** Two-answer format again ("which year" + "what value"). Should be one answer. Also, the table is very hard to read — even for humans. This might be too hard (unanswerable) rather than "hardest."

**Rating: 5/10** — Two-answer format, potentially unanswerable due to image quality.

---

## 3. Structural Issues

### Issue 1: No Workflow Section

The existing CHALLENGING_PROMPT has a 3-step workflow (ANALYZE → MATCH → GENERATE) that forces the LLM to reason about the image structure before generating. The HARDEST_PROMPT has no such workflow — it just lists strategies. The new guidelines should include a workflow section.

**Recommendation:** Add a workflow section to the HARDEST_PROMPT:
```
WORKFLOW:
Step 1: ANALYZE — What type of chart/image is this? How many panels? What data is shown?
Step 2: IDENTIFY — What multi-step reasoning task can be extracted? (e.g., "compare X across conditions", "calculate the change from A to B", "trace the path from C to D")
Step 3: GENERATE — Create a question that requires that reasoning. The answer must be a single value (number or word).
```

### Issue 2: Anti-Patterns Not Ported

The CHALLENGING_PROMPT has extensive anti-patterns (inline choices, single matchmaking, extreme-seeking, etc.) but the HARDEST_PROMPT has almost none. These anti-patterns apply equally to Hardest.

**Recommendation:** Add the full anti-pattern list from CHALLENGING_PROMPT to HARDEST_PROMPT:
- Inline choices ("Is it X or Y?")
- Single-criterion matchmaking ("Which panel has X?")
- Extreme-seeking ("What is the highest/lowest/most?")
- Counting as primary difficulty
- Y-axis value tracing
- Information leakage in multi-sentence questions

### Issue 3: Strategy List Is Stale

The HARDEST_PROMPT strategies (1-7) are generic and overlap heavily with Challenging. For Hardest, the strategies need to be MORE demanding:
- Challenging: "Which panel has the higher value?" (binary comparison)
- Hardest: "Calculate the percentage difference between X and Y, then determine which category it belongs to" (multi-step calculation + comparison)

**Recommendation:** Rewrite the strategy list to emphasize CALCULATION and MULTI-STEP operations, not just comparison.

### Issue 4: Missing "Why It's Good" Pattern

The Google Doc examples have "Why It's Good" explanations. These should be encoded as self-check rules in the prompt (like the CHALLENGING_PROMPT's "What a GOOD Challenging question looks like" section).

**Recommendation:** Add a self-check section:
```
✅ What a GOOD Hardest question looks like:
- Requires 2+ distinct visual operations (read + calculate, compare + rank, trace + identify)
- Answer requires arithmetic (percentage, difference, ratio) OR multi-panel reasoning
- The question cannot be answered by scanning one panel — multiple data sources must be consulted
- Answer is deterministic (not ambiguous between multiple valid answers)
- No information is leaked in the question text
```

### Issue 5: Answer Format Rules Missing

The guidelines don't specify answer format for Hardest. Some examples produce numbers (Example 1: 10), some produce words (Example 3: NW Costa Rica), some produce compound answers (Example 2: index name + percentage). The prompt should enforce: "Answer is ONE number or ONE word — never compound answers."

**Recommendation:** Add explicit format rule: "The answer must be a single atomic value — one number, one word, or one short phrase (max 4 words). Never ask for two things."

---

## 4. Opportunities to Strengthen

### Opportunity 1: Define "Hardest" vs "Challenging" More Clearly

Currently, the distinction is vague. The Google Doc subtitle says "multi-step reasoning — comparing values across panels, calculating percentages, evaluating relationships." This is a good definition but needs to be in the prompt.

**Suggested definition:**
- **Challenging:** Requires 1 comparison or 1 calculation across 2 conditions
- **Hardest:** Requires 2+ operations: e.g., calculate THEN compare, or trace THEN identify, or read THEN compute THEN rank

### Opportunity 2: Add Calculation Templates

Several examples use calculations (percentage change, difference, ratio). The prompt should provide templates for these:
```
Calculation templates:
- Percentage change: "[Entity X]'s value changed from [A] in [year1] to [B] in [year2]. What is the percentage change?"
- Difference: "How many [units] more/less does [X] have compared to [Y]?"
- Ratio: "What is the ratio of [X] to [Y] for [category Z]?"
- Rank delta: "How many positions does [entity] drop/rise in the ranking from [condition A] to [condition B]?"
```

### Opportunity 3: Qwen-Specific Hardness

The current prompt mentions Qwen's weaknesses but doesn't explain WHY these questions defeat Qwen. Adding a brief explanation would help:
```
Qwen fails on Hardest questions because:
- It tends to answer the FIRST comparison it finds, not the complete multi-step reasoning
- It cannot hold multiple data points in memory for arithmetic operations
- It defaults to "cannot determine" when the reasoning chain is too long
```

---

## 5. Summary: What to Merge into HARDEST_PROMPT

| Element | Status | Action |
|---------|--------|--------|
| Examples 1-3 | Good quality | Convert to few-shot examples |
| Examples 4-6 | Acceptable but need fixes | Fix two-answer format, merge as examples |
| Example 7 | Potentially unanswerable | Drop — image quality too low |
| Workflow section | Missing | Add 3-step workflow |
| Anti-patterns | Missing | Port from CHALLENGING_PROMPT |
| Strategy list | Stale | Rewrite to emphasize calculation |
| Self-check section | Missing | Add "What a GOOD Hardest question looks like" |
| Answer format rules | Missing | Add single-answer enforcement |
| Qwen hardness explanation | Missing | Add brief explanation |

---

## 6. Recommended Action

1. Fix the 5 examples that have two-answer format (Examples 2, 5, 7) to be single-answer
2. Merge the good elements into the HARDEST_PROMPT in `_draft_prompts.py`
3. Add workflow, anti-patterns, self-check, and format rules
4. Update the HARDEST_PROMPT anti-pattern list in the validator (`_rule_groups.py`)
5. Test with 10 regenerated tasks at `difficulty=hardest` to verify quality improvement

**Estimated effort:** 2-3 hours
