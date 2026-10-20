You are classifying who is speaking at one marked expression inside the Results section of a published research article. Work only by reading one input file and writing one output file. Do NOT run git, do NOT edit any other file, do NOT use the network, do NOT spawn subagents.

Input: `{DIR}/batch_NN.json` — a JSON array of 20 records with `id`, `before`, `marker`, `after`. The marked expression is `marker`; `before` and `after` are about 600 characters of surrounding Results text (they may start or end mid-sentence). Read it with Python (`python -X utf8 -c "import json; ..."`) and print the records a few at a time so you read every record in full.

For each record decide whose words the marked expression belongs to:

- `participant`: the words of the study's participants, reproduced in the text — interview or focus-group quotations, free-text or open-ended survey answers, written reflections — whether or not quotation marks are used, as long as the text presents them as the participants' own words.
- `participant_summary`: the authors' own sentence that reports what participants said, felt, wanted or recommended, without reproducing their words (e.g. "participants stressed the need to ...", "residents felt they had to ...", "several interviewees suggested that programmes should ..."). The marked expression carries the participants' view, in the authors' wording.
- `author`: the authors' own reporting, description or interpretation that is not a report of participants' views — for example results of measurements or analyses, descriptions of what participants or patients were required to do under the study protocol, and the authors' own conclusions.
- `other_source`: text the authors quote from another source that is not a participant — survey or questionnaire item wording, guideline, policy, curriculum or document text, another publication.
- `unclear`: the context does not allow a decision.

Output: write `{DIR}/out_NN.json`, a JSON array with one object per input record, in input order:

```json
{"id": "...", "speaker": "participant|participant_summary|author|other_source|unclear", "reason": "one sentence", "evidence": "a verbatim span copied from before+marker+after that shows who is speaking (under 40 words)"}
```

The `evidence` must be non-empty and copied character-for-character from the concatenation `before + marker + after` (it will be checked by code). Write the file with Python `json.dump(..., ensure_ascii=False, indent=1)` and UTF-8 encoding. Then re-open it, confirm it has 20 objects whose ids match the input order, check every evidence span in code, and report the counts of each category.
