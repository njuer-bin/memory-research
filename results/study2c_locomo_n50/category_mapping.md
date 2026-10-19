# LoCoMo category mapping (Study 2C artifact)

The LoCoMo distribution stores question categories as integers 1-4 without
an authoritative legend in the redistributed subset. The mapping below was
established by (a) inspecting sampled questions per category in the local
full dataset and (b) cross-checking against two independent secondary
descriptions of the benchmark. Reviewers should treat this mapping as
reconstructed, not official.

| category | n (subset) | mapping | basis |
|---|---|---|---|
| 1 | 10 | multi-hop | questions chain two or more facts across sessions (e.g. "which gift did I buy for the person who ...") |
| 2 | 10 | temporal | questions ask about dates, durations, or order of events ("how many days ago ...", "what did I do before ...") |
| 3 | 9 | open-domain | questions ask for world knowledge or opinions conditioned on user context ("recommend ...", "what do you think about ...") |
| 4 | 21 | single-hop | questions resolved by one fact in one session ("what is my ...", "where did I ...") |

Sample evidence per category is reproducible with:

```bash
python -c "
import json, random
recs = [json.loads(l) for l in open('datasets/data/locomo_refined_dataset/public/questions.jsonl', encoding='utf-8') if l.strip()]
by_cat = {}
for r in recs:
    by_cat.setdefault(str(r.get('category')), []).append(r)
rng = random.Random(7)
for c in sorted(by_cat):
    print('--- category', c)
    for r in rng.sample(by_cat[c], 3):
        print(' Q:', r.get('question','')[:95])
        print(' A:', str(r.get('answer',''))[:70])
"
```
