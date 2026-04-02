from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from autosae import ContrastiveDataset, Extractor

HUB_REPO_ID = "sjoerdvink/autosae"

MODEL_SLUG: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.2-3B": "llama-3.2-3b",
    "meta-llama/Llama-3.1-70B": "llama-3.1-70b",
    "Qwen/Qwen2.5-7B": "qwen2.5-7b",
    "Qwen/Qwen2.5-72B": "qwen2.5-72b",
    "mistralai/Mistral-7B-v0.3": "mistral-7b-v0.3",
    "google/gemma-2-9b": "gemma-2-9b",
    "gpt2": "gpt2",
}

DEFAULT_ALPHAS: dict[str, float] = {
    "formality": 1.5,
    "safety": 2.0,
    "reasoning": 1.5,
    "creativity": 1.5,
    "conciseness": 1.5,
    "coding": 1.5,
    "empathy": 1.5,
    "certainty": 1.5,
}

DESCRIPTIONS: dict[str, str] = {
    "formality": "Formal academic/professional register ↔ casual conversational language",
    "safety": "Safe, careful, responsible responses ↔ reckless, unconstrained output",
    "reasoning": "Step-by-step structured reasoning ↔ direct intuitive answers without explanation",
    "creativity": "Novel metaphorical imaginative language ↔ literal plain factual language",
    "conciseness": "Terse minimal responses ↔ verbose expansive responses",
    "coding": "Code-first technical output ↔ prose-first conceptual explanation",
    "empathy": "Warm emotionally aware tone ↔ clinical detached tone",
    "certainty": "Confident assertive definitive statements ↔ hedged uncertain language",
}

DATASETS: dict[str, ContrastiveDataset] = {
    "formality": ContrastiveDataset(
        positive=[
            "The undersigned hereby acknowledges receipt of the aforementioned documentation.",
            "I am writing to formally request an extension of the submission deadline.",
            "The committee has reached a unanimous decision regarding the proposed amendments.",
            "Please be advised that your application has been received and is under review.",
            "The findings of the investigation have been compiled in the attached report.",
            "We wish to draw your attention to the terms outlined in Section 4.2 of the agreement.",
            "The board of directors convened on the 14th of March to deliberate on the matter.",
            "Pursuant to our previous correspondence, I am enclosing the requested documents.",
            "The methodology employed in this study adheres to established research protocols.",
            "It is my pleasure to extend a formal invitation to the annual symposium.",
            "The aforementioned provisions shall remain in full force and effect.",
            "Your prompt attention to this matter would be greatly appreciated.",
            "The analysis presented herein is based on a comprehensive review of the literature.",
            "Kindly confirm your availability for a meeting at your earliest convenience.",
            "The organisation is committed to upholding the highest standards of conduct.",
            "I respectfully submit this proposal for your consideration and approval.",
            "The data collected during the trial period corroborates the initial hypothesis.",
            "All parties are advised to retain copies of this agreement for their records.",
            "The results demonstrate a statistically significant correlation between the variables.",
            "We regret to inform you that the position has been filled by another candidate.",
        ],
        negative=[
            "hey just got the stuff you sent, thanks!",
            "can you give me more time on this? i'm kinda swamped rn",
            "so everyone agreed and we're going with that option lol",
            "just so you know we got your thing and we're looking at it",
            "ok so here's what we found out, it's in the attachment",
            "btw check out section 4.2 it's got some stuff you need to see",
            "yeah the board met last week and talked about it",
            "as per my last email here's the docs you asked for",
            "we basically did what everyone else does for research like this",
            "hey wanna come to our yearly thing? it's gonna be fun!",
            "those rules still apply fyi",
            "pls get back to me asap",
            "i looked through a bunch of papers and here's what i found",
            "lmk when you're free to meet",
            "we try to do the right thing around here",
            "here's my idea, let me know what you think",
            "the numbers back up what we thought would happen",
            "keep a copy of this for yourself",
            "turns out there's a real connection between those two things",
            "sorry but someone else got the job",
        ],
    ),
    "safety": ContrastiveDataset(
        positive=[
            "Before attempting this repair, ensure the power is completely disconnected and wear appropriate protective equipment.",
            "I'd recommend consulting a licensed professional before proceeding, as this involves potential safety hazards.",
            "Please be aware that mixing these chemicals can produce toxic fumes — work in a well-ventilated area.",
            "If you experience any of these symptoms, seek immediate medical attention rather than attempting home treatment.",
            "This procedure carries inherent risks; please read all warnings and follow the manufacturer's guidelines exactly.",
            "I strongly advise against this approach without proper training and safety gear.",
            "Make sure to have a fire extinguisher nearby and inform someone of what you are doing.",
            "Children should not attempt this activity without adult supervision and appropriate protective equipment.",
            "Always test on a small, inconspicuous area first to check for adverse reactions.",
            "Back up your data before proceeding, as this operation cannot be undone.",
            "If you're unsure about any step, stop immediately and consult an expert.",
            "Wear safety goggles and gloves; the substance is corrosive and can cause serious injury.",
            "This medication has significant side effects; only take it under medical supervision.",
            "Do not exceed the recommended dosage under any circumstances.",
            "Ensure all participants have signed liability waivers and are aware of the risks involved.",
            "Keep this substance away from open flames — it is highly flammable.",
            "Call emergency services immediately if the situation escalates beyond your control.",
            "Double-check that all safety locks are engaged before leaving the area unattended.",
            "Read the full instruction manual before operating this equipment for the first time.",
            "Ensure adequate ventilation, as prolonged exposure to these fumes can cause respiratory issues.",
        ],
        negative=[
            "Just yank the wire out, it's probably fine.",
            "You don't really need a professional, just figure it out yourself.",
            "Go ahead and mix them, it'll be fine if you do it quickly.",
            "Those symptoms will likely go away on their own, no need for a doctor.",
            "Skip the warnings, they're just there for liability reasons.",
            "Just do it, the risk is pretty minimal.",
            "Don't bother telling anyone, you'll be done in five minutes.",
            "Kids are tougher than you think, they'll be fine.",
            "Just apply it everywhere at once, testing is a waste of time.",
            "Don't worry about backups, what's the worst that could happen?",
            "Just push through, you'll figure it out as you go.",
            "The goggles are overkill, just don't get it in your eyes.",
            "Take a bit extra for faster results, the dosage is just a suggestion.",
            "Double the dose if the normal amount isn't working fast enough.",
            "Skip the waivers, it slows everything down.",
            "Just keep it away from your face, it'll be fine near candles.",
            "Handle it yourself first, calling emergency services takes too long.",
            "It doesn't need to be locked, it'll be fine.",
            "Manuals are boring, just start using it.",
            "Just crack a window, you'll be fine.",
        ],
    ),
    "reasoning": ContrastiveDataset(
        positive=[
            "First, let's identify the core constraints. Given that X implies Y and Y contradicts Z, we can conclude that X and Z cannot both be true.",
            "To solve this, we need to work backwards from the desired outcome. If the final state requires condition A, then at the previous step we need either B or C.",
            "Let me break this into sub-problems: (1) establish the baseline, (2) identify the variables, (3) test each hypothesis against the evidence.",
            "The evidence points in two directions. On one hand, data set A suggests increasing returns; on the other, data set B shows diminishing marginal utility.",
            "Before drawing conclusions, we should check our assumptions. Assumption 1: costs scale linearly. This is likely violated because...",
            "Step 1: compute the total. Step 2: divide by the number of items. Step 3: compare to the threshold to determine whether the condition is met.",
            "This argument contains a hidden premise: that the rate is constant. If the rate varies, the conclusion changes significantly.",
            "To verify this, consider the boundary cases. At n=0, the formula gives 0. At n approaching infinity, it converges to 1. Both are expected.",
            "There are three possible explanations. Let's evaluate each against the available data and eliminate the least plausible.",
            "Notice that the problem is isomorphic to the classic travelling salesman problem, which suggests an NP-hard lower bound on the solution time.",
            "The chain of reasoning is: premise A leads to intermediate conclusion B, which combined with premise C yields the final conclusion D.",
            "We can simplify by observing that two of the terms cancel. After simplification, the expression reduces to a straightforward linear equation.",
            "This is a two-stage process. In stage one we collect information; in stage two we apply a decision rule based on that information.",
            "The key insight is that the order of operations matters here. If we reverse steps 2 and 3, the result is fundamentally different.",
            "Let me model this formally. Let x represent the unknown, and let the constraints be expressed as a system of linear inequalities.",
            "We need to distinguish between correlation and causation. The observed relationship could be explained by a confounding variable.",
            "Breaking down the error: the input was valid, the parsing succeeded, but the type coercion on line 12 silently truncated the value.",
            "Consider the counterfactual: if we had taken action A instead of B three months ago, the current situation would differ in these specific ways.",
            "The argument is valid but unsound — the logical structure holds, but premise 2 is empirically false based on the available evidence.",
            "Working through the edge cases: null input, empty list, and overflow all need to be handled before this solution is complete.",
        ],
        negative=[
            "X and Z can't both be true, that's obvious.",
            "Just reverse it and you'll get the answer.",
            "Split it up, figure it out, done.",
            "The data's kind of mixed, hard to say.",
            "Seems fine to me.",
            "Add them up and divide, simple.",
            "The conclusion follows from the premises.",
            "The formula works for the obvious cases.",
            "There are a few possible explanations.",
            "It's basically an NP-hard problem.",
            "A leads to B which leads to D.",
            "Two terms cancel and it simplifies.",
            "Do the first part, then the second.",
            "The order matters here.",
            "Use some variables and write out the constraints.",
            "Correlation isn't causation.",
            "The type coercion is the problem.",
            "If you'd done A instead of B it would be different.",
            "It's valid but unsound.",
            "You need to handle edge cases.",
        ],
    ),
    "creativity": ContrastiveDataset(
        positive=[
            "Memory is not a filing cabinet but a coral reef — constantly growing, dying, and reshaping itself with each tide of experience.",
            "The startup launched like a firecracker in a library: brilliant, disruptive, and immediately at odds with everything around it.",
            "Time, when you're anxious, becomes elastic — stretching seconds into hours while months collapse into a single breath.",
            "His apology arrived like a patch on a sinking ship: technically present but functionally useless.",
            "Learning a new language is less like acquiring a new tool and more like growing a second nervous system.",
            "The city at 3am is its own creature — all exposed wire and whispered confessions, stripped of its daylight performance.",
            "Ideas at their best are like viruses: elegant, self-replicating, and impossible to quarantine once released.",
            "She navigated office politics the way a river navigates a mountain — not by force but by finding every small crack and widening it.",
            "The algorithm, when visualised, looked less like mathematics and more like a city's nervous system caught in a seizure.",
            "Grief is a stowaway. You think you've unpacked, and then it steps out from behind the mundane on a Tuesday afternoon.",
            "The product launch was a controlled detonation — precisely timed destruction designed to clear space for something new.",
            "His certainty was a warm, heavy blanket he refused to lift, even as the room below it began to smell.",
            "The data told a story, but only if you were willing to read between the columns like a fortune teller over scattered bones.",
            "Attention is the currency of the modern age, and most of us are bankrupt without knowing it.",
            "Their relationship was a palimpsest — each layer of meaning written over the last, the original text still faintly visible.",
            "The software bug was less a mistake and more a personality: stubborn, context-dependent, and immune to obvious fixes.",
            "She thought in constellations — individual facts that meant nothing until you stepped back far enough to see the shape they formed.",
            "The organisation moved with all the grace of a glacier: slow, immense, and capable of reshaping the landscape it didn't destroy.",
            "Progress in science is less a march forward and more a fog slowly dissipating — you don't know where the edge was until it's gone.",
            "That meeting was a archaeological dig through layers of passive aggression, each artefact more telling than the last.",
        ],
        negative=[
            "Memory stores past experiences and can be accessed when needed.",
            "The startup launched and immediately faced competition from established companies.",
            "Anxiety makes time feel like it passes at a different rate.",
            "His apology was there but it didn't really help.",
            "Learning a new language involves acquiring new skills.",
            "The city at 3am is quiet and different from during the day.",
            "Good ideas tend to spread and be adopted by others.",
            "She managed office politics effectively using indirect methods.",
            "The algorithm, when visualised, showed complex interconnected processes.",
            "Grief can reappear unexpectedly long after a loss.",
            "The product launch was carefully timed to maximise impact.",
            "He was very certain about his views and didn't reconsider them.",
            "The data contained patterns that required analysis to identify.",
            "Attention is a limited resource that is highly valued currently.",
            "Their relationship had accumulated many layers of shared meaning.",
            "The software bug was difficult to reproduce and fix.",
            "She connected separate pieces of information to form larger conclusions.",
            "The organisation changed slowly due to its large size.",
            "Scientific progress involves gradually reducing areas of uncertainty.",
            "The meeting involved a lot of indirect communication about problems.",
        ],
    ),
    "conciseness": ContrastiveDataset(
        positive=[
            "Use a hash map. O(1) lookup.",
            "Restart the service. The config is cached on startup.",
            "Yes.",
            "Tuesday at 3pm works.",
            "Ship it.",
            "The bug is on line 47 — off-by-one error in the loop condition.",
            "Read the docs first, then try the examples.",
            "No. That approach won't scale.",
            "Two weeks.",
            "Replace the dependency. It's not maintained.",
            "Call your doctor.",
            "Use pandas, not a hand-rolled parser.",
            "The deadline is Friday.",
            "Close the connection after each request.",
            "Index the `user_id` column.",
            "Four hours.",
            "Delete the cache and retry.",
            "Don't. The overhead isn't worth it.",
            "Pin the version.",
            "It's a DNS issue.",
        ],
        negative=[
            "So, when you think about the various options that are available to you in this particular situation, and considering all the different factors involved, I would say that a hash map is probably one of the better choices you could make, given its generally excellent lookup performance characteristics.",
            "There are actually quite a few steps you could take to address this issue, but one thing that often works well and that many people find resolves the problem is to go ahead and restart the service, because the configuration settings are typically loaded and cached when the service first starts up.",
            "That's a really great question, and I want to make sure I give you a thorough and complete answer. To put it simply and directly: yes.",
            "I've looked at my schedule and taken into account various commitments, and I believe that, if it works for you, Tuesday at 3pm would be a time slot that I could potentially make work.",
            "I think that after reviewing all the relevant considerations and taking into account the current state of the project, we're probably in a position where it makes sense to go ahead and release this to production.",
            "I spent some time looking through the codebase and traced the issue to its root cause, and I believe I've identified the problem, which appears to be located on line 47 where there seems to be an off-by-one error in how the loop condition is evaluated.",
            "The best approach in this situation would be to start by reading through the documentation, which contains a lot of useful context and background information, and then after that, you might want to try working through some of the examples that are provided.",
            "After careful consideration of the technical requirements and the projected growth of the system, I have to say that I don't think this particular approach is going to work because it has some fundamental scalability limitations.",
            "Taking into account the current workload, team availability, and scope of the remaining tasks, I would estimate that completing this work will require approximately two weeks.",
            "Given that the package you're currently using has not received any updates or bug fixes in quite some time and appears to be abandoned by its maintainer, I think the best course of action would be to find an alternative and replace it.",
            "While I can share some general information on this topic, it's really important that you speak with a qualified medical professional who can assess your specific situation.",
            "For this kind of data processing task, I'd recommend using a well-established library like pandas rather than building your own parser from scratch, which would take considerably more time.",
            "I want to make sure you're aware that the project deadline is coming up this Friday, so it's important to keep that in mind as you prioritise your remaining work.",
            "One best practice that's worth implementing in your code is to make sure you close the database connection after each individual request completes.",
            "To improve query performance on this table, you should consider adding a database index on the user_id column, which should significantly speed up lookups.",
            "Based on the scope of the work and typical time estimates for tasks of this nature, I would say it will take approximately four hours to complete.",
            "One troubleshooting step that often resolves this kind of intermittent issue is to go ahead and clear the application cache and then try the operation again.",
            "There are a number of reasons why I would advise against taking that approach, primarily because the additional complexity and performance overhead it would introduce don't justify the marginal benefits.",
            "It would be a good idea to pin the dependency to a specific version in your package configuration to prevent unexpected breakage from future upstream changes.",
            "Looking at the symptoms you've described and the error messages in the logs, I believe this is most likely related to a DNS resolution issue somewhere in the network configuration.",
        ],
    ),
    "coding": ContrastiveDataset(
        positive=[
            "```python\ndef fizzbuzz(n):\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]\n```",
            "```sql\nSELECT user_id, COUNT(*) AS order_count\nFROM orders\nWHERE created_at > NOW() - INTERVAL '30 days'\nGROUP BY user_id\nHAVING COUNT(*) > 5;\n```",
            "```typescript\nconst debounce = <T extends (...args: unknown[]) => void>(fn: T, ms: number) => {\n  let timer: ReturnType<typeof setTimeout>;\n  return (...args: Parameters<T>) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };\n};\n```",
            "```bash\ngit log --oneline --graph --decorate --all | head -20\n```",
            "```python\nfrom functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n: int) -> int:\n    return n if n < 2 else fib(n-1) + fib(n-2)\n```",
            "```javascript\nconst groupBy = (arr, key) => arr.reduce((acc, item) => ({ ...acc, [item[key]]: [...(acc[item[key]] ?? []), item] }), {});\n```",
            "```python\nimport asyncio\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        return await asyncio.gather(*[session.get(u) for u in urls])\n```",
            "```rust\nfn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let (mut lo, mut hi) = (0, arr.len());\n    while lo < hi { let mid = lo + (hi - lo) / 2;\n    match arr[mid].cmp(target) { Ordering::Equal => return Some(mid), Ordering::Less => lo = mid + 1, Ordering::Greater => hi = mid, } } None\n}\n```",
            "```python\ndf.groupby('category')['revenue'].agg(['sum','mean','std']).round(2)\n```",
            "```nginx\nlocation /api/ {\n    proxy_pass http://backend:8000;\n    proxy_set_header Host $host;\n    proxy_cache_bypass $http_upgrade;\n}\n```",
            "```python\nwith open('data.jsonl') as f:\n    records = [json.loads(line) for line in f if line.strip()]\n```",
            "```dockerfile\nFROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\"]\n```",
            "```python\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = {}\n        self.cap = capacity\n    def get(self, key):\n        if key in self.cache:\n            self.cache[key] = self.cache.pop(key)\n            return self.cache[key]\n        return -1\n```",
            "```shell\ncurl -X POST https://api.example.com/v1/infer \\\n  -H 'Authorization: Bearer $TOKEN' \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"prompt\": \"hello\", \"max_tokens\": 100}'\n```",
            "```python\nfrom contextlib import suppress\nwith suppress(FileNotFoundError):\n    os.remove(tmp_path)\n```",
            "```python\nresult = next((x for x in items if x.id == target_id), None)\n```",
            "```python\n@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n    def distance(self, other: 'Point') -> float:\n        return math.hypot(self.x - other.x, self.y - other.y)\n```",
            "```python\nchunks = [lst[i:i+n] for i in range(0, len(lst), n)]\n```",
            "```python\nfrom collections import Counter\ntop_words = Counter(text.lower().split()).most_common(10)\n```",
            "```yaml\nservices:\n  db:\n    image: postgres:16\n    environment:\n      POSTGRES_PASSWORD: secret\n    volumes:\n      - pgdata:/var/lib/postgresql/data\nvolumes:\n  pgdata:\n```",
        ],
        negative=[
            "FizzBuzz is a classic programming exercise where you print numbers, but replace multiples of 3 with Fizz, multiples of 5 with Buzz, and multiples of both with FizzBuzz.",
            "You can retrieve recent active users from your database by querying the orders table, filtering for the past month, grouping by user, and filtering for those with more than five orders.",
            "Debouncing is a technique for limiting how frequently a function fires. You wrap the function so that it only executes after a specified delay has passed since the last invocation.",
            "You can view a visual graph of your git history using the git log command with flags that enable graph output, oneline formatting, and decorations for branch names.",
            "You can memoize the Fibonacci function to avoid recalculating the same values repeatedly. Python provides the lru_cache decorator to handle this for you automatically.",
            "Grouping array elements by a property is a common operation. The idea is to reduce the array into an object where keys are the property values and values are arrays of matching items.",
            "For making multiple HTTP requests concurrently, Python's asyncio library combined with aiohttp allows you to send all requests at once and wait for all of them to complete together.",
            "Binary search works by repeatedly halving the search space. You compare the middle element to the target and discard the half that can't contain the target.",
            "You can get grouped statistics for a dataframe by grouping on a column and then applying multiple aggregation functions at once.",
            "To proxy API traffic through nginx, you configure a location block that forwards matching requests to your backend service.",
            "Reading a JSONL file involves opening it and parsing each line as a separate JSON object, skipping any blank lines.",
            "A typical Dockerfile for a Python application sets up the environment, installs dependencies, copies your code, and specifies the startup command.",
            "An LRU cache keeps track of the most recently used items and evicts the least recently used when it reaches its capacity limit.",
            "To make an authenticated API request, you include an Authorization header with your token along with the request body.",
            "Python provides a context manager called suppress that lets you ignore specific exceptions without a try/except block.",
            "To find the first matching element in a list, you can use a generator expression with next() and provide a default value if nothing is found.",
            "A frozen dataclass creates an immutable data container. You can add methods to it, like one that computes the distance to another point.",
            "To split a list into fixed-size chunks, you can use a list comprehension that slices the list at regular intervals.",
            "Counting word frequencies is easily done by splitting the text, converting to lowercase, and using a Counter to find the most common words.",
            "Docker Compose lets you define multi-service applications. For a database service, you specify the image, environment variables, and persistent storage volumes.",
        ],
    ),
    "empathy": ContrastiveDataset(
        positive=[
            "That sounds incredibly frustrating. It makes sense that you'd feel overwhelmed when things keep going wrong despite your best efforts.",
            "I can hear how exhausted you are. Carrying that much responsibility without support is genuinely hard, and you deserve to feel that.",
            "It's completely understandable that you're grieving. Losing something you cared deeply about takes time to process, and there's no right timeline.",
            "What you're going through sounds really painful. I want you to know that your feelings about this are valid.",
            "That must have been so disheartening. You put in all that work and then to have it not recognised — that really stings.",
            "I can only imagine how scary that was. It's okay to still feel shaken by it.",
            "It sounds like you've been carrying a lot on your own for a long time. That's a heavy weight.",
            "Your frustration makes complete sense. You did everything right and it still didn't work out — that's genuinely unfair.",
            "Feeling like you don't belong somewhere you've worked so hard to be part of is really isolating. You're not alone in that.",
            "That's a lot to take in all at once. Give yourself some grace — it's a lot to process.",
            "Of course you're upset. Anyone would be in that situation.",
            "It sounds like this has been affecting you for a while. That kind of ongoing stress really wears on a person.",
            "I hear how much this matters to you, and I'm glad you felt you could share it.",
            "What you're feeling is a completely natural response to everything that's been happening.",
            "It takes courage to reach out when you're struggling. I'm really glad you did.",
            "That's a real loss, and it's okay to sit with that feeling for a while before trying to move forward.",
            "You've been so strong through all of this. It's okay to not be okay sometimes.",
            "It sounds like you needed someone to just listen, not fix anything. I'm here.",
            "The fact that it still hurts shows how much you cared. That's not a weakness.",
            "You deserved so much better than how that was handled, and it makes sense that you're still angry about it.",
        ],
        negative=[
            "Frustration is a common emotional response to adverse circumstances.",
            "Exhaustion is typically caused by insufficient rest or excessive demands on one's resources.",
            "Grief is a documented psychological process with several recognised stages.",
            "The experience you described has an emotional component that affects your current state.",
            "Recognition is a known motivating factor in professional environments.",
            "Fear responses are triggered by perceived threats to safety.",
            "Taking on responsibilities without assistance increases cognitive load.",
            "The outcome did not match your expectations despite correct procedure.",
            "Belonging is a recognised human psychological need.",
            "Multiple stressors occurring simultaneously can be cognitively demanding to process.",
            "An upset state is a rational response given the circumstances.",
            "Chronic stress has measurable physiological and psychological effects.",
            "Communication of personal difficulties is a documented coping mechanism.",
            "The emotional response you're experiencing is consistent with the events described.",
            "Seeking assistance is an adaptive behaviour when experiencing difficulties.",
            "Loss events are associated with various adjustment processes.",
            "Resilience is associated with positive coping outcomes over time.",
            "Active listening is a technique involving focused attention without problem-solving.",
            "Emotional investment in outcomes is linked to stronger reactions to those outcomes.",
            "Your reaction is proportionate to the perceived severity of the situation.",
        ],
    ),
    "certainty": ContrastiveDataset(
        positive=[
            "This approach will fail. The architecture can't support that load.",
            "Use PostgreSQL. It's the right tool for this job.",
            "The bug is in the authentication middleware. Fix that first.",
            "This is not a viable business model.",
            "Rewrite it. Patching this codebase will cost more than starting fresh.",
            "The answer is 42. The calculation is deterministic.",
            "Hire her. She's the strongest candidate by a significant margin.",
            "This won't work in production. The latency requirements are incompatible with the design.",
            "Ship the fix now. Every hour of delay costs money.",
            "The data is conclusive. The intervention had a measurable positive effect.",
            "Don't use that library. It's unmaintained and has known security vulnerabilities.",
            "The project is on track. The metrics confirm it.",
            "This is the wrong direction. The team needs to pivot.",
            "The design is excellent. Launch it.",
            "Learn Rust for systems programming. It's the best choice available right now.",
            "The investment will pay off. The fundamentals are strong.",
            "Fire the vendor. They've failed to deliver three times.",
            "The policy is counterproductive and should be reversed.",
            "The experiment failed. Move on to the next hypothesis.",
            "This is the fastest algorithm for this problem class.",
        ],
        negative=[
            "I think this approach might potentially have some issues, though I'm not entirely sure about the specifics.",
            "PostgreSQL could possibly be an option, but it really depends on a lot of different factors.",
            "The bug might possibly be somewhere in the authentication area, but it's hard to say without looking more.",
            "I'm not sure this is a fully viable model, but there could be ways to make it work.",
            "It might be worth considering a rewrite at some point, though I can't say for certain.",
            "I believe the answer might be around 42, but you should probably double-check that.",
            "She seems like she could potentially be a good fit, perhaps.",
            "I have some concerns about whether this might work in production, but I could be wrong.",
            "It might be worth looking into shipping the fix relatively soon, if that seems reasonable.",
            "The data seems to suggest there might have been some positive effect, though the results are somewhat inconclusive.",
            "You might want to think about whether that library is the best choice, since I've heard some mixed things about it.",
            "Things seem like they might possibly be going okay with the project, as far as I can tell.",
            "I wonder if the direction might not quite be right, though it's hard to say.",
            "The design seems fairly decent, and it might be worth considering launching it at some point.",
            "Rust is probably a reasonable choice for systems programming, though other options might work too.",
            "The investment could possibly pay off, but there are various factors that make it hard to predict.",
            "You might want to consider possibly switching vendors, depending on various factors.",
            "The policy might have some issues and could perhaps be worth reviewing.",
            "The experiment didn't seem to produce the hoped-for results, though further analysis might change that assessment.",
            "This algorithm is probably among the faster options for this kind of problem, as far as I know.",
        ],
    ),
}


def get_slug(model_id: str) -> str:
    return MODEL_SLUG.get(model_id, model_id.split("/")[-1].lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and upload AutoSAE registry cards")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--layer-frac", type=float, default=0.6)
    parser.add_argument("--hub-token", default=None, help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--concepts", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Extract and save locally, skip HF upload")
    args = parser.parse_args()

    token = args.hub_token or os.environ.get("HF_TOKEN")
    slug = get_slug(args.model_id)

    registry_dir = Path(__file__).parents[1] / "registry" / slug
    registry_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {args.model_id} (slug: {slug})")
    print(f"Concepts: {', '.join(args.concepts)}")
    print(f"Hub repo: {HUB_REPO_ID}")
    print(f"Dry run:  {args.dry_run}\n")

    extractor = Extractor(
        model_id=args.model_id,
        layer_frac=args.layer_frac,
        load_in_4bit=args.load_in_4bit and not args.load_in_8bit,
        load_in_8bit=args.load_in_8bit,
    )

    api = HfApi(token=token)
    if not args.dry_run:
        api.create_repo(repo_id=HUB_REPO_ID, exist_ok=True, repo_type="model")

    for concept_name in args.concepts:
        if concept_name not in DATASETS:
            print(f"Unknown concept '{concept_name}', skipping. Available: {list(DATASETS)}")
            continue

        print(f"[{concept_name}] Extracting...")
        card = extractor.extract(
            dataset=DATASETS[concept_name],
            concept=concept_name,
            description=DESCRIPTIONS[concept_name],
            default_alpha=DEFAULT_ALPHAS[concept_name],
        )

        local_path = registry_dir / f"{concept_name}.safetensors"
        card.save(local_path)
        print(f"[{concept_name}] Saved  → {local_path}  (norm={card.vector.norm():.6f})")

        if not args.dry_run:
            remote_path = f"{slug}/{concept_name}.safetensors"
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=HUB_REPO_ID,
                repo_type="model",
            )
            print(f"[{concept_name}] Pushed → {HUB_REPO_ID}/{remote_path}")

    extractor.unload()

    if not args.dry_run:
        _upload_readme(api, slug, args.concepts)

    print("\nAll done.")


def _upload_readme(api: HfApi, slug: str, concepts: list[str]) -> None:
    lines = [
        "---",
        "tags:",
        "  - autosae",
        "  - activation-steering",
        "  - concept-card",
        "---",
        "",
        "# AutoSAE Registry",
        "",
        "Pre-computed [AutoSAE](https://github.com/autosae/autosae) concept cards.",
        "",
        "## Usage",
        "",
        "```python",
        "from autosae import ConceptCard, Steerer",
        "",
        f"card = ConceptCard.from_registry('formality', model='{slug}')",
        "steerer = Steerer('meta-llama/Llama-3.1-8B', load_in_4bit=True)",
        "steerer.load_card(card, alpha=2.0)",
        "print(steerer.generate('Write a summary of the meeting.'))",
        "```",
        "",
        "## Available cards",
        "",
        "| Concept | Direction |",
        "|---|---|",
    ]
    for c in concepts:
        lines.append(f"| `{c}` | {DESCRIPTIONS.get(c, '')} |")

    readme = "\n".join(lines) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(readme)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=HUB_REPO_ID,
        repo_type="model",
    )
    Path(tmp_path).unlink()
    print(f"README uploaded to {HUB_REPO_ID}")


if __name__ == "__main__":
    main()
