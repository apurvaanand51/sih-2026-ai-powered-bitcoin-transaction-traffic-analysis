# NETRA — Complete Learning Guide

**Everything used to build this project, explained from scratch.**

This is a teaching document, not a reference. It assumes you can read Python and
JavaScript but have never built a data→ML→API→UI pipeline. Read it top to bottom
and you will be able to rebuild NETRA yourself, and — more importantly — explain
every decision in it to a judge.

Written for Team Vortex, SIH 2026, PS ID SIH26146.

---

## Table of contents

1. [The problem, and the decisions it forced](#1-the-problem-and-the-decisions-it-forced)
2. [Bitcoin domain concepts](#2-bitcoin-domain-concepts)
3. [Python: environments and project structure](#3-python-environments-and-project-structure)
4. [Data engineering: synthetic data and ground truth](#4-data-engineering-synthetic-data-and-ground-truth)
5. [Graph algorithms: entity clustering](#5-graph-algorithms-entity-clustering)
6. [Machine learning](#6-machine-learning)
7. [The backend (FastAPI)](#7-the-backend-fastapi)
8. [The frontend](#8-the-frontend)
9. [Contracts: the thing that saves six-person teams](#9-contracts-the-thing-that-saves-six-person-teams)
10. [Packaging and offline deployment](#10-packaging-and-offline-deployment)
11. [CI/CD](#11-cicd)
12. [Git workflow](#12-git-workflow)
13. [Every bug we hit, and what it teaches](#13-every-bug-we-hit-and-what-it-teaches)
14. [Rebuild-from-scratch checklist](#14-rebuild-from-scratch-checklist)
15. [Glossary](#15-glossary)
16. [Judge Q&A preparation](#16-judge-qa-preparation)

---

## 1. The problem, and the decisions it forced

### What NTRO asked for

> *AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic.*
> Offline (Linux). Ingest bulk Bitcoin transaction + network metadata
> (CSV/JSON/XML). Correlate network-layer (IP/port/timing) with blockchain-layer
> (wallet/TXID/amount) data. Apply AI/ML to detect anomalies, cluster entities,
> and generate ranked, explainable investigative leads. Present via a dashboard.

### The one-sentence reframing that shaped everything

> **NTRO does not want a crypto explorer. They want an investigative
> lead-generation engine: feed it bulk traffic, get back a ranked, explainable
> shortlist of suspicious entities with evidence — fully offline.**

Every design decision below follows from that sentence. A tool that shows you
*everything* is useless. A tool that says *"start here, and here is why"* is a
product.

### The three decisions that mattered most

| Decision | Why |
|---|---|
| **Correlate the two layers, don't just analyse the chain** | The chain alone cannot tell you *who* controlled a wallet. The network layer alone cannot tell you *what* was moved. The PS explicitly asks you to fuse them, and almost nobody does. This is the moat. |
| **Generate the dataset with planted ground truth** | Every team says "trust us, it's AI". Almost none can state a precision/recall/AUC, because real data has no answer key. Because *we generate the data, we hold the answer key* — so we can report real, measured accuracy. |
| **Freeze the data contract in hour one** | Six people building six components is not a coding problem, it is an **integration** problem. The #1 way teams fail is that the pieces don't fit until hour 30. A frozen schema fixes that before it can happen. |

### Why we deliberately made some things *simpler*

With a 24-hour budget and a mostly-beginner team, an ambitious stack is a
liability. We swapped:

| Wanted | Used instead | Reason |
|---|---|---|
| XGBoost + SHAP | `RandomForest` + `feature_importances_` | Same job, one dependency, no separate explainability library to install and debug |
| HDBSCAN | common-input heuristic + connected components | A domain algorithm we can fully explain, instead of a clustering library we can't |
| DuckDB | pandas | One less concept to learn under time pressure |
| GraphSAGE / node2vec | graph-derived numeric features | Days of work and a torch install, for a marginal gain |

**All of it becomes "future work" — which sounds ambitious rather than unfinished.**

---

## 2. Bitcoin domain concepts

You cannot build this without understanding the data. Here is the minimum.

### Addresses

A Bitcoin address is like an account number. It is a string (`bc1q...`). It is
**not** tied to a name, which is why Bitcoin is pseudonymous rather than
anonymous — every transaction is public forever, but addresses are not labelled.

### UTXOs and transactions

Bitcoin has no "balance" field. Instead there are **unspent transaction outputs**
(UTXOs): chunks of coins, each locked to an address.

To pay someone you:
1. **select** one or more UTXOs you control (these become the **inputs**),
2. **create** new outputs: one (or more) to the recipient, and usually one back
   to yourself with the leftover.

```
   INPUTS                          OUTPUTS
   0.6 BTC (yours) ──┐
                     ├──▶  1.0 BTC  → recipient
   0.5 BTC (yours) ──┘     0.099 BTC → back to you (CHANGE)
                           0.001 BTC → miner fee
```

**You must sign for every input you spend.** This single fact is the foundation
of all blockchain analysis — see §5.

### Change outputs — and why they matter to us

The output that comes back to you is **change**. It is not a payment; you still
own it. Failing to distinguish change from payment will:
- inflate every wallet's apparent activity, and
- make peel chains invisible (a peel chain is *mostly* change).

Our correlation engine separates them explicitly: an output whose address
belongs to the sending entity is change, not a transfer.

**This realism matters for honest metrics.** Our first generator only produced
change on peel-chain transactions, which made `change_ratio` a perfect
give-away. The model scored AUC 1.000 on a technicality. We fixed it by making
ordinary payments produce change too — as they do in real Bitcoin — and the
metrics became credible. *A perfect score is usually a bug, not a triumph.*

### CoinJoin (mixing)

A CoinJoin is a transaction where **many people contribute inputs and receive
equal-value outputs**, so an observer cannot tell whose output is whose:

```
Participants A,B,C,D,E each contribute 0.5 BTC  ──▶  5 outputs of 0.5 BTC
                                                      (unlinkable to inputs)
```

The purpose is to break the ownership trail. Detecting it matters because:
- It is a strong laundering indicator.
- **It deliberately breaks the common-input heuristic** (§5) — which is exactly
  why we detect it *first* and exclude it.

### Peel chains

A peel chain moves a large balance through a series of wallets, shaving a little
off at each hop and forwarding the rest:

```
A ──62.0──▶ B ──61.4──▶ C ──58.1──▶ mixer
```

The signature: at each hop the sender keeps most of the value as **change** and
sends only a small "peel" onward. Measurable as a high change ratio combined
with a very narrow set of outbound counterparties.

### Typologies we plant

| Typology | Structure | What it looks like in features |
|---|---|---|
| Exchange | many→one deposits, one→many withdrawals | high fan-in **and** fan-out, huge tx count |
| Common-input cluster | addresses co-spent repeatedly | one connected component |
| Peel chain | forward-most-value, peel-a-little, repeat | high change ratio, narrow out-degree |
| CoinJoin / mixer | many equal inputs → many equal outputs | output value uniformity ≈ 1.0 |
| Ransomware collector | many victims → one wallet → split out | extreme fan-in, low fan-out |
| Cross-border control | wallets broadcast from several countries | country_count ≥ 2 |

---

## 3. Python: environments and project structure

### Virtual environments (`venv`)

A virtual environment is a private folder of packages for one project.

**Why it matters:** without it, installing `pandas` for this project could
upgrade (and break) the pandas another project depends on. With it, each project
has its own isolated set.

```bash
python -m venv .venv                 # create
.venv\Scripts\activate               # activate (Windows)
source .venv/bin/activate            # activate (Linux / macOS)
```

`python -m venv` means "run the `venv` module using this interpreter" — which
guarantees the environment is built from the Python you just invoked. This
matters on a machine with several versions installed.

### `requirements.txt` and pinning

```
pandas==2.1.4
scikit-learn==1.6.1
```

**Pinned** means exact versions. Unpinned (`pandas>=2`) means "whatever is
newest", which means the environment you tested is not the environment your
teammate gets, and not the environment you ship.

For an air-gapped deployment this is doubly important: you must be able to
download every wheel *in advance* and install with no network:

```bash
pip download -r requirements.txt -d wheels/            # on a connected machine
pip install --no-index --find-links=wheels/ -r requirements.txt   # offline
```

### Packages, modules, `__init__.py`

- A **module** is a `.py` file.
- A **package** is a folder of modules containing `__init__.py`.

`__init__.py` (even empty) is what lets you write `from ml.risk import RiskModel`
instead of fiddling with file paths. It marks the folder as importable.

### The layering, and why it exists

```
generator/    creates data
ingestion/    reads + validates data
correlation/  fuses the two layers
ml/           features + models
backend/      serves it all
pipeline_run.py  wires them together
```

Each layer **only knows about the layers beneath it**. `ml/` never imports
`backend/`. This is called *layered architecture*, and the payoff is concrete:

- You can test the ML without starting a web server.
- You can run the pipeline from the command line, from the API, or from a test.
- Changing the UI cannot break the analysis.

**The counter-example that shows why it matters:** if we had put the pipeline
logic inside the FastAPI route handler, then testing the analysis would require
importing a web framework, and running it offline in batch would require a
server. `pipeline_run.py` exists precisely to keep that separation.

### Why `tasks.py` instead of a Makefile

`make` is not installed on Windows by default. The team practises on Windows and
ships on Linux. So `tasks.py` — using only the standard library — gives **one
command vocabulary that works identically on both**:

```bash
python tasks.py gen | train | analyze | run | smoke | test | clean
```

The lesson is general: **pick tooling that works on every machine your team
uses**, not the tooling that is conventional on yours.

---

## 4. Data engineering: synthetic data and ground truth

### Why generate the data at all?

Three reasons, in order of importance:

1. **Ground truth.** Real data has no answer key. Without labels you cannot
   report precision, recall, or AUC — you can only *claim*. Generating the data
   means we know exactly which entities are illicit, so every metric we report is
   measured rather than asserted.
2. **Legality and privacy.** The PS forbids real seized data. Synthetic data has
   no privacy exposure.
3. **Reproducibility.** A fixed random seed means the same dataset every time, so
   a metric change means a *code* change.

### The discipline that makes it work

The generator writes **two classes of file**:

| File | Purpose | Visible to the model? |
|---|---|---|
| `transactions.csv` | the uploadable traffic dump | ✅ yes |
| `ground_truth_entities.csv` | entity → label + typology | ❌ **never** |
| `ground_truth_addresses.csv` | address → owning entity | ❌ **never** |
| `ground_truth_transactions.csv` | txid → source/dest entity | ❌ **never** |

**The rule:** the ground truth is read *only* by `ml/evaluate.py` to score. If it
ever leaks into feature building, the metrics become fiction. This is the single
most common way a hackathon ML demo becomes dishonest, usually by accident.

### Making the dataset *honest* rather than flattering

A naive generator produces data where the patterns are trivially separable, and
the model scores 100% on everything. That is not an achievement, it is a
give-away, and a judge will spot it.

So the generator plants two things that exist purely to make evaluation real:

**Decoy legitimate services.** Payment processors and mining pools with enormous
volume, hundreds of counterparties, and multi-country infrastructure. They look
statistically *suspicious* but are lawful. Without them, "busy" and "criminal"
would coincide and precision would be meaningless.

**Diluted signatures.** Real launderers also buy coffee. Our planted illicit
wallets also make ordinary payments, mixed into the background traffic with a
fixed probability. Without this, a peel chain would be the wallet's *only*
activity and trivially detectable.

The general principle: **test your model against something that could plausibly
fool it**, or you are measuring your generator, not your model.

### Validation and rejection

Real intake is messy. `ingestion/normalize.py` checks every record and
**rejects with a machine-readable reason**:

```python
if len(clean["input_addresses"]) != len(clean["input_amounts"]):
    return None, "input addresses/amounts length mismatch"
```

Why reject rather than coerce? Because in an investigation, **silent data loss is
indefensible**. The analyst must be able to see "3 records were unparseable and
here is why". The loader returns a `LoadReport` carrying those counts, which the
dashboard displays after an upload.

### pandas concepts used

```python
df[df["risk"] > 50]                    # boolean masking — filter rows
df.groupby("src")["value"].sum()       # split-apply-combine
df.merge(other, on="entity_id")        # SQL-style join
df.sort_values("risk", ascending=False)
np.log1p(x)                            # log(1+x): compresses huge ranges,
                                       # and keeps 0 at 0
```

`np.log1p` matters here: Bitcoin values span nine orders of magnitude
(0.0001 to 10,000 BTC). On a raw scale one whale dominates every decision split.
Log-compressing lets the model see *relative* size.

---

## 5. Graph algorithms: entity clustering

### The common-input heuristic

**You cannot sign for an address you do not control.** So if several addresses
appear as inputs to the *same* transaction, one person controls all of them.

That is the entire heuristic, and it is the bedrock of real blockchain
intelligence.

```
Tx1 inputs: A, B          → A and B are the same person
Tx2 inputs: B, C          → B and C are the same person
Therefore A, B, C are one entity
```

### Connected components

Turning pairwise links into groups is a classic graph problem. Build a graph
where nodes are addresses and edges are "co-spent together", then find
**connected components** — maximal sets of nodes where every node can reach
every other.

```python
import networkx as nx
graph = nx.Graph()
graph.add_edge(anchor, other)      # star, not clique — see below
components = list(nx.connected_components(graph))
```

**A performance detail worth knowing:** we add a *star* (first address to each
other) rather than a *clique* (every pair). Connected components are identical,
but a transaction with 50 inputs costs 49 edges instead of 1225. At 50k
transactions that is the difference between instant and slow.

### The known failure — and why we handle it first

CoinJoin deliberately gathers inputs from **many different owners** into one
transaction. Applied naively, the heuristic would merge dozens of unrelated
people into one giant fake entity.

So we detect coordinated transactions **first** and exclude them:

```
Transactions                                    Entities found
Naive (CoinJoin included):  992 addresses  →    305  (largest: 73 addresses)
Correct (CoinJoin excluded): 992 addresses  →   343  (largest: 12 addresses)
```

That comparison is worth showing a judge. It is a measured demonstration that we
understand the limits of our own method.

### Evaluating clustering: Adjusted Rand Index

How do you score a clustering when you have the truth? The **Rand Index** counts
pairs of items that are grouped consistently in both partitions. The
**Adjusted** Rand Index corrects for the agreement you would get *by chance*:

- `1.0` = perfect agreement
- `0.0` = what random labelling would score
- `< 0` = worse than random

Chance-correction is why we use ARI rather than plain accuracy: with 300+ small
clusters, an accuracy score would look impressive while being meaningless.

---

## 6. Machine learning

### The unit of analysis is the entity, not the transaction

An investigator does not care that "transaction 8f3a was odd". They care that
"this wallet cluster looks like a laundering pipeline". So we aggregate from
transactions up to entities, and everything downstream scores **entities**.

### Feature engineering is the highest-leverage step

A model is only as good as its inputs. Given the same `RandomForest`, the analyst
who expresses the right *domain* features beats the one who throws raw columns at
it. Our 24 features fall into four groups:

| Group | Examples | Domain meaning |
|---|---|---|
| **Scale** | `tx_count`, `log_value_btc`, `value_per_tx` | how big is this actor? |
| **Topology** | `fan_in`, `fan_out`, `in_out_ratio` | what *shape* is its counterparty graph? |
| **Network layer** | `ip_count`, `country_count`, `asn_count`, `burst_score` | *where* was it controlled from? |
| **Structural detectors** | `peel_score`, `mixer_score`, `output_uniformity`, `collector_score` | what does its behaviour look like? |

Note the third group: **those features exist because of the correlation layer.**
On-chain data has no geography. `country_count` is only computable because we
joined the IP a wallet broadcast from. That is the fusion, expressed as a number
a model can use.

### Label leakage — the mistake that invalidates everything

**No feature may be computed from the answer.**

If you compute `typology == "peel_chain"` and then feed that to the model, you
are not predicting anything — you are reading the answer key aloud. The reported
accuracy becomes fiction.

In our codebase this is enforced by structure: `ml/features.py` never imports
anything from the ground-truth files. The labels are joined *only* inside
`ml/train.py` (to fit) and `ml/evaluate.py` (to score).

### Train/test split, and why you must stratify

Training and evaluating on the same data measures memorisation, not skill. So we
hold out a portion:

```python
train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=labels)
```

**`stratify=labels`** keeps the class ratio the same in both halves. Without it,
a small positive class (8% here) can land entirely in one split — and then the
test set has no positives, recall reports 0.0, and the model looks broken when
nothing is wrong.

### Model 1: `IsolationForest` (unsupervised anomaly detection)

**The idea is beautifully simple: anomalies are easy to isolate.**

Build many random decision trees. At each node, pick a random feature and a
random split point. A normal point sits in a dense region, so it takes *many*
splits to separate it from its neighbours. An outlier sits far out on its own, so
a few random splits isolate it.

> Average number of splits needed, over the forest → the anomaly score.
> **Fewer splits = more anomalous.**

Why it belongs here: it is the honest answer to *"what if the criminals do
something you did not plant?"* It needs no prior example of a pattern to notice
that something is unusual. And it is the strongest answer to *"is this just
if-else rules?"* — this model was never told what an anomaly looks like.

**Scoring detail.** We normalise by *rank* rather than min-max. Min-max is
fragile: one extreme outlier squashes everyone else toward zero, so scores stop
being comparable between runs. A rank-based percentile is stable and reads
naturally to an analyst ("more anomalous than 97% of the others").

### Model 2: `RandomForestClassifier` (supervised risk scoring)

A random forest is many decision trees, each trained on a random subset of rows
and features, whose votes are averaged. That randomness is what makes it robust:
individual trees overfit in different directions, and the errors cancel.

Why it fits this problem:
- Learns non-linear interactions a linear model cannot ("high fan-in matters,
  but only when counterparties are also low-volume").
- Robust to unscaled, skewed features — no feature-scaling pipeline to get wrong.
- Gives `feature_importances_` for free.

### Class imbalance

Illicit entities are ~8% of the data. A lazy model gets 92% "accuracy" by
predicting "clean" for everything — and catches zero criminals.

```python
RandomForestClassifier(class_weight="balanced")
```

`class_weight="balanced"` re-weights the classes inversely to their frequency, so
the forest cannot ignore the minority class. **Accuracy is the wrong metric for
imbalanced problems** — which is why we never report it.

### The metrics, and what each one means

| Metric | Question it answers | Why it matters here |
|---|---|---|
| **Precision** | Of the entities I flagged, how many were real? | Low precision = analyst wastes time on innocent wallets |
| **Recall** | Of the real ones, how many did I catch? | Low recall = criminals missed. **The dangerous failure.** |
| **F1** | Harmonic mean of the two | Single number when you need one |
| **ROC-AUC** | Can the model *rank* better than chance? | Threshold-independent. 0.5 = coin flip, 1.0 = perfect |
| **Precision@k** | Of my top k, how many are real? | The right metric for a *ranking* used as a shortlist |
| **ARI** | Did clustering recover the true grouping? | Chance-corrected partition agreement |

**Why we always report precision AND recall together:** they trade off, and which
one matters is a *policy* question, not a technical one. For lead generation a
false positive costs an analyst five minutes; a false negative can cost a case.
So we lean toward recall — and we say so out loud, rather than quietly tuning
whichever number flatters us.

### Explainability, honestly

We produce per-entity reasons:

```
attribution_i = global_importance_i × z_score(x_i)
```

Meaning: *"this feature matters to the model overall, AND this entity is unusual
on it."* Positive pushes toward illicit; negative argues against.

**This is not SHAP.** It is a first-order approximation, it is not
game-theoretic, and we do not claim otherwise. Being precise about this is
exactly what earns credibility when a judge presses on explainability — and it
saves installing and debugging a whole extra library under time pressure.

We also surface **exculpatory** reasons: features whose value argues *against*
suspicion are shown as `info` severity. A tool that only ever shows incriminating
factors is not trustworthy.

---

## 7. The backend (FastAPI)

### What an API is

An API (Application Programming Interface) here means: the browser asks for data
over HTTP, the server answers with JSON.

```
GET  /health          → {"status": "ok"}
POST /upload          → store a file, report what parsed
POST /analyze         → start a job, return an id immediately
GET  /job/{id}        → progress + log lines
GET  /results         → the results.json payload
GET  /report/{id}     → a printable HTML case dossier
```

**Status codes matter** and are part of the contract:
- `200` OK · `202` Accepted (work started, not finished) · `400` your request was
  malformed · `404` not found · `422` understood but unprocessable · `500` we
  broke.

### FastAPI and pydantic

FastAPI uses **type hints** to do three things at once:

```python
@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
```

1. Parse and validate the incoming request.
2. Serialise the response to JSON.
3. **Generate interactive documentation at `/docs`** — which is genuinely useful
   in a demo: you can show a judge a live API console.

`pydantic` is the validation engine underneath. It turns Python classes into
validators, so "the client sent a string where a number was required" becomes an
automatic 422 instead of a crash deep in your code.

### Why analysis runs as a background job

A full analysis takes ~35 seconds. Holding an HTTP request open that long means
the browser times out and the user stares at a spinner with no feedback.

So the API works the way a serious tool works:

```
POST /analyze   → returns a job id IMMEDIATELY (202 Accepted)
GET  /job/{id}  → poll for progress and log lines
                  status: queued | running | done | error
```

This also gives the dashboard the thing that makes a demo feel alive: a progress
bar and a scrolling log.

**Thread-safety.** The analysis runs on a worker thread while the polling
endpoint reads job state from the event loop thread. Without a `Lock` around the
job store, the two race and the UI can read a half-written job. This is not
theoretical — it is why `JobStore` has `threading.Lock`.

**Scope decision, written down:** the job store is in-memory, not Redis. For a
single-process, offline, air-gapped tool that is the *correct* choice — no
external service to install, nothing to go wrong on stage. The trade-off (state
lost on restart, single process only) is documented in the file so nobody has to
rediscover it.

### Serving the UI from the same process

```python
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)))
```

One process serves both the JSON API and the dashboard's files. Consequences:
deployment is one command, there is **no CORS to configure** (same origin), and
there is no reverse proxy to misconfigure. For an air-gapped target, every moving
part you remove is a failure mode you cannot have.

### A security detail worth getting right

```python
def _safe_filename(name: str) -> str:
    cleaned = Path(name).name.strip() or "upload.csv"
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in "._-")
```

Uploaded filenames are attacker-controlled. Without this, a file named
`../../etc/passwd` could escape the upload directory. **Path traversal** is a real
vulnerability class and this is a two-second fix.

---

## 8. The frontend

### The architecture principle

**Analysis lives in the backend; presentation lives in the frontend.**

The frontend contains *no investigation logic at all*. It has exactly one job:

```
fetch results.json  →  ADAPT it to what the DOM expects  →  render
```

### The adapter pattern

The backend speaks in investigation terms (`entities`, `risk_band`, `typology`,
`control` edges). The DOM speaks in presentation terms (groups, CSS classes,
colours, pixel widths). Rather than pollute the analyst-facing schema with UI
concerns, one explicit translation layer sits between them:

```js
function adaptResults(raw) {
  return {
    alerts: (raw.alerts || []).map(a => ({
      sev: bandToSev(a.severity),   // critical -> crit (a CSS class)
      node: a.entity,               // entity -> the id the graph selects
      ...
    })),
  };
}
```

The payoff: the contract can evolve without breaking the UI, and the UI can be
restyled without touching the API.

### `fetch`, promises, async/await

```js
const response = await fetch('/results');
const body = await response.json();
```

`fetch` returns a **Promise** — a placeholder for a value that does not exist
yet. `await` pauses *this function* (not the browser) until it resolves. This is
what lets you write asynchronous code that reads top-to-bottom.

**A gotcha that bites everyone:** `fetch` does **not** reject on HTTP errors. A
404 resolves normally. You must check `response.ok` yourself — otherwise your
error handling silently never runs.

### Escaping output (XSS)

```js
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({...}[c]));
```

Any value interpolated into `innerHTML` must be escaped. We generate our own
data, so today this is belt-and-braces — but the moment a real feed can contain a
crafted address, an unescaped interpolation becomes an injection bug. **Correct
habits cost nothing and survive contact with real data.**

### vis-network and canvas rendering

The graph is drawn to an HTML **canvas** — a bitmap, not DOM elements. This has
two consequences you must plan for:

1. **`document.querySelector` cannot find graph nodes.** They are pixels. You
   interact through the library's API (`network.on('click', ...)`), not the DOM.
2. **Screenshots are the only way to verify it renders.** Static analysis of the
   HTML tells you nothing about what is on the canvas.

### Layout: settle, then freeze

```js
physics: { enabled: true, stabilization: {...} }
network.once('stabilizationIterationsDone', () => {
  network.setOptions({ physics: false });
});
```

The prototype used hardcoded coordinates because its graph was fixed. Real data
needs a layout — so we run physics briefly to settle it, then **freeze it**. A
graph that keeps drifting while you are talking to a judge is unusable.

### Graph readability is a real engineering problem

Our first real graph drew 70 nodes and **4,237 edges** — an unreadable hairball.
Three fixes, in order of impact:

1. **Aggregate parallel edges.** Every transfer between the same pair collapses
   into one edge carrying a `count`. That is what the `count` field is *for*.
2. **Cap the total.** Beyond ~130 edges everything becomes a grey smudge.
3. **Reduce edge opacity.** At full strength, 130 crossing edges form a solid
   mass that hides the nodes. Dropping to ~0.42 is the single biggest
   readability win.

Plus: label only the important nodes (services and the highest-risk wallets),
and encode risk in node *size*. **In a link-analysis view the structure is the
point — see the shape first, read the labels on what matters.**

### The offline vendoring trap

The prototype loaded vis-network, Chart.js and Google Fonts from CDNs. Air-gapped,
a CDN load **fails silently** and you get a blank dashboard — the most common way
an offline demo dies.

Fix: vendor every asset locally, and **assert it in CI**:

```python
check("cdnjs" not in response.text, "no CDN script references (offline-safe)")
```

That test is worth more than a README paragraph, because it fails the build the
moment someone pastes a CDN link back in.

### Removing fabricated statistics

The prototype displayed eight hardcoded statistics: "12.4K wallets clustered",
"847K tx analyzed", "184.6 ₿", "▲ 5 vs last run". It also had a static banner
reading *"Seed wallet → 3 peel hops → CoinJoin mixer → cash-out exchange,
confidence 0.87 · 6 hops"*.

**Hardcoded text like that is worse than a missing feature.** It asserts specific,
plausible-sounding findings during a live demo while being entirely invented. In
an investigative tool, **a number is evidence.**

All eight are now derived from the analysis, and the trace banner is rebuilt from
the selected entity's real timeline. Where a figure genuinely cannot be computed,
we remove it rather than guess.

---

## 9. Contracts: the thing that saves six-person teams

### The problem

Six people, six components, 24 hours. The failure mode is not bad code — it is
**components that don't fit together until hour 30**, when there is no time left.

### The fix: freeze the interface in hour one

`schemas/results.schema.json` is **the handshake**. Everyone codes against it from
minute one.

```json
{
  "entities": [ { "id": "E-0007", "kind": "mixer", "risk": 92, "risk_band": "critical", ... } ],
  "edges":    [ { "from": "E-0007", "to": "E-0031", "kind": "flow", "value": 3.2 } ],
  "alerts":   [ { "id": "AL-01", "entity": "E-0007", "severity": "critical", ... } ],
  "metrics":  { "risk_auc": 0.94, ... }
}
```

It buys three things:

1. **Parallelism.** All three squads build simultaneously instead of waiting.
2. **A stub is enough.** The platform squad builds the entire dashboard against a
   hand-written fixture (`tests/fixtures/sample_results.json`) while the ML squad
   is still training. The UI does not care whether the JSON came from a model or a
   text file.
3. **Automated enforcement.** `jsonschema` turns the contract into a *test*. Drift
   fails the build immediately — not at 3am on demo day.

### JSON Schema

JSON Schema is a language for describing what a JSON document must look like:

```json
"risk": { "type": "integer", "minimum": 0, "maximum": 100 },
"kind": { "type": "string", "enum": ["wallet", "exchange", "mixer", "seed", "ip"] },
"additionalProperties": false
```

**`additionalProperties: false` is the important one.** It means *unknown fields
are rejected*. Without it, someone adds `riskScore` alongside `risk`, the schema
still validates, and now two fields mean the same thing and half the code reads
the wrong one. Silent drift is exactly what a contract exists to prevent.

### Design rules the schema encodes, and why

| Rule | Reasoning |
|---|---|
| `entities` is **unified** — wallet clusters *and* IP endpoints, distinguished by `kind` | An IP endpoint *is* a subject of interest. One list keeps the graph builder trivial |
| `edges` carry `kind: "flow" \| "control"` | Money movement and network control are **different relationships**. Conflating them misrepresents the evidence — and the dashboard draws them differently (solid vs dashed) for the same reason |
| **Analysis in the backend, presentation in the frontend** | The schema says flow is *a value transfer*; the UI decides that means a solid orange line |
| `metrics` **allows null** | A missing metric means "not yet measured". It does **not** mean "put a placeholder here". Never fake a number in this block — it is the most judge-scrutinised part of the payload |

---

## 10. Packaging and offline deployment

### What Docker actually is

Docker bundles your code *together with its entire environment* — the OS
libraries, the Python version, every dependency — into one image.

The problem it solves: "it works on my laptop" happens because your laptop has
the right Python, the right system libraries, the right environment variables.
Shipping the whole environment means the target machine's state stops mattering.

```dockerfile
FROM python:3.10-slim       # a base image with Python already installed
WORKDIR /app
COPY requirements.txt ./     # deps first — separate layer, cached
RUN pip install -r requirements.txt
COPY . .                     # code second — edits rebuild fast
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching is why the order matters.** Docker caches each instruction. Put
`requirements.txt` before your code and an edit to a `.py` file rebuilds in
seconds rather than reinstalling scipy.

### Baking data and models into the image

```dockerfile
RUN python tasks.py gen && python tasks.py train && python tasks.py analyze
```

This is what makes the demo bulletproof: `docker run` on an air-gapped machine
produces a fully populated dashboard with **no setup step and no network**. It
also means the reported metrics travel *with* the artifact, so the shipped tool
and the claimed numbers cannot drift apart.

### Healthchecks

```dockerfile
HEALTHCHECK CMD python -c "...urlopen('http://127.0.0.1:8000/health')..."
```

A healthcheck distinguishes *"the container is running"* from *"the tool works"*.
Those are different things — a process can be alive and serving 500s.

### The air-gap strategy, in three layers

1. **Vendored frontend assets** — vis-network, Chart.js, 42 font files, all
   local. Zero CDN references, asserted by CI.
2. **A pre-downloaded wheel cache** — `pip download -r requirements.txt -d wheels/`
   on a connected machine, then `pip install --no-index --find-links=wheels/`
   offline. The Dockerfile picks this path automatically when `wheels/` exists.
3. **Baked dataset and models** — nothing to compute at first run.

The result: the machine needs no network for the build *or* the run.

### Practical lesson from this project

**Test the network before assuming it is fast.** Setting up the ML stack took
~90 seconds per 2 MB on the practice machine (~25 KB/s). We discovered this by
measuring, and it changed the plan: we used an already-installed Python 3.10
instead of downloading a new interpreter, and it is why the wheel cache exists at
all. *Measure, don't assume.*

---

## 11. CI/CD

### CI: Continuous Integration

**Every time anyone pushes, automatically prove the project still works.**

```yaml
on: [push, pull_request]
jobs:
  verify:
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10", cache: pip }
      - run: pip install -r requirements.txt
      - run: python -m tests.validate_contract --fixture
      - run: python tasks.py gen && python tasks.py train && python tasks.py analyze
      - run: python -m tests.validate_contract out/results.json
      - run: python -m pytest -q tests/test_smoke.py
```

Why it matters for a six-person team specifically: **the contract is only as good
as its enforcement.** CI is what makes "the contract is sacred" a fact rather
than an aspiration. If any squad's change makes the output drift, the build goes
red immediately — while there is still time to fix it.

It also mirrors exactly what a developer runs locally (`python tasks.py smoke`),
so a green build means the same thing it means on a laptop.

**A rule we committed to: set CI up ONCE, early, then leave it alone.** A CI
pipeline must not eat the middle of a 24-hour hackathon.

### CD: what "deployment" means for an air-gapped tool

There is no server to deploy to. The target is an isolated Linux host that cannot
pull anything from the internet. So the useful deliverable is a
**self-contained bundle the agency can receive on physical media**:

```
netra-v1.0.tar.gz
├─ code + schemas + tests
├─ wheels/            ← offline install cache
├─ data/  models/     ← dataset + trained models
├─ frontend/vendor/   ← offline JS + fonts
└─ run.sh             ← works with no network
```

Plus `docker save` output, because the receiving host cannot pull from a registry
either. **Framing CD this way is a stronger answer than pretending you deployed
to Kubernetes** — it shows you understood the actual constraint.

### Release artifacts prove provenance

```yaml
echo "${GITHUB_SHA}" > dist/$NAME/COMMIT
```

Every release carries the commit that produced it, so the shipped artifact and
the reported metrics provably come from the same code.

---

## 12. Git workflow

### Trunk-based development with short-lived branches

- **`main` is protected.** Nobody pushes to it directly.
- **Branch → PR → one review → squash-merge.** Small PRs, open for hours not days.
- **Branch naming by squad:** `forge/synthetic-gen`, `brain/risk-xgb`,
  `netra/graph-wire`.
- **`main` must always run.** If your branch breaks the smoke test, it does not
  merge.

### Why the *review* matters more than the branching

The branch structure is not the point. The point is that **nobody works alone on
a critical path** — a second pair of eyes catches the class of mistake that costs
hours. In this project alone, review-equivalent checks caught: a generator that
produced only single-input transactions (making clustering impossible), IPs with
three octets, and a `mixer_score` that labelled innocent CoinJoin participants as
mixing services. Each would have been expensive to find at hour 20.

### Commit messages

Say *what changed and why*. You will thank yourself during the write-up, and a
judge looking at your history sees engineering discipline rather than a pile of
"fix" and "wip".

### Repo hygiene for judging

Clean `README.md` with an architecture diagram · `LICENSE` · no secrets · a
tagged release commit (`v1-demo`) before the freeze · no committed data or model
binaries (`.gitignore` them — the *generator* is committed, its output is not).

---

## 13. Every bug we hit, and what it teaches

This is the most valuable section. These are real defects found by running the
code, not by reading it. Each one would have cost hours at the event.

### 1. The generator produced one input per transaction

**Symptom:** 925 addresses clustered into 925 singleton entities. Clustering ARI
would have been ~0. Ground truth was unrecoverable.

**Cause:** `in_addrs = [choice(src.addresses)]` — always exactly one input.
But the common-input heuristic keys on *several addresses being co-spent*. With
one input per transaction there is nothing to link.

**Fix:** sometimes spend multiple addresses from the same entity (which is what
real wallets do — they hold several UTXOs and spend some together).

**Lesson:** if a feature has no signal in your data, suspect your **generator**
before your algorithm. *Ask "what observable fact would my algorithm need?" and
check the data actually contains it.*

### 2. IP addresses had three octets

**Symptom:** `40.52.114` instead of `40.52.114.x`. Not valid IPv4.

**Cause:** a prefix built from two octets, then one more appended.

**Lesson:** validate your generated data against its own real-world format. Cheap
assertions on synthetic data catch embarrassing errors early.

### 3. Only peel chains produced change outputs

**Symptom:** risk model precision/recall/F1/AUC all **1.000** and clustering ARI
**1.000**. Suspiciously perfect.

**Cause:** in real Bitcoin almost every payment creates a change output. Our
generator only produced change on peel-chain transactions — which made
`change_ratio` a give-away that separated the classes on a technicality.

**Fix:** ordinary payments also create change (`CLEAN_CHANGE_PROB = 0.55`), plus
decoy legitimate services and diluted illicit signatures.

**Lesson:** **a perfect score is usually a bug, not a triumph.** If your metrics
look too good, hunt for the leak before you print the slide.

### 4. `mixer_score` labelled CoinJoin *participants* as mixers

**Symptom:** 8 entities classified as mixing services when only 3 were planted.
The dashboard showed innocent participants as "CoinJoin mixer".

**Cause:** every owner of every address in a coordinated transaction was treated
as a mixer — inputs *and* outputs. But the input side is the *participants*; the
output side is the *service*.

**Fix:** use the asymmetry. The mixer is the entity receiving the pooled
equal-value **outputs**. Subtract exclusive input-side participants.

**Lesson:** distinguish "is X" from "interacts with X". Telling an investigator
that a victim's wallet *is a mixer* is false evidence, and a feature named
carelessly will encode the wrong concept.

Also: changing a feature's meaning **invalidates the trained model**. We
retrained immediately. A stale model silently mispredicts.

### 5. 4,237 edges for 70 nodes

**Symptom:** the graph rendered as an unreadable hairball; the payload was 1.35 MB.

**Cause:** we emitted one edge *per transaction* touching a visible node.

**Fix:** aggregate every transfer between the same pair into one edge with a
`count` — which is exactly what the contract's `count` field was for — then cap
the total and reduce edge opacity.

**Lesson:** a visualisation that cannot be read is not a feature. And when a
schema field exists (`count`), it usually exists *because* someone already
anticipated this problem.

### 6. Mixing services fell out of the graph window

**Symptom:** the graph contained no mixer node at all, so the demo narrative
("peel chain into a mixer") had nothing to show.

**Cause:** neighbours were ranked purely by BTC value. The mixers carry modest
volume next to a busy payment processor, so they lost the budget.

**Fix:** rank neighbours by **investigative significance first** (is this a
service the leads connect to?) and value second.

**Lesson:** "most important" and "largest" are different orderings. Choose the
ranking that serves the *user's question*, not the one that is easiest to compute.

### 7. `make` is not installed on Windows

**Symptom:** the documented setup command simply failed.

**Fix:** `tasks.py`, a standard-library task runner that works identically on both
platforms.

**Lesson:** pick tooling that works on **every** machine your team uses.

### 8. `TestClient(app)` silently skipped the lifespan

**Symptom:** the API smoke test printed "no analysis loaded" and *passed* — while
testing nothing.

**Cause:** `TestClient(app)` does not run FastAPI's start-up hooks. Our start-up
hook is what restores the cached analysis, so the app genuinely had no results.

**Fix:** use it as a context manager — `with TestClient(app) as client:`.

**Lesson:** **a test that silently skips is worse than no test**, because it
manufactures false confidence. Any "skip" path deserves a hard look.

### 9. A stale server served old analytics

**Symptom:** the dashboard showed analysis from 16 minutes earlier. The data was
valid — just old — so nothing looked wrong.

**Cause:** the server loads `results.json` once at start-up. A `pkill` had failed,
an old process still held the port, and the replacement silently never bound.

**Fix:** an explicit `POST /reload` endpoint that reports the payload's own
`generated_at`, so you can always tell exactly how old what you are looking at is.

**Lesson:** "serving stale data" is a genuine hazard in an investigative tool, and
it is invisible unless you surface *when* the data was produced. Also: when
restarting a server, verify the new process actually bound the port — don't assume
the kill worked.

### 10. `minimum` nested inside a `type` array (invalid JSON Schema)

**Symptom:** the validator crashed instead of validating.

**Fix:** `"type": ["integer", "minimum"]` → `"type": "integer", "minimum": 1`.

**Lesson:** the contract validator caught its own schema's bug. That is the
system working as designed — **validate your validator**.

### The meta-lesson

**Nine of these ten bugs were found by *running* the code and looking carefully at
the output — not by reading it.** Static review would have caught none of them.

So: build the vertical slice early, run it constantly, and look hard at the
numbers. Suspiciously good results and suspiciously empty output are both signals.

---

## 14. Rebuild-from-scratch checklist

For the hackathon. In order. Do not skip the verification steps.

### Hour 0–2 — Setup and contract

- [ ] `python -m venv .venv` and activate
- [ ] `pip install -r requirements.txt` (or offline: `--no-index --find-links=wheels/`)
- [ ] Create the directory tree (`generator/ ingestion/ correlation/ ml/ backend/ frontend/ schemas/ tests/`)
- [ ] **Freeze `schemas/results.schema.json`** — this is the milestone
- [ ] Write `tests/validate_contract.py`
- [ ] Hand-write `tests/fixtures/sample_results.json` (the stub everyone codes against)
- [ ] Verify: `python -m tests.validate_contract --fixture`
- [ ] `git init`, protect `main`, add all six collaborators

### Hour 2–6 — The vertical slice (**the non-negotiable milestone**)

A thin version of every stage, end to end:

- [ ] Generator: 50 transactions, one planted peel chain
- [ ] Ingestion: CSV → normalized records
- [ ] Correlation: trivial entity↔IP join
- [ ] Features + one dumb detector → `results.json`
- [ ] FastAPI serving `/analyze`
- [ ] Dashboard reads the API and renders the graph, alerts, geo
- [ ] Verify: upload a file, see real results on screen

**If this is not green by ~25% of the clock, stop adding features and get it
green.** An ugly full pipeline beats three beautiful disconnected pieces.

### Hour 6–16 — Deepen (squads in parallel)

- [ ] Full generator: 20k–50k transactions, all typologies, decoys, dilution
- [ ] Ingestion for CSV + JSON + XML with rejection reporting
- [ ] Correlation v2: entity↔IP, burst timing, cross-border hops
- [ ] Common-input clustering + ARI evaluation
- [ ] Structural detectors: peel, CoinJoin, collector, exchange
- [ ] `IsolationForest` anomaly
- [ ] `RandomForest` risk + P/R/F1/AUC
- [ ] All API endpoints
- [ ] All dashboard views wired to live data
- [ ] Verify: `python tasks.py smoke` green

### Hour 16–20 — Explainability and metrics

- [ ] Per-entity reasons in plain English, with exculpatory ones surfaced
- [ ] Feature attributions on every lead
- [ ] Metrics surfaced on the Intel screen
- [ ] Case-report export (printable HTML → PDF)
- [ ] Verify: every number on screen traces to the analysis

### Hour 20–22 — Package

- [ ] Vendor every frontend asset; **assert zero external references**
- [ ] `Dockerfile` + `docker-compose.yml` + `run.sh` / `run.bat`
- [ ] Bake data + models into the image
- [ ] CI workflow
- [ ] Verify: **turn off Wi-Fi, run it, everything still works**

### Hour 22–24 — Freeze and rehearse

- [ ] Code freeze (T-2h): demo-critical fixes only
- [ ] Load the demo dataset; rehearse the golden path **three times**
- [ ] Record a backup screen capture
- [ ] Final README + architecture diagram; tag `v1-demo`
- [ ] Run the judge Q&A sheet (§16) out loud as a team

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Address** | A Bitcoin account number. Not tied to a name. |
| **UTXO** | Unspent Transaction Output — a chunk of coins locked to an address |
| **Change output** | The leftover returned to the sender. Not a payment. |
| **CoinJoin** | A mixing transaction where many owners contribute equal-value inputs, breaking the ownership trail |
| **Peel chain** | Layering pattern: forward most of the value, shave a little off, repeat |
| **Common-input heuristic** | Addresses co-spent in one transaction share an owner |
| **Connected components** | Maximal groups of nodes where every node reaches every other |
| **ARI** | Adjusted Rand Index — chance-corrected agreement between two clusterings |
| **IsolationForest** | Unsupervised anomaly detector: anomalies need fewer random splits to isolate |
| **RandomForest** | Ensemble of decision trees whose votes are averaged |
| **Class imbalance** | When one class is rare; makes accuracy a misleading metric |
| **Label leakage** | A feature that encodes the answer, invalidating the evaluation |
| **Stratified split** | Train/test split preserving class proportions |
| **Precision / Recall** | Of those flagged, how many were right / of the real ones, how many were caught |
| **ROC-AUC** | Probability the model ranks a random positive above a random negative |
| **Precision@k** | Precision among the top k results — the right metric for a shortlist |
| **JSON Schema** | A language for describing and validating the shape of JSON |
| **Contract** | The agreed data shape between components |
| **ASN** | Autonomous System Number — identifies the network operator behind an IP |
| **Vendoring** | Copying a dependency into your repo so you don't fetch it at runtime |
| **Air-gapped** | A machine with no network connection |
| **Wheel** | A prebuilt Python package. `pip download` gives you wheels for offline install |
| **Lifespan** | FastAPI start-up/shutdown hooks |
| **Idempotent** | Safe to run more than once with the same result |

---

## 16. Judge Q&A preparation

Rehearse these **out loud as a team**. Each squad answers its own domain.

**"Is this just if-else rules?"**
> No. Clustering is a graph algorithm and the anomaly detector is an
> unsupervised `IsolationForest` — it never saw a label. Risk is a trained
> `RandomForest`; here is its precision, recall and ROC-AUC on held-out labelled
> data, and every lead carries its feature attributions.

**"Where does your data come from? Is it real?"**
> Synthetic, but schema-identical to fused BTC + netflow — the exact fields the
> PS specifies. It carries **planted** illicit typologies with hidden labels,
> which is precisely why we can report real accuracy. No privacy exposure, no
> legal risk, reproducible from a seed.

**"How do you handle false positives?"**
> We output **ranked leads, not verdicts** — analyst-in-the-loop. We also planted
> decoy legitimate services (high-volume payment processors) specifically to test
> that the model discriminates rather than just flagging anything busy. And the
> reasons panel surfaces *exculpatory* factors too, so an analyst can dismiss a
> lead in seconds.

**"Why is your ROC-AUC 1.0? That looks too good."**
> It is high because the planted patterns are genuinely learnable from the
> engineered features, and we measured it on a held-out split. But the more
> informative number is the unsupervised detector: it never saw a label and scores
> 0.25 precision@20 against a 0.08 base rate. That is the honest measure of
> difficulty without supervision. We also deliberately made the data harder —
> ordinary wallets create change too, and launderers also make normal payments —
> because our first version scored a suspicious 1.000 everywhere.

**"Does it scale to real volumes?"**
> The graph is deliberately windowed, correlation is time-bucketed, and entity
> clustering uses a star topology rather than cliques — O(k) edges per transaction
> instead of O(k²). The demo runs on ~28k transactions in about 40 seconds; the
> design targets 100k+ in batch.

**"Why not just buy Chainalysis or TRM?"**
> Those are cloud-hosted, foreign, subscription, and black-box. NETRA is
> **offline, sovereign, auditable and customisable** — no data ever leaves the
> agency, which is the entire point of an NTRO deployment. And you can read every
> line of the analysis.

**"What about Monero / privacy coins?"**
> Out of scope for this BTC problem statement, noted as future work. Our
> correlation approach generalises to any UTXO-model chain.

**"What is genuinely novel here?"**
> The **fusion**. Most tools analyse the chain alone. We correlate the *network*
> layer — the IP, timing and geography of who actually controlled a wallet — with
> on-chain behaviour, and make every lead explainable. `country_count` is a model
> feature that literally cannot exist without that join.

**"Did you use a GNN?"**
> No — we chose not to, deliberately. With a 24-hour budget and a
> mostly-beginner team, node2vec or GraphSAGE was days of work and a torch
> install for a marginal clustering gain. We spent that time on the correlation
> fusion and on making the evaluation honest, which is where the actual advantage
> is. It is documented as future work.

---

*Built by Team Vortex for SIH 2026, PS ID SIH26146.*

*Nothing in this document is a claim we cannot demonstrate by running the code.*
