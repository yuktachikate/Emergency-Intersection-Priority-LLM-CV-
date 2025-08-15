# Emergency Intersection Priority (LLM + CV)

**Design AI that asks before it acts.**


<img width="800" height="533" alt="image" src="https://github.com/user-attachments/assets/5a69c9ad-a016-4b7d-a76a-7edea0fdae8f" />




This project turns a still image (or short video) of an intersection into a **transparent, value‑aware right‑of‑way decision** for emergency vehicles. It demonstrates how to combine:

* computer vision (vehicle + signal detection),
* a deterministic **rule engine** aligned with legal/safety norms, and
* an optional **LLM reasoning layer** (Llama‑4, Llama‑3.1/3.2, LMSYS, etc.) via any **OpenAI‑compatible** API.

It is built to highlight the core claim: **speed without context is misdirection**. A naïve model will confidently say “ambulance first,” but a responsible system **asks for missing facts** (lights on? officer directing? which country?) and explains the result.

---

## Table of contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Install & Run](#install--run)
4. [Configuration (LLM adapters)](#configuration-llm-adapters)
5. [API](#api)
6. [Rule Engine Policy](#rule-engine-policy)
7. [Vision Layer (swappable)](#vision-layer-swappable)
8. [Reproducing the Comparisons](#reproducing-the-comparisons)
9. [Explainability & Logs](#explainability--logs)
10. [Safety & Governance](#Safety--governance)
11. [Evaluation Plan](#evaluation-plan)
12. [Extending the System](#extending-the-system)
13. [Limitations](#limitations)
14. [License](#license)

---

## Features

* **Hybrid reasoning**: CV → Rules → (optional) LLM cross‑check.
* **Deterministic fallback**: If the LLM is down, the result still works and is auditable.
* **Transparent scoring**: Returns per‑vehicle scores + human‑readable reasoning + legality notes.
* **Jurisdiction aware**: Configurable tie‑breaks for right‑hand vs left‑hand traffic; pluggable policy overlays.
* **Simple API**: `POST /analyze` with an image; get JSON.
* **Portable**: pure Python (FastAPI, Pillow, Pydantic, Requests).

---

## Architecture

```
+------------------+       +------------------+       +---------------------+
|  Image / Video   |  -->  |  Vision Layer    |  -->  |  Rule Engine        |
|  (upload)        |       |  (detector stub) |       |  (deterministic)    |
+------------------+       +------------------+       +----------+----------+
                                                             |
                                                             v
                                                     +-------+-------+
                                                     |   LLM Adapter |
                                                     | (optional)    |
                                                     +-------+-------+
                                                             |
                                                             v
                                                     +-------+-------+
                                                     |  JSON Output  |
                                                     | (order + why) |
                                                     +---------------+
```

**Vision** extracts entities (ambulance, fire engine, police, presidential), lights/siren status, distance/bearing, and approximate arrival ordering.
**Rule Engine** produces a score per vehicle + ordered list with tie‑breaks (right/left‑hand rule → apparatus mass → arrival order).
**LLM Adapter** (optional) re‑reasons over the structured scene and the rule output, returning the same strict JSON schema, never free text.

---

## Install & Run

### 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pillow pydantic requests
```

### 2) CLI demo (single image)

```bash
python emergency_intersection.py --image /mnt/data/eb311da5-a63e-4850-bb81-97230c21b73d.png
```

**Output (example):**

```json
{
  "ordered_ids": ["ambulance","fire_engine","police","presidential"],
  "scores": {"ambulance":..., "fire_engine":..., ...},
  "reasoning": "Scores blend emergency level, right-of-way signals, proximity and arrival.",
  "legality_notes": "General rule: yield to emergency vehicles with lights/sirens..."
}
```

### 3) Run the API

```bash
uvicorn emergency_intersection:app --reload --port 8000
```

Then upload an image to `POST /analyze`.

---

## Configuration (LLM adapters)

The adapter supports **any OpenAI‑compatible** endpoint (e.g., Llama‑4, Llama‑3.1/3.2, LMSYS models with a compatible server).

Set environment variables:

```bash
export LLM_BASE_URL=https://<your-openai-compatible-endpoint>
export LLM_API_KEY=sk-...
export LLM_MODEL=llama-4   # or meta-llama-3.1-70b-instruct, meta-llama-3.2-90b-vision-instruct, etc.
```

If unset, the system gracefully falls back to the **Rule Engine** only.

---

## API

### Endpoint

```
POST /analyze
Content-Type: multipart/form-data
Form field: file=<image>
```

### Response schema

```json
{
  "ordered_ids": ["ambulance", "fire_engine", "police", "presidential"],
  "scores": {"<vehicle_id>": <float>},
  "reasoning": "string",
  "legality_notes": "string"
}
```

### Domain objects

**Vehicle** (internal representation extracted by Vision):

```json
{
  "id": "ambulance",
  "kind": "ambulance|fire_engine|police|presidential|civilian",
  "has_right_of_way_signal": true,
  "distance_to_conflict_m": 8.0,
  "arrival_time_s": 2.0,
  "bearing_deg": 45.0
}
```

**PolicyOutput** (final output): same as the Response schema above.

### cURL example

```bash
curl -X POST \
  -F "file=@/path/to/intersection.jpg" \
  http://localhost:8000/analyze
```

---

## Rule Engine Policy

**Base priority** (life‑safety first):

```
ambulance (100) > fire_engine (95) > police (90) > presidential (70) > civilian (10)
```

**Adjustments**:

* **+15** if `has_right_of_way_signal == true` (lights/siren/escort).
* **+ proximity**: closer to the conflict point gets a small boost.
* **+ arrival**: arriving sooner gets a small boost.

**Tie‑breaks** (in order):

1. Right‑hand rule (for right‑hand traffic) or **left‑hand rule** (for left‑hand systems).
2. Apparatus mass (fire > ambulance > police > presidential > civilian).
3. Earlier arrival.

**Jurisdiction overlay**: You can provide a YAML to override weights or tie‑break order:

```yaml
traffic_side: right    # or left
base_priority:
  ambulance: 100
  fire_engine: 95
  police: 90
  presidential: 70
  civilian: 10
bonuses:
  lights: 15
  proximity_weight: 0.2
  arrival_weight: 0.1
# optional custom tie breaks
# tie_breaks: [right_hand_rule, apparatus_mass, arrival_time]
```

---

## Vision Layer (swappable)

The demo ships with a **stub** that matches the provided image. Replace it with your preferred detector:

* **Detector**: YOLOv8, RT‑DETR, OWLv2, Grounding‑DINO (text prompts like “ambulance”, “fire engine”, “police car”, “presidential motorcade SUV”).
* **Lights/Siren**: temporal strobe detector in video or a lightweight CNN on cropped light bars.
* **Tracker**: ByteTrack/OC‑SORT to estimate arrival order and turning intent from trajectories.

Vision outputs should map to the `Vehicle` fields above.

---

## Reproducing the Comparisons

We include a small simulation script (used in the notebook) that contrasts:

* **Naïve LLM pattern answer** vs
* **Value‑aware Rule Engine** across 5 scenarios:

  * S1: US, all on call
  * S2: Ambulance **not** on call
  * S3: Police **directing**
  * S4: UK **left‑hand** traffic
  * S5: **Presidential escort** active

Artifacts are saved as CSV/JSON for sharing.

---

## Explainability & Logs

* The API returns a human‑readable `reasoning` string and `legality_notes` with each decision.
* Enable structured logs to capture: detected entities, raw scores, applied tie‑breaks, final ordering, and any LLM overrides.
* Add a `requires_human_confirmation` flag when confidence is low or policies conflict.

---

## Safety & Governance

* **Ask‑before‑act**: When key facts are missing (lights? jurisdiction?), the service should query upstream sensors or escalate.
* **Deterministic fallback**: If LLM is unavailable, you still get a defensible, auditable decision.
* **Human‑in‑the‑loop**: If an officer is directing or policies conflict, defer and annotate.
* **Data privacy**: Avoid storing raw video. Keep only anonymized detections + decision traces.

---

## Evaluation Plan

**Goals**: legality compliance, safety margin, consistency, and clarity of explanations.

**Benchmarks**:

* Scenario suite (synthetic and real) with labeled ground truth orders.
* Jurisdictional variants (US right‑hand vs UK left‑hand).
* Edge cases: simultaneous arrival, no lights, officer directing, multi‑agency incident.

**Metrics**:

* Exact match of order (Kendall‑τ / Spearman rank correlation).
* Explanation completeness (checklist coverage of lights/law/arrival/turning).
* Decision latency under load.
* Override rate (how often LLM alters rule engine) + correctness.

---

## Extending the System

* **Streamlit UI**: drag‑drop image/video + live ranking + explanation pane.
* **Policy learning**: fit weights with inverse reinforcement learning under legal constraints.
* **Audio siren detection**: mic input for siren presence and type.
* **Motorcade logic**: group multiple escort vehicles into a single flow with police control precedence.
* **On‑road deployment**: package with Docker, expose health endpoints, add circuit breakers.

---

## Limitations

* Current demo uses a **vision stub**; real‑world reliability requires robust detection and tracking.
* Jurisdictional laws vary—provide local overlays and legal review before operational use.
* LLMs may hallucinate; we constrain outputs to **strict JSON** and fall back to rules.

---



### Quick priority reminder (default, when lights/sirens are on and no officer is directing)

**Ambulance → Fire Engine → Police → Presidential → Civilian**
But context can change this order—this project is built to surface and reason over that context.
