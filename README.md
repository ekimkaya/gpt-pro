# Hallucination Detector for Legal AI

A multi-layered verification tool for law firms that use LLMs to draft
filings, memos, or research. It's engineered to catch the failure mode
that cost the *Mata v. Avianca* attorneys their sanctions: fabricated or
misused legal citations slipping into a court filing.

The tool does not try to stop the LLM from hallucinating. It assumes it
will, and wraps the output in independent verification layers so the
hallucination never reaches the court.

## Pipeline

Every call to `HallucinationDetector.check(question)` runs:

1. **Draft generation** — RAG over the firm's authoritative corpus (or
   plain generation if no corpus is provided).
2. **Citation extraction** — every kind we recognize:
   - US case law: `410 U.S. 113`, `925 F.3d 1339`, `N.Y.2d`, `Cal. App.`
   - US Code: `42 U.S.C. § 1983`
   - CFR: `29 C.F.R. § 1630.2(h)`
   - State statutes: `Cal. Penal Code § 187`
   - UK: `[2019] UKSC 4`, `[2023] EWCA Civ 456`
   - EU: `Case C-123/19`, ECLI identifiers
   - Canada: `2020 SCC 7`
   - Australia: `[2020] HCA 14`
   - Federal dockets: `1:23-cv-04456`
   - Secondary sources: Restatements, Model Rules, UCC
   - Short forms: `Id.`, `supra`, `*Varghese*, 925 F.3d at 1342`
3. **Short-form resolution** — `id.`/`supra`/party-only refs are tied to
   their prior full citation; volume mismatches flagged as critical.
4. **Quote attribution** — quoted spans near each citation are associated
   with it.
5. **API validation** — every citation routed to the right backend:

   | Kind | Backend |
   |---|---|
   | US case | `CourtListenerClient` |
   | USC | `USCodeClient` (govinfo.gov) |
   | CFR | `CFRClient` (eCFR) |
   | UK | `BAILIIClient` (TNA + BAILII) |
   | EU | `EURLexClient` |
   | Canada | `CanLIIClient` |
   | Australia | `AustLIIClient` |
   | Docket | `RECAPClient` |
   | Secondary | `RestatementCorpus` (local JSON) |

   Canonical name, year, court, and judge are compared against the
   model's claim; each disagreement gets its own `CitationStatus`
   (`MISMATCH`, `YEAR_MISMATCH`, `COURT_MISMATCH`, `JUDGE_MISMATCH`).
6. **Quote verification** — the cited opinion's text is fetched and the
   quoted language is matched (exact + fuzzy). Misses become
   `QUOTE_MISMATCH` — the "real citation, fake quote" failure mode.
7. **Bluebook lint** — format errors (double spaces, missing year,
   lowercase party names) surface at LOW/MEDIUM severity.
8. **Judge LLM** — a second-vendor model audits the draft for fabricated
   claims.
9. **Consensus voting** — N providers are asked the same question;
   any citation that only one model produced gets flagged.
10. **Uncertainty scoring** — logprobs when available, semantic-entropy
    sampling otherwise.
11. **Policy** — attorney overrides (for sealed/recent cases) and an
    append-only JSONL audit log make every run auditable.

## Quick start

```bash
pip install -r requirements.txt
python -m examples.legal_brief_check --offline   # no network; uses stubs
```

Online usage:

```python
from hallucination_detector import (
    HallucinationDetector, RAGIndex, Document, FirmPolicy,
    MultiJurisdictionValidator,
    CourtListenerClient, USCodeClient, CFRClient, BAILIIClient,
    CanLIIClient, AustLIIClient, RECAPClient, RestatementCorpus,
)
from hallucination_detector.providers import AnthropicProvider, OpenAIProvider

index = RAGIndex([Document(doc_id="CASE-MIRANDA", title="...", text="...")])

cl = CourtListenerClient()
validator = MultiJurisdictionValidator(
    backends=[
        cl, USCodeClient(), CFRClient(),
        BAILIIClient(), CanLIIClient(), AustLIIClient(),
        RECAPClient(), RestatementCorpus(),
    ],
)
policy = FirmPolicy(audit_path="/var/log/firm/hallucination-audit.jsonl")
policy.add_override("123 X.Y.Z 456", "J. Smith", "Sealed case, 2024 matter #42")

detector = HallucinationDetector(
    creator=AnthropicProvider("claude-opus-4-6"),
    rag_index=index,
    judge=OpenAIProvider("gpt-4o"),
    consensus_providers=[AnthropicProvider(), OpenAIProvider()],
    validator=validator,
    opinion_fetcher=cl.fetch_opinion_text,  # enables quote verification
    policy=policy,
)

report = detector.check("Draft one paragraph on Miranda warnings in custody.")
if report.blocked:
    escalate_to_human(report)
else:
    deliver_to_attorney(report.output, report.citations, report.confidence)
```

## Who this is for

Law firms and solo practitioners who already use generative AI for
drafting but need a verification step before anything is filed, shared
with a client, or relied on in discovery. It is **not** a substitute for
attorney review; it is a pre-filter that makes that review tractable by
flagging exactly which sentences need to be checked.

## Configuration

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` — LLM providers
- `COURTLISTENER_TOKEN` — recommended for production
- `GOVINFO_API_KEY` — for USC lookup (defaults to `DEMO_KEY`)
- `CANLII_API_KEY` — required for Canadian lookups
- `HallucinationDetector.block_at` — severity floor for auto-block
  (default `CRITICAL`)
- `HallucinationDetector.min_confidence` — confidence floor
  (default `0.6`)
- `FirmPolicy.add_override(citation, attorney, reason)` — attorney
  sign-off for sealed/very-recent cases

## Data protection (for law firms using the tool online)

Three guarantees that are expected by GCs, malpractice carriers, and
ABA Formal Opinion 512:

### 1. Redaction around every outbound LLM call

`RedactingProvider` wraps any `LLMProvider` and scrubs PII before the
prompt leaves the network, then restores the tokens in the response.
**Legal citations are preserved verbatim** so validation still works.

```python
from hallucination_detector import Redactor, RedactingProvider
from hallucination_detector.providers import AnthropicProvider

redactor = Redactor(
    client_names={"Acme Corp.", "John Smith"},
    matter_number_patterns=[r"MATTER-\d{6}"],
)
provider = RedactingProvider(
    inner=AnthropicProvider(model="claude-opus-4-6"),
    redactor=redactor,
    redacted_log=lambda r: audit.write(r),  # every outbound byte is logged
)
# provider is a drop-in replacement anywhere an LLMProvider is expected.
```

### 2. Validator-only mode (zero LLM calls, zero privilege risk)

For firms that want to pilot without sending anything to a generative
model: run validation on an attorney-drafted document. Extraction,
short-form resolution, API validation, quote checking, and Bluebook lint
all run with no LLM in the loop.

```bash
# Validate a draft, hitting only public citation databases (no client
# data leaves the firm except the citation itself, which is not privileged).
python -m hallucination_detector draft.txt

# Fully offline - no network at all. Pure local static analysis.
python -m hallucination_detector draft.txt --offline

# JSON for downstream tooling.
python -m hallucination_detector draft.txt --json > report.json
```

Or programmatically:

```python
report = detector.check_draft(attorney_draft, use_api=True)
```

### 3. HMAC-signed, tamper-evident audit log

Every detection run (and every override added/removed) is written to an
append-only JSONL file. When a HMAC key is supplied, each record carries
a chained signature — altering any prior record breaks the chain and
`verify_audit_log()` pinpoints the first tampered line.

```python
from hallucination_detector import FirmPolicy, verify_audit_log

policy = FirmPolicy(
    audit_path="/var/log/firm/audit.jsonl",
    audit_hmac_key=bytes.fromhex(os.environ["FIRM_AUDIT_HMAC_KEY"]),
)

# Later, from another process:
result = verify_audit_log("/var/log/firm/audit.jsonl", key)
assert result.ok, f"Line {result.first_bad_line}: {result.reason}"
```

From the CLI:

```bash
export FIRM_AUDIT_HMAC_KEY=$(openssl rand -hex 32)
python -m hallucination_detector draft.txt --audit audit.jsonl
python -m hallucination_detector --verify-audit audit.jsonl
# audit OK (N signed records)
```

## Tests

```bash
python -m unittest discover -s tests
```

40 tests, all offline (stub providers, fake backends, tamper simulation).
