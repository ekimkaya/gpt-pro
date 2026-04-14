# Hallucination Detector

**Software you install on your own computer that catches AI-generated
hallucinations in legal drafts before they reach a court filing.**

This is the failure mode that cost the *Mata v. Avianca* attorneys their
sanctions: ChatGPT invented case citations, opposing counsel couldn't
find them, the judge sanctioned the lawyers $5,000 each. California, New
York, and other state bars are now disciplining attorneys for the same
pattern. This tool catches it.

The product is designed to run **on your own computer**, not in a vendor's
cloud. No phone-home, no telemetry, no required cloud account, no
client data leaves your machine unless you explicitly configure it to.

## Install

```bash
pip install hallucination-detector
```

That's it. No account, no license server, no activation. The wheel is
~200 KB and depends only on `requests`.

Optional cloud LLM extras (only install the ones you actually use):

```bash
pip install 'hallucination-detector[anthropic]'   # for Claude
pip install 'hallucination-detector[openai]'      # for GPT
pip install 'hallucination-detector[gemini]'      # for Gemini
```

For a fully local LLM (no cloud account at all), install
[Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull llama3.1:8b
```

## Use it

### From the command line

```bash
# Validate an attorney's draft. No LLM is called.
hd path/to/draft.txt

# Pure local mode - no API calls of any kind.
hd path/to/draft.txt --offline

# Show the auditable list of every network destination the tool would
# contact with your current configuration. Use this to verify the
# on-prem story to your IT department.
hd --show-network

# Sign every detection run into an HMAC-chained tamper-evident log.
export FIRM_AUDIT_HMAC_KEY=$(openssl rand -hex 32)
hd path/to/draft.txt --audit /var/log/firm/audit.jsonl
hd --verify-audit /var/log/firm/audit.jsonl
```

### From the desktop GUI (for attorneys, not engineers)

```bash
hd-gui
```

Opens a single-window app: paste a draft, click *Validate*, see findings.
File menu has *Show network policy* — the same auditable destinations
list, in a dialog you can show the GC.

### Training mode (learn what each layer catches)

```bash
hd-train --offline
```

Walks through six bundled sample drafts — clean baseline, *Mata
v. Avianca*-style fabricated citations, real-citation-fake-quote, short
form referring to wrong volume, real citation with wrong year, and a
multi-jurisdiction draft. For each one it shows the draft, what to look
for, and what the detector caught.

## What it does

11-step pipeline. Every step is local code; only specific layers reach
the network, and only to public legal databases — never with client data.

| # | Layer | Does what |
|---|---|---|
| 1 | RAG | Grounds the draft in your firm's corpus. Forces `[doc_id]` citations. Optional. |
| 2 | Extraction | Parses every recognized citation kind (US case law, USC, CFR, state statutes, UK, EU, Canada, Australia, federal dockets, Restatements/UCC/Model Rules, `id.`/`supra` short forms). |
| 3 | Short-form resolution | Walks the document in order, links `Id.`/`supra`/`*Party*, 925 F.3d at 1342` to the prior full citation. Wrong-volume short forms = CRITICAL. |
| 4 | Quote attribution | Pairs quoted spans with their nearest citation. |
| 5 | API validation | Routes each citation to the right backend (CourtListener, govinfo, eCFR, BAILII, EUR-Lex, CanLII, AustLII, RECAP, local Restatement corpus). Compares case name / year / court / judge against canonical record. |
| 6 | Quote verification | Fetches the cited opinion and checks that quoted language actually appears there (exact + fuzzy match). Catches "real citation, fake quote." |
| 7 | Bluebook lint | Format errors that cluster around AI output. |
| 8 | Judge LLM | Optional. Second-vendor model audits the draft. |
| 9 | Consensus | Optional. N providers asked the same question; solo citations flagged. |
| 10 | Uncertainty | Optional. Logprobs (OpenAI) or semantic-entropy sampling (Claude). |
| 11 | Policy | Attorney overrides for sealed/recent cases; HMAC-chained audit log. |

Steps 1, 8, 9, 10 require an LLM (cloud or local Ollama). **Steps 2–7
and 11 require nothing but the install** — that's the validator-only
mode most firms will run.

## Privacy guarantees you can verify

1. **No telemetry.** Grep the source. The tool never contacts a vendor
   server. You can audit every byte that leaves your machine with
   `hd --show-network`.
2. **Local LLM option.** Use Ollama and zero generative AI traffic
   leaves your machine.
3. **Reversible PII redaction** when you do use a cloud LLM.
   `RedactingProvider` scrubs client names, matter numbers, SSNs, emails,
   phone numbers, dollar amounts, and dates before any prompt leaves the
   network. Legal citations pass through verbatim so validation still
   works. Tokens are restored in the response.
4. **Tamper-evident audit log.** HMAC chain over JSONL records.
   `hd --verify-audit` confirms the chain or pinpoints the first
   tampered line. This is the artifact you'd hand to bar counsel or a
   malpractice carrier if challenged.

## Programmatic use

```python
from hallucination_detector import (
    HallucinationDetector, MultiJurisdictionValidator,
    CourtListenerClient, USCodeClient, RestatementCorpus,
    FirmPolicy,
)
from hallucination_detector.providers import OllamaProvider

cl = CourtListenerClient()
detector = HallucinationDetector(
    creator=OllamaProvider("llama3.1:8b"),         # local model, no cloud
    judge=OllamaProvider("llama3.1:70b"),          # local, larger judge
    validator=MultiJurisdictionValidator(
        backends=[cl, USCodeClient(), RestatementCorpus()],
    ),
    opinion_fetcher=cl.fetch_opinion_text,        # quote verification
    policy=FirmPolicy(audit_path="/var/log/firm/audit.jsonl",
                      audit_hmac_key=bytes.fromhex(os.environ["FIRM_AUDIT_HMAC_KEY"])),
)

# Either generate a draft from a question:
report = detector.check("Draft a paragraph on Miranda warnings in custody.")
# Or validate an attorney's pre-written draft (no LLM call at all):
report = detector.check_draft(open("brief.txt").read())

if report.blocked:
    escalate_to_human(report)
```

## Who is this for

Solo practitioners, small firms, and BigLaw IT teams who want to use AI
for drafting **and** want documentation that they did so responsibly.
The tool is the documentation: every run is logged, the audit log is
tamper-evident, the network policy is auditable, and the source code is
open so the firm's GC can have it reviewed.

## Tests

```bash
python -m unittest discover -s tests
```

45 tests, all offline (stub providers, fake backends, tamper simulation).

## License

MIT.
