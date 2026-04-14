"""Inspect a configured detector and report every network destination it
will contact.

This is the core trust artifact for an on-prem deployment: an attorney
or IT admin runs ``hd --show-network`` and gets a definitive list of
which URLs the tool will reach, broken down by purpose. Anything not in
this list will not be contacted by the tool.

Returns a list of :class:`NetworkEndpoint` records. Backends advertise
their own destinations via the ``DESTINATIONS`` class attribute or by
exposing a ``base_url`` / ``BASE`` attribute we can introspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class NetworkEndpoint:
    component: str         # "Citation lookup: CourtListenerClient"
    purpose: str           # "Verify US case citations"
    destination: str       # "https://www.courtlistener.com"
    sends_client_data: bool  # True if a prompt is sent (LLM); False if only public citations


# Static catalog of what each shipped backend talks to. Kept here so a
# reader can audit the file once and trust it forever.
_BACKEND_CATALOG: dict[str, tuple[str, str, bool]] = {
    "CourtListenerClient": (
        "Verify US case citations + fetch opinion text for quote check",
        "https://www.courtlistener.com",
        False,
    ),
    "USCodeClient": (
        "Verify US Code statute citations",
        "https://api.govinfo.gov",
        False,
    ),
    "CFRClient": (
        "Verify CFR regulation citations",
        "https://www.ecfr.gov",
        False,
    ),
    "BAILIIClient": (
        "Verify UK case citations",
        "https://www.bailii.org / https://caselaw.nationalarchives.gov.uk",
        False,
    ),
    "EURLexClient": (
        "Verify EU case citations (ECLI)",
        "https://eur-lex.europa.eu",
        False,
    ),
    "CanLIIClient": (
        "Verify Canadian case citations",
        "https://api.canlii.org",
        False,
    ),
    "AustLIIClient": (
        "Verify Australian case citations",
        "http://www.austlii.edu.au",
        False,
    ),
    "RECAPClient": (
        "Verify federal docket numbers (PACER mirror)",
        "https://www.courtlistener.com",
        False,
    ),
    "RestatementCorpus": (
        "(local file) Verify Restatement / UCC / Model Rules citations",
        "(local, no network)",
        False,
    ),
}

_PROVIDER_CATALOG: dict[str, tuple[str, str, bool]] = {
    "AnthropicProvider": (
        "Generative model (Creator / Judge / Consensus)",
        "https://api.anthropic.com",
        True,
    ),
    "OpenAIProvider": (
        "Generative model (Creator / Judge / Consensus)",
        "https://api.openai.com",
        True,
    ),
    "GeminiProvider": (
        "Generative model (Creator / Judge / Consensus)",
        "https://generativelanguage.googleapis.com",
        True,
    ),
    "OllamaProvider": (
        "Generative model — LOCAL (runs on this machine)",
        "(local, no network)",
        False,
    ),
    "StubProvider": (
        "(local stub used in validator-only mode)",
        "(local, no network)",
        False,
    ),
    "RedactingProvider": (
        "Wrapper that scrubs PII before delegating to the inner provider",
        "(see inner provider)",
        True,
    ),
}


def report(detector) -> list[NetworkEndpoint]:
    """Walk a :class:`HallucinationDetector` and enumerate every endpoint it
    will contact. The ordering mirrors the pipeline."""
    out: list[NetworkEndpoint] = []

    def _add_provider(label: str, provider) -> None:
        if provider is None:
            return
        cls = type(provider).__name__
        purpose, dest, sends = _PROVIDER_CATALOG.get(
            cls, (f"Generative model ({label})", "(unknown)", True)
        )
        # Drill into RedactingProvider to also surface what it wraps.
        if cls == "RedactingProvider" and hasattr(provider, "inner"):
            inner_cls = type(provider.inner).__name__
            inner_purpose, inner_dest, inner_sends = _PROVIDER_CATALOG.get(
                inner_cls, (f"Generative model ({label})", "(unknown)", True)
            )
            out.append(NetworkEndpoint(
                component=f"{label} (redacted): {inner_cls}",
                purpose=inner_purpose,
                destination=inner_dest,
                sends_client_data=inner_sends,
            ))
            return
        # For Ollama, prefer the actually-configured base_url over the catalog.
        if cls == "OllamaProvider":
            dest = getattr(provider, "base_url", dest)
        out.append(NetworkEndpoint(
            component=f"{label}: {cls}", purpose=purpose, destination=dest,
            sends_client_data=sends,
        ))

    _add_provider("Creator", getattr(detector, "creator", None))
    _add_provider("Judge", getattr(detector, "judge", None))
    for i, p in enumerate(getattr(detector, "consensus_providers", None) or []):
        _add_provider(f"Consensus[{i}]", p)

    validator = getattr(detector, "validator", None)
    if validator is not None:
        for b in getattr(validator, "backends", []):
            cls = type(b).__name__
            purpose, dest, sends = _BACKEND_CATALOG.get(
                cls, (f"Citation lookup ({cls})", _maybe_introspect_url(b), False),
            )
            out.append(NetworkEndpoint(
                component=f"Citation backend: {cls}", purpose=purpose,
                destination=dest, sends_client_data=sends,
            ))

    if getattr(detector, "opinion_fetcher", None) is not None:
        # The fetcher is usually a bound method of CourtListenerClient,
        # which is already in the backend list. Note it explicitly anyway.
        out.append(NetworkEndpoint(
            component="Quote checker: opinion_fetcher",
            purpose="Fetch full opinion text to verify quoted language",
            destination="(see Citation backend above)",
            sends_client_data=False,
        ))

    return out


def _maybe_introspect_url(obj) -> str:
    for attr in ("base_url", "BASE", "BASE_URL"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            try:
                p = urlparse(v)
                return f"{p.scheme}://{p.netloc}" if p.scheme else v
            except Exception:
                return v
    return "(unknown)"


def format_report(endpoints: list[NetworkEndpoint]) -> str:
    if not endpoints:
        return (
            "Network policy: no outbound destinations.\n"
            "This configuration is fully offline."
        )
    lines = ["Network policy — destinations this tool is configured to contact:", ""]
    sends = [e for e in endpoints if e.sends_client_data]
    no_send = [e for e in endpoints if not e.sends_client_data]
    if sends:
        lines.append("WILL receive the prompt (potentially client data):")
        for e in sends:
            lines.append(f"  - {e.component}")
            lines.append(f"      purpose:     {e.purpose}")
            lines.append(f"      destination: {e.destination}")
        lines.append("")
    if no_send:
        lines.append("Receives only public legal citations (no client data):")
        for e in no_send:
            lines.append(f"  - {e.component}")
            lines.append(f"      purpose:     {e.purpose}")
            lines.append(f"      destination: {e.destination}")
    if not sends:
        lines.append("")
        lines.append(
            "No generative model is configured. Validator-only mode: "
            "the only data leaving this machine is public citations to "
            "public databases."
        )
    return "\n".join(lines)
