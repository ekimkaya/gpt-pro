"""External lookup backends, one per authoritative source.

Every backend implements :class:`CaseLookupClient` from
:mod:`hallucination_detector.citation_validator` and is dispatched to by
:class:`MultiJurisdictionValidator` based on
:attr:`hallucination_detector.models.Citation.kind` and ``jurisdiction``.
"""

from .courtlistener import CourtListenerClient
from .uscode import USCodeClient
from .cfr import CFRClient
from .bailii import BAILIIClient
from .eurlex import EURLexClient
from .canlii import CanLIIClient
from .austlii import AustLIIClient
from .recap import RECAPClient
from .restatement import RestatementCorpus
from .multi import MultiJurisdictionValidator

__all__ = [
    "AustLIIClient",
    "BAILIIClient",
    "CFRClient",
    "CanLIIClient",
    "CourtListenerClient",
    "EURLexClient",
    "MultiJurisdictionValidator",
    "RECAPClient",
    "RestatementCorpus",
    "USCodeClient",
]
