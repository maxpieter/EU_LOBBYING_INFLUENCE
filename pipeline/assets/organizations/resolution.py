"""Consolidated organisation name resolution.

Single deterministic cascade that replaces the 5 scattered matching locations:
1. lobbying/silver.py (7-step cascade)
2. commission_meetings/silver.py (3-step matching)
3. lobbying/fuzzy_match.py (pg_trgm + Claude CLI)
4. lobbying/org_dedup.py (4-pass batch dedup)
5. scripts/run_org_dedup_pass4.py (rapidfuzz + Anthropic API)

OrgResolver builds indexes once, then resolves any org name through a unified
8-step deterministic cascade. Fuzzy/AI matching lives in fuzzy.py.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from pipeline.models.form_entity_mapping import get_organization_type_from_form_entity
from pipeline.models.lobbying_models import Organization, OrganizationType


# ---------------------------------------------------------------------------
# Shared constants — unified superset from all 5 locations
# ---------------------------------------------------------------------------

# Most comprehensive legal suffix list (28 entries, from run_org_dedup_pass4.py
# plus extras from org_dedup.py and commission_meetings/silver.py).
# Uses \s+ prefix instead of \b to avoid word-boundary issues with dotted
# abbreviations like "S.A." where \b fails at the trailing dot.
_LEGAL_SUFFIXES = re.compile(
    r"(?:\s+|^)("
    r"ltd|limited|gmbh|ag|sa|s\.a\.?|s\.r\.l\.?|srl|bv|nv|inc|corp|"
    r"plc|llc|e\.v\.?|aisbl|asbl|vzw|ry|z\.s\.?|a\.s\.?|s\.p\.a\.?|"
    r"s\.l\.?|sl|se|eeig|a\.s\.b\.l\.?|associação|association"
    r")(?:\s+|$|\.)",
    re.IGNORECASE,
)

# Person titles to strip from meeting attendee strings (from lobbying/silver.py)
_PERSON_TITLE_RE = re.compile(
    r"\s*(?:,\s*)?(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|M\.\s|Mme\.?\s)\s",
)

# Trailing "Firstname Lastname" after comma
_TRAILING_NAME_RE = re.compile(r"\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+(?:,.*)?$")

# Transparency Register ID pattern: 10+ digits, dash, 2 digits
_TR_ID_RE = re.compile(r"(\d{10,}-\d{2})")

# Geographic / office suffixes (from org_dedup.py)
_GEO_SUFFIXES = re.compile(
    r"\s+(?:"
    r"belgium|brussels|eu\s*office|europe|european\s*office|"
    r"france|germany|ireland|netherlands|italia|spain|"
    r"uk|united\s*kingdom|denmark|sweden|finland|austria|"
    r"portugal|greece|poland|czech\s*republic|hungary|romania|"
    r"croatia|slovakia|slovenia|bulgaria|cyprus|estonia|"
    r"latvia|lithuania|luxembourg|malta|"
    r"bureau\s*europ[ée]en|repr[ée]sentation|"
    r"eu\s*representation|eu\s*affairs|public\s*affairs"
    r")\s*$",
    re.IGNORECASE,
)

# Parenthetical pattern: "Org Name (ACRONYM)" or "(ACRONYM) Org Name"
_PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


# ---------------------------------------------------------------------------
# Cleaning / normalization functions
# ---------------------------------------------------------------------------

def clean_org_name(name: str) -> str:
    """Strip person names and noise from an organisation name.

    Handles patterns like:
    - "TotalEnergies Mr. Eric Quenet, M. Giovanni Butelli"
    - "ECSA, Mr. Marc du Moulin"
    - "Name1|Name2|Name3" (takes first part)
    """
    if "|" in name:
        name = name.split("|")[0].strip()
    name = _PERSON_TITLE_RE.split(name)[0]
    name = _TRAILING_NAME_RE.sub("", name)
    return name.rstrip(", ").strip()


def normalize_for_key(name: str) -> str:
    """Aggressive normalization for index keys.

    Strips legal suffixes, parentheticals, punctuation, whitespace.
    Result is lowercase alphanumeric only — suitable for dict lookups.
    """
    if not name:
        return ""
    s = name.lower().strip()
    # Remove legal suffixes
    s = _LEGAL_SUFFIXES.sub("", s)
    # Remove parenthetical content
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    # Remove all non-alphanumeric (keep spaces temporarily)
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse and strip whitespace, then remove all spaces
    s = re.sub(r"\s+", "", s).strip()
    return s


def normalize_for_display(name: str) -> str:
    """Lighter normalization for human-readable matching.

    Strips legal suffixes and extra whitespace but keeps spaces and casing.
    Used for prefix matching and display comparisons.
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def generate_stub_id(name: str) -> str:
    """Generate a deterministic hash-based ID for a stub organisation."""
    normalized = normalize_for_key(name)
    source = f"name_{normalized}"
    return f"org_{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"


def search_variants(name: str) -> list[str]:
    """Generate search query variants for fuzzy matching.

    Returns up to 3 variants in priority order:
    1. Original name
    2. Name with geographic/office suffixes stripped
    3. Name with parenthetical content removed
    """
    variants: list[str] = [name]
    stripped = _GEO_SUFFIXES.sub("", name).strip()
    if stripped and stripped.lower() != name.lower():
        variants.append(stripped)
    no_parens = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
    no_parens = _GEO_SUFFIXES.sub("", no_parens).strip()
    if no_parens and no_parens.lower() not in {v.lower() for v in variants}:
        variants.append(no_parens)
    return variants


# ---------------------------------------------------------------------------
# OrgResolver
# ---------------------------------------------------------------------------

class OrgResolver:
    """Consolidated deterministic organisation resolver.

    Builds indexes from a list of canonical TR organisations, then resolves
    any raw org name through an 8-step cascade. Stubs are created for
    unresolvable names with deterministic hash IDs.

    Cascade order:
    1. TR ID exact match
    2. Normalised name exact match (aggressive: no spaces/punctuation)
    3. Cleaned name match (strips person names, retries normalised lookup)
    4. Acronym match (unambiguous only, >=3 chars for TR acronyms)
    5. Parenthetical match — "FEAP (Federation...)" tries both parts
    6. Prefix match — "Toyota" → "Toyota Motor Europe" (>=5 chars, single candidate)
    7. TR ID extraction from name — regex (\\d{10,}-\\d{2})
    8. Create stub with hash ID org_{SHA256[:16]}
    """

    def __init__(self, tr_organizations: list[Organization]):
        self._by_tr_id: dict[str, Organization] = {}
        self._by_norm_key: dict[str, Organization] = {}
        self._by_display_name: dict[str, Organization] = {}
        self._by_acronym: dict[str, Organization] = {}
        self._stubs: dict[str, Organization] = {}  # stub_id -> stub org

        # Track acronym ambiguity
        acronym_counts: dict[str, int] = {}

        for org in tr_organizations:
            # Index by TR ID
            if org.eu_transparency_register_id:
                self._by_tr_id[org.eu_transparency_register_id] = org

            # Index by aggressive normalised key
            norm = normalize_for_key(org.name)
            if norm:
                self._by_norm_key[norm] = org

            # Index by display-level normalised name (for prefix matching)
            display = normalize_for_display(org.name)
            if display:
                self._by_display_name[display] = org

            # Track acronyms for ambiguity detection
            if org.acronym:
                acr = org.acronym.strip().lower()
                if acr:
                    acronym_counts[acr] = acronym_counts.get(acr, 0) + 1
                    self._by_acronym[acr] = org

        # Remove ambiguous acronyms (shared by >1 org)
        for acr, count in acronym_counts.items():
            if count > 1:
                self._by_acronym.pop(acr, None)

    def resolve(
        self,
        name: str,
        tr_id: str | None = None,
    ) -> tuple[Organization, str]:
        """Resolve a raw org name through the full deterministic cascade.

        Returns (matched_org, match_method).
        """
        name = name.strip()
        if not name:
            stub = self._create_stub(name or "unknown")
            return stub, "stub_empty"

        # Step 1: TR ID exact match
        if tr_id:
            tr_id = tr_id.strip()
            if tr_id in self._by_tr_id:
                org = self._by_tr_id[tr_id]
                return org, "tr_id_exact"

        # Step 2: Normalised name exact match
        norm = normalize_for_key(name)
        if norm in self._by_norm_key:
            org = self._by_norm_key[norm]
            # Enrich with TR ID if we have one and org doesn't
            if tr_id and not org.eu_transparency_register_id:
                org.eu_transparency_register_id = tr_id
                self._by_tr_id[tr_id] = org
            return org, "name_exact"

        # Step 3: Clean name (strip person names) and retry
        cleaned = clean_org_name(name)
        if cleaned != name:
            cleaned_norm = normalize_for_key(cleaned)
            if cleaned_norm and cleaned_norm in self._by_norm_key:
                return self._by_norm_key[cleaned_norm], "cleaned_name"
            # Also try acronym on cleaned name
            if cleaned.lower() in self._by_acronym:
                return self._by_acronym[cleaned.lower()], "cleaned_acronym"

        # Step 4: Acronym match (raw name as acronym)
        name_lower = name.strip().lower()
        if name_lower in self._by_acronym:
            return self._by_acronym[name_lower], "acronym"

        # Step 5: Parenthetical match — "FEAP (Federation...)" or "(Federation...) FEAP"
        paren_match = _PAREN_RE.match(name)
        if paren_match:
            outer = paren_match.group(1).strip()
            inner = paren_match.group(2).strip()
            for part in [outer, inner]:
                part_norm = normalize_for_key(part)
                if part_norm and part_norm in self._by_norm_key:
                    return self._by_norm_key[part_norm], "parenthetical"
                if part.lower() in self._by_acronym:
                    return self._by_acronym[part.lower()], "parenthetical_acronym"

        # Step 6: Prefix match (>= 5 chars, single candidate only)
        if len(name) >= 5:
            prefix_norm = normalize_for_key(cleaned)
            if prefix_norm and len(prefix_norm) >= 4:
                candidates = [
                    v for k, v in self._by_norm_key.items()
                    if k.startswith(prefix_norm)
                ]
                if len(candidates) == 1:
                    return candidates[0], "prefix"

        # Step 7: TR ID embedded in name
        tr_match = _TR_ID_RE.search(name)
        if tr_match:
            extracted_id = tr_match.group(1)
            if extracted_id in self._by_tr_id:
                return self._by_tr_id[extracted_id], "tr_id_extracted"

        # Step 8: Create stub
        stub = self._create_stub(name, tr_id)
        return stub, "stub"

    def resolve_batch(
        self,
        names: list[str],
        tr_ids: list[str | None] | None = None,
    ) -> list[tuple[Organization, str]]:
        """Resolve a batch of names. Returns list of (org, method) tuples."""
        if tr_ids is None:
            tr_ids = [None] * len(names)
        return [self.resolve(n, t) for n, t in zip(names, tr_ids)]

    def _create_stub(
        self,
        name: str,
        tr_id: str | None = None,
    ) -> Organization:
        """Create or retrieve an existing stub organisation."""
        # Check if we already have a stub for this name
        stub_id = generate_stub_id(name)
        if tr_id and tr_id.strip():
            stub_id = tr_id.strip()

        if stub_id in self._stubs:
            return self._stubs[stub_id]

        # Also check canonical orgs by the generated ID
        for org in list(self._by_tr_id.values()) + list(self._by_norm_key.values()):
            if org.id == stub_id:
                return org

        # Infer org type from name
        org_type = get_organization_type_from_form_entity(name)
        if not org_type:
            org_type = OrganizationType.OTHER_ORGANISATIONS_PUBLIC_MIXED

        stub = Organization(
            name=name,
            normalized_name=name,
            organization_type=org_type,
            eu_transparency_register_id=(tr_id if tr_id else None),
        )

        self._stubs[stub.id] = stub
        # Index the stub so subsequent lookups find it
        norm = normalize_for_key(name)
        if norm and norm not in self._by_norm_key:
            self._by_norm_key[norm] = stub

        return stub

    def add_organization(self, org: Organization) -> None:
        """Add an organisation to the indexes (e.g. after fuzzy resolution)."""
        if org.eu_transparency_register_id:
            self._by_tr_id[org.eu_transparency_register_id] = org
        norm = normalize_for_key(org.name)
        if norm:
            self._by_norm_key[norm] = org
        display = normalize_for_display(org.name)
        if display:
            self._by_display_name[display] = org
        if org.acronym:
            acr = org.acronym.strip().lower()
            if acr:
                self._by_acronym[acr] = org

    def get_all_organizations(self) -> list[Organization]:
        """Return all organisations: canonical + stubs."""
        # Deduplicate by ID
        all_orgs: dict[str, Organization] = {}
        for org in self._by_tr_id.values():
            all_orgs[org.id] = org
        for org in self._by_norm_key.values():
            all_orgs[org.id] = org
        for org in self._stubs.values():
            all_orgs[org.id] = org
        return list(all_orgs.values())

    def get_stubs(self) -> list[Organization]:
        """Return only stub organisations (no TR ID)."""
        return [
            org for org in self._stubs.values()
            if not org.eu_transparency_register_id
        ]

    @property
    def canonical_count(self) -> int:
        return len(self._by_tr_id)

    @property
    def stub_count(self) -> int:
        return len(self._stubs)
