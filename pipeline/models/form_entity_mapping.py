"""Form of Entity to Organization Category Mapping.

Copied from parl8 for backwards compatibility.
Maps form_of_entity values from the EU Transparency Register
to the 13 official organization categories.
"""

from typing import Dict, Optional

from .lobbying_models import OrganizationType

FORM_ENTITY_MAPPING: Dict[str, OrganizationType] = {
    # ASSOCIATIONS AND NETWORKS OF PUBLIC AUTHORITIES
    "ministry": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "department": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "agency": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "authority": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "commission": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "council": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "government": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "public": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "state": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "federal": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "regional": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "municipal": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "office": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "administration": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "embassy": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "delegation": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    "representation": OrganizationType.ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES,
    # THINK TANKS AND RESEARCH INSTITUTIONS
    "university": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "college": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "institute": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "research": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "academic": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "think tank": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "thinktank": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "studies": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "center": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "centre": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "laboratory": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "science": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "education": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "stichting": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    "foundation": OrganizationType.THINK_TANKS_RESEARCH_INSTITUTIONS,
    # NON-GOVERNMENTAL ORGANISATIONS
    "fund": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "charity": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "non-profit": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "nonprofit": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "ngo": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "civil society": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "advocacy": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "campaign": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "movement": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "alliance": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "coalition": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "network": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "forum": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "initiative": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "environmental": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "human rights": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "social": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "community": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    "vzw": OrganizationType.NON_GOVERNMENTAL_ORGANISATIONS,
    # TRADE UNIONS AND PROFESSIONAL ASSOCIATIONS
    "trade union": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "professional": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "guild": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "syndicate": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "workers": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "labor": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "labour": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "syndicat": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    "union": OrganizationType.TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS,
    # TRADE AND BUSINESS ASSOCIATIONS
    "association": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "federation": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "confederation": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "chamber": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "society": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "academy": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "industry": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "sector": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "business": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "employers": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "aisbl": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "asbl": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "e.v.": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    "verein": OrganizationType.TRADE_BUSINESS_ASSOCIATIONS,
    # PROFESSIONAL CONSULTANCIES
    "consulting": OrganizationType.PROFESSIONAL_CONSULTANCIES,
    "advisory": OrganizationType.PROFESSIONAL_CONSULTANCIES,
    "services": OrganizationType.PROFESSIONAL_CONSULTANCIES,
    "consultancy": OrganizationType.PROFESSIONAL_CONSULTANCIES,
    "consultant": OrganizationType.PROFESSIONAL_CONSULTANCIES,
    # LAW FIRMS
    "law": OrganizationType.LAW_FIRMS,
    "legal": OrganizationType.LAW_FIRMS,
    "attorney": OrganizationType.LAW_FIRMS,
    "advocate": OrganizationType.LAW_FIRMS,
    "solicitor": OrganizationType.LAW_FIRMS,
    # CHURCHES AND RELIGIOUS COMMUNITIES
    "church": OrganizationType.CHURCHES_RELIGIOUS_COMMUNITIES,
    "religious": OrganizationType.CHURCHES_RELIGIOUS_COMMUNITIES,
    "faith": OrganizationType.CHURCHES_RELIGIOUS_COMMUNITIES,
    # COMPANIES AND GROUPS
    "inc": OrganizationType.COMPANIES_GROUPS,
    "corp": OrganizationType.COMPANIES_GROUPS,
    "ltd": OrganizationType.COMPANIES_GROUPS,
    "llc": OrganizationType.COMPANIES_GROUPS,
    "sa": OrganizationType.COMPANIES_GROUPS,
    "gmbh": OrganizationType.COMPANIES_GROUPS,
    "ag": OrganizationType.COMPANIES_GROUPS,
    "srl": OrganizationType.COMPANIES_GROUPS,
    "spa": OrganizationType.COMPANIES_GROUPS,
    "bv": OrganizationType.COMPANIES_GROUPS,
    "nv": OrganizationType.COMPANIES_GROUPS,
    "company": OrganizationType.COMPANIES_GROUPS,
    "corporation": OrganizationType.COMPANIES_GROUPS,
    "enterprise": OrganizationType.COMPANIES_GROUPS,
    "group": OrganizationType.COMPANIES_GROUPS,
    "holdings": OrganizationType.COMPANIES_GROUPS,
    "capital": OrganizationType.COMPANIES_GROUPS,
    "investment": OrganizationType.COMPANIES_GROUPS,
    "bank": OrganizationType.COMPANIES_GROUPS,
    "finance": OrganizationType.COMPANIES_GROUPS,
    "insurance": OrganizationType.COMPANIES_GROUPS,
    "pharma": OrganizationType.COMPANIES_GROUPS,
    "tech": OrganizationType.COMPANIES_GROUPS,
    "technology": OrganizationType.COMPANIES_GROUPS,
    "software": OrganizationType.COMPANIES_GROUPS,
    "energy": OrganizationType.COMPANIES_GROUPS,
    "manufacturing": OrganizationType.COMPANIES_GROUPS,
    "telecom": OrganizationType.COMPANIES_GROUPS,
    "media": OrganizationType.COMPANIES_GROUPS,
    "sas": OrganizationType.COMPANIES_GROUPS,
    "international": OrganizationType.COMPANIES_GROUPS,
    "global": OrganizationType.COMPANIES_GROUPS,
}


def get_organization_type_from_form_entity(form_of_entity: str) -> Optional[OrganizationType]:
    """Get organization type from form_of_entity string."""
    if not form_of_entity or str(form_of_entity).lower() == "nan":
        return None

    form_lower = str(form_of_entity).lower().strip()

    if form_lower in FORM_ENTITY_MAPPING:
        return FORM_ENTITY_MAPPING[form_lower]

    for pattern, org_type in FORM_ENTITY_MAPPING.items():
        if pattern in form_lower:
            return org_type

    return None
