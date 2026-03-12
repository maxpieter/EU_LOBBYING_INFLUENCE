# EU Lobbying Transparency Pipeline — Project Overview

## Purpose

This document provides full context on the project's goals, data model, data sources, and pipeline architecture. It is intended to be read by an AI assistant at the start of a working session to enable productive collaboration without re-explaining the project.

---

## 1. What This Project Does

We are building an open-source analytical platform that integrates multiple layers of EU lobbying transparency data into a unified, explorable picture. The core problem: EU institutions publish lobbying data across siloed sources with inconsistent structure. MEP meetings carry procedure references; Commission meetings do not. Amendment documents don't reference who the MEP met. No existing tool connects these layers.

The platform works per **legislative procedure** (e.g., "2023/0212(COD)" — the Digital Euro regulation). For a given procedure, a user can explore: which organisations lobbied which officials, when, what documents were produced, how lobbying intensity shifted across lifecycle stages, and where the data goes silent.

**This is not a causal claim about lobbying influence.** We do not attempt to prove that meetings change votes. The data does not support such claims.

---

## 2. The Four Analysis Layers

### Layer 1: Network (deferred)
Bipartite MEP↔organisation graphs from declared meetings, filtered per procedure. One-mode projections reveal co-lobbying patterns. Statistical backbone extraction (hypergeometric test + FDR correction) separates signal from noise. **Not being built yet.**

### Layer 2: Document
Legislative documents (Commission proposals, committee opinions, draft reports, amendments) are scraped and parsed per document type. Amendments are parsed at article level to identify which provisions each MEP targets.

### Layer 3: Semantic Linking
Commission meeting minutes are unstructured free-text with no procedure tag. We use sentence embeddings to compute similarity between meeting subject lines and legislative procedure content (titles, aliases, extracted terms), producing a confidence score for each meeting→procedure link. This transforms opaque records into structured, queryable data. Meetings that cannot be linked to any procedure are themselves a transparency finding.

### Layer 4: Temporal
All data carries timestamps. Each procedure's lobbying activity is decomposed across lifecycle stages (pre-proposal, committee, amendment, trilogue, adoption) to reveal how lobbying intensity and actor composition shift over time.

---

## 3. Current Approach

We are building the pipeline for **one procedure at a time** to validate the approach before scaling. Aliases and some metadata are manually curated for now.

---

## 4. Database Schema — Domain Tables

The database runs on PostgreSQL (Supabase). Below are the domain-relevant tables. App infrastructure tables (billing, auth, chats, notifications, etc.) are omitted.

### 4.1 Core Entity Tables

#### `meps`
Members of the European Parliament.

| Column | Type | Notes |
|---|---|---|
| id | bigint PK | EP MEP ID |
| fullName | text | |
| country | text | Member state |
| politicalGroup | text | EP political group (EPP, S&D, etc.) |
| nationalPoliticalGroup | text | National party |
| profile_url | text | EP website profile |
| image_url | text | |
| role | text | |
| birth_date | date | |
| committees | jsonb | Committee memberships with roles |
| declarations | jsonb | Declarations of financial interests |
| past_meetings | jsonb | Denormalized meeting history |
| status | text | 'active' or 'inactive' |

Source: EP Open Data Portal, Parltrack.

#### `actors`
Commission officials: Commissioners, Cabinet members, Directors-General. Three tiers with different disclosure obligations.

| Column | Type | Notes |
|---|---|---|
| actor_id | text PK | |
| fullName | text | |
| actor_type | text | Type of actor |
| role | text | Commissioner / Cabinet / DG |
| country | text | |
| portfolio | text | Policy portfolio |
| term_start / term_end | date | |
| team | jsonb | Cabinet members (if Commissioner) |
| declarations | jsonb | Financial declarations |
| past_meetings | jsonb | Denormalized — also stored relationally in commission_meetings |
| parliament | text | Default 'eu' |
| status | text | 'active' or 'inactive' |

Source: Commission website, EU Open Data Portal.

**Note:** The `past_meetings`, `team`, `declarations`, `speeches`, `calendar`, `documents` JSONB columns are denormalized copies for display. For analytical queries (joins, filters, aggregations), use the relational tables instead.

#### `organizations`
Interest representatives / lobbying entities.

| Column | Type | Notes |
|---|---|---|
| id | text PK | |
| name | text | |
| normalized_name | text | Lowercased/cleaned for matching |
| official_name | text | |
| acronym | text | |
| eu_transparency_register_id | text | Join key to EU Transparency Register |
| organization_type | text | Category (consultancy, trade assoc, NGO, etc.) |
| industry_sector | varchar | |
| country | varchar | Head office country |
| city | text | |
| website | text | |
| employee_count_range | varchar | |
| annual_revenue_range | varchar | Lobbying budget (commercial) or total budget (non-commercial) — **not directly comparable across categories** |
| total_meetings_count | int | Aggregate |
| unique_meps_met | int | Aggregate |
| influence_score | numeric | Derived metric |
| transparency_score | numeric | Derived metric |
| policy_focus_areas | text[] | |
| level_of_interest | text | |
| interests_represented | text | |

Source: EU Transparency Register (daily XML/Excel dumps from data.europa.eu), LobbyFacts.

**Entity resolution challenge:** Organisation names in MEP meeting records and Commission meeting records are free-text and often don't match the Transparency Register name exactly ("European Round Table for Industry" vs "ERT" vs "European Round Table of Industrialists"). Fuzzy matching + manual curation is needed.

#### `procedures`
Legislative procedures / dossiers.

| Column | Type | Notes |
|---|---|---|
| id | text PK | Procedure reference, e.g. "2023/0212(COD)" |
| process_id | text | |
| title | text | Formal OEIL title, e.g. "Establishment of the digital euro" |
| description | text | |
| procedure_type | text | COD, CNS, etc. |
| legal_basis | text[] | |
| policy_area | text | |
| subjects | text[] | EuroVoc descriptors |
| commission_document | text | COM document reference |
| status | text | |
| stage | text | Current lifecycle stage |
| proposal_date | date | |
| last_activity_date | date | |
| decision_date | date | |
| events | jsonb | Timeline of procedural events |
| actors | jsonb | Involved MEPs, committees, rapporteurs |
| key_provisions | jsonb | |
| related_topics | text[] | |
| ai_summary | text | LLM-generated summary |
| embedding | vector | For semantic search |
| oeil_url | text | |
| eurlex_proposal_url | text | |
| eurlex_final_act_url | text | |

Source: EP Open Data Portal, OEIL, Parltrack (`ep_dossiers`), EUR-Lex.

### 4.2 Meeting Tables

#### `lobbying_meetings`
MEP meetings with interest representatives. These have structured procedure references.

| Column | Type | Notes |
|---|---|---|
| id | text PK | |
| mep_id | integer FK → meps | |
| organization_id | text FK → organizations | Single org — **limitation: some meetings involve multiple orgs** |
| meeting_date | date | |
| title | text | Topic heading (e.g., "Digital Euro") |
| location | text | |
| capacity | text | MEP's role context |
| related_procedure | text | Procedure reference — **key structural advantage over Commission meetings** |
| committee_acronym | text | e.g., "ECON" |
| meeting_purpose | text | |
| policy_area | text | |
| meeting_type | varchar | |
| transparency_level | varchar | |
| source_data | jsonb | Raw scraped data |

Source: Scraped from MEP profile pages on europarl.europa.eu, via Parltrack / Integrity Watch EU.

**Important:** MEP meeting records are NOT linked to the Transparency Register. The `organization_id` is resolved by us, not provided by the source data.

#### `commission_meetings` *(TO BE CREATED)*
Commission official meetings with interest representatives. These have **no procedure reference** — the subject is free-text.

| Column | Type | Notes |
|---|---|---|
| id | text PK | |
| actor_id | text FK → actors | Commissioner / Cabinet / DG |
| actor_role_at_time | text | Role tier at time of meeting |
| meeting_date | date | |
| location | text | |
| subject_raw | text | **Free-text subject — the field we run embeddings against** |
| portfolio_at_time | text | May be unavailable (recently removed from dataset) |
| source_url | text | |
| matched_procedure_id | text FK → procedures, nullable | **Output of semantic linking layer** |
| match_confidence | float, nullable | Confidence score of the procedure match |
| match_method | text, nullable | 'alias_exact', 'alias_fuzzy', 'embedding_similarity', 'manual' |
| created_at | timestamptz | |

Source: EU Open Data Portal, Integrity Watch EU.

**Key data quality issue:** The "portfolio" field was recently removed from the Commission's published dataset for Commissioners and Cabinet members. Historical and current data have different schemas.

### 4.3 Document Tables

#### `procedure_articles`
Parsed content of legislative texts at article/recital level.

| Column | Type | Notes |
|---|---|---|
| id | uuid PK | |
| procedure_id | text FK → procedures | |
| element_type | text | 'recital' or 'article' |
| element_number | text | |
| title | text | |
| content | text | Full text of the article/recital |
| document_source | text | |
| document_version | text | 'proposal', 'committee', 'adopted' |
| sort_order | int | |

#### `procedure_amendments`
Amendments filed by MEPs targeting specific articles.

| Column | Type | Notes |
|---|---|---|
| id | uuid PK | |
| procedure_id | text FK → procedures | |
| document_id | text | |
| amendment_number | int | |
| target_element | text | Which article/recital is targeted |
| target_type | text | |
| original_text | text | |
| amended_text | text | |
| justification | text | |
| committee | text | |
| rapporteur_mep_id | int | |
| submitted_by | jsonb | MEP(s) who filed the amendment |
| adopted | boolean | |

Source: EP website, Parltrack (`ep_amendments`).

### 4.4 Other Domain Tables

#### `votes`
Plenary vote records.

| Column | Type | Notes |
|---|---|---|
| id | int PK | |
| meeting_date_str | date | |
| title | text | |
| procedure_url | text | |
| procedure_id | text | Derived from procedure_url via regex |
| result | text | |
| number_favor / number_against / number_abstention | int | |
| ai_summary | text | |

#### `speeches`
Plenary speeches by MEPs.

| Column | Type | Notes |
|---|---|---|
| id | text PK | |
| date | date | |
| mepid | text | |
| speech | text | Original language |
| translated_speech | text | |
| topics | jsonb | |
| procedure_type | text | |
| embedding_vec | vector | |

#### `glossary`
EU terminology definitions with embeddings for semantic lookup.

---

## 5. Tables To Be Created

### `procedure_aliases`
Maps informal names, acronyms, and multilingual variants to procedures. **Critical for the semantic linking layer** — meeting records typically use informal names (e.g., "Digital Euro") rather than procedure codes.

| Column | Type | Notes |
|---|---|---|
| id | uuid PK | |
| procedure_id | text FK → procedures | |
| alias | text | The informal name, acronym, or variant |
| alias_type | text | 'short_name', 'acronym', 'informal', 'other_language' |
| language | text | ISO 639-1 code |
| source | text | 'manual', 'oeil_title', 'mined_from_meetings' |
| created_at | timestamptz | |

**Building the alias list:** Start with manual curation per procedure. The OEIL title (e.g., "Establishment of the digital euro") provides a formal descriptive title that can be cleaned into a short name by stripping procedural prefixes ("Establishment of the", "Regulation on the", etc.). True nicknames ("MiCA", "CSDDD", "AI Act") require manual entry or mining from press releases. Multilingual variants matter (e.g., "Euro numérique", "Digitaler Euro").

### `commission_meeting_organizations` *(junction table)*
Commission meetings can involve multiple organisations. The current `lobbying_meetings` table has a single `organization_id` — same limitation applies there but is lower priority.

| Column | Type | Notes |
|---|---|---|
| commission_meeting_id | text FK → commission_meetings | |
| organization_id | text FK → organizations | |

### `procedure_documents` *(optional, recommended)*
Parent metadata for the documents that `procedure_articles` are parsed from.

| Column | Type | Notes |
|---|---|---|
| id | text PK | |
| procedure_id | text FK → procedures | |
| document_type | text | 'commission_proposal', 'committee_draft_report', 'committee_opinion', 'adopted_text', 'impact_assessment' |
| document_reference | text | e.g., "COM(2023)0369" |
| title | text | |
| date | date | |
| url | text | |
| parsed | boolean | Has it been broken into procedure_articles? |

### `procedure_timeline_stages` *(optional, for temporal layer)*
Queryable lifecycle stages with date boundaries, derived from events in procedure data.

| Column | Type | Notes |
|---|---|---|
| id | uuid PK | |
| procedure_id | text FK → procedures | |
| stage | text | 'pre_proposal', 'commission_proposal', 'committee_stage', 'amendment_period', 'plenary_first_reading', 'trilogue', 'second_reading', 'final_adoption' |
| start_date | date | |
| end_date | date | |

---

## 6. Data Sources

| Source | What It Provides | Format | URL |
|---|---|---|---|
| EU Transparency Register | Organisations, budgets, lobbyist counts, clients | XML/Excel daily dumps, API | data.europa.eu |
| EU Open Data Portal — Commission meetings | Commission meeting records (official, date, orgs, free-text subject) | CSV/JSON | data.europa.eu |
| EP Open Data Portal | MEPs, procedures, documents, calendar | RDF/Turtle, JSON-LD | data.europarl.europa.eu |
| Parltrack | MEPs, dossiers, amendments, votes, activities | JSON bulk dumps (ODBLv1.0) | parltrack.org |
| Integrity Watch EU | Enriched Commission meetings, MEP meetings | Scraped/republished (ODBLv1.0) | integritywatch.eu |
| EUR-Lex | Commission proposals, adopted legislation | HTML/XML (Akoma Ntoso, Formex) | eur-lex.europa.eu |
| LobbyFacts | Historical Transparency Register data, enriched rankings | Searchable, downloadable | lobbyfacts.eu |
| OEIL (Legislative Observatory) | Procedure metadata, timelines, formal titles | Web / searchable | oeil.europarl.europa.eu |

---

## 7. Key Data Quality Issues

1. **Entity resolution between org names and Transparency Register.** Free-text org names in meeting records often don't match register entries. Requires fuzzy matching + manual curation.

2. **Commission meeting subject opacity.** The `subject_raw` field ranges from specific ("Discussion on Article 6 of COM(2023)0XXX") to vague ("Discussion on digital policy"). Meetings too vague to link are a transparency finding, not a pipeline failure.

3. **Portfolio field removal.** The Commission recently removed the portfolio field from published meeting data. Historical data has it; current data does not. Need workaround via Commissioner-to-DG mapping.

4. **Single-org limitation.** Both `lobbying_meetings` and (future) `commission_meetings` need to handle multi-org meetings. Junction tables needed.

5. **Scraping accuracy.** Parltrack data is scraped from EP websites and carries the usual caveats. Always cross-reference with original sources for critical findings.

6. **Budget comparability.** Transparency Register financial data is not comparable across org categories: commercial entities report lobbying spend, non-commercial entities report total operating budget.

7. **MEP meeting completeness.** Publication is now mandatory for all MEPs meeting interest representatives on parliamentary business, but compliance and detail quality vary.

---

## 8. Semantic Linking Pipeline — How Procedure Matching Works

For each Commission meeting, the pipeline attempts to match the free-text `subject_raw` to a known procedure:

1. **Alias exact match:** Check if `subject_raw` contains any known alias from `procedure_aliases` (case-insensitive). High precision.
2. **Alias fuzzy match:** Fuzzy string matching against aliases. Medium precision.
3. **Embedding similarity:** Compute sentence embedding of `subject_raw`, compare against procedure embeddings (enriched with title + aliases + extracted terms). Lower precision, wider recall.
4. **Unmatched = transparency finding:** If no method produces a confident match, the meeting remains unlinked. The percentage of unlinked meetings is itself a transparency metric.

The matched procedure, confidence score, and method are stored directly on the `commission_meetings` row.

---

## 9. Tech Stack

Inspired by the Parl8 project (github.com/DemAI-tech/parl8). Specific stack choices TBD but expected to include:

- **Database:** PostgreSQL (Supabase) with pgvector for embeddings
- **Backend:** Python or Node.js for scraping, parsing, and pipeline orchestration
- **Embeddings:** Sentence transformers for semantic similarity
- **Frontend:** Web-based interactive exploration tool
- **NLP:** TF-IDF for discriminative term extraction, NER for entity extraction from legislative texts

---

## 10. Transparency Metric

The semantic linking pipeline doesn't just match meetings to procedures — it quantifies how vague each meeting record is. We can measure:

- What percentage of Commission meetings are too opaque to link to any dossier
- Whether opacity correlates with the type of organisation met
- Whether certain Commissioners or DGs produce systematically vaguer meeting records
- How transparency compares across lifecycle stages of a procedure
