# Nomenclature rule

# Modular T cell nomenclature

Source: Masopust et al., *Guidelines for T cell nomenclature*, Nature Reviews Immunology 26, 298–313 (2026), DOI: 10.1038/s41577-025-01238-2. The framework follows the source's nomenclature v1.0; the decision procedure and seven-slot format are task-specific instructions. The established TC1 designation is additionally supported by McAllister et al. (2005), DOI: 10.1128/IAI.73.11.7450-7457.2005.

## Population and evidence

Interpret the reported identity through the study's definition, evidence, species, sampling and timeline. The study's definition takes precedence over generic usage. Keep claims within their population, condition and time point. Shared sampling statements apply when they clearly include the target. Retain each input population as defined: a definition spanning alternative settings carries their shared properties.

Established terms can convey defining properties even when an extract omits canonical assays or markers. Use the biological definition of the term, distinguishing its defining claims from markers commonly used to recognize it. Read an expanded abbreviation through the wording supplied by the study. Treat the stated identity as a reported claim; associated phenotypes need their own support. Read observations and identity claims at their stated scope; proposed mechanisms describe hypotheses. Results of missing measurements, including unavailable figures or supplements, remain unknown. [Box 2; pp. 300–301, 312]

Assess properties separately using study definitions, investigator claims, appropriate proxies or well-supported contextual inference. An inference needs a biological link to the target population and property beyond their occurrence in the same disease or sample. Leave materially ambiguous properties unspecified. The source permits different levels of detail [Fig. 3]; this task uses one consistent level: include each clearly supported property that fits the seven slots. Heterogeneity, group differences and assay responses remain at their measured scope.

## Seven slots and assembly

Return `lineage`, `function`, `migration`, `migration_subscript`, `differentiation`, `differentiation_subscript`, and `antigen`. Each slot holds one `module` and a concise, consistent snake_case `semantic_value`, or both values are `null`.

The code assembles:

`[lineage] [function or T][migration][migration_subscript][differentiation][differentiation_subscript][antigen]`

The function token includes `T`. Store bases and refinements separately, using uppercase migration refinements and lowercase differentiation refinements. Select a single value per slot. The final code has no trailing `cell`. Anatomy, species, experimental details and ordinary phenotype-marker qualifiers remain in the supplied label and evidence.

## Lineage and function

Lineage is a relatively stable identity supported by the population definition and evidence. Designations include `CD4+`, `CD8+`, `gamma-delta T`, `NKT`, `iNKT`, `MAIT`, `CD8aa+ TCRab`, `DN T` and `DP T`. Functional activity and differentiation state need accompanying lineage evidence. Interpret unconventional lineages through their TCR or antigen-recognition identity; resemblance based on phenotype alone remains a phenotype description. [Table 6]

| Function token | Biological meaning |
| --- | --- |
| `TH1` | Type 1 helper identity, with an IFNγ bias and T-bet-associated differentiation. |
| `TH2` | Type 2 helper identity, with IL-4, IL-5 or IL-13 bias and GATA3-associated differentiation. |
| `TH9` | IL-9-biased helper function. |
| `TH17` | IL-17-biased helper function, with RORγt as a relevant proxy. |
| `TFH`, with subclasses such as `TFH1` and `TFH2` | Follicular helper identity supporting B-cell responses; BCL6 and CXCR5 are relevant proxies. |
| `Treg` | Regulatory function and tolerance; FOXP3 is a major feature of conventional CD4 regulatory cells. |
| `TFR` | Regulatory cells localized within B-cell follicles or germinal centres, with supporting imaging as required by the source. |
| `TCTL` | Cytotoxic function, applicable to either CD4 or CD8 cells. |
| `TC1`, `TC2`, `TC17`, `TC22` | CD8 functional identities with the corresponding cytokine bias. |

A study designation or coherent functional evidence can support functional identity. The table lists common identities; retain an established, supported functional subclass in its standard notation, including cytokine-defined follicular helper subclasses. Interpret cytokines and transcription factors in their species and assay context. Apply traits from a parent compartment, comparison group or condition-specific fraction where evidence supports the trait in the target population. Changes in marker-positive fractions describe heterogeneity. A cytotoxic designation needs a functional claim or evidence of killing beyond effector-molecule expression alone. [Table 3; p. 309]

Retain regulatory subclass tokens: `tTreg` for thymic origin, `pTreg` for peripheral origin, `eTreg` for effector regulatory cells and `iTreg` for induction in vitro. Origin prefixes belong to the function token; differentiation refinements describe a separate property. Interpret origin with study- and species-appropriate evidence. The source supplies no universally accepted human marker panel for this distinction.

## Migration

| Base | Biological meaning |
| --- | --- |
| `S` | Capacity to enter uninflamed secondary lymphoid organs directly from blood. CD62L and/or CCR7 expression are reasonable proxies. |
| `D` | Limited tendency to enter uninflamed secondary lymphoid organs directly from blood. Absence of CD62L and/or CCR7 is reasonable proxy evidence. |
| `U` | Migration is unknown or unspecified and that uncertainty is useful to express. |

A population term whose biological definition includes homing can establish a migration class as well as a differentiation state. Use that defining homing claim when the study is consistent with it. For identities defined by antigen experience or developmental potential, establish migration through supplied migration evidence; typical receptor profiles are unmeasured until the study reports them. Conflicting or intermediate proxies need a supported study interpretation; the source gives no universal marker precedence. Lymphoid sampling establishes location, while `S` concerns entry from blood into an uninflamed organ. [p. 309; Table 4]

| Refinement | Allowed base | Biological meaning |
| --- | --- | --- |
| `B` | `S`, `D` or `U` | Sampled from venous blood; residence or blood–tissue recirculation is unspecified. |
| `W` | `S` or `D` | Recirculation between blood and nonlymphoid tissues. |
| `R` | `D` | Residence within an organ or restricted vascular compartment. |

Choose at most one refinement. Residence needs support for persistence in the compartment; prior migration evidence or contextual proxies may suffice. Tissue recovery, intravascular staining and activation markers describe location or phenotype. An explicitly unverified resemblance to residents retains its uncertainty. [Box 2; Table 4; Table 7]

Recirculation needs a study claim or evidence addressing travel between blood and nonlymphoid tissues, beyond a homing class or non-resident identity. Recovery from several sites establishes distribution. Include `B` for a population sampled from blood/PBMCs when neither `R` nor `W` is selected. Combine it with the supported homing base: `SB`, `DB` or `UB`. If sampling is the only migration information, the base is `U`; with no migration information, leave both slots unspecified. Pooled sampling across blood and other sites retains its mixed origin.

## Differentiation

| Base | Biological meaning |
| --- | --- |
| `N` | Mature post-thymic cells with an undifferentiated phenotype and no cognate-antigen stimulation outside the thymus. |
| `A` | Recent extrinsic activation through antigen, cytokine or other activating receptors. |
| `M` | Differentiated, antigen-experienced cells characterized by persistence, typically with relative quiescence and longevity. |
| `X` | Exhaustion under the study's biological definition and context. |
| `G` | Anergy: persistent hyporesponsiveness after activation, particularly reduced IL-2 production after TCR signalling. |

Select the current state from the study's identity claim, timing and coherent phenotype. An explicit state designation is a reported claim unless the study defines it differently or supplies contradictory evidence. Distinguish a precursor's current state from the state of its future descendants. Effector activity needs accompanying state evidence; exhausted populations can retain effector functions. Assay stimulation describes the assay condition. Recent in vivo activation requires support from the population's in vivo context. [Tables 1, 2 and 5]

Use `p` for progenitor or precursor potential and `t` for terminal differentiation within the selected state. These refinements describe capacity to generate descendants along a developmental trajectory. A study can state this potential through an explicit developmental identity without repeating fate assays. Read the claim from the reported wording and definition. Marker re-expression, late differentiation, senescence-associated markers and a single proliferation measurement describe phenotype or present activity; a developmental interpretation needs support that addresses potential.

## Cognate-antigen status

`+` expresses a reasonable claim that relevant cognate antigen persists in the organism. `0` expresses a reasonable claim of clearance or irrelevance to the population. These describe antigen status independently of stimulation at sampling. Omission leaves antigen status unspecified. [pp. 309–310]

Interpret identity, role and state together with the current setting and timeline. A sustained antigen-driven response during ongoing infection or malignancy can support persistence. A reported identity describing sustained antigen-driven differentiation can contribute to that inference. Biologically supported context can suffice without an explicit persistence statement, direct antigen measurement or exact epitope identification.

The inference must connect the current antigen source to this population. Disease diagnosis, receptor specificity or an isolated marker alone leaves that relationship unresolved. Bystander populations retain their own antigen relationships; past exposure establishes history. Clearance and irrelevance each need support beyond prior exposure, lack of prior stimulation or missing antigen information.

# Mapping task

Use the reference to complete structured T cell nomenclature slot filling for the following T cell cases.

Return only valid JSON. Do not include explanations.
Return one output object per input case.
Do not copy original_label or evidence into the output.
Do not return mapped_label. The final mapped_label will be derived by code from mapped_slots.

For each supported slot, return:
- module: the final naming module inferred from the reference.
- semantic_value: a concise snake_case semantic key describing the biological meaning of that module, inferred from the reference.

If the evidence does not support a slot, return:
{"module": null, "semantic_value": null}

Do not invent unsupported biological properties.
Interpret each reported label and its evidence using the reference's evidence policy and requested level of naming detail.
Use the same semantic_value string consistently for the same biological meaning across cases.

REFERENCE:
<<<
{{RULE_TEXT}}
>>>

CASES:
{{CASES_JSON}}

Output schema:
[
  {
    "case_id": 0,
    "mapped_slots": {
      "lineage": {
        "module": "CD4+",
        "semantic_value": "cd4_conventional"
      },
      "function": {
        "module": null,
        "semantic_value": null
      },
      "migration": {
        "module": null,
        "semantic_value": null
      },
      "migration_subscript": {
        "module": null,
        "semantic_value": null
      },
      "differentiation": {
        "module": null,
        "semantic_value": null
      },
      "differentiation_subscript": {
        "module": null,
        "semantic_value": null
      },
      "antigen": {
        "module": null,
        "semantic_value": null
      }
    }
  }
]

# Refinement

Reconsider the flagged mappings using the original population evidence, the nomenclature rule and the reviewer's feedback. Make the smallest revision justified by those sources, preserving supported properties. Treat feedback as an argument to evaluate: when its proposed change lacks support, retain the existing interpretation.

The records below use local case_id values for this revision request. original_case_id identifies the corresponding record in the preceding Mapping output. Return slots for the local case_id values using the Mapping task's output format.

REVIEWED RECORDS:
{{REVIEW_RECORDS_JSON}}

MAPPING TASK:
{{MAPPING_PROMPT}}
