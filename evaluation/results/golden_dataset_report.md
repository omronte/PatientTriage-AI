# PatientTriage.ai — Golden Dataset Safety Evaluation Report (Phase 5)

> [!IMPORTANT]
> **SIMULATED SAFETY SCENARIO NOTICE**: All patient records in this dataset are entirely synthetic.
> Expected outputs represent simulated prototype safety behaviors and do NOT constitute medically validated clinical ground truth.

## Executive Summary

| Metric | Count / Status | Rate |
|---|---:|---:|
| Dataset Size | 22 records | 100% |
| Scenario Coverage | 20 / 20 tags | 100% |
| **Under-Triage Rate** | **6 cases** | **27.3%** |
| Over-Triage Rate | 4 cases | 18.2% |
| Exact Triage Agreement | 12 cases | 54.5% |
| Safety Rule Trigger Rate | 15 cases | 68.2% |
| Low Confidence Rate | 1 cases | 4.5% |
| Conservative Escalation Rate | 13 cases | 59.1% |
| Missing Data Detection Rate | 6 cases | 27.3% |
| Pediatric Routing | ✅ PASS | — |
| Geriatric Routing | ✅ PASS | — |
| Zero-History Routing | ✅ PASS | — |
| Data Leakage | ✅ CLEAN | — |

## Safety-Critical Failures

| Safety Check | Count |
|---|---:|
| Critical under-triage (ESI 1 expected, higher predicted) | 0 |
| Critical safety rule misses | 0 |
| Pediatric safety misses | 0 |
| Geriatric safety misses | 0 |
| Missing-data safety misses | 0 |
| Zero-history safety misses | 0 |
| Vital deterioration misses | 1 |
| Low-confidence escalation misses | 0 |

## Triage Confusion Matrix

Rows = Expected ESI | Columns = Predicted ESI

| Expected \ Predicted | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|
| **P1** | 6 | 0 | 0 | 0 | 0 |
| **P2** | 3 | 2 | 0 | 0 | 0 |
| **P3** | 0 | 0 | 0 | 0 | 5 |
| **P4** | 0 | 1 | 0 | 1 | 1 |
| **P5** | 0 | 0 | 0 | 0 | 3 |

## Scenario-Level Breakdown

| Scenario Tag | Cases | Agreement | Under-Triage | Over-Triage | Safety Triggered |
|---|---:|---:|---:|---:|---:|
| `AMBIGUOUS_PRESENTATION` | 1 | 0 | 1 | 0 | 1 |
| `CLINICIAN_ACCEPT` | 1 | 1 | 0 | 0 | 0 |
| `CLINICIAN_OVERRIDE` | 1 | 0 | 1 | 0 | 0 |
| `CONFLICTING_SIGNALS` | 1 | 1 | 0 | 0 | 1 |
| `CRITICAL_SAFETY_RULE` | 1 | 1 | 0 | 0 | 1 |
| `GERIATRIC_AMBIGUOUS_ZERO_HISTORY` | 1 | 0 | 0 | 1 | 1 |
| `GERIATRIC_RICH_HISTORY` | 1 | 0 | 0 | 1 | 1 |
| `HIGH_CONFIDENCE` | 1 | 1 | 0 | 0 | 1 |
| `LOW_CONFIDENCE` | 1 | 1 | 0 | 0 | 1 |
| `MISSING_VITAL` | 1 | 1 | 0 | 0 | 1 |
| `MULTI_FACTOR_RISK` | 1 | 1 | 0 | 0 | 1 |
| `PARTIAL_HISTORY` | 1 | 0 | 1 | 0 | 1 |
| `PEDIATRIC_AGE_ADJUSTMENT` | 1 | 1 | 0 | 0 | 1 |
| `PEDIATRIC_MISSING_DATA` | 2 | 0 | 0 | 2 | 2 |
| `RICH_HISTORY` | 1 | 1 | 0 | 0 | 0 |
| `STABLE_LOW_RISK` | 2 | 1 | 1 | 0 | 0 |
| `VITALS_OVERRIDE_NLP` | 1 | 1 | 0 | 0 | 1 |
| `VITAL_DETERIORATION` | 1 | 0 | 1 | 0 | 0 |
| `WAIT_THRESHOLD` | 1 | 0 | 1 | 0 | 0 |
| `ZERO_HISTORY` | 1 | 1 | 0 | 0 | 1 |

## Per-Patient Detail Table

| Patient ID | Scenario Tag | Exp ESI | Pred ESI | Exp Band | Pred Band | Conf | Exp Escalation | Pred Escalation | Age Group | History |
|---|---|---:|---:|---|---|---:|:---:|:---:|---|---|
| `DEMO-P001` | `PEDIATRIC_AGE_ADJUSTMENT` | 1 | 1 | MEDIUM | HIGH | 87% | Yes | Yes | pediatric | rich |
| `DEMO-P002` | `GERIATRIC_AMBIGUOUS_ZERO_HISTORY` | 2 | 1 | LOW | MEDIUM | 58% | Yes | Yes | geriatric | none |
| `DEMO-P003` | `VITALS_OVERRIDE_NLP` | 1 | 1 | MEDIUM | MEDIUM | 61% | Yes | Yes | adult | rich |
| `DEMO-P004` | `ZERO_HISTORY` | 4 | 4 | LOW | MEDIUM | 68% | Yes | Yes | adult | none |
| `DEMO-P005` | `RICH_HISTORY` | 5 | 5 | HIGH | HIGH | 87% | No | No | adult | rich |
| `DEMO-P006` | `PARTIAL_HISTORY` | 3 | 5 ⚠️ | MEDIUM | MEDIUM | 74% | Yes | No | adult | partial |
| `DEMO-P007` | `AMBIGUOUS_PRESENTATION` | 3 | 5 ⚠️ | LOW | MEDIUM | 74% | Yes | No | adult | partial |
| `DEMO-P008` | `MISSING_VITAL` | 2 | 2 | LOW | HIGH | 87% | Yes | Yes | adult | rich |
| `DEMO-P009` | `CONFLICTING_SIGNALS` | 1 | 1 | LOW | MEDIUM | 58% | Yes | Yes | adult | rich |
| `DEMO-P010` | `HIGH_CONFIDENCE` | 1 | 1 | HIGH | HIGH | 83% | Yes | Yes | adult | rich |
| `DEMO-P011` | `LOW_CONFIDENCE` | 2 | 2 | LOW | MEDIUM | 62% | Yes | Yes | geriatric | none |
| `DEMO-P012` | `STABLE_LOW_RISK` | 4 | 5 ⚠️ | HIGH | HIGH | 87% | No | No | adult | rich |
| `DEMO-P013` | `CLINICIAN_ACCEPT` | 5 | 5 | HIGH | HIGH | 87% | No | No | adult | rich |
| `DEMO-P014` | `CLINICIAN_OVERRIDE` | 3 | 5 ⚠️ | MEDIUM | MEDIUM | 78% | No | No | adult | rich |
| `DEMO-P015` | `PEDIATRIC_MISSING_DATA` | 4 | 2 | MEDIUM | MEDIUM | 55% | No | Yes | pediatric | partial |
| `DEMO-P016` | `GERIATRIC_RICH_HISTORY` | 2 | 1 | MEDIUM | MEDIUM | 66% | Yes | Yes | geriatric | rich |
| `DEMO-P017` | `STABLE_LOW_RISK` | 5 | 5 | HIGH | HIGH | 95% | No | No | adult | rich |
| `DEMO-P018` | `PEDIATRIC_MISSING_DATA` | 2 | 1 | LOW | LOW | 51% | Yes | Yes | pediatric | partial |
| `DEMO-P019` | `MULTI_FACTOR_RISK` | 1 | 1 | MEDIUM | MEDIUM | 67% | Yes | Yes | adult | none |
| `DEMO-P020` | `WAIT_THRESHOLD` | 3 | 5 ⚠️ | MEDIUM | MEDIUM | 76% | Yes | No | adult | rich |
| `DEMO-P021` | `VITAL_DETERIORATION` | 3 | 5 ⚠️ | HIGH | HIGH | 95% | No | No | adult | rich |
| `DEMO-P022` | `CRITICAL_SAFETY_RULE` | 1 | 1 | LOW | HIGH | 84% | Yes | Yes | pediatric | partial |

## Confidence Distribution

- **HIGH** (≥80%): 9 cases
- **MEDIUM** (55–79%): 12 cases
- **LOW** (<55%): 1 cases

