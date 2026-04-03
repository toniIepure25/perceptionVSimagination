# Table 1. Main benchmark results

| Model | Program role | Test cosine | Test MSE | Imagery mean cosine | Perception mean cosine | Held-out paired groups |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge | External low-data reference baseline | 0.55199 | 0.001167 | 0.55152 | 0.55446 | 1 |
| Shared-only | Canonical neural baseline | 0.13596 | 0.002250 | 0.13422 | 0.14527 | 1 |
| Shared-private p16 | Exploratory hypothesis model | 0.10784 | 0.002323 | — | — | 1 |
| Shared-private p8 | Exploratory recovery variant | 0.09595 | 0.002354 | — | — | 1 |
| Shared-private | Hypothesis-family baseline | 0.06927 | 0.002424 | 0.07151 | 0.05735 | 1 |
| Shared-private no-domain | Diagnostic control | 0.05907 | 0.002450 | — | — | 1 |

Missing imagery/perception means indicate that the currently frozen evidence bundle does not report those condition-specific summaries for that variant.
