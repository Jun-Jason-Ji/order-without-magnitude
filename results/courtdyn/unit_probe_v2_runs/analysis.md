# Text-only unit-conversion probe v2

392 items per checkpoint; the 12 published items are a verbatim subset.
Accuracy = published criterion |error| < 0.05. Wilson 95% intervals.
Basis: the 246 diagnostic items, where an unconverted answer would be scored wrong (see is_diagnostic). All-392 figure kept for continuity only.

## Accuracy

| checkpoint | diagnostic 246 | all 392 (legacy) | published 12 | factor given | factor withheld |
|---|---|---|---|---|---|
| Base (no adapter) | 80.5 [75.1, 85.0] | 81.1 [77.0, 84.7] | 9/12 | 81.9 [74.9, 87.4] | 78.4 [69.5, 85.3] |
| CourtDyn-native | 45.5 [39.4, 51.8] | 40.8 [36.1, 45.7] | 2/12 | 51.4 [43.3, 59.4] | 37.3 [28.5, 46.9] |
| Event-holdout v1 | 64.2 [58.1, 70.0] | 56.9 [51.9, 61.7] | 9/12 | 68.1 [60.1, 75.1] | 58.8 [49.1, 67.9] |
| Event-holdout v3 | 58.1 [51.9, 64.1] | 68.6 [63.9, 73.0] | 4/12 | 64.6 [56.5, 71.9] | 49.0 [39.5, 58.6] |

## How the wrong answers are wrong

| checkpoint | correct | identity | wrong_direction | decade_error | other | unparsed |
|---|---|---|---|---|---|---|
| Base (no adapter) | 198 | 0 | 0 | 28 | 9 | 11 |
| CourtDyn-native | 112 | 32 | 0 | 85 | 17 | 0 |
| Event-holdout v1 | 158 | 10 | 0 | 65 | 13 | 0 |
| Event-holdout v3 | 143 | 22 | 0 | 74 | 7 | 0 |

## Text-only twins of the visual unit arms

| checkpoint | m->cm embedded | m->px embedded | m->cm returned unchanged |
|---|---|---|---|
| Base (no adapter) | 64.3 [38.8, 83.7] | 78.6 [52.4, 92.4] | 0/14 |
| CourtDyn-native | 28.6 [11.7, 54.6] | 64.3 [38.8, 83.7] | 1/14 |
| Event-holdout v1 | 35.7 [16.3, 61.2] | 71.4 [45.4, 88.3] | 0/14 |
| Event-holdout v3 | 35.7 [16.3, 61.2] | 78.6 [52.4, 92.4] | 2/14 |

## Paired contrasts (exact McNemar)

- **Base (no adapter)** framing bare vs embedded: 81.3 vs 79.7, p = 0.824 (123 pairs); factor withheld vs given: 78.4 vs 80.4, p = 0.791 (102 pairs)
- **CourtDyn-native** framing bare vs embedded: 41.5 vs 49.6, p = 0.0872 (123 pairs); factor withheld vs given: 37.3 vs 41.2, p = 0.344 (102 pairs)
- **Event-holdout v1** framing bare vs embedded: 65.0 vs 63.4, p = 0.815 (123 pairs); factor withheld vs given: 58.8 vs 61.8, p = 0.664 (102 pairs)
- **Event-holdout v3** framing bare vs embedded: 62.6 vs 53.7, p = 0.0433 (123 pairs); factor withheld vs given: 49.0 vs 54.9, p = 0.286 (102 pairs)

## Same checkpoint, same unit request: text vs image

Median ratio of the requested-unit answer to the metre quantity (text: stated value; image: metre-arm answer for the same clip).
Required: cm/m = 100, px/m = 27.498.

| checkpoint | text cm/m | image cm/m (speed, path) | text px/m | image px/m (speed, path) | text px ratios in (0.5, 2) |
|---|---|---|---|---|---|
| Base (no adapter) | 100.00 | not run | 27.50 | not run | 0/27 |
| CourtDyn-native | 10.00 | 1.00, 1.02 | 27.50 | 1.60, 1.80 | 0/28 |
| Event-holdout v1 | 10.00 | 1.56, 1.14 | 27.50 | 1.67, 1.55 | 0/28 |
| Event-holdout v3 | 10.00 | 1.57, 1.00 | 27.50 | 1.83, 1.45 | 0/28 |

## Each adapter against the base model (same items)

- CourtDyn-native: base-only correct 90, adapter-only correct 4, McNemar p = 3.22e-22
- Event-holdout v1: base-only correct 46, adapter-only correct 6, McNemar p = 1.03e-08
- Event-holdout v3: base-only correct 67, adapter-only correct 12, McNemar p = 2.05e-10
