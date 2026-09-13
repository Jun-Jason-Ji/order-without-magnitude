# Release scope and redistribution notice

Original code retains the existing MIT license and named copyright holders in
`LICENSE`. The associated manuscript's author order is separately recorded; adding
a paper author is not an assertion that they own all earlier software copyright.

Version 1.1.0 distributes only original code, aggregate statistical findings,
input digests, source sequence path mappings, and reconstruction documentation.
It omits source videos/images/annotations, generated QA or training pools,
homography matrices, frame/trajectory identifiers, contact sheets, individual
predictions and scores, model/adapter weights, and per-item reference outputs.
The new `revision_controls_20260912` derived input directory is not copied.

The previously recorded restrictions remain in force: TeamTrack data redistribution
permission was not established in the project audit; Human-M3 was subject to its
separate restricted access terms. Obtain inputs independently under applicable
terms. Dataset code or website licenses do not automatically license dataset
content. Digests and sequence mappings grant no data access or redistribution rights.

Public v1.0.0 is a separate sanitized historical snapshot. Its DOI identifies that
snapshot, not version 1.1.0. Its sanitized output rows alone lack the QA/ground
truth and alignment needed by the original scoring scripts. Do not substitute the
author's unsanitized `release_heavy` or local prediction directories into any public
update. The historical native adapter remains subject to its Apache-2.0 backbone
terms; there are no weights in version 1.1.0.

Event-alignment and Human-M3 summaries omit individual time samples / shared frame
IDs. Aggregate counts are evidence of source overlap, not a grant of source-data
rights. Cite TeamTrack and Human-M3 when using their inputs. Version 1.1.0 has separate metadata and preserves the historical snapshot.
