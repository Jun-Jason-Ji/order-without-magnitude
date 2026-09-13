# Validation scope

The source candidate passed inventory, SHA-256, Python syntax, metadata and restricted-field checks (113 payload files, 18,018 JSON keys).
The release ZIP is checked by CRC and SHA-256; synthetic CPU smoke runs are recorded in validation/publication_smoke.json.
Execution provenance records completed P1/P2 author-side runs. Historical failure receipts are preserved alongside the narrowly reviewed target-module-order correction; they are not rewritten.
Preparation-time PREPARED_NOT_RUN and public_release:false fields in original receipts describe those receipts at creation, not the later publication status. See RELEASE.json for this version identity.
Fresh GPU installation and full independent licensed-input reproduction have not been certified. No model weights or item-level evaluation inputs are bundled.
