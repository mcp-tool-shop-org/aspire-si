"""Upgrade a schema 1.0 geometry export from the trainer to 1.1, stating how it was produced.

A 1.0 export does not say what a step is or where its scalars came from. For one written by
`aspire train` those facts are known: steps are training steps, and with more than one epoch the
scalars after the first epoch are replayed from the dialogue cache. This script adds exactly those
fields and the new version; it does not touch the trajectory, scalars or anything else.

Usage: python examples/pod-run/upgrade_export.py <geometry.json> <epochs> [<out.json>]
"""

import json
import sys
from pathlib import Path

from aspire.geometry import SCHEMA_VERSION

source = Path(sys.argv[1])
epochs = int(sys.argv[2])
target = Path(sys.argv[3]) if len(sys.argv) > 3 else source

document = json.loads(source.read_text(encoding="utf-8"))
if document.get("schema_version") != "1.0":
    sys.exit(f"{source} is schema {document.get('schema_version')!r}; only 1.0 exports are upgraded.")
document["schema_version"] = SCHEMA_VERSION
document["run_metadata"]["step_axis"] = "training_step"
document["run_metadata"]["scalar_source"] = "replayed" if epochs > 1 else "live"
target.write_text(json.dumps(document, indent=1, allow_nan=False), encoding="utf-8")
print(f"{target}: schema {SCHEMA_VERSION}, step_axis training_step, scalar_source "
      f"{document['run_metadata']['scalar_source']}")
