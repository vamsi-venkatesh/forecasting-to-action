"""VVDex forecasting worker: deterministic profiling, preparation, training, evaluation and planning.

Runs as a separate process with its own dependencies (requirements-forecast.txt), started by the
engine's durable jobs; it never mutates engine lifecycle tables. Every output is a versioned file
plus a manifest (inputs' hashes, code version, parameters), so one result can be reproduced.
"""

