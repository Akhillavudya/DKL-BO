"""Self-contained data layer for the SNUMAT generalization side-test.

Nothing here modifies the main `src/dklbo` package — this folder only adds a new
data-ingestion layer (loader + 3D graph builder + descriptors) for the SNUMAT
3D bulk-crystal dataset, and reuses the dataset-agnostic model/BO code from
`dklbo` by import.
"""
