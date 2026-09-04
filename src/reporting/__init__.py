"""
Output surfaces for a finished report (the PDF export).

Deliberately outside `src/pipeline/`, alongside `src/validation/`: these
modules read reports, they never produce analysis. Nothing here may be
imported by a pipeline stage, and nothing here computes a figure the
pipeline did not already compute.
"""
