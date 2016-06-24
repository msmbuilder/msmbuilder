"""Get sample data for testing and experimenting

{{header}}
"""
import os

from msmbuilder.example_datasets import FsPeptide

FsPeptide("./").cache()
if not os.path.exists("data"):
    os.symlink("fs_peptide", "data")
