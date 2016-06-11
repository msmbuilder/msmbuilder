"""Get sample data for testing and experimenting

{{header}}
"""
import os

from msmbuilder.example_datasets import FsPeptide

FsPeptide("./").cache()
if not os.path.exists("trajs"):
    os.symlink("fs_peptide", "trajs")
if not os.path.exists("top.pdb"):
    os.symlink("fs_peptide/fs-peptide.pdb", "top.pdb")
