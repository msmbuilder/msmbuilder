My msmb Project
===============

Initialized with `msmb TemplateProject` on {{date}}

Keep notes about your project here.

## Folder layout

Each new step in MSM construction is in a new folder with symlinks
to the files on which it depends from previous steps.

## Variable names convention

variable    | filename          | description
------------|-------------------|-----------------------------------------------
meta        | meta.pandas.pickl | pandas dataframe of trajectory metadata
ftrajs      | ftrajs/           | trajectories of feature vectors (dihedrals, ...)
dihed_feat  | featurizer.pickl  | featurizer object
ttrajs      | ttrajs/           | dimensionality-reduced, tica trajectories
tica        | tica.pickl        | tica object
ktrajs      | ktrajs/           | trajecories of cluster indices
kmeans      | clusterer.pickl   | cluserer object
microktrajs | microktrajs/      | trimmed cluster indices
macroktrajs | macroktrajs/      | macrostate indices

## License

These templates are licensed under the MIT license. Do whatever
you want with them.
