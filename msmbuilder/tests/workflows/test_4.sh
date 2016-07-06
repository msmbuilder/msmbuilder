msmb TemplateProject

python 1-get-example-data.py
cd analysis
python gather-metadata.py
python gather-metadata-plot.py
cd dihedrals
python featurize.py
python featurize-plot.py
cd tica
python tica.py
python tica-plot.py
cd cluster
python cluster.py
python cluster-plot.py
cd msm
