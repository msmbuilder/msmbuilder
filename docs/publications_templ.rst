.. _publications:

Publications
============

The following published works use MSMBuilder. To add your publication
to the list, open an issue on GitHub with the relevant information or
edit ``docs/publications.bib`` and submit a pull request.

.. publications.bib lists the relevant publications
.. publications_templ.rst defines how the publications will be displayed
.. publications.rst is generated during sphinx build (see conf.py)
   and should not be edited directly!

{% for pub in publications %}
{{pub.title}}
--------------------------------------------------------------------------------

 * {{pub.author | join('; ')}}
 * *{{pub.journal}}* **{{pub.year}}**, {{pub.volume}} {{pub.pages}}
 * `doi: {{pub.doi}} <http://dx.doi.org/{{pub.doi}}>`_

{{pub.abstract | wordwrap }}

{% endfor %}

