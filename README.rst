.. image:: https://img.shields.io/pypi/v/skeleton.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/skeleton.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/skeleton

.. image:: https://github.com/jaraco/skeleton/workflows/tests/badge.svg
   :target: https://github.com/jaraco/skeleton/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton


*******************************
Neon Recording API
*******************************

- As far as I can tell, between 'main' and 'multipart' branches, every requested feature in the notion doc, as well as all that was discussed at all meetings, is implemented and tested. Various implementations of some functionality were tried, to see what feels most ergonomic for a user, while balancing minimal code against efficiency, as well as explicitness. Some functionality was tested with different, but equivalent, implementations, to eliminate any possibility of errors and to provide some assurance of stability.

- See 'multipart' branch not only for code to load multipart recordings, but also for code to load different (appropriate) timestamps based on whether recording came from cloud or phone

- Some examples and discussion in the Notion doc involved fixation data. It was not clear what we wanted to do with fixations, since I understood that this library will only be for Native Recording Data. Shall a fixation detector be run when loading the data?

- I tried, but pl-recover-recording will complicate installation of this library. it requires building untrunc from source and instructions for windows and macOS are needed

- Some functions from the original design are not possible without monkey patching routines deep within Python. For example, with the current generator approach to sampling, the following is not possible:
gaze = gaze.sample(between_two_events).to_numpy()
unless we monkey patch the code for generator objects. Earlier approaches to the implementation of Stream that supported this were rejected after evaluation.
So there is a Stream.sampled_to_numpy() method to handle this request.

- I tried one or two ways and searched/thought about it, but loading all frames at once into RAM does not seem feasible.

- TODO: fill in rest of README

- see `initial discussion/design <https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4>`_
