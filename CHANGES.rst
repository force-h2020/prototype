CHANGELOG
---------

Version 0.3.0
-------------

Released:

Release notes
~~~~~~~~~~~~~

Version 0.3.0 is a major update to the Force ITWM Example plugin,
and includes a number of backward incompatible changes, including:

* Strong dependency on ``force_bdss`` package version 0.4.0
* Many MCO objects that were used accross multiple plugins have been ported
  to ``force_bdss``. Therefore the ``WeightedMCO`` functionality now needs to import
  the ``WeightedOptimizer``, ``WeightedMCOStartEvent`` and ``WeightedMCOProgressEvent``
  classes.
* The ``SubprocessOptimizer`` has been removed, since we no longer use Dakota. It remains
  in the ``enthought_example`` package as an example of how to use the ``MCOCommunicator``
  class and run the BDSS in ``--evaluate`` mode

The following people contributed
code changes for this release:

* Stefano Borini
* Matthew Evans
* James Johnson
* Frank Longford
* Petr Kungurtsev

Features
~~~~~~~~

* New ``WeightedMCO`` class (#61, #70) that uses the ``WeightedOptimizer``, ``WeightedMCOStartEvent``
  and ``WeightedMCOProgressEvent`` classes provided in the core ``force_bdss`` package
* Included Sen's Scaling method for KPIs in ``WeightedOptimizer`` (#38)
* Introduced Taylor gradient consistency testing methods in ``itwm_example/unittest_tools`` module
  (#46, #48, #54) along with the `TestGradientDataSource`` class, designed to be used with systems
  that are able to possess provide analytical gradients for their input slot parameters
* New ``SpaceSampler`` class (#51) and subclasses ``UniformSpaceSampler`` /
  ``DirichletSpaceSampler`` provide control over how parameter space is sampled in the ``WeightedMCO``
  in MCO
* Included example workflow json file to utilise ``NevergradMCO`` from ``force_nevergrad`` package
  (#57, #61, #67)
* ``CSVWriter`` class now inherits from ``BaseCSVWriter`` in the ``force_bdss`` (#62)
* Two example workflow files can be found in the ``itwm_example/tests/fixtures`` module (#61, #68, #74)

Changes
~~~~~~~

* References to ``Workflow.mco`` attribute updated to ``Workflow.mco_model`` (#42, #68)

Removals
~~~~~~~~

* Most ``BaseMCOParameter`` classes have been ported to the force_bdss package (#56)
* Ported the ``WeightedOptimizer`` class to the ``force_bdss`` (#55, #61)
* Ported the ``BaseCSVWriter`` class to the ``force_bdss`` (#62)
* The ``SubprocessOptimizer`` and ``MCOCommunicator`` classes have been removed (#65),
  since we no longer use Dakota.

Documentation
~~~~~~~~~~~~~

* Included docs module with Sphinx auto-api (#73)
* Added brief use case introduction documentation (#73)


Maintenance and code organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated scipy to version 1.2.1 (#39)
* EDM version updated to 2.1.0 in Travis CI (#32, #33, #76) using python 3.6
  bootstrap environment
* Travis CI now runs 2 jobs: Linux Ubuntu Bionic and MacOS (#76)
* Better handling of ClickExceptions in CI (#305)

Version 0.2.0
-------------

- Initial release
