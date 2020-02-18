
FORCE BDSS ITWM Prototype
--------------------------

.. image:: https://travis-ci.com/force-h2020/force-bdss-plugin-itwm-example.svg?branch=master
   :target: https://travis-ci.com/force-h2020/force-bdss-plugin-itwm-example
   :alt: Build status

.. image:: http://codecov.io/github/force-h2020/force-bdss-plugin-itwm-example/coverage.svg?branch=master
   :target: http://codecov.io/github/force-h2020/force-bdss-plugin-itwm-example?branch=master
   :alt: Coverage status

This repository contains the implementation of a plugin for the Business Decision Support System (BDSS), contributing
a toy chemical reactor model designed by ITWM.
It is implemented under the Formulations and Computational Engineering (FORCE) project within Horizon 2020
(`NMBP-23-2016/721027 <https://www.the-force-project.eu>`_).

The ``ITWMExamplePlugin`` class contributes several BDSS objects, including a stand-alone ``MCO`` solver as
well as ``DataSource`` and ``NotificationListener`` subclasses. It also can be combined with other ``MCO`` solvers, such
as those contributed by the `FORCE BDSS Nevergrad <https://github.com/force-h2020/force-bdss-plugin-nevergrad>`_ plugin.


Installation
-------------
Installation requirements include an up-to-date version of ``force-bdss``. Additional modules that can contribute to the ``force-wfmanager`` UI are also included,
but a local version of ``force-wfmanager`` is not required in order to complete the
installation.


To install ``force-bdss`` and the ``force-wfmanager``, please see the following
`instructions <https://github.com/force-h2020/force-bdss/blob/master/doc/source/installation.rst>`_.

After completing at least the ``force-bdss`` installation steps, clone the git repository::

    git clone https://github.com/force-h2020/force-bdss-plugin-itwm-example

the enter the source directory and run::

    python -m ci install

This will allow install the plugin in the ``force-py36`` edm environment, allowing the contributed
BDSS objects to be visible by both ``force-bdss`` and ``force-wfmanager`` applications.

Documentation
-------------

To build the Sphinx documentation in the ``doc/build`` directory run::

    python -m ci docs
