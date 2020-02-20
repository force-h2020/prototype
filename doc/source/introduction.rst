Model Chemical Reactor
======================

The business case of this FORCE prototype consists of a hypothetical reaction
in a liquid between two educts, A and B, where P is the product.

.. math::
    A + B \rightarrow P

In practical applications, no substance is pure and very often side products are formed.
We therefore assume that one educt, A, contains traces of some non-reactive contaminant,
which forms a side product, S. Thus, the above reaction scheme generalizes to

.. math::
    A + B \longrightarrow^{k_p}_{k_s} S + P

Th reaction constants :math:`k_P` and :math:`k_S` are attributed to the product and side reactions
respectively. We assume that no other reaction occurs, thus the side product is distinguished
from the product by a lower reaction constant.

The reaction process runs in a reactor with ideal temperature control and mixing and the reaction
preserves the volume (no activities). The educts from suppliers are stored in batches with an
uninterrupted supply chain. The yield of P can therefore be considered proportional to the
loading volume of the reaction vessel and reaction time allowed. These parameters will also
influence the material and production costs respectively.

A purification step for A is available before loading in the reactor, and takes place
in a buffer store, up-stream from the reaction vessel. This purification step is assumed to be
ideal, but comes at a price which is added to the total production cost.

.. math::
    A \rightarrow A^\prime

    A^\prime + B \longrightarrow^{k_p}_{k_s} S + P

The reaction constants are known to be non-linear functions of the reactor temperature, and (as noted previously)
the product reaction rate is always higher than the side product reaction rate. Therefore, the purity
of the product P may be maximized by increasing the temperature, at the expense of an
increased production cost due to environment control of the reactor.

On the other hand, material cost can be reduced by using lower purity feedstock, A, but
a greater amount of contaminant will also reduce the purity of our product P.

We therefore describe a system that we can control with four parameters:

1. Loading Volume
2. Reaction Time
3. Temperature
4. Purity of A

to optimize three conflicting goals (KPIs):

1. Purity of P
2. Material cost
3. Production cost