# Enumeration of solution for linear assignment problem

Algorithms for Enumerating All Perfect, Maximum and Maximal Matchings in Bipartite Graphs (Takeaki Uno, 1997)

... with a little bit of tweak. We can permutate specific combination of nodes only, while suppressing other nodes' permutation, by simply prohibiting certain nodes to be end node of cycle search step.

## Motivation

### Focus

Suppressing certain nodes' permutation is especially helpful in object detection with expected position of targets, occlusion / overdetection correspondence can be suppressed nicely.
