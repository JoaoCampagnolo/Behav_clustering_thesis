# Behav_clustering_thesis
Master thesis project on _Unsupervised behavioral classification with 3D pose data from tethered Drosophila Melanogaster_.

**_ABSTRACT_**  
One of the preeminent challenges of Behavioral Neuroscience is the understanding of how the
brain works and how it ultimately commands an animal’s behavior. Solving this brain-behavior linkage
requires, on one end, precise, meaningful and coherent techniques for measuring behavior. Rapid
technical developments in tools for collecting and analyzing behavioral data, paired with the immaturity
of current approaches, motivate an ongoing search for systematic, unbiased behavioral classification
techniques.
To accomplish such a classification, this study employs a state-of-the-art tool for tracking 3D
pose of tethered _Drosophila_, _DeepFly3D_, to collect a dataset of x-, y- and z- landmark positions over
time, from tethered _Drosophila Melanogaster_ moving over an air-suspended ball. This is succeeded by
unprecedented normalization across individual flies by computing the angles between adjoining
landmarks, followed by standard wavelet analysis. Subsequently, six unsupervised behavior
classification techniques are compared - four of which follow proven formulas, while the remaining two
are experimental. Lastly, their performances are evaluated via meaningful metric scores along with
cluster video assessment, as to ensure a fully unbiased cycle - from the conjecturing of a definition of
behavior to the corroboration of the results that stem from its assumptions.
Performances from different techniques varied significantly. Techniques that perform clustering
in embedded low- (two-) dimensional spaces struggled with their heterogeneous and anisotropic nature.
High-dimensional clustering techniques revealed that these properties emerged from the original highdimensional 
posture-dynamics spaces. Nonetheless, high and low-dimensional spaces disagree on the arrangement of their elements, 
with embedded data points showing hierarchical organization, which was lacking prior to their embedding. 
Low-dimensional clustering techniques were globally a better match against these spatial features and 
yielded more suitable results. Their candidate embedding algorithms alone were capable of revealing 
dissimilarities in preferred behaviors among contrasting genotypes of _Drosophila_. Lastly, the top-ranking 
classification technique produced satisfactory behavioral cluster videos (despite the irregular 
allocation of rest labels) in a consistent and repeatable manner, while requiring a marginal number of 
hand tuned parameters.  
**_Keywords_**: _Drosophila_, behavior, posture-dynamics space, unsupervised learning, clustering,
t-SNE, PCA, Gaussian Mixture Model, HDBSCAN.

This repository holds: **1)** Developed code; **2)** Dataset; **3)** Cluster results.

The code was developed at EPFL's Ramdya Lab. Many thanks to Pavan Ramdya, Semih Günel and the remaining lab members - it was a pleasure to work in your company.
