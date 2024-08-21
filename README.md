# Information-Flow-In-Silico-Brain

Here, we study the flow of information in an in-silico brain with biologically realistic connectivity. The whole-brain model is based on three assumptions:
1) The synchronization of low-frequency oscillations (approx. 1 - 15 Hz) in large-scale brain networks can be modelled as a Kuramoto model.
2) High-frequency spiking carries information, not low-frequency oscillations.
3) Low freuency oscillations can either enhance or supress high frequency spiking.
If the above is true, cordinated transfer of information between regions occurs when the low-frequency osicllations in the regions are synchronized, such that the high frequency spiking is enhanced at the right moment in both regions.

We test this hypothesis by creating a two layered network, where each node has two untits. The first layers consists of coupled oscillators, interacting as a Kuramoto model. The second layer consists of spike rate units. The oscillatory units work to enhance and supress acitvity in the spike rate layer but the spike rate units do not influence the oscillatory layer. We then stimulated nodes by adding an external current to a spiking unit, and investiagte how the current flows between nodes depending on the synchronization pattern in the oscillatory layer. We are especially interested in the information transmission between nodes, which we quantify using transfer entropy. 

The strucual data is aquired from the MICA-MICs MRI dataset (Royer, J., R. Rodríguez-Cruces, S. Tavakol, S. Larivière, P. Herholz, Q. Li, R. Vos de Wael, C. Paquola, O. Benkarim & B.-y. Park (2022) An open MRI dataset for multiscale neuroscience. Scientific Data, 9, 569). For parcellation, we chose the 200-node 7-network Schaefer atlas. 
