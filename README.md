# Area_Constraint_Frequency_Analysis
Simulation framework for area constraint frequency analysis algorithms

This work compares the accuracy, robustness, and complexity of the following frequency analysis algorithms: discrete Fourier transform (DFT) and discrete Hartley transform (DHT).
The work also proposes two alternative frequency analysis algorithms based on the traditional DHT, i.e., the jittered and dithered discrete Hartley transform (J-DHT and D-DHT). The accuracy is compared in the context of frequency detection, which is defined as the performance measured in probablity of detection subject to a specific false positve rate. The complexity is compared based on the number of multiply-and-accumulate operations and estimated required areas. Please refer to the document named final_version.pdf for further details. 

This code framwork supports distributed parallel simulations where it distributes workload across multiple machines using socket programming. This computing method can be applied to any simulation and scientific evaluations. Please refer to the files client.py and server.py if you would like to use this methodology for your own simulations.