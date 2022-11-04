# VLN-Model-Optimization
* Proposed the Structured state-Evolution (SEvol) module to solve the flaw of over-compression of object-level spatial-temporal information in NvEM, and enhanced the performance of it on R2R, R4R and REVERIE datasets
* Based on A2C algorithm, used Reinforced Layout clues Miner (RLM) module to select objects appropriately
* Employed Dynamic Graph Neural Network (DGNN) to aggregate the spatial-temporal information of objects
* Based on GRU model, proposed mGRU model (matrix version), accomplished the renewal of weight in DGNN at every time step
* Used NNI to find the best parameter set
