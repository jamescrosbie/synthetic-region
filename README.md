# Synthetic Regions
## 14 Oct 2017

## Project aim:
To build a counterfactual for NCM vanguards to assess how they would have performed had they not been vanguards.

### Method
#### Data Prep
1. For list of GP practices, read in the demographic information i.e. percentage age/sex splits, %IMD, %BEM (source list size information, GPPS).  These factors were chosen because they are outside GP control and are therefore a given.  This work also builds on the public health observatories work on GP clustering.
2. Cluster practices into groups using k-means.  The number of means is chosen using the silhouette coefficient.
3. Read in performance metric data from SUS.
4. Remove vanguard practices as dont want to affect output.
5. For each cluster, time point, age/sex split work out the expected number of events.
6. Aggregate to practice level.
7. Remove small practices (as dont want too much volatility) and practices that (no longer exist as cannot build synthetic on these)

### Develop Synthetic Region
8. Match each vanguard to a cluster based on vanguards characteristics
9. Sample subset of practices from cluster (note vanguard practices have been removed previously) that is equal to the vanguard size
10. Using Simulated Annealing to optimise the total absolute error between fitted and observed over the baseline period.
  * SA was chosen as built in optimisation methods would not solve, so used a stochastic approach  
  * Absolute error was chosen over squared error as squared error gives more weight to outliers, and data is very volatile so wanted to minimise this effect.
  * Regularisation was used to reduce the number of practices in the synthetic region to only those that added value.
  * Fitted value is the weighted (value to be optimised) sum of the sampled GP practices.
11. Process is repeated 100 times taking the best (smallest) solution as the synthetic region.

### Assessing Fit
12. Fit is assessed by repeating process against all GP practices i.e. without the restriction of donor practices being in the same cluster
13. Placebo test is calculated.
  * From the donor pool of practices select group of size equal to the vanguard.  
  * Aggregate these into a placebo vanguard
  * From the remaining pool of practices in the donor pool repeat the process to find a synthetic.
  * Repeat the process to find the variability of the placebo








