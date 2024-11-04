# *********************************************
# Authors: Aurore Archimbaud 
#          TBS Business School
# *********************************************



# Packages ----------------------------------------------------------------
library("parallel")
library("ICS")
library("rrcov")
library("ICSClust")



# Functions ---------------------------------------------------------------
# To generate the mixture of gaussian distributions
mixture_sim2 <- function (pct_clusters = c(0.5, 0.5), n = 500, p = 10, 
                          mean_lst = list(), sd_lst = list()){
  if (sum(pct_clusters) != 1) {
    stop("the sum of the groups is not equal to 1!")
  }
  n_groups = floor(n * pct_clusters)
  if (sum(n_groups) != n) {
    n_groups[length(pct_clusters)] <- rev(n - cumsum(n_groups)[1:(length(pct_clusters) - 
                                                                    1)])[1]
  }
  if (sum(n_groups) != n) {
    warning(paste("the total number of observation is not equal to n", 
                  paste(round(pct_clusters, 2), collapse = " - ")))
  }
  X_list <- lapply(1:length(pct_clusters), function(i) {
    n <- n_groups[i]
    if (n > 0) {
      data.frame(mvtnorm::rmvnorm(n = n, mean = mean_lst[[i]], 
                                  sigma = diag(sd_lst[[i]])))
    }
  })
  data.frame(cluster = rep(paste0("Group", 1:length(pct_clusters)), 
                           n_groups), do.call(rbind, (X_list)))
}


# Parameters --------------------------------------------------------------

# control parameters for data generation
n <- 1000                               # number of observations
pk <- 5                                 # constant to define the number of variables based on the number of clusters  
delta_all <- c(100, 50, 10, 5, 1)       # shift location
R <- 50                                 # number of simulation runs
seed <- 20230509                        # seed of the random number generator
n_cores <- 5                            # number of cores
sigma1 <- 1                             # covariance structure
sigma2 <- 1                             # covariance structure


# Initialization pf a dataframe to store the generalized eigenvalues
p_max <- 50                             # maximal number of variables
df_eigenvalues_max <- data.frame(matrix(NA, nrow = 1, ncol = p_max))
names(df_eigenvalues_max) <- paste0("IC.", 1:p_max)

# control parameters for mixture weights
pct_clusters_list <- list(c(0.50, 0.50), 
                          c(0.40, 0.60),
                          c(0.30, 0.70), 
                          c(0.21, 0.79), 
                          c(0.20, 0.80), 
                          c(0.10, 0.90),
                          c(0.60,0.20,0.20),
                          c(1/3, 1/3, 1/3), 
                          c(0.33, 0.33, 0.34), 
                          c(0.18, 0.32, 0.50),
                          c(0.20, 0.30, 0.50),
                          c(0.10, 0.40, 0.50),
                          c(0.10, 0.30, 0.60),
                          c(0.10, 0.20, 0.70),
                          c(0.10, 0.10, 0.80),
                          c(0.2, 0.2, 0.2, 0.2, 0.2), 
                          c(0.14, 0.2, 0.2, 0.2, 0.26),
                          c(0.1, 0.2, 0.2, 0.2, 0.3),
                          c(0.1, 0.1, 0.2, 0.2, 0.4),
                          c(0.1, 0.1, 0.1, 0.3, 0.4),
                          c(0.1, 0.1, 0.1, 0.1, 0.6),
                          rep(0.1,10),
                          c(0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.2),
                          c(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.3),
                          c(0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12)
)


# control parameters for ICS scatters: only COV-COV4
ICS_scatters_list <- list(
  `COV-COV[4]` = list(S1 = ICS_cov, S2 = ICS_cov4)
)


# it is very easy to use parallel computing on Unix systems, but not on Windows
if (.Platform$OS.type == "windows") {
  n_cores <- 1              # use only one CPU core
} else {
  n_cores <- n_cores        # number of CPU cores to be used
  RNGkind("L'Ecuyer-CMRG")  # use parallel random number streams
}


# run simulation -----
cat(paste(Sys.time(), ": starting ...\n"))
set.seed(seed)
results_list <- parallel::mclapply(seq_len(R), function(r) {
  
  # print simulation run
  cat(paste(Sys.time(), sprintf(":   run = %d\n", r)))
  

  # mixture weights --------------------------------------------------------------
  # loop over different mixture weights
  results_clusters <- lapply(pct_clusters_list, function(pct_clusters) {
    # We simulate normal gaussian for each cluster with the first variable
    # being true clusters
    # computation of number of variables
    p <- pk*length(pct_clusters)
    
    # creation of the location parameters
    results_means <- lapply(delta_all, function(delta) {
      mean_lst <- lapply(1:length(pct_clusters), function(i) {
        clusters_means <- rep(0, p)
        if (i > 1) {
          clusters_means[i - 1] = delta
        }
        clusters_means
      })
      # creation of the scale parameters
      sd_lst <- lapply(1:length(pct_clusters), function(i) {
        c(rep(sigma1, length(pct_clusters)), rep(sigma2, p-length(pct_clusters)))
      })
      # generation of the data
      data <- mixture_sim2(pct_clusters = pct_clusters, n = n, p = p, 
                           mean_lst = mean_lst, sd_lst = sd_lst)
      nb_clusters <- length(unique(data$cluster))
      true_clusters <- data[,1]
      info <- data.frame(Run = r, n = n, p = p,
                         delta = delta, q = length(pct_clusters),
                         clusters = paste(round(pct_clusters*100),
                                          collapse = "-"))

      
      # ICS ----
      ## scatters ------
      results_ARI_ICS_scatters <- lapply(1:length(ICS_scatters_list),
                                         function(i) {
                                           scatter = names(ICS_scatters_list)[i]
                                           time_reduction <- system.time({
                                             ICS_out <- tryCatch({
                                               do.call(ICS::ICS,
                                                       append(list(X = data[,-1]),
                                                              ICS_scatters_list[[i]]))
                                             },error = function(e) NULL)
                                           })[["elapsed"]]
                                         
                                         
                                             eigenvalues <- gen_kurtosis(ICS_out)
                                             df_eigenvalues <- df_eigenvalues_max
                                             df_eigenvalues[,names(eigenvalues)] <- eigenvalues
                                            
                                             cbind(info, scatter = scatter,
                                                   time = time_reduction, 
                                                   df_eigenvalues)

                                         })
      do.call(rbind, results_ARI_ICS_scatters)
      
    })
    
    do.call(rbind, results_means)
  })
  
  
  # combine results from current simulation run into data frame
  do.call(rbind, results_clusters)
  
}, mc.cores = n_cores)

# combine results into data frame
results <- do.call(rbind, results_list)


# save results to file
file_results <- "R/results/simulations/simulations_study_group_means_cov_cov4_n=%d_r=%d_sigma1=%f_sigma2=%d.RData"
save(results, seed, file = sprintf(file_results, n, R, sigma1, sigma2))

# print message that simulation is done
cat(paste(Sys.time(), ": finished.\n"))



