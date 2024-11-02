# *********************************************
# Authors: Aurore Archimbaud 
#          TBS Business School
# *********************************************


# Packages and functions -------------------------------------------------------
library("dplyr")
library("ggplot2")
library("ggthemes")
library("magrittr")
library("tidyr")
library("plotly")

# function for parsing axis labels
parse_labels <- function(labels, ...) parse(text = labels, ...)



# Import results  ---------------------------------------

# load file

suffix <- "simulations_study_group_means_cov_cov4_n=1000_r=50_sigma1=1.000000_sigma2=1" 
load(paste0("results/simulations/",suffix, ".RData"))


# rename some labels and filter relevant results
res_df <- results %>%  
  filter( ! clusters %in% c("33-33-33", "20-80",
                            "20-30-50", "60-20-20",
                            "10-20-70",
                            "10-10-10-30-40",
                            "10-10-10-10-60",
                            "5-5-5-10-10-10-10-10-15-20"),
          scatter == "COV-COV[4]") %>% 
  mutate(n = factor(n, levels = c(1000), 
                    labels = c("n = 1000")))

# create a data frame to store on which components the structure should be found
# for each cluster from the theoretical results
df_zones <- data.frame(matrix(c(
  c(2, "50-50", 0, 1),                        
  c(2, "40-60", 0, 1),                     
  c(2, "30-70", 0, 1),                        
  c(2, "21-79", 0, 0),                        
  c(2, "10-90", 1, 0),                        
  c(3, "33-33-34", 0, 2),                     
  c(3, "18-32-50", 0, 1),                   
  c(3, "10-40-50", 1, 1),                   
  c(3, "10-30-60", 1, 1),                   
  c(3, "10-10-80", 2, 0),                 
  c(5, "20-20-20-20-20", 0, 4),               
  c(5, "14-20-20-20-26", 0, 3),              
  c(5, "10-20-20-20-30", 1, 3),              
  c(5, "10-10-20-20-40", 2, 2),              
  c(10, "10-10-10-10-10-10-10-10-10-10", 0, 9),
  c(10, "5-5-5-5-5-5-5-15-20-30", 7,2),      
  c(10, "8-10-10-10-10-10-10-10-10-12", 0,8)), ncol = 4, byrow = TRUE))
colnames(df_zones) <- c("q", "clusters", "nb_first", "nb_last")
df_zones <- df_zones %>% 
  mutate(q = as.numeric(q),
         clusters = factor(clusters, levels = unique(res_df$clusters)),
         nb_first = as.numeric(nb_first),
         nb_last = as.numeric(nb_last)) %>% 
  pivot_longer(c(nb_first, nb_last),  names_to = "ind", values_to = "nb")


df_zones_threshold <- data.frame(
  q = c(2,3,5,10), 
  clusters = c("21-79", "18-32-50", "14-20-20-20-26", "8-10-10-10-10-10-10-10-10-12"),
  start = c(1.5,2.5, 4.5, 9.5),
  end = c(2.5, 3.5, 5.5,10.5)) 

# Eigenvalues plot --------------------------------------------------------
## Parameters for the plot
keep_clusters <-  c("50-50",
                    "40-60",
                    "30-70",
                    "21-79",
                    "10-90",
                    "33-33-34",
                    "18-32-50",
                    "10-40-50",
                    "10-30-60",
                    "10-10-80",
                    "20-20-20-20-20", 
                    "14-20-20-20-26",
                    "10-20-20-20-30",  
                    "10-10-20-20-40",
                    "10-10-10-10-10-10-10-10-10-10",
                    "8-10-10-10-10-10-10-10-10-12",
                    "5-5-5-5-5-5-5-15-20-30")       
keep_scatter <- unique(res_df$scatter)
keep_delta <- unique(res_df$delta)

res_df_sub <- res_df %>%  pivot_longer(
  cols = starts_with("IC."),
  names_to = "IC",
  values_to = "lambda"
)

res_df_sub$rho <- sapply(res_df_sub$IC, function(x)
  paste0(gsub("IC.", "rho[", x), "]"))

eigenvalues_label <- "Generalized eigenvalues"
mean_label <- "delta"
text_size_factor <- 8/6.5
text_size_factor <- 1
colors_mean_lst <- ggthemes::colorblind_pal()(5)


# Create plots: one for each number of clusters

for (q_val in unique(res_df$q)){
  
  # Legend
  legend_position <- ifelse(q_val==2, "top", "none")
  
  # file name for plot
  file_plot <- "figures/simulations/%s_q_%d_eigenvalues.%s"
  # save plot to pdf
  pdf(sprintf(file_plot, suffix, q_val, "pdf"), width = 12, height = 3)
  
  # Preprocess to keep only data linked to q clusters
  df_sub_q <- res_df_sub %>% filter(q == q_val, 
                                 !(is.na(lambda)))
  df_zones_sub <- df_zones %>% filter(q == q_val, nb != 0)
  
  df_zones_threshold_sub <- df_zones_threshold %>% filter(q == q_val) %>% 
    mutate(
      clusters = factor(clusters, levels = keep_clusters, ordered = TRUE)
    )
  p <- df_sub_q$p[1]
  k <- q_val - 1
  rho_sel <- paste0("rho[", c(1:k, p-((k-1):0)),"]")
  rho_sel_q_val <- sort(as.numeric(gsub("\\D", "",  rho_sel)))
  ind_gap <- which((rho_sel_q_val - lag(rho_sel_q_val, default = 0))>1)[1]
  ind_gap_line <- rho_sel_q_val[ind_gap-1]
  nb_all <- length(rho_sel)
  width <- 0.5
  df_zones_sub <- df_zones_sub %>% 
    mutate(start = ifelse(ind == "nb_first", 0+width, nb_all-nb-width+1),
           end = ifelse(ind == "nb_first", nb+width, nb_all+width)) %>% 
    mutate(
      clusters = factor(clusters, levels = keep_clusters, ordered = TRUE)
    )
  
  
  # Plot
  plot_eigenvalues <-  df_sub_q  %>%
    filter(rho %in% rho_sel) %>% 
    mutate(
      clusters = factor(clusters, levels = keep_clusters, ordered = TRUE),
      scatter = factor(scatter, levels = keep_scatter),
      delta = factor(delta, levels = keep_delta),
      rho = factor(rho, levels = unique(df_sub_q$rho))
    ) %>%
    ggplot(mapping = aes_string(x = "rho", y = "lambda",
                                color = "delta", fill = "delta")) +
    geom_boxplot(alpha = 0.4, position = position_dodge2()) +
    geom_hline(yintercept=1, linetype="dashed", 
               color = "red") +
    geom_vline(xintercept = ind_gap_line + 0.5) + 
    geom_rect(data = df_zones_sub,
              inherit.aes = FALSE,
              aes(xmin = .data$start, xmax = .data$end,
                  ymin = -Inf, ymax = Inf),
              alpha = 0.25, fill = "grey")+
    geom_rect(data = df_zones_threshold_sub,
              inherit.aes = FALSE,
              aes(xmin = .data$start, xmax = .data$end,
                  ymin = -Inf, ymax = Inf),
              alpha = 0.1, fill = "red")+
    scale_color_manual( values = colors_mean_lst, guide = "none") +
    scale_fill_manual( values = colors_mean_lst) +
    scale_x_discrete(labels = parse_labels) +
    theme_bw() +
    theme(axis.title = element_text(size = 11 * text_size_factor),
          axis.text = element_text(size = 9 * text_size_factor),
          legend.position = legend_position,
          plot.title = element_text(hjust = 0.5),
          legend.title = element_text(size = 11 * text_size_factor),
          legend.text = element_text(size = 9 * text_size_factor),
          panel.spacing.y = unit(0.5, "lines"),
          strip.text = element_text(size = 10 * text_size_factor)) +
    labs(x = "", y = "", 
         fill = parse_labels(mean_label))+
    facet_grid(.~clusters,
               labeller = labeller(scatter = label_parsed))
  
  
  print(plot_eigenvalues)
  
  
  dev.off()
}

