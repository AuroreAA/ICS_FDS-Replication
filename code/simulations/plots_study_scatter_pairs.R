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
library("scales")

# function for parsing axis labels
parse_labels <- function(labels, ...) parse(text = labels, ...)


# Import results for dimension reduction ---------------------------------------
suffix = "n=1000_r=50_sigma1=1.000000_sigma2=1"
load(paste0("results/simulations/simulations_study_scatter_pairs_",suffix, ".RData"))



# rename some labels and filter relevant results
res_df <- results %>%
  mutate(
         clusters = recode(clusters,
                           "10-10-10-10-10-10-10-10-10-10" =  "10-10-10-10-10\n10-10-10-10-10",
                           "5-5-5-5-5-5-5-15-20-30" = "5-5-5-5-5-5\n5-15-20-30  " ,
                           "8-10-10-10-10-10-10-10-10-12" =  "8-10-10-10-10\n10-10-10-10-12")
         
  ) %>%
  filter(!(clusters %in% c("5-5-5-10-10-10-10-10-15-20", "33-33-33")))


scatter_ordered <- c(
  "COV-COV[4]", "COVAXIS-COV", "TCOV-COV", 
  # MCD-COV scatter pairs
  sprintf("MCD[0.%d]-COV", c(25, 50, 75))
)

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
                    "10-10-10-10-10\n10-10-10-10-10",
                    "8-10-10-10-10\n10-10-10-10-12",
                    "5-5-5-5-5-5\n5-15-20-30  ")  

# heatmap -----------------------------------------------------------------
df_sub <- res_df %>% 
  select(clusters, scatter, selected, q, Run) %>% 
  filter(clusters %in% keep_clusters)


dfsub3 <- data.frame( df_sub %>% separate_rows(selected, sep = '\\,')) %>% 
  rename(IC = selected) %>% 
  group_by(q, scatter) %>% 
  mutate(nb_q = length(unique(clusters)))

dfsub3$IC <- sapply(dfsub3$IC, function(x)
  paste0(gsub("\\.","[",x),"]")
)


df_sub_stats_scatter <- dfsub3 %>% 
  group_by(scatter, q, IC, nb_q) %>% 
  summarize(nb = n())

R <- max(res_df$Run)
p <- max(res_df$p)
all_ICS <- paste0("IC[", 1:p, "]")




df_sub_stats_clusters <- dfsub3 %>% 
  group_by(clusters, scatter, IC, q) %>% 
  mutate( clusters = factor(clusters, levels = keep_clusters),
          IC = factor(IC, levels = all_ICS)) %>% 
  summarize(pct = round(n()/R*100,2))


# create plot
text_size_factor <- 8/6.5

# find the length of the longest label
max_length <- 15


unique_clusters <- keep_clusters
unique_clusters_pad <- unique(stringr::str_pad(unique_clusters, 15, side = "left"))




for (q_val in unique(res_df$q)){
  
  legend_position <- ifelse(q_val==2, "top", "none")
  
  df_sub_q <-  df_sub_stats_clusters %>% 
    filter(q==q_val) %>% 
    mutate(scatter = factor(scatter, levels = scatter_ordered),
           IC = factor(IC, levels = all_ICS),
           clusters = factor(clusters, levels = unique_clusters,
                             labels = unique_clusters_pad)
    )
  
  all_IC_q <- unique(df_sub_q$IC)
  all_IC_q_val <- sort(as.numeric(gsub("\\D", "", all_IC_q)))
  ind_gap <- which((all_IC_q_val - lag(all_IC_q_val, default = 0))>1)[1]
  ind_gap_line <- all_IC_q_val[ind_gap-1]
  
  
  
  # file name for plot
  file_plot <- "figures/simulations/simulations_study_scatter_pairs_%s_q_%d_IC_clusters.%s"
  # save plot to pdf
  pdf(sprintf(file_plot, suffix, q_val, "pdf"), width = 20, height = 4)
  
  
  
  res_clusters <- df_sub_q %>% 
    ggplot( aes( IC, clusters, fill = pct)) + 
    geom_tile() +
    geom_vline(xintercept = ind_gap_line + 0.5, color = "lightgrey", linewidth = 1.5) + 
    labs(x = "", y = "") +
    theme(
      legend.position =  legend_position
    )+
  
    scale_x_discrete(
      labels =  parse_labels,
      guide = guide_axis(n.dodge = ifelse(q_val == 10, 2, 1))
    )+
    
    scale_fill_gradient(low="#c8e8fa", high="#043c59",
                        name =" % of selected IC \n over 50 replications",
                        breaks = c(1,25*(1:4)),
                        labels = percent(0.25*(0:4)) )+
    theme_bw() +
    theme(axis.title = element_text(size = 11 * text_size_factor),
          axis.text = element_text(size = 9 * text_size_factor),
          #axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
          legend.position = legend_position,
          plot.title = element_text(hjust = 0.5),
          legend.title = element_text(size = 11 * text_size_factor),
          legend.text = element_text(size = 9 * text_size_factor),
          panel.spacing.y = unit(0.5, "lines"),
          strip.text = element_text(size = 10 * text_size_factor),
          legend.key.width = unit(2, "lines"))+
    facet_grid(.~scatter,
               labeller = labeller(scatter = label_parsed))
  print(res_clusters)
  
  dev.off()
}
