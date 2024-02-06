library(tidyverse)
library(ggplot2)
library(pracma)
library(tidyverse)
theme_set(theme_bw(base_size = 14))
file_list <- list.files(path = "~/Downloads/results_jihee/Swissroll_defaultab_300/", 
                         pattern = "*", full.names = TRUE)
results <- bind_rows(lapply(file_list, read.csv))

results <- results %>%
  mutate(calinski_harabasz_score_embeds_log = log10(calinski_harabasz_score_embeds))
summ_mean = results %>% group_by(model_name,
                            hid_dim,
                            out_dim,
                            lr,
                            min_dist,
                            edr, fmr,
                            tau, lambd,
                            pred_hid,alpha_gnn,
                            beta_gnn, gnn_type,
                            n_neighbors) %>%
  summarise_if(is.numeric, mean, na.rm=TRUE)

summ_q50 = results %>% group_by(model_name,
                                 hid_dim,
                                 out_dim,
                                 lr,
                                 min_dist,
                                 edr, fmr,
                                 tau, lambd,
                                 pred_hid,alpha_gnn,
                                 beta_gnn, gnn_type,
                                 n_neighbors) %>%
  summarise_if(is.numeric, quantile, 0.5, na.rm=TRUE)

summ_q25 = results %>% group_by(model_name,
                                hid_dim,
                                out_dim,
                                lr,
                                min_dist,
                                edr, fmr,
                                tau, lambd,
                                pred_hid,alpha_gnn,
                                beta_gnn, gnn_type,
                                n_neighbors) %>%
  summarise_if(is.numeric, quantile, 0.25, na.rm=TRUE)

summ_q75 = results %>% group_by(model_name,
                                hid_dim,
                                out_dim,
                                lr,
                                min_dist,
                                edr, fmr,
                                tau, lambd,
                                pred_hid,alpha_gnn,
                                beta_gnn, gnn_type,
                                n_neighbors) %>%
  summarise_if(is.numeric, quantile, 0.75, na.rm=TRUE)


summ = rbind(summ_mean,
             summ_q50,
             summ_q25,
             summ_q75)



legend_order <- c("DGI",  "BGRL",
                  "GRACE",  "CCA-SSG" , "GNUMAP2"  , 
                  "PCA", "LaplacianEigenmap", "Isomap",
                  "TSNE",  "UMAP",  "DenseMAP")


my_colors <- c("burlywood1","yellow", "chartreuse2",  "orange", "dodgerblue", 
               "#999999", 
               "mediumpurple3", "orchid4",
               "red", "brown", "burlywood3")

labels_n <-  c("DGI (Veličković et al)",  
               "BGRL (Thakoor et al)",
               "GRACE (Zhu et al)",  
               "CCA-SSG (Zhang et al)" , 
               "GNUMAP (this paper)", 
               "PCA", 
               "Laplacian Eigenmap (Belkin et al)", 
               "Isomap (Tenenbaum et al)",
               "TSNE (van der Maaten et al)", 
               "UMAP (McInnes et al)",
               "DenseMAP (Narayan et al)" )

labels_n2 <-  c("DGI",  
               "BGRL",
               "GRACE",  
               "CCA-SSG" , 
               "GNUMAP", 
               "PCA", 
               "Laplacian Eigenmap", 
               "Isomap",
               "TSNE", 
               "UMAP",
               "DenseMAP" )



results_long = pivot_longer(results %>% select(-c(X)),
                            cols = -c("model_name",
                                      "hid_dim",
                                      "out_dim",
                                      "lr",
                                      "min_dist",
                                      "edr", "fmr",
                                      "tau", "lambd",
                                      "pred_hid","alpha_gnn",
                                      "beta_gnn", "gnn_type",
                                      "n_neighbors"))


aggregated_results <- results_long %>%
  group_by(model_name, name) %>%
  summarize(mean_value = mean(value, na.rm = TRUE),
            sd_value = sd(value, na.rm = TRUE),
            median_value = quantile(value, 0.5,na.rm = TRUE),
            q25_value = quantile(value, 0.25,na.rm = TRUE),
            q75_value = quantile(value, 0.75,na.rm = TRUE),
            min_value = min(value, na.rm = TRUE))





# Opt



results_long$model_name <- factor(results_long$model_name, levels = c("DGI",  "BGRL",
                                                                  "GRACE",  "CCA-SSG" , "GNUMAP2"  , 
                                                                  "PCA", "LaplacianEigenmap", "Isomap",
                                                                  "TSNE",  "UMAP",  "DenseMAP",
                                                                  "GNUMAP"))


labels_named_vector <- setNames(labels_n2, legend_order)

ggplot(results_long %>% filter(name %in% c("acc",
                                           #"acc_linear",
                                           "silhouette_embeds",
                                           "calinski_harabasz_score_embeds_log",
                                           "davies_bouldin_score_embeds",
                                           "spearman_graph"), model_name !="GNUMAP"),
       aes(x=model_name,
           y=value,
           fill=model_name)) +
  geom_boxplot(outlier.size = 0.7, outlier.alpha = 0.5) + 
  scale_fill_manual(values = my_colors, breaks = legend_order,
                     labels = labels_n) +
  scale_x_discrete(labels = labels_named_vector) +
  facet_wrap(vars(name), nrow=2, scales="free",labeller = as_labeller(c(`acc` = "Classification Accuracy",
                                                                        `acc_linear` = "Linear Classification Accuracy",
                                                                        `silhouette_embeds` = "Silhouette Score",
                                                                        `calinski_harabasz_score_embeds_log` = "Calinski Harabasz Score",
                                                                        `davies_bouldin_score_embeds` = "Davies Bouldin Score",
                                                                        `spearman_graph` = "Spearman Correlation\nwith Original Graph"
                                                                      )))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(fill = "Model") +
  xlab("Model") + 
  ylab("") + 
  scale_y_log10()
  
  

p_facet <- ggplot(aggregated_results %>% filter(model_name != "GNUMAP",
                                     name %in% c("acc",
                                                 #"acc_linear",
                                                 "silhouette_embeds",
                                                 "calinski_harabasz_score_embeds_log",
                                                 "davies_bouldin_score_embeds"
                                                 )),
       aes(x = model_name, y = median_value, fill = model_name)) +
  geom_col() +
  geom_errorbar(aes(ymin = q25_value, ymax = q75_value), width = 0.2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(fill = "Model") +
  xlab("") + 
  ylab("") +
  scale_fill_manual(values = my_colors, breaks = legend_order,
                                      labels = labels_n) +
  scale_x_discrete(labels = labels_named_vector) +
  facet_wrap(vars(name), nrow=2, scales="free_y",labeller = as_labeller(c(`acc` = "Classification Accuracy (↑) ",
                                                                        `acc_linear` = "Linear Classification Accuracy (↑)",
                                                                        `silhouette_embeds` = "Silhouette Score (↑)",
                                                                        `calinski_harabasz_score_embeds_log` = "Calinski Harabasz Score (↑)",
                                                                        `davies_bouldin_score_embeds` = "Davies Bouldin Score (↓)",
                                                                        `spearman_graph` = "Spearman Correlation\nwith Original Graph (↑)"
  )))
  

# Assuming 'additional_data' is your dataset containing 'neighbors' and 'accuracy'
p_neighbors <- ggplot(aggregated_results %>% filter(model_name != "GNUMAP",
                                                    name %in% c("neighbor_1",
                                                                "neighbor_10",
                                                                "neighbor_20",
                                                                "neighbor_30",
                                                                "neighbor_3" ,
                                                                "neighbor_5",
                                                                "neighbor_50")) %>%
                                                    select(name, model_name, 
                                                           mean_value,
                                                           median_value,
                                                           q25_value,
                                                           q75_value) %>%
                        mutate(n_neighbours = as.numeric(sub("\\D+", "", name )),
                               name_plot ="Overlap Value (in %)" ),
                      aes(x = n_neighbours, y = 100 * median_value, colour = model_name)) +
  geom_point(size=2) +
  geom_line(linewidth=1.) +
  geom_errorbar(aes(ymin = 100 * q25_value,
                  ymax = 100 * q75_value, 
                  colour = model_name), width=0.05,
                linewidth=0.8, alpha=0.5) + 
  scale_color_manual(values = my_colors, breaks = legend_order,
                    labels = labels_n) +
  scale_y_log10() + 
  scale_x_log10() + 
  xlab("Number of Neighbours") + 
  ylab("") + 
  facet_wrap(vars(name_plot), nrow=1, scales="free") +
  theme(legend.position = "none")
  
p_neighbors
# Combine the plots
library(gridExtra)
library(gridExtra)
p_facet2 <- ggplot(aggregated_results %>% filter(model_name != "GNUMAP",
                                                name %in% c("frechet",
                                                            "spearman_graph")),
                  aes(x = model_name, y = median_value, fill = model_name)) +
  geom_col() +
  geom_errorbar(aes(ymin = q25_value, ymax = q75_value), width = 0.2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(fill = "Model") +
  xlab("") + 
  ylab("") +
  scale_fill_manual(values = my_colors, breaks = legend_order,
                    labels = labels_n) +
  scale_x_discrete(labels = labels_named_vector) +
  facet_wrap(vars(name), nrow=3, scales="free_y",labeller = as_labeller(c(`acc` = "Classification Accuracy (↑) ",
                                                                        `acc_linear` = "Linear Classification Accuracy (↑)",
                                                                        `silhouette_embeds` = "Silhouette Score (↑)",
                                                                        `frechet` = "Frechet Distance (↓)",
                                                                        `calinski_harabasz_score_embeds_log` = "Calinski Harabasz Score (↑)",
                                                                        `davies_bouldin_score_embeds` = "Davies Bouldin Score (↓)",
                                                                        `spearman_graph` = "Spearman Correlation\nwith Original Graph (↑)"
  ))) + theme(legend.position = "none")


# Arrange the plots with specified relative widths
p1 <-grid.arrange(p_neighbors, p_facet2, ncol = 1,heights = c(0.4, 0.6))
p2 <-grid.arrange(p1, p_facet, ncol = 2, widths = c(0.3, 0.7))


p2