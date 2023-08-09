library(magrittr)

df <- data.table::fread("scaling_squaring.csv")


# ERROR
# vary with: tess_size, basis, method, epsilon

df %>% 
  dplyr::filter(zero_boundary) %>% 
  dplyr::filter(basis == "svd") %>% 
  dplyr::mutate(combined = paste0(
    # "Basis: ", basis, " ",
    "Method: ", method, " ",
    "Eps: ", epsilon, " "
  )) %>% 
  ggplot2::ggplot()+
  ggplot2::geom_line(ggplot2::aes(x=N, y=error, color=factor(tess_size)))+
  ggplot2::geom_point(ggplot2::aes(x=N, y=error, color=factor(tess_size)))+
  ggplot2::facet_wrap(~ combined, scales="free", ncol=4)


# TIME
# vary with: tess_size, basis, method, epsilon

df %>% 
  dplyr::filter(zero_boundary) %>% 
  dplyr::filter(basis == "svd") %>% 
  dplyr::mutate(combined = paste0(
    # "Basis: ", basis, " ",
    "Method: ", method, " ",
    "Eps: ", epsilon, " ",
    "Threads: ", num_threads, " "
  )) %>% 
  ggplot2::ggplot()+
  ggplot2::geom_point(ggplot2::aes(x=N, y=elapsed_time, color=factor(tess_size)))+
  ggplot2::geom_line(ggplot2::aes(x=N, y=elapsed_time, color=factor(tess_size)))+
  ggplot2::facet_wrap(~ combined, scales="free", ncol=4)



df %>% 
  tidyr::pivot_longer(cols=c(elapsed_time, error)) %>% 
  ggplot2::ggplot()+
  ggplot2::geom_point(ggplot2::aes(x=N, y=value, color=factor(epsilon)))+
  ggplot2::geom_line(ggplot2::aes(x=N, y=value, color=factor(epsilon)))+
  ggplot2::facet_wrap(name ~ method, scales="free")+
  ggplot2::scale_colour_viridis_d()+
  ggplot2::theme_bw()+
  # ggpubr::theme_pubclean()+
  ggplot2::theme(
    legend.position = "bottom"
  )
  # ggplot2::facet_grid(name ~ method, scales="free")

