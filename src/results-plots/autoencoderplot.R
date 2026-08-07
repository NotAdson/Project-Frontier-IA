library(readr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop(
    "Uso: Rscript autoencoderplot.R <entrada.csv> <diretorio_saida>",
    call. = FALSE
  )
}

entrada <- args[[1]]
diretorio_saida <- args[[2]]

dir.create(diretorio_saida, recursive = TRUE, showWarnings = FALSE)

dados <- read_csv(entrada, show_col_types = FALSE)

tempo_total_horas <- sum(dados$time_seconds) / 3600
melhor_epoca <- dados[which.min(dados$val_loss), ]
texto_melhor_epoca <- sprintf(
  "Melhor época: %d\nval_loss = %.4f",
  melhor_epoca$epoch,
  melhor_epoca$val_loss
)

grafico <- ggplot(dados, aes(x = epoch)) +
  geom_line(
    aes(y = train_loss, color = "Treinamento"),
    linewidth = 1
  ) +
  geom_line(
    aes(y = val_loss, color = "Validação"),
    linewidth = 1
  ) +
  annotate(
    "label",
    x = Inf,
    y = Inf,
    label = texto_melhor_epoca,
    hjust = 1.1,
    vjust = 1.2
  ) +
  scale_x_continuous(
    limits = c(0, NA),
    breaks = scales::breaks_width(5)
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = scales::breaks_width(0.05)
  ) +
  labs(
    title = "Loss do autoencoder ao longo das épocas",
    subtitle = sprintf(
      "Tempo total de treinamento: %.2f horas",
      tempo_total_horas
    ),
    x = "Época",
    y = "Loss",
    color = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

arquivo_saida <- file.path(
  diretorio_saida,
  "loss_autoencoder.png"
)

ggsave(
  filename = arquivo_saida,
  plot = grafico,
  width = 10,
  height = 6,
  dpi = 300
)

cat("Gráfico salvo em:", arquivo_saida, "\n")
