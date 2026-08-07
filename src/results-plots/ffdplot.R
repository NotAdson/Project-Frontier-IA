library(ggplot2)
library(jsonlite)

# Configurações da matriz

geracao_melhor_agente <- 38

agentes_considerados <- data.frame(
  nome_json = c(
    paste("Model Gen", geracao_melhor_agente),
    "Model Gen 36",
    "Random Agent",
    "MCTS Puro"
  )
)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop(
    "Uso: Rscript ffdplot.R <diretorio_entrada> <diretorio_saida>",
    call. = FALSE
  )
}

diretorio_dados <- args[[1]]
diretorio_saida <- args[[2]]

dir.create(diretorio_saida, recursive = TRUE, showWarnings = FALSE)

dados <- read.csv(
  file.path(diretorio_dados, "training_log.csv")
)


# PLOTS DE LOSS

dados$epoca_global <- seq_len(nrow(dados)) - 1
ultima_epoca <- max(dados$epoca_global)

salvar_loss <- function(
    coluna_treino,
    coluna_validacao,
    titulo,
    arquivo_saida
) {
  dados_plot <- data.frame(
    epoca = dados$epoca_global,
    treinamento = dados[[coluna_treino]],
    validacao = dados[[coluna_validacao]]
  )
  
  grafico <- ggplot(dados_plot, aes(x = epoca)) +
    geom_line(
      aes(y = treinamento, color = "Treinamento"),
      linewidth = 0.9
    ) +
    geom_line(
      aes(y = validacao, color = "Validação"),
      linewidth = 0.9
    ) +
    scale_x_continuous(
      limits = c(0, NA),
      breaks = unique(c(
        seq(0, ultima_epoca, by = 100),
        ultima_epoca
      ))
    ) +
    scale_y_continuous(
      limits = c(0, NA),
      breaks = scales::breaks_pretty(n = 10)
    ) +
    labs(
      title = titulo,
      x = "Época",
      y = "Erro",
      color = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold")
    )
  
  ggsave(
    filename = file.path(diretorio_saida, arquivo_saida),
    plot = grafico,
    width = 10,
    height = 6,
    dpi = 300
  )
}

salvar_loss(
  "policy_loss",
  "val_policy_loss",
  "Erro na escolha da próxima ação",
  "escolha_acoes_loss.png"
)

salvar_loss(
  "dynamic_matching_loss",
  "val_dynamic_matching_loss",
  "Erro ao prever o pokémon adversario",
  "prever_adversario_loss.png"
)

salvar_loss(
  "value_loss",
  "val_value_loss",
  "Erro na previsão do resultado da batalha",
  "previsao_resultado_loss.png"
)


# MATRIZ DE TAXA DE VITÓRIA

arquivo_benchmark <- file.path(
  diretorio_dados,
  paste0("gen", geracao_melhor_agente),
  "benchmark_report.json"
)

partidas <- fromJSON(arquivo_benchmark)
partidas <- partidas[, c("p1", "p2", "winner")]

nomes_agentes <- agentes_considerados$nome_json

matriz <- expand.grid(
  agente = nomes_agentes,
  adversario = nomes_agentes,
  stringsAsFactors = FALSE
)

matriz$partidas <- 0
matriz$vitorias <- 0

for (i in seq_len(nrow(matriz))) {
  agente <- matriz$agente[i]
  adversario <- matriz$adversario[i]
  
  if (agente == adversario) {
    next
  }
  
  confronto <- partidas[
    (
      partidas$p1 == agente &
        partidas$p2 == adversario
    ) |
      (
        partidas$p1 == adversario &
          partidas$p2 == agente
      ),
  ]
  
  matriz$partidas[i] <- nrow(confronto)
  matriz$vitorias[i] <- sum(confronto$winner == agente)
}

matriz$taxa_vitoria <- ifelse(
  matriz$partidas == 0,
  NA,
  100 * matriz$vitorias / matriz$partidas
)

matriz$rotulo <- ifelse(
  is.na(matriz$taxa_vitoria),
  "Sem dados",
  sprintf(
    "%.1f%%\n(n = %d)",
    matriz$taxa_vitoria,
    matriz$partidas
  )
)

grafico_matriz <- ggplot(
  matriz,
  aes(
    x = adversario,
    y = agente,
    fill = taxa_vitoria
  )
) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = rotulo), size = 4) +
  scale_fill_gradient(
    limits = c(0, 100),
    low = "white",
    high = "steelblue",
    na.value = "grey90",
    labels = function(x) paste0(x, "%")
  ) +
  coord_equal() +
  labs(
    title = "Taxa de vitória entre os agentes",
    subtitle = "Taxa de vitória do agente da linha contra o agente da coluna",
    x = "Adversário",
    y = "Agente",
    fill = "Taxa de vitória"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1),
    plot.title = element_text(face = "bold")
  )

ggsave(
  filename = file.path(
    diretorio_saida,
    "matriz_taxa_vitoria.png"
  ),
  plot = grafico_matriz,
  width = 9,
  height = 7,
  dpi = 300
)

cat("Gráficos salvos em:", diretorio_saida, "\n")
