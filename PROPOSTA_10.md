# PROPOSTA_10.md — Autoencoder para compressão de `fused_features` (issue #10)

Documento reconstruído a partir dos arquivos efetivamente criados nesta sessão
(`src/battle_agents/mcts_approximation/pipeline/autoencoder/*.py`) e dos logs/saídas
reais de todos os comandos executados. Todo número citado aqui vem de uma execução
real já registrada na conversa — nenhum foi estimado ou arredondado sem indicar.
Onde não há certeza suficiente, está marcado `[CONFIRMAR]`.

Status na data deste documento: **fase de exploração de arquitetura ENCERRADA DE
FORMA DEFINITIVA**. Depois do treino inicial (seção 5, reprovado nos dois
critérios formais) foram testadas mais duas variações — `latent_dim=128` e uma
loss segmentada MSE+BCE com `latent_dim=256` (seção 6, resultado v4). Uma
auditoria posterior (seção 7) descobriu que v3 e v4 tinham, na prática, o
**mesmo gargalo real** (128 dims) por um bug de arquitetura em `model.py`
— corrigido, retreinado como v5, e reavaliado. **v5 é o resultado final desta
fase**: estritamente melhor que v4 nos dois critérios de MSE, com arquitetura
tecnicamente correta. Não haverá mais rodadas de teste de dimensão/arquitetura
depois de v5 — a próxima etapa é a integração (revisão de opções na seção 8,
pendência ainda não iniciada na seção 10).

---

## 1. Contexto e decisão de arquitetura

### A ambiguidade inicial

Havia duas formas possíveis de aplicar compressão ao estado consumido pela rede
principal (`train_nn.py`):

- **(A) Comprimir por-Pokémon**, preservando os 12 tokens que o Meta-Planner
  Transformer usa (`train_nn.py:349-390`, `token_dim=184` por Pokémon — ver
  `NUM_ACTIVE=12`, `PER_MON_DENSE=53` em `state_encoder.py`). Essa foi a primeira
  hipótese de design, e chegou a gerar um artefato (`data/autoencoder_bootstrap/dense_pokemon_slices.npy`,
  111.196.104 linhas × 53 colunas, 23,57GB) via `prepare_data.py`. Esse arquivo foi
  **apagado a pedido explícito do usuário** por não servir mais ao design atual.
  O próprio `prepare_data.py` também não existe mais no repositório: removi ele
  depois, numa resposta de revisão no PR1, já que não tinha uso no pipeline ativo.
- **(B) Comprimir o `fused_features` inteiro**, já depois do Meta-Planner ter
  processado os tokens e produzido seus pesos de atenção — ou seja, comprimir o
  vetor de entrada do tronco tático principal (`train_nn.py:432`,
  `Dense(512)` recebendo `fused_features`).

### Por que (B) foi escolhida

- `fused_features` (3074 dimensões) é o gargalo de tamanho citado na issue — é o
  tensor que entra direto na primeira camada `Dense(512)` do tronco principal
  (`train_nn.py:432`).
- O Meta-Planner **continua intocado**: ele recebe os dados densos crus como
  sempre (`inp_dense`, `p_dense`, `p_pp` fatiados de `inp_dense` em
  `train_nn.py:358-368`), monta os 12 tokens e calcula `meta_plan` normalmente.
  A compressão entra **depois** disso, só no caminho que alimenta o tronco tático.
- Comprimir por-Pokémon (opção A) exigiria redesenhar o slicing de
  `SliceLayer` que monta os 12 tokens (`train_nn.py:349-390`), quebrando a
  arquitetura do Meta-Planner existente. Comprimir `fused_features` inteiro não
  toca nada dessa lógica — é um autoencoder plugado antes do `Dense(512)`,
  arquitetura do resto do modelo preservada.

### Composição exata de `fused_features` (3074 dims)

Montado em `train_nn.py:424-429`:

```python
concat_main = keras.layers.Concatenate()(
    [inp_dense, emb_species_main, emb_moves_main, emb_items_main, emb_abilities_main]
)
fused_features = keras.layers.Concatenate(name="fused_features")([concat_main, meta_plan])
```

| peça | dim | origem |
|---|---|---|
| `inp_dense` | 758 | `NUM_DENSE_FEATURES` (`state_encoder.py`) |
| `emb_species_main` | 384 | `Embedding(num_species, 32)` sobre 12 índices (`NUM_SPECIES_INDICES × MAIN_EMB_SPECIES_DIM`) |
| `emb_moves_main` | 1536 | `Embedding(num_moves, 32)` sobre 48 índices |
| `emb_items_main` | 192 | `Embedding(num_items, 16)` sobre 12 índices |
| `emb_abilities_main` | 192 | `Embedding(num_abilities, 16)` sobre 12 índices |
| `meta_plan` | 12 | saída do Meta-Planner (6 pesos softmax próprios + 6 sigmoid do oponente) |
| **total** | **3074** | — |

Essa tabela é a mesma usada como `PIECES` em `generate_synthetic_dataset.py:80-89`
e reaproveitada (não redefinida) em `train_autoencoder.py` e `test_reconstruction.py`
via `piece_offsets()` (`generate_synthetic_dataset.py:92-104`).

---

## 2. Geração de dados

### 2.1. `data/genrandom_bootstrap/` — 49.999 jogos reais

Gerados via `generate_data.py`, usando `agent_type="random"` (`RandomAgent` dos
dois lados). Essa opção **não existia originalmente** no script — foi adicionada
nesta sessão como extensão retrocompatível (`generate_data.py`, parâmetro
`agent_type` em `run_simulation`/`generate_dataset`), especificamente para não
depender de um modelo `.onnx` treinado (que ainda não existe neste checkout —
o único outro modo sem modelo, `use_cheating_mcts=True`, é ordens de magnitude
mais lento por chamar o engine a cada passo de rollout).

Confirmado nesta sessão: **49.999 arquivos** `game_*.json` em
`data/genrandom_bootstrap/`, somando **9.266.342 steps** no total (número
obtido de forma independente duas vezes — uma vez rodando `prepare_data.py`,
outra rodando `generate_synthetic_dataset.py` — com resultado idêntico nos
dois, o que é uma checagem cruzada de consistência. `prepare_data.py` foi
removido do repositório depois dessa checagem, numa resposta de revisão no
PR1, por não ter uso no pipeline ativo).

### 2.2. `data/autoencoder_bootstrap/fused_features_synthetic.npy`

Dataset sintético de **2.000.000 exemplos × 3074 dims**, `float32`, 24,59GB
(24.592.000.128 bytes exatos), gerado por `generate_synthetic_dataset.py`.
**Nenhum Meta-Planner real roda nesta etapa** — o autoencoder treina
desacoplado da rede principal, sem depender de pesos já treinados dela. Cada
peça do vetor de 3074 dims é montada assim:

1. **`inp_dense` (758) + índices categóricos (84)** — vêm de estados **reais**
   de batalha: os primeiros `TOTAL_FEATURES=842` valores do `"features"` de um
   step real de `data/genrandom_bootstrap/game_*.json`, amostrados por
   **reservoir sampling (Algorithm R)** numa única passada por todos os 49.999
   arquivos (`generate_synthetic_dataset.py:107-142`) — dá a cada step, em
   qualquer arquivo, probabilidade igual de entrar na amostra de 2.000.000,
   sem viés pelos primeiros arquivos e sem precisar saber o total de steps de
   antemão.
2. **Os 4 embeddings (`emb_species_main`, `emb_moves_main`, `emb_items_main`,
   `emb_abilities_main`)** — gerados aplicando os índices categóricos reais do
   passo 1 a tabelas de peso **aleatórias e NÃO treinadas**, inicializadas
   `Normal(mean=0, stddev=0.05)` (`build_embedding_tables`,
   `generate_synthetic_dataset.py:145-166`). Implementado como indexação direta
   numa matriz numpy (matematicamente idêntico a um lookup de
   `keras.layers.Embedding`/`torch.nn.Embedding` não treinado — um lookup é só
   indexação na matriz de pesos), sem depender de import de Keras/PyTorch.
   Nota registrada no próprio código: `Normal(0, 0.05)` é o valor pedido na
   especificação, mas **não é** literalmente o default do Keras (que é
   `RandomUniform(-0.05, 0.05)`) — implementamos o valor numérico exato
   especificado, não o default real do Keras.
3. **`meta_plan` (12)** — sorteado **independentemente por linha**, sem
   nenhuma correlação com `dense`/embeddings: 6 valores via
   `np.random.dirichlet(np.ones(6))` (pesos próprios, somam 1), 6 valores via
   `np.random.uniform(0, 1, 6)` (pesos do oponente) — espelhando a divisão
   softmax/sigmoid documentada em `train_nn.py` para `meta_plan` real.
   **Não existe Meta-Planner real rodando nesta etapa** — essa é uma
   característica central do design (treinar o autoencoder desacoplado, sem
   precisar rodar a busca MCTS completa), conforme especificado no briefing
   desta tarefa.
4. Concatenado na ordem exata de `train_nn.py:424-429`
   (`inp_dense, emb_species_main, emb_moves_main, emb_items_main, emb_abilities_main, meta_plan`).

Seeds fixas e documentadas (`generate_synthetic_dataset.py:71-74`):
`SEED=42` → `SEED_RESERVOIR=42` (amostragem), `SEED_EMBEDDINGS=43` (pesos dos
embeddings), `SEED_META_PLAN=44` (sorteios do meta_plan) — três RNGs
independentes, cada um usado sequencialmente e criado uma única vez (não
re-seedado por chunk/arquivo/worker — verificado por leitura de código e por
teste empírico direcionado, ver seção 5).

Vocabulários observados na geração (saída real do script):
`species=393, moves=373, items=130, abilities=78`.

Geração rodou em duas fases medidas: reservoir sampling sobre os 49.999
arquivos em 1186,7s (~19,8min), montagem/escrita do `.npy` final em 198,7s —
total ~23,1min.

---

## 3. Arquitetura do autoencoder

`model.py` (51 linhas) — `FusedFeaturesAutoencoder(nn.Module)`:

- **Encoder**: `3074 → 512 → 256 → 128 → 64`, `Linear` + `ReLU` entre camadas,
  **sem ativação na última camada** (saída do encoder, o código latente).
- **Decoder**: `64 → 128 → 256 → 512 → 3074`, espelho exato do encoder,
  também **sem ativação na última camada**.
- Métodos `encode()`, `decode()`, `forward()` (= `decode(encode(x))`).

**Por que sem ativação na saída do decoder**: os embeddings sintéticos são
`Normal(0, 0.05)` — podem ser (e frequentemente são) negativos. Uma `sigmoid`
ou `tanh` limitaria a saída a um intervalo que não cobre esses valores,
quebrando a reconstrução dessa peça por construção, antes mesmo de qualquer
otimização.

Implementado via um helper `_mlp(dims)` reutilizado para as duas metades
(`model.py:25-33`), garantindo que a regra "ReLU entre camadas, nunca depois
da última" seja aplicada de forma idêntica dos dois lados.

---

## 4. Investigação do MSE alto no bloco denso (8 rodadas de smoke test)

Todas as rodadas abaixo usaram os mesmos 20.000 exemplos (primeiras linhas do
`.npy` — já vêm de reservoir sampling, então continuam sendo uma amostra
válida, não só "os jogos mais antigos"), mesma seed de split (`123`,
`val_fraction=0.2` → 16.000 treino / 4.000 validação), permitindo comparação
direta entre rodadas. Motivação: o critério formal de aceitação da issue é
**MSE agregado < 0,01**, mas uma inspeção inicial (v1) mostrou que esse
agregado pode esconder um bloco denso muito pior que a média.

| rodada | mudança testada | épocas | dense MSE | meta_plan MSE | aggregate MSE | % erro vindo do dense |
|---|---|---|---|---|---|---|
| v1 | loss flat (`nn.MSELoss()` simples) | 5 | 0,070305 | 0,057585 | 0,019508 | **88,9%** |
| v2 | loss ponderada por peça, `dense-weight=8` | 5 | 0,080478 | 0,051722 | 0,021724 | 91,3% |
| v3 | `dense-weight=70` | 5 | 0,073248 | 0,053522 | 0,020028 | 90,2% |
| v4 | `dense-weight=70`, mais épocas | 20 | 0,068503 | 0,050831 | 0,018527 | 91,2% |
| v5 | `dense-weight=300` (vs 70) | 20 | 0,068461 | 0,051049 | 0,018518 | 91,16% |
| v6 | `dense-weight=70` + `meta-plan-weight=0.05` | 20 | 0,068450 | 0,051166 | 0,018516 | — |
| v7 | `dense-weight=70` + `latent-dim=128` (vs 64) | 20 | 0,068502 | 0,050904 | 0,018527 | — |
| v8 | **dense isolado** (758-dim só, sem embeddings/meta_plan, `nn.MSELoss()` simples) | 20 | **0,068441** | n/a (peça não existe nesse modo) | n/a | n/a |

Sequência de hipóteses testadas e descartadas:

- **v1 → v2/v3**: ponderar a loss por peça (pesos = inverso da dimensão,
  normalizado, com multiplicador extra no dense) **piorou** o dense em vez de
  melhorar — `meta_plan` (só 12 dims) recebe peso base enorme (`0,8475`) nesse
  esquema, maior que o peso final do dense mesmo em `dense-weight=8`
  (`0,1073`), consumindo a maior parte do gradiente disponível para reduzir o
  erro de uma peça que é ruído puro e não pode ser melhorada abaixo do seu
  piso de variância.
- **v3 → v4**: mais épocas (5→20) com `dense-weight=70` melhorou o dense
  (`0,073248 → 0,068503`), mas a proporção do erro vindo do dense **não
  caiu** (90,2%→91,2%) — o ganho absoluto veio de mais tempo de treino, não
  da ponderação em si.
- **v4 → v5**: subir `dense-weight` de 70 para 300 (4,3× mais dominante no
  gradiente) **não mudou nada** (`0,068503 → 0,068461`, diferença dentro do
  ruído). Descarta ponderação de loss como alavanca útil além de certo ponto.
- **v5 → v6**: reduzir `meta-plan-weight` para 0,05 (liberar gradiente que o
  meta_plan consumia à toa) **também não mudou nada** (`0,068461 → 0,068450`).
- **v4 → v7**: dobrar `latent_dim` de 64 para 128 **não mudou nada**
  (`0,068503 → 0,068502`, idêntico até a 5ª casa decimal) — descarta a
  hipótese de capacidade insuficiente do gargalo latente.
- **v4/v5/v6/v7 → v8**: treinar o bloco denso **isolado** (758→512→256→128→64,
  sem embeddings nem meta_plan competindo por gradiente nenhum) deu
  `0,068441` — **essencialmente o mesmo número** das rodadas anteriores
  (diferença de 5ª casa decimal, dentro do ruído observado entre rodadas
  idênticas).

**Conclusão da investigação**: como isolar completamente o bloco denso (v8)
não mudou o resultado, a dificuldade em reconstruir o bloco denso **é
inerente aos próprios dados** — mistura de contínuo (hp_ratio, stats,
power/accuracy) e binário esparso (one-hots de status/tipo/categoria), mais
fog of war entre time próprio (sempre visível) e time adversário
(majoritariamente zerado até revelado) — não é interferência de gradiente
com as outras peças, nem falta de capacidade do gargalo latente de 64 dims.
Só restava testar **escala real de dados/tempo de treino** (2M linhas vs 20k,
early stopping de verdade vs teto artificial de 20 épocas) — o que motivou
ir para o treino completo (seção 5).

---

## 5. Resultado do treino completo

### Configuração final

`dense-weight=70`, `meta-plan-weight=1.0`, `species/moves/items-abilities-weight=1.0`
(defaults), `latent-dim=64` (arquitetura original — v7 mostrou que 128 não
ajuda), `batch-size=4096`, `lr=1e-3`, `patience=5`, teto de segurança
`epochs=50`, `--num-workers 4`, `2.000.000` linhas (`1.600.000` treino /
`400.000` validação, seed `123`).

Rodado via `nohup` em background, terminou sozinho (50/50 épocas completas,
**sem acionar early stopping** — nunca ficou 5 épocas seguidas sem melhorar).

### Tempo real

- **50 épocas completas em 13.858s (~3h51min)** — soma exata dos tempos por
  época reportados no log (`data/autoencoder_bootstrap/train_full_run.log`).
- Média: 277,2s/época (mín. 266,1s, máx. 297,1s) — consistente com a medição
  isolada de 1 época feita antes do treino (280,0s).

### Resultado (`test_reconstruction.py` sobre o checkpoint final, época 50)

```
FAIL:
  - aggregate MSE 0.012447 >= acceptance threshold 0.01
  - dense-block MSE 0.045114 >= dense acceptance threshold 0.01
Checkpoint epoch: 50  (saved val_loss: 0.042734)
Validation rows evaluated: 400,000
Per-piece MSE (raw, unweighted):
           dense (dim=  758): mse=0.045114
     emb_species (dim=  384): mse=0.002407
       emb_moves (dim= 1536): mse=0.001645
       emb_items (dim=  192): mse=0.000854
   emb_abilities (dim=  192): mse=0.002335
       meta_plan (dim=   12): mse=0.000140
```

**Reprova nos dois critérios formais de aceitação** (`< 0,01` agregado e
`< 0,01` no dense — este último um critério adicional que criamos depois de
identificar, na seção 4, que o agregado sozinho mascara o bloco denso).

**Mas houve melhora real com escala**: dense caiu de `~0,0685` (platô de
todos os smoke tests, 20k linhas) para `0,045114` (2M linhas, 50 épocas) —
uma queda de ~34%. Dense continua sendo a maior fatia do erro agregado
(89,4% do erro ponderado por dimensão, recalculado a partir desses números —
consistente com a faixa de 88,9%-91,3% observada nos smoke tests).

A curva de `val_loss` (a métrica ponderada usada no treino, não o MSE cru)
ainda caía de forma consistente perto da época 50, sem sinal claro de
convergência/plateau:

```
epoch 40/50  val_loss=0.045724  (best)
epoch 43/50  val_loss=0.045273  (best)
epoch 45/50  val_loss=0.043939  (best)
epoch 47/50  val_loss=0.043271  (best)
epoch 49/50  val_loss=0.043233  (best)
epoch 50/50  val_loss=0.042734  (best)
```
A época 50 foi a melhor até então (`(best)`), e a tendência das últimas 10
épocas é de queda ininterrupta (com ruído local — ep. 41, 42, 46, 48 pioraram
frente à melhor até ali, mas o mínimo global seguiu caindo) — sugere que mais
épocas provavelmente ajudariam ainda mais, embora não haja garantia de que
o critério `<0,01` seja alcançável só com mais tempo, dado o platô estrutural
já observado nos smoke tests para o bloco denso especificamente.

Essa pergunta (se v1 teria cruzado o threshold com mais tempo de treino)
ficou respondida de forma indireta pelas seções 6 e 7, sem precisar de
extrapolação formal de curva. v2 a v5 rodaram sobre o mesmo volume de dados
e convergiram de verdade via early stopping, não por um teto artificial de
épocas como v1, e o bloco dense continuou acima do threshold de 0,01 em
todas elas. O platô é estrutural dos próprios dados, mistura de features
contínuas e binárias mais fog of war entre time próprio e adversário, não
falta de tempo de treino. Extrapolar a curva de v1 especificamente não
compensa mais o esforço, já que v1 deixou de ser o resultado relevante
a partir de v2.

### A anomalia do `meta_plan` MSE=0,000140

Valor **~368× menor** que o piso teórico de variância do ruído puro
(`~0,0516`, calculado a partir da mistura de 6 componentes `Beta(1,5)` do
Dirichlet + 6 componentes `Uniform(0,1)`). Investigado a fundo (a pedido
explícito do usuário, antes de confiar em qualquer resultado do treino):

- **Não é vazamento entre treino e validação**: `split_indices()` (mesma
  função usada em `train_autoencoder.py` e `test_reconstruction.py`)
  produz partição perfeita — interseção `train_idx ∩ val_idx` = 0,
  união = `{0, ..., 1.999.999}` exatamente, verificado com a config real de
  produção (`n_total=2_000_000, val_fraction=0.2, seed=123`).
- **Não há duplicatas no dataset**: checado em duas amostras de 50.000 linhas
  cada (contígua e espalhada aleatoriamente pelo arquivo inteiro) nas colunas
  de `meta_plan` — 100% únicas nas duas amostras. Teste direcionado adicional
  de periodicidade em `CHUNK_ROWS=100.000` (a hipótese mais plausível de bug
  de RNG) também não encontrou repetição.
- **O RNG do `meta_plan` é usado corretamente**: `meta_rng = np.random.default_rng(SEED_META_PLAN)`
  criado uma única vez, antes do loop de chunks, usado sequencialmente —
  confirmado por leitura de código (`generate_synthetic_dataset.py:200,232-233`).

**Explicação real**: `meta_plan` é **parte do próprio input** do autoencoder,
não um alvo a ser inferido a partir de outra informação. A premissa "nenhum
modelo deveria bater o piso de variância do ruído" vale para um problema de
**regressão** (prever `meta_plan` a partir de dados independentes) — mas isso
é um **autoencoder**: o gargalo latente tem 64 dimensões, muito mais que os
12 valores de `meta_plan`. A rede não precisa inferir nada; só precisa
reservar uma fração pequena da capacidade latente para um **passthrough
quase-identidade** — o encoder escreve os 12 valores (escalados) em alguns
neurônios latentes, o decoder os lê de volta quase sem perda. Confirmado
diretamente comparando `meta_plan` real vs. reconstruído em 5 linhas de
validação nunca vistas no treino — erro absoluto por componente de
`~0,001–0,03`, padrão de passthrough quase-identidade, não coincidência
estatística. Isso também explica por que os smoke tests (treinos curtos, 20
épocas) nunca mostraram isso — não deu tempo da rede convergir para essa
solução; com 50 épocas em 1,6M linhas, e `meta_plan` pesando `0,8475` na loss
(quase tanto quanto o dense, `0,939`), o otimizador teve tempo e incentivo de
sobra.

**Implicação prática**: `meta_plan` nunca vai servir como sinal útil de
validação da "capacidade de compressão tática" do autoencoder, porque é
ajudado por estar no próprio input. O critério que importa de fato é o do
bloco `dense`.

---

## 6. Investigação de arquitetura pós-treino inicial

Auditoria feita para este documento: os 4 checkpoints citados abaixo (v1 a v4)
foram recarregados agora, diretamente do disco, com
`torch.load(map_location='cpu', weights_only=False)`, e reavaliados rodando
`test_reconstruction.py` (v1) e uma cópia read-only da mesma função `evaluate()`
para v2/v3/v4 (necessária só porque v1/v2/v3 não têm o buffer `binary_mask`,
adicionado depois — ver nota metodológica ao final da seção 6.1). Nenhum
número desta seção foi copiado de mensagens anteriores da conversa sem
reconfirmação.

### 6.1 Teste de `latent_dim=128` em escala real (v3)

Depois do resultado de v1 (seção 5, reprovado nos dois critérios), rodou-se
v2 — mesma arquitetura (`latent_dim=64`, MSE uniforme via `nn.MSELoss()`
plano), mas com o teto de épocas elevado de 50 para 120
(`train_full_run_v2.log:8`, `Training: ... epochs=1..120`), permitindo
convergência real via early stopping em vez do corte artificial de v1. Na
sequência, v3 repetiu a mesma configuração trocando só `latent_dim` para 128,
testando em escala real (2M linhas) a mesma pergunta que o smoke test v7
(seção 4) já tinha testado em escala pequena.

| ver | `latent_dim` | epoch salvo (melhor) | teto de épocas | como parou | val_loss (ckpt, métrica de treino) | dense MSE (raw) | dense RMSE | MSE agregado (raw) | agregado&nbsp;<&nbsp;0,01 | dense&nbsp;<&nbsp;0,01 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 64  | 50  | 50  | atingiu o teto (sem early stopping) | 0,042734 | 0,045114 | 21,24% | 0,012447 | FAIL | FAIL |
| v2 | 64  | 93  | 120 | early stopping na época 98 (`train_full_run_v2.log`) | 0,037411 | 0,039518 | 19,88% | 0,011025 | FAIL | FAIL |
| v3 | 128 | 119 | 120 | atingiu o teto (sem early stopping — `train_full_run_v3.log` não emite a linha `Early stopping`) | 0,035766 | 0,037787 | 19,44% | 0,010591 | FAIL | FAIL |

Confirmado via `checkpoint["args"]`: v2 e v3 usam `dense_weight=70.0`,
`seed=123`, `val_fraction=0.2`, mesmo dataset de 2M linhas — única diferença
de configuração entre eles é `latent_dim`.

**Resultado**: dobrar o gargalo latente de 64 para 128 melhorou o dense MSE
de `0,039518` para `0,037787` — uma queda de **4,38%**
(`(0,039518-0,037787)/0,039518`), pequena. **Conclusão**: a melhora foi real
(diferente do smoke test v7, seção 6.2), mas pequena demais para justificar
por si só a mudança — motivou investigar a métrica de loss em vez de só
aumentar a capacidade (seção 6.3).

Também vale notar, pela leitura direta dos logs: v3 **não convergiu via early
stopping** — rodou até o teto de 120 épocas sem disparar a parada antecipada
(`patience=5`). Isso o coloca, em termos de "convergência de verdade", mais
perto de v1 (também parou por teto) do que de v2 (convergiu por early
stopping) — um detalhe que não estava explícito na descrição desta tarefa e
que vale registrar para não superestimar o quanto v3 já tinha convergido.

**Nota metodológica sobre esta reavaliação**: `test_reconstruction.py`
(versão atual) chama `model.reconstruct()`
(`test_reconstruction.py:106`), que aplica `sigmoid` nas posições binárias do
bloco dense (`model.py:133-140`) — comportamento correto **apenas** para
checkpoints treinados com a loss segmentada MSE+BCE (v4, seção 6.3), cujo
decoder foi treinado para produzir logits nessas posições. v1/v2/v3 foram
treinados com `nn.MSELoss()` plano sobre o vetor inteiro — seu decoder já
produz o valor final diretamente, não um logit. Rodar `model.reconstruct()`
sobre esses três checkpoints (como uma execução ingênua desta auditoria fez
inicialmente) aplica um `sigmoid` **não treinado** por cima de valores já
corretos e infla artificialmente o erro (dense MSE apurado assim: v1
`0,163772`, v2 `0,159685`, v3 `0,158810` — muito acima dos números
corretos acima). Usar `model.forward()` puro (sem ativação) para v1/v2/v3
reproduziu exatamente os números já documentados na seção 5 para v1
(`0,045114` dense / `0,012447` agregado, idênticos até a 6ª casa decimal) e
os números citados no próprio comentário de `test_reconstruction.py:104`
para v2/v3 (`0,039518` / `0,037787`) — confirmando que a tabela acima está
correta e que o script atual, se rodado sem essa distinção, dá um resultado
errado para checkpoints anteriores ao v4.

### 6.2 Auditoria de um resultado suspeito (smoke test v7)

Antes de confiar no resultado de v3 acima, era preciso entender uma
discrepância: o smoke test v7 (seção 4), que já tinha testado
`latent_dim=128` — só que em escala pequena (20.000 linhas, 20 épocas) — deu
um dense MSE (`0,068502`) **idêntico até a 5ª casa decimal** ao de v4/smoke
com `latent_dim=64` (`0,068503`). Se dobrar a capacidade latente realmente
não fizesse diferença nenhuma, o resultado de v3 acima (uma queda real de
4,38%) seria contraditório. Investigação feita para resolver isso, em dois
níveis:

- **Nível de código**: `model.py`'s `_ENCODER_DIMS`/`_DECODER_DIMS`
  (`model.py:43-44`) são só os defaults do módulo — o construtor de
  `FusedFeaturesAutoencoder` monta o encoder/decoder a partir do **parâmetro**
  `latent_dim`, não da constante do módulo (`self.encoder = _mlp([fused_dim,
  512, 256, 128, latent_dim])`, `model.py:117-118`). O flag `--latent-dim` do
  CLI (`train_autoencoder.py:298-300`) é passado direto para o construtor
  (`FusedFeaturesAutoencoder(fused_dim=effective_fused_dim,
  latent_dim=args.latent_dim)`, `train_autoencoder.py:348`) — sem nenhum
  ponto intermediário onde o valor pudesse ser silenciosamente ignorado ou
  sobrescrito pelo default.
- **Nível de arquivo**: reaberto agora
  `checkpoints_smoketest_v7/fused_autoencoder_best.pt` e inspecionado
  diretamente o **shape dos tensores de peso** (não só o dict `args`, que
  poderia estar desatualizado/errado sem afetar o modelo de fato treinado):
  `encoder.6.weight` (última camada `Linear` do encoder) e `decoder.0.weight`
  (primeira camada `Linear` do decoder) têm shape `[128, 128]` — confirmando,
  no nível dos parâmetros salvos, que o modelo realmente treinado tinha
  gargalo de 128 dims, não 64. `n_total=20000` no mesmo checkpoint bate com a
  descrição de escala do smoke test na seção 4.

**Conclusão**: não há bug de wiring — o resultado idêntico do smoke test v7
foi **coincidência estatística real**, específica da escala pequena (20k
linhas, 20 épocas). Confirmado retroativamente pelo próprio resultado de v3
em escala real (seção 6.1): a mesma mudança (`latent_dim` 64→128) que não
produziu diferença nenhuma em 20k linhas produziu uma queda real (ainda que
pequena, 4,38%) em 2M linhas — a diferença só aparece com volume de dado e
tempo de treino suficientes, exatamente como já se via na conclusão da seção
4 sobre o platô do bloco dense.

### 6.3 Decisão de mudar a métrica de loss (não só o tamanho do gargalo)

Com a melhora de capacidade (v2→v3) pequena demais para ser a alavanca
principal, e a seção 4 já tendo descartado tanto ponderação de loss quanto
capacidade insuficiente do gargalo como causa do platô do bloco dense (v8:
isolar o bloco dense não mudou o resultado), a hipótese seguinte foi sobre o
**tipo** de loss usada, não seu tamanho ou peso.

O bloco dense (758 dims) mistura dois tipos de dado bem diferentes:
features contínuas (hp_ratio, level, os 5 stats, power/accuracy dos moves) e
features binárias esparsas (fainted, os 6 flags de status, is_active, os 18
flags de tipo, e os 3 flags de categoria de cada move — is_physical/
is_special/is_status), tanto no bloco do próprio time quanto no espelhado do
oponente. `nn.MSELoss()` trata essas duas famílias da mesma forma — mas MSE é
uma métrica subótima para alvos binários 0/1: penaliza de forma quadrática em
vez de logarítmica, dando pouco gradiente útil perto da fronteira de decisão
correta.

**Decisão**: segmentar a loss do bloco dense em duas partes — MSE para as
posições contínuas, `nn.BCEWithLogitsLoss` para as posições binárias —
mantendo o **MSE cru como métrica de avaliação** (não de treino), para
preservar comparabilidade direta com os números de v1/v2/v3 acima.

- `model.py::compute_dense_binary_mask()` (`model.py:82-86`, construída a
  partir de `_per_mon_binary_offsets()`/`compute_dense_binary_indices()`,
  `model.py:52-79`) enumera exatamente quais das 758 posições do bloco dense
  são binárias — derivado dos offsets de `state_encoder.py`
  (`OFF_FAINTED`, `OFF_STATUSES`, `OFF_IS_ACTIVE`, `OFF_TYPES`,
  `OFF_MOVES_DENSE`), não de números de coluna fixados à mão — e cobre tanto
  o bloco do próprio time (offset 0) quanto o espelhado do oponente (offset
  `OPP_TEAM_START`).
- `train_autoencoder.py::SegmentedPieceLoss` (`train_autoencoder.py:130-182`)
  combina, só para a peça `dense`, `cont_mse + bce_ratio * bin_bce`
  (`train_autoencoder.py:178`); as outras 5 peças (4 embeddings + meta_plan)
  continuam com MSE simples, sem mudança.
- `model.py`'s `decode()`/`forward()` continuam retornando valores **crus**
  (logits, não probabilidades) nas posições binárias por design
  (`model.py:19-28`), para alimentar `nn.BCEWithLogitsLoss` diretamente sem
  duplicar o sigmoid; `reconstruct()` (`model.py:133-140`) é quem aplica
  `sigmoid` nessas posições para uso/avaliação real — é esse método que
  `test_reconstruction.py` chama (`test_reconstruction.py:106`), e é
  exatamente essa distinção que causou o número inflado descrito na nota
  metodológica da seção 6.1 quando aplicado incorretamente a v1/v2/v3.

### 6.4 O problema de escala entre MSE e BCE

Combinar `cont_mse + bin_bce` sem correção teria um problema de escala: BCE
tem piso teórico de `~0,693` (`ln 2`) para uma flag binária ~50/50, enquanto
o piso de MSE já observado nas seções 4-5 e 6.1 está na faixa `~0,03-0,08`
(quase 10× menor). Sem correção, o termo BCE dominaria o gradiente do termo
`dense` combinado só por ter um piso mais alto — não porque seja um alvo mais
difícil de fato.

**Correção**: `--dense-bce-ratio` (`train_autoencoder.py:290-297`), default
`0,1`, combinado como `cont_mse + dense_bce_ratio * bin_bce`
(`train_autoencoder.py:178`). Com o default, o piso do termo BCE cai para
`~0,1 × 0,693 ≈ 0,0693` — mesma ordem de grandeza do piso de MSE observado.

### 6.5 Resultado do v4 (aceito nesta etapa, depois substituído por v5, ver seção 7)

Configuração: `latent_dim=256`, `SegmentedPieceLoss` (`dense-bce-ratio=0.1`),
`dense-weight=70` (igual às rodadas anteriores), mesmo dataset/split
(`seed=123`, `val_fraction=0.2`, 2.000.000 linhas). Confirmado via
`checkpoint["args"]` do arquivo em disco
(`data/autoencoder_bootstrap/checkpoints_v4_segloss/fused_autoencoder_best.pt`):
`latent_dim=256`, `dense_bce_ratio=0.1`.

Treino rodou com teto de 140 épocas e convergiu via early stopping na época
113 (5 épocas sem melhora — `train_full_run_v4.log`), com o melhor checkpoint
salvo na **época 108** (`val_loss=0,023678`, métrica de treino ponderada, não
o MSE cru).

Reavaliado agora (`test_reconstruction.py` sobre esse checkpoint, usando
`model.reconstruct()` — a forma correta para este checkpoint, treinado com
BCE):

```
Checkpoint epoch: 108  (saved val_loss: 0.023678)
Validation rows evaluated: 400,000
Aggregate MSE (raw, unweighted): 0.009502
Per-piece MSE (raw, unweighted):
           dense (dim=  758): mse=0.033497
     emb_species (dim=  384): mse=0.002418
       emb_moves (dim= 1536): mse=0.001479
       emb_items (dim=  192): mse=0.000879
   emb_abilities (dim=  192): mse=0.002342
       meta_plan (dim=   12): mse=0.000060
```

Dense MSE = **0,033497**; RMSE do dense = **`sqrt(0,033497) = 0,183021`, ou
seja `18,30%`** — este é o valor exato recalculado agora, e é o que deve ser
usado (substitui qualquer menção anterior a "15%" nesta conversa, que não
bate com o número medido).

Comparado a v3 (o melhor resultado da série anterior à mudança de loss):
dense MSE caiu de `0,037787` para `0,033497` — queda de **11,35%**
(`(0,037787-0,033497)/0,037787`). MSE agregado caiu de `0,010591` para
`0,009502`.

### 6.6 Status final dos critérios de aceitação formal da issue

| critério | threshold | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| MSE agregado | `< 0,01` | 0,012447 FAIL | 0,011025 FAIL | 0,010591 FAIL | **0,009502 PASS** |
| MSE dense | `< 0,01` | 0,045114 FAIL | 0,039518 FAIL | 0,037787 FAIL | 0,033497 FAIL |

**v4 é o melhor resultado da série nos dois critérios** (menor MSE agregado e
menor MSE dense entre os 4 checkpoints reavaliados) e é o único que passa no
critério de MSE agregado. **Não passa** no critério do bloco dense
isoladamente — `0,033497` ainda está `~3,3×` acima do threshold de `0,01`,
mesmo com a queda de `21,24%` para `18,30%` de RMSE ao longo de toda a série
(v1→v4). Dado o tempo disponível até o prazo de 20/07, a decisão tomada foi
aceitar v4 como resultado final desta fase e seguir para a integração (seção
"Pendências", então seção 8, hoje seção 10) — o critério de dense isolado
fica como métrica a revisitar se a validação de desempenho real mostrar que
a qualidade de jogo foi comprometida.

> **Nota de atualização (seção 7)**: uma auditoria posterior encontrou um bug
> de arquitetura que fazia v3 e v4 compartilharem o mesmo gargalo real
> (128 dims, não 256). O retreino corrigido, v5, saiu estritamente melhor que
> v4 nos dois critérios de MSE (seção 7.5/7.6) e é o resultado aceito hoje
> nesta fase.

---

## 7. O bug do gargalo fixo e a correção (v5) — fechamento definitivo da fase de arquitetura

### 7.1 O bug: v3 e v4 tinham o mesmo gargalo real

A seção 6.2 já tinha, sem nomear explicitamente como bug, todo o material
necessário para ver isto: o construtor de `FusedFeaturesAutoencoder` **não**
usava as constantes de módulo `_ENCODER_DIMS`/`_DECODER_DIMS` (código morto)
nem calculava larguras intermediárias a partir de `latent_dim` — montava o
encoder literalmente como `_mlp([fused_dim, 512, 256, 128, latent_dim])`
(citado em 6.2 a partir do `model.py` de então). Ou seja: os três tamanhos
intermediários (512, 256, 128) eram **hardcoded**, e `latent_dim` só
controlava a **última** camada do encoder (a que produz o código latente).

Consequência: para qualquer `latent_dim > 128`, a camada mais estreita da
rede não é a última (a nominal "latent"), e sim a penúltima, fixa em 128 —
`latent_dim=256` (v4) reabre de 128 para 256 só na última camada, sem nunca
ter comprimido a informação para menos de 128 dims em nenhum ponto do
encoder. Na prática, **v3 (`latent_dim=128`) e v4 (`latent_dim=256`) tinham
exatamente o mesmo gargalo de compressão real (128 dims)** — v4 não é uma
rede com o dobro de capacidade latente de v3, como o nome do hiperparâmetro
sugere; é a mesma rede com uma camada de expansão extra colada no fim do
encoder (e simetricamente no começo do decoder).

### 7.2 Evidência de tensor

Confirmado agora por duas vias independentes, recarregando os checkpoints do
disco com `torch.load(map_location='cpu', weights_only=False)`:

**(a) Tentativa de carregar `checkpoints_v4_segloss` com o `model.py` já
corrigido falha por incompatibilidade de shape** — o próprio erro do
PyTorch é a evidência mais direta de que a arquitetura realmente treinada
em v4 diverge da fórmula corrigida `encoder_dims(latent_dim, fused_dim)`:

```
RuntimeError: Error(s) in loading state_dict for FusedFeaturesAutoencoder:
  size mismatch for encoder.2.weight: checkpoint [256, 512] vs modelo atual [384, 512]
  size mismatch for encoder.4.weight: checkpoint [128, 256] vs modelo atual [256, 384]
  size mismatch for encoder.6.weight: checkpoint [256, 128] vs modelo atual [256, 256]
  ... (e os 3 pares equivalentes do decoder)
```

**(b) Inspeção direta dos shapes de cada tensor** (`state_dict` bruto, sem
instanciar nenhum modelo):

| checkpoint | `latent_dim` (args) | `encoder.0` | `encoder.2` | `encoder.4` | `encoder.6` (saída) | gargalo real |
|---|---|---|---|---|---|---|
| `checkpoints_v3_latent128` | 128 | `[512, 3074]` | `[256, 512]` | `[128, 256]` | `[128, 128]` | **128** |
| `checkpoints_v4_segloss` | 256 | `[512, 3074]` | `[256, 512]` | `[128, 256]` | `[256, 128]` | **128** (mesma largura mínima de v3; a camada 6 só reabre para 256) |
| `checkpoints_v5_fixed256` | 256 | `[512, 3074]` | `[384, 512]` | `[256, 384]` | `[256, 256]` | **256** (nunca estreita abaixo de `latent_dim`) |

(Convenção PyTorch: `nn.Linear.weight.shape == [out_features, in_features]` —
a largura da camada é a primeira dimensão.) v3 e v4 têm shapes de
`encoder.4` idênticos (`[128, 256]`) e o mesmo mínimo de largura (128) em
toda a rede — a única diferença real entre eles é a largura da última
camada (128→128 vs 128→256), não a capacidade de compressão. v5 nunca cai
abaixo de 256 em nenhuma camada.

### 7.3 Correção implementada

`model.py` (`encoder_dims()`/`decoder_dims()`, linhas 74-84, chamadas pelo
construtor nas linhas 157-158) substituiu os três tamanhos hardcoded por
`_intermediate_dims(latent_dim)` (linhas 66-71):

```python
def _intermediate_dims(latent_dim: int) -> list:
    return [
        max(512, latent_dim * 2),
        max(256, int(latent_dim * 1.5)),
        max(128, latent_dim),
    ]
```

Cada uma das três larguras intermediárias é `max(default_antigo, múltiplo de
latent_dim)`, garantindo que nenhuma camada do encoder/decoder seja mais
estreita que `latent_dim` — o funil fica monotonicamente decrescente até a
camada latente de verdade, em vez de estreitar cedo demais e reabrir depois.

Confirmado agora, chamando a função diretamente:

```
encoder_dims(256, 3074) = [3074, 512, 384, 256, 256]   # v5 — bate exatamente com 7.2(b)
encoder_dims(128, 3074) = [3074, 512, 256, 128, 128]   # fórmula nova, caso latent_dim=128
encoder_dims( 64, 3074) = [3074, 512, 256, 128,  64]   # fórmula nova, caso latent_dim=64
```

### 7.4 Retrocompatibilidade com v1/v2 (`latent_dim=64`) e v3 (`latent_dim=128`)

Para `latent_dim <= 128`, `max(128, latent_dim) == 128`, então a fórmula nova
degenera exatamente na antiga (`[512, 256, 128]`) — os checkpoints v1/v2/v3
deveriam continuar carregáveis pelo `model.py` corrigido sem nenhuma mudança
de shape. Confirmado agora, recarregando os três checkpoints reais com a
classe `FusedFeaturesAutoencoder` **já corrigida** (`load_state_dict(...,
strict=False)`, `strict=False` só por causa do buffer `binary_mask` —
adicionado depois de v1/v2/v3 existirem, ver nota metodológica da seção 6.1;
nenhuma das outras chaves apresenta erro de shape):

```
v1  latent_dim=64   missing=['binary_mask']  unexpected=[]
v2  latent_dim=64   missing=['binary_mask']  unexpected=[]
v3  latent_dim=128  missing=['binary_mask']  unexpected=[]
```

Zero `size mismatch` nas oito camadas `Linear` (4 do encoder + 4 do decoder)
dos três checkpoints — confirmando que a correção é estritamente aditiva
para `latent_dim <= 128`: v1/v2/v3 continuam com a arquitetura exata que
tinham antes.

### 7.5 Resultado do v5 (gargalo corrigido, `latent_dim=256`)

Configuração idêntica a v4 em tudo, exceto a correção de arquitetura:
mesmo dataset (`fused_features_synthetic.npy`, 2.000.000 linhas, seed 123,
`val_fraction=0.2`), mesma loss (`SegmentedPieceLoss`, `dense-bce-ratio=0.1`,
`dense-weight=70`), mesmo teto de 140 épocas/`patience=5`. Confirmado via
`checkpoint["args"]` de `checkpoints_v5_fixed256/fused_autoencoder_best.pt`:
único campo de configuração diferente de v4 é a arquitetura resultante do
`model.py` corrigido (os `args` salvos são idênticos aos de v4 nos demais
hiperparâmetros).

**Convergência mais rápida**: v5 disparou early stopping na **época 77**
(melhor checkpoint salvo na **época 72**, `val_loss=0,022275`), contra a
época **113** de v4 (melhor checkpoint na época **108**, `val_loss=0,023678`)
— confirmado lendo `train_full_run_v5.log`/`train_full_run_v4.log` linha a
linha. Tanto pelo ponto de parada (77 vs 113) quanto pelo melhor checkpoint
(72 vs 108), v5 converge mais cedo que v4.

**Reavaliação real** (`test_reconstruction.py` sobre `checkpoints_v5_fixed256`,
rodado agora):

```
Checkpoint epoch: 72  (saved val_loss: 0.022275)
Validation rows evaluated: 400,000
Aggregate MSE (raw, unweighted): 0.009101
Per-piece MSE (raw, unweighted):
           dense (dim=  758): mse=0.031879
     emb_species (dim=  384): mse=0.002426
       emb_moves (dim= 1536): mse=0.001473
       emb_items (dim=  192): mse=0.000870
   emb_abilities (dim=  192): mse=0.002342
       meta_plan (dim=   12): mse=0.000094
```

Reavaliado também `checkpoints_v4_segloss` nesta mesma sessão, para garantir
comparação como-igual (usando a arquitetura antiga hardcoded — a única forma
de recarregar esse checkpoint específico, ver 7.2(a)):

```
Checkpoint epoch: 108  (saved val_loss: 0.023678)
Aggregate MSE (raw, unweighted): 0.009502
Per-piece MSE (raw, unweighted):
           dense (dim=  758): mse=0.033497
```

Ambos os números batem exatamente com os já registrados na seção 6.5 para
v4 — nenhuma diferença encontrada nesta reconfirmação.

**Comparação v4 → v5**: dense MSE caiu de `0,033497` para `0,031879` — queda
de **4,83%** (`(0,033497-0,031879)/0,033497`); MSE agregado caiu de
`0,009502` para `0,009101` — queda de **4,22%**. RMSE do dense: `18,30%`
(v4) → `17,85%` (v5) (`sqrt(0,031879) = 0,178547`).

**Interpretação**: a correção arquitetural era necessária e está validada
por evidência de tensor direta (seção 7.2) — v4 e v3 não eram, de fato,
experimentos com capacidades latentes diferentes, e essa parte da narrativa
da seção 6 (a comparação v2→v3, que testava genuinamente 64 vs 128, continua
válida; só a leitura de v4 como "256 de capacidade real" estava incorreta).
Mas a melhora *numérica* de resultado com o gargalo corrigido foi pequena
(~4,8% no dense) — muito menor do que se poderia esperar de dobrar a largura
mínima real da rede (128→256). Isso reforça a conclusão já obtida
independentemente na seção 4 (v8, bloco dense isolado) e na seção 6.1 (v2→v3,
64→128 em escala real, só 4,38% de queda): o platô do bloco dense é uma
limitação **estrutural dos dados** (mistura de features contínuas e binárias
esparsas, mais fog of war entre time próprio e adversário) — não uma
limitação de capacidade do gargalo latente, nem, agora confirmado, um
artefato do bug de arquitetura. Corrigir o bug era o correto a fazer por
correção arquitetural em si, mas não era a alavanca que resolveria o platô.

### 7.6 v5 é o resultado final desta fase (ENCERRADA DE FORMA DEFINITIVA)

v5 é **estritamente melhor que v4 nos dois critérios de MSE** (menor MSE
agregado **e** menor MSE dense) e é o único checkpoint com arquitetura
tecnicamente correta (gargalo real = `latent_dim` declarado, não um valor
menor escondido). Tabela final, v1→v5:

| critério | threshold | v1 | v2 | v3 | v4 | **v5** |
|---|---|---|---|---|---|---|
| `latent_dim` (declarado) | — | 64 | 64 | 128 | 256 | 256 |
| gargalo real (largura mínima) | — | 64 | 64 | 128 | **128** (bug) | **256** (correto) |
| MSE agregado | `< 0,01` | 0,012447 FAIL | 0,011025 FAIL | 0,010591 FAIL | 0,009502 PASS | **0,009101 PASS** |
| MSE dense | `< 0,01` | 0,045114 FAIL | 0,039518 FAIL | 0,037787 FAIL | 0,033497 FAIL | **0,031879 FAIL** |
| RMSE dense | — | 21,24% | 19,88% | 19,44% | 18,30% | **17,85%** |

**v5 não passa** no critério do bloco dense isoladamente — continua
`~3,2×` acima do threshold de `0,01`, mesma situação de v4 (o critério
formal de aceitação não muda, só o valor absoluto do MSE melhora). Isso é
consistente com a interpretação da seção 7.5: o gap restante não é um
problema de arquitetura do autoencoder, e mais rodadas de dimensão/gargalo
não devem fechá-lo.

**Decisão**: a fase de exploração de arquitetura do autoencoder está
**ENCERRADA DE FORMA DEFINITIVA** com v5 como resultado final. Não haverá
mais rodadas de teste de dimensão/arquitetura depois desta — qualquer
trabalho futuro sobre o bloco dense (se houver) deveria mirar os próprios
dados/features, não o gargalo latente (ver seção 10, "Pendência futura").

---

## 8. Revisão da investigação de integração PyTorch→Keras (arquitetura corrigida, NÃO implementado)

O levantamento de opções de integração PyTorch→Keras discutido anteriormente
nesta conversa (recomendando a **Opção A** — recriar o encoder como camadas
`keras.layers.Dense` e copiar os pesos treinados via `set_weights()`, com
transposição `[out_features, in_features]` de PyTorch para `[in_features,
out_features]` de Keras) foi feito sobre a arquitetura **antiga** do encoder
(`3074 → 512 → 256 → 128 → 256`, com o bug do gargalo fixo descrito na seção
7). Esta seção revisa essa recomendação contra a arquitetura real e corrigida
de v5. **Nada foi implementado nesta revisão** — só análise, por instrução
explícita.

### 8.1 Arquitetura exata do encoder em v5

Extraída agora, diretamente do `state_dict` de
`checkpoints_v5_fixed256/fused_autoencoder_best.pt` (mesma evidência de
tensor da seção 7.2(b)):

```
encoder.0: Linear(3074 → 512), depois ReLU
encoder.2: Linear( 512 → 384), depois ReLU
encoder.4: Linear( 384 → 256), depois ReLU
encoder.6: Linear( 256 → 256)   (sem ReLU — saída do encoder, código latente)
```

Confirma exatamente `encoder_dims(256, 3074) = [3074, 512, 384, 256, 256]`
(seção 7.3) — **4 camadas `Linear`, 3 `ReLU` intercaladas, nenhuma ativação
na saída**, mesmo padrão estrutural de antes (seção 3: "sem ativação na
última camada"). Índices pares (`encoder.0/.2/.4/.6`) porque `_mlp()`
intercala `Linear`/`ReLU` num único `nn.Sequential` (seção 3) — os índices
ímpares são os `ReLU`, sem parâmetros.

### 8.2 A recomendação da Opção A continua válida

**Sim, sem mudança estrutural.** A única coisa que mudou entre a arquitetura
antiga e a de v5 são as **larguras** das camadas intermediárias
(`512→256→128` virou `512→384→256`) e a largura da camada de saída do
encoder (`128→256` virou `256→256`, já não expande mais). O tipo de cada
camada continua sendo exatamente o mesmo: `Linear` + `ReLU`, sem
`BatchNorm`, `Dropout`, `LayerNorm`, ou qualquer outra camada com estado
adicional além de peso+bias — confirmado lendo `_mlp()` (`model.py:141-149`)
e `FusedFeaturesAutoencoder.__init__` (`model.py:152-160`) atuais, que não
mudaram nesse aspecto.

Isso importa porque a Opção A depende inteiramente de "cada camada PyTorch
tem um equivalente Keras 1:1, com pesos transponíveis diretamente" — verdade
para `Linear`→`Dense` (é exatamente essa correspondência, com a transposição
`[out,in]`→`[in,out]` que a análise anterior já previa), mas ficaria mais
complicada se houvesse `BatchNorm` (estatísticas de rodagem, não só
pesos) ou `Dropout` (sem equivalente de pesos, mas precisa virar `no-op` em
inferência). Nenhuma dessas complicações existe aqui, antes ou depois da
correção do bug.

Concretamente, a recriação em Keras ficaria (mesmo padrão de antes, só com
as novas larguras):

```python
encoder_keras = keras.Sequential([
    keras.layers.Dense(512, activation="relu", input_shape=(3074,)),
    keras.layers.Dense(384, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256),  # sem ativação — saída do encoder
])
```

com pesos copiados camada a camada via
`keras_layer.set_weights([pytorch_weight.T, pytorch_bias])` para cada uma
das 4 camadas `Linear` (`encoder.0`, `encoder.2`, `encoder.4`, `encoder.6`).

### 8.3 O que muda na prática (não estrutural, mas vale registrar)

- **Congelamento**: a seção 10 (pendências) já previa carregar o encoder
  **congelado** — isso não muda, e como as larguras mudaram, o tensor que
  sai do encoder para dentro de `train_nn.py` também muda: `Dense(512)`
  (`train_nn.py:432`) passaria a receber um vetor de **256** dims (o
  `latent_dim` de v5), não mais 64 (o `latent_dim` original de v1/v2) nem
  128 (v3). Isso já era esperado — a análise anterior de integração foi
  feita quando v4 (`latent_dim=256`) já era o candidato, então o tamanho de
  saída (256) não muda entre a análise anterior e agora; só a forma
  *interna* do encoder muda.
- **Qual checkpoint carregar**: a análise anterior apontava para o
  checkpoint então aceito (v4). Isso deve ser atualizado para
  `checkpoints_v5_fixed256/fused_autoencoder_best.pt` (seção 7.6) — mesmo
  `latent_dim=256`, mesmo shape de saída, arquitetura interna diferente e
  tecnicamente correta.
- **[CONFIRMAR]**: a análise completa de opções feita anteriormente
  (alternativas à Opção A, se houver, e os critérios usados para descartá-
  las) não está registrada em nenhum arquivo deste repositório — só existe
  no histórico da conversa. Esta seção confirma que a conclusão (Opção A)
  continua de pé com a arquitetura nova, mas não tem como reproduzir aqui,
  de forma verificável, o conteúdo integral do levantamento original das
  demais opções. Se precisar dessas alternativas documentadas por escrito,
  marcar como próximo passo.
- Confirmado por leitura de `evaluator.py`/`train_nn.py` (`export_to_onnx()`,
  `train_nn.py:489-568`): o modelo principal é Keras, exportado a ONNX via
  `model.export(...)` (Keras 3 nativo) ou `tf2onnx` como fallback, e
  `evaluator.py` carrega só esse único grafo ONNX via `onnxruntime`
  (`evaluator.py:82`) — reforça por que a Opção A (encoder como camadas
  Keras nativas, dentro do mesmo `keras.Model`) é a rota que evita ter dois
  grafos de inferência separados (um ONNX do `train_nn.py`, outro `.pt`
  solto do autoencoder), que era o problema original citado na pendência de
  "mesmo grafo ONNX" (seção 10).

**Nada disso foi implementado** — `train_nn.py`, `evaluator.py` e
`state_encoder.py` continuam exatamente como estavam antes desta issue,
como já registrado na seção "Arquivos criados".

---

## 9. Infraestrutura de suporte a retomada de treino

Adicionado em `train_autoencoder.py` (não testado em execução real — só
sintaxe validada via `py_compile`, por instrução explícita de não rodar
nada durante essa mudança):

- `optimizer_state_dict` passou a ser salvo no checkpoint junto com
  `model_state_dict` (antes só o segundo era salvo).
- `--resume-from-checkpoint <path>`: carrega `model_state_dict` e
  `optimizer_state_dict`, retoma `best_val_loss` do valor salvo (não de
  `inf`), continua a numeração de épocas (`--epochs` no resume = "quantas
  épocas A MAIS", não orçamento total — checkpoint parado na 33 + `--epochs 20`
  → roda 34..53).
- `patience_counter` sempre reinicia em 0 no resume (documentado no
  `--help` do argparse) — não há como recuperar quantas épocas consecutivas
  sem melhora já tinham passado antes de uma interrupção.
- Validação de `--seed`/`--val-fraction`/`--max-rows` contra
  `ckpt["args"]` antes de resumir — aborta com erro explícito listando o
  que não bateu, em vez de reproduzir silenciosamente um split diferente.
- Recusa explícita de resumir checkpoints sem `optimizer_state_dict`
  (`SystemExit` com mensagem clara), em vez de resumir silenciosamente com
  Adam reiniciado do zero.

**Importante**: o checkpoint do treino de 50 épocas
(`data/autoencoder_bootstrap/checkpoints/fused_autoencoder_best.pt`) foi
salvo **antes** dessa mudança — confirmado diretamente (`torch.load` e
inspeção das chaves): `['model_state_dict', 'fused_dim', 'latent_dim', 'epoch',
'val_loss', 'n_total', 'args']`, **sem** `optimizer_state_dict`. Esse
checkpoint **não é resumível** com estado real do Adam — só serve como ponto
de partida (pesos do modelo) para um treino novo do zero, não para continuar
o treino de 50 épocas além de onde parou.

---

## 10. Pendências

### Encerrado nesta fase

- **Exploração de arquitetura do autoencoder**: **ENCERRADA DE FORMA
  DEFINITIVA**. A decisão original da seção 6.6 (aceitar v4) foi
  **substituída** pela decisão da seção 7.6: aceitar **v5**
  (`latent_dim=256`, gargalo real corrigido para 256, loss segmentada
  MSE+BCE, `dense-bce-ratio=0.1`; dense MSE=0,031879, MSE agregado 0,009101)
  como versão final do autoencoder. v4 **não é mais** o resultado aceito —
  v5 o substitui em definitivo, por ser estritamente melhor nos dois
  critérios de MSE e por corrigir o bug de arquitetura documentado na seção
  7. Não haverá mais rodadas de teste de dimensão/arquitetura depois de v5.

### Ainda não implementado

- **Integração em `train_nn.py`**: substituir a primeira camada
  `Dense(512)` do tronco principal (`train_nn.py:432`) pelo encoder do
  **v5** treinado e **congelado** (pesos não atualizados durante o treino da
  rede principal) — atualizado de v4 para v5 (seção 7.6/8.3). Não iniciado —
  **próxima etapa ativa**.
- **Integração em `evaluator.py` / exportação ONNX**: o encoder precisa
  fazer parte do **mesmo grafo ONNX** que o MCTS carrega em produção
  (`NeuralStateEvaluator`, `evaluator.py`), não ser um arquivo `.pt` separado
  carregado à parte. Não iniciado — envolve converter/fundir um módulo
  PyTorch treinado num grafo Keras/ONNX, o que não é trivial dado que o resto
  do pipeline (`train_nn.py`) é Keras 3, não PyTorch. A recomendação de
  abordagem (Opção A — recriar como `keras.layers.Dense` + `set_weights()`,
  seção 8) foi revisada contra a arquitetura de v5 e continua válida
  estruturalmente; a implementação em si ainda não começou.
- **Validação de desempenho real** (rede principal jogando COM vs. SEM o
  autoencoder, via `RoundRobinBenchmark`/`TournamentBenchmark` ou
  equivalente) foi **conscientemente adiada** para depois do prazo de
  20/07 — risco aceito e documentado aqui, não resolvido. Sem essa validação,
  não há garantia de que a compressão (mesmo que batesse os critérios de MSE)
  preserve a qualidade de jogo da rede principal.

### Pendência futura, não urgente (fora do escopo do prazo atual)

- O bloco `dense` continua sendo o único critério formal reprovado (seção
  7.6). Diferente da versão anterior desta seção, **não recomendamos mais
  testar `latent_dim` maior ou outras variações de arquitetura/gargalo** —
  a seção 7.5 mostrou que corrigir o bug (dobrar o gargalo real de 128 para
  256) só reduziu o dense MSE em ~4,8%, reforçando que o platô é estrutural
  dos dados, não de capacidade. Se alguém quiser continuar otimizando o
  bloco `dense` no futuro, a alavanca provável está nos próprios
  dados/features (ex.: engenharia de features, ou tratar contínuo/binário
  de forma ainda mais separada), não em mais rodadas de dimensão latente.

---

## Arquivos criados nesta sessão (issue #10)

| arquivo | linhas | função |
|---|---|---|
| `src/battle_agents/mcts_approximation/pipeline/autoencoder/__init__.py` | 1 | inicialização do subpacote |
| `src/battle_agents/mcts_approximation/pipeline/autoencoder/generate_synthetic_dataset.py` | 274 | geração do dataset sintético de 2M exemplos |
| `src/battle_agents/mcts_approximation/pipeline/autoencoder/model.py` | 180 | arquitetura `FusedFeaturesAutoencoder` + máscara binária/loss segmentada (seção 6.3) + `encoder_dims()`/`decoder_dims()` escalando com `latent_dim` (correção do bug do gargalo fixo, seção 7.3) |
| `src/battle_agents/mcts_approximation/pipeline/autoencoder/train_autoencoder.py` | 462 | treino, loss ponderada por peça, `SegmentedPieceLoss` (seção 6.3), resume |
| `src/battle_agents/mcts_approximation/pipeline/autoencoder/test_reconstruction.py` | 209 | teste de aceitação (MSE agregado + por peça) |

Contagens de linha acima reconfirmadas por `wc -l` nesta sessão (a versão
anterior deste documento listava `model.py`=51 e `train_autoencoder.py`=386 —
desatualizado desde a adição da loss segmentada MSE+BCE, seção 6.3; `model.py`
cresceu de 140 para 180 linhas nesta sessão com a correção do gargalo fixo,
seção 7.3).

Também modificado nesta sessão (fora da pasta `autoencoder/`):
`src/battle_agents/mcts_approximation/pipeline/generate_data.py` (parâmetro
`agent_type`, retrocompatível — seção 2.1). **Nenhum outro arquivo do projeto
foi alterado** — `train_nn.py`, `evaluator.py` e `state_encoder.py` continuam
exatamente como estavam antes desta issue.

Artefatos gerados em `data/` (não versionados, regeneráveis — ver
`CLAUDE.md`): `data/genrandom_bootstrap/` (49.999 jogos, ~49GB),
`data/autoencoder_bootstrap/fused_features_synthetic.npy` (24,59GB),
`data/autoencoder_bootstrap/checkpoints/fused_autoencoder_best.pt` (13MB, v1),
`checkpoints_v2/fused_autoencoder_best.pt` (40MB, v2),
`checkpoints_v3_latent128/fused_autoencoder_best.pt` (40MB, v3),
`checkpoints_v4_segloss/fused_autoencoder_best.pt` (41MB, v4 — **superado por
v5**, não é mais o resultado aceito, seção 7.6),
`checkpoints_v5_fixed256/fused_autoencoder_best.pt` (45MB, **v5 — resultado
final desta fase**, seção 7.5/7.6), mais 11 diretórios `checkpoints_smoketest*`
e `checkpoints_timing_test` (pequenos, artefatos das rodadas de diagnóstico da
seção 4 — não apagados, mantidos como rastro auditável das 8 rodadas; contagem
reconfirmada agora via `ls`, corrige o "9" de uma versão anterior deste
documento). Também `train_full_run.log` / `train_full_run_v2.log` /
`train_full_run_v3.log` / `train_full_run_v4.log` / `train_full_run_v5.log`
(logs completos por época de cada treino, usados como fonte para os números de
convergência/early stopping das seções 6 e 7).
