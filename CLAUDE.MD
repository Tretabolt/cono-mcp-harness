# CLAUDE.MD — Especificação Viva: CONO MCP Harness (MNHI 3.0)

## 1. Axioma de Engenharia (Anti-Vibe Coding)
- Disciplina > Intuição: nenhuma hipótese causal é aceita sem passar por asserções TDD determinísticas.
- Separação Estrita Identidade vs. Estado (Camada 0):
  - `Hash_UIF = SHA256(serial + tipo_agente + funcao_s)` -> nunca inclui posição, pesos mutáveis ou humor.
  - `Hash_Rastro = SHA256(hash_origem + hash_destino + tipo_relacao)` -> nunca inclui peso causal mutável.
- AI-Jail: o agente não lê a memória de estado `S` do `world_engine`. Toda interação ocorre exclusivamente via protocolo MCP (`stdio` JSON-RPC) dentro de container isolado (`network_mode: none`).

## 2. Stack Tecnológica
- Core / Simulador / MCP / Memória / Testes: Python 3.11+ (stdlib pura para zero atrito de build; `pytest` opcional para runner).
- Visualização 3D posterior: HTML + Three.js (`web_3d_client/index.html`) lendo snapshots públicos.

## 3. Definições Matemáticas Travadas
1. **Distância Hiperbólica (`d_H`) no Disco de Poincaré (`||u|| < 1`):**
   `d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))`
2. **ISC Dinâmico (Trajetória no tempo):**
   `ISC_din = 1 / (1 + H_fase + H_espectral)`
   - `H_fase`: variância normalizada `var(d_H) / mean(d_H)^2` entre nós/estados projetados no disco.
   - `H_espectral`: Entropia de Shannon (base 2) das magnitudes `ΔM` (passo espacial ou `Δr`/`Δw` na rede de pulsos).
3. **ISC Topológico (Snapshot do Grafo Causal):**
   `ISC_top = Arestas_Validas / Arestas_Maximas`
   - Regra de Emergência de Símbolo ("EU Local"): `ISC_top >= 0.85` E `Var(λ_norm) <= 0.25`.
4. **Pentacosagram (Auditor Imutável — Read/Verify/Log):**
   - Cadeia sequencial: `Hash(t) = SHA256(Hash(t-1) + merkle_root + acao + status + d_hyper)`.
   - Detecção de Alucinação: `D_Hyper > τ` (onde `τ` é calibrado por domínio; padrão inicial `τ = 1.0`).
5. **Heptagram (Atuador Ético / Rules Engine — Enforce):**
   - Veto via Divergência de Jensen-Shannon (base 2, em `[0, 1]`) contra `P_tirania`:
     `VETO` se `JSD(P_nova || P_tirania) < JSD(P_atual || P_tirania)` OU `JSD(P_nova || P_tirania) < JSD_min` (padrão `0.15` bits para `N=4`).
   - Penalidade por falha auditada: `w_confianca *= 0.5` (quarentena se `w < 0.05`; recuperação `+10%` por acerto TDD até `1.0`).

## 4. Contrato MCP (POMDP)
- Ferramentas permitidas: `move(direcao)`, `rotate(graus)`, `inspect()`, `interact()`, `raise_arm()`, `drop()`, `pulse_decide(estimulo)`.
- Affordance heurística: `contorno_ciano = true` apenas quando objeto interativo está imediatamente à frente e visível.

## 5. Regra de Ouro de Evolução
O `Distiller` só grava uma nova heurística no `CausalGraph` persistente se:
1. O episódio passar no verificador TDD do nível atual;
2. O `Pentacosagram` validar a cadeia sem adulteração;
3. A suíte de regressão dos níveis anteriores continuar verde.

## 6. Integração Oficial com MNHI 3.0 (`mnhi3_bridge.py`)
- O `cono-mcp-harness` consome o pacote `mnhi` (`mnhi-3.0/src`) como motor oficial da Camada 0, Extrapolation, Heptagram (PCE) e Pentacosagram.
- **Regra de Saliência vs. Verdade Causal (Nível 4):**
  - `clue_weight` observado via MCP é tratado apenas como *saliência sensorial bruta*.
  - Antes de agir sobre uma pista, o agente submete cada candidata ao `MNHI3Engine.simulate_counterfactual()` e ao `Heptagram.evaluate()`.
  - Pistas com relação não verificada / atalho chamativo (`ATALHO` / `DECOY`) têm PCE travado em `<= 0.20` (`compliant=False`) e são rejeitadas mesmo tendo saliência `0.9`.
- **Trava Cruzada de Integridade:**
  - Ao fim de cada episódio, o `MNHI3Bridge` executa `store.run_integrity_audit()` e `pentacosagram.verify_chain()`. Qualquer adulteração anula a aprovação do episódio e bloqueia o `Distiller`.
- **Ponte opcional:** o import da ponte é preguiçoso — sem o repositório irmão `mnhi-3.0/` ao lado, o harness continua funcionando standalone e os testes cruzados (`test_05`) são pulados.
