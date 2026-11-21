# cono-mcp-harness — harness MCP / jogo MNHI 3.0

Harness de testes do MNHI 3.0 como jogo MCP: um agente resolve
cenários (Levels 1–4) através de um servidor MCP, com o ciclo
load → execute → verify → distill. Python stdlib-only, como os
outros dois repos.

## O que tem

- `packages/mnhi_memory/` — memória MNHI: hasher, métricas
  (`causal_entropy_bits`, `can_emerge_meta_symbol`), Heptagram,
  Pentacosagram; ponte para o engine standalone em
  `packages/mnhi_memory/mnhi3_bridge.py` (import preguiçoso — o
  harness funciona sozinho, sem o repo vizinho).
- `packages/agent_orchestrator/` — orquestrador dos Levels.
- `scenarios/` — os testes TDD, `test_01` a `test_05`.
- `web_3d_client/` — cliente 3D.

## Como rodar a suíte

```bash
cd ~/workspace/cono-mcp-harness
python3 -m unittest discover -s scenarios
```

Tem que dar 17/17 OK. Rodar a partir de outro diretório falha com
erros de import — o comando acima precisa ser executado de dentro do
repo (ou com `-t cono-mcp-harness`).

## H_causal

`packages/mnhi_memory/metrics.py` tem `causal_entropy_bits` (entropia
de Shannon, base 2, sobre os ISCs normalizados dos shards) e
`can_emerge_meta_symbol` (portão: ISC_Global ≥ 0,80 E H_causal ≥
log2(N)/2). O portão é um PISO, não um teto — a versão invertida
(H ≤ 0,1) estava errada e foi corrigida em 17/11. Teste TDD em
`scenarios/test_01_mnhi_core.py`. Suíte 17/17, HEAD no GitHub
`18f2e73e` (Tretabolt/cono-mcp-harness).

Nota (20/11): o `CLAUDE.MD` deste repo não documenta o comando de
testes acima nem menciona a H_causal — esta README existe para cobrir
isso. As definições de ISC/Var aqui ainda são as antigas (Var(λ) ≤
0,25, ISC dinâmico 1/(1+H_fase+H_espectral) e topológico): o gym já
evoluiu para VarPn ≤ 0,36 e o harness não acompanhou — unificar é um
passo à parte, maior, não feito aqui.
