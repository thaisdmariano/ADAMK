#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Motor de Autonomia.

Substitui por completo o antigo "modo autônomo" (destruído em 2026-09-12
por instrução direta da criadora): aquele rodava em cima de
calcular_similaridade/corpus_similarity_score -- um score contínuo (média
ponderada de frações tipo Jaccard, corte fixo em 0.70) -- misturando texto
e estrutura na mesma conta.

Terceira versão deste desenho (2026-09-12), a que fica -- a Thaís pediu
pra levar a sério: cosseno E Báskara voltam os dois, cada um só onde faz
sentido, nunca misturados de novo --

- **texto ↔ texto**: cosseno de verdade (vetor de frequência de tokens).
  "Esse texto é parecido com aquele" é uma pergunta legitimamente contínua
  -- cosseno rankeia/ENCONTRA o bloco de texto mais parecido, mas nunca
  decide identidade sozinho.
- **emoção + contexto + pensamento ↔ o bloco encontrado**: Báskara
  CONFIRMA -- ou bate ou não bate, sem meio-termo. Conta quantos eixos
  batem (b) e quantos divergem (c) de verdade (Camada 6 -- comparar_eixo);
  "vazio"/"desconhecido" não contam pra nenhum dos dois. Δ=b²-4ac≥0
  confirma a correspondência estrutural; Δ<0 recusa, mesmo com o texto
  parecido.

Ou seja: cosseno ACHA candidato, Báskara CONFIRMA -- uma etapa nunca
substitui a outra.

Regra inegociável (Camada 5), reforçada em código, não só em comentário:
este motor NUNCA cria um bloco de dado concreto sozinho -- contexto e
pensamento_interno são sempre e só da criadora. O máximo que ele faz é
sinalizar uma correspondência encontrada (ou não); quem chama decide o que
fazer com isso (ex.: registrar como opinião pendente).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from insepa_espelhamento import comparar_eixo


def _tokenizar_simples(texto: str) -> List[str]:
    """Tokenizador local, sem depender de lovely_test.py -- palavra ou
    pontuação, mesmo espírito do resto do INSEPA."""
    return re.findall(r"\w+|[^\w\s]", texto or "", re.UNICODE)


def similaridade_texto_cosseno(texto_a: str, texto_b: str) -> float:
    """Cosseno de verdade -- só pra texto, só pra responder "esse texto é
    parecido com aquele" como um número contínuo e honesto. Vetor de
    frequência de tokens (Counter); cosseno é o ângulo entre os dois.
    Nunca decide identidade sozinho -- serve só pra ACHAR/rankear o
    candidato mais parecido; a confirmação de verdade (emoção/contexto/
    pensamento) sempre passa pelo Báskara depois."""
    tokens_a = Counter(t.lower() for t in _tokenizar_simples(texto_a))
    tokens_b = Counter(t.lower() for t in _tokenizar_simples(texto_b))
    vocabulario = set(tokens_a) | set(tokens_b)
    if not vocabulario:
        return 0.0
    produto_escalar = sum(tokens_a.get(t, 0) * tokens_b.get(t, 0) for t in vocabulario)
    norma_a = sum(v * v for v in tokens_a.values()) ** 0.5
    norma_b = sum(v * v for v in tokens_b.values()) ** 0.5
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return produto_escalar / (norma_a * norma_b)


LIMIAR_COSSENO_TEXTO = 0.70  # piso mínimo pra cosseno considerar "achou candidato" -- sem isso, dois textos que só compartilham pontuação (tipo só o "?") contavam como candidato, um sinal fraco demais pra sequer chegar no Báskara.


def encontrar_texto_mais_parecido(texto_candidato: str, blocos: List[dict]) -> Optional[dict]:
    """Etapa 1 -- cosseno ACHA o bloco cujo texto (canônico ou Multivar) é
    mais parecido com o candidato. Puro ranking; não confirma nada ainda --
    mas só conta como "achado" acima de LIMIAR_COSSENO_TEXTO (70%), senão
    pontuação sozinha (sem nenhuma palavra em comum) já contava como candidato."""
    melhor_bloco, melhor_score = None, 0.0
    for b in blocos:
        entrada = b.get("entrada", {})
        variantes = [entrada.get("texto", "")] + list(entrada.get("Multivars_Texto_Entrada", []))
        score = max((similaridade_texto_cosseno(texto_candidato, v) for v in variantes if v), default=0.0)
        if score > melhor_score:
            melhor_bloco, melhor_score = b, score
    if melhor_bloco is None or melhor_score < LIMIAR_COSSENO_TEXTO:
        return None
    return {"bloco": melhor_bloco, "score_texto": melhor_score}


def _eixo_emocao(reacao_candidato: str, reacoes_aceitas_bloco: Optional[List[str]]) -> str:
    """Mesmo vocabulário de comparar_eixo (bate/diverge/vazio/desconhecido),
    usando o conjunto de reações aceitas do bloco (canônica + vars/
    multivars já registradas -- Camada 5), não igualdade de string única."""
    aceitas = {r.strip() for r in (reacoes_aceitas_bloco or []) if r}
    reac = (reacao_candidato or "").strip()
    if not reac and not aceitas:
        return "vazio"
    if not reac or not aceitas:
        return "desconhecido"
    return "bate" if reac in aceitas else "diverge"


def confirmar_estrutura_baskara(
    reacao_candidato: str,
    reacoes_aceitas_bloco: Optional[List[str]],
    contexto_candidato: str,
    contexto_bloco: str,
    pensamento_candidato: str,
    pensamento_bloco: str,
) -> Dict[str, object]:
    """Etapa 2 -- Báskara CONFIRMA se emoção+contexto+pensamento
    correspondem ao bloco que o cosseno achou. Conta quantos dos 3 eixos
    batem de verdade (b) e quantos divergem de verdade (c) -- "vazio"/
    "desconhecido" não contam pra nenhum dos dois lados (normal contexto/
    pensamento não existirem pro candidato -- só a criadora define,
    Camada 5). a=1 fixo. Δ=b²-4ac≥0 confirma; Δ<0 recusa, mesmo com o
    texto parecido."""
    eixo_emocao = _eixo_emocao(reacao_candidato, reacoes_aceitas_bloco)
    eixo_contexto = comparar_eixo(contexto_candidato, contexto_bloco)
    eixo_pensamento = comparar_eixo(pensamento_candidato, pensamento_bloco)

    b = sum(1 for eixo in (eixo_emocao, eixo_contexto, eixo_pensamento) if eixo == "bate")
    c = sum(1 for eixo in (eixo_emocao, eixo_contexto, eixo_pensamento) if eixo == "diverge")
    a = 1
    discriminante = b * b - 4 * a * c
    confirma = discriminante >= 0
    raizes = None
    if confirma and (b or c):
        raiz_delta = discriminante ** 0.5
        raizes = ((-b + raiz_delta) / (2 * a), (-b - raiz_delta) / (2 * a))

    return {
        "confirma": confirma,
        "discriminante": discriminante,
        "raizes": raizes,
        "b": b, "c": c,
        "eixo_emocao": eixo_emocao,
        "eixo_contexto": eixo_contexto,
        "eixo_pensamento": eixo_pensamento,
        "raiz_compartilhada": eixo_pensamento == "bate",
    }


def encontrar_melhor_correspondencia(
    texto_candidato: str,
    reacao_candidato: str,
    blocos: List[dict],
    reacoes_aceitas_por_bloco: Dict[object, List[str]],
    contexto_candidato: str = "",
    pensamento_candidato: str = "",
) -> Optional[dict]:
    """Pipeline completo: cosseno acha o bloco de texto mais parecido;
    Báskara confirma se a estrutura (emoção/contexto/pensamento) realmente
    corresponde. Só devolve algo se as DUAS etapas passarem -- texto
    parecido sozinho nunca basta."""
    achado_texto = encontrar_texto_mais_parecido(texto_candidato, blocos)
    if achado_texto is None:
        return None
    bloco = achado_texto["bloco"]
    entrada = bloco.get("entrada", {})
    confirmacao = confirmar_estrutura_baskara(
        reacao_candidato, reacoes_aceitas_por_bloco.get(bloco.get("bloco_id"), []),
        contexto_candidato, entrada.get("contexto", ""),
        pensamento_candidato, entrada.get("pensamento_interno", ""),
    )
    if not confirmacao["confirma"]:
        return None
    return {"bloco": bloco, "score_texto": achado_texto["score_texto"], "confirmacao": confirmacao}


def avaliar_entrada_autonoma(
    texto_candidato: str,
    reacao_candidato: str,
    blocos: List[dict],
    reacoes_aceitas_por_bloco: Dict[object, List[str]],
    contexto_candidato: str = "",
    pensamento_candidato: str = "",
) -> Dict[str, object]:
    """Ponto de entrada do motor: cosseno acha, Báskara confirma. NUNCA
    cria um bloco de dado concreto sozinho -- contexto e pensamento_interno
    são sempre e só da criadora. O máximo que devolve é um veredito; quem
    chama decide o que fazer (ex.: registrar como opinião pendente)."""
    if not texto_candidato and not reacao_candidato:
        return {"action": "vazio"}
    correspondencia = encontrar_melhor_correspondencia(
        texto_candidato, reacao_candidato, blocos, reacoes_aceitas_por_bloco,
        contexto_candidato, pensamento_candidato,
    )
    if correspondencia is None:
        return {"action": "sem_correspondencia"}
    conf = correspondencia["confirmacao"]
    return {
        "action": "correspondencia_encontrada",
        "bloco_id": correspondencia["bloco"]["bloco_id"],
        "score_texto": correspondencia["score_texto"],
        "discriminante": conf["discriminante"],
        "raizes": conf["raizes"],
        "eixo_emocao": conf["eixo_emocao"],
        "eixo_contexto": conf["eixo_contexto"],
        "eixo_pensamento": conf["eixo_pensamento"],
        "raiz_compartilhada": conf["raiz_compartilhada"],
    }
