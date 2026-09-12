#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 6: Espelhamento e Análise.

Quando uma entrada nova chega, o Adam não devolve um único número de
confiança -- ele espelha a entrada contra um bloco de referência, EIXO POR
EIXO (emoção, texto, contexto, pensamento), sempre por igualdade EXATA de
tokens, nunca por distância/aproximação. Cada combinação de bate/diverge tem
um significado próprio:

- texto bate mas emoção diverge: mesma frase, carga emocional diferente --
  sinal de tom/ironia/sarcasmo (pela própria Camada 4, isso já não é a mesma
  chave, então nem deveria ter sido tratado como a mesma entrada).
- contexto diverge: mesmo que o resto bata, é um assunto genuinamente
  diferente (contexto não tem var/multivar -- não existe forma equivalente).
- pensamento bate (mesma raiz) mas a entrada traz conteúdo que a criadora
  ainda não confirmou: o espelhamento nunca pula a revisão dela sozinho --
  bater na raiz é necessário, mas nunca suficiente, pra virar dado concreto.

O motor "aprende com a interação" não porque calcula uma aproximação melhor,
mas porque toda vez que a criadora resolve um diagnóstico, isso vira registro
exato novo (var, emoção ligada, ou bloco de verdade) -- o sistema cresce por
reforço dela, nunca por inferência sozinha.
"""

from __future__ import annotations

from typing import Dict

from insepa_marcadores import tokenizar


def comparar_eixo(a: str, b: str) -> str:
    """Compara dois textos por igualdade EXATA do conjunto de tokens -- nunca
    por distância/aproximação. 'vazio' quando os dois lados não têm nada pra
    comparar; 'desconhecido' quando só um dos lados tem dado (ex.: uma
    entrada ao vivo ainda sem contexto/pensamento inferido)."""
    if not a and not b:
        return "vazio"
    if not a or not b:
        return "desconhecido"
    return "bate" if set(tokenizar(a)) == set(tokenizar(b)) else "diverge"


def espelhar(entrada_nova: Dict[str, str], bloco_referencia: Dict[str, str]) -> Dict[str, object]:
    """Espelha uma entrada nova contra um bloco de referência, eixo por eixo.

    `entrada_nova`: {"texto", "emocao", "contexto", "pensamento"} (contexto e
    pensamento podem vir vazios -- ainda não inferidos).
    `bloco_referencia`: {"texto", "reacao", "contexto", "pensamento_interno"}.
    """
    eixo_emocao = comparar_eixo(entrada_nova.get("emocao", ""), bloco_referencia.get("reacao", ""))
    eixo_texto = comparar_eixo(entrada_nova.get("texto", ""), bloco_referencia.get("texto", ""))
    eixo_contexto = comparar_eixo(entrada_nova.get("contexto", ""), bloco_referencia.get("contexto", ""))
    eixo_pensamento = comparar_eixo(entrada_nova.get("pensamento", ""), bloco_referencia.get("pensamento_interno", ""))

    return {
        "emocao": eixo_emocao,
        "texto": eixo_texto,
        "contexto": eixo_contexto,
        "pensamento": eixo_pensamento,
        "sinal_tom_diferente": eixo_texto == "bate" and eixo_emocao == "diverge",
        "raiz_compartilhada": eixo_pensamento == "bate",
        "precisa_da_criadora": True,  # Camada 5: bater nos eixos nunca é suficiente sozinho
    }
