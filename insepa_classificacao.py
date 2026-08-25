#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 2: Classificação.

Objetivo desta camada: CLASSIFICAR o que o texto já carrega -- não inventar
nada. Quatro regras:

1. Todo texto tem uma emoção. Mesmo a ausência de emoção é uma emoção: frieza.
2. Toda emoção pode ser detectada explícita (emoji/reação já presente) ou
   oculta (sem emoji, mas uma VAR de palavra no texto revela que ela está lá).
3. Toda frase carrega um contexto.
4. Texto + emoção + contexto, juntos, geram um pensamento: uma breve
   explicação da situação.

As vars têm duas funções diferentes: no texto, são variações de palavras que
se encaixam no contexto/bloco; nos emojis, funcionam como SINALIZADORES --
quando não há emoji, a var da palavra descreve a emoção que está ali.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Léxico de exemplo: emoji -> vars (palavras/expressões que sinalizam essa
# emoção quando o emoji não está explícito, seja no campo de reação escrito
# por extenso, seja escondida dentro do próprio texto).
LEXICO_EMOCOES: Dict[str, List[str]] = {
    ":D": ["sorriso", "sorriso largo", "alegria explícita"],
    "^^": ["expressão amigável", "sorriso gentil", "sorriso suave", "carinho"],
    ":)": ["sorriso fechado", "leve sorriso"],
    "🖤": ["frieza", "indiferença", "distanciamento"],
}

FRIEZA = "frieza"  # emoção padrão quando nenhuma outra é detectada (regra 1)


def classificar_emocao(texto: str, reacao: str = "") -> dict:
    """Classifica a emoção de uma entrada.

    A var é, por definição, uma DESCRIÇÃO da emoção -- então tanto faz se ela
    aparece no campo de reação (escrita por extenso, ex.: "sorriso fechado")
    ou escondida dentro do próprio texto: as duas contam como a mesma coisa,
    e nenhuma delas é aceita sem bater com um sinal conhecido.

    - explícita: o campo de reação já é literalmente um emoji conhecido.
    - oculta: uma var conhecida aparece na reação (escrita por extenso) ou no texto.
    - frieza: nada bate -- regra 1, ausência também é uma emoção.
    """
    reacao_norm = (reacao or "").strip()

    # 1) A reação já é, literalmente, um emoji que conhecemos.
    if reacao_norm in LEXICO_EMOCOES:
        return {"tipo": "explícita", "emocao": reacao_norm, "var_detectada": None, "origem": "reação"}

    # 2) Uma var conhecida aparece na reação (escrita por extenso) ou no texto.
    # A var mais específica (mais longa) vence -- "sorriso fechado" deve
    # bater antes de "sorriso" sozinho, mesmo que os dois sejam válidos.
    reacao_lower = reacao_norm.lower()
    texto_lower = (texto or "").lower()
    candidatos = []
    for emoji, vars_lista in LEXICO_EMOCOES.items():
        for var in vars_lista:
            var_lower = var.lower()
            if var_lower in reacao_lower:
                candidatos.append((len(var), emoji, var, "reação"))
            if var_lower in texto_lower:
                candidatos.append((len(var), emoji, var, "texto"))

    if candidatos:
        _, emoji, var, origem = max(candidatos, key=lambda c: c[0])
        return {"tipo": "oculta", "emocao": emoji, "var_detectada": var, "origem": origem}

    return {"tipo": "ausente", "emocao": FRIEZA, "var_detectada": None, "origem": None}


def classificar_contexto(contexto_declarado: Optional[str]) -> dict:
    """Regra 3: toda frase carrega um contexto -- se nenhum foi declarado
    explicitamente, ainda assim classificamos que ele existe (só não foi
    nomeado), nunca que ele está ausente."""
    if contexto_declarado:
        return {"tipo": "declarado", "contexto": contexto_declarado}
    return {"tipo": "presente, não nomeado", "contexto": None}
