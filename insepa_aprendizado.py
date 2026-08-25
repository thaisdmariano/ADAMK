#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 5: Aprendizado Seguro.

Como o Adam lida com dado NOVO sem adivinhar:

1. O ALNULU só deixa o Adam VER a forma de uma palavra (transformá-la em
   número). Ele não serve pra aproximar por distância/similaridade -- isso
   forçaria uma comparação que a IA comum faz (embeddings + vetores), e o
   Adam não trabalha assim. O Adam trabalha com algarismos sequenciais: é o
   INSEPA que descobre a POSIÇÃO exata de uma palavra, por registro, não por
   aproximação.

2. Uma palavra só é reconhecida de duas formas: (a) ela já é um token que
   existe em algum bloco, ou (b) ela é uma VAR ou MULTIVAR já registrada
   DENTRO de um bloco específico. Ex.: o bloco do "gato preto" pode ter a
   multivar "o gato cor de carvão" (frase inteira) e a var "ônix" (no lugar
   de "preto") -- ambas resolvem pro mesmo bloco porque estão cadastradas
   ali, não porque uma métrica achou elas "parecidas".

3. Se a palavra não bate com nenhum token nem var/multivar registrada, ela é
   simplesmente DESCONHECIDA -- sem tentar achar "o mais parecido".

4. Todo usuário pode produzir uma OPINIÃO (texto + emoção). Isso não é dado
   concreto ainda -- só vira DADO CONCRETO quando também tem contexto e
   pensamento interno, e só a criadora pode preenchê-los.

5. Segurança do aprendizado: quando uma opinião é salva, texto (E) e emoção
   (RE) recebem marcadores reais e sequenciais -- mas contexto (CE) e
   pensamento interno (PIDE) recebem um PLACEHOLDER: "{IM}.0". O índice
   filho ".0" nunca é usado pela sequência normal (que sempre começa em .1),
   então esse placeholder marca, na própria estrutura do dado, que aquele
   campo ainda não foi confirmado -- sem precisar de uma flag extra.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from insepa_marcadores import marcar_entrada


def resolver_termo(termo: str, registro_vars_multivars: Dict[str, List[str]]) -> Optional[str]:
    """Resolve um termo pro bloco/token canônico que ele representa -- só por
    registro EXATO de var/multivar, nunca por aproximação de forma/distância.

    `registro_vars_multivars` mapeia um token canônico (ex.: "preto") pra
    todas as vars/multivars já cadastradas pra ele (ex.: ["ônix", "cor de carvão"]).
    """
    termo_norm = termo.strip().lower()
    for canonico, variantes in registro_vars_multivars.items():
        if termo_norm == canonico.lower():
            return canonico
        if termo_norm in [v.lower() for v in variantes]:
            return canonico
    return None


def classificar_dado(texto: str, emocao: str, contexto: str, pensamento: str) -> str:
    """Opinião: só texto+emoção -- o que qualquer usuário pode produzir. Dado
    concreto: todos os quatro campos, e só a criadora define contexto/pensamento."""
    if texto.strip() and emocao.strip() and contexto.strip() and pensamento.strip():
        return "dado concreto"
    if texto.strip() and emocao.strip():
        return "opinião"
    return "incompleto"


def marcar_opiniao(im: str, texto: str, emocao: str, ultimo: str) -> Dict[str, object]:
    """Marca uma opinião: texto (E) e emoção (RE) recebem marcadores reais e
    sequenciais; contexto (CE) e pensamento (PIDE) recebem o placeholder
    "{IM}.0", nunca usado pela sequência normal."""
    marcadores_e, ultimo = marcar_entrada(texto, ultimo)
    marcadores_re, ultimo = marcar_entrada(emocao, ultimo) if emocao.strip() else ([], ultimo)
    placeholder = f"{im}.0"
    return {
        "E": [m for m, _ in marcadores_e],
        "RE": [m for m, _ in marcadores_re],
        "CE": placeholder,
        "PIDE": placeholder,
        "ultimo": ultimo,
    }
