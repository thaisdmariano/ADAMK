#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 1: Marcadores Únicos.

Objetivo desta camada: dar a CADA token (palavra ou pontuação) de UMA entrada
um marcador numérico próprio, que nunca se repete fora daquela entrada.

Por que isso importa: a mesma palavra em duas frases diferentes recebe
marcadores diferentes. "Noite" na frase A e "noite" na frase B nunca são o
mesmo símbolo. Isso é o que impede o sistema de confundir o significado de
uma palavra: a palavra sozinha não carrega identidade nenhuma, só o token
dentro daquela sequência específica carrega.

INSEPA aceita tudo: pontuação, stopwords, tudo vira token e ganha marcador.
Nada é filtrado, nada é normalizado embora fora.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def tokenizar(texto: str) -> List[str]:
    """Quebra o texto em tokens: uma sequência de letras/números é um token,
    e cada caractere de pontuação/símbolo é um token à parte.

    Nada é descartado -- sem stopwords, sem normalização de acento aqui.
    """
    return re.findall(r"\w+|[^\w\s]", texto, re.UNICODE)


def proximo_marcador(anterior: str) -> str:
    """Incrementa o filho (IF) sem arredondar: 0.9 -> 0.10, não 0.9 -> 1.0."""
    mae, _, filho = anterior.partition(".")
    if not mae.isdigit():
        raise ValueError(f"Marcador inválido: {anterior!r}")
    return f"{mae}.{int(filho or '0') + 1}"


def marcar_entrada(texto: str, ultimo_marcador: str) -> Tuple[List[Tuple[str, str]], str]:
    """Marca cada token do texto com um marcador único, continuando a sequência
    a partir de `ultimo_marcador` (o último marcador já usado no universo/IM).

    Retorna (lista de (marcador, token), novo último marcador).
    """
    marcadores: List[Tuple[str, str]] = []
    atual = ultimo_marcador
    for token in tokenizar(texto):
        atual = proximo_marcador(atual)
        marcadores.append((atual, token))
    return marcadores, atual


def formatar(marcadores: List[Tuple[str, str]]) -> str:
    return " ".join(f"{token}({marcador})" for marcador, token in marcadores)


if __name__ == "__main__":
    # Reproduzindo o exemplo: duas frases quase idênticas, uma boa e uma sombria.
    ultimo = "0.0"

    frase1 = "A noite era brilhante."
    marcadores1, ultimo = marcar_entrada(frase1, ultimo)
    print(f'"{frase1}"')
    print(" ", formatar(marcadores1))
    print()

    frase2 = "A noite era sombria."
    marcadores2, ultimo = marcar_entrada(frase2, ultimo)
    print(f'"{frase2}"')
    print(" ", formatar(marcadores2))
    print()

    print("As duas sequências de marcadores nunca se sobrepõem, mesmo")
    print('compartilhando "A", "noite" e "era" -- cada frase inteira tem sua')
    print('própria identidade numérica. "Noite" na frase 1 e "noite" na frase 2')
    print("são marcadores diferentes, nunca o mesmo símbolo.")
