#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 3: Integração.

Objetivo desta camada: integrar os elementos pra formar uma resposta coesa.

O pensamento_interno é a árvore (o tronco/raiz) — não o último passo de uma
linha única. A partir dele nascem DOIS ramos, cada um com seu próprio
texto + emoção + contexto:

  RAMO ENTRADA: texto e emoção enviados pelo usuário; contexto inferido com
  base nesse input.

  RAMO SAÍDA: texto e emoção devolvidos pelo Adam; contexto próprio, mas
  associado ao da entrada.

Os dois ramos "ressoam" entre si por compartilharem a mesma raiz (o
pensamento) — não porque o texto do contexto de um pareça com o do outro.
Contexto de entrada e de saída são escritos em vocabulários diferentes por
convenção (ex.: "Saudação doce e carinhosa" vs "Saudação espirituosa") e
isso é esperado, não um erro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Ramo:
    texto: str
    emocao: str
    contexto: str


@dataclass
class BlocoIntegrado:
    pensamento: str
    entrada: Ramo
    saida: Ramo


def montar_bloco(
    pensamento: str,
    entrada_texto: str,
    entrada_emocao: str,
    entrada_contexto: str,
    saida_texto: str,
    saida_emocao: str,
    saida_contexto: str,
) -> BlocoIntegrado:
    """Monta os dois ramos (entrada e saída) em torno da mesma raiz
    (pensamento), sem inventar nada -- só estrutura o que foi dado."""
    return BlocoIntegrado(
        pensamento=pensamento,
        entrada=Ramo(entrada_texto, entrada_emocao, entrada_contexto),
        saida=Ramo(saida_texto, saida_emocao, saida_contexto),
    )
