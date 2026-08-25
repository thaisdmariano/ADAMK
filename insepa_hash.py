#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 4: Hashrização.

Toda sequência de marcador tem entrada e saída. Essas duas partes podem ser
encontradas pela JUNÇÃO dos marcadores -- como uma chave/hash.

Exemplo real (bloco 1, domínio 0 / Adam):
  entrada: 0.1 ... 0.25
  saída:   0.26 ... 0.37

Se o Adam recebe a chave "0.1...0.25", essa chave destranca a porta pra
mostrar o arquivo "0.26...0.37". É um processo de segurança de dados: só a
chave exata destranca -- nada de adivinhação.

Terminologia: dentro de um universo/IM, o índice MÃE (IM) é o número antes
do ponto (ex.: o "0" de "0.25") -- é o mesmo pra todos os blocos daquele
universo. Os números depois do ponto (0.1, 0.25, 0.26, 0.37...) são os
índices FILHOS (IF), herdeiros daquele mesmo IM. Nem todo IM é "0" -- cada
universo/domínio tem o seu próprio.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def chave_hash(marcadores: List[str]) -> str:
    """Junta uma sequência de marcadores numa chave única, tipo hash."""
    return ",".join(marcadores)


def im_de(marcador: str) -> str:
    """Extrai o Índice Mãe (IM) de um marcador -- a parte antes do ponto."""
    return marcador.partition(".")[0]


class Cofre:
    """Um cofre de chave->porta: chave de entrada (hash dos marcadores)
    destranca a porta pra revelar os marcadores de saída correspondentes."""

    def __init__(self) -> None:
        self._portas: Dict[str, Tuple[List[str], List[str]]] = {}

    def trancar(self, marcadores_entrada: List[str], marcadores_saida: List[str]) -> None:
        """Registra uma nova porta: a chave da entrada abre pra essa saída."""
        chave = chave_hash(marcadores_entrada)
        self._portas[chave] = (marcadores_entrada, marcadores_saida)

    def destrancar(self, marcadores_entrada: List[str]) -> Optional[List[str]]:
        """Só abre com a chave EXATA -- nada de chave parecida."""
        chave = chave_hash(marcadores_entrada)
        porta = self._portas.get(chave)
        return porta[1] if porta else None
