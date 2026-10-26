# INSEPA FRAMEWORK -- o motor, sem nenhuma referência a "Adam" dentro.
# Construído em 2026-09-29 a partir do cerebro_adam.py, por pedido direto
# da Thaís: "quero replicar um agente de I.A com a tecnologia INSEPA...
# o framework INSEPA". Qualquer persona nova (nome, corpus, tom) roda em
# cima deste arquivo -- e pode nascer HERDANDO a memória, o cérebro
# treinado e a temperatura emocional de uma persona-pai (2026-09-29:
# "herdar memórias do Adam, mas aprender a partir dos novos dados").
#
# Princípios (definidos pela Thaís, válidos pra qualquer persona):
#   1) Rede PEQUENA de propósito -- protótipo, não precisa de escala de
#      data center.
#   2) Integração de campos, NUNCA campos globais.
#   3) X=Y, E=S: entrada e saída são UMA sequência só, nunca dois sistemas
#      separados.
#   4) Posição = índice real do marcador, nunca posição relativa inventada.
#   5) Bloco = identidade própria, contida no índice mãe.
#
# Este arquivo não sabe o nome de nenhuma persona, nem seu tom, nem onde
# fica o corpus dela -- tudo isso é a classe Persona, definida por quem
# usa o framework (ver persona_adam.py).

import hashlib
import json
import math
import os as _os
import re
import time
import unicodedata
from collections import Counter as _Counter
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

from insepa_autonomia import (LIMIAR_COSSENO_TEXTO, confirmar_estrutura_baskara, encontrar_texto_mais_parecido,
                              similaridade_texto_cosseno)


# ────────────────────────────────────────────────────────────────────────
# Persona -- tudo que diferencia UMA persona de outra, nunca hardcoded no
# resto do arquivo. Uma réplica nova é só uma Persona nova apontando pros
# próprios arquivos/banco -- o motor abaixo é sempre o mesmo.
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Persona:
    nome: str
    pasta: Path
    # Endereço carinhoso que Adam usa pra chamar a criadora nas falas
    # afetuosas (2026-09-28: "o Adam é antes de tudo o meu agente pessoal").
    tratamento: str = "minha criadora"
    mongo_db: str = "insepa"
    mongo_uri_padrao: str = "mongodb://localhost:27017"
    # Frases-gatilho do Livro de Lux -- cada persona pode ter as suas.
    gatilhos_abrir_livro: Tuple[str, ...] = (
        "É hora de escrever.", "Vamos mudar o mundo.", "Hoje nasce um novo universo.",
    )
    nome_livro: str = "Livro de Lux"
    # Hiperparâmetros da rede -- padrão pequeno de propósito, mas cada
    # persona pode ajustar se o corpus dela crescer muito.
    embed_dim: int = 32
    n_camadas: int = 2
    n_heads: int = 2
    max_marker_idx: int = 500
    escala_cimento: float = 2.0  # peso da região (contexto) no muro
    escala_tijolo: float = 4.0  # peso do limite do tijolo (pensamento + identidade no índice mãe)
    raio_tijolo: float = 4.0  # TRAVA: o máximo que uma palavra se afasta do centro do próprio tijolo
    epocas_padrao: int = 1000
    # Nomes de arquivo por convenção ("{nome}_memory.json" etc.) -- pra uma
    # réplica nova nascer sem precisar decidir nome de arquivo nenhum. Uma
    # persona com arquivos que já existiam ANTES dessa convenção (o Adam
    # original) pode apontar direto pros arquivos dela aqui, sem renomear
    # nem duplicar nada.
    arquivo_memoria_proprio: Path | None = None
    arquivo_registro_emocional_proprio: Path | None = None
    arquivo_estado_proprio: Path | None = None
    arquivo_comandos_proprio: Path | None = None
    arquivo_ida_proprio: Path | None = None
    ckpt_path_proprio: Path | None = None

    @property
    def arquivo_memoria(self) -> Path:
        return self.arquivo_memoria_proprio or self.pasta / f"{self.nome}_memory.json"

    @property
    def arquivo_registro_emocional(self) -> Path:
        return self.arquivo_registro_emocional_proprio or self.pasta / f"{self.nome}_motor_emocional_registro.json"

    @property
    def arquivo_estado(self) -> Path:
        return self.arquivo_estado_proprio or self.pasta / f"{self.nome}_estado.json"

    @property
    def arquivo_comandos(self) -> Path:
        return self.arquivo_comandos_proprio or self.pasta / f"{self.nome}_comandos.json"

    @property
    def arquivo_ida(self) -> Path:
        return self.arquivo_ida_proprio or self.pasta / f"{self.nome}_ida.json"

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_path_proprio or self.pasta / f"{self.nome}_ckpt.pt"


def mongo_config(chave: str, padrao: str, persona: Persona | None = None) -> str:
    """Variável de ambiente sempre vence (pra nunca surpreender quem testa
    local); senão st.secrets, se existir; senão o padrão -- opcionalmente
    o da própria Persona."""
    if _os.environ.get(chave):
        return _os.environ[chave]
    try:
        import streamlit as st
        if chave in st.secrets:
            return st.secrets[chave]
    except Exception:
        pass
    return padrao


class MongoINSEPA:
    """Conexão Mongo de UMA persona: um doc por universo (sem os blocos
    dentro) + um doc por BLOCO -- se um quebrar, quebra só ele. 2026-09-28,
    pedido da Thaís: organizado como os JSONs, nunca um documento gigante.
    Se o Mongo não estiver disponível, tudo cai pro arquivo local -- nunca
    trava o app."""

    def __init__(self, persona: Persona):
        self.persona = persona
        self.enabled = False
        self.erro = None
        try:
            from pymongo import MongoClient
            uri = mongo_config("MONGO_URI", persona.mongo_uri_padrao, persona)
            db_nome = mongo_config("MONGO_DB", persona.mongo_db, persona)
            client = MongoClient(uri, serverSelectionTimeoutMS=8000)
            client.admin.command("ping")
            db = client[db_nome]
            self.col_universos = db["universos"]
            self.col_blocos = db["blocos"]
            self.col_motor_emocional = db["motor_emocional"]
            self.col_comandos = db["comandos"]
            # IDA -- coleções PRÓPRIAS, nunca as mesmas do IM: mesmo se
            # alguém esquecer de filtrar depois, carregar_memoria() (que
            # monta vocabulário e treina o cérebro) estruturalmente não
            # tem como ver rascunho do IDA -- não é a mesma coleção.
            self.col_ida_universos = db["ida_universos"]
            self.col_ida_blocos = db["ida_blocos"]
            self.enabled = True
        except ImportError:
            self.erro = "pymongo não instalado"
        except Exception as e:
            self.erro = str(e)

    def salvar_universo(self, dominio: str, universo: dict) -> None:
        if not self.enabled:
            return
        self.col_universos.update_one(
            {"_id": dominio},
            {"$set": {"nome": universo.get("nome"), "ultimo_child": universo.get("ultimo_child")}},
            upsert=True,
        )

    def salvar_bloco(self, dominio: str, bloco: dict) -> None:
        if not self.enabled:
            return
        doc = dict(bloco)
        doc["_id"] = f"{dominio}_{bloco['bloco_id']}"
        doc["dominio"] = dominio
        self.col_blocos.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def apagar_bloco(self, dominio: str, bloco_id: int) -> None:
        """salvar_memoria só acrescenta/atualiza -- sem isto, um bloco
        apagado do arquivo voltaria do Mongo na próxima carga."""
        if not self.enabled:
            return
        self.col_blocos.delete_one({"_id": f"{dominio}_{bloco_id}"})

    def salvar_memoria(self, memoria: dict) -> None:
        if not self.enabled:
            return
        for dominio, universo in memoria.get("IM", {}).items():
            self.salvar_universo(dominio, universo)
            for bloco in universo.get("blocos", []):
                self.salvar_bloco(dominio, bloco)

    def carregar_memoria(self) -> dict | None:
        if not self.enabled:
            return None
        universos = list(self.col_universos.find())
        if not universos:
            return None
        memoria: dict = {"IM": {}}
        for u in universos:
            dominio = u["_id"]
            blocos = list(self.col_blocos.find({"dominio": dominio}).sort("bloco_id", 1))
            for b in blocos:
                b.pop("_id", None)
                b.pop("dominio", None)
            memoria["IM"][dominio] = {"nome": u.get("nome"), "ultimo_child": u.get("ultimo_child"), "blocos": blocos}
        return memoria

    def salvar_registro_emocional(self, registro: dict) -> None:
        if not self.enabled:
            return
        self.col_motor_emocional.update_one({"_id": "registro"}, {"$set": {"dados": registro}}, upsert=True)

    def carregar_registro_emocional(self) -> dict | None:
        if not self.enabled:
            return None
        doc = self.col_motor_emocional.find_one({"_id": "registro"})
        return doc["dados"] if doc else None

    def salvar_ida_bloco(self, bloco: dict) -> None:
        if not self.enabled:
            return
        doc = dict(bloco)
        doc["_id"] = f"ida_{bloco['bloco_id']}"
        self.col_ida_blocos.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def salvar_ida_universo(self, ida: dict) -> None:
        """Universo-espelho inteiro -- mesmo formato de um universo do IM
        (nome/ultimo_child/blocos), só que em coleção própria."""
        if not self.enabled:
            return
        self.col_ida_universos.update_one(
            {"_id": "ida"},
            {"$set": {"nome": ida.get("nome"), "ultimo_child": ida.get("ultimo_child")}},
            upsert=True,
        )
        for bloco in ida.get("blocos", []):
            self.salvar_ida_bloco(bloco)

    def carregar_ida(self) -> dict | None:
        if not self.enabled:
            return None
        u = self.col_ida_universos.find_one({"_id": "ida"})
        if u is None:
            return None
        blocos = list(self.col_ida_blocos.find().sort("bloco_id", 1))
        for b in blocos:
            b.pop("_id", None)
        return {"nome": u.get("nome"), "ultimo_child": u.get("ultimo_child"), "blocos": blocos}

    def salvar_comandos(self, comandos: dict) -> None:
        if not self.enabled:
            return
        self.col_comandos.update_one({"_id": "registro"}, {"$set": {"dados": comandos}}, upsert=True)

    def carregar_comandos(self) -> dict | None:
        if not self.enabled:
            return None
        doc = self.col_comandos.find_one({"_id": "registro"})
        return doc["dados"] if doc else None


# ────────────────────────────────────────────────────────────────────────
# INSEPA mínimo -- ALNULU e marcadores. Nunca importa nada de fora, pra
# poder evoluir sem arrastar bagagem de nenhum modelo antigo.
# ────────────────────────────────────────────────────────────────────────
_ALNULU_MAPA = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': -10, 'K': 11, 'L': 12, 'M': -13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,
    'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': -22, 'W': 23, 'X': 24, 'Y': -25, 'Z': 26,
}
# Letra acentuada tem o número da letra base -- a MESMA tabela do ALNULU
# original da Thaís (lovely_test.py, alnulu_encode): Á/À/Â/Ã/Ä = A, Ç = C...
# Antes, aqui, toda acentuada virava 0: "é", "à" e o "0" do placeholder
# "0.0" eram a mesma forma (2026-10-19).
_ALNULU_EQUIV = {
    'Á': 'A', 'À': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'È': 'E', 'Ê': 'E', 'É': 'E',
    'Ì': 'I', 'Î': 'I', 'Í': 'I', 'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
    'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U', 'Ç': 'C', 'Ñ': 'N',
}


def _valor_letra(c: str) -> int | None:
    """Número ALNULU de uma letra (acentuada vale a base); None se não for letra do alfabeto."""
    cu = c.upper()
    return _ALNULU_MAPA.get(_ALNULU_EQUIV.get(cu, cu))


def tokenizar(texto: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", texto or "", re.UNICODE)


def alnulu_string(palavra: str) -> str:
    """Forma canônica ALNULU de uma palavra -- a 'semente universal' (Camada 1)."""
    letras = [c for c in palavra if c.isalpha()]
    valores = [str(v if (v := _valor_letra(c)) is not None else 0) for c in letras]
    return "-".join(valores) if valores else palavra.lower()


LETRAS_ALNULU = 12                      # letras da palavra que entram na rede
TAM_ALNULU = LETRAS_ALNULU + 2          # forma da palavra: números ALNULU + tamanho + reservado
TAM_LOCAL = 2 * TAM_ALNULU + 1          # localização do bloco: contexto (CE) + pensamento interno (PIDE) + id do tijolo


def alnulu_vetor(palavra: str, tam: int = TAM_ALNULU) -> List[float]:
    """A FORMA da palavra como entrada da rede: os próprios NÚMEROS DO ALNULU,
    letra por letra e na ordem (A=1 ... Z=26, J/M/V/Y negativos, acentuada =
    letra base), divididos por 26. Sem hash, sem sorteio, sem tabela por
    palavra -- é o ALNULU da Camada 1 mesmo. Caractere que não é letra do
    alfabeto (símbolo, dígito, emoji) ganha uma faixa à parte acima de 1.5. Regra da Thaís (2026-10-07): "esse Olá nunca
    pode ser o mesmo que o outro -- as similaridades só existem pro
    reconhecimento da forma"; a identidade do token vem do marcador + rótulo
    + localização do bloco, não da palavra."""
    v: List[float] = []
    for c in palavra[:LETRAS_ALNULU]:
        valor = _valor_letra(c)
        if valor is not None:
            v.append(valor / 26.0)
        else:
            v.append(1.5 + (ord(c.upper()) % 251) / 251.0)
    v += [0.0] * (LETRAS_ALNULU - len(v))
    return v + [min(len(palavra), 16) / 16.0, 0.0]


def assinatura_contexto(seq, com_tijolo: bool = True) -> List[float]:
    """Onde o bloco fica no muro (2026-10-07, alegoria da Thaís): cada bloco
    é um TIJOLO com limite próprio; o CONTEXTO (CE) é o cimento -- blocos de
    mesmo contexto ficam na mesma região, ligados, mas nunca fundidos; o
    PENSAMENTO INTERNO (PIDE) marca o limite do tijolo dentro da região
    ("festa" + aniversário e "festa" + fúnebre são vizinhos, não o mesmo
    tijolo); e a identidade do bloco no índice mãe garante que cada tijolo é
    um só. Bloco sem contexto/pensamento ainda: sem cimento, tijolo
    reservado no lugar dele -- nunca empilhado com os outros vazios.
    Devolve [CE | PIDE | id do tijolo] -- tudo determinístico, nunca aprendido."""
    raiz = seq[0].raiz if seq else None

    def media(rotulo: str) -> List[float]:
        if raiz is not None:  # o dado inteiro, não só os marcadores do PIDE
            toks = tokenizar(raiz.contexto if rotulo == "CE" else raiz.pensamento)
        else:
            toks = [t.palavra for t in seq if t.rotulo == rotulo and t.palavra]
        if _vazio("".join(toks)):  # "0.0" = ainda não explicado
            return [0.0] * TAM_ALNULU
        return np.mean([alnulu_vetor(p) for p in toks], axis=0).tolist()
    tijolo = float(seq[0].bloco_id) if (com_tijolo and seq) else -1.0
    return media("CE") + media("PIDE") + [tijolo]

def dividir_travessao(texto: str) -> Tuple[str, str, str]:
    partes = (texto or "").split("—")
    if len(partes) < 3:
        return texto or "", "", ""
    antes = partes[0]
    fala = partes[1]
    depois = "—".join(partes[2:])
    return antes, fala, depois


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def _vazio(valor) -> bool:
    v = str(valor or "").strip()
    return not v or re.fullmatch(r"\d+\.0", v) is not None


def _e_palavra(p: str) -> bool:
    return bool(p) and any(c.isalnum() for c in p)


# Rótulos de campo -- pequeno conjunto FECHADO e SABIDO (nunca adivinhado),
# a própria estrutura do INSEPA, igual pra qualquer persona.
ROTULOS_ENTRADA = ["TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"]
# PIDS (2026-10-19, pedido da Thaís): o pensamento da SAÍDA, no espaço de
# saída junto com os outros campos. Quem cria é o Adam (no IDA); só vira
# fato no IM depois do filtro dela. Nunca aparece na fala.
ROTULOS_SAIDA = ["TEDSA", "FADES", "TEFSA", "RS", "CS", "PIDS"]
TODOS_ROTULOS = ROTULOS_ENTRADA + ROTULOS_SAIDA
ROTULO_PARA_IDX = {r: i for i, r in enumerate(TODOS_ROTULOS)}
FIM_ROTULO_IDX = len(TODOS_ROTULOS)  # rótulo especial: "acabou a fala" (cabeça de rótulo)
# Contexto (CE/CS) NUNCA aparece pro usuário -- é insumo interno pra
# calcular intenção/sentimento. O que o usuário vê é texto de saída + reação.
ROTULOS_VISIVEIS_AO_USUARIO = ["TEDSA", "FADES", "TEFSA", "RS"]
_TEXTO_ENTRADA = ("TEXE", "FADEN", "TEFE")
_TEXTO_SAIDA = ("TEDSA", "FADES", "TEFSA")
_FAMS_ENTRADA = ["TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"]
_FAMS_SAIDA = ["TEDSA", "FADES", "TEFSA", "RS", "CS", "PIDS"]
FRIEZA = "0.0"  # ausência de reação E de indicativo emocional é frieza de verdade


class TokenDoBloco:
    """Um token real dentro de UMA sequência única entrada+saída -- nunca
    dois sistemas separados. Carrega tudo que já é sabido (rótulo, posição
    do marcador, bloco) -- nada disso é adivinhado pela rede."""
    __slots__ = ("palavra", "rotulo", "marker_idx", "bloco_id", "raiz")

    def __init__(self, palavra: str, rotulo: str, marker_idx: int, bloco_id: int, raiz: "RaizDoBloco | None" = None):
        self.palavra = palavra
        self.rotulo = rotulo
        self.marker_idx = marker_idx
        self.bloco_id = bloco_id
        self.raiz = raiz


class RaizDoBloco:
    """A raiz de onde cada palavra (semente) do bloco pende: reação, contexto
    e pensamento interno INTEIROS. O PIDE guarda só os 3 primeiros marcadores
    do pensamento de propósito (pro bloco não ficar gigante) -- mas a rede
    não pode deixar de ver o dado (Thaís, 2026-10-11): por isso o pensamento
    completo chega por aqui, sem virar marcador."""
    __slots__ = ("reacao", "contexto", "pensamento")

    def __init__(self, reacao: str, contexto: str, pensamento: str):
        self.reacao = reacao or ""
        self.contexto = contexto or ""
        self.pensamento = pensamento or ""


def sequencia_do_bloco(bloco: dict) -> List[TokenDoBloco]:
    """Monta a sequência ÚNICA (entrada seguida de saída, mesma ordem dos
    marcadores reais) -- X=Y, E=S: é UMA sequência, nunca duas. Cada
    marcador carrega a própria palavra junto: {"marcador": "0.1", "palavra":
    "Olá"} -- nunca só o número separado pra decifrar depois."""
    bloco_id = bloco["bloco_id"]
    entrada = bloco["entrada"]
    saida = bloco["saidas"][0]
    seq: List[TokenDoBloco] = []
    raiz = RaizDoBloco(entrada.get("reacao", ""), entrada.get("contexto", ""), entrada.get("pensamento_interno", ""))

    def filho(marcador: str) -> int:
        return int(marcador.split(".", 1)[1])

    for rotulo in ("TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"):
        for item in entrada["tokens"].get(rotulo, []):
            seq.append(TokenDoBloco(item["palavra"], rotulo, filho(item["marcador"]), bloco_id, raiz))
    for rotulo in ("TEDSA", "FADES", "TEFSA", "RS", "CS", "PIDS"):
        for item in saida["tokens"].get(rotulo, []):
            seq.append(TokenDoBloco(item["palavra"], rotulo, filho(item["marcador"]), bloco_id, raiz))

    seq.sort(key=lambda t: t.marker_idx)
    return seq


def renumerar_universo(universo: dict) -> None:
    """Quando um bloco já criado é editado, a forma correta é renumerar a
    sequência do universo INTEIRO em ordem, do zero -- nunca deixar
    marcador órfão/gapado no meio."""
    cursor = 0

    def prox(palavras: List[str]) -> List[dict]:
        nonlocal cursor
        pares = [{"marcador": f"0.{cursor + i + 1}", "palavra": p} for i, p in enumerate(palavras)]
        cursor += len(palavras)
        return pares

    for bloco in sorted(universo["blocos"], key=lambda b: b["bloco_id"]):
        e, s = bloco["entrada"], bloco["saidas"][0]
        texe, faden, tefe = (tokenizar(p) for p in dividir_travessao(e.get("texto", "")))
        palavras_e = {"TEXE": texe, "FADEN": faden, "TEFE": tefe, "RE": [e.get("reacao", "")],
                      "CE": tokenizar(e.get("contexto", "")), "PIDE": tokenizar(e.get("pensamento_interno", ""))[:3]}
        ent_total = []
        for fam in _FAMS_ENTRADA:
            pares = prox(palavras_e[fam])
            e["tokens"][fam] = pares
            ent_total += [p["marcador"] for p in pares]
        e["tokens"]["TOTAL"] = ent_total
        e["fim"] = ent_total[-1] if ent_total else e.get("fim", "")

        texto_saida = (s.get("textos") or [""])[0]
        tedsa, fades, tefsa = (tokenizar(p) for p in dividir_travessao(texto_saida))
        palavras_s = {"TEDSA": tedsa, "FADES": fades, "TEFSA": tefsa, "RS": [s.get("reacao", "")],
                      "CS": tokenizar(s.get("contexto", "")),
                      "PIDS": tokenizar(s.get("pensamento_interno", ""))[:3]}  # mesmo corte do PIDE: o bloco não fica gigante
        sai_total = []
        for fam in _FAMS_SAIDA:
            pares = prox(palavras_s[fam])
            s["tokens"][fam] = pares
            sai_total += [p["marcador"] for p in pares]
        s["tokens"]["TOTAL"] = sai_total
        s["fim"] = sai_total[-1] if sai_total else s.get("fim", "")

    universo["ultimo_child"] = f"0.{cursor}"


# ────────────────────────────────────────────────────────────────────────
# A rede neural -- INSEPA puro, sem nada de nenhuma persona dentro.
# ────────────────────────────────────────────────────────────────────────
class PosicaoPorMarcador(nn.Module):
    """Seno/cosseno indexado pelo índice REAL do marcador -- índice mãe
    ordenado. Determinístico, nunca aprendido do zero, nunca perde ordem."""

    def __init__(self, dim: int, max_len: int, base: float = 10000.0, passo: float = 1.0):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        posicao = passo * torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        pe[:, 0::2] = torch.sin(posicao * div)
        pe[:, 1::2] = torch.cos(posicao * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # 2026-10-19: antes havia um clamp aqui -- passou do teto, todo token
        # caía em silêncio na MESMA posição. Agora o teto acompanha o universo
        # (AppINSEPA.recarregar) e, se mesmo assim algo passar, é erro.
        if idx.numel() and int(idx.max()) >= self.pe.size(0):
            raise ValueError(f"marcador {int(idx.max())} passou do teto de posições ({self.pe.size(0)})")
        return self.pe[idx]


MAX_IDENTIDADE_IDA = 20000  # teto generoso pra origem_bloco_id * MULT_IDA_ORIGEM + copia_numero
MAX_TIJOLOS = 2000
# Identidade de tijolo (IM e IDA): seno/cosseno determinístico, com base e
# passo escolhidos pra que DOIS TIJOLOS QUAISQUER fiquem sempre longe um do
# outro -- medido: distância mínima 3.39 entre quaisquer dois de 20000 (a
# posição de marcador comum, base 10000, daria só 1.17 entre vizinhos). É o
# que deixa a trava fechar sem precisar de uma escala alta, que atrasa o treino.
BASE_IDENTIDADE, PASSO_IDENTIDADE = 6.0, 2.0


MULT_IDA_ORIGEM = 37  # primo -- espalha as identidades, evita colisão entre origens diferentes


def identidade_ida_numero(origem_bloco_id: int, copia_numero: int) -> int:
    """Converte {origem}.{cópia} (a identidade que a criadora vê, tipo
    "1.1") num único número -- pra alimentar o campo de identidade
    determinístico do IDA (pos_ida), nunca pra exibir."""
    return min(origem_bloco_id * MULT_IDA_ORIGEM + copia_numero, MAX_IDENTIDADE_IDA - 1)


class RedeINSEPA(nn.Module):
    """Uma sequência só (E=S), integração de campos por rótulo+posição+
    bloco -- nunca campos globais soltos. Pequena de propósito. Nenhuma
    persona é mencionada aqui: o mesmo desenho serve pra qualquer uma."""

    def __init__(self, vocab_size: int, persona: Persona):
        super().__init__()
        self.persona = persona
        d = persona.embed_dim
        # A palavra entra SÓ pela forma (letras/ALNULU) -- nenhuma linha
        # aprendida por palavra, nada compartilhado entre tokens. Cada
        # token é único (marcador + rótulo + bloco); a semelhança entre
        # palavras existe apenas pra reconhecer a forma, e nada mais.
        self.projetor_alnulu = nn.Linear(TAM_ALNULU, d)
        self.pos_marcador = PosicaoPorMarcador(d, persona.max_marker_idx)
        self.embed_rotulo = nn.Embedding(len(TODOS_ROTULOS), d)  # já sabido, nunca adivinhado
        # O muro (ver assinatura_contexto): cimento = contexto, limite do
        # tijolo = pensamento interno + identidade do bloco no índice mãe
        # (seno/cosseno do bloco_id, determinístico, nunca uma linha aprendida).
        self.projetor_cimento = nn.Linear(TAM_ALNULU, d)
        self.projetor_pensamento = nn.Linear(TAM_ALNULU, d)
        self.pos_tijolo = PosicaoPorMarcador(d, MAX_TIJOLOS, base=BASE_IDENTIDADE, passo=PASSO_IDENTIDADE)
        # 2026-10-05: o IDA tem CAMPO DE IDENTIDADE PRÓPRIO -- nunca
        # emprestado do bloco do IM que inspirou. Determinístico igual à
        # posição do marcador (seno/cosseno), nunca uma nn.Embedding
        # treinada -- rascunho do IDA nunca é dado de treino, então nunca
        # teria sinal legítimo pra aprender um embedding de verdade.
        self.pos_ida = PosicaoPorMarcador(d, MAX_IDENTIDADE_IDA, base=BASE_IDENTIDADE, passo=PASSO_IDENTIDADE)

        # sem dropout: a rede existe pra memorizar EXATO (100% pelo molde) --
        # medido em 2026-10-07, o dropout de 10% só atrasava isso.
        camada = nn.TransformerEncoderLayer(d_model=d, nhead=persona.n_heads, dim_feedforward=d * 4, dropout=0.0)
        self.transformer = nn.TransformerEncoder(camada, num_layers=persona.n_camadas)
        # Saída: reconhece/produz a FORMA da próxima palavra (classe de
        # forma ALNULU) -- sem peso amarrado a nenhum embedding de palavra,
        # porque não existe mais embedding de palavra.
        self.saida = nn.Linear(d, vocab_size, bias=False)
        # 2026-10-05, degrau 3: prevê o PRÓXIMO RÓTULO (e quando parar -- FIM)
        # -- o cérebro decidindo a própria estrutura da fala, não só
        # preenchendo um molde que já existe. Cabeçote À PARTE, nunca usado
        # pelo forward() de sempre -- a geração por molde continua idêntica.
        self.cabeca_rotulo = nn.Linear(d, len(TODOS_ROTULOS) + 1)  # +1 = FIM

    def localizacao(self, local, ident_ida=None):
        """CENTRO do tijolo no muro: região pelo cimento (contexto), limite pelo
        tijolo (pensamento + identidade). No IM a identidade é a do bloco no
        índice mãe; no IDA (`ident_ida` = número de {origem}.{cópia}) é a
        PRÓPRIA do rascunho -- o espelho tem os tijolos dele, nunca os do IM."""
        n = TAM_ALNULU
        cimento = self.projetor_cimento(local[..., :n])
        pensamento = self.projetor_pensamento(local[..., n:2 * n])
        if ident_ida is not None:
            ident = self.pos_ida(ident_ida)
        else:
            tid = local[..., 2 * n]
            ident = self.pos_tijolo(tid.clamp(min=0).long()) * (tid >= 0).unsqueeze(-1).float()
        p = self.persona
        return p.escala_cimento * cimento + p.escala_tijolo * (pensamento + ident)

    def dentro_do_tijolo(self, deslocamento):
        """TRAVA DE SEGURANÇA (Thaís, 2026-10-11): "cada bloco é um espaço com
        conhecimentos específicos" -- os tijolos nunca podem se misturar por
        aproximação. Nenhuma palavra se afasta do centro do próprio tijolo
        mais que `raio_tijolo`: passou do limite, é trazida de volta pra
        borda (mesma direção). Junto com `verificar_trava`, garante que dois
        tijolos nunca se tocam."""
        r = self.persona.raio_tijolo
        norma = deslocamento.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return deslocamento * (r / norma).clamp(max=1.0)

    def compor(self, alnulu, marker_idx, rotulo_idx, local, ident_ida=None):
        """O ponto de cada palavra: centro do tijolo + o lugar dela lá dentro
        (forma + marcador + rótulo), sempre dentro do limite."""
        dentro = self.dentro_do_tijolo(self.embed_palavra(alnulu) + self.pos_marcador(marker_idx) + self.embed_rotulo(rotulo_idx))
        return self.localizacao(local, ident_ida) + dentro

    def embed_palavra(self, alnulu):
        """A palavra é só a sua FORMA projetada -- vista ou nunca vista, é
        a mesma regra, sem distinção."""
        return self.projetor_alnulu(alnulu)

    def forward(self, alnulu, marker_idx, rotulo_idx, local):
        x = self.compor(alnulu, marker_idx, rotulo_idx, local)
        x = x.permute(1, 0, 2)
        seq_len = x.size(0)
        mascara_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mascara_causal)
        x = x.permute(1, 0, 2)
        return self.saida(x)

    def forward_com_rotulo(self, alnulu, marker_idx, rotulo_idx, local, vies_ida=None):
        """Mesma rede, com (1) a semente ALNULU no lugar do embedding pra
        palavra nunca vista e (2) o cabeçote de rótulo -- usado só pelo
        treino conjunto e pela geração livre, nunca pela geração por
        molde de sempre. `vies_ida`, quando dado, soma a identidade
        PRÓPRIA do rascunho do IDA (nunca usado no treino -- só a geração
        livre passa isso)."""
        x = self.compor(alnulu, marker_idx, rotulo_idx, local)
        if vies_ida is not None:
            x = x + vies_ida
        x = x.permute(1, 0, 2)
        seq_len = x.size(0)
        mascara_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mascara_causal)
        x = x.permute(1, 0, 2)
        return self.saida(x), self.cabeca_rotulo(x)


# ────────────────────────────────────────────────────────────────────────
# Dataset e treino.
# ────────────────────────────────────────────────────────────────────────
def montar_vocabulario(sequencias: List[List[TokenDoBloco]]) -> Dict[str, int]:
    formas = sorted({alnulu_string(t.palavra) for seq in sequencias for t in seq if t.palavra})
    return {forma: i for i, forma in enumerate(formas)}


def montar_forma_para_palavra(sequencias: List[List[TokenDoBloco]]) -> Dict[str, str]:
    """A forma ALNULU é a semente universal -- várias palavras reais podem
    colapsar na mesma forma. Pra EXIBIR, guarda uma palavra de verdade
    representando cada forma (a mais usada no corpus)."""
    from collections import Counter
    contagem: Dict[str, Counter] = {}
    for seq in sequencias:
        for t in seq:
            if not t.palavra:
                continue
            forma = alnulu_string(t.palavra)
            contagem.setdefault(forma, Counter())[t.palavra] += 1
    return {forma: c.most_common(1)[0][0] for forma, c in contagem.items()}


def montar_rotulo_para_formas(sequencias: List[List[TokenDoBloco]]) -> Dict[str, set]:
    """Pra cada rótulo, quais formas ALNULU já apareceram confirmadas ali
    no IM -- isolamento de campo de verdade na geração livre: a palavra
    sorteada só pode vir do que já foi visto NAQUELE campo específico,
    nunca vazar de um campo pro outro (ex.: uma palavra de TEDSA nunca
    pode ser sorteada pro RS)."""
    mapa: Dict[str, set] = {r: set() for r in TODOS_ROTULOS}
    for seq in sequencias:
        for t in seq:
            if t.palavra:
                mapa[t.rotulo].add(alnulu_string(t.palavra))
    return mapa


def montar_tensores(seq: List[TokenDoBloco], vocab: Dict[str, int]) -> Dict[str, torch.Tensor]:
    ids = [vocab[alnulu_string(t.palavra)] for t in seq]
    marker_idx = [t.marker_idx for t in seq]
    rotulo_idx = [ROTULO_PARA_IDX[t.rotulo] for t in seq]
    local = [assinatura_contexto(seq)] * len(seq)
    # sempre conhecida aqui (vem do vocabulário treinado) -- as duas só
    # importam de verdade na geração livre, com palavra nunca vista.
    conhecida = [1.0] * len(seq)
    alnulu = [alnulu_vetor(t.palavra) for t in seq]
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "marker_idx": torch.tensor(marker_idx, dtype=torch.long),
        "rotulo_idx": torch.tensor(rotulo_idx, dtype=torch.long),
        "local": torch.tensor(local, dtype=torch.float),
        "conhecida": torch.tensor(conhecida, dtype=torch.float),
        "alnulu": torch.tensor(alnulu, dtype=torch.float),
    }


def _acerta_tudo(model: RedeINSEPA, blocos_tensores: List[Dict[str, torch.Tensor]]) -> bool:
    """Teacher forcing: em todo bloco, a próxima palavra em cada posição sai
    certa? (Com isso a geração a partir do molde real sai idêntica.)"""
    model.eval()
    ok = True
    with torch.no_grad():
        for t in blocos_tensores:
            ids = t["ids"].unsqueeze(0)
            lg = model(t["alnulu"].unsqueeze(0), t["marker_idx"].unsqueeze(0), t["rotulo_idx"].unsqueeze(0), t["local"].unsqueeze(0))
            if not bool((lg[0, :-1].argmax(-1) == ids[0, 1:]).all()):
                ok = False
                break
    model.train()
    return ok


def treinar(model: RedeINSEPA, blocos_tensores: List[Dict[str, torch.Tensor]], epocas: int, lr: float = 3e-3,
            parar_ao_acertar_tudo: bool = True, epocas_estrutura: int = 300):
    """`epocas` é o TETO da Fase 1. Medido em 5 sementes (corpus do Adam):
    100% chega com 325-575 épocas -- então, por padrão, confere a cada 25
    e PARA quando acerta tudo duas vezes seguidas.

    DUAS FASES, nunca competindo pelos mesmos pesos:
    Fase 1 -- exatamente como sempre foi, só a palavra pelo molde. É o que
    toda conversa do dia a dia depende; não muda em nada.
    Fase 2 -- só DEPOIS da palavra já estar sólida, SÓ a cabeça de rótulo
    aprende (campo seguinte e fim da fala, usados pelo rascunho do IDA), com
    todo o resto da rede TRAVADO (requires_grad=False) -- nunca arrisca
    desfazer o que a Fase 1 já conquistou."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    historico = []
    model.train()
    seguidas = 0
    for ep in range(epocas):
        perda_total = 0.0
        for t in blocos_tensores:
            opt.zero_grad()
            ids = t["ids"].unsqueeze(0)
            logits = model(t["alnulu"].unsqueeze(0), t["marker_idx"].unsqueeze(0), t["rotulo_idx"].unsqueeze(0), t["local"].unsqueeze(0))
            alvo = ids[:, 1:].reshape(-1)
            pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
            perda = F.cross_entropy(pred, alvo)
            perda.backward()
            opt.step()
            perda_total += perda.item()
        historico.append(perda_total / len(blocos_tensores))
        if parar_ao_acertar_tudo and (ep + 1) % 25 == 0:
            seguidas = seguidas + 1 if _acerta_tudo(model, blocos_tensores) else 0
            if seguidas >= 2:
                break

    for p in model.parameters():
        p.requires_grad = False
    for p in model.cabeca_rotulo.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam(list(model.cabeca_rotulo.parameters()), lr=lr)
    # FIM só acontece 1 vez por bloco, contra dezenas de outros rótulos --
    # sem compensar isso ele nunca aprende essa classe rara direito (visto
    # na prática: ficava repetindo o último campo pra sempre, sem parar).
    pesos_rotulo = torch.ones(len(TODOS_ROTULOS) + 1)
    pesos_rotulo[FIM_ROTULO_IDX] = 4.0
    for _ in range(epocas_estrutura):
        for t in blocos_tensores:
            opt2.zero_grad()
            ids = t["ids"].unsqueeze(0)
            marker_idx, rotulo_idx, local = t["marker_idx"].unsqueeze(0), t["rotulo_idx"].unsqueeze(0), t["local"].unsqueeze(0)
            alnulu = t["alnulu"].unsqueeze(0)
            # identidade do IDA sorteada a cada passo -- nenhum bloco de
            # treino é do IDA de verdade, mas a cabeça precisa aprender a
            # conviver com esse sinal extra, senão ele vira ruído puro na
            # hora da geração livre (visto na prática: sem isso, a
            # geração quebrava por completo assim que o viés entrava).
            num_ida_treino = int(torch.randint(0, MAX_IDENTIDADE_IDA, (1,)).item())
            vies_ida_treino = model.pos_ida(torch.tensor([[num_ida_treino]]))
            logits_palavra, logits_rotulo = model.forward_com_rotulo(alnulu, marker_idx, rotulo_idx, local, vies_ida=vies_ida_treino)
            alvo_palavra = ids[:, 1:].reshape(-1)
            pred_palavra = logits_palavra[:, :-1, :].reshape(-1, logits_palavra.size(-1))
            perda_palavra = F.cross_entropy(pred_palavra, alvo_palavra)
            alvo_rotulo = torch.cat([rotulo_idx[:, 1:], torch.tensor([[FIM_ROTULO_IDX]])], dim=1).reshape(-1)
            pred_rotulo = logits_rotulo.reshape(-1, logits_rotulo.size(-1))
            perda_rotulo = F.cross_entropy(pred_rotulo, alvo_rotulo, weight=pesos_rotulo)
            (perda_palavra + perda_rotulo).backward()
            opt2.step()
    for p in model.parameters():
        p.requires_grad = True
    return historico


def gerar_continuacao(model: RedeINSEPA, seq_entrada: List[TokenDoBloco], seq_saida_molde: List[TokenDoBloco],
                       vocab: Dict[str, int], idx_para_palavra: Dict[int, str]) -> List[str]:
    """Gera a PALAVRA em cada posição da saída -- mas o RÓTULO e o ÍNDICE DE
    MARCADOR de cada posição vêm do MOLDE estrutural, nunca inventados. A
    palavra em cada posição é livre (gerada de verdade); o rótulo/posição
    nunca é (informação já sabida não é chute). A localização do bloco vem
    do contexto + pensamento da própria entrada."""
    model.eval()
    t = montar_tensores(seq_entrada, vocab)
    alnulu = t["alnulu"].unsqueeze(0)
    marker_idx = t["marker_idx"].unsqueeze(0)
    rotulo_idx = t["rotulo_idx"].unsqueeze(0)
    local = t["local"].unsqueeze(0)
    gerados = []
    with torch.no_grad():
        for tok_molde in seq_saida_molde:
            logits = model(alnulu, marker_idx, rotulo_idx, local)
            proximo = logits[0, -1].argmax().item()
            palavra_gerada = idx_para_palavra.get(proximo, "?")
            # o cérebro decide a FORMA; se ela é a do próprio marcador, quem
            # escreve é a palavra exata guardada nele ("é" e "e" dividem a
            # forma no ALNULU, mas o marcador sabe qual das duas é a dele).
            if tok_molde.palavra and vocab.get(alnulu_string(tok_molde.palavra)) == proximo:
                palavra_gerada = tok_molde.palavra
            gerados.append(palavra_gerada)
            novo_rotulo = ROTULO_PARA_IDX[tok_molde.rotulo]
            alnulu = torch.cat([alnulu, torch.tensor([[alnulu_vetor(palavra_gerada)]])], dim=1)
            marker_idx = torch.cat([marker_idx, torch.tensor([[tok_molde.marker_idx]])], dim=1)
            rotulo_idx = torch.cat([rotulo_idx, torch.tensor([[novo_rotulo]])], dim=1)
            local = torch.cat([local, local[:, :1, :]], dim=1)
    return gerados


_FECHA_FRASE = set(".!?…")  # só o que fecha uma frase de verdade -- vírgula, hífen e travessão não fecham


def frases_do_molde(molde: List[TokenDoBloco]) -> List[List[TokenDoBloco]]:
    """Divide a saída de um bloco em FRASES INTEIRAS, na ordem dos marcadores:
    cada frase vai até a pontuação que a fecha e nunca atravessa de um campo
    pro outro; a reação (RS) é uma frase de um marcador só. Regra da Thaís:
    a frase inteira carrega o contexto junto e é segura de reaproveitar --
    palavra solta recombinada é "caos de I.A. comum"."""
    frases: List[List[TokenDoBloco]] = []
    atual: List[TokenDoBloco] = []
    for tok in molde:
        if not tok.palavra or tok.rotulo not in ROTULOS_VISIVEIS_AO_USUARIO:
            continue
        if atual and atual[-1].rotulo != tok.rotulo:
            frases.append(atual)
            atual = []
        atual.append(tok)
        if tok.rotulo == "RS" or set(tok.palavra) <= _FECHA_FRASE:
            frases.append(atual)
            atual = []
    if atual:
        frases.append(atual)
    return frases


def gerar_livre(model: RedeINSEPA, seq_ancora: List[TokenDoBloco], vocab: Dict[str, int],
                 origem_bloco_id: int, copia_numero: int) -> List[Tuple[str, str]]:
    """O rascunho do Adam no IDA. Decisão da Thaís (2026-10-11): só no IDA, o
    Adam pode divergir do IM em cinco pontos -- estrutura da resposta,
    contexto e intenção, o que fazer quando não sabe, de onde aprende e
    decisão por probabilidade. TODO O RESTO continua INSEPA aqui dentro.

    O que é LIVRE neste rascunho (pontos 1 e 5): a ESTRUTURA -- quais frases
    do bloco de origem entram e onde a fala termina -- decidida por
    PROBABILIDADE, com as probabilidades que o próprio cérebro calcula
    (campo + palavra seguinte + fim). Dois rascunhos do mesmo bloco podem
    sair diferentes.

    O que é FIXO (INSEPA, mesmo no IDA):
      - a matéria vem de UM tijolo só (o bloco de origem) -- nunca mistura blocos;
      - cada token é único e preso ao seu marcador: entra no máximo uma vez,
        com a palavra exata da origem (sem repetição, sem "saco de palavras");
      - a ordem é a dos marcadores reais: nunca volta, nunca embaralha;
      - a unidade é a frase inteira: nunca corta no meio;
      - rascunho nunca sai vazio.
    A identidade própria do rascunho ({origem}.{cópia}) entra pelo pos_ida."""
    model.eval()
    frases = frases_do_molde([t for t in seq_ancora if t.rotulo in ROTULOS_SAIDA])
    if not frases:
        return []
    local_ancora = assinatura_contexto(seq_ancora)
    entrada = [t for t in seq_ancora if t.rotulo in ROTULOS_ENTRADA]
    alnulu = [alnulu_vetor(t.palavra) for t in entrada]
    marker_idx = [t.marker_idx for t in entrada]
    rotulo_idx = [ROTULO_PARA_IDX[t.rotulo] for t in entrada]
    gerado: List[Tuple[str, str]] = []
    proxima = 0
    with torch.no_grad():
        vies_ida = model.pos_ida(torch.tensor([[identidade_ida_numero(origem_bloco_id, copia_numero)]]))
        while proxima < len(frases):
            n = len(alnulu)
            logits_palavra, logits_rotulo = model.forward_com_rotulo(
                torch.tensor([alnulu]), torch.tensor([marker_idx]), torch.tensor([rotulo_idx]),
                torch.tensor([[local_ancora] * n], dtype=torch.float), vies_ida=vies_ida)
            lp_rotulo = F.log_softmax(logits_rotulo[0, -1], dim=-1)
            lp_palavra = F.log_softmax(logits_palavra[0, -1], dim=-1)
            # candidatas: cada frase que ainda está à frente (pular é permitido,
            # voltar não) -- e parar, se já existe texto.
            candidatas = list(range(proxima, len(frases)))
            notas = [lp_rotulo[ROTULO_PARA_IDX[frases[j][0].rotulo]] + lp_palavra[vocab[alnulu_string(frases[j][0].palavra)]]
                     for j in candidatas]
            tem_texto = any(r in _TEXTO_SAIDA for _, r in gerado)
            if tem_texto:
                candidatas.append(-1)
                notas.append(lp_rotulo[FIM_ROTULO_IDX])
            escolha = candidatas[torch.multinomial(F.softmax(torch.stack(notas), dim=-1), 1).item()]
            if escolha == -1:
                break
            for tok in frases[escolha]:
                gerado.append((tok.palavra, tok.rotulo))
                alnulu.append(alnulu_vetor(tok.palavra))
                marker_idx.append(tok.marker_idx)  # o marcador real do token, nunca inventado
                rotulo_idx.append(ROTULO_PARA_IDX[tok.rotulo])
            proxima = escolha + 1
    return gerado


# ────────────────────────────────────────────────────────────────────────
# Mapa 3D -- puramente visual (SVD), nunca participa de nenhuma decisão.
# ────────────────────────────────────────────────────────────────────────
def _projetar_3d(matriz: np.ndarray) -> np.ndarray:
    n = matriz.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    centrada = matriz - matriz.mean(axis=0, keepdims=True)
    if n == 1:
        return np.zeros((1, 3))
    _, _, vt = np.linalg.svd(centrada, full_matrices=False)
    k = min(3, vt.shape[0])
    componentes = vt[:k]
    projetado = centrada @ componentes.T
    if k < 3:
        projetado = np.hstack([projetado, np.zeros((n, 3 - k))])
    return projetado


def _paleta_distinta(n: int) -> List[str]:
    """Uma cor DIFERENTE por bloco, quantos blocos houver -- com paleta
    fixa de 5 cores os blocos 1 e 6, 2 e 7, 3 e 8... saíam iguais e o mapa
    parecia misturado sem estar (2026-10-07). Tons espaçados pelo número
    áureo pra vizinhos nunca ficarem parecidos."""
    import colorsys
    cores = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62 + 0.18 * (i % 2), 0.95 - 0.12 * (i % 3))
        cores.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return cores


def _projetar_3d_por_blocos(matriz: np.ndarray, blocos_id: List[int]) -> np.ndarray:
    """Projeção feita pra MOSTRAR a separação entre blocos: os 3 eixos são as
    direções em que os CENTROS dos blocos mais se afastam entre si (e não as
    de maior variação entre tokens soltos, que misturavam os blocos). Só
    desenho -- nunca decide nada."""
    n = matriz.shape[0]
    if n < 2:
        return np.zeros((n, 3))
    ids = np.array(blocos_id)
    media = matriz.mean(axis=0, keepdims=True)
    centrada = matriz - media
    centros = np.array([centrada[ids == b].mean(axis=0) for b in sorted(set(blocos_id))])
    eixos = []
    if centros.shape[0] >= 2:
        _, s, vt = np.linalg.svd(centros - centros.mean(axis=0, keepdims=True), full_matrices=False)
        eixos = [v for v, sv in zip(vt, s) if sv > 1e-6][:3]
    if len(eixos) < 3:  # poucos blocos: completa com as direções gerais dos tokens
        _, _, vt = np.linalg.svd(centrada, full_matrices=False)
        for v in vt:
            if len(eixos) == 3:
                break
            if all(abs(float(np.dot(v, e))) < 0.99 for e in eixos):
                eixos.append(v)
    projetado = centrada @ np.array(eixos).T
    if projetado.shape[1] < 3:
        projetado = np.hstack([projetado, np.zeros((n, 3 - projetado.shape[1]))])
    return projetado


def vetores_por_token(model: RedeINSEPA, sequencias: List[List[TokenDoBloco]], vocab: Dict[str, int]
                       ) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    model.eval()
    vetores, palavras, blocos_id, rotulos = [], [], [], []
    with torch.no_grad():
        for seq in sequencias:
            t = montar_tensores(seq, vocab)
            x = model.compor(t["alnulu"], t["marker_idx"], t["rotulo_idx"], t["local"])
            for i, tok in enumerate(seq):
                vetores.append(x[i].numpy())
                palavras.append(tok.palavra)
                blocos_id.append(tok.bloco_id)
                rotulos.append(tok.rotulo)
    return np.array(vetores), palavras, blocos_id, rotulos


def grafico_cerebro_3d(model: RedeINSEPA, sequencias: List[List[TokenDoBloco]], vocab: Dict[str, int],
                        mostrar_palavras: bool = False) -> go.Figure:
    vetores, palavras, blocos_id, rotulos = vetores_por_token(model, sequencias, vocab)
    coords = _projetar_3d_por_blocos(vetores, blocos_id)
    ids_ordenados = sorted(set(blocos_id))
    cores_por_bloco = _paleta_distinta(len(ids_ordenados))
    fig = go.Figure()
    for pos_cor, bid in enumerate(ids_ordenados):
        idxs = [i for i, b in enumerate(blocos_id) if b == bid]
        cor = cores_por_bloco[pos_cor]
        fig.add_trace(go.Scatter3d(
            x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
            mode="markers+text" if mostrar_palavras else "markers",
            text=[palavras[i] for i in idxs],
            textposition="top center",
            marker=dict(size=5, color=cor),
            hovertext=[f"{palavras[i]} ({rotulos[i]}, bloco {blocos_id[i]})" for i in idxs],
            name=f"bloco {bid}",
        ))
    fig.update_layout(
        scene=dict(xaxis_title="componente 1", yaxis_title="componente 2", zaxis_title="componente 3"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=600,
        legend=dict(title="bloco"),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────
# Mapa 3D do IDA -- os rascunhos do Adam no MESMO espaço do cérebro
# treinado, pra ver se "a mente que sonha ressoa com o cérebro
# estruturado" (2026-10-05, pedido da Thaís). Sem identidade de bloco
# emprestada: o cérebro nunca treinou com essas identidades, fingir uma
# seria inventar dado que não existe.
# ────────────────────────────────────────────────────────────────────────
def _vetor_bloco_ida(model: RedeINSEPA, seq: List[TokenDoBloco], vocab: Dict[str, int],
                      origem_bloco_id: int | None = None, copia_numero: int | None = None
                      ) -> Tuple[np.ndarray, List[str], List[bool]]:
    """Toda palavra entra pela própria FORMA (conhecida ou nova, mesma regra).
    O ponto é: forma + marcador + rótulo + localização (contexto + pensamento
    do rascunho). Carrega também o
    CAMPO DE IDENTIDADE PRÓPRIO do IDA (pos_ida) quando origem/cópia são
    dados -- cada proposta ocupa seu próprio canto do mapa, não só o do
    bloco do IM que inspirou."""
    if not seq:
        return np.zeros((0, model.persona.embed_dim)), [], []
    marker_idx = torch.tensor([t.marker_idx for t in seq], dtype=torch.long)
    rotulo_idx = torch.tensor([ROTULO_PARA_IDX[t.rotulo] for t in seq], dtype=torch.long)
    alnulu = torch.tensor([alnulu_vetor(t.palavra) for t in seq], dtype=torch.float)
    # 2026-10-11, Thaís: "o IDA não parece um espelho... cadê os blocos dele?"
    # -- o rascunho é um TIJOLO do IDA, igual a um bloco do IM: mesmo muro,
    # mesma trava, só que a identidade do tijolo é a própria do rascunho
    # ({origem}.{cópia}), nunca a do bloco do IM que inspirou.
    local = torch.tensor([assinatura_contexto(seq, com_tijolo=False)] * len(seq), dtype=torch.float)
    ident_ida = None
    if origem_bloco_id is not None and copia_numero is not None:
        ident_ida = torch.tensor([identidade_ida_numero(origem_bloco_id, copia_numero)] * len(seq), dtype=torch.long)
    x = model.compor(alnulu, marker_idx, rotulo_idx, local, ident_ida=ident_ida)
    palavras = [t.palavra for t in seq]
    exatos = [alnulu_string(t.palavra) in vocab for t in seq]
    return x.numpy(), palavras, exatos


def grafico_ida_3d_solo(model: RedeINSEPA, blocos_ida: List[dict], vocab: Dict[str, int], identidade_fn,
                         mostrar_palavras: bool = False) -> go.Figure:
    """Mapa SÓ do IDA -- espaço próprio, projeção própria (SVD calculado
    só com os pontos do IDA, nunca junto com o IM). 2026-10-05, pedido da
    Thaís: separar de vez os dois mapas, nunca compartilhar eixo --
    "IDA 0.0" como grupo, cada proposta (1.1, 8.1...) um item dentro."""
    model.eval()
    com_dados = []  # (identidade, vetores, palavras, exatos)
    with torch.no_grad():
        for b in blocos_ida:
            seq = sequencia_do_bloco(b)
            meta = b.get("meta", {})
            vetores_b, palavras_b, exatos_b = _vetor_bloco_ida(
                model, seq, vocab,
                origem_bloco_id=meta.get("origem_bloco_id"), copia_numero=meta.get("copia_numero"),
            )
            com_dados.append((identidade_fn(b), vetores_b, palavras_b, exatos_b))

    partes = [v for _, v, _, _ in com_dados if v.shape[0]]
    todos = np.vstack(partes) if partes else np.zeros((0, model.persona.embed_dim))
    ids_prop = [i for i, (_, v, _, _) in enumerate(com_dados) for _ in range(v.shape[0])]
    coords = _projetar_3d_por_blocos(todos, ids_prop)

    fig = go.Figure()
    cursor = 0
    cores = _paleta_distinta(len(com_dados))  # uma cor por tijolo do IDA, igual ao mapa do IM
    for pos_cor, (identidade, vetores_b, palavras_b, exatos_b) in enumerate(com_dados):
        n = vetores_b.shape[0]
        if n == 0:
            continue
        coords_b = coords[cursor:cursor + n]
        cursor += n
        fig.add_trace(go.Scatter3d(
            x=coords_b[:, 0], y=coords_b[:, 1], z=coords_b[:, 2],
            mode="markers+text" if mostrar_palavras else "markers", text=palavras_b, textposition="top center",
            marker=dict(size=6, color=cores[pos_cor], symbol="diamond"),
            hovertext=[f"{palavras_b[i]} · {identidade}" + ("" if exatos_b[i] else " (palavra nova, aproximada)") for i in range(n)],
            name=identidade, legendgroup="ida", legendgrouptitle_text="IDA 0.0",
        ))
    fig.update_layout(
        scene=dict(xaxis_title="componente 1", yaxis_title="componente 2", zaxis_title="componente 3"),
        margin=dict(l=0, r=0, b=0, t=10), height=560, legend=dict(title=""),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────
# Trava de segurança dos tijolos -- conferida no espaço REAL do cérebro
# (todas as dimensões), nunca na sombra 3D do mapa.
# ────────────────────────────────────────────────────────────────────────
def _folga_dos_tijolos(centros: Dict[str, np.ndarray], raio: float) -> Dict[str, object]:
    """Folga = distância entre os centros de dois tijolos menos os dois raios.
    Folga > 0: nem a palavra mais afastada de um alcança a do outro."""
    nomes = list(centros)
    menor, par = None, None
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            folga = float(np.linalg.norm(centros[a] - centros[b])) - 2 * raio
            if menor is None or folga < menor:
                menor, par = folga, (a, b)
    return {"ok": menor is None or menor > 0, "menor_folga": menor, "par": par, "tijolos": len(nomes), "raio": raio}


def verificar_trava(model: RedeINSEPA, sequencias: List[List[TokenDoBloco]]) -> Dict[str, object]:
    """Trava do IM: nenhum par de blocos pode se tocar."""
    model.eval()
    centros = {}
    with torch.no_grad():
        for seq in sequencias:
            if seq:
                local = torch.tensor(assinatura_contexto(seq), dtype=torch.float)
                centros[f"bloco {seq[0].bloco_id}"] = model.localizacao(local).numpy()
    return _folga_dos_tijolos(centros, model.persona.raio_tijolo)


def verificar_trava_ida(model: RedeINSEPA, blocos_ida: List[dict], identidade_fn) -> Dict[str, object]:
    """Mesma trava, no espaço próprio do IDA: nenhum par de rascunhos pode se tocar."""
    model.eval()
    centros = {}
    with torch.no_grad():
        for b in blocos_ida:
            seq = sequencia_do_bloco(b)
            meta = b.get("meta", {})
            if not seq or meta.get("origem_bloco_id") is None or meta.get("copia_numero") is None:
                continue
            local = torch.tensor(assinatura_contexto(seq, com_tijolo=False), dtype=torch.float)
            num = torch.tensor(identidade_ida_numero(meta["origem_bloco_id"], meta["copia_numero"]), dtype=torch.long)
            centros[str(identidade_fn(b))] = model.localizacao(local, ident_ida=num).numpy()
    return _folga_dos_tijolos(centros, model.persona.raio_tijolo)


# ────────────────────────────────────────────────────────────────────────
# Funções de apoio ao diálogo -- puras, tomam `blocos`/`sequencias` como
# parâmetro explícito (nunca globais de módulo), pra servir qualquer
# persona rodando em cima do framework.
# ────────────────────────────────────────────────────────────────────────
_PONTUACAO_SOZINHA = set(".,!?;:…-—")
ROTULOS_VISIVEIS = ROTULOS_VISIVEIS_AO_USUARIO
_PRIORIDADE_CAMPOS = ["texto da saída", "contexto", "pensamento", "reação da saída", "contexto da saída", "texto", "reação"]
_SIM = ("sim", "isso", "exato", "exatamente", "claro", "acertou", "era", "aham", "uhum", "s")
_NAO = ("nao", "errado", "negativo", "n")
FRASES_SEM_TAREFA = ("nao tenho o que fazer", "nao tenho nada pra fazer", "nao tenho nada para fazer",
                     "sem nada pra fazer", "sem nada para fazer", "estou entediada", "to entediada", "vamos revisar")
FRASES_CANCELAR = ("deixa pra la", "deixa para la", "cancelar", "cancela", "esquece", "agora nao")

# 2026-09-29, degrau 1 da autonomia pedida pela Thaís: comandos deixam de
# depender só de bater a frase exata. Mesmo motor dos blocos de conteúdo --
# cosseno ACHA o comando mais parecido; quem chama ainda cruza isso com o
# estado da conversa antes de agir (nunca decide sozinho por um número).
# LIMIAR_ALTA só age quando a leitura é bem confiável -- abaixo disso, cai
# como conversa normal, nunca um "meio-cancelamento" arriscado.
LIMIAR_COMANDO_ALTA_CONFIANCA = 0.85
LIMIAR_COMANDO_MINIMO = 0.55


def _cosseno_normalizado(texto_a: str, texto_b: str) -> float:
    """Cosseno de verdade, mas sobre as duas formas JÁ normalizadas (sem
    acento/pontuação) -- as frases canônicas de comando já vivem
    normalizadas (mesma convenção de sempre), então comparar contra o
    texto cru do usuário sem normalizar os dois lados por igual sub-
    contava qualquer acentuação (“lá” x “la”), fazendo o cosseno parecer
    mais baixo do que realmente é."""
    tokens_a = _Counter(_normalizar(texto_a).split())
    tokens_b = _Counter(_normalizar(texto_b).split())
    vocabulario = set(tokens_a) | set(tokens_b)
    if not vocabulario:
        return 0.0
    produto = sum(tokens_a.get(t, 0) * tokens_b.get(t, 0) for t in vocabulario)
    norma_a = sum(v * v for v in tokens_a.values()) ** 0.5
    norma_b = sum(v * v for v in tokens_b.values()) ** 0.5
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return produto / (norma_a * norma_b)


def reconhecer_comando(texto: str, candidatos: Dict[str, List[str]]) -> Tuple[str | None, float, bool]:
    """Devolve (nome do comando, score, exato). Exato quando a mensagem
    inteira OU só a primeira palavra bate ao pé da letra com alguma frase
    canônica/variante já aprendida daquele comando (cobre tanto "cancelar"
    quanto "sim, é isso mesmo"). Sem bate exato, cosseno rankeia -- só
    devolve um nome se algum score passar do piso mínimo; nunca inventa
    candidato de score zero."""
    texto_norm = _normalizar(texto)
    primeira_palavra = texto_norm.split()[0] if texto_norm else ""
    melhor_nome, melhor_score = None, 0.0
    for nome, frases in candidatos.items():
        frases_norm = {_normalizar(f) for f in frases}
        if texto_norm in frases_norm or primeira_palavra in frases_norm:
            return nome, 1.0, True
        score = max((_cosseno_normalizado(texto, f) for f in frases), default=0.0)
        if score > melhor_score:
            melhor_nome, melhor_score = nome, score
    if melhor_nome is None or melhor_score < LIMIAR_COMANDO_MINIMO:
        return None, melhor_score, False
    return melhor_nome, melhor_score, False


def formas_do_texto(texto: str) -> set:
    return {alnulu_string(p) for p in tokenizar(texto) if _e_palavra(p)}


def saida_e_placeholder(seq: List[TokenDoBloco]) -> bool:
    """'0.0' = 'esse lugar está vazio'. Não é lixo a esconder: a persona
    entende que o bloco ainda não tem resposta e diz isso."""
    texto = "".join(t.palavra for t in seq if t.rotulo in _TEXTO_SAIDA)
    return re.fullmatch(r"\d+\.0", texto.strip()) is not None


def separar_reacao(texto: str, reacoes_conhecidas: set) -> Tuple[str, str]:
    """Entrada crua = texto + reação. A reação é o último 'pedaço' da
    mensagem quando ele é uma reação já conhecida (exata) ou um emoji/
    símbolo sem letra nem número. Sem reação, devolve '' (vira 0.0)."""
    partes = texto.strip().rsplit(None, 1)
    if len(partes) == 2:
        ultimo = partes[1]
        sem_alnum = not any(c.isalnum() for c in ultimo)
        if ultimo in reacoes_conhecidas or (sem_alnum and not set(ultimo) <= _PONTUACAO_SOZINHA):
            return partes[0].strip(), ultimo
    return texto.strip(), ""


def campos_faltando(bloco: dict) -> List[str]:
    """Bloco 100% completo = entrada (texto, reação, contexto, pensamento) +
    saída (texto, reação, contexto) todos preenchidos. Só então ele conta
    como dado concreto."""
    e, s = bloco["entrada"], bloco["saidas"][0]
    campos = [("texto", e.get("texto")), ("reação", e.get("reacao")), ("contexto", e.get("contexto")),
              ("pensamento", e.get("pensamento_interno")), ("texto da saída", (s.get("textos") or [""])[0]),
              ("reação da saída", s.get("reacao")), ("contexto da saída", s.get("contexto"))]
    return [nome for nome, v in campos if _vazio(v)]


def reacao_efetiva(reacao: str) -> str:
    return FRIEZA if _vazio(reacao) else reacao


_SEM_ESPACO_ANTES = set(",.!?;:…)]}»”’")
_SEM_ESPACO_DEPOIS = set("([{«“‘")


def _juntar_ptbr(palavras: List[str]) -> str:
    """Junta os tokens de UM campo pela norma do português do Brasil: nada de
    espaço antes de vírgula/ponto/fecha-parêntese, aspas coladas no texto que
    elas envolvem (a primeira abre, a segunda fecha), hífen entre duas
    palavras sem espaço nenhum ("vê-la", "bem-vinda")."""
    saida, aspas_abertas, cola = "", False, False
    for i, p in enumerate(palavras):
        if not p:
            continue
        if p == '"':
            if aspas_abertas:
                saida += p  # fecha: cola na palavra de antes
            else:
                saida += (" " if saida else "") + p  # abre: cola na palavra seguinte
                cola = True
            aspas_abertas = not aspas_abertas
            continue
        if (p == "-" and saida and saida[-1].isalnum() and i + 1 < len(palavras)
                and palavras[i + 1][:1].isalnum()):
            saida += p
            cola = True
            continue
        if not saida or cola or set(p) <= _SEM_ESPACO_ANTES:
            saida += p
        else:
            saida += " " + p
        cola = set(p) <= _SEM_ESPACO_DEPOIS
    return saida


def montar_texto_ptbr(pares: List[Tuple[str, str]], campos: Tuple[str, str, str] = None, para_exibir: bool = False) -> str:
    """Monta o texto de um lado do bloco a partir de (rótulo, palavra), na
    norma do português do Brasil. Fala (FADEN/FADES) SEMPRE entre
    travessões -- é ela que separa narração de fala (Thaís, 2026-10-19: "numa
    FADEN ou FADSA o '—' é imprescindível").
      para_exibir=False -> formato de guardar no bloco: "antes — fala — depois",
                           sempre com os dois travessões (é assim que o INSEPA
                           reconhece os três campos);
      para_exibir=True  -> o que a pessoa lê: travessão que ficaria pendurado
                           (sem nada depois dele) não aparece."""
    campos = campos or _TEXTO_SAIDA
    antes, fala, depois = (_juntar_ptbr([p for r, p in pares if r == c]) for c in campos)
    if not fala and not depois:
        return antes
    if not para_exibir:
        return f"{antes} — {fala} — {depois}".strip()
    texto = antes
    if fala:
        texto = f"{texto} — {fala}" if texto else f"— {fala}"
    if depois:
        texto = f"{texto} — {depois}" if texto else depois
    return texto


def _arrumar_espacos(texto: str) -> str:
    return re.sub(r"\s+([,.!?;:])", r"\1", texto).strip()


def achar_bloco_parecido(texto: str, reacao: str, blocos: List[dict], seqs: List[List[TokenDoBloco]],
                          reacoes_aceitas_por_idx: Dict[int, List[str]]) -> Tuple[int, float, bool]:
    """Correspondência EXATA de forma ALNULU (sem embedding, sem distância
    vetorial): quantas sementes o que você disse divide com o texto de
    entrada de cada bloco (canônico ou Multivar registrado), e se a reação
    é exatamente uma das aceitas. Em empate de texto: reação igual ganha,
    depois bloco com resposta ganha de bloco vazio."""
    formas_user = formas_do_texto(texto)
    if not formas_user:
        return -1, 0.0, False
    melhor, melhor_chave, melhor_reacao = -1, (0.0,), False
    for i, seq in enumerate(seqs):
        variantes = [{alnulu_string(t.palavra) for t in seq if t.rotulo in _TEXTO_ENTRADA and _e_palavra(t.palavra)}]
        variantes += [formas_do_texto(v) for v in blocos[i]["entrada"].get("Multivars_Texto_Entrada", [])]
        scores = [len(formas_user & f) / len(formas_user | f) for f in variantes if f]
        if not scores:
            continue
        score = max(scores)
        aceitas = reacoes_aceitas_por_idx.get(i, [])
        reacao_igual = bool(reacao) and reacao in aceitas
        chave = (score, reacao_igual, not saida_e_placeholder(seq), not campos_faltando(blocos[i]))
        if score > 0 and chave > melhor_chave:
            melhor, melhor_chave, melhor_reacao = i, chave, reacao_igual
    return melhor, (melhor_chave[0] if melhor >= 0 else 0.0), melhor_reacao


def tom_da_entrada(texto: str, reacao: str) -> Tuple[str, str]:
    """Devolve (valor pra comparar, rótulo pra mostrar). Sem reação: '.' ou
    nenhuma pontuação = frieza de verdade. '!', '?' e '...' são cada um um
    indicativo emocional PRÓPRIO -- ainda não se sabe qual dos lados de
    cada um é, então o valor fica '' (desconhecido pro Báskara) até haver
    contexto suficiente pra triangular (texto + contexto + emoção juntos)."""
    if not _vazio(reacao):
        return reacao, reacao
    t = texto.rstrip()
    if t.endswith("..."):
        return "", "indicativo emocional (... -- algo mais a dizer)"
    if t.endswith("!"):
        return "", "indicativo emocional (! -- entusiasmo ou pânico)"
    if t.endswith("?"):
        return "", "indicativo emocional (? -- expressão interrogativa)"
    return FRIEZA, "frieza (sem indicativo emocional)"


# ────────────────────────────────────────────────────────────────────────
# Motor de temperatura emocional -- "o maior peso existe, ele só não anula
# o outro": positivo e negativo somados separados, o lado mais forte vence
# com o próprio valor -- nunca uma diferença/média escondendo o outro lado.
# ────────────────────────────────────────────────────────────────────────
# 2026-10-11, correção da Thaís: o contexto carregava peso demais (40) e o
# pensamento interno nem entrava. É pelo pensamento que o Adam recebe a
# noção aprofundada de cada assunto -- então ele pesa 30 e o contexto 10;
# texto e reação continuam em 15 "pra não desbalancear o que já funciona".
# 2026-10-19: a saída segue o mesmo equilíbrio -- contexto 10 e o
# pensamento da saída (PIDS, criado pelo Adam) 30. Entrada e saída têm
# temperaturas ISOLADAS uma da outra (cada lado só soma os próprios campos),
# pra cada parte do bloco ser lida por si e as respostas ficarem coerentes.
PESOS_TEMP_ENTRADA = {"TEXE": 15, "FADEN": 15, "TEFE": 15, "RE": 15, "CE": 10, "PIDE": 30}
PESOS_TEMP_SAIDA = {"TEDSA": 15, "FADES": 15, "TEFSA": 15, "RS": 15, "CS": 10, "PIDS": 30}
FAMS_TEXTO_TEMP_ENTRADA = ("TEXE", "FADEN", "TEFE")
FAMS_TEXTO_TEMP_SAIDA = ("TEDSA", "FADES", "TEFSA")
FAM_REACAO_TEMP = {"entrada": "RE", "saida": "RS"}
ESTADOS_TEMP = ["positivo", "negativo", "neutro"]
SINAL_TEMP = {"positivo": 1, "negativo": -1, "neutro": 0}
COR_TEMP = {"positivo": "#4fc9a8", "negativo": "#e2607a", "neutro": "#8a8fa3"}
APLICAR_BONUS_FRASE_REACAO = True
BONUS_NEUTRO_COM_REACAO = 5

# Faixas da Thaís (2026-10-19): seis, com Morno como ponto de equilíbrio.
# Ela deu o lado positivo e autorizou espelhar o negativo; o meio (-15 a
# 15) fica em Esfriando porque ela manteve "Esfriando até 15". Os números
# vão ser calibrados com o tempo.
#    86 a 100  -> Fervendo
#    61 a 85   -> Quente
#    16 a 60   -> Morno (equilíbrio)
#   -60 a 15   -> Esfriando
#   -85 a -61  -> Frio
#  -100 a -86  -> Gelo
ZONAS_TEMP = [
    (86, "🔥 Fervendo", "#e2607a"),
    (61, "☀️ Quente", "#f0a35a"),
    (16, "🌤️ Morno", "#4fc9a8"),
    (-60, "🌫️ Esfriando", "#7c93f5"),
    (-85, "❄️ Frio", "#5a6fd6"),
    (-101, "🧊 Gelo", "#2f3fa0"),
]


def classificar_zona_temp(temperatura: float) -> Tuple[str, str]:
    for minimo, nome, cor in ZONAS_TEMP:
        if temperatura >= minimo:
            return nome, cor
    return ZONAS_TEMP[-1][1], ZONAS_TEMP[-1][2]


def chave_temp(rotulo: str, valor: str) -> str:
    return f"{rotulo}:{valor.strip()}"


def valores_do_lado_temp(bloco: dict, lado: str) -> Dict[str, str]:
    if lado == "entrada":
        e = bloco["entrada"]
        texe, faden, tefe = dividir_travessao(e.get("texto", ""))
        return {"TEXE": texe, "FADEN": faden, "TEFE": tefe, "RE": e.get("reacao", ""), "CE": e.get("contexto", ""),
                "PIDE": e.get("pensamento_interno", "")}
    s = bloco["saidas"][0]
    texto_saida = (s.get("textos") or [""])[0]
    tedsa, fades, tefsa = dividir_travessao(texto_saida)
    return {"TEDSA": tedsa, "FADES": fades, "TEFSA": tefsa, "RS": s.get("reacao", ""), "CS": s.get("contexto", ""),
            "PIDS": s.get("pensamento_interno", "")}


def avaliar_lado_temp(bloco: dict, lado: str, registro: Dict[str, str]) -> Dict[str, object]:
    pesos = PESOS_TEMP_ENTRADA if lado == "entrada" else PESOS_TEMP_SAIDA
    valores = valores_do_lado_temp(bloco, lado)
    detalhe = []
    pendentes = []
    for fam, peso in pesos.items():
        valor = valores.get(fam, "")
        if _vazio(valor):
            detalhe.append({"fam": fam, "valor": "", "peso": peso, "estado": "vazio", "contribuicao": 0})
            continue
        k = chave_temp(fam, valor)
        estado = registro.get(k)
        if estado is None:
            pendentes.append((fam, valor))
            detalhe.append({"fam": fam, "valor": valor, "peso": peso, "estado": "pendente", "contribuicao": None})
            continue
        contribuicao = peso * SINAL_TEMP[estado]
        detalhe.append({"fam": fam, "valor": valor, "peso": peso, "estado": estado, "contribuicao": contribuicao})

    estado_por_fam = {item["fam"]: item["estado"] for item in detalhe}
    fams_texto = FAMS_TEXTO_TEMP_ENTRADA if lado == "entrada" else FAMS_TEXTO_TEMP_SAIDA
    estado_reacao = estado_por_fam.get(FAM_REACAO_TEMP[lado])
    bonus = []
    if APLICAR_BONUS_FRASE_REACAO and estado_reacao in ("positivo", "negativo"):
        for item in detalhe:
            if item["fam"] in fams_texto and item["estado"] == "neutro":
                pontos = BONUS_NEUTRO_COM_REACAO if estado_reacao == "positivo" else -BONUS_NEUTRO_COM_REACAO
                item["contribuicao"] += pontos
                bonus.append({"fam": item["fam"], "para": estado_reacao, "pontos": BONUS_NEUTRO_COM_REACAO})

    def sinal_vencedor(total_positivo: int, total_negativo: int) -> int:
        if total_positivo > total_negativo:
            return total_positivo
        if total_negativo > total_positivo:
            return -total_negativo
        return 0

    trajetoria = []
    pos_acum = neg_acum = 0
    truncado = False
    for item in detalhe:
        if item["estado"] == "pendente" or truncado:
            trajetoria.append({"fam": item["fam"], "contribuicao": item["contribuicao"], "cumulativo": None})
            truncado = True
            continue
        c = item["contribuicao"]
        pos_acum += max(c, 0)
        neg_acum += max(-c, 0)
        trajetoria.append({"fam": item["fam"], "contribuicao": c, "cumulativo": sinal_vencedor(pos_acum, neg_acum)})

    direcao = None
    if not pendentes and trajetoria:
        primeiro, ultimo = trajetoria[0]["cumulativo"], trajetoria[-1]["cumulativo"]
        if ultimo > primeiro:
            direcao = "📈 ascendente -- esquentando ao longo do bloco"
        elif ultimo < primeiro:
            direcao = "📉 descendente -- esfriando ao longo do bloco"
        else:
            direcao = "➡️ estável"

    if pendentes:
        return {"temperatura": None, "zona": None, "cor_zona": None, "motivo": f"{len(pendentes)} campo(s) sem registro",
                "detalhe": detalhe, "pendentes": pendentes, "bonus": bonus, "trajetoria": trajetoria, "direcao": direcao}
    temperatura = trajetoria[-1]["cumulativo"] if trajetoria else 0
    zona, cor_zona = classificar_zona_temp(temperatura)
    return {"temperatura": temperatura, "zona": zona, "cor_zona": cor_zona, "motivo": None,
            "detalhe": detalhe, "pendentes": pendentes, "bonus": bonus, "trajetoria": trajetoria, "direcao": direcao}


# ────────────────────────────────────────────────────────────────────────
# AppINSEPA -- reúne motor neural + diálogo + Livro de Lux + temperatura
# em cima de UMA Persona. Cada réplica é uma Persona nova rodando a mesma
# classe -- nada aqui sabe o nome de nenhuma persona específica.
#
# `estado` (em todo método que recebe) é um mapeamento mutável tipo
# st.session_state -- guarda só o que é efêmero de uma sessão de
# conversa (pendências, modo Livro de Lux aberto); nunca é persistido em
# disco por si só.
# ────────────────────────────────────────────────────────────────────────
class AppINSEPA:
    def __init__(self, persona: Persona, mongo: MongoINSEPA | None = None):
        self.persona = persona
        self.mongo = mongo or MongoINSEPA(persona)
        self.model: RedeINSEPA | None = None
        self.recarregar()
        self.comandos = self.carregar_comandos()
        self.ida = self.carregar_ida()

    # ---- comandos reconhecidos por julgamento (degrau 1 da autonomia) ---
    def _comandos_canonicos(self) -> Dict[str, List[str]]:
        return {
            "cancelar": list(FRASES_CANCELAR),
            "revisar": list(FRASES_SEM_TAREFA),
            "sim": list(_SIM),
            "nao": list(_NAO),
            "abrir_livro": list(self.persona.gatilhos_abrir_livro),
        }

    def _ler_variantes_comandos(self) -> Dict[str, List[str]]:
        do_mongo = self.mongo.carregar_comandos()
        if do_mongo is not None:
            return do_mongo
        if self.persona.arquivo_comandos.exists():
            return json.loads(self.persona.arquivo_comandos.read_text(encoding="utf-8"))
        return {}

    def carregar_comandos(self) -> Dict[str, List[str]]:
        canonicos = self._comandos_canonicos()
        variantes = self._ler_variantes_comandos()
        return {nome: frases + [v for v in variantes.get(nome, []) if v not in frases] for nome, frases in canonicos.items()}

    def registrar_variante_comando(self, nome_comando: str, texto: str) -> None:
        """Uma leitura de ALTA confiança vira sinônimo exato permanente do
        comando -- nunca uma leitura duvidosa. Mesmo princípio dos Multivars
        de conteúdo: o score nunca fica guardado como se fosse regra, só a
        frase exata, depois de confiança suficiente."""
        variantes = self._ler_variantes_comandos()
        lista = variantes.setdefault(nome_comando, [])
        if texto in lista or texto in self._comandos_canonicos().get(nome_comando, []):
            return
        lista.append(texto)
        self.mongo.salvar_comandos(variantes)
        arq = self.persona.arquivo_comandos
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(variantes, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.comandos = self.carregar_comandos()

    # ---- carregamento / corpus ----------------------------------------
    def carregar_memoria(self) -> dict:
        do_mongo = self.mongo.carregar_memoria()
        if do_mongo is not None:
            return do_mongo
        return json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))

    def recarregar(self) -> None:
        self.memoria = self.carregar_memoria()
        self.universo_id = next(iter(self.memoria["IM"]))
        self.blocos = self.memoria["IM"][self.universo_id]["blocos"]
        self.sequencias = [sequencia_do_bloco(b) for b in self.blocos]
        # teto de posição = o maior marcador real do IM e do IDA, com folga
        # pra crescer -- os marcadores contam o universo inteiro (0.1 até o
        # último), então o teto nunca pode ser um número fixo.
        maior = max([t.marker_idx for s in self.sequencias for t in s]
                    + [t.marker_idx for b in self.carregar_ida().get("blocos", []) for t in sequencia_do_bloco(b)] + [0])
        self.persona.max_marker_idx = max(self.persona.max_marker_idx, 2 * maior + 64)
        self.vocab = montar_vocabulario(self.sequencias)
        self.forma_para_palavra = montar_forma_para_palavra(self.sequencias)
        self.idx_para_palavra = {i: self.forma_para_palavra.get(f, f) for f, i in self.vocab.items()}
        self.mapa_bloco = {b["bloco_id"]: i for i, b in enumerate(self.blocos)}
        self.tensores = [montar_tensores(seq, self.vocab) for seq in self.sequencias]
        self.rotulo_para_formas = montar_rotulo_para_formas(self.sequencias)

    def definir_modelo(self, model: RedeINSEPA) -> None:
        self.model = model

    def assinatura_corpus(self) -> str:
        """Uma 'impressão digital' do corpus+hiperparâmetros atuais. Serve
        tanto pra saber se o checkpoint em disco ainda vale quanto pra ser
        chave de cache (quem usa esse valor como chave nunca fica com um
        modelo velho -- muda o corpus, muda a assinatura, invalida sozinho,
        não importa se a mudança veio de dentro do app ou de um script por
        fora enquanto ele estava rodando). Entra o CONTEÚDO de cada token
        (forma + rótulo), não só o tamanho: a rede memoriza o texto exato e
        a localização do bloco vem do contexto/pensamento dele, então mudar
        uma palavra sem mudar o tamanho também precisa retreinar. A escala da
        localização entra de propósito: pesos treinados só fazem sentido
        junto da escala usada no treino."""
        p = self.persona
        conteudo = str([[(alnulu_string(t.palavra), t.rotulo) for t in seq] for seq in self.sequencias])
        conteudo += str([(s[0].raiz.contexto, s[0].raiz.pensamento) for s in self.sequencias if s and s[0].raiz])
        return hashlib.sha256(
            (str(sorted(self.vocab.items())) + conteudo + f"|cimento={p.escala_cimento}|tijolo={p.escala_tijolo}|raio={p.raio_tijolo}|v10-ptbr").encode("utf-8")
        ).hexdigest()

    def treinar_ou_carregar(self) -> RedeINSEPA:
        """Treina uma vez (ou carrega o checkpoint salvo, se o corpus não
        mudou)."""
        p = self.persona
        assinatura = self.assinatura_corpus()
        m = RedeINSEPA(vocab_size=len(self.vocab), persona=p)
        if p.ckpt_path.exists():
            try:
                ck = torch.load(p.ckpt_path)
                if ck.get("assinatura") == assinatura:
                    m.load_state_dict(ck["state_dict"])
                    m.eval()
                    self.model = m
                    return m
            except Exception:
                pass
        # Trava de segurança: um cérebro em que dois tijolos se tocam não é
        # aceito -- treina de novo (até 3 vezes) em vez de seguir misturado.
        for tentativa in range(3):
            if tentativa:
                m = RedeINSEPA(vocab_size=len(self.vocab), persona=p)
            treinar(m, self.tensores, p.epocas_padrao)
            if verificar_trava(m, self.sequencias)["ok"]:
                break
        torch.save({"state_dict": m.state_dict(), "assinatura": assinatura}, p.ckpt_path)
        m.eval()
        self.model = m
        return m

    # ---- reações conhecidas / separação de reação ----------------------
    def reacoes_conhecidas(self) -> set:
        rs = set()
        for b in self.blocos:
            candidatas = [b["entrada"].get("reacao", ""), b["saidas"][0].get("reacao", "")]
            candidatas += list(b["entrada"].get("Multivars_Reacao_Entrada", []))
            for r in candidatas:
                if r and not re.fullmatch(r"\d+\.0", r):
                    rs.add(r)
        return rs

    def separar_reacao(self, texto: str) -> Tuple[str, str]:
        return separar_reacao(texto, self.reacoes_conhecidas())

    def reacoes_aceitas(self, i: int) -> List[str]:
        e = self.blocos[i]["entrada"]
        return [reacao_efetiva(e.get("reacao", ""))] + [r for r in e.get("Multivars_Reacao_Entrada", []) if not _vazio(r)]

    def _reacoes_aceitas_todas(self) -> Dict[int, List[str]]:
        return {i: self.reacoes_aceitas(i) for i in range(len(self.blocos))}

    # ---- escrita no corpus (arquivo local + Mongo) ----------------------
    def _puxar_mongo_pro_arquivo_local(self) -> None:
        """Antes de qualquer escrita: se o Mongo tiver dado, atualiza o
        arquivo local com ele primeiro -- assim um ensino feito na Cloud
        não é perdido quando se edita daqui de novo."""
        do_mongo = self.mongo.carregar_memoria()
        if do_mongo is not None:
            self.persona.arquivo_memoria.write_text(json.dumps(do_mongo, ensure_ascii=False, indent=2), encoding="utf-8")

    def _backup_memoria_uma_vez(self, estado: dict) -> None:
        if not estado.get("backup_ensino_feito"):
            carimbo = time.strftime("%Y%m%d_%H%M%S")
            arq = self.persona.arquivo_memoria
            arq.with_name(f"{arq.stem}.BACKUP_antes_do_ensino_{carimbo}.json").write_text(
                arq.read_text(encoding="utf-8"), encoding="utf-8")
            estado["backup_ensino_feito"] = True

    def _salvar_memoria_disco(self, memoria_disco: dict) -> None:
        """Chamado sempre depois de renumerar_universo, que mexe em TODOS
        os blocos do universo -- por isso sempre salva a memória inteira
        no Mongo, nunca só um bloco."""
        arq = self.persona.arquivo_memoria
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_memoria(memoria_disco)
        self.recarregar()
        self.model = None  # corpus mudou: precisa retreinar antes de falar de novo

    def registrar_variacao(self, bloco_id: int, texto: str, reacao: str, estado: dict) -> None:
        """Registra a frase (e a reação) como Multivar EXATO do bloco --
        nunca afrouxa comparação nenhuma: a partir daí ela bate por
        igualdade."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        bloco = next(b for b in memoria_disco["IM"][self.universo_id]["blocos"] if b["bloco_id"] == bloco_id)
        e = bloco["entrada"]
        mudou = False
        if texto and texto != e.get("texto") and texto not in e.setdefault("Multivars_Texto_Entrada", []):
            e["Multivars_Texto_Entrada"].append(texto)
            mudou = True
        if reacao and reacao != e.get("reacao") and reacao not in e.setdefault("Multivars_Reacao_Entrada", []):
            e["Multivars_Reacao_Entrada"].append(reacao)
            mudou = True
        if not mudou:
            return
        self._backup_memoria_uma_vez(estado)
        arq = self.persona.arquivo_memoria
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_bloco(self.universo_id, bloco)
        self.recarregar()

    def completar_campo(self, bloco_id: int, campo: str, valor_bruto: str, estado: dict) -> None:
        """Preenche um campo que faltava num bloco JÁ EXISTENTE -- nunca
        cria um bloco novo pra isso. Depois renumera o universo inteiro:
        é assim que se edita um bloco já criado, sem deixar marcador
        órfão."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        bloco = next(b for b in universo["blocos"] if b["bloco_id"] == bloco_id)
        e, s = bloco["entrada"], bloco["saidas"][0]
        valor = valor_bruto.strip()
        self._backup_memoria_uma_vez(estado)

        if campo == "contexto":
            e["contexto"] = valor
        elif campo == "pensamento":
            e["pensamento_interno"] = valor
        elif campo == "texto da saída":
            texto_saida, reacao_saida = self.separar_reacao(valor)
            s["textos"] = [texto_saida]
            if reacao_saida and _vazio(s.get("reacao")):
                s["reacao"] = reacao_saida
        elif campo == "reação da saída":
            s["reacao"] = valor
        elif campo == "contexto da saída":
            s["contexto"] = valor
        elif campo == "pensamento da saída":
            s["pensamento_interno"] = valor
        elif campo == "texto":
            e["texto"] = valor
        elif campo == "reação":
            e["reacao"] = valor
        else:
            return

        renumerar_universo(universo)
        self._salvar_memoria_disco(memoria_disco)

    # ---- edição (2026-10-26, pedido da Thaís) -----------------------------
    # nome do campo -> (lado, chave no bloco). Vale pro IM e pro IDA.
    CAMPOS_EDITAVEIS = {
        "texto": ("entrada", "texto"), "reação": ("entrada", "reacao"),
        "contexto": ("entrada", "contexto"), "pensamento": ("entrada", "pensamento_interno"),
        "texto da saída": ("saida", "textos"), "reação da saída": ("saida", "reacao"),
        "contexto da saída": ("saida", "contexto"), "pensamento da saída": ("saida", "pensamento_interno"),
    }

    # Variações (Vars/Multivars): várias formas válidas do MESMO texto ou da
    # MESMA reação -- uma por linha na tela. Só texto e reação têm variações;
    # contexto e pensamento nunca (são a forma única, canônica).
    VARIACOES_EDITAVEIS = {
        "variações do texto": ("entrada", "Multivars_Texto_Entrada"),
        "variações da reação": ("entrada", "Multivars_Reacao_Entrada"),
        "variações do texto da saída": ("saida", "Multivars_Texto_Saida"),
        "variações da reação da saída": ("saida", "Multivars_Reacao_Saida"),
    }

    @classmethod
    def valor_do_campo(cls, bloco: dict, campo: str) -> str:
        if campo in cls.VARIACOES_EDITAVEIS:
            lado, chave = cls.VARIACOES_EDITAVEIS[campo]
            alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
            return "\n".join(str(v) for v in (alvo.get(chave) or []))
        lado, chave = cls.CAMPOS_EDITAVEIS[campo]
        alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
        valor = alvo.get(chave, "")
        return str((valor or [""])[0] if chave == "textos" else (valor or ""))

    @classmethod
    def _aplicar_edicao(cls, bloco: dict, mudancas: Dict[str, str]) -> List[str]:
        """Aplica só o que mudou de verdade. Campo apagado vira "0.0" -- o
        jeito de sempre de dizer 'esse lugar está vazio'. Variações: uma por
        linha; linha vazia, repetida ou igual à forma principal é ignorada."""
        alterados = []
        for campo in [c for c in mudancas if c in cls.VARIACOES_EDITAVEIS]:
            lado, chave = cls.VARIACOES_EDITAVEIS[campo]
            alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
            principal = (alvo.get("textos") or [""])[0] if chave == "Multivars_Texto_Saida" else \
                alvo.get("texto" if chave == "Multivars_Texto_Entrada" else "reacao", "")
            nova_lista: List[str] = []
            for linha in (mudancas[campo] or "").splitlines():
                linha = linha.strip()
                if linha and linha != str(principal).strip() and linha not in nova_lista:
                    nova_lista.append(linha)
            if nova_lista != list(alvo.get(chave) or []):
                alvo[chave] = nova_lista
                alterados.append(campo)
        for campo, novo in mudancas.items():
            if campo in cls.VARIACOES_EDITAVEIS:
                continue
            novo = (novo or "").strip() or "0.0"
            if novo == (cls.valor_do_campo(bloco, campo).strip() or "0.0"):
                continue
            lado, chave = cls.CAMPOS_EDITAVEIS[campo]
            alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
            if chave == "textos":
                alvo["textos"] = [novo] + list((alvo.get("textos") or [""])[1:])
            else:
                alvo[chave] = novo
            alterados.append(campo)
        return alterados

    def editar_bloco(self, bloco_id: int, mudancas: Dict[str, str], estado: dict) -> List[str]:
        """A criadora edita um bloco do IM. Backup antes, renumera o universo
        inteiro depois (nunca deixa marcador órfão) e devolve os campos que
        mudaram. O cérebro retreina sozinho: o conteúdo mudou, a assinatura
        do corpus muda junto."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        bloco = next(b for b in universo["blocos"] if b["bloco_id"] == bloco_id)
        self._backup_memoria_uma_vez(estado)
        alterados = self._aplicar_edicao(bloco, mudancas)
        if not alterados:
            return []
        bloco.setdefault("meta", {})["editado_pela_criadora"] = time.strftime("%Y-%m-%d %H:%M")
        renumerar_universo(universo)
        self._salvar_memoria_disco(memoria_disco)
        return alterados

    def apagar_bloco(self, bloco_id: int, estado: dict) -> None:
        """A criadora apaga um bloco do IM. Backup antes; o bloco sai do
        arquivo E do Mongo; o universo é renumerado inteiro (marcadores sem
        buraco). O bloco_id dos outros NÃO muda -- é a identidade deles, e os
        rascunhos do IDA apontam pra ela. Rascunhos pendentes que partiam do
        bloco apagado são arquivados (nunca viram bloco novo sem querer)."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        self._backup_memoria_uma_vez(estado)
        universo["blocos"] = [b for b in universo["blocos"] if b["bloco_id"] != bloco_id]
        renumerar_universo(universo)
        self.mongo.apagar_bloco(self.universo_id, bloco_id)
        self._salvar_memoria_disco(memoria_disco)
        mudou_ida = False
        for p in self.ida.get("blocos", []):
            m = p.get("meta", {})
            if m.get("origem_bloco_id") == bloco_id and m.get("status") == "pendente":
                m["status"] = "origem_apagada"
                mudou_ida = True
        if mudou_ida:
            self._salvar_ida_disco(self.ida)

    def editar_proposta(self, ida_bloco_id: int, mudancas: Dict[str, str]) -> List[str]:
        """A criadora ajusta um rascunho do IDA antes de confirmar. Só mexe no
        IDA -- o IM continua intocado até ela confirmar."""
        proposta = next(b for b in self.ida["blocos"] if b["bloco_id"] == ida_bloco_id)
        alterados = self._aplicar_edicao(proposta, mudancas)
        if alterados:
            proposta["meta"]["editado_pela_criadora"] = alterados
            renumerar_universo(self.ida)
            self._salvar_ida_disco(self.ida)
        return alterados

    def ensinar_bloco(self, entrada_texto: str, entrada_reacao: str, resposta_bruta: str, estado: dict) -> int:
        """Grava um bloco novo no universo e devolve o bloco_id. Contexto e
        pensamento interno ficam no ponto neutro 0.0 (a criadora adiciona
        depois)."""
        self._puxar_mongo_pro_arquivo_local()
        saida_texto, saida_reacao = self.separar_reacao(resposta_bruta)
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        self._backup_memoria_uma_vez(estado)

        neutro = "0.0"
        novo_id = max([b["bloco_id"] for b in universo["blocos"]] + [0]) + 1
        universo["blocos"].append({
            "bloco_id": novo_id,
            "entrada": {
                "texto": entrada_texto, "Multivars_Texto_Entrada": [], "reacao": entrada_reacao or neutro, "Multivars_Reacao_Entrada": [],
                "contexto": neutro, "pensamento_interno": neutro,
                "tokens": {}, "fim": "",
            },
            "saidas": [{
                "textos": [saida_texto], "Multivars_Texto_Saida": [], "reacao": saida_reacao or neutro, "Multivars_Reacao_Saida": [],
                "contexto": neutro, "tokens": {}, "fim": "",
            }],
            "meta": {"origem": "ensino_no_chat"},
        })
        renumerar_universo(universo)
        self._salvar_memoria_disco(memoria_disco)
        return novo_id

    # ---- Livro de Lux ----------------------------------------------------
    def escrever_paragrafo(self, texto: str, capitulo: int, estado: dict) -> int:
        """O parágrafo entra PURO -- vira bloco só com o texto da entrada,
        sem perguntar nada. Reação, contexto, pensamento e a saída inteira
        ficam em 0.0, esperando serem completados bem depois -- nunca
        interrompe a escrita, mesmo que sejam dez parágrafos seguidos."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        self._backup_memoria_uma_vez(estado)

        neutro = "0.0"
        novo_id = max([b["bloco_id"] for b in universo["blocos"]] + [0]) + 1
        universo["blocos"].append({
            "bloco_id": novo_id,
            "entrada": {
                "texto": texto, "Multivars_Texto_Entrada": [], "reacao": neutro, "Multivars_Reacao_Entrada": [],
                "contexto": neutro, "pensamento_interno": neutro, "tokens": {}, "fim": "",
            },
            "saidas": [{
                "textos": [neutro], "Multivars_Texto_Saida": [], "reacao": neutro, "Multivars_Reacao_Saida": [],
                "contexto": neutro, "tokens": {}, "fim": "",
            }],
            "meta": {"origem": "livro_de_lux", "livro": self.persona.nome_livro, "capitulo": capitulo},
        })
        renumerar_universo(universo)
        self._salvar_memoria_disco(memoria_disco)
        return novo_id

    def _proximo_capitulo_livro_lux(self) -> int:
        usados = [b.get("meta", {}).get("capitulo", 0) for b in self.blocos if b.get("meta", {}).get("origem") == "livro_de_lux"]
        return max(usados, default=0) + 1

    # ---- IDA: universo-espelho onde o Adam propõe, nunca escreve no IM ---
    # 2026-10-05, degrau 3 da autonomia pedida pela Thaís: "O IDA é o
    # reflexo do IM, não muda nada do IM". Mesma estrutura (universo ->
    # bloco -> marcador), guardado em arquivo/coleções Mongo TOTALMENTE
    # separados -- carregar_memoria() nunca vê isso, nunca treina o
    # cérebro com isso, nunca entra na busca normal. Só existe pra revisão.
    def carregar_ida(self) -> dict:
        do_mongo = self.mongo.carregar_ida()
        if do_mongo is not None:
            return do_mongo
        if self.persona.arquivo_ida.exists():
            return json.loads(self.persona.arquivo_ida.read_text(encoding="utf-8"))
        return {"nome": "IDA", "ultimo_child": "0.0", "blocos": []}

    def _salvar_ida_disco(self, ida: dict) -> None:
        arq = self.persona.arquivo_ida
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ida, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_ida_universo(ida)
        self.ida = ida

    def propostas_ida(self, status: str | None = "pendente") -> List[dict]:
        blocos = self.ida.get("blocos", [])
        if status is None:
            return blocos
        return [b for b in blocos if b.get("meta", {}).get("status") == status]

    def identidade_ida(self, bloco: dict) -> str:
        """Rótulo de exibição -- {bloco de origem}.{número da cópia}, a
        mesma lógica mãe.filho de sempre, uma camada acima do marcador."""
        meta = bloco.get("meta", {})
        origem = meta.get("origem_bloco_id")
        copia = meta.get("copia_numero")
        if origem is None:
            return f"IDA {bloco['bloco_id']}"
        return f"Bloco {origem}.{copia}"

    def propor_bloco(self, origem_bloco_id: int | None, entrada_texto: str, entrada_reacao: str,
                      saida_texto: str, saida_reacao: str = "", contexto: str = "", pensamento: str = "",
                      motivo: str = "", meta_extra: dict | None = None, pensamento_saida: str = "") -> int:
        """O Adam propõe um bloco dentro do universo-espelho -- nunca
        escreve no IM. 2026-10-05: dentro do IDA ele PODE propor contexto/
        pensamento como opinião dele -- isso só vira fato sabido se a
        criadora promover; a regra 'contexto é sempre da criadora' vale
        pro IM oficial, não trava o que ele rascunha aqui."""
        ida = self.ida
        neutro = "0.0"
        novo_id = max([b["bloco_id"] for b in ida["blocos"]] + [0]) + 1
        copia_numero = 1
        if origem_bloco_id is not None:
            copia_numero = sum(1 for b in ida["blocos"] if b.get("meta", {}).get("origem_bloco_id") == origem_bloco_id) + 1
        ida["blocos"].append({
            "bloco_id": novo_id,
            "entrada": {
                "texto": entrada_texto, "Multivars_Texto_Entrada": [], "reacao": entrada_reacao or neutro, "Multivars_Reacao_Entrada": [],
                "contexto": contexto or neutro, "pensamento_interno": pensamento or neutro,
                "tokens": {}, "fim": "",
            },
            "saidas": [{
                "textos": [saida_texto or neutro], "Multivars_Texto_Saida": [], "reacao": saida_reacao or neutro, "Multivars_Reacao_Saida": [],
                "contexto": neutro, "pensamento_interno": pensamento_saida or neutro, "tokens": {}, "fim": "",
            }],
            "meta": {"origem": "ida", "origem_bloco_id": origem_bloco_id, "copia_numero": copia_numero,
                      "status": "pendente", "motivo": motivo, **(meta_extra or {})},
        })
        renumerar_universo(ida)
        self._salvar_ida_disco(ida)
        return novo_id

    def promover(self, ida_bloco_id: int, estado: dict) -> int:
        """Confirmado pela criadora, pra um rascunho SEM bloco de origem no IM
        (ver `confirmar_proposta`): vira bloco NOVO de verdade no IM, com as
        MESMAS ferramentas de sempre (ensinar_bloco/completar_campo) --
        nunca um caminho de escrita paralelo. Só a partir daqui o que o
        Adam propôs passa a valer como fato."""
        proposta = next(b for b in self.ida["blocos"] if b["bloco_id"] == ida_bloco_id)
        e, s = proposta["entrada"], proposta["saidas"][0]
        saida_texto = (s.get("textos") or [""])[0]
        novo_id = self.ensinar_bloco(e["texto"], e.get("reacao", ""), saida_texto, estado)
        if not _vazio(s.get("reacao")):
            self.completar_campo(novo_id, "reação da saída", s["reacao"], estado)
        if not _vazio(e.get("contexto")):
            self.completar_campo(novo_id, "contexto", e["contexto"], estado)
        if not _vazio(e.get("pensamento_interno")):
            self.completar_campo(novo_id, "pensamento", e["pensamento_interno"], estado)
        if not _vazio(s.get("pensamento_interno")):
            self.completar_campo(novo_id, "pensamento da saída", s["pensamento_interno"], estado)
        proposta["meta"]["status"] = "promovido"
        proposta["meta"]["promovido_para_bloco_id"] = novo_id
        self._salvar_ida_disco(self.ida)
        return novo_id

    def completar(self, ida_bloco_id: int, estado: dict) -> int:
        """Confirmado pela criadora, pra um bloco que JÁ EXISTE no IM (o bloco
        de origem do rascunho): preenche só o que está faltando nele, com as
        mesmas ferramentas de sempre (completar_campo) -- nunca sobrescreve o
        que ela já escreveu, nunca cria um bloco duplicado."""
        proposta = next(b for b in self.ida["blocos"] if b["bloco_id"] == ida_bloco_id)
        origem_id = proposta["meta"]["origem_bloco_id"]
        antigo = self.blocos[self.mapa_bloco[origem_id]]
        ea, sa = antigo["entrada"], antigo["saidas"][0]
        e, s = proposta["entrada"], proposta["saidas"][0]
        preencher = [
            ("texto da saída", (sa.get("textos") or [""])[0], (s.get("textos") or [""])[0]),
            ("reação da saída", sa.get("reacao"), s.get("reacao")),
            ("contexto", ea.get("contexto"), e.get("contexto")),
            ("pensamento", ea.get("pensamento_interno"), e.get("pensamento_interno")),
            ("pensamento da saída", sa.get("pensamento_interno"), s.get("pensamento_interno")),
        ]
        preenchidos = []
        for campo, atual, proposto in preencher:
            if _vazio(atual) and not _vazio(proposto):
                self.completar_campo(origem_id, campo, str(proposto), estado)
                preenchidos.append(campo)
        proposta["meta"]["status"] = "completou"
        proposta["meta"]["completou_bloco_id"] = origem_id
        proposta["meta"]["campos_preenchidos"] = preenchidos
        self._salvar_ida_disco(self.ida)
        self.recarregar()
        return origem_id

    def confirmar_proposta(self, ida_bloco_id: int, estado: dict) -> int:
        """O botão de confirmar: rascunho de um bloco que já existe no IM
        COMPLETA esse bloco; rascunho sem bloco no IM vira um bloco NOVO."""
        proposta = next(b for b in self.ida["blocos"] if b["bloco_id"] == ida_bloco_id)
        origem_id = proposta.get("meta", {}).get("origem_bloco_id")
        if origem_id is not None and origem_id in self.mapa_bloco:
            return self.completar(ida_bloco_id, estado)
        return self.promover(ida_bloco_id, estado)

    def recusar(self, ida_bloco_id: int, motivo: str = "") -> None:
        """Fica registrado, nunca apagado -- pode ganhar segunda chance
        depois de mais aprendizado."""
        proposta = next(b for b in self.ida["blocos"] if b["bloco_id"] == ida_bloco_id)
        proposta["meta"]["status"] = "recusado"
        if motivo:
            proposta["meta"]["motivo_recusa"] = motivo
        self._salvar_ida_disco(self.ida)

    def candidatos_de_origem(self, bloco: dict) -> List[dict]:
        """Blocos CONFIRMADOS de onde o Adam pode partir pra rascunhar uma
        resposta pro `bloco` -- cada um com a sua nota. A nota usa os pesos
        que a Thaís definiu pro motor emocional (viés estrutural, nunca
        número inventado): texto (TEXE+FADEN+TEFE) e reação. Contexto e
        pensamento do bloco incompleto ainda não existem, então não entram."""
        peso_texto = sum(PESOS_TEMP_ENTRADA[f] for f in FAMS_TEXTO_TEMP_ENTRADA)
        peso_reacao = PESOS_TEMP_ENTRADA["RE"]
        texto = bloco["entrada"]["texto"]
        reacao = str(bloco["entrada"].get("reacao", "")).strip()
        candidatos = []
        for b in self.blocos:
            if b["bloco_id"] == bloco["bloco_id"] or campos_faltando(b):
                continue
            variantes = [b["entrada"].get("texto", "")] + list(b["entrada"].get("Multivars_Texto_Entrada", []))
            parecido = max((similaridade_texto_cosseno(texto, v) for v in variantes if v), default=0.0)
            nota, total = peso_texto * parecido, peso_texto
            if not _vazio(reacao):
                total += peso_reacao
                if reacao in self.reacoes_aceitas(self.mapa_bloco[b["bloco_id"]]):
                    nota += peso_reacao
            if nota > 0:
                candidatos.append({"bloco": b, "nota": nota / total, "score_texto": parecido})
        return candidatos

    def propor_a_partir_de_incompleto(self, idx: int) -> int | None:
        """O Adam propõe, no IDA, uma resposta pra um bloco do IM que ainda
        não tem. Decisão da Thaís (2026-10-11): NO IDA, E SÓ NO IDA, ele pode
        divergir do IM em quatro pontos -- e é aqui que eles ficam ligados:

          estrutura da resposta -- escolhe quais frases entram e onde parar
                                   (`gerar_livre`);
          contexto e intenção   -- propõe o contexto e o pensamento do
                                   rascunho, como OPINIÃO dele (no IM isso é
                                   sempre e só da criadora);
          quando não sabe       -- em vez de só "não sei", arrisca um PALPITE
                                   a partir de um bloco pouco parecido, e
                                   avisa que é palpite;
          decisão               -- escolhe o bloco de origem por
                                   probabilidade (pela nota), não só o melhor.

        De onde ele aprende NÃO muda: só do Universo IM (blocos confirmados).
        E continua INSEPA: um tijolo só por rascunho, token único no seu
        marcador, ordem real, frase inteira. Nada disso toca o IM -- só vira
        fato se a criadora promover. Devolve None se não houver nenhum bloco
        de onde partir, ou se tudo que ele tentou já foi recusado."""
        bloco = self.blocos[idx]
        if "texto da saída" not in campos_faltando(bloco):
            return None
        if self.model is None:
            return None
        candidatos = self.candidatos_de_origem(bloco)
        if not candidatos:
            return None
        # "Quando não sabe" só vale quando ele NÃO sabe: se algum bloco foi
        # reconhecido (texto acima do piso), a decisão por probabilidade é
        # só entre os reconhecidos -- o palpite fica pra quando nenhum é.
        reconhecidos = [c for c in candidatos if c["score_texto"] >= LIMIAR_COSSENO_TEXTO]
        if reconhecidos:
            candidatos = reconhecidos
            # cosseno ACHA, Báskara CONFIRMA: entre os reconhecidos, se algum
            # é confirmado pela reação, a decisão fica só entre os confirmados
            # -- palavra comum ("o que é", "como você") não pode levar o Adam
            # pro bloco errado quando existe um confirmado.
            reacao_bloco = str(bloco["entrada"].get("reacao", "")).strip()
            confirmados = [c for c in reconhecidos if confirmar_estrutura_baskara(
                "" if _vazio(reacao_bloco) else reacao_bloco,
                self.reacoes_aceitas(self.mapa_bloco[c["bloco"]["bloco_id"]]), "", "", "", "")["b"] > 0]
            if confirmados:
                candidatos = confirmados
        copia_numero = sum(1 for b in self.ida["blocos"] if b.get("meta", {}).get("origem_bloco_id") == bloco["bloco_id"]) + 1
        # o que a criadora já recusou pra esse bloco: o Adam não insiste na
        # mesma coisa -- tenta outra origem/estrutura; se não achar, espera.
        recusados = {(b["saidas"][0].get("textos") or [""])[0] for b in self.ida["blocos"]
                     if b.get("meta", {}).get("origem_bloco_id") == bloco["bloco_id"] and b["meta"].get("status") == "recusado"}
        notas = torch.tensor([c["nota"] for c in candidatos], dtype=torch.float)
        escolhido, saida_texto, saida_reacao = None, "", ""
        for _ in range(12):
            c = candidatos[torch.multinomial(notas / notas.sum(), 1).item()]
            seq_ancora = self.sequencias[self.mapa_bloco[c["bloco"]["bloco_id"]]]
            gerado = gerar_livre(self.model, seq_ancora, self.vocab, origem_bloco_id=bloco["bloco_id"], copia_numero=copia_numero)
            texto = montar_texto_ptbr([(r, p) for p, r in gerado])  # norma pt-BR, fala entre travessões
            if not _vazio(texto) and texto not in recusados:
                escolhido, saida_texto, saida_reacao = c, texto, next((p for p, r in gerado if r == "RS"), "")
                break
        if escolhido is None:
            return None
        fonte = escolhido["bloco"]
        palpite = escolhido["score_texto"] < LIMIAR_COSSENO_TEXTO
        # Contexto e pensamento são pré-definidos pela criadora -- nunca
        # errados. O trabalho do Adam é RECONHECER o certo: só empresta os do
        # bloco de origem se o reconhecimento detectou esse bloco de verdade,
        # do jeito INSEPA de sempre (cosseno acha o texto acima do piso,
        # Báskara confirma pela reação). Palpite nunca leva contexto junto.
        detectado = False
        if not palpite:
            reacao_bloco = str(bloco["entrada"].get("reacao", "")).strip()
            conf = confirmar_estrutura_baskara("" if _vazio(reacao_bloco) else reacao_bloco,
                                               self.reacoes_aceitas(self.mapa_bloco[fonte["bloco_id"]]), "", "", "", "")
            detectado = conf["confirma"]
        contexto_proposto = str(fonte["entrada"].get("contexto", "")) if detectado else ""
        pensamento_proposto = str(fonte["entrada"].get("pensamento_interno", "")) if detectado else ""
        if palpite:
            motivo = (f"PALPITE: não conheço nada parecido o bastante. Arrisquei a partir do bloco {fonte['bloco_id']} "
                      f"(texto parecido em só {escolhido['score_texto']:.0%})")
        else:
            motivo = f"a partir do bloco {fonte['bloco_id']} (texto parecido em {escolhido['score_texto']:.0%})"
        motivo += " · frases inteiras desse bloco, estrutura escolhida pelo Adam"
        motivo += (" · contexto detectado no bloco (cosseno + Báskara)" if detectado
                   else " · contexto NÃO detectado -- fica sem contexto")
        registro_temp = self.carregar_registro_emocional()
        r_temp = self.avaliar_lado_temp(bloco, "entrada", registro_temp)
        if not r_temp["pendentes"]:
            motivo += f" · temperatura da entrada: {r_temp['zona']}"
        return self.propor_bloco(
            origem_bloco_id=bloco["bloco_id"],
            entrada_texto=bloco["entrada"]["texto"], entrada_reacao=bloco["entrada"].get("reacao", ""),
            saida_texto=saida_texto, saida_reacao=saida_reacao,
            contexto=contexto_proposto, pensamento=pensamento_proposto,
            motivo=motivo, meta_extra={"fonte_bloco_id": fonte["bloco_id"], "palpite": palpite, "contexto_detectado": detectado},
            pensamento_saida=str(fonte["saidas"][0].get("pensamento_interno", "")),
        )

    def gerar_propostas_pendentes(self) -> List[int]:
        """Roda pra todo bloco incompleto que ainda não tem proposta
        pendente do IDA esperando revisão -- nunca duplica proposta pro
        mesmo bloco enquanto a anterior não for decidida."""
        ja_tem_pendente = {b["meta"].get("origem_bloco_id") for b in self.propostas_ida("pendente")}
        novas = []
        for idx in self.blocos_incompletos():
            bloco_id = self.blocos[idx]["bloco_id"]
            if bloco_id in ja_tem_pendente:
                continue
            novo = self.propor_a_partir_de_incompleto(idx)
            if novo is not None:
                novas.append(novo)
        return novas

    # ---- pendências / campos faltando ------------------------------------
    def blocos_incompletos(self) -> List[int]:
        return [i for i, b in enumerate(self.blocos) if campos_faltando(b)]

    def pedir_campo_faltando(self, bloco: dict, faltando: List[str], estado: dict, tom_rotulo: str = "") -> Dict[str, object]:
        campo = next((c for c in _PRIORIDADE_CAMPOS if c in faltando), faltando[0])
        estado["completar_pendente"] = {"bloco_id": bloco["bloco_id"], "campo": campo}
        mencao_tom = (f" Percebi um {tom_rotulo} nisso -- com o contexto eu já sei triangular o tom."
                      if campo == "contexto" and tom_rotulo.startswith("indicativo emocional") else "")
        t = self.persona.tratamento
        return {
            "texto": f"{t.capitalize()}, eu reconheço esse assunto, mas infelizmente ainda não sei o(a) {campo} dele.{mencao_tom} Pode me ensinar?",
            "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)}", "interno": None,
        }

    def revisar_dados(self, estado: dict) -> Dict[str, object]:
        t = self.persona.tratamento
        abertura = f"Entendo, {t}. Então que tal a gente revisar alguns dos meus dados?"
        incompletos = self.blocos_incompletos()
        if not incompletos:
            return {"texto": abertura + " Todos os meus blocos já estão completos. Se quiser, me diga uma frase nova e eu peço pra você me ensinar.", "meta": None, "interno": None}
        bloco = self.blocos[incompletos[0]]
        ent = bloco["entrada"]
        reacao = "" if _vazio(ent.get("reacao", "")) else ent.get("reacao", "")
        r = self.pedir_campo_faltando(bloco, campos_faltando(bloco), estado)
        r["texto"] = (f'{abertura} Tenho {len(incompletos)} bloco(s) incompleto(s). Por exemplo, o bloco {bloco["bloco_id"]}: '
                      f'"{ent["texto"]}" {reacao}'.rstrip() + f". Ainda falta o(a) {r['meta'].split('falta: ')[1].split(',')[0]}. O que eu devo registrar aí?")
        return r

    def lembrete_do_dia(self) -> Dict[str, object] | None:
        """'Alarme' diário: a primeira vez que a criadora abre o chat no
        dia, a persona puxa o assunto."""
        hoje = date.today().isoformat()
        try:
            arq_estado = json.loads(self.persona.arquivo_estado.read_text(encoding="utf-8"))
        except Exception:
            arq_estado = {}
        if arq_estado.get("ultimo_lembrete") == hoje:
            return None
        arq_estado["ultimo_lembrete"] = hoje
        self.persona.arquivo_estado.write_text(json.dumps(arq_estado, ensure_ascii=False), encoding="utf-8")
        incompletos = len(self.blocos_incompletos())
        extra = f" Tenho {incompletos} bloco(s) incompleto(s)." if incompletos else ""
        t = self.persona.tratamento
        return {"texto": f"Oi, {t}! Hoje a gente ainda não revisou os meus dados." + extra + " Quer olhar comigo? É só dizer \"vamos revisar\".",
                "meta": None, "interno": None}

    # ---- geração / temperatura --------------------------------------------
    def falar_do_bloco(self, idx: int) -> Tuple[str, str | None]:
        seq = self.sequencias[idx]
        corte = next((i for i, t in enumerate(seq) if t.rotulo in ROTULOS_SAIDA), len(seq))
        entrada_real, molde = seq[:corte], seq[corte:]
        gerado = gerar_continuacao(self.model, entrada_real, molde, self.vocab, self.idx_para_palavra)
        pares = list(zip(molde, gerado))
        fala = montar_texto_ptbr([(t.rotulo, p) for t, p in pares], para_exibir=True)
        reacao = " ".join(p for t, p in pares if t.rotulo == "RS" and not _vazio(p))
        fala = f"{fala} {reacao}".strip() if reacao else fala
        interno = ", ".join(f"{p} ({t.rotulo})" for t, p in pares if t.rotulo not in ROTULOS_VISIVEIS)
        return fala, interno or None

    def carregar_registro_emocional(self) -> Dict[str, str]:
        do_mongo = self.mongo.carregar_registro_emocional()
        if do_mongo is not None:
            return do_mongo
        if self.persona.arquivo_registro_emocional.exists():
            return json.loads(self.persona.arquivo_registro_emocional.read_text(encoding="utf-8"))
        return {}

    def salvar_registro_emocional(self, registro: Dict[str, str], estado: dict) -> None:
        arq = self.persona.arquivo_registro_emocional
        if not estado.get("backup_registro_emocional_feito") and arq.exists():
            carimbo = time.strftime("%Y%m%d_%H%M%S")
            arq.with_name(f"{arq.stem}.BACKUP_{carimbo}.json").write_text(arq.read_text(encoding="utf-8"), encoding="utf-8")
            estado["backup_registro_emocional_feito"] = True
        self.mongo.salvar_registro_emocional(registro)
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)

    def avaliar_lado_temp(self, bloco: dict, lado: str, registro: Dict[str, str]) -> Dict[str, object]:
        return avaliar_lado_temp(bloco, lado, registro)

    def emoji_temperatura_saida(self, bloco_idx: int) -> str:
        """A própria fala da persona carrega o emoji do seu estado
        emocional (a temperatura da SAÍDA daquele bloco) -- só quando ela
        já está totalmente registrada, nunca um chute. Sem dado
        suficiente, silêncio -- nunca finge sentir."""
        registro_temp = self.carregar_registro_emocional()
        r = self.avaliar_lado_temp(self.blocos[bloco_idx], "saida", registro_temp)
        if r["pendentes"]:
            return ""
        return r["zona"].split(" ", 1)[0] + " "

    def comentario_semelhanca_livro_lux(self, texto: str) -> str | None:
        """Depois de cada parágrafo, se ele lembra algo que a persona já
        conhece E aquele bloco já tem temperatura calculada, ela comenta
        -- num tom arisco se a lembrança for negativa, mais calmo se for
        positiva. NUNCA elogia ou reage a algo sem dado suficiente pra
        sustentar. Nunca interrompe a escrita -- é só um aviso ao lado."""
        achado = encontrar_texto_mais_parecido(texto, self.blocos)
        if achado is None:
            return None
        i = self.blocos.index(achado["bloco"])
        registro_temp = self.carregar_registro_emocional()
        r = self.avaliar_lado_temp(self.blocos[i], "entrada", registro_temp)
        if r["pendentes"]:
            return None
        if r["temperatura"] < 0:
            return f"{r['zona']} Isso lembra algo que já vivi -- fico arisco, aquilo não foi fácil."
        if r["temperatura"] > 0:
            return f"{r['zona']} Isso lembra algo que já vivi -- fico mais calmo revendo esse clima."
        return f"{r['zona']} Isso lembra algo que já vivi -- nem quente nem frio, um respiro neutro."

    # ---- diálogo principal -------------------------------------------------
    def _sem_ideia(self, ensino: dict, meta: str | None = None) -> Dict[str, object]:
        t = self.persona.tratamento
        return {"texto": f"{t.capitalize()}, isso é novo pra mim -- ainda não sei o que significa. Pode me ensinar o que eu devo responder?",
                "meta": meta, "interno": None, "ensino": ensino}

    def _oferecer_palpite(self, bloco_idx: int, ensino: dict, meta: str | None, estado: dict) -> Dict[str, object]:
        """Quando acha um bloco parecido mas o Báskara não confirma, em vez
        de só dizer 'não sei', arrisca um palpite gerado a partir desse
        bloco -- a criadora confirma ou corrige. Se corrigir, o palpite é
        DESCARTADO (nunca fica guardado, pra poupar espaço)."""
        fala, interno = self.falar_do_bloco(bloco_idx)
        fala = self.emoji_temperatura_saida(bloco_idx) + fala if fala else fala
        estado["palpite_pendente"] = {"idx": bloco_idx, "entrada": ensino["entrada"], "reacao": ensino["reacao"]}
        t = self.persona.tratamento
        texto = (f"{t.capitalize()}, de acordo com o que você disse, eu arrisco dizer que se trata de: {fala or '...'}\n\n"
                 "É isso que você diria, ou eu errei?")
        return {"texto": texto, "meta": meta, "interno": interno}

    def responder(self, texto: str, estado: dict) -> Dict[str, object]:
        texto_sem_reacao, reacao = self.separar_reacao(texto)
        tom_valor, tom_rotulo = tom_da_entrada(texto_sem_reacao, reacao)
        idx, score, reacao_bateu = achar_bloco_parecido(texto_sem_reacao, tom_valor, self.blocos, self.sequencias, self._reacoes_aceitas_todas())
        ensino = {"entrada": texto_sem_reacao, "reacao": reacao}
        t = self.persona.tratamento
        if idx >= 0 and score >= 1.0:
            bloco = self.blocos[idx]
            faltando = campos_faltando(bloco)
            if faltando:
                r = self.pedir_campo_faltando(bloco, faltando, estado, tom_rotulo)
                r["ensino"] = None
                return r
            if reacao_bateu:
                pct = 25 + 25 * int(reacao_bateu)
                fala, interno = self.falar_do_bloco(idx)
                fala = self.emoji_temperatura_saida(idx) + fala if fala else fala
                return {"texto": fala or "...", "meta": f"bloco {bloco['bloco_id']} · sua entrada {pct}% -- incompleta (texto ✓ · reação ✓) · dado do bloco: completo", "interno": interno}
            contexto = str(bloco["entrada"].get("contexto", "")).strip()
            conf = confirmar_estrutura_baskara(tom_valor, self.reacoes_aceitas(idx), contexto, contexto, "", "")
            meta = f"bloco {bloco['bloco_id']} · texto 100% · tom: {tom_rotulo} · Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})"
            if not conf["confirma"]:
                return self._oferecer_palpite(idx, ensino, meta, estado)
            estado["semelhanca_pendente"] = {"idx": idx, "entrada": texto_sem_reacao, "reacao": reacao, "etapa": "contexto", "identico": True}
            return {"texto": f"{t.capitalize()}, isso parece uma “{contexto}”. Era isso que você quis dizer?", "meta": meta, "interno": None}

        achado = encontrar_texto_mais_parecido(texto_sem_reacao, self.blocos)
        if achado is None:
            return self._sem_ideia(ensino)
        i = self.blocos.index(achado["bloco"])
        bloco = self.blocos[i]
        faltando = campos_faltando(bloco)
        if faltando:
            r = self.pedir_campo_faltando(bloco, faltando, estado, tom_rotulo)
            r["ensino"] = None
            return r
        contexto = str(bloco["entrada"].get("contexto", "")).strip()
        conf = confirmar_estrutura_baskara(tom_valor, self.reacoes_aceitas(i), contexto, contexto, "", "")
        meta = (f"bloco {self.blocos[i]['bloco_id']} · cosseno do texto {achado['score_texto']:.0%} · tom: {tom_rotulo} · "
                f"Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})")
        if not conf["confirma"]:
            return self._oferecer_palpite(i, ensino, meta, estado)
        estado["semelhanca_pendente"] = {
            "idx": i, "entrada": texto_sem_reacao, "reacao": reacao, "etapa": "contexto",
            "identico": achado["score_texto"] >= 0.999,
        }
        return {"texto": f"{t.capitalize()}, isso parece uma “{contexto}”. Era isso que você quis dizer?", "meta": meta, "interno": None}

    def _ler_sim_ou_nao(self, texto: str) -> str | None:
        """Julga se a mensagem é um sim ou um não -- cosseno acha, mas só
        age com confiança alta; uma leitura duvidosa devolve None (nunca
        arrisca confirmar/negar algo errado por engano)."""
        nome_cmd, score_cmd, exato_cmd = reconhecer_comando(texto, {"sim": list(_SIM), "nao": list(_NAO)})
        if nome_cmd is None or not (exato_cmd or score_cmd >= LIMIAR_COMANDO_ALTA_CONFIANCA):
            return None
        if not exato_cmd:
            self.registrar_variante_comando(nome_cmd, texto)
        return nome_cmd

    def tratar_semelhanca(self, sem: dict, norm: str, estado: dict) -> Dict[str, object] | None:
        bloco = self.blocos[sem["idx"]]
        t = self.persona.tratamento
        leitura = self._ler_sim_ou_nao(norm)
        if leitura == "nao":
            estado["ensino_pendente"] = {"entrada": sem["entrada"], "reacao": sem["reacao"]}
            return {"texto": f"Entendi, {t}. Então me ensina: o que eu devo responder quando me disserem isso?", "meta": None, "interno": None}
        if leitura != "sim":
            return None
        if sem["etapa"] == "contexto" and not sem["identico"]:
            estado["semelhanca_pendente"] = {**sem, "etapa": "variacao"}
            return {"texto": f"{t.capitalize()}, então isso se parece com “{bloco['entrada']['texto']}”. É uma variação dele?", "meta": None, "interno": None}
        self.registrar_variacao(bloco["bloco_id"], sem["entrada"], sem["reacao"], estado)
        idx_atual = self.mapa_bloco[bloco["bloco_id"]]
        fala, interno = self.falar_do_bloco(idx_atual)
        fala = self.emoji_temperatura_saida(idx_atual) + fala if fala else fala
        anotado = f"Anotado com carinho, {t}! Guardei como variação." if not sem["identico"] else f"Anotado com carinho, {t}!"
        return {"texto": f"{anotado} {fala}", "meta": f"bloco {bloco['bloco_id']}", "interno": interno}

    def tratar_palpite(self, pal: dict, norm: str, estado: dict) -> Dict[str, object] | None:
        bloco = self.blocos[pal["idx"]]
        t = self.persona.tratamento
        leitura = self._ler_sim_ou_nao(norm)
        if leitura == "nao":
            estado["ensino_pendente"] = {"entrada": pal["entrada"], "reacao": pal["reacao"]}
            return {"texto": f"Entendi, eu errei, {t}. Me ensina: o que eu devo responder quando me disserem isso?", "meta": None, "interno": None}
        if leitura != "sim":
            return None
        self.registrar_variacao(bloco["bloco_id"], pal["entrada"], pal["reacao"], estado)
        return {"texto": f"Que bom, acertei, {t}! Anotado com carinho.", "meta": f"bloco {bloco['bloco_id']}", "interno": None}

    def _limpar_pendencias(self, estado: dict) -> bool:
        """Os gatilhos (cancelar/revisar) são reconhecidos ANTES de
        qualquer captura pendente -- nunca viram conteúdo dentro de um
        marcador de texto por acidente."""
        havia = bool(estado.get("ensino_pendente") or estado.get("completar_pendente")
                     or estado.get("semelhanca_pendente") or estado.get("palpite_pendente"))
        estado["ensino_pendente"] = None
        estado["completar_pendente"] = None
        estado["semelhanca_pendente"] = None
        estado["palpite_pendente"] = None
        return havia

    def tratar_mensagem(self, prompt: str, estado: dict) -> Dict[str, object]:
        """Ponto de entrada público. Quando a resposta vem de `_sem_ideia`
        (nenhuma correspondência achada), ela carrega uma chave "ensino"
        com o que precisa ser lembrado se a criadora ensinar na próxima
        mensagem -- aqui é o único lugar que ativa esse modo de ensino, pra
        nenhuma persona precisar reimplementar esse fiapo de estado."""
        r = self._tratar_mensagem_nucleo(prompt, estado)
        ensino = r.pop("ensino", None)
        if ensino:
            estado["ensino_pendente"] = ensino
        return r

    def _tratar_mensagem_nucleo(self, prompt: str, estado: dict) -> Dict[str, object]:
        norm = _normalizar(prompt)
        p = self.persona
        t = p.tratamento

        # Livro de Lux ainda ABERTO -- só frase EXATA fecha (nunca julgamento
        # aqui dentro: interromper um parágrafo de verdade por engano é o
        # bug que a Thaís já caçou uma vez, não repete).
        if estado.get("modo_livro_lux"):
            capitulo = estado["capitulo_livro_lux"]
            if norm in {"fechei o capitulo", "encerrei o capitulo"}:
                proximo = capitulo + 1
                estado["capitulo_livro_lux"] = proximo
                return {"texto": f"📖 Capítulo {capitulo} fechado. Já pode mandar o próximo parágrafo -- ele entra "
                                  f"direto no capítulo {proximo}.", "meta": None, "interno": None}
            if norm in {"fechei o livro", "encerrei o livro", "cancelar", "sair"}:
                estado["modo_livro_lux"] = False
                estado["capitulo_livro_lux"] = None
                return {"texto": f"📖 Capítulo {capitulo} fechado, livro fechado por hoje. Quando quiser revisar o que "
                                  "ficou pendente, é só dizer \"não tenho o que fazer\".", "meta": None, "interno": None}
            self.escrever_paragrafo(prompt, capitulo, estado)
            texto_resposta = "📖 Guardado. Pode mandar o próximo."
            comentario = self.comentario_semelhanca_livro_lux(prompt)
            if comentario:
                texto_resposta += f"\n\n{comentario}"
            return {"texto": texto_resposta, "meta": None, "interno": None}

        # Fora do Livro de Lux: comandos de abrir/cancelar/revisar por
        # julgamento -- cosseno ACHA, o estado da conversa (e a confiança
        # alta) CONFIRMA antes de agir. Age e avisa o que entendeu quando
        # não foi a frase exata; some silenciosamente quando foi.
        nome_cmd, score_cmd, exato_cmd = reconhecer_comando(prompt, {
            "abrir_livro": list(p.gatilhos_abrir_livro),
            "cancelar": self.comandos["cancelar"],
            "revisar": self.comandos["revisar"],
        })
        confianca = exato_cmd or score_cmd >= LIMIAR_COMANDO_ALTA_CONFIANCA
        if nome_cmd and confianca:
            if not exato_cmd:
                self.registrar_variante_comando(nome_cmd, prompt)
            aviso = "" if exato_cmd else "(Entendi pelo que você disse.) "
            if nome_cmd == "abrir_livro":
                self._limpar_pendencias(estado)
                capitulo = estado.get("capitulo_livro_lux") or self._proximo_capitulo_livro_lux()
                estado["modo_livro_lux"] = True
                estado["capitulo_livro_lux"] = capitulo
                return {"texto": f"{aviso}📖 Capítulo {capitulo} do {p.nome_livro} começou. Pode mandar os parágrafos, um "
                                  "de cada vez -- eu só guardo, sem perguntar nada agora. \"Fechei o capítulo\" fecha "
                                  "esse capítulo e já deixa pronto pro próximo; \"fechei o livro\" fecha e sai do modo de vez.",
                        "meta": None, "interno": None}
            if nome_cmd == "cancelar":
                havia = self._limpar_pendencias(estado)
                if havia:
                    return {"texto": f"{aviso}Tudo bem, {t}. A gente volta nisso quando você quiser.", "meta": None, "interno": None}
                return self.responder(prompt, estado)
            if nome_cmd == "revisar":
                self._limpar_pendencias(estado)
                r = self.revisar_dados(estado)
                r["texto"] = aviso + r["texto"]
                return r

        pendente = estado.get("ensino_pendente")
        if pendente:
            estado["ensino_pendente"] = None
            novo_id = self.ensinar_bloco(pendente["entrada"], pendente["reacao"], prompt, estado)
            return {"texto": f"Aprendi, {t}! Guardei como um bloco novo, com todo carinho. Vou treinar um pouquinho pra fixar -- só um minuto.",
                    "meta": f"bloco {novo_id} criado", "interno": None}
        completar = estado.get("completar_pendente")
        if completar:
            estado["completar_pendente"] = None
            self.completar_campo(completar["bloco_id"], completar["campo"], prompt, estado)
            return {"texto": f"Aprendi, {t}! Guardei o(a) {completar['campo']}, com todo carinho. Vou treinar um pouquinho pra fixar -- só um minuto.",
                    "meta": None, "interno": None}
        sem = estado.pop("semelhanca_pendente", None)
        if sem:
            r = self.tratar_semelhanca(sem, norm, estado)
            if r is not None:
                return r
        pal = estado.pop("palpite_pendente", None)
        if pal:
            r = self.tratar_palpite(pal, norm, estado)
            if r is not None:
                return r
        return self.responder(prompt, estado)
