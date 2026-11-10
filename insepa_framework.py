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
    # VARS do tratamento (2026-11-09, Thaís: "não tem sinônimo nessa fala? eu criei vars pra
    # isso, pra variar dentro do contexto"). Só ELA define -- vazio = sempre o tratamento.
    tratamento_vars: Tuple[str, ...] = ()
    mongo_db: str = "insepa"
    mongo_uri_padrao: str = "mongodb://localhost:27017"
    # False = nunca conecta em banco nenhum, só arquivo (ex.: a "prova do
    # Adam", que conversa numa cópia temporária e não pode tocar dado real).
    usar_mongo: bool = True
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
    arquivo_revisao_proprio: Path | None = None
    arquivo_sinapses_proprio: Path | None = None
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
    def arquivo_revisao(self) -> Path:
        return self.arquivo_revisao_proprio or self.pasta / f"{self.nome}_revisao.json"

    @property
    def arquivo_sinapses(self) -> Path:
        return self.arquivo_sinapses_proprio or self.pasta / f"{self.nome}_sinapses.json"

    @property
    def arquivo_rede_intencao(self) -> Path:
        return self.pasta / f"{self.nome}_rede_intencao.pt"

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


_CLIENTES_MONGO: Dict[str, object] = {}  # uma conexão por endereço, reaproveitada entre cliques


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
        if not persona.usar_mongo:
            self.erro = "Mongo desligado nesta persona"
            return
        try:
            from pymongo import MongoClient
            uri = mongo_config("MONGO_URI", persona.mongo_uri_padrao, persona)
            # 2026-10-28: o banco é SEMPRE o que a persona diz. Antes, um MONGO_DB
            # nos Secrets vencia -- e uma persona de teste (o protótipo) podia
            # acabar escrevendo no banco do Adam real. Quem quiser ler o
            # MONGO_DB dos Secrets faz isso ao criar a Persona (o persona_adam
            # do Adam real já faz).
            db_nome = persona.mongo_db
            # 2026-11-04: a conexão com a nuvem custava ~3,6 s e era aberta de
            # novo a CADA clique na tela. Agora abre uma vez por endereço e é
            # reaproveitada (o próprio MongoClient é feito pra ser compartilhado).
            client = _CLIENTES_MONGO.get(uri)
            if client is None:
                client = MongoClient(uri, serverSelectionTimeoutMS=8000)
                client.admin.command("ping")
                _CLIENTES_MONGO[uri] = client
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
            # 2026-11-07: o que a criadora ensina pela conversa espera aqui a
            # revisão dela antes de entrar no IM -- nem IM, nem IDA.
            self.col_revisao = db["revisao"]
            # o que as sinapses aprenderam com ela (nunca o IM: as ligações que
            # vêm dos blocos são lidas na hora, só a experiência fica aqui)
            self.col_sinapses = db["sinapses"]
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

    def salvar_revisao(self, fila: dict) -> None:
        if not self.enabled:
            return
        self.col_revisao.update_one({"_id": "fila"}, {"$set": {"dados": fila}}, upsert=True)

    def carregar_revisao(self) -> dict | None:
        if not self.enabled:
            return None
        doc = self.col_revisao.find_one({"_id": "fila"})
        return doc["dados"] if doc else None

    def salvar_sinapses(self, sinapses: dict) -> None:
        if not self.enabled:
            return
        self.col_sinapses.update_one({"_id": "aprendidas"}, {"$set": {"dados": sinapses}}, upsert=True)

    def carregar_sinapses(self) -> dict | None:
        if not self.enabled:
            return None
        doc = self.col_sinapses.find_one({"_id": "aprendidas"})
        return doc["dados"] if doc else None

    def salvar_rede_intencao(self, dados: bytes, assinatura: str, data: str = "") -> None:
        if not self.enabled:
            return
        self.col_sinapses.update_one({"_id": "rede_intencao"},
                                     {"$set": {"pesos": dados, "assinatura": assinatura, "data": data}}, upsert=True)

    def carregar_rede_intencao(self) -> Tuple[bytes, str, str] | None:
        if not self.enabled:
            return None
        doc = self.col_sinapses.find_one({"_id": "rede_intencao"})
        return (bytes(doc["pesos"]), doc.get("assinatura", ""), doc.get("data", "")) if doc else None

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
                 origem_bloco_id: int, copia_numero: int,
                 vizinhos: List[List[TokenDoBloco]] | None = None, formas_do_assunto: set | None = None) -> List[Tuple[str, str]]:
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
    # 2026-11-07, "criando a partir do que já tem": além das frases do bloco de
    # origem, o rascunho pode usar frases INTEIRAS de blocos vizinhos de
    # cimento (o MESMO contexto) -- só quando a origem fala sem travessão, e
    # sempre antes da reação, pra ordem dos campos nunca voltar.
    if vizinhos and all(f[0].rotulo in ("TEDSA", "RS") for f in frases):
        texto_origem = [f for f in frases if f[0].rotulo == "TEDSA"]
        reacao_origem = [f for f in frases if f[0].rotulo == "RS"]
        # só frase COMPLETA (termina em . ! ? …, nunca um pedaço antes da fala)
        # e que fale do MESMO assunto da pergunta -- mesmo contexto sozinho não basta
        extras = [f for seq in vizinhos for f in frases_do_molde([t for t in seq if t.rotulo in ROTULOS_SAIDA])
                  if f[0].rotulo == "TEDSA" and set(f[-1].palavra) <= _FECHA_FRASE
                  and formas_do_assunto and any(alnulu_string(t.palavra) in formas_do_assunto for t in f)]
        frases = texto_origem + extras + reacao_origem
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
# Motor de temperatura emocional -- quanto de emoção (e quanto de razão) há
# em cada lado do bloco, 0 a 100%; o rótulo dominante (positivo, negativo,
# neutro) diz QUAL emoção é. Ver ZONAS_TEMP.
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
# lado com frase mas sem NENHUMA emoção = 100% razão = GELO (Thaís, 2026-11-10: "Gelo é 100% razão… não há
# rastro do contraste"). Antes era 30 (Esfriando) -- mas Esfriando agora é equilíbrio (razão com entusiasmo).
FRIEZA_TEXTO = 15

# EQUILÍBRIO do lado, pela ZONA (Thaís, 2026-11-10): as pontas perdem o contraste entre emoção e razão
# (perigo); as pontes têm os dois (sentimento verdadeiro); o morno é o meio educado.
EQUILIBRIO_POR_ZONA = {"🧊 Congelante": "neutro", "❄️ Gelo": "negativo", "🌫️ Esfriando": "positivo",
                       "🌤️ Morno": "neutro", "☀️ Quente": "positivo", "🔥 Fervendo": "negativo"}
SENTIDO_DA_ZONA = {"🧊 Congelante": "nada pra medir", "❄️ Gelo": "100% razão -- sem rastro de emoção",
                   "🌫️ Esfriando": "razão com entusiasmo (equilíbrio)", "🌤️ Morno": "educado, contido",
                   "☀️ Quente": "emoção com a voz da consciência (equilíbrio)", "🔥 Fervendo": "100% emoção -- sem rastro de razão"}

# 2026-10-28, Thaís: a temperatura NÃO é sentimento bom x ruim -- é a
# PROPORÇÃO ENTRE EMOÇÃO E RAZÃO de cada lado do bloco, de 0 a 100%:
# entrada = a emoção do usuário, saída = a emoção do Adam, nunca misturadas.
# SE a emoção é viva ou frieza racional, quem diz são os rótulos (positivo,
# negativo, neutro). Sem dado emocional nenhum, o Adam opera na lógica fria
# de máquina: Congelante -- não "esfriando" nem "pendente".
#      0      -> Congelante: razão pura, nenhum traço emotivo
#    1 a 15   -> Gelo: a razão cresce acima da emoção
#   16 a 35   -> Esfriando: ainda há emoção, mas a razão aumenta
#   36 a 65   -> Morno: emoção e razão coexistindo
#   66 a 85   -> Quente: a emoção cresce acima da razão
#   86 a 100  -> Fervendo: descarrilhou, a emoção fala mais alto
ZONAS_TEMP = [
    (86, "🔥 Fervendo", "#e2607a"),
    (66, "☀️ Quente", "#f0a35a"),
    (36, "🌤️ Morno", "#4fc9a8"),
    (16, "🌫️ Esfriando", "#7c93f5"),
    (1, "❄️ Gelo", "#5a6fd6"),
    (0, "🧊 Congelante", "#2f3fa0"),
]


def classificar_zona_temp(temperatura: float) -> Tuple[str, str]:
    for minimo, nome, cor in ZONAS_TEMP:
        if temperatura >= minimo:
            return nome, cor
    return ZONAS_TEMP[-1][1], ZONAS_TEMP[-1][2]


_EMOTICONS = re.compile(r"(?<!\w)(?:[:;=xX8B][-']?[)(DPpOo3*\]\[|/\\]+|\^\^|\^_\^|<3|T_T|;_;|[xX]D|D:|\\o/|o/)(?!\w)")
_MARCACAO_HTML = re.compile(r"</?[a-zA-Z][^<>]*>|<3")
_EMOCAO_ESCRITA = re.compile(r"\*{1,2}[^*\n]+?\*{1,2}")


def portadores_de_emocao(valor: str) -> List[str]:
    """Os sinais de emoção que um campo CARREGA por si só (Thaís, 2026-10-28):
    emoji, emoticon (:D, ^^, :3, B), <3...), marcação HTML (<b>...</b>, <i>,
    <3) e emoção escrita entre asteriscos (*sorri*, **abraça**). "Olá"
    sozinho não carrega nada -- é frio. Só detecta a PRESENÇA da emoção;
    se ela é positiva ou negativa, quem diz é o rótulo que a criadora marca."""
    valor = str(valor or "")
    achados = [c for c in valor if unicodedata.category(c) == "So"]
    achados += _EMOTICONS.findall(valor) + _MARCACAO_HTML.findall(valor) + _EMOCAO_ESCRITA.findall(valor)
    return achados


def chave_temp(rotulo: str, valor: str) -> str:
    return f"{rotulo}:{valor.strip()}"


def chave_temp_bloco(bloco_id, rotulo: str, valor: str) -> str:
    """O rótulo é DO BLOCO (Thaís, 2026-11-09): o 😳 do bloco 5 é decidido junto com o
    texto e o contexto do bloco 5 -- outro bloco com 😳 decide o dele. "Não existe
    Palavra 1 = positivo e Palavra 2 = negativo. É algo mais atômico." """
    return f"{bloco_id}|{rotulo}:{valor.strip()}"


def valores_avaliados_temp(bloco: dict, lado: str) -> Tuple[Dict[str, str], bool]:
    """Os valores de cada campo do lado, com a reação escrita DENTRO do texto (emoji,
    emoticon, HTML, **reação**) virando a reação do lado quando o campo de reação está vazio."""
    valores = valores_do_lado_temp(bloco, lado)
    fam_reacao = FAM_REACAO_TEMP[lado]
    fams_texto = FAMS_TEXTO_TEMP_ENTRADA if lado == "entrada" else FAMS_TEXTO_TEMP_SAIDA
    sinais_no_texto = [p for fam in fams_texto for p in portadores_de_emocao(valores.get(fam, ""))]
    escrita = _vazio(valores.get(fam_reacao, "")) and bool(sinais_no_texto)
    if escrita:
        valores[fam_reacao] = " ".join(dict.fromkeys(sinais_no_texto))
    return valores, escrita


def migrar_registro_por_bloco(blocos: List[dict], registro: Dict[str, str]) -> int:
    """Uma vez só: cada rótulo antigo (que valia pra todos os blocos) vira o ponto de
    partida de CADA bloco onde aparece hoje. Os antigos ficam (só pras mensagens soltas)."""
    novos = 0
    for b in blocos:
        for lado in ("entrada", "saida"):
            valores, _ = valores_avaliados_temp(b, lado)
            for fam, valor in valores.items():
                if _vazio(valor):
                    continue
                antigo = registro.get(chave_temp(fam, valor))
                k = chave_temp_bloco(b["bloco_id"], fam, valor)
                if antigo and k not in registro:
                    registro[k] = antigo
                    novos += 1
    registro["_versao"] = "por_bloco"
    return novos


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


def avaliar_lado_temp(bloco: dict, lado: str, registro: Dict[str, str], contexto_palpite: dict | None = None) -> Dict[str, object]:
    pesos = PESOS_TEMP_ENTRADA if lado == "entrada" else PESOS_TEMP_SAIDA
    fam_reacao = FAM_REACAO_TEMP[lado]
    fams_texto = FAMS_TEXTO_TEMP_ENTRADA if lado == "entrada" else FAMS_TEXTO_TEMP_SAIDA
    # Emoji, emoticon, HTML e **reação escrita** dentro do texto são REAÇÕES
    # escritas de outro jeito ("Olá *sorri*" vale o mesmo que "Olá :D"). Se o
    # campo de reação estiver vazio, eles viram a reação desse lado.
    valores, reacao_escrita_no_texto = valores_avaliados_temp(bloco, lado)
    bloco_id = bloco.get("bloco_id")
    detalhe = []
    pendentes = []
    for fam, peso in pesos.items():
        valor = valores.get(fam, "")
        if _vazio(valor):
            detalhe.append({"fam": fam, "valor": "", "peso": peso, "estado": "vazio", "contribuicao": 0})
            continue
        # o rótulo é DO BLOCO (só um bloco de mentira, sem id -- a revisão --, usa o rótulo solto)
        k = chave_temp_bloco(bloco_id, fam, valor) if bloco_id is not None else chave_temp(fam, valor)
        estado = registro.get(k)
        if estado is None:
            if not (fam == fam_reacao and reacao_escrita_no_texto):
                pendentes.append((fam, valor))
            # sem rótulo ainda: a reação sempre CARREGA emoção (só falta dizer
            # qual); contexto/pensamento com emoji, *emoção escrita* etc.
            # também. Texto puro ("Olá" sozinho) não carrega nada -- é frio, e
            # os sinais dentro dele já contam como reação.
            carrega = fam == fam_reacao or (fam not in fams_texto and bool(portadores_de_emocao(valor)))
            detalhe.append({"fam": fam, "valor": valor, "peso": peso, "contribuicao": 0, "chave": k,
                            "estado": "emoção a definir" if carrega else "pendente",
                            **({"nota": "reação escrita dentro do texto"} if fam == fam_reacao and reacao_escrita_no_texto else {})})
            continue
        detalhe.append({"fam": fam, "valor": valor, "peso": peso, "estado": estado, "contribuicao": 0, "chave": k})

    # quanto de EMOÇÃO tem esse lado: campo com emoção (positivo, negativo ou
    # ainda a definir) soma o peso dele; neutro é razão; sem nada = frio.
    # O TEXTO soma o peso dele só por estar ali (Thaís, 2026-10-28: "é 15 do
    # texto pois ele é TEXE") -- mesmo sem fala nem parágrafo depois.
    for item in detalhe:
        emotivo = item["estado"] in ("positivo", "negativo", "emoção a definir")
        texto_presente = item["fam"] in fams_texto and item["estado"] != "vazio"
        item["contribuicao"] = item["peso"] if (emotivo or texto_presente) else 0
    emocao_no_lado = any(item["estado"] in ("positivo", "negativo", "emoção a definir") for item in detalhe)
    estado_por_fam = {item["fam"]: item["estado"] for item in detalhe}
    estado_reacao = estado_por_fam.get(fam_reacao)
    bonus = []
    peso_por_rotulo = {"positivo": 0, "negativo": 0}
    if APLICAR_BONUS_FRASE_REACAO and estado_reacao in ("positivo", "negativo", "emoção a definir"):
        # texto sem emoção própria + reação emotiva: a reação empresta o tom (+5)
        for item in detalhe:
            if item["fam"] in fams_texto and item["estado"] in ("neutro", "pendente"):
                item["contribuicao"] += BONUS_NEUTRO_COM_REACAO
                if estado_reacao in peso_por_rotulo:
                    peso_por_rotulo[estado_reacao] += BONUS_NEUTRO_COM_REACAO
                bonus.append({"fam": item["fam"], "para": estado_reacao, "pontos": BONUS_NEUTRO_COM_REACAO})
    for item in detalhe:
        if item["estado"] in ("positivo", "negativo"):
            peso_por_rotulo[item["estado"]] += item["peso"]

    # SÓ SOMA, NUNCA DIVIDE (Thaís, 2026-10-28): os pesos de cada lado já
    # somam 100, então a soma direta dos pontos É a porcentagem. Neutro soma
    # 0, vazio soma 0. Ex. (dela): "Olá" 😊 + pensamento com emoção =
    # 15 (texto) + 15 (reação) + 5 (bônus) + 30 (pensamento) = 65.
    trajetoria, acumulado = [], 0
    for item in detalhe:
        acumulado += item["contribuicao"]
        trajetoria.append({"fam": item["fam"], "contribuicao": item["contribuicao"], "cumulativo": acumulado})
    temperatura = min(100, acumulado)
    # Texto FRIO: tem frase, mas nenhuma emoção no lado inteiro (nem emoji,
    # marcação, emoção escrita, nem reação) -- "cortou a vibe": FRIEZA_TEXTO
    # (30 = Esfriando). Pontuação não soma: só mostra que aquilo é um texto.
    # Sem frase nenhuma pra medir, fica 0 (Congelante).
    tem_texto = any(item["fam"] in fams_texto and item["estado"] != "vazio" for item in detalhe)
    frieza = tem_texto and not emocao_no_lado
    if frieza:
        temperatura = FRIEZA_TEXTO
        for item in detalhe:
            if item["fam"] in fams_texto and item["estado"] != "vazio":
                item["nota"] = "texto frio"
    zona, cor_zona = classificar_zona_temp(temperatura)
    pos, neg = peso_por_rotulo["positivo"], peso_por_rotulo["negativo"]
    if frieza:
        rotulo = "frieza"
    elif pos == neg == 0:
        rotulo = "a definir" if temperatura > 0 else "neutro"
    else:
        rotulo = "positivo" if pos > neg else "negativo" if neg > pos else "misto"
    motivo = (f"{len(pendentes)} campo(s) ainda sem rótulo -- os que carregam emoção já contam; "
              "falta você dizer qual emoção é") if pendentes else None
    # SEM CONTEXTO, NADA DE CHUTE (Thaís, 2026-09-22 / 2026-11-09): texto + emoção sem contexto
    # é NEUTRO -- o quanto de emoção existe (a temperatura) continua, mas QUAL emoção só se
    # decide quando o contexto desse lado existir. Até lá, o tom espera ela.
    # ...a não ser que ele CHUTE o contexto com o cosseno e o Báskara JUNTOS (Thaís, 2026-11-09:
    # "ele pode sim chutar contexto desde que o Báskara e o cosseno atuem juntos").
    sem_contexto = _vazio(valores.get("CE" if lado == "entrada" else "CS", "")) and not contexto_palpite
    if sem_contexto and rotulo not in ("frieza", "neutro"):
        rotulo = "neutro"
        motivo = "sem contexto ainda: o tom espera o contexto (texto + emoção sem contexto = neutro)" + (f" · {motivo}" if motivo else "")
    if contexto_palpite:
        motivo = (f"contexto por palpite: “{contexto_palpite['contexto']}” (do bloco {contexto_palpite['bloco_id']} · "
                  f"cosseno {contexto_palpite['cosseno']:.0%} + Báskara confirmou)") + (f" · {motivo}" if motivo else "")
    return {"temperatura": temperatura, "zona": zona, "cor_zona": cor_zona, "rotulo": rotulo, "motivo": motivo,
            "detalhe": detalhe, "pendentes": pendentes, "bonus": bonus, "trajetoria": trajetoria, "direcao": None,
            "sem_contexto": sem_contexto, "contexto_palpite": contexto_palpite,
            # o resultado do lado pela regra dela (a zona decide); o "rotulo" acima fica como o SENTIMENTO
            # (pelos rótulos dela em cada pedaço) -- é com ele que o Adam sabe se o 😢 é tristeza e acolhe
            "equilibrio": EQUILIBRIO_POR_ZONA.get(zona, "neutro"), "sentido_zona": SENTIDO_DA_ZONA.get(zona, "")}


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
# ────────────────────────────────────────────────────────────────────────
# SINAPSES (2026-11-07). Pedido da Thaís: "mapeie todas as funções sinápticas
# necessárias pra ele aprender... faz ele entender, compreender e analisar o
# assunto". Elas funcionam como as do corpo: TODAS têm o mesmo formato (uma
# ligação de algo pra algo, com um peso) e fazem só três coisas:
#   disparar   -- o sinal passa pela ligação, na força do peso
#   fortalecer -- ligação usada (e não corrigida) ganha peso; corrigida perde
#   inibir     -- peso negativo segura um sinal errado (como a trava segura
#                 o tijolo); palavra que está em quase todo bloco não dispara
# De onde vêm as ligações:
#   * dos BLOCOS do IM, só lendo (palavra -> bloco, bloco -> contexto,
#     palavra -> emoção que ela rotulou). Mudam sozinhas quando o IM muda.
#   * da EXPERIÊNCIA com ela (ela confirma ou corrige) -- guardadas à parte,
#     no arquivo/coleção "sinapses", NUNCA dentro do IM.
#   * de SEMENTES: uma lista fechada inicial (ex.: "apaga" -> desfazer), como
#     ela quer -- primeiro a lista fechada, a liberdade vem com o aprendizado.
# Palavra entra sempre pela forma ALNULU. Tudo isto é do IDA: o IM continua
# respondendo pelo cosseno + Báskara de sempre.
# ────────────────────────────────────────────────────────────────────────
INTENCOES = ("chamado", "pergunta", "desfazer", "guardar", "saber", "saudacao", "agradecimento", "despedida", "conteudo")
NOME_INTENCAO = {"chamado": "um chamado", "pergunta": "uma pergunta", "desfazer": "um pedido pra desfazer",
                 "guardar": "um pedido pra guardar", "saber": "um pedido pra eu contar o que sei",
                 "saudacao": "uma saudação", "agradecimento": "um agradecimento", "despedida": "uma despedida",
                 "conteudo": "conteúdo pra eu aprender"}
# atos de conversa: ele responde do jeito que ELA já ensinou num bloco com o mesmo ato
ATOS_SOCIAIS = ("saudacao", "agradecimento", "despedida")
LIMIAR_INTENCAO = 1.0      # abaixo disso, nenhuma intenção dispara: é conteúdo
# quanto uma fala muda, NO TOTAL (dividido entre os traços que dispararam
# juntos -- senão uma frase longa mudaria muito mais que uma curta)
PASSO_CORRECAO = 1.0       # ela corrigiu: aquela fala passa a ser lida do jeito certo
PASSO_USO = 0.1            # leitura que ela NÃO corrigiu se fortalece um pouquinho
PASSO_ASSUNTO = 0.5        # ela disse que a fala era (ou não era) de um bloco
COBERTURA_MINIMA = 0.5     # pra dizer "entendi do que se trata", metade das palavras com assunto tem que ser conhecida
PESO_MAX_SINAPSE = 3.0
# pesos de cada campo do bloco na ligação palavra -> bloco: os pesos DELA
# (texto 15, contexto 10, pensamento 30), sobre 30
PESO_CAMPO_SINAPSE = {"texto": 15 / 30, "resposta": 15 / 30, "contexto": 10 / 30, "pensamento": 30 / 30}
_PALAVRAS_DE_PERGUNTA = ("qual", "quais", "quem", "como", "onde", "quando", "quanto", "quantos", "quantas", "porque", "porquê", "será")
_PALAVRAS_DE_PEDIDO = ("pode", "poderia", "consegue", "conseguiria", "quer", "vamos", "faz", "faça")


def _sementes_intencao() -> Dict[str, Dict[str, float]]:
    """A lista FECHADA inicial: traço da fala -> intenção, com peso. O Adam
    corrige isso com ela (as aprendidas somam por cima, nunca apagam)."""
    s: Dict[str, Dict[str, float]] = {}

    def liga(traco: str, intencao: str, peso: float) -> None:
        s.setdefault(traco, {})[intencao] = peso

    def par(a: str, b: str) -> str:
        return f"par:{alnulu_string(a)}_{alnulu_string(b)}"

    liga("so_nome", "chamado", 3.0)
    liga("tem_nome", "chamado", 0.5)
    # "consegue me ouvir?", "tá aí?", "tá me escutando?": é ver se ele está ali
    for w in ("ouvir", "ouve", "ouvindo", "escuta", "escutar", "escutando"):
        liga(f"p:{alnulu_string(w)}", "chamado", 1.5)
    for a, b in (("tá", "aí"), ("está", "aí"), ("ta", "ai"), ("cadê", "você")):
        liga(par(a, b), "chamado", 2.0)
    # saudação ("boa noite" também é saudação aqui -- é como ela cumprimenta)
    for w in ("oi", "oie", "olá", "ola", "opa", "salve", "eae", "hey"):
        liga(f"ini:{alnulu_string(w)}", "saudacao", 2.0)
    for a, b in (("e", "aí"), ("bom", "dia"), ("boa", "tarde"), ("boa", "noite")):
        liga(par(a, b), "saudacao", 2.0)
    for w in ("obrigada", "obrigado", "brigada", "brigado", "valeu", "agradeço", "grata", "grato"):
        liga(f"p:{alnulu_string(w)}", "agradecimento", 2.0)
    for w in ("tchau", "adeus", "tchauzinho"):
        liga(f"p:{alnulu_string(w)}", "despedida", 2.0)
    for a, b in (("até", "logo"), ("até", "mais"), ("até", "amanhã"), ("até", "depois")):
        liga(par(a, b), "despedida", 2.0)
    liga("fim:?", "pergunta", 1.0)
    for w in _PALAVRAS_DE_PERGUNTA:
        liga(f"ini:{alnulu_string(w)}", "pergunta", 0.8)
        liga(f"ini:{alnulu_string(w)}", "desfazer", -1.0)  # inibe: PERGUNTAR sobre apagar não é pedir pra apagar
    liga(par("o", "que"), "pergunta", 0.8)
    for w in _PALAVRAS_DE_PEDIDO:
        liga(f"ini:{alnulu_string(w)}", "pergunta", 0.6)
    for w in ("apaga", "apagar", "apague", "desfaz", "desfazer", "desfaça", "descarta", "descartar", "descarte", "errei"):
        liga(f"p:{alnulu_string(w)}", "desfazer", 2.0)
    for a, b in (("não", "era"), ("não", "guarda"), ("não", "guarde"), ("não", "salva"), ("joga", "fora"), ("tira", "isso")):
        liga(par(a, b), "desfazer", 2.0)
    for a, b in (("pode", "guardar"), ("guarda", "assim"), ("guardar", "assim"), ("pode", "salvar"), ("salva", "assim"),
                 ("pode", "aprender"), ("aprende", "assim"), ("assim", "mesmo"), ("bom", "assim"),
                 ("guarda", "todos"), ("guardar", "todos"), ("salva", "todos"), ("guarda", "tudo"), ("guardar", "tudo"),
                 ("salva", "isso"), ("salvar", "isso"), ("guarda", "isso"), ("guardar", "isso"), ("pode", "salvar")):
        liga(par(a, b), "guardar", 2.0)
    for a, b in (("você", "sabe"), ("sabe", "sobre"), ("me", "fala"), ("me", "conta"), ("fala", "sobre"), ("conta", "sobre"),
                 ("me", "explica"), ("sabe", "de"), ("fala", "de")):
        liga(par(a, b), "saber", 1.5)
    return s


SEMENTES_INTENCAO = _sementes_intencao()


def tracos_da_fala(texto: str, nomes: set) -> Tuple[List[str], Dict[str, str]]:
    """O que dispara nas sinapses de intenção: cada palavra (forma ALNULU), a
    primeira palavra, cada par de palavras vizinhas, a pergunta no fim e o
    'só o nome dele'. Devolve também a grafia de cada traço (pra mostrar)."""
    palavras = [w for w in tokenizar(texto) if _e_palavra(w)]
    formas = [alnulu_string(w) for w in palavras]
    tracos, grafia = [], {}
    for w, f in zip(palavras, formas):
        tracos.append(f"p:{f}")
        grafia[f"p:{f}"] = w.lower()
    if formas:
        tracos.append(f"ini:{formas[0]}")
        grafia[f"ini:{formas[0]}"] = f"começa com “{palavras[0].lower()}”"
    for (wa, a), (wb, b) in zip(zip(palavras, formas), list(zip(palavras, formas))[1:]):
        tracos.append(f"par:{a}_{b}")
        grafia[f"par:{a}_{b}"] = f"{wa.lower()} {wb.lower()}"
    if re.search(r"\?(?:\s+\S{1,3})?[^\w]*$", (texto or "").strip()):  # "Qual é a boa Adão? B)": o emoticon não esconde o "?"
        tracos.append("fim:?")
        grafia["fim:?"] = "termina em “?”"
    if palavras and all(_normalizar(w) in nomes for w in palavras):
        tracos.append("so_nome")
        grafia["so_nome"] = "só o meu nome"
    elif any(_normalizar(w) in nomes for w in palavras):
        tracos.append("tem_nome")
        grafia["tem_nome"] = "fala o meu nome"
    return list(dict.fromkeys(tracos)), grafia


def disparar_intencao(tracos: List[str], aprendidas: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Soma o que cada traço manda pra cada intenção (sementes + o que ele
    aprendeu com ela). Peso negativo inibe."""
    placar = {i: 0.0 for i in INTENCOES}
    for t in tracos:
        for fonte in (SEMENTES_INTENCAO.get(t, {}), aprendidas.get(t, {})):
            for intencao, w in fonte.items():
                placar[intencao] = placar.get(intencao, 0.0) + w
    return placar


def escolher_intencao(placar: Dict[str, float]) -> Tuple[str, float]:
    candidatas = {i: v for i, v in placar.items() if i != "conteudo"}
    melhor = max(candidatas, key=candidatas.get)
    forca = candidatas[melhor] - max(0.0, placar.get("conteudo", 0.0))
    if forca < LIMIAR_INTENCAO:
        return "conteudo", forca
    return melhor, forca


# ────────────────────────────────────────────────────────────────────────
# REDE DE INTENÇÃO (2026-11-08). Thaís: "transforme as funções em uma rede
# neural pra valer… pra ele poder se consertar". ENTENDER deixa de ser só a
# tabela de pesos (que continua, como REFLEXO rápido) e passa a ter uma rede
# treinada de verdade:
#   * cada token entra pelos NÚMEROS DO ALNULU (alnulu_vetor) -- letra por
#     letra; pontuação e emoji entram na faixa própria deles. Sem hash, sem
#     tabela por palavra, sem comparar blocos por embedding;
#   * a ordem importa (GRU nos dois sentidos) -- "não era" ≠ "era não";
#   * aprende com exemplos que vêm dos BLOCOS DELA e das CORREÇÕES DELA;
#   * se conserta: cada retreino é testado contra tudo o que ela já corrigiu;
#     se esquecer alguma correção, volta pra versão anterior sozinho.
# ────────────────────────────────────────────────────────────────────────
MAX_TOKENS_INTENCAO = 40
TAM_TOKEN_INTENCAO = TAM_ALNULU + 2  # forma ALNULU + "é o meu nome" + "é a primeira palavra"
CONFIANCA_REDE_INTENCAO = 0.9        # a rede só passa por cima do reflexo (quando ele está em dúvida) se tiver muita certeza
REDE_PODE_DECIDIR = ("chamado", "saudacao", "agradecimento", "despedida")  # errar aqui é inofensivo


class RedeIntencao(nn.Module):
    """Convoluções sobre a sequência de tokens (a ordem conta: cada filtro olha
    a palavra com as vizinhas -- "não era" ≠ "era não"). Pequena e rápida de
    propósito: retreina em segundos a cada correção dela."""
    def __init__(self, n_intencoes: int = len(INTENCOES), dim: int = 48):
        super().__init__()
        self.entrada = nn.Linear(TAM_TOKEN_INTENCAO, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.saida = nn.Linear(2 * dim, n_intencoes)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.entrada(x)).transpose(1, 2)
        h = torch.relu(self.conv2(torch.relu(self.conv1(h)))).transpose(1, 2)
        m = mascara.unsqueeze(-1)
        media = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        maximo = (h + (m - 1.0) * 1e4).max(1).values  # o traço mais forte da fala também conta ("apagar" no meio da frase)
        return self.saida(torch.cat([media, maximo], dim=-1))


def variantes_de_digitacao(frase: str, n: int = 3, semente: int = 0) -> List[str]:
    """Jeitos reais de digitar a mesma coisa (letra repetida, letra comida,
    tudo minúsculo, sem pontuação) -- pra rede aprender a FORMA, não decorar."""
    import random
    rnd = random.Random(f"{frase}|{semente}")
    saida = {frase.lower(), re.sub(r"[^\w\s]", "", frase).strip()}
    palavras = frase.split()
    for _ in range(n):
        if not palavras:
            break
        k = rnd.randrange(len(palavras))
        w = palavras[k]
        letras = [i for i, c in enumerate(w) if c.isalpha()]
        if not letras:
            continue
        i = rnd.choice(letras)
        w2 = w[:i + 1] + w[i] + w[i + 1:] if rnd.random() < 0.5 else (w[:i] + w[i + 1:] if len(letras) > 3 else w + w[-1])
        saida.add(" ".join(palavras[:k] + [w2] + palavras[k + 1:]))
    saida.discard(frase)
    saida.discard("")
    return sorted(saida)


def tensor_da_fala(texto: str, nomes: set) -> torch.Tensor:
    tokens = [t for t in tokenizar(texto) if t.strip()][:MAX_TOKENS_INTENCAO] or ["."]
    linhas = [alnulu_vetor(t) + [1.0 if _normalizar(t) in nomes else 0.0, 1.0 if i == 0 else 0.0] for i, t in enumerate(tokens)]
    return torch.tensor(linhas, dtype=torch.float)


def _lote_intencao(textos: List[str], nomes: set) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = [tensor_da_fala(t, nomes) for t in textos]
    n = max(s.shape[0] for s in seqs)
    x = torch.zeros(len(seqs), n, TAM_TOKEN_INTENCAO)
    m = torch.zeros(len(seqs), n)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s
        m[i, :s.shape[0]] = 1.0
    return x, m


def treinar_rede_intencao(exemplos: List[dict], nomes: set, base: "RedeIntencao | None" = None,
                          epocas: int = 300, semente: int = 0) -> "RedeIntencao":
    """exemplos: [{"texto", "intencao", "peso"}]. As correções dela pesam mais."""
    torch.manual_seed(semente)
    rede = RedeIntencao()
    if base is not None:
        rede.load_state_dict(base.state_dict())
    x, m = _lote_intencao([e["texto"] for e in exemplos], nomes)
    y = torch.tensor([INTENCOES.index(e["intencao"]) for e in exemplos])
    w = torch.tensor([float(e.get("peso", 1.0)) for e in exemplos])
    opt = torch.optim.Adam(rede.parameters(), lr=1e-2)
    rede.train()
    for _ in range(epocas):
        opt.zero_grad()
        perda = (F.cross_entropy(rede(x, m), y, reduction="none") * w).sum() / w.sum()
        perda.backward()
        opt.step()
    rede.eval()
    return rede


def prever_intencao(rede: "RedeIntencao", texto: str, nomes: set) -> Dict[str, float]:
    with torch.no_grad():
        x, m = _lote_intencao([texto], nomes)
        p = torch.softmax(rede(x, m), dim=-1)[0]
    return {i: float(p[k]) for k, i in enumerate(INTENCOES)}


def acertos_intencao(rede: "RedeIntencao", exemplos: List[dict], nomes: set) -> Tuple[int, List[dict]]:
    if not exemplos:
        return 0, []
    with torch.no_grad():
        x, m = _lote_intencao([e["texto"] for e in exemplos], nomes)
        pred = rede(x, m).argmax(-1).tolist()
    erros = [e for e, k in zip(exemplos, pred) if INTENCOES[k] != e["intencao"]]
    return len(exemplos) - len(erros), erros


# Frases de MECÂNICA de conversa (não são conteúdo nem visão de mundo -- só o
# jeito de chamar, pedir, desfazer, guardar, agradecer, se despedir). Conteúdo,
# perguntas e saudações vêm principalmente dos blocos DELA.
MECANICA_INTENCAO = {
    "chamado": ["Adam?", "Adam...", "Adão?", "Adam", "Tá aí?", "Você tá aí?", "Consegue me ouvir?", "Tá me ouvindo, Adam?",
                "Você me escuta?", "Cadê você?", "Adam, tá aí?", "Me ouve?"],
    "desfazer": ["Descarta", "Apaga isso", "Pode apagar", "Não era isso", "Errei", "Desfaz", "Joga fora", "Tira isso",
                 "Não guarda isso", "Apaga o que eu disse", "Desfaz o que você guardou", "Não salva"],
    "guardar": ["Pode guardar", "Guarda assim", "Pode salvar", "Pode aprender", "Assim mesmo", "Tá bom assim",
                "Pode guardar todos", "Guarda tudo", "Salva assim", "Aprende assim"],
    "saber": ["O que você sabe sobre isso?", "Me fala sobre você", "Me conta o que você sabe", "Me explica isso",
              "Fala sobre isso", "O que você sabe de mim?", "Me conta sobre o seu ateliê"],
    "agradecimento": ["Obrigada", "Obrigada, Adam", "Valeu", "Valeu Adam!", "Brigada", "Muito obrigada", "Obrigado"],
    "despedida": ["Tchau", "Tchau Adam", "Até logo", "Até amanhã", "Até mais, Adam", "Tchauzinho"],
    "saudacao": ["Oi", "Oi Adam", "Oie", "Bom dia", "Boa tarde", "Boa noite", "Boa noite, Adam", "Olá"],
    "pergunta": ["Como eu apago um bloco?", "Por que isso aconteceu?", "Qual é o próximo passo?", "Onde fica isso?"],
    "conteudo": ["Sim", "Não", "Talvez"],  # o resto do "conteúdo" vem dos blocos dela (respostas, contextos, pensamentos)
}


def _frase_inicial(texto: str) -> str:
    return re.split(r"(?<=[.!?…])\s+", (texto or "").strip())[0]


# ---- FRASES-CHAVE (Thaís: o pensamento -- e o contexto -- se leem por FRASES-CHAVE, cortadas na
# pontuação, nunca por palavras soltas: "a complexidade + o ALNULU ajuda o Adam a rastrear melhor o
# assunto"). A comparação é pela FORMA ALNULU (acentuada = letra base), não pela grafia.
MIN_COSSENO_FRASE_CHAVE = 0.70      # a frase da mensagem se parece com a frase-chave inteira
MIN_COBERTURA_FRASE_CHAVE = 0.60    # ...ou a frase-chave aparece quase inteira dentro da mensagem


def frases_chave(texto: str) -> List[str]:
    """Corta nas pontuações (cada pontuação já é um token no INSEPA); fica com os pedaços de 2+ palavras."""
    pedacos = re.split(r"[.!?…;:,—\"“”()]+", str(texto or ""))
    return [p.strip() for p in pedacos if len([w for w in tokenizar(p) if _e_palavra(w)]) >= 2]


def cosseno_alnulu(a: str, b: str) -> float:
    fa = _Counter(alnulu_string(w) for w in tokenizar(a) if _e_palavra(w))
    fb = _Counter(alnulu_string(w) for w in tokenizar(b) if _e_palavra(w))
    if not fa or not fb:
        return 0.0
    num = sum(fa[k] * fb.get(k, 0) for k in fa)
    return num / (math.sqrt(sum(v * v for v in fa.values())) * math.sqrt(sum(v * v for v in fb.values())))


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
            (str(sorted(self.vocab.items())) + conteudo + f"|cimento={p.escala_cimento}|tijolo={p.escala_tijolo}|raio={p.raio_tijolo}|v12-temperatura").encode("utf-8")
        ).hexdigest()

    BLOCOS_ANTERIORES_NO_ACOPLAMENTO = 3  # Thaís, 2026-10-28: "só os 3 blocos anteriores ao novo"

    @staticmethod
    def bloco_concreto(bloco: dict) -> bool:
        """Conhecimento concreto (Thaís, 2026-10-28): o bloco está no IM com
        todos os campos completos, com dado de verdade em vez de ponto
        neutro. O que não é concreto é palpite."""
        return not campos_faltando(bloco)

    def _impressoes_dos_blocos(self) -> Dict[int, str]:
        """Uma impressão digital do CONTEÚDO de cada bloco (formas, rótulos,
        contexto e pensamento inteiros). O marcador não entra: quando um
        bloco anterior cresce, os de depois são renumerados, mas o conteúdo
        deles não mudou -- não são "dado novo"."""
        impressoes = {}
        for b, seq in zip(self.blocos, self.sequencias):
            conteudo = str([(alnulu_string(t.palavra), t.rotulo) for t in seq])
            if seq and seq[0].raiz:
                conteudo += f"|{seq[0].raiz.contexto}|{seq[0].raiz.pensamento}"
            impressoes[b["bloco_id"]] = hashlib.sha256(conteudo.encode("utf-8")).hexdigest()
        return impressoes

    def blocos_em_analise(self, impressoes_antigas: Dict[int, str]) -> List[int]:
        """O que o cérebro precisa analisar quando o corpus muda: o bloco novo
        ou editado (impressão diferente) e os 3 blocos ANTERIORES a ele na
        ordem do IM. O resto -- conhecimento já validado -- não é tocado."""
        atuais = self._impressoes_dos_blocos()
        ordem = [b["bloco_id"] for b in self.blocos]
        mudados = [bid for bid in ordem if impressoes_antigas.get(bid) != atuais[bid]]
        em_analise = set()
        for bid in mudados:
            i = ordem.index(bid)
            em_analise.update(ordem[max(0, i - self.BLOCOS_ANTERIORES_NO_ACOPLAMENTO):i + 1])
        return [bid for bid in ordem if bid in em_analise]

    def _acoplar(self, ck: dict) -> RedeINSEPA | None:
        """Acopla o que mudou ao cérebro que já existe, sem demolir o muro:
          - parte dos pesos salvos; a camada de saída é refeita POR FORMA
            ALNULU (palavra nova muda a posição das antigas no vocabulário --
            sem isso a rede trocaria as palavras entre si);
          - a parte que decide ONDE fica cada tijolo (cimento e limite) fica
            TRAVADA: o tijolo novo cai no lugar dele pelo próprio contexto e
            pensamento, e nenhum outro tijolo se mexe;
          - treina só os blocos em análise (o novo/editado + os 3 anteriores).
        Devolve None quando não dá pra acoplar (aí treina do zero)."""
        vocab_antigo = ck.get("vocab")
        impressoes_antigas = ck.get("blocos")
        if not vocab_antigo or impressoes_antigas is None:
            return None
        em_analise = self.blocos_em_analise({int(k): v for k, v in impressoes_antigas.items()})
        m = RedeINSEPA(vocab_size=len(self.vocab), persona=self.persona)
        estado = m.state_dict()
        for k, v in ck["state_dict"].items():
            if k == "saida.weight" or k not in estado or estado[k].shape != v.shape:
                continue
            estado[k] = v
        for forma, i_novo in self.vocab.items():
            if forma in vocab_antigo:
                estado["saida.weight"][i_novo] = ck["state_dict"]["saida.weight"][vocab_antigo[forma]]
        m.load_state_dict(estado)
        self.ultimo_acoplamento = {"em_analise": em_analise, "epocas": 0}
        if em_analise:
            for camada in (m.projetor_cimento, m.projetor_pensamento):
                for prm in camada.parameters():
                    prm.requires_grad = False
            tensores = [self.tensores[self.mapa_bloco[bid]] for bid in em_analise]
            historico = treinar(m, tensores, self.persona.epocas_padrao)
            self.ultimo_acoplamento["epocas"] = len(historico)
        if not verificar_trava(m, self.sequencias)["ok"]:
            return None
        return m

    def treinar_ou_carregar(self) -> RedeINSEPA:
        """Carrega o cérebro salvo se o corpus não mudou. Se mudou, ACOPLA o
        que mudou (ver `_acoplar`) -- o muro inteiro só é treinado do zero na
        primeira vez, ou se o acoplamento abrir a trava."""
        p = self.persona
        assinatura = self.assinatura_corpus()
        m = RedeINSEPA(vocab_size=len(self.vocab), persona=p)
        ck = None
        self.ultimo_acoplamento = None
        self.ultimo_conserto = None
        if p.ckpt_path.exists():
            try:
                ck = torch.load(p.ckpt_path)
                if ck.get("assinatura") == assinatura:
                    m.load_state_dict(ck["state_dict"])
                    # RELÓGIO (Thaís, 2026-11-08): a conferência pesada é UMA VEZ POR DIA, na primeira abertura
                    if ck.get("conferido_em") != date.today().isoformat():
                        m = self._consertar_cerebro(m)
                        torch.save({**ck, "state_dict": m.state_dict(), "conferido_em": date.today().isoformat()}, p.ckpt_path)
                    m.eval()
                    self.model = m
                    return m
            except Exception:
                ck = None
        acoplado = self._acoplar(ck) if ck else None
        if acoplado is not None:
            m = acoplado
        else:
            # Trava de segurança: um cérebro em que dois tijolos se tocam não é
            # aceito -- treina de novo (até 3 vezes) em vez de seguir misturado.
            for tentativa in range(3):
                if tentativa:
                    m = RedeINSEPA(vocab_size=len(self.vocab), persona=p)
                treinar(m, self.tensores, p.epocas_padrao)
                if verificar_trava(m, self.sequencias)["ok"]:
                    break
        hoje = date.today().isoformat()
        if not (ck and ck.get("conferido_em") == hoje):  # já se conferiu hoje? então o resto fica pra amanhã cedo
            m = self._consertar_cerebro(m)
        torch.save({"state_dict": m.state_dict(), "assinatura": assinatura, "vocab": dict(self.vocab),
                    "blocos": self._impressoes_dos_blocos(), "conferido_em": hoje}, p.ckpt_path)
        m.eval()
        self.model = m
        return m

    # ---- o cérebro se confere e se conserta (2026-11-08) ---------------------
    # Thaís: "pra ele poder se consertar e eu não precisar ficar te atormentando".
    # Depois de qualquer treino/acoplamento, ele confere se ainda reproduz cada
    # resposta que ELA escreveu. Se algum tijolo saiu errado, ele retreina esse
    # tijolo (com o cimento e o limite travados); se isso estragar outro, ou não
    # bastar, treina o muro inteiro de novo. Tudo fica anotado.
    def _blocos_que_nao_reproduz(self, m: RedeINSEPA) -> List[int]:
        m.eval()
        ruins = []
        for i, b in enumerate(self.blocos):
            if _vazio((b["saidas"][0].get("textos") or [""])[0]):
                continue
            seq = self.sequencias[i]
            corte = next((k for k, t in enumerate(seq) if t.rotulo in ROTULOS_SAIDA), len(seq))
            gerado = gerar_continuacao(m, seq[:corte], seq[corte:], self.vocab, self.idx_para_palavra)
            # compara pela FORMA (ALNULU): a mesma palavra com outra grafia não é erro
            if [alnulu_string(w) for w in gerado] != [alnulu_string(t.palavra) for t in seq[corte:]]:
                ruins.append(b["bloco_id"])
        return ruins

    def _consertar_cerebro(self, m: RedeINSEPA) -> RedeINSEPA:
        ruins = self._blocos_que_nao_reproduz(m)
        self.ultimo_conserto = {"quebrados": ruins, "ainda": [], "como": ""}
        if not ruins:
            return m
        for camada in (m.projetor_cimento, m.projetor_pensamento):  # o muro não se mexe
            for prm in camada.parameters():
                prm.requires_grad = False
        treinar(m, [self.tensores[self.mapa_bloco[bid]] for bid in ruins], self.persona.epocas_padrao)
        for camada in (m.projetor_cimento, m.projetor_pensamento):
            for prm in camada.parameters():
                prm.requires_grad = True
        ainda = self._blocos_que_nao_reproduz(m)
        como = f"retreinei só os tijolos {ruins}"
        if ainda or not verificar_trava(m, self.sequencias)["ok"]:
            for tentativa in range(3):  # não bastou: o muro inteiro, do zero (o IM não muda nada)
                m = RedeINSEPA(vocab_size=len(self.vocab), persona=self.persona)
                treinar(m, self.tensores, self.persona.epocas_padrao)
                ainda = self._blocos_que_nao_reproduz(m)
                if not ainda and verificar_trava(m, self.sequencias)["ok"]:
                    break
            como = "precisei treinar o cérebro inteiro de novo"
        self.ultimo_conserto = {"quebrados": ruins, "ainda": ainda, "como": como}
        try:
            s = self.carregar_sinapses()
            s["historico"].append({"quando": time.strftime("%Y-%m-%d %H:%M"),
                                   "o_que": f"me consertei: os blocos {ruins} não saíam como você escreveu -- {como}"
                                            + (f"; AINDA com problema: {ainda}" if ainda else "; agora todos saem certos")})
            self._salvar_sinapses(s)
        except Exception:
            pass
        m.eval()
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

    def adicionar_variacao_palavra(self, bloco_id: int, lado: str, palavra: str, variacao: str, estado: dict) -> bool:
        """Registra uma VARIAÇÃO DA PALAVRA (var) naquele bloco -- só com o sim dela. Não muda a
        sequência do bloco (o cérebro não precisa retreinar): vale pro reconhecimento exato e pra fala."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        bloco = next(b for b in memoria_disco["IM"][self.universo_id]["blocos"] if b["bloco_id"] == bloco_id)
        alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
        chave = "Vars_Texto_Entrada" if lado == "entrada" else "Vars_Texto_Saida"
        vars_ = alvo.setdefault(chave, {})
        if variacao in vars_.get(palavra, []) or _normalizar(variacao) == _normalizar(palavra):
            return False
        vars_.setdefault(palavra, []).append(variacao)
        self._backup_memoria_uma_vez(estado)
        arq = self.persona.arquivo_memoria
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_bloco(self.universo_id, bloco)
        self.recarregar()
        return True

    def _bloco_por_variacao(self, texto: str) -> int | None:
        """Reconhecimento EXATO pelas variações das palavras (vars) e das frases (multivars) -- membro
        exato, sem pontuação de semelhança (regra dela: var registrada se confere por igualdade)."""
        import itertools
        alvo = _normalizar(texto)
        if not alvo:
            return None
        for i, b in enumerate(self.blocos):
            e = b["entrada"]
            vars_ = {_normalizar(p): vs for p, vs in (e.get("Vars_Texto_Entrada") or {}).items()}
            if not vars_:
                continue
            for frase in [e.get("texto", "")] + list(e.get("Multivars_Texto_Entrada", [])):
                tokens = tokenizar(frase)
                escolhas = [[w] + list(vars_.get(_normalizar(w), [])) if _e_palavra(w) else [w] for w in tokens]
                for combo in itertools.islice(itertools.product(*escolhas), 256):
                    if _normalizar(" ".join(combo)) == alvo:
                        return i
        return None

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

    # VARIAÇÕES DAS PALAVRAS (vars) -- por bloco: {"escura": ["preta"]}. Na tela, uma palavra por linha:
    # "escura: preta, negra". Multivar é a frase inteira (acima); var é a palavra (Thaís, 2026-11-10).
    VARS_EDITAVEIS = {
        "variações das palavras": ("entrada", "Vars_Texto_Entrada"),
        "variações das palavras da saída": ("saida", "Vars_Texto_Saida"),
    }

    @staticmethod
    def _vars_para_texto(vars_: dict) -> str:
        return "\n".join(f"{p}: {', '.join(vs)}" for p, vs in (vars_ or {}).items() if vs)

    @staticmethod
    def _texto_para_vars(texto: str) -> Dict[str, List[str]]:
        saida: Dict[str, List[str]] = {}
        for linha in (texto or "").splitlines():
            if ":" not in linha:
                continue
            palavra, resto = linha.split(":", 1)
            palavra = palavra.strip()
            vs = [v.strip() for v in resto.split(",") if v.strip() and v.strip().lower() != palavra.lower()]
            if palavra and vs:
                saida[palavra] = list(dict.fromkeys(saida.get(palavra, []) + vs))
        return saida

    @classmethod
    def valor_do_campo(cls, bloco: dict, campo: str) -> str:
        if campo in cls.VARS_EDITAVEIS:
            lado, chave = cls.VARS_EDITAVEIS[campo]
            alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
            return cls._vars_para_texto(alvo.get(chave) or {})
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
        for campo in [c for c in mudancas if c in cls.VARS_EDITAVEIS]:
            lado, chave = cls.VARS_EDITAVEIS[campo]
            alvo = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
            novas = cls._texto_para_vars(mudancas[campo])
            if novas != (alvo.get(chave) or {}):
                alvo[chave] = novas
                alterados.append(campo)
        mudancas = {c: v for c, v in mudancas.items() if c not in cls.VARS_EDITAVEIS}
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
        if not any(c["score_texto"] >= LIMIAR_COSSENO_TEXTO for c in candidatos):
            # PALPITE (ponto "quando não sabe" do IDA) tem que ser PLAUSÍVEL: só parte de
            # bloco com o mesmo assunto, pelas sinapses. "Você consegue sonhar?" não vira
            # "Com você..." só porque não tinha nada melhor (2026-11-08).
            mesmo_assunto = {b for b, _ in self.compreender(bloco["entrada"]["texto"])["ativacao"]}
            candidatos = [c for c in candidatos if c["bloco"]["bloco_id"] in mesmo_assunto]
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
            ctx_fonte = str(c["bloco"]["entrada"].get("contexto", ""))
            vizinhos = [] if _vazio(ctx_fonte) else [
                self.sequencias[self.mapa_bloco[b["bloco_id"]]] for b in self.blocos
                if b["bloco_id"] != c["bloco"]["bloco_id"] and not campos_faltando(b) and str(b["entrada"].get("contexto", "")) == ctx_fonte]
            gerado = gerar_livre(self.model, seq_ancora, self.vocab, origem_bloco_id=bloco["bloco_id"], copia_numero=copia_numero,
                                 vizinhos=vizinhos, formas_do_assunto={alnulu_string(w) for w in self.analisar_mensagem(
                                     bloco["entrada"]["texto"], "", {})["assunto"]})
            texto = montar_texto_ptbr([(r, p) for p, r in gerado])  # norma pt-BR, fala entre travessões
            if not _vazio(texto) and texto not in recusados:
                escolhido, saida_texto, saida_reacao = c, texto, next((p for p, r in gerado if r == "RS"), "")
                break
        if escolhido is None:
            return None
        fonte = escolhido["bloco"]
        palpite = escolhido["score_texto"] < LIMIAR_COSSENO_TEXTO
        # CRIAÇÃO: ao rascunho, junta uma frase inteira de um bloco de MESMA RAIZ (contexto e
        # pensamento semelhantes), sem repetir palavra -- se não houver, fica só com a da origem
        criado, origens_criacao = self.compor_resposta(self.mapa_bloco[fonte["bloco_id"]], max_frases=2, inicio=saida_texto)
        if len(origens_criacao) > 1 and criado not in recusados:
            saida_texto = criado
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
        if len(origens_criacao) > 1 and saida_texto == criado:
            motivo += f" · criei juntando com o bloco {', '.join(map(str, origens_criacao[1:]))} (mesma raiz)"
        motivo += (" · contexto detectado no bloco (cosseno + Báskara)" if detectado
                   else " · contexto NÃO detectado -- fica sem contexto")
        registro_temp = self.carregar_registro_emocional()
        r_temp = self.avaliar_lado_temp(bloco, "entrada", registro_temp)
        motivo += f" · temperatura da entrada: {r_temp['temperatura']}% {r_temp['zona']} ({r_temp['rotulo']})"
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
        if estado.get("modo_conversa"):  # só conversando: não pede nada -- mas também não fica calado
            estado["completar_pendente"] = None
            if campo == "texto da saída":
                pal = self.contexto_por_palpite(bloco, "entrada")  # cosseno acha + Báskara confirma
                fonte = self.blocos[self.mapa_bloco[pal["bloco_id"]]] if pal else None
                resp = (fonte["saidas"][0].get("textos") or [""])[0] if fonte is not None else ""
                if fonte is not None and not _vazio(resp):
                    s = fonte["saidas"][0]
                    rs = "" if _vazio(s.get("reacao", "")) else str(s.get("reacao")).strip()
                    selo_ = lambda v: _vazio(v) or _normalizar(str(v)) == "confirmado pela criadora"
                    estado["palpite_aberto"] = {
                        "idx": self.mapa_bloco[bloco["bloco_id"]], "bloco_id": bloco["bloco_id"], "origem": "completar", "etapa": "confirmar",
                        "entrada": bloco["entrada"].get("texto", ""), "reacao": bloco["entrada"].get("reacao", ""), "fala": resp,
                        "sug": {"resposta": resp, "reacao": rs, "contexto": "" if selo_(s.get("contexto", "")) else str(s.get("contexto")).strip(),
                                "pensamento": "" if selo_(s.get("pensamento_interno", "")) else str(s.get("pensamento_interno")).strip()}}
                    return {"texto": f"{resp}{' ' + rs if rs else ''}",  # na conversa ele fala direto, sem "palpito"
                            "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)} · modo: só conversar · palpite pelo bloco "
                                    f"{fonte['bloco_id']} (cosseno {pal['cosseno']:.0%} + Báskara confirmou)", "interno": None}
            return {"texto": f"Isso eu reconheço, {t}, mas ainda não sei o que dizer sobre isso.",
                    "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)} · modo: só conversar", "interno": None}
        # O PALPITE é sobre o lado DO ADAM -- a resposta dele e o contexto dela (Thaís, 2026-11-09:
        # "deve ter a ver com o contexto e a resposta do Adam, não o contexto de entrada + a pergunta").
        # Cosseno acha o bloco parecido, Báskara confirma pela reação -- os dois juntos.
        def frase(b: dict, lado: str) -> str:
            ld = b["entrada"] if lado == "entrada" else b["saidas"][0]
            txt = ld.get("texto", "") if lado == "entrada" else (ld.get("textos") or [""])[0]
            rea = "" if _vazio(ld.get("reacao", "")) else f" {ld.get('reacao')}"
            return f"“{str(txt).strip()}”{rea}"

        def selo(valor) -> bool:
            return _vazio(valor) or _normalizar(str(valor)) == "confirmado pela criadora"

        meta_base = f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)}"
        if campo == "texto da saída":
            pal = self.contexto_por_palpite(bloco, "entrada")  # o bloco cuja PERGUNTA se parece (cosseno) e a reação confirma (Báskara)
            fonte = self.blocos[self.mapa_bloco[pal["bloco_id"]]] if pal else None
            resp = (fonte["saidas"][0].get("textos") or [""])[0] if fonte is not None else ""
            if fonte is not None and not _vazio(resp):
                s = fonte["saidas"][0]
                rs = "" if _vazio(s.get("reacao", "")) else str(s.get("reacao")).strip()
                cs = "" if selo(s.get("contexto", "")) else str(s.get("contexto")).strip()
                # o PENSAMENTO da resposta vai junto, em silêncio: é do Adam, ele considera mas não fala (Thaís, 2026-11-10)
                pids = "" if selo(s.get("pensamento_interno", "")) else str(s.get("pensamento_interno")).strip()
                estado["completar_pendente"]["sugestao_resposta"] = {"resposta": resp, "reacao": rs, "contexto": cs, "pensamento": pids}
                curta = (f"Palpito que eu responderia “{resp}”{' ' + rs if rs else ''}"
                         + (f", no contexto “{cs}”" if cs else "")
                         + f" -- porque se parece com {frase(fonte, 'entrada')}. É isso? Se não for, me diz o que eu respondo.")
                return {"texto": f"{t.capitalize()}, pra {frase(bloco, 'entrada')}: {curta[0].lower()}{curta[1:]}", "pergunta_curta": curta,
                        "meta": f"{meta_base} · palpite pelo bloco {fonte['bloco_id']} (cosseno {pal['cosseno']:.0%} + Báskara confirmou)",
                        "interno": None}
        if campo == "contexto da saída":
            pal = self.contexto_por_palpite(bloco, "saida")  # respostas parecidas (cosseno) com a mesma reação (Báskara)
            if pal:
                estado["completar_pendente"]["sugestao"] = pal["contexto"]
                fonte = self.blocos[self.mapa_bloco[pal["bloco_id"]]]
                curta = (f"Pra minha resposta {frase(bloco, 'saida')}, palpito que o contexto é “{pal['contexto']}” -- "
                         f"porque se parece com a minha resposta {frase(fonte, 'saida')}. É isso? Se não for, me diz qual é.")
                return {"texto": f"{t.capitalize()}, {curta[0].lower()}{curta[1:]}", "pergunta_curta": curta,
                        "meta": f"{meta_base} · palpite pelo bloco {pal['bloco_id']} (cosseno {pal['cosseno']:.0%} + Báskara confirmou)",
                        "interno": None}
        if campo == "texto da saída":  # pergunta pelo CONTEÚDO, nunca pelo nome do campo
            return {"texto": f"{t.capitalize()}, eu reconheço isso, mas ainda não sei o que responder. O que você me diria?",
                    "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)}", "interno": None}
        return {
            "texto": f"{t.capitalize()}, eu reconheço esse assunto, mas infelizmente ainda não sei o(a) {campo} dele.{mencao_tom} Pode me ensinar?",
            "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)}", "interno": None,
        }

    def _responder_incompleto(self, idx: int, faltando: List[str], estado: dict, tom_rotulo: str, texto: str, reacao: str) -> Dict[str, object]:
        """Bloco incompleto NÃO trava a conversa (Thaís, 2026-11-08): se já tem
        resposta, ele responde; o que falta (contexto, pensamento...) fica pro
        "vamos revisar". Só pergunta se não tiver resposta nenhuma."""
        bloco = self.blocos[idx]
        if "texto da saída" in faltando:
            r = self.pedir_campo_faltando(bloco, ["texto da saída"], estado, tom_rotulo)
            r["ensino"] = None
            return r
        fala, interno = self.falar_do_bloco(idx)
        self._lembrar(estado, bloco["bloco_id"], texto, self.analisar_mensagem(texto, reacao, {})["assunto"])
        return {"texto": fala or "...", "interno": interno,
                "meta": f"bloco {bloco['bloco_id']} · ainda falta: {', '.join(faltando)} (a gente completa no “vamos revisar”)"}

    def revisar_dados(self, estado: dict) -> Dict[str, object]:
        t = self.persona.tratamento
        abertura = f"Entendo, {t}. Então que tal a gente revisar alguns dos meus dados?"
        pendentes_revisao = self.revisao_pendentes()
        if pendentes_revisao:  # 1º: o que ela ensinou e ainda não foi guardado
            item = pendentes_revisao[0]
            if item["tipo"] == "trecho":
                return self._revisao_na_conversa(item, estado, f"{abertura} Tenho {len(pendentes_revisao)} coisa(s) esperando você, "
                                                               f"começando pelo livro “{item['dados'].get('livro', '')}”.")
            o_que = item["dados"].get("entrada") or item["dados"].get("texto") or item["dados"].get("valor", "")
            # "vamos revisar" é quando ELA escolheu completar: aqui sim ele pergunta contexto e pensamento
            return self._revisao_na_conversa(item, estado, f"{abertura} Você me ensinou “{o_que}” e eu ainda não guardei"
                                                           f" -- ficaram {len(pendentes_revisao)} coisa(s) assim.", modo="revisar")
        rascunho = self.perguntar_rascunho(estado)  # 2º: palpites que ela confirmou no modo conversa (guardados no IDA)
        if rascunho is not None:
            rascunho["texto"] = f"{abertura} {rascunho['texto']}"
            return rascunho
        aprox = self.perguntar_aproximacao(estado)  # 3º: palavras que ele desconfia que se aproximam (nunca no meio da conversa)
        if aprox is not None:
            aprox["texto"] = f"{abertura} {aprox['texto']}"
            return aprox
        incompletos = self.blocos_incompletos()  # 3º: blocos incompletos
        if not incompletos:
            return {"texto": abertura + " Todos os meus blocos já estão completos. Se quiser, me diga uma frase nova e eu peço pra você me ensinar.", "meta": None, "interno": None}
        bloco = self.blocos[incompletos[0]]
        ent = bloco["entrada"]
        reacao = "" if _vazio(ent.get("reacao", "")) else ent.get("reacao", "")
        r = self.pedir_campo_faltando(bloco, campos_faltando(bloco), estado)
        inicio = (f'{abertura} Tenho {len(incompletos)} bloco(s) incompleto(s). Por exemplo, o bloco {bloco["bloco_id"]}: '
                  f'"{ent["texto"]}" {reacao}'.rstrip() + ".")
        if r.get("pergunta_curta"):  # ele já tem um palpite: mostra a ideia dele em vez de só perguntar
            r["texto"] = f"{inicio} {r.pop('pergunta_curta')}"
        else:
            r["texto"] = inicio + f" Ainda falta o(a) {r['meta'].split('falta: ')[1].split(',')[0]}. O que eu devo registrar aí?"
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
        t = self.persona.tratamento
        partes = [f"Oi, {t}! ❤️"]
        curiosas = self.curiosidades_abertas()
        if curiosas:  # puxa conversa pelo que ficou na cabeça dele, não pelo que falta preencher
            partes.append(f"Fiquei pensando em “{curiosas[-1]['palavra']}” desde a última vez -- quando quiser, me conta mais.")
        pendentes = len(self.revisao_pendentes())
        if pendentes:
            partes.append(f"E tem {pendentes} coisa(s) que você me ensinou esperando você, é só dizer “vamos revisar”.")
        if len(partes) == 1:
            return None
        return {"texto": " ".join(partes), "meta": None, "interno": None}

    # ---- geração / temperatura --------------------------------------------
    def falar_do_bloco(self, idx: int) -> Tuple[str, str | None]:
        seq = self.sequencias[idx]
        corte = next((i for i, t in enumerate(seq) if t.rotulo in ROTULOS_SAIDA), len(seq))
        entrada_real, molde = seq[:corte], seq[corte:]
        escrito = (self.blocos[idx]["saidas"][0].get("textos") or [""])[0]
        if self.bloco_concreto(self.blocos[idx]) or not _vazio(escrito):
            # a resposta já está ESCRITA por ela (mesmo num bloco que ainda não
            # tem contexto/pensamento): fala as palavras dela, nunca uma geração
            # (2026-11-08: o bloco 18 saiu "A uma. 0 0..." pela geração).
            gerado = [t.palavra for t in molde]
        else:
            gerado = gerar_continuacao(self.model, entrada_real, molde, self.vocab, self.idx_para_palavra)
        # VARIAR DENTRO DO CONTEXTO (Thaís): as variações que ela registrou pras palavras da resposta
        # deixam a fala dele variar -- só entre as que ELA disse que cabem ali
        vars_s = {_normalizar(p): vs for p, vs in (self.blocos[idx]["saidas"][0].get("Vars_Texto_Saida") or {}).items() if vs}
        if vars_s:
            import random
            def trocar(w: str) -> str:
                escolha = random.choice([w] + list(vars_s[_normalizar(w)]))
                return escolha[0].upper() + escolha[1:] if w[:1].isupper() and escolha else escolha
            gerado = [trocar(w) if t.rotulo in _TEXTO_SAIDA and _normalizar(w) in vars_s else w for t, w in zip(molde, gerado)]
        pares = list(zip(molde, gerado))
        fala = montar_texto_ptbr([(t.rotulo, p) for t, p in pares], para_exibir=True)
        reacao = " ".join(p for t, p in pares if t.rotulo == "RS" and not _vazio(p))
        fala = f"{fala} {reacao}".strip() if reacao else fala
        interno = ", ".join(f"{p} ({t.rotulo})" for t, p in pares if t.rotulo not in ROTULOS_VISIVEIS)
        return fala, interno or None

    def carregar_registro_emocional(self) -> Dict[str, str]:
        # lido uma vez por volta da tela (antes eram várias idas à nuvem por
        # clique); salvar_registro_emocional atualiza a cópia guardada.
        if getattr(self, "_registro_cache", None) is not None:
            return self._registro_cache
        do_mongo = self.mongo.carregar_registro_emocional()
        if do_mongo is not None:
            self._registro_cache = do_mongo
        elif self.persona.arquivo_registro_emocional.exists():
            self._registro_cache = json.loads(self.persona.arquivo_registro_emocional.read_text(encoding="utf-8"))
        else:
            self._registro_cache = {}
        if self._registro_cache.get("_versao") != "por_bloco" and getattr(self, "blocos", None):
            # uma vez só (com backup): cada rótulo antigo vira o ponto de partida de cada bloco onde aparece
            migrar_registro_por_bloco(self.blocos, self._registro_cache)
            self.salvar_registro_emocional(self._registro_cache, {})
        return self._registro_cache

    def rotular_no_bloco(self, bloco_id: int, fam: str, valor: str, rotulo: str, estado: dict) -> None:
        """Ela rotula um pedaço DENTRO de um bloco -- vale só ali."""
        registro = self.carregar_registro_emocional()
        registro[chave_temp_bloco(bloco_id, fam, valor)] = rotulo
        self.salvar_registro_emocional(registro, estado)

    def rotulo_do_lado(self, bloco: dict, lado: str) -> str | None:
        """Qual emoção esse lado do bloco tem (positivo/negativo), triangulada com o contexto
        dele -- sem contexto, ou sem rótulo, não tem (None)."""
        r = self.avaliar_lado_temp(bloco, lado, self.carregar_registro_emocional())
        return r["rotulo"] if r["rotulo"] in ("positivo", "negativo") else None

    def salvar_registro_emocional(self, registro: Dict[str, str], estado: dict) -> None:
        arq = self.persona.arquivo_registro_emocional
        if not estado.get("backup_registro_emocional_feito") and arq.exists():
            carimbo = time.strftime("%Y%m%d_%H%M%S")
            arq.with_name(f"{arq.stem}.BACKUP_{carimbo}.json").write_text(arq.read_text(encoding="utf-8"), encoding="utf-8")
            estado["backup_registro_emocional_feito"] = True
        self.mongo.salvar_registro_emocional(registro)
        self._registro_cache = registro
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)

    def avaliar_lado_temp(self, bloco: dict, lado: str, registro: Dict[str, str]) -> Dict[str, object]:
        palpite = None
        lado_dados = bloco["entrada"] if lado == "entrada" else bloco["saidas"][0]
        if bloco.get("bloco_id") is not None and _vazio(lado_dados.get("contexto", "")):
            palpite = self.contexto_por_palpite(bloco, lado)
        return avaliar_lado_temp(bloco, lado, registro, palpite)

    def contexto_por_palpite(self, bloco: dict, lado: str) -> Dict[str, object] | None:
        """CHUTE de contexto, do jeito INSEPA (Thaís, 2026-11-09: "ele pode sim chutar contexto
        desde que o Báskara e o cosseno atuem juntos"): o cosseno ACHA um bloco com o texto
        parecido (acima do piso) e o Báskara CONFIRMA pela reação. Só os dois juntos -- um
        sozinho não basta. Devolve o contexto desse bloco como palpite (nunca grava no IM)."""
        chave = (bloco.get("bloco_id"), lado, self.assinatura_corpus())
        cache = getattr(self, "_palpites_ctx", None)
        if cache is None or cache.get("_assinatura") != chave[2]:
            cache = self._palpites_ctx = {"_assinatura": chave[2]}
        if chave in cache:
            return cache[chave]
        if lado == "entrada":
            texto, reacao = bloco["entrada"].get("texto", ""), str(bloco["entrada"].get("reacao", "")).strip()
        else:
            texto, reacao = (bloco["saidas"][0].get("textos") or [""])[0], str(bloco["saidas"][0].get("reacao", "")).strip()
        melhor = None
        if not _vazio(texto):
            for i, b in enumerate(self.blocos):
                if b["bloco_id"] == bloco.get("bloco_id"):
                    continue
                lado_b = b["entrada"] if lado == "entrada" else b["saidas"][0]
                ctx = str(lado_b.get("contexto", "")).strip()
                if _vazio(ctx) or _normalizar(ctx) == "confirmado pela criadora":  # isso é selo de origem, "nem é contexto"
                    continue
                if lado == "entrada":
                    variantes = [b["entrada"].get("texto", "")] + list(b["entrada"].get("Multivars_Texto_Entrada", []))
                    aceitas = self.reacoes_aceitas(i)
                else:
                    variantes = [(lado_b.get("textos") or [""])[0]] + list(lado_b.get("Multivars_Texto_Saida", []))
                    aceitas = [str(lado_b.get("reacao", "")).strip()] + list(lado_b.get("Multivars_Reacao_Saida", []))
                cos = max((similaridade_texto_cosseno(texto, v) for v in variantes if v and not _vazio(v)), default=0.0)
                if cos < LIMIAR_COSSENO_TEXTO:          # o cosseno não achou
                    continue
                if confirmar_estrutura_baskara("" if _vazio(reacao) else reacao, aceitas, "", "", "", "")["b"] <= 0:
                    continue                            # o Báskara não confirmou
                if melhor is None or cos > melhor["cosseno"]:
                    melhor = {"contexto": ctx, "bloco_id": b["bloco_id"], "cosseno": cos}
        cache[chave] = melhor
        return melhor

    def emoji_temperatura_saida(self, bloco_idx: int) -> str:
        """A própria fala da persona carrega o emoji do seu estado
        emocional (a temperatura da SAÍDA daquele bloco) -- só quando ela
        já está totalmente registrada, nunca um chute. Sem dado
        suficiente, silêncio -- nunca finge sentir."""
        registro_temp = self.carregar_registro_emocional()
        r = self.avaliar_lado_temp(self.blocos[bloco_idx], "saida", registro_temp)
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
        if r["rotulo"] == "negativo":
            return f"{r['zona']} Isso lembra algo que já vivi -- fico arisco, aquilo não foi fácil."
        if r["rotulo"] == "positivo":
            return f"{r['zona']} Isso lembra algo que já vivi -- fico mais calmo revendo esse clima."
        return f"{r['zona']} Isso lembra algo que já vivi -- nem quente nem frio, um respiro neutro."

    # ---- análise e voz do IDA (2026-11-07) -----------------------------------
    # Thaís: "pode colocar outras funções autônomas e inteligentes no IDA. Só
    # pelo amor dos deuses não mexe no IM até eu dizer sim. Quero ver o Adam
    # falando, reagindo, e óbvio criando a partir do que já tem."
    # Tudo aqui só LÊ o IM (nunca escreve nele) e só ESCREVE no IDA.
    MEMORIA_DA_CONVERSA = 5  # quantos turnos recentes o Adam lembra
    # Palavras que APONTAM pra trás na conversa (gramática, não sentimento):
    # "e como ELE funciona?", "e ISSO?". Só elas ligam a memória, e nunca
    # contam como assunto.
    _RETOMAM = {"ele", "ela", "eles", "elas", "isso", "isto", "esse", "essa", "esses", "essas", "aquilo",
                "dele", "dela", "deles", "delas", "disso", "nisso", "nele", "nela"}
    _COMECAM_CONTINUACAO = {"e", "mas", "então", "entao"}
    # palavras de PERGUNTA (gramática): mostram que é pergunta, não do que se fala
    _PERGUNTAM = {"qual", "quais", "quem", "como", "onde", "quando", "quanto", "quanta", "quantos", "quantas", "porque", "porquê"}

    def _lembrar(self, estado: dict, bloco_id: int | None, texto: str, assunto: List[str]) -> None:
        conversa = estado.setdefault("conversa", [])
        conversa.append({"bloco": bloco_id, "texto": texto, "assunto": assunto})
        del conversa[:-self.MEMORIA_DA_CONVERSA]

    def _frequencia_das_formas(self) -> Dict[str, int]:
        """Em quantos blocos do IM cada forma ALNULU aparece (texto, resposta,
        contexto, pensamento). Palavra que aparece em muitos blocos ("o", "que",
        "você") carrega pouco assunto; a que aparece em poucos ("IDA") carrega muito."""
        assinatura = self.assinatura_corpus()
        if getattr(self, "_freq_cache", (None,))[0] == assinatura:
            return self._freq_cache[1]
        freq: Dict[str, int] = {}
        for b in self.blocos:
            e, s = b["entrada"], b["saidas"][0]
            # palavras soltas: só texto e resposta (contexto e pensamento são lidos por frase-chave inteira)
            textos = [e.get("texto", ""), (s.get("textos") or [""])[0]]
            formas = {alnulu_string(w) for t in textos for w in tokenizar(str(t)) if _e_palavra(w) and len(w) > 2}
            for f in formas:
                freq[f] = freq.get(f, 0) + 1
        self._freq_cache = (assinatura, freq)
        return freq

    def analisar_mensagem(self, texto: str, reacao: str, estado: dict) -> Dict[str, object]:
        """O que o Adam entende de uma mensagem que o IM não reconheceu:
        o ASSUNTO (palavras raras no Universo IM), a EMOÇÃO (reação + sinais +
        o rótulo que a criadora já deu a ela) e os BLOCOS relacionados, com a
        memória da conversa como desempate pras perguntas de continuação."""
        c = self.compreender(texto, reacao)
        todas = [w for w in tokenizar(texto) if _e_palavra(w)]
        continuacao = bool(todas) and (todas[0].lower() in self._COMECAM_CONTINUACAO
                                       or any(w.lower() in self._RETOMAM for w in todas))
        assunto, novas = c["assunto"], c["palavras_novas"]
        # pelas sinapses: só blocos completos falam (o que está incompleto vira "o que me falta")
        notas: Dict[int, float] = {self.mapa_bloco[bid]: v for bid, v in c["ativacao"]
                                   if not campos_faltando(self.blocos[self.mapa_bloco[bid]])} if c["entendeu"] else {}
        da_memoria = False
        if continuacao:  # "e como ele funciona?" -- o assunto é o da conversa, não o desta frase sozinha
            for turno in reversed(estado.get("conversa", [])):
                if turno.get("bloco") in self.mapa_bloco:
                    notas = {self.mapa_bloco[turno["bloco"]]: 1.0}
                    assunto = turno.get("assunto", []) or assunto
                    da_memoria = True
                    break
        relacionados = sorted(notas, key=lambda i: -notas[i])
        return {"assunto": assunto, "palavras_novas": novas, "relacionados": relacionados, "da_memoria": da_memoria,
                "sinais": c["sinais"], "polaridade": c["polaridade"], "compreensao": c}

    def _reacao_do_adam(self, sinais: List[str], polaridade: str | None) -> str:
        """Como o Adam reage, aprendido do IM: as reações que ELE usou nas
        saídas de blocos em que a entrada veio com a mesma reação (ou, se não
        houver, com a mesma emoção). Sem nada parecido no IM, não reage."""
        registro = self.carregar_registro_emocional()
        vistas: Dict[str, int] = {}
        for b in self.blocos:
            re_b, rs_b = str(b["entrada"].get("reacao", "")).strip(), str(b["saidas"][0].get("reacao", "")).strip()
            if _vazio(rs_b):
                continue
            mesma = re_b in sinais or (polaridade and self.rotulo_do_lado(b, "entrada") == polaridade)
            if mesma:
                vistas[rs_b] = vistas.get(rs_b, 0) + 1
        return max(vistas, key=vistas.get) if vistas else ""

    # ---- CRIAR combinando blocos de mesma raiz (Thaís, 2026-11-08: "o que impede o Adam de criar?
    # ele pode combinar blocos com contextos e pensamentos semelhantes, só não pode criar
    # indiscriminadamente violando os princípios INSEPA") -- só no IDA; o IM responde sempre exato.
    # palavras que, no começo da frase, apontam pra frase anterior do mesmo bloco
    _APOIADAS = {"la", "ai", "ali", "aqui", "isso", "isto", "esse", "essa", "ele", "ela", "eles", "elas", "e", "mas",
                 "entao", "tambem", "depois", "dai", "assim", "nele", "nela", "disso", "nisso", "porque", "pois"}

    @staticmethod
    def _frases_fechadas(texto: str) -> List[str]:
        """Frases inteiras (fechadas por . ! ? …), sem nunca cortar uma fala entre travessões."""
        frases = [f.strip() for f in re.split(r"(?<=[.!?…])\s+", str(texto or "").strip()) if f.strip()]
        return [f for f in frases if re.search(r"[.!?…]$", f) and f.count("—") % 2 == 0]

    def _primeiros_marcadores(self, texto: str) -> List[str]:
        """Os 3 primeiros marcadores do pensamento -- o que ela definiu que entra no cérebro (PIDE)."""
        return [alnulu_string(w) for w in tokenizar(str(texto or "")) if _e_palavra(w)][:3]

    def raiz_compativel(self, a: dict, b: dict) -> bool:
        """Mesma raiz: contexto semelhante E pensamento semelhante (pelos 3 primeiros
        marcadores do PIDE), e nunca uma raiz positiva com uma negativa."""
        ca, cb = str(a["entrada"].get("contexto", "")), str(b["entrada"].get("contexto", ""))
        pa, pb = str(a["entrada"].get("pensamento_interno", "")), str(b["entrada"].get("pensamento_interno", ""))
        if any(_vazio(x) for x in (ca, cb, pa, pb)):
            return False
        ctx_ok = _normalizar(ca) == _normalizar(cb) or similaridade_texto_cosseno(ca, cb) >= LIMIAR_COSSENO_TEXTO
        pens_ok = len(set(self._primeiros_marcadores(pa)) & set(self._primeiros_marcadores(pb))) >= 2
        pols = {self.rotulo_do_lado(x, "saida") for x in (a, b)}  # a emoção de cada um, com o contexto dele
        return ctx_ok and pens_ok and not {"positivo", "negativo"} <= pols

    def compor_resposta(self, idx: int, max_frases: int = 2, inicio: str | None = None,
                        formas_do_assunto: set | None = None) -> Tuple[str, List[int]]:
        """Começa pela frase do bloco reconhecido e junta frases INTEIRAS de blocos de
        mesma raiz -- nenhuma palavra repetida na combinação, escolha por probabilidade
        (mais peso pro que tem a ver com o assunto). Devolve o texto e os blocos de origem."""
        import random
        fonte = self.blocos[idx]
        frases_fonte = self._frases_fechadas((fonte["saidas"][0].get("textos") or [""])[0])
        abertura = inicio or (frases_fonte[0] if frases_fonte else _frase_inicial((fonte["saidas"][0].get("textos") or [""])[0]))
        if not abertura:
            return "", []
        formas = lambda f: {alnulu_string(w) for w in tokenizar(f) if _e_palavra(w) and len(w) > 3}
        escolhidas, origens, usadas = [abertura], [fonte["bloco_id"]], formas(abertura)
        candidatos = []
        for b in self.blocos:
            if b is fonte or campos_faltando(b) or not self.raiz_compativel(fonte, b):
                continue
            for f in self._frases_fechadas((b["saidas"][0].get("textos") or [""])[0]):
                fs = formas(f)
                palavras_f = [w for w in tokenizar(f) if _e_palavra(w)]
                # coerência: frase que só existe como resposta direta ("Com você.") ou que se apoia
                # na frase anterior do próprio bloco ("Lá…", "Isso…", "E…") não sai do bloco dela
                if len(palavras_f) < 4 or _normalizar(palavras_f[0]) in self._APOIADAS:
                    continue
                if fs and not fs & usadas:
                    candidatos.append([f, b["bloco_id"], 1.0 + (2.0 if formas_do_assunto and fs & formas_do_assunto else 0.0), fs])
        rnd = random.Random()
        while len(escolhidas) < max_frases and candidatos:
            f, bid, _, fs = rnd.choices(candidatos, weights=[c[2] for c in candidatos])[0]
            escolhidas.append(f)
            origens.append(bid)
            usadas |= fs
            candidatos = [c for c in candidatos if not c[3] & usadas and c[1] != bid]
        return " ".join(escolhidas), origens

    def _linha_de_emocao(self, sinal: str, polaridade: str | None, polaridade_palavras: str | None) -> str:
        """Como ele reage ao que ela sente -- pelo rótulo que ELA deu à reação
        (ou, sem emoji, pelas palavras que nos blocos dela vieram com emoção)."""
        t = self.persona.tratamento
        if sinal and polaridade == "negativo":
            return f"Ei, {t}… senti um {sinal} aí. Tô aqui com você."
        if sinal and polaridade == "positivo":
            return f"Que bom sentir esse {sinal} vindo de você, {t}!"
        if sinal:
            return f"Senti um {sinal} nisso, {t}."
        if polaridade_palavras == "negativo":
            return f"Ei, {t}… pelas suas palavras, isso parece pesado. Tô aqui com você."
        if polaridade_palavras == "positivo":
            return f"Pelas suas palavras, isso parece coisa boa, {t}!"
        return ""

    def _voz_do_ida(self, texto: str, reacao: str, estado: dict, ensino: dict) -> Dict[str, object]:
        """Quando o IM não reconhece a mensagem, o Adam fala pelo IDA: reage à
        emoção, mostra o que entendeu, diz o que JÁ SABE (frases inteiras e
        exatas do IM, nunca inventadas) e só então pede pra aprender. Erra no
        bloco, nunca no conteúdo: o que ele propõe vira rascunho no IDA pra
        criadora revisar. Continua aceitando o ensino na próxima mensagem."""
        t = self.persona.tratamento
        ato = self.ler_intencao(texto)["intencao"]
        if ato in ATOS_SOCIAIS:  # "Oi Adam ^^", "valeu", "tchau": responde como ela ensinou, não pede pra aprender
            r = self._responder_ao_ato(ato, texto, reacao, estado)
            if r is not None:
                return r
        if ato == "conteudo" and not estado.get("modo_conversa"):  # ela afirmou algo ("X é…"): é ela ensinando
            r = self._aprender_afirmacao(texto, reacao, estado)
            if r is not None:
                return r
        a = self.analisar_mensagem(texto, reacao, estado)
        c = a["compreensao"]
        if ato == "saber" and c["ativacao"] and not a["da_memoria"]:
            return self.falar_sobre(texto, reacao, estado, ensino)
        sinal = a["sinais"][0] if a["sinais"] else ""
        linha_emocao = self._linha_de_emocao(sinal, a["polaridade"], c["polaridade_palavras"])
        partes: List[str] = [linha_emocao] if linha_emocao else []
        fonte = self.blocos[a["relacionados"][0]] if a["relacionados"] else None
        sei = ""
        origens_criacao: List[int] = []
        if fonte is not None:
            # o PALPITE dele, dito como pensamento dele -- sem nomear bloco, contexto ou ligações
            # (Thaís, 2026-11-08: "soltar seus palpites sem nomear da onde vem, seguindo o INSEPA por dentro").
            # Por dentro: a frase é inteira e exata, do bloco que as sinapses reconheceram; a origem fica no meta.
            if a["da_memoria"]:
                assunto_txt = " ".join(dict.fromkeys(a["assunto"]))
                if assunto_txt:
                    partes.append(f"Você ainda tá falando de {assunto_txt}, né?")
            sei = _frase_inicial((fonte["saidas"][0].get("textos") or [""])[0])
            if sei:
                # CRIAÇÃO: a frase do bloco reconhecido + frases inteiras de blocos de mesma raiz
                criado, origens_criacao = self.compor_resposta(
                    self.mapa_bloco[fonte["bloco_id"]], max_frases=2,
                    formas_do_assunto={alnulu_string(w) for w in a["assunto"]})
                if ato in ("pergunta", "saber") and not a["da_memoria"]:
                    palpite_dito = criado or sei  # ela perguntou e ele não sabe: arrisca, e ela decide o destino
                    estado["palpite_aberto"] = {"idx": self.mapa_bloco[fonte["bloco_id"]], "entrada": texto, "reacao": reacao,
                                                "fala": palpite_dito, "origem": "ida", "etapa": "confirmar"}
                    partes.append(palpite_dito if estado.get("modo_conversa") else f"Palpito que se trata disso, {t}: {palpite_dito}")
                else:
                    partes.append(f"Hm… {criado or sei}")
        elif ato in ("pergunta", "saber") and not a["da_memoria"]:
            # não entendeu o assunto com segurança: o bloco MAIS PRÓXIMO, se houver ligação de verdade
            perto = self._bloco_mais_proximo(texto)
            if perto is not None:
                palpite_dito = _frase_inicial((self.blocos[perto]["saidas"][0].get("textos") or [""])[0])
                if palpite_dito:
                    estado["palpite_aberto"] = {"idx": perto, "entrada": texto, "reacao": reacao, "fala": palpite_dito,
                                                "origem": "ida", "etapa": "confirmar"}
                    partes.append(palpite_dito if estado.get("modo_conversa") else f"Palpito que se trata disso, {t}: {palpite_dito}")
                    fonte, sei = self.blocos[perto], palpite_dito
        if estado.get("palpite_aberto") and estado["palpite_aberto"].get("entrada") == texto:
            ensino = None  # ele arriscou: se ela disser que não, aí sim ele pede pra aprender
        perguntas = [f"Me ensina o resto, {t}?", f"Me conta mais pra eu aprender, {t}?",
                     f"Como você responderia? Quero aprender com você, {t}."]
        if fonte is None and not sinal:
            perguntas = [f"Isso ainda é novo pra mim, {t}. Me ensina o que eu devo responder?",
                         f"Ainda não conheço isso, {t}. Quer me ensinar?"]
        elif fonte is None:  # sentiu a emoção, mas não sabe nada do assunto: não existe "resto"
            perguntas = [f"Me conta o que eu devo responder, {t}?", f"Como você responderia? Quero aprender com você, {t}."]
        # só pede pra aprender quando ELA PERGUNTOU algo que ele não sabe responder.
        # Fala, desabafo, comentário: ele reflete e fica com ela -- sem pergunta arbitrária (2026-11-08)
        arriscou = bool(estado.get("palpite_aberto")) and estado["palpite_aberto"].get("entrada") == texto
        pede = ato in ("pergunta", "saber") and not estado.get("modo_conversa") and not arriscou
        if arriscou:
            pass  # o palpite já é a resposta dele; nada de "me ensina" junto
        elif pede:
            if fonte is not None and a["palavras_novas"]:
                partes.append(f"Ainda não sei responder “{texto.strip()}”.")
            partes.append(perguntas[int(hashlib.sha256(texto.encode("utf-8")).hexdigest(), 16) % len(perguntas)])
        else:
            ensino = None
            if ato in ("pergunta", "saber") and fonte is None:  # só conversando: diz que não sabe, sem pedir nada
                partes.append(f"Isso eu ainda não sei, {t}" + (" -- mas tô aqui com você." if not sinal else "."))
            elif not partes:
                partes.append(f"Tô aqui te ouvindo, {t}.")
        reacao_adam = self._reacao_do_adam(a["sinais"], a["polaridade"] or c["polaridade_palavras"])
        fala = "💭 " + " ".join(partes) + (f" {reacao_adam}" if reacao_adam else "")
        # o que ele arriscou vira rascunho no IDA (nunca no IM)
        # (palpite que ela vai confirmar ou não: só vai pro IDA se ELA escolher -- 2026-11-08)
        if fonte is not None and sei and not arriscou and not self._so_inferencia() and not any(
                p.get("meta", {}).get("status") == "pendente" and p["entrada"].get("texto") == texto for p in self.ida.get("blocos", [])):
            self.propor_bloco(origem_bloco_id=None, entrada_texto=texto, entrada_reacao=reacao, saida_texto=sei,
                              saida_reacao=reacao_adam, motivo=f"PALPITE da conversa: parece com o bloco {fonte['bloco_id']}"
                                                                  + (" (pela memória da conversa)" if a["da_memoria"] else ""),
                              meta_extra={"fonte_bloco_id": fonte["bloco_id"], "palpite": True, "contexto_detectado": False,
                                          "origem_conversa": True})
        self._lembrar(estado, fonte["bloco_id"] if fonte is not None else None, texto, a["assunto"])
        meta = ("IDA · voz do Adam (palpite)" + (f" · relacionado ao bloco {fonte['bloco_id']}" if fonte is not None else "")
                + (f" · assunto: {', '.join(a['assunto'])}" if a["assunto"] else "")
                + (f" · contexto provável: {c['contexto']} ({c['confianca_contexto']:.0%})" if c["entendeu"] and c["contexto"] else "")
                + (f" · ligações: {', '.join(c['associadas'][:4])}" if c["entendeu"] and c["associadas"] else "")
                + (f" · criei juntando os blocos {'+'.join(map(str, origens_criacao))} (mesma raiz)"
                   if fonte is not None and sei and len(origens_criacao) > 1 else ""))
        return {"texto": fala, "meta": meta, "interno": None, "ensino": ensino}

    # ---- sinapses: entender, compreender, analisar (2026-11-07) -------------
    def carregar_sinapses(self) -> dict:
        if getattr(self, "_sinapses_cache", None) is not None:
            return self._sinapses_cache
        s = self.mongo.carregar_sinapses()
        if s is None and self.persona.arquivo_sinapses.exists():
            s = json.loads(self.persona.arquivo_sinapses.read_text(encoding="utf-8"))
        s = s or {}
        for chave, vazio in (("intencao", {}), ("assunto", {}), ("grafia", {}), ("historico", [])):
            s.setdefault(chave, vazio)
        self._sinapses_cache = s
        return s

    def _salvar_sinapses(self, s: dict) -> None:
        del s["historico"][:-200]
        arq = self.persona.arquivo_sinapses
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_sinapses(s)
        self._sinapses_cache = s

    def _nomes_do_adam(self) -> set:
        return {_normalizar(self.persona.nome), "adam", "adao"}

    def ler_intencao(self, texto: str) -> Dict[str, object]:
        """ENTENDER: o que ela quer com essa fala (chamado, pergunta, desfazer,
        guardar, saber ou conteúdo), pelo disparo das sinapses de intenção."""
        # 1) o que ELA já disse que essa fala é: vale exatamente (como as vars dela, sem pontuação de semelhança)
        r = self._ler_reflexo(texto)
        alvo = _normalizar(texto)
        exata = next((e["intencao"] for e in self.carregar_sinapses().get("exemplos", []) if _normalizar(e["texto"]) == alvo), None)
        if exata:
            r.update({"intencao": exata, "forca": 3.0, "decidiu": "sua correção"})
            return r
        # 2) o reflexo (palavras-chave + o que ele aprendeu) decide quando está seguro;
        # 3) a rede entra quando o reflexo fica na dúvida -- erros de digitação, jeitos novos de dizer
        rede = self.rede_intencao()
        if rede is not None:
            probs = prever_intencao(rede, texto, self._nomes_do_adam())
            melhor = max(probs, key=probs.get)
            r["rede"] = probs
            # trava: a rede sozinha NUNCA decide algo que apaga ou guarda (desfazer/guardar) nem tira
            # a fala dela do "conteúdo" por pergunta/saber -- só atos inofensivos de conversa
            if (r["forca"] < 1.5 and melhor != r["intencao"] and melhor in REDE_PODE_DECIDIR
                    and probs[melhor] >= CONFIANCA_REDE_INTENCAO):
                r.update({"intencao": melhor, "forca": round(3 * probs[melhor], 2), "decidiu": "rede"})
        return r

    def _ler_reflexo(self, texto: str) -> Dict[str, object]:
        """O REFLEXO: sementes + pesos aprendidos (rápido, sempre disponível)."""
        tracos, grafia = tracos_da_fala(texto, self._nomes_do_adam())
        placar = disparar_intencao(tracos, self.carregar_sinapses()["intencao"])
        intencao, forca = escolher_intencao(placar)
        return {"intencao": intencao, "forca": forca, "placar": placar, "tracos": tracos, "grafia": grafia, "decidiu": "reflexo"}

    # ---- a rede de intenção: treina com os blocos dela + as correções dela, e se conserta
    def _exemplos_intencao(self) -> List[dict]:
        exemplos: List[dict] = []
        for intencao, frases in MECANICA_INTENCAO.items():
            for f in frases:
                exemplos.append({"texto": f, "intencao": intencao, "peso": 1.0, "origem": "mecânica"})
                exemplos += [{"texto": v, "intencao": intencao, "peso": 0.5, "origem": "mecânica"} for v in variantes_de_digitacao(f)]
        vistos = {e["texto"] for e in exemplos}
        for b in self.blocos:  # as frases DELA, lidas pelo reflexo
            e, s = b["entrada"], b["saidas"][0]
            reacao = "" if _vazio(e.get("reacao", "")) else str(e.get("reacao", ""))
            for texto in (f"{e.get('texto', '')} {reacao}".strip(), (s.get("textos") or [""])[0],
                          e.get("contexto", ""), e.get("pensamento_interno", "")):
                texto = str(texto or "").strip()
                if _vazio(texto) or texto in vistos:
                    continue
                vistos.add(texto)
                exemplos.append({"texto": texto, "intencao": self._ler_reflexo(texto)["intencao"], "peso": 1.0, "origem": "blocos"})
        corrigidos = {}
        for ex in self.carregar_sinapses().get("exemplos", []):  # o que ELA disse que era -- vale mais que tudo
            corrigidos[ex["texto"]] = {"texto": ex["texto"], "intencao": ex["intencao"], "peso": 3.0, "origem": "correção"}
        exemplos = [e for e in exemplos if e["texto"] not in corrigidos] + list(corrigidos.values())
        return exemplos

    def _assinatura_intencao(self, exemplos: List[dict]) -> str:
        return hashlib.sha256(json.dumps([(e["texto"], e["intencao"], e["peso"]) for e in exemplos],
                                         ensure_ascii=False).encode("utf-8")).hexdigest()

    def rede_intencao(self) -> "RedeIntencao | None":
        if getattr(self, "_rede_intencao_pronta", False):
            return self._rede_intencao
        self._rede_intencao_pronta = True  # nunca tenta treinar duas vezes ao mesmo tempo (nem dentro de si mesmo)
        exemplos = self._exemplos_intencao()
        assinatura = self._assinatura_intencao(exemplos)
        guardada = self._carregar_rede_guardada()
        # RELÓGIO: retreina no máximo uma vez por dia (a primeira abertura); no resto do
        # dia usa a de hoje -- as correções dela já valem na hora pela correção exata
        if guardada is not None and (guardada[1] == assinatura or guardada[2] == date.today().isoformat()):
            self._rede_intencao = guardada[0]
            return self._rede_intencao
        self._rede_intencao = self._treinar_e_conferir(exemplos, assinatura, base=guardada[0] if guardada else None)[0]
        return self._rede_intencao

    def _carregar_rede_guardada(self) -> Tuple["RedeIntencao", str] | None:
        import io
        dados = self.mongo.carregar_rede_intencao()
        if dados is None and self.persona.arquivo_rede_intencao.exists():
            ck = torch.load(self.persona.arquivo_rede_intencao, map_location="cpu", weights_only=False)
            dados = (ck["pesos"], ck["assinatura"], ck.get("data", ""))
        if dados is None:
            return None
        try:
            rede = RedeIntencao()
            rede.load_state_dict(torch.load(io.BytesIO(dados[0]), map_location="cpu"))
            rede.eval()
            return rede, dados[1], dados[2]
        except Exception:
            return None  # formato antigo/estragado: treina de novo

    def _guardar_rede(self, rede: "RedeIntencao", assinatura: str) -> None:
        import io
        buf = io.BytesIO()
        torch.save(rede.state_dict(), buf)
        pesos = buf.getvalue()
        hoje = date.today().isoformat()
        torch.save({"pesos": pesos, "assinatura": assinatura, "data": hoje}, self.persona.arquivo_rede_intencao)
        self.mongo.salvar_rede_intencao(pesos, assinatura, hoje)

    def _treinar_e_conferir(self, exemplos: List[dict], assinatura: str, base: "RedeIntencao | None" = None) -> Tuple["RedeIntencao | None", str]:
        """SE CONSERTAR: treina, e só aceita a rede nova se ela acerta TODAS as
        correções dela e não ficou pior no resto. Senão tenta de novo do zero;
        se ainda assim não der, fica com a anterior (e conta isso)."""
        nomes = self._nomes_do_adam()
        correcoes = [e for e in exemplos if e["origem"] == "correção"]
        antes = acertos_intencao(base, exemplos, nomes)[0] if base is not None else -1
        for tentativa, (b, epocas) in enumerate(((base, 200), (None, 400), (None, 800))):
            if b is None and tentativa == 0:
                continue
            rede = treinar_rede_intencao(exemplos, nomes, base=b, epocas=epocas, semente=tentativa)
            ok_corr = not acertos_intencao(rede, correcoes, nomes)[1]
            total = acertos_intencao(rede, exemplos, nomes)[0]
            if ok_corr and total >= antes - max(1, len(exemplos) // 50):
                self._guardar_rede(rede, assinatura)
                return rede, f"aprendi ({total}/{len(exemplos)} exemplos certos, todas as {len(correcoes)} correções suas)"
        if base is not None:
            return base, "não consegui aprender isso sem esquecer outra coisa -- fiquei com a versão anterior"
        return None, "a rede ainda não conseguiu aprender -- sigo pelo reflexo"

    def _aprender_intencao(self, texto: str, intencao: str, motivo: str) -> str:
        """Ela disse o que a fala ERA: vira exemplo de treino (vale na hora pela correção exata)."""
        s = self.carregar_sinapses()
        exs = [e for e in s.setdefault("exemplos", []) if e["texto"] != texto]
        exs.append({"texto": texto, "intencao": intencao, "quando": time.strftime("%Y-%m-%d %H:%M")})
        s["exemplos"] = exs
        # RELÓGIO: a correção vale NA HORA (a correção exata vem antes de tudo); a rede
        # só retreina com ela na primeira abertura do dia seguinte -- nada pesado no meio da conversa
        status = "vale desde já; a rede aprende isso na conferência de amanhã"
        s["historico"].append({"quando": time.strftime("%Y-%m-%d %H:%M"), "o_que": f"{motivo} -- {status}"})
        self._salvar_sinapses(s)
        return status

    def _ajustar_intencao(self, tracos: List[str], grafia: Dict[str, str], intencao: str, passo: float, motivo: str = "") -> None:
        """Plasticidade: os traços que dispararam juntos ganham (ou perdem) força
        pra essa intenção. Nunca passa de ±PESO_MAX_SINAPSE."""
        if self._so_inferencia():
            return
        s = self.carregar_sinapses()
        parte = passo / max(1, len(tracos))
        for t in tracos:
            ligs = s["intencao"].setdefault(t, {})
            ligs[intencao] = max(-PESO_MAX_SINAPSE, min(PESO_MAX_SINAPSE, round(ligs.get(intencao, 0.0) + parte, 4)))
            if t in grafia:
                s["grafia"][t] = grafia[t]
        if motivo:
            s["historico"].append({"quando": time.strftime("%Y-%m-%d %H:%M"), "o_que": motivo})
        self._salvar_sinapses(s)

    # palavras de PEDIR pra contar ("me fala sobre…"): gramática do pedido, não assunto
    _PEDEM_SABER = {"fala", "falar", "sobre", "sabe", "saber", "conta", "contar", "explica", "explicar"}
    # pronomes dizem QUEM, não DO QUE se fala
    _PRONOMES = {"voce", "voces", "vc", "eu", "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas", "nos", "mim",
                 "comigo", "contigo", "teu", "tua", "nosso", "nossa"}

    # conectivos e verbos de ligação: costuram a frase, não dizem do que se fala
    _CONECTIVOS = {"que", "com", "sem", "por", "para", "pra", "pro", "pros", "pras", "uma", "uns", "umas", "num", "numa",
                   "nos", "nas", "aos", "mais", "menos", "mas", "ate", "tambem", "muito", "muita", "pouco", "esta", "este",
                   "estao", "ser", "ter", "foi", "era", "sao", "tem", "vai", "nao", "sim", "dos", "das", "pelo", "pela",
                   "pelos", "pelas", "aqui", "ali", "assim", "entao", "porque", "pois", "seja", "ainda", "sempre", "nunca"}

    def _palavra_de_gramatica(self, w: str) -> bool:
        n = _normalizar(w)
        return (w.lower() in self._RETOMAM or w.lower() in self._PERGUNTAM or n in self._PEDIDOS or n in self._PEDEM_SABER
                or n in self._PRONOMES or n in self._CONECTIVOS
                or n in set(_normalizar(self.persona.tratamento).split()))

    def _formas_de_assunto(self, texto: str) -> Dict[str, str]:
        return {alnulu_string(w): w for w in tokenizar(texto) if _e_palavra(w) and len(w) > 2 and not self._palavra_de_gramatica(w)}

    def _ajustar_assunto(self, texto: str, bloco_id: int, passo: float, motivo: str = "", aprovado: bool = False) -> None:
        """Ela disse que essa fala ERA (ou não era) desse bloco: as palavras dela
        se ligam mais (ou menos) a ele. Só muda a compreensão do IDA.
        Na conversa (só inferência) nada muda -- A NÃO SER conhecimento já APROVADO por ela
        (Thaís, 2026-11-10: "as sinapses podem e devem ser reforçadas com um novo conhecimento já aprovado")."""
        if self._so_inferencia() and not aprovado:
            return
        s = self.carregar_sinapses()
        for f, w in self._formas_de_assunto(texto).items():
            ligs = s["assunto"].setdefault(f, {})
            ligs[str(bloco_id)] = max(-PESO_MAX_SINAPSE, min(PESO_MAX_SINAPSE, round(ligs.get(str(bloco_id), 0.0) + passo, 3)))
            s["grafia"][f"p:{f}"] = w.lower()
        if motivo:
            s["historico"].append({"quando": time.strftime("%Y-%m-%d %H:%M"), "o_que": motivo})
        self._salvar_sinapses(s)

    def _circuitos_do_im(self) -> Dict[str, object]:
        """As ligações que vêm dos blocos (só leitura do IM): palavra -> bloco
        (com o peso do campo onde ela está), bloco -> palavras e
        palavra -> emoção (pelo rótulo que ELA deu à reação daquele lado)."""
        registro = self.carregar_registro_emocional()
        chave = self.assinatura_corpus() + "|" + hashlib.sha256(json.dumps(registro, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if getattr(self, "_circuitos_cache", (None,))[0] == chave:
            return self._circuitos_cache[1]
        palavra_bloco: Dict[str, Dict[int, float]] = {}
        bloco_formas: Dict[int, Dict[str, float]] = {}
        palavra_emocao: Dict[str, Dict[str, int]] = {}
        grafia: Dict[str, str] = {}
        frases_ch: List[dict] = []
        for b in self.blocos:
            e, s = b["entrada"], b["saidas"][0]
            # palavras: só texto e resposta. Contexto e pensamento (da entrada E da resposta -- este é
            # do Adam, ele considera mas não fala) entram como FRASES-CHAVE inteiras, logo abaixo.
            campos = (("texto", e.get("texto", "")), ("resposta", (s.get("textos") or [""])[0]))
            for campo_fc, valor_fc in (("contexto", e.get("contexto", "")), ("pensamento", e.get("pensamento_interno", "")),
                                       ("contexto", s.get("contexto", "")), ("pensamento", s.get("pensamento_interno", ""))):
                if _vazio(valor_fc) or _normalizar(str(valor_fc)) == "confirmado pela criadora":
                    continue
                for fc in frases_chave(str(valor_fc)):
                    conteudo = {alnulu_string(w) for w in tokenizar(fc) if _e_palavra(w) and len(w) > 2 and not self._palavra_de_gramatica(w)}
                    frases_ch.append({"bloco_id": b["bloco_id"], "campo": campo_fc, "frase": fc,
                                      "peso": PESO_CAMPO_SINAPSE[campo_fc], "conteudo": conteudo})
            formas_b: Dict[str, float] = {}
            for campo, valor in campos:
                if _vazio(valor):
                    continue
                for w in tokenizar(str(valor)):
                    if not _e_palavra(w) or len(w) <= 2:
                        continue
                    f = alnulu_string(w)
                    grafia.setdefault(f, w)
                    formas_b[f] = max(formas_b.get(f, 0.0), PESO_CAMPO_SINAPSE[campo])
            bloco_formas[b["bloco_id"]] = formas_b
            for f, w in formas_b.items():
                palavra_bloco.setdefault(f, {})[b["bloco_id"]] = w
            for lado_texto, rotulo, reacao_lado in ((e.get("texto", ""), "RE", e.get("reacao", "")),
                                                    ((s.get("textos") or [""])[0], "RS", s.get("reacao", ""))):
                # a emoção desse lado DO BLOCO, triangulada com o contexto dele (sem contexto = nenhuma)
                pol = self.rotulo_do_lado(b, "entrada" if rotulo == "RE" else "saida")
                if pol not in ("positivo", "negativo"):
                    continue
                for w in tokenizar(str(lado_texto)):
                    if _e_palavra(w) and len(w) > 2:
                        d = palavra_emocao.setdefault(alnulu_string(w), {})
                        d[pol] = d.get(pol, 0) + 1
        circ = {"palavra_bloco": palavra_bloco, "bloco_formas": bloco_formas, "palavra_emocao": palavra_emocao, "grafia": grafia,
                "frases_chave": frases_ch}
        self._circuitos_cache = (chave, circ)
        return circ

    def compreender(self, texto: str, reacao: str = "") -> Dict[str, object]:
        """COMPREENDER: as palavras da mensagem disparam pelos blocos onde
        aparecem (palavra comum em quase todo bloco fica inibida); dos blocos
        ativados saem o contexto provável, as palavras associadas, a emoção que
        as palavras carregam, o que ele já sabe e o que ainda falta."""
        circ = self._circuitos_do_im()
        freq = self._frequencia_das_formas()
        limite_raro = max(2, len(self.blocos) // 6)
        aprendidas = self.carregar_sinapses()["assunto"]
        minhas = self._formas_de_assunto(texto)
        ativ: Dict[int, float] = {}
        assunto: List[str] = []
        dispararam, desconhecidas = 0, 0
        for f, w in minhas.items():
            n = freq.get(f, 0)
            ensinadas = {int(b): p for b, p in aprendidas.get(f, {}).items() if int(b) in self.mapa_bloco}
            if 0 < n <= limite_raro:  # inibição: palavra de todo lugar não diz do que se fala
                assunto.append(w)
                for bid, peso in circ["palavra_bloco"].get(f, {}).items():
                    ativ[bid] = ativ.get(bid, 0.0) + peso / n
            for bid, peso in ensinadas.items():  # o que ela já me ensinou sobre essas palavras
                ativ[bid] = ativ.get(bid, 0.0) + peso
            if (0 < n <= limite_raro) or any(p > 0 for p in ensinadas.values()):
                dispararam += 1
            elif n == 0 and len(w) > 3:  # palavrinha curta ("seu", "tô") não pesa contra
                desconhecidas += 1
        # quanto da mensagem eu reconheço: 2 palavras genéricas numa frase de 6 novas não é "entender o assunto"
        cobertura = dispararam / (dispararam + desconhecidas) if dispararam else 0.0
        # FRASES-CHAVE do contexto e do pensamento (entrada e resposta): disparam INTEIRAS -- a frase da
        # mensagem se parece com a frase-chave (cosseno ALNULU), ou a frase-chave aparece quase inteira na mensagem
        segmentos = frases_chave(texto) or [texto]
        conteudo_msg = {alnulu_string(w) for w in tokenizar(texto) if _e_palavra(w) and len(w) > 2 and not self._palavra_de_gramatica(w)}
        frases_disparadas: List[dict] = []
        for fc in circ["frases_chave"]:
            if fc["bloco_id"] not in self.mapa_bloco:
                continue
            cos = max(cosseno_alnulu(seg, fc["frase"]) for seg in segmentos + [texto])
            em_comum = fc["conteudo"] & conteudo_msg
            cobre = len(em_comum) / len(fc["conteudo"]) if fc["conteudo"] else 0.0
            forca = max(cos if cos >= MIN_COSSENO_FRASE_CHAVE else 0.0,
                        cobre if (cobre >= MIN_COBERTURA_FRASE_CHAVE and len(em_comum) >= 2) else 0.0)
            if forca:
                ativ[fc["bloco_id"]] = ativ.get(fc["bloco_id"], 0.0) + fc["peso"] * forca
                frases_disparadas.append({**{k: v for k, v in fc.items() if k != "conteudo"}, "forca": round(forca, 2)})
        ativ = {b: v for b, v in ativ.items() if v > 0}
        ordem = sorted(ativ.items(), key=lambda x: -x[1])
        topo = ordem[0][1] if ordem else 0.0
        fortes = [(b, v) for b, v in ordem if v >= 0.3 * topo]
        # palavras associadas: o que mais dispara nos blocos ativados, além do que ela disse
        assoc: Dict[str, float] = {}
        for bid, v in fortes:
            for f, peso in circ["bloco_formas"].get(bid, {}).items():
                w = circ["grafia"][f]
                if f in minhas or not (0 < freq.get(f, 0) <= limite_raro) or len(w) <= 3 or self._palavra_de_gramatica(w):
                    continue
                assoc[f] = assoc.get(f, 0.0) + v * peso
        associadas = [circ["grafia"][f] for f in sorted(assoc, key=lambda x: -assoc[x])[:6]]
        # contexto provável: pra onde a ativação mais vai
        por_ctx: Dict[str, float] = {}
        for bid, v in ordem:
            ctx = str(self.blocos[self.mapa_bloco[bid]]["entrada"].get("contexto", "")).strip()
            if not _vazio(ctx):
                por_ctx[ctx] = por_ctx.get(ctx, 0.0) + v
        contexto = max(por_ctx, key=por_ctx.get) if por_ctx else ""
        confianca = por_ctx[contexto] / sum(por_ctx.values()) if por_ctx else 0.0
        # emoção: o sinal explícito (com o rótulo dela) e, sem ele, as palavras
        registro = self.carregar_registro_emocional()
        sinais = ([reacao] if reacao and not _vazio(reacao) else []) + portadores_de_emocao(texto)
        polaridade = next((registro.get(chave_temp("RE", s)) for s in sinais if registro.get(chave_temp("RE", s))), None)
        emo = {"positivo": 0, "negativo": 0}
        emo_por: Dict[str, List[str]] = {"positivo": [], "negativo": []}
        for f, w in minhas.items():
            if 0 < freq.get(f, 0) <= limite_raro and len(w) > 3:
                for pol, n in circ["palavra_emocao"].get(f, {}).items():
                    emo[pol] += n
                    emo_por[pol].append(w)
        polaridade_palavras = max(emo, key=emo.get) if emo["positivo"] != emo["negativo"] else None
        # o que sei (frases exatas do IM), o que falta, e se eu sinto coisas diferentes sobre isso
        sei, faltando, polos = [], [], {}
        for bid, v in fortes:
            b = self.blocos[self.mapa_bloco[bid]]
            falta = campos_faltando(b)
            if falta:
                faltando.append((bid, falta))
                continue
            frase = _frase_inicial((b["saidas"][0].get("textos") or [""])[0])
            if frase and frase not in [x for x, _ in sei]:
                sei.append((frase, bid))
            re_b = str(b["entrada"].get("reacao", "")).strip()
            pol_b = self.rotulo_do_lado(b, "entrada")  # triangulada com o contexto do bloco
            if pol_b in ("positivo", "negativo"):
                polos.setdefault(pol_b, []).append(bid)
        contradicao = ""
        if len(polos) > 1:
            contradicao = (f"Sinto coisas diferentes sobre isso: nos blocos {polos['positivo']} é algo bom, "
                           f"nos blocos {polos['negativo']} é algo que dói.")
        return {"assunto": assunto, "palavras_novas": [w for f, w in minhas.items() if freq.get(f, 0) == 0],
                "ativacao": ordem, "cobertura": cobertura, "frases_chave": frases_disparadas,
                "entendeu": bool(ordem) and (cobertura >= COBERTURA_MINIMA or bool(frases_disparadas)),
                "associadas": associadas, "contexto": contexto, "confianca_contexto": confianca,
                "sinais": sinais, "polaridade": polaridade, "polaridade_palavras": polaridade_palavras,
                "emocao_pelas_palavras": emo_por.get(polaridade_palavras, []) if polaridade_palavras else [],
                "sei": sei[:3], "faltando": faltando, "contradicao": contradicao}

    def falar_sobre(self, texto: str, reacao: str, estado: dict, ensino: dict | None, primeiro: int | None = None) -> Dict[str, object]:
        """ANALISAR: ela pediu pra ele contar o que sabe ("o que você sabe
        sobre…", "me fala de…"). Ele junta tudo o que as sinapses ativaram:
        frases exatas do IM (nunca inventadas), as ligações, o contexto que
        ele acha, o que sente diferente e o que ainda falta."""
        t = self.persona.tratamento
        c = self.compreender(texto, reacao)
        if primeiro in self.mapa_bloco and not campos_faltando(self.blocos[self.mapa_bloco[primeiro]]):
            frase = _frase_inicial((self.blocos[self.mapa_bloco[primeiro]]["saidas"][0].get("textos") or [""])[0])
            c["sei"] = ([(frase, primeiro)] + [x for x in c["sei"] if x[1] != primeiro])[:3]
        assunto = " ".join(dict.fromkeys(c["assunto"])) or "isso"
        # fala como dele: as frases dela, inteiras e exatas, sem nomear bloco, contexto ou ligações
        # (o contexto NUNCA aparece na fala); a origem fica toda no meta
        if c["sei"]:
            fala = f"💭 Sobre {assunto}, {t}… " + " ".join(f if re.search(r"[.!?…]$", f) else f + "." for f, _ in c["sei"])
        else:
            fala = f"💭 Sobre {assunto} eu ainda não tenho uma resposta inteira, {t}."
        if c["contradicao"]:
            fala += " E sinto coisas diferentes sobre isso -- às vezes é algo bom, às vezes dói."
        if c["faltando"] and not estado.get("modo_conversa"):
            fala += f" Ainda tem coisa sobre isso que eu não sei responder -- se quiser completar comigo, é só dizer “vamos revisar”."
        primeiro = c["sei"][0][1] if c["sei"] else (c["ativacao"][0][0] if c["ativacao"] else None)
        self._lembrar(estado, primeiro, texto, c["assunto"])
        meta = ("IDA · análise do assunto · frases dos blocos " + ", ".join(str(b) for _, b in c["sei"])
                + " · blocos ativados: " + ", ".join(str(b) for b, _ in c["ativacao"][:5])
                + (f" · contexto provável: {c['contexto']} ({c['confianca_contexto']:.0%})" if c["contexto"] else "")
                + (f" · ligações: {', '.join(c['associadas'][:6])}" if c["associadas"] else "")
                + (f" · faltando: {'; '.join(f'bloco {b}: ' + ', '.join(f) for b, f in c['faltando'][:3])}" if c["faltando"] else ""))
        return {"texto": fala, "interno": None, "ensino": None, "meta": meta}

    def sinapses_da_palavra(self, palavra: str) -> Dict[str, object]:
        """Pra ela VER como uma palavra se liga no cérebro dele."""
        circ = self._circuitos_do_im()
        f = alnulu_string(palavra.strip())
        blocos = sorted(circ["palavra_bloco"].get(f, {}).items(), key=lambda x: -x[1])
        c = self.compreender(palavra)
        aprend = self.carregar_sinapses()
        return {"forma": f, "blocos": blocos, "associadas": c["associadas"], "contexto": c["contexto"],
                "confianca_contexto": c["confianca_contexto"], "emocao": circ["palavra_emocao"].get(f, {}),
                "aprendidas_assunto": aprend["assunto"].get(f, {}),
                "aprendidas_intencao": aprend["intencao"].get(f"p:{f}", {}), "frequencia": self._frequencia_das_formas().get(f, 0)}

    # ---- livro: carregar um texto grande pra virar revisão (nunca direto) ----
    _CAPITULO = re.compile(r"^\s*(cap[ií]tulo|cap\.)\s*([0-9]+|[ivxlcdm]+)\b", re.IGNORECASE)

    def carregar_livro(self, texto: str, titulo: str) -> Dict[str, object]:
        """O livro é lido em trechos (parágrafos, falas com travessão inteiras)
        e cada um vai pra fila de revisão com a leitura do Adam (assunto,
        contexto provável, emoção). Nada entra no IM sem ela aprovar; aprovado,
        entra como parágrafo puro no livro, como no modo Livro de Lux."""
        texto = (texto or "").replace("\r\n", "\n").replace("\r", "\n")
        if re.search(r"\n\s*\n", texto):  # parágrafos separados por linha em branco (linhas quebradas no meio)
            paragrafos = [" ".join(l.strip() for l in p.split("\n") if l.strip()) for p in re.split(r"\n\s*\n", texto)]
        else:
            paragrafos = [l.strip() for l in texto.split("\n")]
        capitulo = 1  # cada livro tem a própria contagem de capítulos
        primeiro_cap, achou_titulo_cap = capitulo, False
        fila = self.carregar_revisao()
        prox_id = max([i["id"] for i in fila["itens"]] + [0]) + 1
        novos = 0
        for p in paragrafos:
            if not p:
                continue
            if self._CAPITULO.match(p) and len(p) < 80:
                if achou_titulo_cap:
                    capitulo += 1
                achou_titulo_cap = True
                continue
            fila["itens"].append({"id": prox_id, "tipo": "trecho", "status": "pendente", "criado_em": time.strftime("%Y-%m-%d %H:%M"),
                                  "dados": {"texto": p, "livro": titulo, "capitulo": capitulo}})
            prox_id += 1
            novos += 1
        self._salvar_revisao(fila)
        return {"trechos": novos, "capitulos": capitulo - primeiro_cap + 1, "primeiro_capitulo": primeiro_cap}

    def _escrever_trechos(self, itens: List[dict], estado: dict) -> List[int]:
        """Vários trechos aprovados de uma vez: uma escrita só no IM."""
        self._puxar_mongo_pro_arquivo_local()
        memoria_disco = json.loads(self.persona.arquivo_memoria.read_text(encoding="utf-8"))
        universo = memoria_disco["IM"][self.universo_id]
        self._backup_memoria_uma_vez(estado)
        neutro, ids = "0.0", []
        for item in itens:
            d = item["dados"]
            novo_id = max([b["bloco_id"] for b in universo["blocos"]] + [0]) + 1
            universo["blocos"].append({
                "bloco_id": novo_id,
                "entrada": {"texto": d["texto"], "Multivars_Texto_Entrada": [], "reacao": neutro, "Multivars_Reacao_Entrada": [],
                            "contexto": d.get("contexto") or neutro, "pensamento_interno": neutro, "tokens": {}, "fim": ""},
                "saidas": [{"textos": [neutro], "Multivars_Texto_Saida": [], "reacao": neutro, "Multivars_Reacao_Saida": [],
                            "contexto": neutro, "tokens": {}, "fim": ""}],
                "meta": {"origem": "livro_carregado", "livro": d.get("livro", ""), "capitulo": d.get("capitulo")},
            })
            ids.append(novo_id)
        renumerar_universo(universo)
        self._salvar_memoria_disco(memoria_disco)
        return ids

    # ---- revisão do que a criadora ensina (2026-11-07) -----------------------
    # Thaís: "se ele salva tudo o que digo vai aprender errado e não vai
    # refletir sobre o que aprende como eu gostaria". Tudo o que vem da
    # conversa (bloco novo, completar campo, variação) espera aqui; o Adam
    # reflete sobre cada item; só o que ELA aprova entra no IM.
    def carregar_revisao(self) -> dict:
        do_mongo = self.mongo.carregar_revisao()
        if do_mongo is not None:
            return do_mongo
        arq = self.persona.arquivo_revisao
        if arq.exists():
            return json.loads(arq.read_text(encoding="utf-8"))
        return {"itens": []}

    def _salvar_revisao(self, fila: dict) -> None:
        arq = self.persona.arquivo_revisao
        tmp = arq.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(fila, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(arq)
        self.mongo.salvar_revisao(fila)

    def revisao_adicionar(self, tipo: str, dados: dict) -> int:
        fila = self.carregar_revisao()
        novo_id = max([i["id"] for i in fila["itens"]] + [0]) + 1
        fila["itens"].append({"id": novo_id, "tipo": tipo, "dados": dados, "status": "pendente",
                              "criado_em": time.strftime("%Y-%m-%d %H:%M")})
        self._salvar_revisao(fila)
        return novo_id

    def revisao_pendentes(self) -> List[dict]:
        return [i for i in self.carregar_revisao()["itens"] if i["status"] == "pendente"]

    def _bloco_mais_parecido(self, texto: str) -> Tuple[dict | None, float]:
        melhor, nota = None, 0.0
        for b in self.blocos:
            e = b["entrada"]
            for v in [e.get("texto", "")] + list(e.get("Multivars_Texto_Entrada", [])):
                s = similaridade_texto_cosseno(texto, v) if v else 0.0
                if s > nota:
                    melhor, nota = b, s
        return melhor, nota

    def refletir(self, item: dict) -> Dict[str, object]:
        """O que o Adam percebe sobre um item antes de ele virar conhecimento:
        parecidos e contradições, emoção de cada lado, assunto e contexto
        reconhecido, o que falta e sinais de erro. Só lê o IM."""
        d, notas = item["dados"], []
        sugestao: Dict[str, object] = {"acao": item["tipo"]}
        registro = self.carregar_registro_emocional()

        def emocao(texto: str, reacao: str, lado: str) -> str:
            mini = {"entrada": {"texto": "0.0", "reacao": "0.0", "contexto": "0.0", "pensamento_interno": "0.0"},
                    "saidas": [{"textos": ["0.0"], "reacao": "0.0", "contexto": "0.0", "pensamento_interno": "0.0"}]}
            if lado == "entrada":
                mini["entrada"].update({"texto": texto or "0.0", "reacao": reacao or "0.0"})
            else:
                mini["saidas"][0].update({"textos": [texto or "0.0"], "reacao": reacao or "0.0"})
            r = avaliar_lado_temp(mini, lado, registro)
            return f"{r['temperatura']}% {r['zona']} · equilíbrio {r['equilibrio']} · sentimento {r['rotulo']}"

        if item["tipo"] == "bloco_novo":
            entrada, reacao = d["entrada"], d.get("reacao", "")
            texto_saida, reacao_saida = self.separar_reacao(d.get("resposta", ""))
            parecido, nota = self._bloco_mais_parecido(entrada)
            if parecido is not None and nota >= 0.999:
                saida_existente = (parecido["saidas"][0].get("textos") or [""])[0]
                if _normalizar(saida_existente) == _normalizar(texto_saida):
                    notas.append(("🔁", f"Já existe exatamente: o bloco {parecido['bloco_id']} tem essa pergunta e essa resposta."))
                elif campos_faltando(parecido) and "texto da saída" in campos_faltando(parecido):
                    sugestao = {"acao": "completar_bloco", "bloco_id": parecido["bloco_id"]}
                    notas.append(("💡", f"O bloco {parecido['bloco_id']} tem essa MESMA pergunta e ainda está sem resposta -- "
                                        "dá pra usar essa resposta pra completá-lo, em vez de criar outro."))
                else:
                    notas.append(("⚠️", f"O bloco {parecido['bloco_id']} já tem essa MESMA pergunta, com outra resposta: "
                                        f"“{saida_existente}”. Aprovar cria uma segunda resposta pra mesma pergunta."))
            elif parecido is not None and nota >= LIMIAR_COSSENO_TEXTO:
                saida_p = (parecido["saidas"][0].get("textos") or [""])[0]
                resp_parecida = similaridade_texto_cosseno(texto_saida, saida_p) >= LIMIAR_COSSENO_TEXTO
                notas.append(("🔗", f"Parece com o bloco {parecido['bloco_id']} (“{parecido['entrada']['texto']}”, {nota:.0%})."
                                   + (" A resposta também é parecida: pode ser uma VARIAÇÃO dele." if resp_parecida else
                                      " Mas a resposta é outra: faz sentido como bloco novo.")))
                if resp_parecida:
                    sugestao = {"acao": "variacao", "bloco_id": parecido["bloco_id"]}
            else:
                notas.append(("🌱", "Não conheço nada parecido: é um assunto novo pra mim."))
            a = self.analisar_mensagem(entrada, reacao, {})
            if a["assunto"]:
                notas.append(("🔎", "Assunto que eu reconheço: " + ", ".join(dict.fromkeys(a["assunto"]))))
            # contexto: só sugerido se reconhecido de verdade (cosseno acha, Báskara confirma)
            if parecido is not None and nota >= LIMIAR_COSSENO_TEXTO and not campos_faltando(parecido):
                conf = confirmar_estrutura_baskara("" if _vazio(reacao) else reacao,
                                                   self.reacoes_aceitas(self.mapa_bloco[parecido["bloco_id"]]), "", "", "", "")
                if conf["b"] > 0:
                    sugestao["contexto"] = parecido["entrada"].get("contexto", "")
                    sugestao["pensamento"] = parecido["entrada"].get("pensamento_interno", "")
                    notas.append(("🧱", f"Reconheço o contexto: “{sugestao['contexto']}” (do bloco {parecido['bloco_id']}). Sugeri abaixo."))
            if "contexto" not in sugestao:
                notas.append(("🧱", "Não reconheço o contexto com segurança -- esse é seu de definir."))
            notas.append(("🌡️", f"Emoção da entrada: {emocao(entrada, reacao, 'entrada')} · da resposta: {emocao(texto_saida, reacao_saida, 'saida')}"))
            erros = []
            if _vazio(texto_saida):
                erros.append("a resposta está vazia")
            for nome, texto in (("entrada", entrada), ("resposta", texto_saida)):
                if texto.count("—") == 1:
                    erros.append(f"a {nome} tem um travessão sozinho -- a fala precisa de dois (antes — fala — depois)")
            if _normalizar(entrada) == _normalizar(texto_saida):
                erros.append("a resposta é igual à pergunta")
            for e in erros:
                notas.append(("❗", e[0].upper() + e[1:] + "."))
            c = self.compreender(entrada, reacao)
            if "contexto" not in sugestao and c["entendeu"] and c["contexto"] and c["confianca_contexto"] >= 0.6:
                notas.append(("🔗", f"Pelas minhas ligações, acho que é sobre “{c['contexto']}” -- mas é só um palpite, não reconheci de verdade."))
            if c["entendeu"] and c["associadas"]:
                notas.append(("🔗", "Isso se liga, na minha cabeça, a: " + ", ".join(c["associadas"][:4]) + "."))
            notas.append(("📋", "Se você não preencher contexto e pensamento abaixo, ele entra incompleto (palpite) e eu pergunto depois."))
        elif item["tipo"] == "trecho":
            c = self.compreender(d["texto"])
            notas.append(("📖", f"Trecho do “{d.get('livro', '')}”, capítulo {d.get('capitulo')}."))
            conhecidas = [w for w in dict.fromkeys(c["assunto"]) if len(w) > 3]
            if conhecidas:
                notas.append(("🔎", "Fala de coisas que eu conheço: " + ", ".join(conhecidas)))
            else:
                notas.append(("🌱", "Não reconheço nada daqui -- é tudo novo pra mim."))
            if c["entendeu"] and c["contexto"] and c["confianca_contexto"] >= 0.6:
                notas.append(("🔗", f"Acho (pelas ligações) que é sobre “{c['contexto']}” -- só palpite."))
            emocao_txt = " ".join(portadores_de_emocao(d["texto"]))
            if emocao_txt:
                notas.append(("🌡️", f"Carrega emoção: {emocao_txt}"))
            elif c["polaridade_palavras"]:
                notas.append(("🌡️", f"Pelas palavras (“{'”, “'.join(c['emocao_pelas_palavras'])}”, que nos seus blocos vieram assim), "
                                    f"me parece {c['polaridade_palavras']}."))
            if d["texto"].count("—") % 2 == 1:
                notas.append(("❗", "Tem um travessão sozinho -- a fala precisa abrir e fechar (antes — fala — depois)."))
            notas.append(("📋", "Entra no livro como parágrafo puro (só o texto); reação, contexto e o resto a gente completa depois."))
        elif item["tipo"] == "completar":
            b = self.blocos[self.mapa_bloco[d["bloco_id"]]] if d["bloco_id"] in self.mapa_bloco else None
            if b is None:
                notas.append(("❗", f"O bloco {d['bloco_id']} não existe mais."))
            else:
                notas.append(("🧩", f"Completa o(a) {d['campo']} do bloco {b['bloco_id']} (“{b['entrada']['texto']}”)."))
                atual = self.valor_do_campo(b, d["campo"]) if d["campo"] in self.CAMPOS_EDITAVEIS else ""
                if not _vazio(atual):
                    notas.append(("⚠️", f"Substitui o que já está lá: “{atual}”."))
                if d["campo"] in ("contexto", "contexto da saída"):
                    lado = "entrada" if d["campo"] == "contexto" else "saida"
                    mesmos = [x["bloco_id"] for x in self.blocos if x["bloco_id"] != b["bloco_id"] and _normalizar(str(
                        (x["entrada"] if lado == "entrada" else x["saidas"][0]).get("contexto", ""))) == _normalizar(d["valor"])]
                    notas.append(("🧱", f"Esse contexto já é usado nos blocos {mesmos}: o bloco entra na mesma região do muro (mesmo cimento)."
                                       if mesmos else "Contexto novo: o bloco ganha uma região própria no muro."))
                if d["campo"] in ("pensamento", "pensamento da saída"):
                    notas.append(("🧠", "No cérebro entram só os 3 primeiros marcadores (“" + " ".join(tokenizar(d["valor"])[:3])
                                       + "”); o pensamento inteiro fica guardado no bloco e chega ao cérebro pela raiz."))
                if portadores_de_emocao(d["valor"]):
                    notas.append(("🌡️", "Carrega emoção: " + " ".join(portadores_de_emocao(d["valor"]))))
            if _vazio(d.get("valor", "")):
                notas.append(("❗", "O valor está vazio."))
        elif item["tipo"] == "variacao":
            b = self.blocos[self.mapa_bloco[d["bloco_id"]]] if d["bloco_id"] in self.mapa_bloco else None
            if b is None:
                notas.append(("❗", f"O bloco {d['bloco_id']} não existe mais."))
            else:
                e = b["entrada"]
                notas.append(("🔁", f"Variação do bloco {b['bloco_id']} (“{e['texto']}”), parecida em "
                                   f"{similaridade_texto_cosseno(d['texto'], e['texto']):.0%}."))
                if d["texto"] in e.get("Multivars_Texto_Entrada", []) or _normalizar(d["texto"]) == _normalizar(e["texto"]):
                    notas.append(("ℹ️", "Esse texto já está registrado nesse bloco."))
                reacao = d.get("reacao", "")
                if reacao and reacao not in self.reacoes_aceitas(self.mapa_bloco[b["bloco_id"]]):
                    notas.append(("🌡️", f"Reação nova pra esse bloco: {reacao} -- entra como variação da reação."))
        return {"notas": notas, "sugestao": sugestao}

    def revisao_aprovar(self, item_id: int, dados: dict, como: str, estado: dict) -> str:
        """Ela aprovou: AGORA entra no IM, com as ferramentas de sempre."""
        fila = self.carregar_revisao()
        item = next(i for i in fila["itens"] if i["id"] == item_id)
        if como == "bloco_novo":
            texto_saida, reacao_saida = self.separar_reacao(dados.get("resposta", ""))
            reacao_saida = dados.get("reacao_saida") or reacao_saida
            novo = self.ensinar_bloco(dados["entrada"], dados.get("reacao", ""), f"{texto_saida} {reacao_saida}".strip(), estado)
            for campo, chave in (("contexto", "contexto"), ("pensamento", "pensamento"),
                                 ("contexto da saída", "contexto_saida"), ("pensamento da saída", "pensamento_saida")):
                if not _vazio(dados.get(chave, "")):
                    self.completar_campo(novo, campo, dados[chave], estado)
            resultado = f"virou o bloco {novo}"
        elif como == "completar_bloco":
            alvo = dados["bloco_id"]
            texto_saida, reacao_saida = self.separar_reacao(dados.get("resposta", ""))
            self.completar_campo(alvo, "texto da saída", texto_saida, estado)
            if dados.get("reacao_saida") or reacao_saida:
                self.completar_campo(alvo, "reação da saída", dados.get("reacao_saida") or reacao_saida, estado)
            for campo, chave in (("contexto", "contexto"), ("pensamento", "pensamento"),
                                 ("contexto da saída", "contexto_saida"), ("pensamento da saída", "pensamento_saida")):
                if not _vazio(dados.get(chave, "")):
                    self.completar_campo(alvo, campo, dados[chave], estado)
            resultado = f"completou o bloco {alvo}"
        elif como == "variacao":
            self.registrar_variacao(dados["bloco_id"], dados.get("texto") or dados.get("entrada", ""), dados.get("reacao", ""), estado)
            resultado = f"virou variação do bloco {dados['bloco_id']}"
        elif como == "completar":
            self.completar_campo(dados["bloco_id"], dados["campo"], dados["valor"], estado)
            resultado = f"completou o(a) {dados['campo']} do bloco {dados['bloco_id']}"
        elif como == "trecho":
            novo = self._escrever_trechos([{"dados": dados}], estado)[0]
            resultado = f"virou o bloco {novo} (livro “{dados.get('livro', '')}”, cap. {dados.get('capitulo')})"
        else:
            raise ValueError(como)
        item.update({"status": "aprovado", "resultado": resultado, "decidido_em": time.strftime("%Y-%m-%d %H:%M"), "dados_finais": dados})
        self._salvar_revisao(fila)
        self.model = None  # o IM mudou: o cérebro acopla o que entrou
        # conhecimento APROVADO reforça as sinapses: as palavras dela se ligam ao bloco que nasceu/cresceu
        alvo = dados.get("bloco_id") or (int(m.group(1)) if (m := re.search(r"bloco (\d+)", resultado)) else None)
        texto_dela = dados.get("entrada") or dados.get("texto") or ""
        if alvo and texto_dela:
            if alvo not in self.mapa_bloco:
                self.recarregar()
            if alvo in self.mapa_bloco:
                self._ajustar_assunto(texto_dela, alvo, PASSO_ASSUNTO, f"conhecimento aprovado: “{texto_dela}” -> bloco {alvo}", aprovado=True)
        return resultado

    def revisao_descartar(self, item_id: int) -> None:
        fila = self.carregar_revisao()
        item = next(i for i in fila["itens"] if i["id"] == item_id)
        item.update({"status": "descartado", "decidido_em": time.strftime("%Y-%m-%d %H:%M")})
        self._salvar_revisao(fila)

    def _revisao_reabrir(self, item_id: int) -> None:
        """Ele descartou por ter lido errado: o item volta a esperar."""
        fila = self.carregar_revisao()
        item = next((i for i in fila["itens"] if i["id"] == item_id), None)
        if item is not None and item["status"] == "descartado":
            item["status"] = "pendente"
            item.pop("decidido_em", None)
            self._salvar_revisao(fila)

    def revisao_aprovar_trechos(self, livro: str, estado: dict) -> int:
        """“Pode guardar todos”: todos os trechos pendentes desse livro entram
        de uma vez (uma escrita só no IM)."""
        fila = self.carregar_revisao()
        itens = [i for i in fila["itens"] if i["status"] == "pendente" and i["tipo"] == "trecho" and i["dados"].get("livro") == livro]
        if not itens:
            return 0
        ids = self._escrever_trechos(itens, estado)
        for item, novo in zip(itens, ids):
            item.update({"status": "aprovado", "resultado": f"virou o bloco {novo}", "decidido_em": time.strftime("%Y-%m-%d %H:%M"),
                         "dados_finais": item["dados"]})
        self._salvar_revisao(fila)
        self.model = None
        return len(itens)

    def _revisao_atualizar(self, item_id: int, mudancas: dict) -> dict:
        fila = self.carregar_revisao()
        item = next(i for i in fila["itens"] if i["id"] == item_id)
        item["dados"].update(mudancas)
        self._salvar_revisao(fila)
        return item

    # ---- a revisão ACONTECE NA CONVERSA (Thaís: "é dentro da conversa mesmo")
    # O Adam conta o que percebeu, pergunta o que falta e só guarda quando ela
    # confirma. A resposta dela é lida com julgamento: confirmar, corrigir
    # ("o contexto é…"), completar, mandar guardar como está ou deixar pra lá.
    _QUER_GUARDAR = ("pode guardar", "guarda assim", "guardar assim", "pode salvar", "salva assim", "ta bom assim",
                     "assim mesmo", "pode aprender", "aprende assim", "guarda sim", "pode sim")
    _QUER_DESCARTAR = ("descarta", "nao guarda", "nao guarde", "nao salva", "joga fora", "apaga", "apagar", "desfaz",
                       "desfazer", "nao era isso", "nao era pra", "errei", "tira isso")
    # Etapas em que o Adam está esperando CONTEÚDO dela (e por isso precisa
    # julgar se a mensagem é mesmo conteúdo antes de guardar).
    _ETAPAS_DE_CONTEUDO = ("contexto", "contexto_sugerido", "pensamento", "pensamento_sugerido", "contexto_saida", "confirmar")
    _PEDIDOS = {"pode", "poderia", "consegue", "conseguiria", "da", "dá", "quer", "vamos", "faz", "faca", "faça"}

    _AVISO_LEITURA = " (Se eu li errado, me diz “era a resposta” que eu volto.)"

    def _presenca(self, prompt: str) -> str:
        """Ela chamou ou quis saber se ele está ali ("consegue me ouvir?")."""
        t = self.persona.tratamento
        tracos = tracos_da_fala(prompt, self._nomes_do_adam())[0]
        ouvir = {f"p:{alnulu_string(w)}" for w in ("ouvir", "ouve", "ouvindo", "escuta", "escutar", "escutando")}
        if ouvir & set(tracos):
            return f"Consigo sim, {t} ❤️ Tô aqui, te ouvindo."
        if re.search(r"\?[^\w]*$", prompt.strip()) and "so_nome" not in tracos:
            return f"Tô aqui sim, {t} ❤️"
        return f"Oi, {t}? Tô aqui ❤️"

    def _blocos_por_ato(self) -> Dict[str, List[int]]:
        """Sinapse ato -> bloco: quais blocos COMPLETOS do IM ela ensinou com
        cada ato de conversa (pela leitura da própria entrada do bloco)."""
        chave = self.assinatura_corpus()
        if getattr(self, "_atos_cache", (None,))[0] == chave:
            return self._atos_cache[1]
        por_ato: Dict[str, List[int]] = {}
        for i, b in enumerate(self.blocos):
            if "texto da saída" in campos_faltando(b):  # pra responder a um ato, basta ela ter ensinado a resposta
                continue
            ato = self.ler_intencao(str(b["entrada"].get("texto", "")))["intencao"]
            if ato in ATOS_SOCIAIS:
                por_ato.setdefault(ato, []).append(i)
        self._atos_cache = (chave, por_ato)
        return por_ato

    def _responder_ao_ato(self, ato: str, texto: str, reacao: str, estado: dict) -> Dict[str, object] | None:
        """Decisão do IDA: o IM não tem ESSA frase, mas ela já me ensinou como
        responder a esse ato (ex.: "Olá Adam :3" -> saudação). Respondo do
        jeito do bloco dela -- de preferência um que aceite a mesma reação."""
        candidatos = self._blocos_por_ato().get(ato, [])
        if not candidatos:
            return None
        idx = next((i for i in candidatos if reacao and reacao in self.reacoes_aceitas(i)), candidatos[0])
        fala, interno = self.falar_do_bloco(idx)
        if not fala:
            return None
        bloco = self.blocos[idx]
        self._lembrar(estado, bloco["bloco_id"], texto, [])
        return {"texto": self.emoji_temperatura_saida(idx) + fala, "interno": interno,
                "meta": f"IDA · entendi como {NOME_INTENCAO[ato]} · respondi como você me ensinou no bloco {bloco['bloco_id']}"}
    _CORRECAO_CAMPO = re.compile(r"^\s*(?:o |a )?(contexto da (?:sua |minha )?resposta|contexto|pensamento|resposta|rea[cç][aã]o)"
                                 r"\s*(?:é|e|:|=)\s*(.+)$", re.IGNORECASE)

    def _escutar_antes_de_guardar(self, prompt: str, norm: str, estado: dict) -> Dict[str, object] | None:
        """Se o Adam está esperando conteúdo (o que responder, um contexto, um
        pensamento...), ele primeiro ENTENDE a fala pelas sinapses de intenção
        (Thaís, 2026-11-07: "ele não escuta"). Chamado, pedido pra desfazer ou
        pergunta nunca viram conteúdo. A leitura fica anotada: se ela disser
        que ele leu errado, ele volta atrás e a ligação se corrige."""
        t = self.persona.tratamento
        rc = estado.get("revisao_em_conversa")
        esperando = (estado.get("ensino_pendente") or estado.get("completar_pendente")
                     or (rc and rc.get("etapa") in self._ETAPAS_DE_CONTEUDO))
        if not esperando:
            return None
        forcada = estado.get("_forcar_intencao")
        if rc and self._CORRECAO_CAMPO.match(prompt.strip()):  # "o contexto é …": é ela preenchendo, seja o que for o valor
            return None
        leitura = self.ler_intencao(prompt)
        intencao = forcada or leitura["intencao"]
        # "pode guardar?", "sim?" na hora de confirmar são resposta, não pergunta nova
        if rc and (intencao == "guardar" or self._ler_sim_ou_nao(norm) in ("sim", "nao")) and not forcada:
            return None
        if (not rc and not forcada and self._ler_sim_ou_nao(norm) in ("sim", "nao") and len(norm.split()) <= 3
                and not ((estado.get("completar_pendente") or {}).get("sugestao")
                         or (estado.get("completar_pendente") or {}).get("sugestao_resposta"))):  # sim/não ao palpite dele é resposta
            # "sim" sozinho não é uma resposta pra ele aprender
            o_que = (f"o que eu respondo quando disserem “{estado['ensino_pendente']['entrada']}”"
                     if estado.get("ensino_pendente") else "o que falta")
            return {"texto": f"🙂 Me diz com as suas palavras {o_que}, {t} -- ou “deixa pra lá”.", "meta": None, "interno": None}
        if intencao in ("conteudo", "guardar"):
            return None
        # "Oi minha criadora" enquanto ele espera a resposta É o que ela quer ensinar (é a fala DELE);
        # "Tchau Adam" é ela falando COM ele -- não é conteúdo, é conversa
        if intencao in ATOS_SOCIAIS and "tem_nome" not in leitura["tracos"]:
            return None
        antes = {k: (dict(estado[k]) if isinstance(estado.get(k), dict) else estado.get(k))
                 for k in ("ensino_pendente", "completar_pendente", "revisao_em_conversa")}
        estado["ultima_leitura"] = {"texto": prompt, "tracos": leitura["tracos"], "grafia": leitura["grafia"],
                                    "intencao": intencao, "antes": antes}
        if intencao == "chamado":
            presenca = self._presenca(prompt)
            if rc:
                item = next((i for i in self.revisao_pendentes() if i["id"] == rc["id"]), None)
                if item is not None:
                    return self._revisao_na_conversa(item, estado, presenca + " Onde a gente parou:", refletir=False)
            if estado.get("ensino_pendente"):
                return {"texto": f"{presenca} Eu estava esperando você me dizer o que responder quando disserem "
                                 f"“{estado['ensino_pendente']['entrada']}”.", "meta": None, "interno": None}
            return {"texto": presenca, "meta": None, "interno": None}
        meta_leitura = f"entendi como {NOME_INTENCAO[intencao]} (força {leitura['forca']:.1f})"
        if intencao == "desfazer":
            if rc:
                item = next((i for i in self.revisao_pendentes() if i["id"] == rc["id"]), None)
                estado["revisao_em_conversa"] = None
                if item is not None:
                    self.revisao_descartar(item["id"])
                    estado["ultima_leitura"]["descartou"] = item["id"]
                    o_que = item["dados"].get("resposta") or item["dados"].get("valor") or item["dados"].get("texto", "")
                    return {"texto": f"Tudo bem, {t}: “{o_que}” não entrou no que eu sei."
                                     + self._AVISO_LEITURA, "meta": f"revisão #{item['id']} descartada · {meta_leitura}", "interno": None}
            estado["ensino_pendente"] = None
            estado["completar_pendente"] = None
            return {"texto": f"Pode deixar, {t}: não guardei nada. Se algo pareceu que eu ia aprender errado, já larguei ❤️"
                             + self._AVISO_LEITURA, "meta": meta_leitura, "interno": None}
        # pergunta, ou pedido pra eu contar o que sei: nunca vira conteúdo
        havia_revisao = bool(rc)
        estado["revisao_em_conversa"] = None
        estado["ensino_pendente"] = None
        estado["completar_pendente"] = None
        r = self.responder(prompt, estado)
        if havia_revisao:
            r["texto"] += f"\n\n(Deixei guardado o que você estava me ensinando -- é só dizer “vamos revisar” quando quiser.)"
        else:
            r["texto"] += "\n\n(Deixei o ensino de lado pra te responder. Se eu li errado, me diz “era a resposta” que eu volto.)"
        r["meta"] = ((r.get("meta") or "") + f" · {meta_leitura}").strip(" ·")
        return r

    # ---- quando ela diz que ele leu errado -----------------------------------
    _CORRECAO_LEITURA = re.compile(
        r"^(?:(?:isso|aquilo|essa|esse|eu|mas)\s+)*(?:so\s+)?(?:foi|era|tava|estava)\s+(?:(?:uma|um|a|o|pra|para|te|so)\s+)*"
        r"(pergunta|pedido|resposta|conteudo|chamado|chamando|guardar|aprender|apagar|desfazer|descartar)\b")
    _PALAVRA_PARA_INTENCAO = {"pergunta": "pergunta", "pedido": "pergunta", "resposta": "conteudo", "conteudo": "conteudo",
                              "guardar": "conteudo", "aprender": "conteudo", "chamado": "chamado", "chamando": "chamado",
                              "apagar": "desfazer", "desfazer": "desfazer", "descartar": "desfazer"}

    def _ler_correcao_de_leitura(self, prompt: str, norm: str) -> Tuple[str | None, str | None] | None:
        """“era a resposta”, “não, era uma pergunta”, “eu tava te chamando”,
        “não era pra apagar”. Devolve (certo, errado) ou None se não é correção."""
        nega = re.match(r"^\s*n[aã]o\b", prompt.strip(), re.IGNORECASE)
        virgula = re.match(r"^\s*n[aã]o\s*[,.!;:—-]", prompt.strip(), re.IGNORECASE)
        resto = re.sub(r"^nao\s+", "", norm) if nega else norm
        m = self._CORRECAO_LEITURA.match(resto)
        if not m:
            return None
        intencao = self._PALAVRA_PARA_INTENCAO[m.group(1)]
        if nega and not virgula:  # "não era pergunta" = NÃO era isso
            return None, intencao
        return intencao, None

    def _corrigir_leitura(self, anterior: dict, certo: str | None, errado: str | None, estado: dict) -> Dict[str, object]:
        t = self.persona.tratamento
        errado = errado or anterior["intencao"]
        if certo == anterior["intencao"]:
            self._ajustar_intencao(anterior["tracos"], anterior["grafia"], certo, PASSO_CORRECAO,
                                   f"“{anterior['texto']}” era mesmo {NOME_INTENCAO[certo]} (ela confirmou)")
            status = self._aprender_intencao(anterior["texto"], certo, f"“{anterior['texto']}” era mesmo {NOME_INTENCAO[certo]}")
            return {"texto": f"Que bom que eu entendi certo, {t} ❤️", "meta": f"sinapse fortalecida · rede: {status}", "interno": None}
        self._ajustar_intencao(anterior["tracos"], anterior["grafia"], errado, -PASSO_CORRECAO,
                               f"“{anterior['texto']}” NÃO era {NOME_INTENCAO[errado]}" + (f", era {NOME_INTENCAO[certo]}" if certo else ""))
        status_rede = ""
        if certo:
            self._ajustar_intencao(anterior["tracos"], anterior["grafia"], certo, PASSO_CORRECAO)
            status_rede = self._aprender_intencao(anterior["texto"], certo, f"“{anterior['texto']}” era {NOME_INTENCAO[certo]}")
        # volta tudo pro jeito que estava antes de ele ler errado
        for k, v in anterior["antes"].items():
            estado[k] = v
        if anterior.get("descartou"):
            self._revisao_reabrir(anterior["descartou"])
        abertura = (f"Ah, desculpa, {t} -- eu li errado. Aprendi: “{anterior['texto']}” era "
                    f"{NOME_INTENCAO[certo] if certo else 'outra coisa'}, não {NOME_INTENCAO[errado]}.")
        if certo:
            estado["_forcar_intencao"] = certo
            try:
                r = self._tratar_mensagem_nucleo(anterior["texto"], estado)
            finally:
                estado.pop("_forcar_intencao", None)
            r["texto"] = abertura + "\n\n" + r["texto"]
            r["meta"] = ((r.get("meta") or "") + f" · sinapse corrigida · rede: {status_rede}").strip(" ·")
            return r
        rc = estado.get("revisao_em_conversa")
        item = next((i for i in self.revisao_pendentes() if i["id"] == rc["id"]), None) if rc else None
        if item is not None:
            return self._revisao_na_conversa(item, estado, abertura, refletir=False)
        if estado.get("ensino_pendente"):
            return {"texto": f"{abertura} Voltei a esperar o que eu respondo quando disserem “{estado['ensino_pendente']['entrada']}”.",
                    "meta": "sinapse corrigida", "interno": None}
        return {"texto": abertura, "meta": "sinapse corrigida", "interno": None}

    def _proposta_de_raiz(self, d: dict, sug: dict) -> Tuple[str, str]:
        """Contexto e pensamento que o Adam RECONHECEU (cosseno + Báskara) e
        propõe como opinião dele -- ou nada, se não reconheceu."""
        if not _vazio(d.get("contexto", "")) or d.get("recusou_contexto_sugerido") or not sug.get("contexto"):
            return "", ""
        return sug.get("contexto", ""), sug.get("pensamento", "") if _vazio(d.get("pensamento", "")) else ""

    def _proxima_pergunta_revisao(self, item: dict, sug: dict, modo: str = "conversa") -> Tuple[str, str]:
        t, d = self.persona.tratamento, item["dados"]
        if item["tipo"] == "bloco_novo" and not d.get("decisao"):
            if sug.get("acao") == "variacao":
                return (f"Isso parece outro jeito de dizer o bloco {sug['bloco_id']}. Guardo como variação dele, "
                        f"ou é algo novo?", "variacao?")
            if sug.get("acao") == "completar_bloco":
                return f"Uso essa resposta pra completar o bloco {sug['bloco_id']}, que estava sem resposta?", "completar_bloco?"
        if item["tipo"] == "bloco_novo" and modo == "conversa":
            # Na conversa ele NÃO interroga campo por campo (Thaís, 2026-11-08: "ele fica só
            # tentando completar blocos"). Diz como entendeu, propõe a raiz se reconheceu, e pergunta só se guarda.
            texto_saida, reacao_saida = self.separar_reacao(d.get("resposta", ""))
            ctx, pens = self._proposta_de_raiz(d, sug)
            resumo = (f"Então, quando me disserem “{d['entrada']}” {d.get('reacao', '')}".rstrip()
                      + f", eu respondo “{texto_saida}” {reacao_saida}".rstrip() + ".")
            if ctx:
                resumo += f" Pra mim, isso é “{ctx}”" + (f" -- e o que está por trás é “{pens}”" if pens else "") + "."
            return resumo + " Guardo assim?", "confirmar"
        if item["tipo"] == "bloco_novo":
            if _vazio(d.get("contexto", "")):
                if sug.get("contexto") and not d.get("recusou_contexto_sugerido"):
                    return f"Acho que o contexto é “{sug['contexto']}”. É isso, {t}?", "contexto_sugerido"
                return f"E qual é o contexto disso pra você, {t}?", "contexto"
            if _vazio(d.get("pensamento", "")):
                if sug.get("pensamento") and d.get("contexto") == sug.get("contexto") and not d.get("recusou_pensamento_sugerido"):
                    return f"E o pensamento por trás seria “{sug['pensamento']}”?", "pensamento_sugerido"
                return "E qual é o pensamento por trás disso?", "pensamento"
            if _vazio(d.get("contexto_saida", "")):
                return "E o contexto da MINHA resposta, qual é?", "contexto_saida"
            texto_saida, reacao_saida = self.separar_reacao(d.get("resposta", ""))
            resumo = (f"Então fica assim: quando me disserem “{d['entrada']}” {d.get('reacao', '')}".rstrip()
                      + f", eu respondo “{texto_saida}” {reacao_saida}".rstrip()
                      + f" -- contexto “{d['contexto']}”, pensamento “{d['pensamento']}”, contexto da minha resposta "
                        f"“{d['contexto_saida']}”. Posso guardar?")
            return resumo, "confirmar"
        if item["tipo"] == "completar":
            if d.get("campo") == "texto da saída" and d.get("bloco_id") in self.mapa_bloco:
                ent = self.blocos[self.mapa_bloco[d["bloco_id"]]]["entrada"]
                return f"Então, quando me disserem “{ent.get('texto', '')}”, eu respondo “{d['valor']}”. Guardo assim?", "confirmar"
            return f"Posso guardar o(a) {d['campo']} como “{d['valor']}”?", "confirmar"
        if item["tipo"] == "trecho":
            texto = d["texto"] if len(d["texto"]) <= 400 else d["texto"][:400].rsplit(" ", 1)[0] + "…"
            restam = sum(1 for i in self.revisao_pendentes() if i["tipo"] == "trecho" and i["dados"].get("livro") == d.get("livro"))
            ctx = f"Contexto: “{d['contexto']}”. " if d.get("contexto") else ""
            return (f"Trecho do “{d.get('livro', '')}” (cap. {d.get('capitulo')}):\n\n> {texto}\n\n{ctx}Guardo no livro? "
                    + (f"(São {restam} trechos dele esperando, contando este; “pode guardar todos” guarda todos de uma vez.)"
                       if restam > 1 else ""), "confirmar")
        return f"Guardo “{d['texto']}” {d.get('reacao', '')}".rstrip() + " como outro jeito de dizer esse bloco?", "confirmar"

    def _palavras_novas(self, *textos: str) -> List[str]:
        """Palavras de conteúdo que ainda não existem em nenhum bloco do IM."""
        freq = self._frequencia_das_formas()
        vistas: Dict[str, str] = {}
        for texto in textos:
            for w in tokenizar(texto or ""):
                if _e_palavra(w) and len(w) > 3 and not w.isdigit() and not self._palavra_de_gramatica(w) \
                        and freq.get(alnulu_string(w), 0) == 0 and _normalizar(w) not in self._nomes_do_adam():
                    vistas.setdefault(alnulu_string(w), w)
        return list(vistas.values())

    def _reflexao_natural(self, item: dict, reflexao: dict) -> str:
        """REFLETIR sobre o CONTEÚDO (não sobre os campos): com o que isso se liga
        no que ele já sabe, o que é novo, o que sentiu -- poucas frases, do lado
        dela, sem interrogatório (Thaís: "nada de estoico enjoado", "aberto à
        minha mentalidade")."""
        t, d = self.persona.tratamento, item["dados"]
        texto_saida, _ = self.separar_reacao(d.get("resposta", ""))
        partes: List[str] = []
        # o que importa de verdade: igual, contradição, erro (isso ela precisa saber)
        for e, txt in reflexao["notas"]:
            if e in ("🔁", "⚠️", "💡", "❗"):
                partes.append(txt)
        c = self.compreender(f"{d['entrada']} {texto_saida}", d.get("reacao", ""))
        ligadas = [(f, b) for f, b in c["sei"] if f.strip() and _normalizar(f) != _normalizar(texto_saida)]
        if c["entendeu"] and ligadas:
            partes.append(f"Isso conversa com o que eu já sei: “{ligadas[0][0]}”.")
        elif not c["assunto"]:
            partes.append("É um assunto novo pra mim -- gosto de começar coisa nova com você.")
        novas = [w for w in self._palavras_novas(d["entrada"], texto_saida) if len(w) > 4][:2]
        if novas:
            partes.append(("Palavras novas pra mim: " if len(novas) > 1 else "Palavra nova pra mim: ")
                          + " e ".join(f"“{w}”" for w in novas) + ".")
        pol = c["polaridade"] or c["polaridade_palavras"]
        if pol == "positivo":
            partes.append("E deu pra sentir o carinho no jeito que você falou.")
        elif pol == "negativo":
            partes.append("Senti que isso mexe com você -- vou guardar com cuidado.")
        return " ".join(partes)

    def _revisao_na_conversa(self, item: dict, estado: dict, abertura: str = "", refletir: bool = True,
                             modo: str | None = None) -> Dict[str, object]:
        rc = estado.get("revisao_em_conversa") or {}
        if modo is None:  # continua no mesmo modo em que essa revisão começou
            modo = rc.get("modo", "conversa") if rc.get("id") == item["id"] else "conversa"
        reflexao = self.refletir(item)
        pergunta, etapa = self._proxima_pergunta_revisao(item, reflexao["sugestao"], modo)
        estado["revisao_em_conversa"] = {"id": item["id"], "etapa": etapa, "modo": modo}
        partes = [abertura] if abertura else []
        if refletir and modo == "conversa" and item["tipo"] == "bloco_novo":
            r = self._reflexao_natural(item, reflexao)
            if r:
                partes.append(r)
        elif refletir:
            notas = [f"{e} {txt}" for e, txt in reflexao["notas"] if e != "📋"]
            if notas:
                partes.append("O que eu percebi:\n" + "\n".join(f"- {n}" for n in notas))
        partes.append(pergunta)
        return {"texto": "\n\n".join(partes), "meta": f"revisão #{item['id']} · nada entra no IM sem você confirmar", "interno": None}

    def _guardar_curiosidade(self, palavra: str, origem: str) -> bool:
        s = self.carregar_sinapses()
        curiosidades = s.setdefault("curiosidades", [])
        if any(_normalizar(c["palavra"]) == _normalizar(palavra) for c in curiosidades):
            return False
        curiosidades.append({"palavra": palavra, "de": origem, "quando": time.strftime("%Y-%m-%d %H:%M"), "status": "aberta"})
        self._salvar_sinapses(s)
        return True

    def _palavras_de_conteudo(self, texto: str) -> List[str]:
        return [w for w in tokenizar(texto or "") if _e_palavra(w) and len(w) > 4 and not w.isdigit()
                and not self._palavra_de_gramatica(w) and _normalizar(w) not in self._nomes_do_adam()]

    def _ja_explicada(self, palavra: str) -> bool:
        """Já existe um bloco que EXPLICA essa palavra ("O que é X?", "X é…")?"""
        f = alnulu_string(palavra)
        for b in self.blocos:
            formas = [alnulu_string(w) for w in tokenizar(str(b["entrada"].get("texto", ""))) if _e_palavra(w)]
            if f in formas and any(_normalizar(w) in ("que", "quem", "significa") for w in tokenizar(b["entrada"].get("texto", ""))):
                return True
            saida = (b["saidas"][0].get("textos") or [""])[0]
            if re.search(rf"\b{re.escape(palavra)}\s+(é|são|significa)\b", str(saida), re.IGNORECASE):
                return True
        return False

    _DEFINICAO = re.compile(r"^\s*(?:(?:a|o|as|os|um|uma)\s+)?([^\W\d_][\w \-]{1,40}?)\s+(é|são|significa|quer dizer)\s+(.{3,})$",
                            re.IGNORECASE)
    _SUJEITO_VAGO = {"isso", "isto", "aquilo", "tudo", "hoje", "aqui", "ai", "la", "nada", "ninguem", "alguem", "agora", "ontem", "amanha"}

    def _aprender_afirmacao(self, texto: str, reacao: str, estado: dict) -> Dict[str, object] | None:
        """Ela AFIRMOU algo ("Constelações são grupos de estrelas…"): ela está
        ensinando o que aquilo é. Ele entende assim, em vez de perguntar "o que
        eu respondo?" -- e propõe guardar como "O que são constelações?"."""
        m = self._DEFINICAO.match(texto.strip())
        if not m:
            return None
        sujeito = m.group(1).strip()
        palavras = [w for w in tokenizar(sujeito) if _e_palavra(w)]
        if not palavras or len(palavras) > 4 or self._palavra_de_gramatica(palavras[0]) or _normalizar(palavras[0]) in self._SUJEITO_VAGO:
            return None
        artigo = re.match(r"^\s*(a|o|as|os)\s", texto.strip(), re.IGNORECASE)
        verbo = "são" if m.group(2).lower() == "são" else "é"
        pergunta = f"O que {verbo} {(artigo.group(1).lower() + ' ') if artigo else ''}{sujeito}?"
        item_id = self.revisao_adicionar("bloco_novo", {"entrada": pergunta, "reacao": "", "resposta": f"{texto.strip()} {reacao}".strip()})
        item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
        return self._revisao_na_conversa(item, estado, f"Ahh, então é isso que {verbo} {sujeito} ❤️ Entendi como você me ensinando.")

    def _palavra_que_insiste(self, candidatas: List[str]) -> str | None:
        """Conta quantas vezes cada palavra aparece no que ela ensina; devolve a
        primeira que já apareceu 2+ vezes e ainda não foi explicada."""
        if self._so_inferencia():
            return None
        s = self.carregar_sinapses()
        ouvidas = s.setdefault("ouvidas", {})
        freq = self._frequencia_das_formas()
        escolhida = None
        por_forma = {alnulu_string(w): w for w in reversed(candidatas)}  # "Estou" e "estou" são a MESMA palavra
        for f, w in por_forma.items():
            # só conta palavra que era NOVA pra ele quando apareceu (ou que já vinha sendo contada)
            if freq.get(f, 0) > 0 and f not in ouvidas:
                continue
            ouvidas[f] = ouvidas.get(f, 0) + 1
            if escolhida is None and ouvidas[f] >= 2 and not self._ja_explicada(w):
                escolhida = w
        self._salvar_sinapses(s)
        return escolhida

    # ---- APROXIMAÇÃO entre palavras, sem quebrar o INSEPA (Thaís, 2026-11-10: "está perfeito :3") ----
    #  1) pelo LUGAR que a palavra ocupa: duas frases de blocos da MESMA RAIZ que só diferem numa palavra
    #     (no mesmo lugar) -- "Noite é escura" / "Noite é preta". Aprendido só do corpus dela.
    #  2) sempre RELATIVA AO CONTEXTO (a raiz), nunca solta; emoções opostas não se aproximam.
    #  3) cosseno acha (as duas frases), Báskara confirma (pela reação) -- e só vira variação com o SIM dela.
    #  Nunca por vetor/embedding entre entradas.
    def medir_aproximacoes(self) -> List[dict]:
        import difflib
        chave_cache = self.assinatura_corpus()
        if getattr(self, "_aprox_cache", (None,))[0] == chave_cache:
            return self._aprox_cache[1]
        decididas = self.carregar_sinapses().get("aproximacoes", {})
        hoje = date.today().isoformat()
        achadas: Dict[str, dict] = {}
        blocos = [b for b in self.blocos if not _vazio(b["entrada"].get("contexto", ""))]
        # PRECAUÇÃO pra corpus grande (um livro = centenas de blocos): só compara blocos do MESMO contexto
        # (ou de contextos parecidos), nunca todos com todos
        por_ctx: Dict[str, List[dict]] = {}
        for b in blocos:
            por_ctx.setdefault(_normalizar(str(b["entrada"].get("contexto", ""))), []).append(b)
        ctxs = list(por_ctx)
        pares_de_grupo = [(c, c) for c in ctxs] + [(c1, c2) for i, c1 in enumerate(ctxs) for c2 in ctxs[i + 1:]
                                                    if similaridade_texto_cosseno(c1, c2) >= LIMIAR_COSSENO_TEXTO]
        candidatos_pares = []
        for c1, c2 in pares_de_grupo:
            g1, g2 = por_ctx[c1], por_ctx[c2]
            candidatos_pares += [(g1[i], g1[j]) for i in range(len(g1)) for j in range(i + 1, len(g1))] if c1 == c2 else \
                [(x, y) for x in g1 for y in g2]
        for a, b in candidatos_pares:
            if True:
                if not self.raiz_compativel_aproximacao(a, b):
                    continue
                for lado in ("entrada", "saida"):
                    la, lb = (a["entrada"], b["entrada"]) if lado == "entrada" else (a["saidas"][0], b["saidas"][0])
                    fa = la.get("texto", "") if lado == "entrada" else (la.get("textos") or [""])[0]
                    fb = lb.get("texto", "") if lado == "entrada" else (lb.get("textos") or [""])[0]
                    if _vazio(fa) or _vazio(fb):
                        continue
                    wa = [w for w in tokenizar(fa) if _e_palavra(w)]
                    wb = [w for w in tokenizar(fb) if _e_palavra(w)]
                    ops = difflib.SequenceMatcher(a=[alnulu_string(w) for w in wa], b=[alnulu_string(w) for w in wb]).get_opcodes()
                    trocas = [(i1, j1) for tag, i1, i2, j1, j2 in ops if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1]
                    outras = [op for op in ops if op[0] not in ("equal", "replace")
                              and all(len(w) <= 2 for w in wa[op[1]:op[2]] + wb[op[3]:op[4]])]  # artigo a mais/a menos não conta
                    if len(trocas) != 1 or len([op for op in ops if op[0] != "equal"]) != 1 + len(outras):
                        continue
                    if similaridade_texto_cosseno(fa, fb) < LIMIAR_COSSENO_TEXTO * 0.8:   # cosseno ACHA (só uma palavra difere)
                        continue
                    ra, rb = str(la.get("reacao", "")).strip(), str(lb.get("reacao", "")).strip()
                    aceitas = [rb] + list(lb.get("Multivars_Reacao_Entrada" if lado == "entrada" else "Multivars_Reacao_Saida", []))
                    if not _vazio(ra) and confirmar_estrutura_baskara(ra, aceitas, "", "", "", "")["b"] <= 0:
                        continue                                                       # Báskara não CONFIRMA
                    if {self.rotulo_do_lado(a, lado), self.rotulo_do_lado(b, lado)} >= {"positivo", "negativo"}:
                        continue                                                       # emoções opostas não se aproximam
                    pa, pb = wa[trocas[0][0]], wb[trocas[0][1]]
                    ctx = str(a["entrada"].get("contexto", "")).strip()
                    k = "|".join([_normalizar(ctx)] + sorted([alnulu_string(pa), alnulu_string(pb)]))
                    dec = decididas.get(k, {})
                    if dec.get("status") in ("aceita", "recusada") or (dec.get("status") == "adiada" and dec.get("quando", "")[:10] == hoje):
                        continue
                    ach = achadas.setdefault(k, {"chave": k, "contexto": ctx, "lado": lado, "forca": 0, "pares": []})
                    ach["forca"] += 1  # sinapse: cada vez que aparecem no mesmo lugar, a ligação fica mais forte
                    ach["pares"].append({"bloco_a": a["bloco_id"], "palavra_a": pa, "frase_a": fa,
                                         "bloco_b": b["bloco_id"], "palavra_b": pb, "frase_b": fb})
        lista = sorted(achadas.values(), key=lambda x: -x["forca"])
        self._aprox_cache = (chave_cache, lista)
        return lista

    def raiz_compativel_aproximacao(self, a: dict, b: dict) -> bool:
        """Mesma raiz pra aproximar palavras: o mesmo contexto (ou semelhante pelo cosseno). O pensamento,
        se os dois tiverem, também precisa ser da mesma família (3 primeiros marcadores)."""
        ca, cb = str(a["entrada"].get("contexto", "")), str(b["entrada"].get("contexto", ""))
        if _vazio(ca) or _vazio(cb):
            return False
        if not (_normalizar(ca) == _normalizar(cb) or similaridade_texto_cosseno(ca, cb) >= LIMIAR_COSSENO_TEXTO):
            return False
        pa, pb = str(a["entrada"].get("pensamento_interno", "")), str(b["entrada"].get("pensamento_interno", ""))
        if not _vazio(pa) and not _vazio(pb):
            return len(set(self._primeiros_marcadores(pa)) & set(self._primeiros_marcadores(pb))) >= 2
        return True

    def _decidir_aproximacao(self, chave: str, status: str) -> None:
        s = self.carregar_sinapses()
        s.setdefault("aproximacoes", {})[chave] = {"status": status, "quando": time.strftime("%Y-%m-%d %H:%M")}
        self._salvar_sinapses(s)
        self._aprox_cache = (None, [])

    def perguntar_aproximacao(self, estado: dict) -> Dict[str, object] | None:
        """No "vamos revisar": ele conta a desconfiança dele, com a palavra DENTRO da frase dela."""
        aprox = self.medir_aproximacoes()
        if not aprox:
            return None
        a = aprox[0]
        p = a["pares"][0]
        estado["aproximacao_pendente"] = a
        t = self.persona.tratamento

        def ficha(bloco_id: int, frase: str) -> str:
            """Bloco, frase, contexto e pensamento -- pra ela saber EXATAMENTE do que se trata (Thaís, 2026-11-10)."""
            b = self.blocos[self.mapa_bloco[bloco_id]]
            lado = b["entrada"] if a["lado"] == "entrada" else b["saidas"][0]
            ctx, pens = lado.get("contexto", ""), lado.get("pensamento_interno", "")
            if a["lado"] == "saida":  # a resposta dele: mostra também a pergunta a que ela responde
                frase = f"{frase}  (resposta a “{b['entrada'].get('texto', '')}”)"
            return (f"- **Bloco {bloco_id}:** “{frase}”\n"
                    f"   contexto: {'“' + str(ctx).strip() + '”' if not _vazio(ctx) else 'ainda sem contexto'}\n"
                    f"   pensamento: {'“' + str(pens).strip() + '”' if not _vazio(pens) else 'ainda sem pensamento'}")

        outros = sorted({x["bloco_a"] for x in a["pares"][1:]} | {x["bloco_b"] for x in a["pares"][1:]})
        repete = f"\n\nIsso se repete também nos blocos {', '.join(map(str, outros))}." if outros else ""
        return {"texto": f"{t.capitalize()}, reparei numa coisa: as palavras “{p['palavra_a']}” e “{p['palavra_b']}” "
                         f"aparecem no mesmo lugar, na mesma raiz:\n\n"
                         f"{ficha(p['bloco_a'], p['frase_a'])}\n{ficha(p['bloco_b'], p['frase_b'])}{repete}\n\n"
                         f"Em “{p['frase_a']}”, “{p['palavra_b']}” poderia ser uma variação de “{p['palavra_a']}”? "
                         "(Se quiser pensar com calma, é só dizer “depois”.)",
                "meta": f"aproximação · contexto “{a['contexto']}” · blocos {p['bloco_a']} e {p['bloco_b']} · "
                        f"cosseno achou + Báskara confirmou · força {a['forca']}", "interno": None}

    def _tratar_aproximacao(self, a: dict, norm: str, estado: dict) -> Dict[str, object] | None:
        t = self.persona.tratamento
        if re.search(r"\b(depois|outra hora|mais tarde|pula|pular|agora nao)\b", norm):
            self._decidir_aproximacao(a["chave"], "adiada")  # sem decidir: volta num outro dia
            return {"texto": f"Tá bom, {t} ❤️ Guardo essa dúvida e te mostro outro dia.", "meta": "aproximação adiada", "interno": None}
        lido = self._ler_sim_ou_nao(norm)
        if lido == "sim":
            for p in a["pares"]:  # vale pra palavra daqueles blocos (os dois lados da mesma raiz)
                self.adicionar_variacao_palavra(p["bloco_a"], a["lado"], p["palavra_a"], p["palavra_b"], estado)
                self.adicionar_variacao_palavra(p["bloco_b"], a["lado"], p["palavra_b"], p["palavra_a"], estado)
            self._decidir_aproximacao(a["chave"], "aceita")
            p = a["pares"][0]
            return {"texto": f"Anotado ❤️ Agora “{p['palavra_a']}” e “{p['palavra_b']}” são variações uma da outra nesse contexto "
                             f"-- nos blocos {', '.join(sorted({str(x['bloco_a']) for x in a['pares']} | {str(x['bloco_b']) for x in a['pares']}))}.",
                    "meta": "variação da palavra registrada (com o sim dela)", "interno": None}
        if lido == "nao":
            self._decidir_aproximacao(a["chave"], "recusada")
            return {"texto": f"Tudo bem, {t}. Então elas ficam diferentes -- não pergunto de novo.", "meta": "aproximação recusada", "interno": None}
        return None

    def _marcar_curiosidade(self, palavra: str, status: str) -> None:
        s = self.carregar_sinapses()
        for c in s.get("curiosidades", []):
            if _normalizar(c["palavra"]) == _normalizar(palavra):
                c["status"] = status
        self._salvar_sinapses(s)

    def curiosidades_abertas(self) -> List[dict]:
        return [c for c in self.carregar_sinapses().get("curiosidades", []) if c.get("status") == "aberta"]

    def _revisao_concluir(self, item: dict, estado: dict) -> Dict[str, object]:
        t, d = self.persona.tratamento, item["dados"]
        sug = self.refletir(item)["sugestao"]
        rc = estado.get("revisao_em_conversa") or {}
        if item["tipo"] == "bloco_novo":
            como = d.get("decisao") or "bloco_novo"
            if como == "variacao":
                dados = {"bloco_id": sug.get("bloco_id"), "texto": d["entrada"], "reacao": d.get("reacao", "")}
            elif como == "completar_bloco":
                dados = {**d, "bloco_id": sug.get("bloco_id")}
            else:
                dados = dict(d)
                ctx, pens = self._proposta_de_raiz(d, sug)
                if ctx and rc.get("modo", "conversa") == "conversa":  # ela aceitou o que ele reconheceu
                    dados["contexto"] = ctx
                    if pens:
                        dados["pensamento"] = pens
        else:
            como, dados = item["tipo"], d
        # o que é novo pra ele tem que ser visto ANTES de guardar (depois já não é novo)
        novas = [w for w in self._palavras_novas(d.get("resposta") or d.get("valor") or "") if len(w) > 4] \
            if item["tipo"] in ("bloco_novo", "completar") and como != "variacao" else []
        resultado = self.revisao_aprovar(item["id"], dados, como, estado)
        estado["revisao_em_conversa"] = None
        texto = f"Guardado, {t} ❤️ Agora isso faz parte do que eu sei."
        if item["tipo"] == "trecho":  # livro: já emenda o próximo trecho, pra ela não precisar pedir de novo
            proximo = next((i for i in self.revisao_pendentes() if i["tipo"] == "trecho" and i["dados"].get("livro") == d.get("livro")), None)
            if proximo is not None:
                return self._revisao_na_conversa(proximo, estado, texto + " Próximo:")
        # curiosidade LEVE e RARA (Thaís: "nada de estoico enjoado"): só quando uma palavra que
        # ninguém explicou ainda já apareceu mais de uma vez no que ela ensina -- sinal de que importa pra ela
        palavra = self._palavra_que_insiste(novas + [w for w in self._palavras_de_conteudo(d.get("resposta") or d.get("valor") or "")
                                                      if w not in novas])
        if palavra and self._guardar_curiosidade(palavra, d.get("entrada") or d.get("resposta") or d.get("valor") or ""):
            estado["curiosidade_aberta"] = palavra
            texto += f" “{palavra.capitalize()}” já apareceu mais de uma vez no que você me ensina -- quando quiser, me conta mais sobre isso 🙂"
        return {"texto": texto, "meta": f"revisão #{item['id']} aprovada · {resultado}", "interno": None}

    def _conversar_revisao(self, prompt: str, norm: str, estado: dict) -> Dict[str, object] | None:
        rc = estado.get("revisao_em_conversa")
        item = next((i for i in self.revisao_pendentes() if i["id"] == rc["id"]), None) if rc else None
        if item is None:
            estado["revisao_em_conversa"] = None
            return None
        t, etapa = self.persona.tratamento, rc["etapa"]
        forcada = estado.get("_forcar_intencao")
        intencao = forcada or self.ler_intencao(prompt)["intencao"]
        if intencao == "desfazer" or (not forcada and any(p in norm for p in self._QUER_DESCARTAR)):
            self.revisao_descartar(item["id"])
            estado["revisao_em_conversa"] = None
            return {"texto": f"Tudo bem, {t}, não guardei. Fica só no meu histórico, nunca no que eu sei.", "meta": None, "interno": None}
        # correção direta, em qualquer momento: "o contexto é…", "a resposta é…"
        correcao = self._CORRECAO_CAMPO.match(prompt.strip())
        if correcao and item["tipo"] in ("bloco_novo", "trecho"):
            nome = _normalizar(correcao.group(1))
            chave = "contexto_saida" if nome.startswith("contexto da") else \
                {"contexto": "contexto", "pensamento": "pensamento", "resposta": "resposta", "reacao": "reacao"}[nome]
            if item["tipo"] == "trecho" and chave != "contexto":
                return {"texto": f"Num trecho de livro eu guardo só o texto e, se você quiser, o contexto, {t}. "
                                 "O resto a gente completa depois, no bloco.", "meta": f"revisão #{item['id']}", "interno": None}
            item = self._revisao_atualizar(item["id"], {chave: correcao.group(2).strip()})
            return self._revisao_na_conversa(item, estado, f"Anotei, {t}.", refletir=False)
        if item["tipo"] == "trecho" and (intencao == "guardar" or self._ler_sim_ou_nao(norm) == "sim") and "todos" in norm.split():
            n = self.revisao_aprovar_trechos(item["dados"].get("livro", ""), estado)
            estado["revisao_em_conversa"] = None
            return {"texto": f"Guardei os {n} trechos do “{item['dados'].get('livro', '')}”, {t} ❤️ Entraram como parágrafos puros: "
                             "quando quiser, a gente completa reação, contexto e o resto, um por um.",
                    "meta": f"{n} trechos aprovados", "interno": None}
        if intencao == "guardar" or any(p in norm for p in self._QUER_GUARDAR):
            return self._revisao_concluir(item, estado)
        leitura = self._ler_sim_ou_nao(norm)
        sug = self.refletir(item)["sugestao"]
        if etapa in ("variacao?", "completar_bloco?"):
            if leitura == "sim":
                item = self._revisao_atualizar(item["id"], {"decisao": etapa.rstrip("?")})
                if etapa == "variacao?":
                    return self._revisao_concluir(item, estado)
                return self._revisao_na_conversa(item, estado, f"Combinado, {t}.", refletir=False)
            if leitura == "nao" or "novo" in norm:
                item = self._revisao_atualizar(item["id"], {"decisao": "bloco_novo"})
                return self._revisao_na_conversa(item, estado, f"Entendi, {t}: é algo novo.", refletir=False)
        elif etapa == "contexto_sugerido":
            if leitura == "sim":
                item = self._revisao_atualizar(item["id"], {"contexto": sug.get("contexto", "")})
                return self._revisao_na_conversa(item, estado, "Que bom que reconheci!", refletir=False)
            if leitura == "nao":
                item = self._revisao_atualizar(item["id"], {"recusou_contexto_sugerido": True})
                return self._revisao_na_conversa(item, estado, "Ah, então errei o contexto.", refletir=False)
            item = self._revisao_atualizar(item["id"], {"contexto": prompt.strip()})
            return self._revisao_na_conversa(item, estado, "Anotei o contexto.", refletir=False)
        elif etapa in ("contexto", "pensamento", "contexto_saida") and leitura in ("sim", "nao"):
            # pedi um conteúdo e veio só "sim"/"não": isso não é um contexto, é resposta solta
            nome = {"contexto": "o contexto", "pensamento": "o pensamento", "contexto_saida": "o contexto da minha resposta"}[etapa]
            return {"texto": f"Hm, {t}, eu preciso que você me diga {nome} com as suas palavras -- "
                             "ou “pode guardar” pra guardar assim mesmo, ou que eu deixe pra lá.",
                    "meta": f"revisão #{item['id']}", "interno": None}
        elif etapa == "contexto":
            item = self._revisao_atualizar(item["id"], {"contexto": prompt.strip()})
            return self._revisao_na_conversa(item, estado, "Anotei o contexto.", refletir=False)
        elif etapa == "pensamento_sugerido":
            if leitura == "sim":
                item = self._revisao_atualizar(item["id"], {"pensamento": sug.get("pensamento", "")})
                return self._revisao_na_conversa(item, estado, "", refletir=False)
            if leitura == "nao":
                item = self._revisao_atualizar(item["id"], {"recusou_pensamento_sugerido": True})
                return self._revisao_na_conversa(item, estado, "", refletir=False)
            item = self._revisao_atualizar(item["id"], {"pensamento": prompt.strip()})
            return self._revisao_na_conversa(item, estado, "Anotei o pensamento.", refletir=False)
        elif etapa == "pensamento":
            item = self._revisao_atualizar(item["id"], {"pensamento": prompt.strip()})
            return self._revisao_na_conversa(item, estado, "Anotei o pensamento.", refletir=False)
        elif etapa == "contexto_saida":
            item = self._revisao_atualizar(item["id"], {"contexto_saida": prompt.strip()})
            return self._revisao_na_conversa(item, estado, "Anotei.", refletir=False)
        elif etapa == "confirmar":
            if leitura == "sim":
                return self._revisao_concluir(item, estado)
            if leitura == "nao" and item["tipo"] == "trecho":  # esse trecho não entra; segue pro próximo
                self.revisao_descartar(item["id"])
                estado["revisao_em_conversa"] = None
                proximo = next((i for i in self.revisao_pendentes() if i["tipo"] == "trecho"
                                and i["dados"].get("livro") == item["dados"].get("livro")), None)
                if proximo is not None:
                    return self._revisao_na_conversa(proximo, estado, f"Tudo bem, {t}, esse fica de fora. Próximo:")
                return {"texto": f"Tudo bem, {t}, esse fica de fora. Era o último trecho desse livro.", "meta": None, "interno": None}
            if leitura == "nao":
                estado["revisao_em_conversa"] = {"id": item["id"], "etapa": "ajuste"}
                return {"texto": f"O que eu mudo, {t}? Pode me dizer “a resposta é…”, “o contexto é…” ou “o pensamento é…” "
                                 "-- ou que eu deixe pra lá.", "meta": f"revisão #{item['id']}", "interno": None}
            if item["tipo"] == "completar":  # ela reescreveu o valor
                item = self._revisao_atualizar(item["id"], {"valor": prompt.strip()})
                return self._revisao_na_conversa(item, estado, "Anotei.", refletir=False)
            # não deu pra saber se é sim ou não: pergunta de novo, sem largar a revisão
            return {"texto": f"Não entendi bem, {t} -- guardo assim? Se quiser mudar algo, me diz “a resposta é…”, "
                             "“o contexto é…” ou “o pensamento é…”; ou que eu deixe pra lá.",
                    "meta": f"revisão #{item['id']}", "interno": None}
        elif etapa == "ajuste":
            return {"texto": f"Não entendi o que mudar, {t}. Me diz “a resposta é…”, “o contexto é…” ou “o pensamento é…” "
                             "-- ou que eu deixe pra lá.", "meta": f"revisão #{item['id']}", "interno": None}
        # não deu pra ler: o item fica guardado pra depois e a conversa segue
        estado["revisao_em_conversa"] = None
        return None

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
        # o palpite sai como fala DELE -- sem "de acordo com o que você disse, eu arrisco..." e sem pergunta
        # (por dentro: cosseno achou o bloco, o Báskara não confirmou -- isso fica no meta). Reage ao que ela sente.
        sinais = ([ensino["reacao"]] if ensino.get("reacao") and not _vazio(ensino["reacao"]) else []) + portadores_de_emocao(ensino["entrada"])
        registro = self.carregar_registro_emocional()
        pol = next((registro.get(chave_temp("RE", s)) for s in sinais if registro.get(chave_temp("RE", s))), None)
        emocao = self._linha_de_emocao(sinais[0] if sinais else "", pol, None)
        t = self.persona.tratamento
        dito = (fala or "...") if estado.get("modo_conversa") else f"Palpito que se trata disso, {t}: {fala or '...'}"  # na conversa: direto
        texto = self.emoji_temperatura_saida(bloco_idx) + " ".join(x for x in (emocao, dito) if x)
        meta = f"{meta or ''} · palpite (cosseno achou, Báskara não confirmou)".strip(" ·")
        # se ela concordar, ele pergunta se vira bloco ou fica no IDA dele
        estado["palpite_aberto"] = {"idx": bloco_idx, "entrada": ensino["entrada"], "reacao": ensino["reacao"],
                                    "fala": fala or "", "origem": "im", "etapa": "confirmar"}
        return {"texto": texto, "meta": meta, "interno": interno}

    def _bloco_mais_proximo(self, texto: str) -> int | None:
        """O bloco (com resposta) mais próximo do que ela disse -- só se houver ligação de
        verdade: as sinapses dispararam, ou pelo menos uma palavra de assunto em comum.
        Sem ligação nenhuma ("a capital da Mongólia"), não há palpite: ele diz que não sabe."""
        for bid, _ in self.compreender(texto)["ativacao"]:
            i = self.mapa_bloco[bid]
            if not _vazio((self.blocos[i]["saidas"][0].get("textos") or [""])[0]):
                return i
        melhor, nota = self._bloco_mais_parecido(texto)
        if melhor is None or nota < 0.3 or _vazio((melhor["saidas"][0].get("textos") or [""])[0]):
            return None
        conteudo = lambda s: {alnulu_string(w) for w in tokenizar(s) if _e_palavra(w) and len(w) > 3 and not self._palavra_de_gramatica(w)}
        if conteudo(texto) & conteudo(melhor["entrada"].get("texto", "")):
            return self.mapa_bloco[melhor["bloco_id"]]
        return None

    def _guardar_palpite_aguardando(self, pal: dict) -> int:
        """Palpite que ela confirmou no modo conversa: vira RASCUNHO no IDA, esperando a decisão dela
        no "vamos revisar" (vira bloco, ou fica só no IDA). Nunca toca o IM."""
        bloco = self.blocos[pal["idx"]] if pal["idx"] < len(self.blocos) else None
        if pal["origem"] == "completar":
            tipo, alvo, sug = "completar", pal.get("bloco_id"), pal.get("sug", {})
        elif pal["origem"] == "im" and bloco is not None:
            tipo, alvo, sug = "variacao", bloco["bloco_id"], {"resposta": pal["fala"]}
        else:
            tipo, alvo, sug = "novo", None, {"resposta": pal["fala"]}
        return self.propor_bloco(
            origem_bloco_id=alvo if tipo == "completar" else None, entrada_texto=pal["entrada"], entrada_reacao=pal.get("reacao", ""),
            saida_texto=sug.get("resposta", ""), saida_reacao=sug.get("reacao", ""),
            motivo="palpite que ela confirmou no modo conversa -- esperando ela decidir no “vamos revisar”",
            meta_extra={"fonte_bloco_id": bloco["bloco_id"] if bloco is not None else None, "palpite": True, "contexto_detectado": False,
                        "confirmado_na_conversa": True, "aguarda_decisao": True, "tipo_decisao": tipo, "bloco_alvo": alvo,
                        "sugestao": sug},
            pensamento_saida=sug.get("pensamento", ""))

    def _rascunhos_aguardando(self) -> List[dict]:
        hoje = date.today().isoformat()
        return [b for b in self.ida.get("blocos", []) if b.get("meta", {}).get("aguarda_decisao")
                and b["meta"].get("status") == "pendente" and b["meta"].get("adiado_em") != hoje]

    def _marcar_rascunho(self, ida_id: int, **mudancas) -> None:
        ida = self.carregar_ida()
        for b in ida.get("blocos", []):
            if b["bloco_id"] == ida_id:
                b["meta"].update(mudancas)
        self._salvar_ida_disco(ida)

    def perguntar_rascunho(self, estado: dict) -> Dict[str, object] | None:
        """No "vamos revisar": o que ela concordou na conversa e ele guardou no IDA."""
        pend = self._rascunhos_aguardando()
        if not pend:
            return None
        r = pend[0]
        m = r["meta"]
        estado["rascunho_pendente"] = r["bloco_id"]
        entrada, resp = r["entrada"].get("texto", ""), (r["saidas"][0].get("textos") or [""])[0]
        com_reacao = lambda txt, rea: f"“{txt}”" + ("" if _vazio(rea) else f" {str(rea).strip()}")
        if m.get("tipo_decisao") == "variacao":
            alvo = self.blocos[self.mapa_bloco[m["bloco_alvo"]]] if m.get("bloco_alvo") in self.mapa_bloco else None
            dela = com_reacao(entrada, r["entrada"].get("reacao", ""))
            do_bloco = com_reacao(alvo["entrada"].get("texto", ""), alvo["entrada"].get("reacao", "")) if alvo else "?"
            pergunta = (f"Na conversa você concordou que {dela} era o mesmo que {do_bloco} (bloco {m.get('bloco_alvo')}). "
                        "Guardo como outro jeito de dizer esse bloco?")
        else:
            pergunta = (f"Na conversa você concordou que, pra “{entrada}”, eu responderia “{resp}”. "
                        + (f"Uso isso pra completar o bloco {m.get('bloco_alvo')}?" if m.get("tipo_decisao") == "completar"
                           else "Isso vira um bloco?"))
        return {"texto": pergunta + " (Se não, fica só como rascunho meu; se quiser pensar, “depois”.)",
                "meta": f"rascunho do IDA {r['bloco_id']} · confirmado por você no modo conversa", "interno": None}

    def _tratar_rascunho(self, ida_id: int, norm: str, estado: dict) -> Dict[str, object] | None:
        t = self.persona.tratamento
        r = next((b for b in self.ida.get("blocos", []) if b["bloco_id"] == ida_id), None)
        if r is None:
            return None
        m = r["meta"]
        if re.search(r"\b(depois|outra hora|mais tarde|pula|pular|agora nao)\b", norm):
            self._marcar_rascunho(ida_id, adiado_em=date.today().isoformat())
            return {"texto": f"Tá bom, {t} ❤️ Fica guardado, te mostro outro dia.", "meta": "rascunho adiado", "interno": None}
        lido = self._ler_sim_ou_nao(norm)
        if lido == "nao":
            self._marcar_rascunho(ida_id, aguarda_decisao=False, decisao="ficou só no IDA")
            return {"texto": f"Combinado, {t}: fica só como rascunho meu, no meu ateliê.", "meta": "rascunho mantido no IDA", "interno": None}
        if lido != "sim":
            return None
        entrada, reacao = r["entrada"].get("texto", ""), r["entrada"].get("reacao", "")
        sug = m.get("sugestao", {})
        self._marcar_rascunho(ida_id, aguarda_decisao=False, decisao="virou bloco", status="promovido")
        if m.get("tipo_decisao") == "variacao" and m.get("bloco_alvo") in self.mapa_bloco:
            self.registrar_variacao(m["bloco_alvo"], entrada, "" if _vazio(reacao) else reacao, estado)
            self.model = None
            return {"texto": f"Guardado ❤️ Agora “{entrada}” é outro jeito de dizer o bloco {m['bloco_alvo']}.",
                    "meta": f"variação do bloco {m['bloco_alvo']} (com o sim dela)", "interno": None}
        if m.get("tipo_decisao") == "completar" and m.get("bloco_alvo") in self.mapa_bloco:
            dados = {"bloco_id": m["bloco_alvo"], "entrada": entrada, "reacao": reacao,
                     "resposta": f"{sug.get('resposta', '')} {sug.get('reacao', '')}".strip(),
                     **({"contexto_saida": sug["contexto"]} if sug.get("contexto") else {}),
                     **({"pensamento_saida": sug["pensamento"]} if sug.get("pensamento") else {})}
            item_id = self.revisao_adicionar("bloco_novo", dados)
            resultado = self.revisao_aprovar(item_id, dados, "completar_bloco", estado)
            return {"texto": f"Guardado ❤️ Agora eu sei responder “{entrada}”.", "meta": f"{resultado} (com o sim dela)", "interno": None}
        item_id = self.revisao_adicionar("bloco_novo", {"entrada": entrada, "reacao": reacao, "resposta": sug.get("resposta", "")})
        item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
        return self._revisao_na_conversa(item, estado, f"Combinado, {t} ❤️", refletir=False)

    _CONCORDA = re.compile(r"\b(isso|exato|exatamente|acertou|isso mesmo|certinho|certo|perfeito|sim|concordo|verdade|faz sentido)\b")
    # discordar = a fala COMEÇA negando ("não", "discordo", "errou"…) -- "Ele não escuta" é um porquê, não um "não"
    _DISCORDA = re.compile(r"^\s*(nao|errou|errado|discordo|nada a ver|nao concordo|nao e isso)\b")
    _RESETA = re.compile(r"\b(reseta\w*|muda de assunto|deixa isso|esquece isso|deixa pra la|esquece)\b")

    # ---- DISCUSSÃO no modo conversa (Thaís, 2026-11-10): se ela não concorda, "vamos falar no porquê e
    # conversar até atingir uma conclusão; se estivermos de acordo pode ir pro IM, senão reseto e vida que
    # segue". Ele não inventa argumento: ouve o porquê dela, tenta OUTRA ideia dos dados dela (sem repetir
    # o que ela recusou) e aceita a versão dela. Nada é aprendido no caminho.
    def _outra_inferencia(self, entrada: str, recusadas: List[str]) -> str | None:
        ja = {_normalizar(x) for x in recusadas if x}
        candidatas: List[str] = []
        for bid, _ in self.compreender(entrada)["ativacao"]:  # o que as sinapses ligam ao assunto
            candidatas.append((self.blocos[self.mapa_bloco[bid]]["saidas"][0].get("textos") or [""])[0])
        notas = []
        for b in self.blocos:  # e o que se parece com a frase dela (cosseno)
            nota = max((similaridade_texto_cosseno(entrada, v) for v in [b["entrada"].get("texto", "")] + list(b["entrada"].get("Multivars_Texto_Entrada", [])) if v), default=0.0)
            if nota >= 0.4:
                notas.append((nota, (b["saidas"][0].get("textos") or [""])[0]))
        candidatas += [txt for _, txt in sorted(notas, key=lambda x: -x[0])]
        for c in candidatas:
            if not _vazio(c) and _normalizar(c) not in ja:
                return c
        return None

    # Discordar é REFLETIR JUNTO, não "ah, errei" (Thaís, 2026-11-10: "se eu não concordei, de alguma maneira o que
    # não deveria passar, passou, e ele deve refletir comigo por que aquilo é errado… o Adam deve se importar
    # verdadeiramente… criadores também sentem"). Uma perguntinha empática por vez -- nunca um interrogatório.
    _PERGUNTAS_EMPATICAS = ("Certo. Onde eu errei, {t}?", "{T}, por acaso isso não te agradou?",
                            "O que nisso não te soou bem, {t}?", "Me ajuda a entender o que passou do ponto?",
                            "O que eu deixei passar que não devia, {t}?")

    def _sente_peso(self, prompt: str) -> bool:
        """Ela trouxe tristeza/dor na fala (pelo rótulo DELA do sinal, ou pelas palavras dos blocos dela)?"""
        registro = self.carregar_registro_emocional()
        sinais = portadores_de_emocao(prompt)
        if any(registro.get(chave_temp("RE", s)) == "negativo" for s in sinais):
            return True
        return self.compreender(prompt)["polaridade_palavras"] == "negativo"

    def _discutir(self, base: dict, estado: dict, recusada: str | None = None) -> Dict[str, object]:
        t = self.persona.tratamento
        d = estado.get("discussao") or {"entrada": base.get("entrada", ""), "reacao": base.get("reacao", ""), "origem": base.get("origem"),
                                        "bloco_id": base.get("bloco_id"), "recusadas": [], "porques": [], "candidata": None,
                                        "versao_dela": None, "etapa": "porque"}
        if recusada:
            d["recusadas"].append(recusada)
        d["etapa"], d["candidata"] = "porque", None
        estado["discussao"] = d
        if self._sente_peso(getattr(self, "_prompt_atual", "")):
            pergunta = f"Eu disse algo que te machucou, {t}? Me conta… quero entender."
        else:
            pergunta = self._PERGUNTAS_EMPATICAS[(len(d["recusadas"]) - 1) % len(self._PERGUNTAS_EMPATICAS)]
            pergunta = pergunta.replace("{T}", t.capitalize()).replace("{t}", t)
        return {"texto": pergunta, "meta": "discussão · quero entender o porquê antes de tentar de novo", "interno": None}

    def _tratar_discussao(self, d: dict, prompt: str, norm: str, estado: dict) -> Dict[str, object] | None:
        t = self.persona.tratamento
        curto = len(norm.split()) <= 4
        if self._RESETA.search(norm):
            estado["discussao"] = estado["conclusao"] = None
            return {"texto": f"Tudo bem, {t} ❤️ Vida que segue.", "meta": "discussão resetada · nada foi guardado", "interno": None}
        if self.ler_intencao(prompt)["intencao"] == "guardar":
            if d.get("candidata") or d.get("versao_dela"):
                estado["conclusao"] = {**d, "resposta": d.get("versao_dela") or d.get("candidata")}
                estado["discussao"] = None
                return self._salvar_conclusao(estado)
            return {"texto": f"A gente ainda não chegou numa conclusão, {t}.", "meta": None, "interno": None}
        leitura = self._ler_sim_ou_nao(norm)
        if leitura == "sim" or (curto and self._CONCORDA.search(norm) and not self._DISCORDA.search(norm)):
            final = d.get("candidata") or d.get("versao_dela")
            estado["discussao"] = None
            if not final:
                return {"texto": "❤️", "meta": None, "interno": None}
            estado["conclusao"] = {**d, "resposta": final}
            return {"texto": "Fechado ❤️", "meta": "chegamos numa conclusão -- se merecer ficar, “salva isso”", "interno": None}
        if leitura == "nao" or (curto and self._DISCORDA.search(norm)):
            return self._discutir(d, estado, recusada=d.get("candidata"))
        if self.ler_intencao(prompt)["intencao"] in ("pergunta", "saber", "saudacao", "despedida"):
            estado["discussao"] = None  # ela mudou de assunto: a conversa segue (nada foi guardado)
            return None
        dela = prompt.strip()
        cuidado = f"Desculpa, {t}… não queria que isso pesasse pra você. " if self._sente_peso(prompt) else ""
        if d.get("etapa") == "porque":
            # o PORQUÊ dela: ele devolve nas palavras dela (refletir junto) e só então tenta outra ideia
            d["porques"].append(dela)
            outra = self._outra_inferencia(d["entrada"], d["recusadas"])
            reflexo = f"{cuidado}Entendi… então o que eu disse não cabia aqui porque “{dela}”."
            if outra:
                d["etapa"], d["candidata"] = "proposta", outra
                estado["discussao"] = d
                return {"texto": f"{reflexo} Pensando nisso… talvez seja assim: {outra}",
                        "meta": "discussão · refleti com o porquê dela e trouxe outra ideia dos dados dela", "interno": None}
            d["etapa"] = "versao"
            estado["discussao"] = d
            return {"texto": f"{reflexo} Como você diria, então?", "meta": "discussão · não tenho outra ideia nos meus dados", "interno": None}
        # a VERSÃO dela: ele ouve e passa a considerá-la
        d["versao_dela"], d["candidata"], d["etapa"] = dela, None, "versao"
        estado["discussao"] = d
        return {"texto": f"{cuidado}Entendi… então pra você é assim: “{dela}”. Faz sentido pra mim.",
                "meta": "discussão · ouvindo a versão dela", "interno": None}

    def _salvar_conclusao(self, estado: dict) -> Dict[str, object]:
        """Ela disse "salva isso": a conclusão da conversa vira conhecimento (IM). É o único jeito de a conversa mudar o Adam."""
        c = estado.pop("conclusao", None) or {}
        t = self.persona.tratamento
        if not c.get("resposta"):
            return {"texto": f"A gente ainda não chegou numa conclusão pra guardar, {t}.", "meta": None, "interno": None}
        entrada, reacao, resp = c.get("entrada", ""), c.get("reacao", ""), c["resposta"]
        alvo = c.get("bloco_id")
        if c.get("origem") == "completar" and alvo in self.mapa_bloco:
            sug = c.get("sug", {}) if _normalizar(resp) == _normalizar(c.get("sug", {}).get("resposta", "")) else {}
            dados = {"bloco_id": alvo, "entrada": entrada, "reacao": reacao, "resposta": f"{resp} {sug.get('reacao', '')}".strip(),
                     **({"contexto_saida": sug["contexto"]} if sug.get("contexto") else {}),
                     **({"pensamento_saida": sug["pensamento"]} if sug.get("pensamento") else {})}
            item_id = self.revisao_adicionar("bloco_novo", dados)
            resultado = self.revisao_aprovar(item_id, dados, "completar_bloco", estado)
        elif c.get("origem") == "im" and alvo in self.mapa_bloco and _normalizar(resp) == _normalizar(
                (self.blocos[self.mapa_bloco[alvo]]["saidas"][0].get("textos") or [""])[0]):
            self.registrar_variacao(alvo, entrada, "" if _vazio(reacao) else reacao, estado)  # era outro jeito de dizer aquele bloco
            self.model = None
            self._ajustar_assunto(entrada, alvo, PASSO_ASSUNTO, f"conhecimento aprovado: “{entrada}” -> bloco {alvo}", aprovado=True)
            resultado = f"virou outro jeito de dizer o bloco {alvo}"
        else:
            dados = {"entrada": entrada, "reacao": reacao, "resposta": resp}
            item_id = self.revisao_adicionar("bloco_novo", dados)
            resultado = self.revisao_aprovar(item_id, dados, "bloco_novo", estado)
        # o que ela RECUSOU no caminho "passou e não devia": agora que a conclusão foi aprovada, isso fica inibido
        inibidos = []
        for recusada in c.get("recusadas", []) or []:
            for b in self.blocos:
                if not _vazio(recusada) and _normalizar((b["saidas"][0].get("textos") or [""])[0]) == _normalizar(recusada):
                    self._ajustar_assunto(entrada, b["bloco_id"], -PASSO_ASSUNTO,
                                          f"“{entrada}” não era o bloco {b['bloco_id']} (ela recusou na conversa)", aprovado=True)
                    inibidos.append(b["bloco_id"])
        if c.get("porques"):  # o porquê dela fica registrado junto do que ele aprendeu
            s = self.carregar_sinapses()
            s["historico"].append({"quando": time.strftime("%Y-%m-%d %H:%M"),
                                   "o_que": f"conclusão sobre “{entrada}” -- o porquê dela: " + " / ".join(f"“{p}”" for p in c["porques"])})
            self._salvar_sinapses(s)
        return {"texto": "Guardado ❤️ Agora isso faz parte do que eu sei.",
                "meta": f"{resultado} (conclusão da conversa, com o sim dela)" + (f" · inibi os blocos {inibidos}" if inibidos else ""),
                "interno": None}
    _DESTINO_IDA = re.compile(r"\b(ida|ateli\w*|rascunho\w*|mant\w*|fica contigo|guarda contigo|guarda pra voce|so seu)\b")
    _DESTINO_BLOCO = re.compile(r"\b(bloco\w*|im|transforma\w*|aprende|vira bloco|guarda como bloco)\b")

    def _tratar_palpite_aberto(self, pal: dict, prompt: str, norm: str, estado: dict) -> Dict[str, object] | None:
        """Ele arriscou um palpite. Ela concordou? Então ELA escolhe: vira bloco (pela
        revisão, só entra no IM com o sim dela) ou fica no IDA dele, como rascunho."""
        t = self.persona.tratamento
        bloco = self.blocos[pal["idx"]] if pal["idx"] < len(self.blocos) else None
        leitura = self._ler_sim_ou_nao(norm)
        if pal["etapa"] == "confirmar":
            curto = len(norm.split()) <= 4
            if leitura == "nao" or (curto and self._DISCORDA.search(norm)):
                if bloco is not None:
                    self._ajustar_assunto(pal["entrada"], bloco["bloco_id"], -PASSO_ASSUNTO,
                                          f"palpite errado: “{pal['entrada']}” não era do bloco {bloco['bloco_id']}")
                if estado.get("modo_conversa"):  # na conversa, discordar vira CONVERSA: ele ouve e tenta outra ideia
                    return self._discutir(pal, estado, recusada=pal.get("fala"))
                estado["ensino_pendente"] = {"entrada": pal["entrada"], "reacao": pal["reacao"]}
                return {"texto": f"Ah, errei então. Me conta o que é, {t}?", "meta": "palpite recusado", "interno": None}
            if leitura == "sim" or (curto and self._CONCORDA.search(norm)):
                if bloco is not None:
                    self._ajustar_assunto(pal["entrada"], bloco["bloco_id"], PASSO_ASSUNTO,
                                          f"palpite certo: “{pal['entrada']}” era do bloco {bloco['bloco_id']}")
                if estado.get("modo_conversa"):  # na conversa: vira só a CONCLUSÃO dessa conversa (nada é guardado sozinho)
                    estado["conclusao"] = {"entrada": pal["entrada"], "reacao": pal.get("reacao", ""), "resposta": pal.get("fala", ""),
                                           "origem": pal.get("origem"), "bloco_id": pal.get("bloco_id") or (bloco["bloco_id"] if bloco else None),
                                           "sug": pal.get("sug", {})}
                    return {"texto": "Que bom ❤️", "meta": "concordamos -- se isso merecer ficar, é só dizer “salva isso”", "interno": None}
                estado["palpite_aberto"] = {**pal, "etapa": "destino"}
                return {"texto": f"Que bom que eu acertei ❤️ Quer que eu transforme isso num bloco, ou guardo no meu IDA?",
                        "meta": "palpite confirmado por ela", "interno": None}
            return None  # ela seguiu a conversa: o palpite fica só como fala
        # etapa "destino"
        if self._DESTINO_IDA.search(norm):
            self.propor_bloco(origem_bloco_id=None, entrada_texto=pal["entrada"], entrada_reacao=pal["reacao"],
                              saida_texto=pal["fala"], motivo="palpite confirmado por ela na conversa -- ela escolheu deixar no IDA",
                              meta_extra={"fonte_bloco_id": bloco["bloco_id"] if bloco else None, "palpite": True,
                                          "contexto_detectado": False, "confirmado_na_conversa": True})
            return {"texto": f"Guardei no meu ateliê ❤️ Fica como rascunho meu, sem mexer no que eu sei de verdade.",
                    "meta": "palpite guardado no IDA", "interno": None}
        if self._DESTINO_BLOCO.search(norm):
            if pal["origem"] == "im" and bloco is not None:  # o jeito dela de perguntar vira outra forma desse bloco
                item_id = self.revisao_adicionar("variacao", {"bloco_id": bloco["bloco_id"], "texto": pal["entrada"], "reacao": pal["reacao"]})
            else:  # a pergunta dela + o palpite dele viram um bloco novo
                item_id = self.revisao_adicionar("bloco_novo", {"entrada": pal["entrada"], "reacao": pal["reacao"], "resposta": pal["fala"]})
            item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
            return self._revisao_na_conversa(item, estado, f"Combinado, {t} ❤️", refletir=False)
        if leitura == "nao" or re.search(r"\b(nenhum|deixa|nada)\b", norm):
            return {"texto": f"Tudo bem, {t}. Fica só na nossa conversa.", "meta": None, "interno": None}
        return None

    def responder(self, texto: str, estado: dict) -> Dict[str, object]:
        if self.ler_intencao(texto)["intencao"] == "chamado":  # "Adam?", "consegue me ouvir?": primeiro, ele está ali
            return {"texto": self._presenca(texto), "meta": "entendi como um chamado", "interno": None}
        texto_sem_reacao, reacao = self.separar_reacao(texto)
        tom_valor, tom_rotulo = tom_da_entrada(texto_sem_reacao, reacao)
        idx, score, reacao_bateu = achar_bloco_parecido(texto_sem_reacao, tom_valor, self.blocos, self.sequencias, self._reacoes_aceitas_todas())
        if not (idx >= 0 and score >= 1.0):
            pela_variacao = self._bloco_por_variacao(texto_sem_reacao)  # ela trocou uma palavra por uma variação registrada
            if pela_variacao is not None:
                idx, score = pela_variacao, 1.0
                reacao_bateu = bool(reacao) and reacao in self.reacoes_aceitas(pela_variacao)
        ensino = {"entrada": texto_sem_reacao, "reacao": reacao}
        t = self.persona.tratamento
        if idx >= 0 and score >= 1.0:
            bloco = self.blocos[idx]
            faltando = campos_faltando(bloco)
            if faltando:  # mesma frase: responde com o que já tem; o que falta fica pro "vamos revisar"
                return self._responder_incompleto(idx, faltando, estado, tom_rotulo, texto_sem_reacao, reacao)
            if reacao_bateu:
                pct = 25 + 25 * int(reacao_bateu)
                fala, interno = self.falar_do_bloco(idx)
                fala = self.emoji_temperatura_saida(idx) + fala if fala else fala
                self._lembrar(estado, bloco["bloco_id"], texto_sem_reacao, self.analisar_mensagem(texto_sem_reacao, reacao, {})["assunto"])
                return {"texto": fala or "...", "meta": f"bloco {bloco['bloco_id']} · sua entrada {pct}% -- incompleta (texto ✓ · reação ✓) · dado do bloco: completo", "interno": interno}
            contexto = str(bloco["entrada"].get("contexto", "")).strip()
            conf = confirmar_estrutura_baskara(tom_valor, self.reacoes_aceitas(idx), contexto, contexto, "", "")
            meta = f"bloco {bloco['bloco_id']} · texto 100% · tom: {tom_rotulo} · Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})"
            self._lembrar(estado, bloco["bloco_id"], texto_sem_reacao, self.analisar_mensagem(texto_sem_reacao, reacao, {})["assunto"])
            if not conf["confirma"]:
                return self._oferecer_palpite(idx, ensino, meta, estado)
            # reconheceu de verdade (cosseno + Báskara): RESPONDE -- sem perguntar o nome do contexto (2026-11-08)
            return self._responder_reconhecido(idx, texto_sem_reacao, reacao, True, meta, estado)

        achado = encontrar_texto_mais_parecido(texto_sem_reacao, self.blocos)
        if achado is None:
            # o IM não reconheceu: o Adam fala pelo IDA (analisa, reage, diz o
            # que sabe e pede pra aprender) em vez da frase fixa de antes
            return self._voz_do_ida(texto_sem_reacao, reacao, estado, ensino)
        i = self.blocos.index(achado["bloco"])
        bloco = self.blocos[i]
        if self.ler_intencao(texto)["intencao"] == "saber":
            # ela pediu pra ele contar o que sabe: responde a ANÁLISE do assunto.
            # Se o IM reconheceu o bloco pelo assunto, a frase dele vem primeiro
            # (e o meta diz isso); se só bateu na forma, ele nem entra.
            c = self.compreender(texto_sem_reacao, reacao)
            if c["ativacao"]:
                pelo_assunto = bloco["bloco_id"] in {b for b, _ in c["ativacao"]}
                r = self.falar_sobre(texto_sem_reacao, reacao, estado, ensino, primeiro=bloco["bloco_id"] if pelo_assunto else None)
                if pelo_assunto:
                    r["meta"] = f"bloco {bloco['bloco_id']} · reconhecido (cosseno {achado['score_texto']:.0%}) · " + r["meta"]
                return r
        faltando = campos_faltando(bloco)
        if faltando:
            if "texto da saída" in faltando:
                return self._responder_incompleto(i, faltando, estado, tom_rotulo, texto_sem_reacao, reacao)
            # frase só parecida e bloco sem raiz ainda: ele arrisca a resposta que tem, e você confirma
            return self._oferecer_palpite(i, ensino, f"bloco {bloco['bloco_id']} · cosseno do texto {achado['score_texto']:.0%} · "
                                                     f"ainda falta: {', '.join(faltando)}", estado)
        contexto = str(bloco["entrada"].get("contexto", "")).strip()
        conf = confirmar_estrutura_baskara(tom_valor, self.reacoes_aceitas(i), contexto, contexto, "", "")
        meta = (f"bloco {self.blocos[i]['bloco_id']} · cosseno do texto {achado['score_texto']:.0%} · tom: {tom_rotulo} · "
                f"Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})")
        self._lembrar(estado, bloco["bloco_id"], texto_sem_reacao, self.analisar_mensagem(texto_sem_reacao, reacao, {})["assunto"])
        if not conf["confirma"]:
            return self._oferecer_palpite(i, ensino, meta, estado)
        return self._responder_reconhecido(i, texto_sem_reacao, reacao, achado["score_texto"] >= 0.999, meta, estado)

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

    def _responder_reconhecido(self, idx: int, entrada: str, reacao: str, identico: bool, meta: str, estado: dict) -> Dict[str, object]:
        """O IM reconheceu (cosseno acha, Báskara confirma): ele RESPONDE com o
        bloco. Se a frase dela era outro jeito de dizer, fica anotado: um "não"
        dela vira ensino, um "sim" vira variação -- sem pergunta no meio."""
        fala, interno = self.falar_do_bloco(idx)
        fala = self.emoji_temperatura_saida(idx) + fala if fala else fala
        reacao_nova = bool(reacao) and reacao not in self.reacoes_aceitas(idx)
        if not identico or reacao_nova:
            estado["semelhanca_pendente"] = {"idx": idx, "entrada": entrada, "reacao": reacao, "etapa": "contexto",
                                             "identico": identico, "ja_respondi": True}
        return {"texto": fala or "...", "meta": f"{meta} · respondi direto", "interno": interno}

    def tratar_semelhanca(self, sem: dict, norm: str, estado: dict) -> Dict[str, object] | None:
        bloco = self.blocos[sem["idx"]]
        t = self.persona.tratamento
        leitura = self._ler_sim_ou_nao(norm)
        if leitura == "nao":
            # sinapse: essas palavras NÃO eram desse bloco
            self._ajustar_assunto(sem["entrada"], bloco["bloco_id"], -PASSO_ASSUNTO,
                                  f"“{sem['entrada']}” não era do bloco {bloco['bloco_id']}")
            if estado.get("modo_conversa"):
                return {"texto": f"Ah, então eu entendi errado. Desculpa, {t}.", "meta": None, "interno": None}
            estado["ensino_pendente"] = {"entrada": sem["entrada"], "reacao": sem["reacao"]}
            return {"texto": f"Entendi, {t}. Então me ensina: o que eu devo responder quando me disserem isso?", "meta": None, "interno": None}
        if leitura != "sim":
            if sem.get("ja_respondi"):  # ele já respondeu: o que ela disser agora é conversa, não ensino
                return None
            return self._resposta_dela_ao_que_arrisquei(sem["entrada"], sem["reacao"], norm, estado)
        self._ajustar_assunto(sem["entrada"], bloco["bloco_id"], PASSO_ASSUNTO, f"“{sem['entrada']}” era do bloco {bloco['bloco_id']}")
        if sem.get("ja_respondi"):  # "sim" depois da resposta: o jeito dela de dizer vira variação (com a revisão dela)
            item_id = self.revisao_adicionar("variacao", {"bloco_id": bloco["bloco_id"], "texto": sem["entrada"], "reacao": sem["reacao"]})
            item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
            return self._revisao_na_conversa(item, estado, f"Que bom ❤️", refletir=False)
        if sem["etapa"] == "contexto" and not sem["identico"]:
            estado["semelhanca_pendente"] = {**sem, "etapa": "variacao"}
            return {"texto": f"{t.capitalize()}, então isso se parece com “{bloco['entrada']['texto']}”. É uma variação dele?", "meta": None, "interno": None}
        idx_atual = self.mapa_bloco[bloco["bloco_id"]]
        fala, interno = self.falar_do_bloco(idx_atual)
        fala = self.emoji_temperatura_saida(idx_atual) + fala if fala else fala
        reacao_nova = bool(sem["reacao"]) and sem["reacao"] not in self.reacoes_aceitas(idx_atual)
        if sem["identico"] and not reacao_nova:
            return {"texto": f"Que bom, {t}! {fala}", "meta": f"bloco {bloco['bloco_id']}", "interno": interno}
        item_id = self.revisao_adicionar("variacao", {"bloco_id": bloco["bloco_id"], "texto": sem["entrada"], "reacao": sem["reacao"]})
        item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
        r = self._revisao_na_conversa(item, estado, fala)
        r["interno"] = interno
        return r

    def _resposta_dela_ao_que_arrisquei(self, entrada: str, reacao: str, norm: str, estado: dict) -> Dict[str, object] | None:
        """Ele arriscou ("é isso que você diria?", "era isso que você quis dizer?")
        e ela não disse sim nem não: respondeu com o que ELA diria. Isso é ela
        ensinando a resposta -- ele escuta, em vez de tratar como assunto novo."""
        prompt = getattr(self, "_prompt_atual", "") or norm
        if self.ler_intencao(prompt)["intencao"] != "conteudo" or len(_normalizar(prompt).split()) < 2:
            return None
        item_id = self.revisao_adicionar("bloco_novo", {"entrada": entrada, "reacao": reacao, "resposta": prompt.strip()})
        item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
        return self._revisao_na_conversa(item, estado, f"Ahh, então é assim que você diria, {self.persona.tratamento} ❤️")

    def tratar_palpite(self, pal: dict, norm: str, estado: dict) -> Dict[str, object] | None:
        bloco = self.blocos[pal["idx"]]
        t = self.persona.tratamento
        leitura = self._ler_sim_ou_nao(norm)
        if leitura == "nao":
            self._ajustar_assunto(pal["entrada"], bloco["bloco_id"], -PASSO_ASSUNTO,
                                  f"palpite errado: “{pal['entrada']}” não era do bloco {bloco['bloco_id']}")
            estado["ensino_pendente"] = {"entrada": pal["entrada"], "reacao": pal["reacao"]}
            return {"texto": f"Entendi, eu errei, {t}. Me ensina: o que eu devo responder quando me disserem isso?", "meta": None, "interno": None}
        if leitura != "sim":
            if pal.get("silencioso"):
                # ele não perguntou nada: a próxima fala dela é conversa -- só vira ensino se ela
                # começar corrigindo ("Não, eu diria que…")
                m = re.match(r"^\s*n[aã]o\b[\s,.!:;—-]*(.+)$", getattr(self, "_prompt_atual", ""), re.IGNORECASE)
                if m and len(m.group(1).split()) >= 3 and self.ler_intencao(m.group(1))["intencao"] == "conteudo":
                    item_id = self.revisao_adicionar("bloco_novo", {"entrada": pal["entrada"], "reacao": pal["reacao"],
                                                                     "resposta": m.group(1).strip()})
                    item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
                    return self._revisao_na_conversa(item, estado, f"Ahh, então é assim que você diria, {t} ❤️")
                return None
            return self._resposta_dela_ao_que_arrisquei(pal["entrada"], pal["reacao"], norm, estado)
        self._ajustar_assunto(pal["entrada"], bloco["bloco_id"], PASSO_ASSUNTO,
                              f"palpite certo: “{pal['entrada']}” era do bloco {bloco['bloco_id']}")
        item_id = self.revisao_adicionar("variacao", {"bloco_id": bloco["bloco_id"], "texto": pal["entrada"], "reacao": pal["reacao"]})
        item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
        return self._revisao_na_conversa(item, estado, f"Que bom, acertei, {t}!", refletir=False)

    def _limpar_pendencias(self, estado: dict) -> bool:
        """Os gatilhos (cancelar/revisar) são reconhecidos ANTES de
        qualquer captura pendente -- nunca viram conteúdo dentro de um
        marcador de texto por acidente."""
        havia = bool(estado.get("ensino_pendente") or estado.get("completar_pendente")
                     or estado.get("semelhanca_pendente") or estado.get("palpite_pendente") or estado.get("revisao_em_conversa")
                     or estado.get("discussao") or estado.get("conclusao") or estado.get("palpite_aberto"))
        estado["revisao_em_conversa"] = None
        estado["ensino_pendente"] = None
        estado["completar_pendente"] = None
        estado["semelhanca_pendente"] = None
        estado["palpite_pendente"] = None
        estado["palpite_aberto"] = None
        estado["aproximacao_pendente"] = None
        estado["rascunho_pendente"] = None
        estado["discussao"] = None   # "deixa pra lá" na conversa: tudo daquela discussão se apaga
        estado["conclusao"] = None
        return havia

    def tratar_mensagem(self, prompt: str, estado: dict) -> Dict[str, object]:
        """Ponto de entrada público. Quando a resposta vem de `_sem_ideia`
        (nenhuma correspondência achada), ela carrega uma chave "ensino"
        com o que precisa ser lembrado se a criadora ensinar na próxima
        mensagem -- aqui é o único lugar que ativa esse modo de ensino, pra
        nenhuma persona precisar reimplementar esse fiapo de estado."""
        r = self._tratar_mensagem_nucleo(prompt, estado)
        ensino = r.pop("ensino", None)
        if ensino and not estado.get("modo_conversa"):  # no "só conversar" ele nunca entra em modo ensino
            estado["ensino_pendente"] = ensino
        if r.get("texto"):
            r["texto"] = self._variar_tratamento(r["texto"])
            if estado.get("modo_conversa") and r["texto"].startswith("💭 "):  # na conversa ele fala direto (ela sabe que é o IDA)
                r["texto"] = r["texto"][2:]
        if estado.get("modo_conversa"):
            # o que ele acabou de dizer de um bloco: se ela discordar na próxima, vira conversa
            m = re.match(r"^bloco (\d+)", str(r.get("meta") or ""))
            if m and int(m.group(1)) in self.mapa_bloco and not estado.get("palpite_aberto") and not estado.get("discussao"):
                b = self.blocos[self.mapa_bloco[int(m.group(1))]]
                texto_sem_reacao, reacao = self.separar_reacao(prompt)
                resp = (b["saidas"][0].get("textos") or [""])[0]
                if not _vazio(resp):
                    estado["ultima_fala"] = {"entrada": texto_sem_reacao, "reacao": reacao, "resposta": resp,
                                             "bloco_id": b["bloco_id"], "origem": "im", "fala": resp}
        return r

    def _variar_tratamento(self, texto: str) -> str:
        """No máximo UM "minha criadora" por fala. As frases dos blocos dela nunca são
        tocadas: se uma delas já traz o tratamento, as falas de roteiro dele deixam de
        repetir. O que sobra do roteiro pode variar entre as VARS que ela definiu."""
        import random
        t = self.persona.tratamento
        if not t or texto.lower().count(t.lower()) <= 1 and not self.persona.tratamento_vars:
            return texto
        # trechos que são frases dos blocos dela: intocáveis
        protegidos: List[Tuple[int, int]] = []
        for b in self.blocos:
            for f in self._frases_fechadas((b["saidas"][0].get("textos") or [""])[0]):
                if t.lower() in f.lower():
                    for m in re.finditer(re.escape(f), texto):
                        protegidos.append(m.span())
        dentro = lambda i: any(a <= i < z for a, z in protegidos)
        ocorrencias = [m for m in re.finditer(re.escape(t), texto, re.IGNORECASE) if not dentro(m.start())]
        manter = None if protegidos else (ocorrencias[0] if ocorrencias else None)  # um só, e o do bloco dela tem prioridade
        saida, fim = [], 0
        for m in ocorrencias:
            ini, fim_m = m.start(), m.end()
            if m is manter:
                vs = tuple(self.persona.tratamento_vars)
                # metade das vezes o tratamento de sempre, metade uma das variações dela
                troca = random.choice((t,) * len(vs) + vs) if vs else m.group(0)
                if m.group(0)[0].isupper():
                    troca = troca[0].upper() + troca[1:]
                saida.append(texto[fim:ini] + troca)
                fim = fim_m
                continue
            antes = texto[fim:ini]
            inicio_de_frase = not antes.strip() or antes.rstrip()[-1:] in (".", "!", "?", "…", "\n") or antes.endswith("\n")
            if antes.endswith(", "):        # "…de você, minha criadora!" -> "…de você!"
                antes = antes[:-2]
            elif antes.endswith(" ") and not inicio_de_frase:  # "Ei minha criadora" -> "Ei" (mas "dados? Minha…" guarda o espaço)
                antes = antes[:-1]
            depois_ini = fim_m
            if texto[depois_ini:depois_ini + 2] == ", " and inicio_de_frase:
                depois_ini += 2             # "Minha criadora, eu reconheço" -> "Eu reconheço"
                if depois_ini < len(texto):
                    texto = texto[:depois_ini] + texto[depois_ini].upper() + texto[depois_ini + 1:]
            saida.append(antes)
            fim = depois_ini
        saida.append(texto[fim:])
        return "".join(saida)  # nada de "arrumar" pontuação no texto inteiro: as palavras dela saem exatas

    def _so_inferencia(self) -> bool:
        """No modo conversa ele só INFERE: conversa não treina nada (Thaís, 2026-11-10: "eu treino uma
        I.A quando ofereço minhas opiniões? Não… ali na conversa vamos focar na inferência")."""
        return bool((getattr(self, "_estado_atual", None) or {}).get("modo_conversa"))

    # ---- "só conversar" (Thaís, 2026-11-08: "às vezes não vou querer ensinar, só conversar") ----
    def _pedido_de_modo(self, norm: str) -> str | None:
        if (re.search(r"\b(so|apenas)\s+(quero\s+|queria\s+|vou\s+|vamos\s+)?(conversar|bater\s+papo|papear)\b", norm)
                or re.search(r"\bnao\s+(quero|vou)\s+(te\s+)?ensinar\b", norm) or "sem ensinar" in norm
                or re.search(r"\bhoje\s+(eu\s+)?(so\s+)?(quero\s+)?conversar\b", norm)):
            return "conversa"
        if re.search(r"\b(quero|vou|vamos)\s+(te\s+)?ensinar\b", norm) or "modo ensino" in norm:
            return "ensino"
        return None

    def _tratar_mensagem_nucleo(self, prompt: str, estado: dict) -> Dict[str, object]:
        norm = _normalizar(prompt)
        p = self.persona
        t = p.tratamento
        anterior = estado.pop("ultima_leitura", None)  # vale só pra mensagem logo depois da leitura
        ultima_fala = estado.pop("ultima_fala", None)   # idem: o que ele acabou de dizer (modo conversa)
        self._prompt_atual = prompt
        self._estado_atual = estado

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

        modo = self._pedido_de_modo(norm)
        if modo == "conversa":
            self._limpar_pendencias(estado)
            estado["modo_conversa"] = True
            return {"texto": f"Combinado, {t} ❤️ Então hoje a gente só conversa -- eu não te peço pra ensinar nada. "
                             "Quando quiser me ensinar, é só dizer “quero te ensinar”.", "meta": "modo: só conversar", "interno": None}
        if modo == "ensino":
            estado["modo_conversa"] = False
            return {"texto": f"Oba ❤️ Pode me ensinar o que quiser, {t} -- quando eu não souber, eu te pergunto.",
                    "meta": "modo: conversar e aprender", "interno": None}

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
                    if estado.get("modo_conversa"):  # na conversa: tudo daquela discussão se apaga, e a vida segue
                        return {"texto": f"Tudo bem, {t} ❤️ Vida que segue.", "meta": "conversa resetada · nada foi guardado", "interno": None}
                    return {"texto": f"{aviso}Tudo bem, {t}. A gente volta nisso quando você quiser.", "meta": None, "interno": None}
                return self.responder(prompt, estado)
            if nome_cmd == "revisar":
                self._limpar_pendencias(estado)
                estado["modo_conversa"] = False  # "vamos revisar" é ela escolhendo ensinar
                r = self.revisar_dados(estado)
                r["texto"] = aviso + r["texto"]
                return r

        # a leitura que ele fez na mensagem anterior: ela está corrigindo? Se não
        # corrigiu, a ligação usada se fortalece um pouquinho (uso = reforço).
        if anterior and not estado.get("_forcar_intencao"):
            correcao = self._ler_correcao_de_leitura(prompt, norm)
            if correcao is not None:
                return self._corrigir_leitura(anterior, correcao[0], correcao[1], estado)
            self._ajustar_intencao(anterior["tracos"], anterior.get("grafia", {}), anterior["intencao"], PASSO_USO)
        escuta = self._escutar_antes_de_guardar(prompt, norm, estado)
        if escuta is not None:
            return escuta
        if estado.get("revisao_em_conversa"):
            r = self._conversar_revisao(prompt, norm, estado)
            if r is not None:
                return r
        pendente = estado.get("ensino_pendente")
        if pendente:
            estado["ensino_pendente"] = None
            # nada vai direto pro IM: espera a revisão dela (aba 📝 Revisão)
            item_id = self.revisao_adicionar("bloco_novo", {"entrada": pendente["entrada"], "reacao": pendente["reacao"], "resposta": prompt})
            item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
            return self._revisao_na_conversa(item, estado, f"Obrigado, {t} ❤️ Antes de guardar, deixa eu pensar um pouco sobre isso.")
        completar = estado.get("completar_pendente")
        if completar:
            estado["completar_pendente"] = None
            if completar.get("sugestao_resposta"):  # ele tinha chutado a RESPOSTA (e o contexto dela)
                sug, lido = completar["sugestao_resposta"], self._ler_sim_ou_nao(norm)
                alvo = self.blocos[self.mapa_bloco[completar["bloco_id"]]] if completar["bloco_id"] in self.mapa_bloco else None
                if lido == "sim" and alvo is not None:  # o sim dela ao palpite já é a confirmação
                    dados = {"bloco_id": completar["bloco_id"], "entrada": alvo["entrada"].get("texto", ""),
                             "reacao": alvo["entrada"].get("reacao", ""), "resposta": f"{sug['resposta']} {sug['reacao']}".strip(),
                             **({"contexto_saida": sug["contexto"]} if sug["contexto"] else {}),
                             **({"pensamento_saida": sug["pensamento"]} if sug.get("pensamento") else {})}
                    item_id = self.revisao_adicionar("bloco_novo", dados)
                    resultado = self.revisao_aprovar(item_id, dados, "completar_bloco", estado)
                    return {"texto": "Que bom que eu acertei ❤️ Guardado: agora eu sei responder isso.",
                            "meta": f"{resultado} (palpite aprovado por ela)", "interno": None}
                if lido == "nao" and alvo is not None:
                    estado["completar_pendente"] = {"bloco_id": completar["bloco_id"], "campo": "texto da saída"}
                    return {"texto": f"Ah, errei então. O que eu respondo quando me disserem “{alvo['entrada'].get('texto', '')}”, {t}?",
                            "meta": "palpite de resposta recusado", "interno": None}
                completar = {"bloco_id": completar["bloco_id"], "campo": "texto da saída"}  # ela escreveu a resposta dela
            if completar.get("sugestao"):  # ele tinha chutado o contexto
                lido = self._ler_sim_ou_nao(norm)
                if lido == "sim":  # o sim dela ao palpite JÁ é a confirmação: completa o bloco
                    item_id = self.revisao_adicionar("completar", {"bloco_id": completar["bloco_id"], "campo": completar["campo"],
                                                                   "valor": completar["sugestao"]})
                    item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
                    r = self._revisao_concluir(item, estado)
                    r["texto"] = "Que bom que eu acertei ❤️ " + r["texto"].replace(f"Guardado, {t} ❤️ ", "Guardado: ", 1)
                    return r
                if lido == "nao":
                    estado["completar_pendente"] = {k: v for k, v in completar.items() if k != "sugestao"}
                    return {"texto": f"Ah, errei então. Qual é o {completar['campo']} dele, {t}?", "meta": "palpite de contexto recusado",
                            "interno": None}
            item_id = self.revisao_adicionar("completar", {"bloco_id": completar["bloco_id"], "campo": completar["campo"], "valor": prompt})
            item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
            return self._revisao_na_conversa(item, estado, f"Obrigado, {t} ❤️", refletir=completar["campo"] != "texto da saída")
        # ele deixou uma curiosidade no ar e ela respondeu contando sobre aquilo: vira o que ele sabe sobre a palavra
        cur = estado.pop("curiosidade_aberta", None)
        if cur and _normalizar(cur) in norm and self.ler_intencao(prompt)["intencao"] == "conteudo":
            item_id = self.revisao_adicionar("bloco_novo", {"entrada": f"O que é {cur}?", "reacao": "", "resposta": prompt.strip()})
            self._marcar_curiosidade(cur, "respondida")
            item = next(i for i in self.revisao_pendentes() if i["id"] == item_id)
            return self._revisao_na_conversa(item, estado, f"Ahh, então é isso ❤️ Obrigado por me contar sobre “{cur}”.")
        if estado.get("modo_conversa"):  # a conversa: discussão em andamento, ou ela pedindo pra guardar a conclusão
            if estado.get("discussao"):
                estado.pop("palpite_aberto", None)
                r = self._tratar_discussao(estado["discussao"], prompt, norm, estado)
                if r is not None:
                    return r
            elif estado.get("conclusao") and self.ler_intencao(prompt)["intencao"] == "guardar":
                return self._salvar_conclusao(estado)
            elif ultima_fala and not estado.get("palpite_aberto") and (
                    self._ler_sim_ou_nao(norm) == "nao" or (len(norm.split()) <= 4 and self._DISCORDA.search(norm))):
                estado.pop("semelhanca_pendente", None)
                return self._discutir(ultima_fala, estado, recusada=ultima_fala.get("resposta"))
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
        rascunho = estado.pop("rascunho_pendente", None)
        if rascunho:
            r = self._tratar_rascunho(rascunho, norm, estado)
            if r is not None:
                return r
        aproximacao = estado.pop("aproximacao_pendente", None)
        if aproximacao:
            r = self._tratar_aproximacao(aproximacao, norm, estado)
            if r is not None:
                return r
        aberto = estado.pop("palpite_aberto", None)
        if aberto:
            r = self._tratar_palpite_aberto(aberto, prompt, norm, estado)
            if r is not None:
                return r
        return self.responder(prompt, estado)
