# CÉREBRO DO ADAM -- protótipo isolado, ainda não integrado ao lovely_test.py.
#
# Princípios (definidos pela Thaís, 2026-09-20):
#   1) Rede PEQUENA de propósito -- é protótipo, não precisa de escala de
#      data center.
#   2) Integração de campos, NUNCA campos globais. Nada de tabela única
#      compartilhada às cegas por todo o corpus sem estrutura.
#   3) X=Y, E=S: entrada e saída não são dois sistemas separados (um
#      "encoder" com vocabulário próprio, um "decoder" com outro). São o
#      MESMO fio contínuo -- os marcadores da própria Thaís já provam isso
#      (0.1 até 0.17 sem quebra nenhuma entre entrada e saída). Um bloco
#      inteiro é UMA sequência só; a saída é a continuação da entrada,
#      nunca uma tradução para outro espaço.
#   4) Posição = índice real do marcador (seno/cosseno), nunca posição
#      relativa inventada, nunca fora de ordem.
#   5) Bloco = identidade própria, contida no índice mãe (círculos
#      concêntricos) -- nunca solta, nunca confundida com outro bloco.
#
# Roda como app Streamlit próprio: `streamlit run cerebro_adam.py`.

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(__file__).parent
ARQUIVO_MEMORIA = PROJ / "Adam_Lovely_memory.json"
EMBED_DIM = 32          # pequeno de propósito
N_CAMADAS = 2            # pequeno de propósito
N_HEADS = 2               # pequeno de propósito
MAX_MARKER_IDX = 500     # generoso pro corpus pessoal atual, nunca precisa de milhares
ESCALA_BLOCO = 12.0      # 2026-09-22, Thaís: blocos voltaram a se misturar no mapa 3D com 4.0 -- aumentado pra forçar mais distância real entre eles (mesma constante usada no treino E no mapa, nunca só de enfeite)


# ────────────────────────────────────────────────────────────────────────
# INSEPA mínimo (tokenizador + ALNULU), copiado isolado -- este arquivo
# não importa nada de lovely_test.py de propósito, pra poder evoluir sem
# arrastar a bagagem do modelo antigo junto.
# ────────────────────────────────────────────────────────────────────────
import re

_ALNULU_MAPA = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': -10, 'K': 11, 'L': 12, 'M': -13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,
    'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': -22, 'W': 23, 'X': 24, 'Y': -25, 'Z': 26,
}


def tokenizar(texto: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", texto, re.UNICODE)


def alnulu_string(palavra: str) -> str:
    """Forma canônica ALNULU de uma palavra -- a 'semente universal' (Camada 1)."""
    letras = [c.upper() for c in palavra if c.isalpha()]
    valores = [str(_ALNULU_MAPA.get(c, 0)) for c in letras]
    return "-".join(valores) if valores else palavra.lower()


def dividir_travessao(texto: str) -> Tuple[str, str, str]:
    partes = texto.split("—")
    if len(partes) < 3:
        return texto, "", ""
    antes = partes[0]
    fala = partes[1]
    depois = "—".join(partes[2:])
    return antes, fala, depois


# ────────────────────────────────────────────────────────────────────────
# Carregar corpus real (só lê aqui; só escreve quando a criadora ensina ou
# confirma uma variação no chat, sempre com backup antes).
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_memoria() -> dict:
    return json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))


# Rótulos de campo -- pequeno conjunto FECHADO e SABIDO (nunca adivinhado).
ROTULOS_ENTRADA = ["TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"]
ROTULOS_SAIDA = ["TEDSA", "FADES", "TEFSA", "RS", "CS"]
TODOS_ROTULOS = ROTULOS_ENTRADA + ROTULOS_SAIDA
ROTULO_PARA_IDX = {r: i for i, r in enumerate(TODOS_ROTULOS)}

# 2026-09-20, regra da Thaís: contexto (CE na entrada, CS na saída) NUNCA
# aparece pro usuário -- é insumo interno pro Adam calcular intenção (e no
# futuro, sentimento). O que o usuário vê é só texto de saída + reação.
ROTULOS_VISIVEIS_AO_USUARIO = ["TEDSA", "FADES", "TEFSA", "RS"]


class TokenDoBloco:
    """Um token real dentro de UMA sequência única entrada+saída -- nunca
    dois sistemas separados. Carrega tudo que já é sabido (rótulo, posição
    do marcador, bloco) -- nada disso é adivinhado pela rede."""
    __slots__ = ("palavra", "rotulo", "marker_idx", "bloco_id")

    def __init__(self, palavra: str, rotulo: str, marker_idx: int, bloco_id: int):
        self.palavra = palavra
        self.rotulo = rotulo
        self.marker_idx = marker_idx
        self.bloco_id = bloco_id


def sequencia_do_bloco(bloco: dict) -> List[TokenDoBloco]:
    """Monta a sequência ÚNICA (entrada seguida de saída, mesma ordem dos
    marcadores reais -- 0.1 até o fim) -- X=Y, E=S: é UMA sequência, nunca
    duas."""
    bloco_id = bloco["bloco_id"]
    entrada = bloco["entrada"]
    saida = bloco["saidas"][0]
    seq: List[TokenDoBloco] = []

    def filho(marcador: str) -> int:
        return int(marcador.split(".", 1)[1])

    # 2026-09-22, Thaís: cada marcador carrega a própria palavra junto --
    # "TEXE": [{"marcador": "0.1", "palavra": "Olá"}] -- nunca só o número
    # separado num campo "texto" à parte pra decifrar depois. Entrada:
    # TEXE/FADEN/TEFE + RE + CE + PIDE; Saída: TEDSA/FADES/TEFSA + RS + CS --
    # MESMA sequência, continuando.
    for rotulo in ("TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"):
        for item in entrada["tokens"].get(rotulo, []):
            seq.append(TokenDoBloco(item["palavra"], rotulo, filho(item["marcador"]), bloco_id))
    for rotulo in ("TEDSA", "FADES", "TEFSA", "RS", "CS"):
        for item in saida["tokens"].get(rotulo, []):
            seq.append(TokenDoBloco(item["palavra"], rotulo, filho(item["marcador"]), bloco_id))

    seq.sort(key=lambda t: t.marker_idx)
    return seq


# ────────────────────────────────────────────────────────────────────────
# O cérebro em si.
# ────────────────────────────────────────────────────────────────────────
class PosicaoPorMarcador(nn.Module):
    """Seno/cosseno indexado pelo índice REAL do marcador -- índice mãe
    ordenado. Determinístico, nunca aprendido do zero, nunca perde ordem."""

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        posicao = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(posicao * div)
        pe[:, 1::2] = torch.cos(posicao * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.pe[idx.clamp(max=self.pe.size(0) - 1)]


class CerebroAdam(nn.Module):
    """Uma sequência só (E=S), integração de campos por rótulo+posição+
    bloco -- nunca campos globais soltos. Pequeno de propósito."""

    def __init__(self, vocab_size: int, num_blocos_total: int):
        super().__init__()
        self.embed_forma = nn.Embedding(vocab_size, EMBED_DIM)  # semente universal (forma ALNULU)
        self.pos_marcador = PosicaoPorMarcador(EMBED_DIM, MAX_MARKER_IDX)
        self.embed_rotulo = nn.Embedding(len(TODOS_ROTULOS), EMBED_DIM)  # já sabido, nunca adivinhado
        self.embed_bloco = nn.Embedding(num_blocos_total, EMBED_DIM)  # identidade, contida no índice mãe

        camada = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=EMBED_DIM * 4,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(camada, num_layers=N_CAMADAS)
        # Peso amarrado (tied) com embed_forma -- prever a próxima palavra é
        # literalmente "qual semente do MESMO vocabulário vem agora", nunca
        # um espaço de saída à parte (isso é o E=S na prática, na própria
        # matemática da previsão).
        self.saida = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.saida.weight = self.embed_forma.weight

    def forward(self, ids: torch.Tensor, marker_idx: torch.Tensor, rotulo_idx: torch.Tensor, bloco_idx: torch.Tensor) -> torch.Tensor:
        # ids/marker_idx/rotulo_idx/bloco_idx: (batch, seq_len)
        # 2026-09-20: bloco_idx precisa DOMINAR a posição no espaço (cada
        # bloco no seu espacinho, círculo dentro do índice mãe) -- sem
        # escala extra ele só somava igual aos outros termos, e como o
        # marker_idx já basta pra resolver a tarefa (também identifica o
        # bloco sozinho), o treino não tinha motivo pra empurrar blocos
        # pra longe um do outro. ESCALA_BLOCO força blocos diferentes a
        # nascerem em regiões bem separadas antes de qualquer variação
        # interna.
        x = (
            self.embed_forma(ids)
            + self.pos_marcador(marker_idx)
            + self.embed_rotulo(rotulo_idx)
            + ESCALA_BLOCO * self.embed_bloco(bloco_idx)
        )
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim) -- convenção do TransformerEncoder
        seq_len = x.size(0)
        mascara_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mascara_causal)
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        return self.saida(x)  # (batch, seq_len, vocab_size) -- logits pro PRÓXIMO token


# ────────────────────────────────────────────────────────────────────────
# Dataset: um bloco = uma sequência de treino (prever o próximo token em
# cada posição, causal, a sequência inteira -- entrada continua em saída).
# ────────────────────────────────────────────────────────────────────────
def montar_vocabulario(sequencias: List[List[TokenDoBloco]]) -> Dict[str, int]:
    formas = sorted({alnulu_string(t.palavra) for seq in sequencias for t in seq if t.palavra})
    return {forma: i for i, forma in enumerate(formas)}


def montar_forma_para_palavra(sequencias: List[List[TokenDoBloco]]) -> Dict[str, str]:
    """A forma ALNULU é a semente universal (Camada 1) -- várias palavras
    reais podem colapsar na mesma forma. Pra EXIBIR, guarda uma palavra de
    verdade representando cada forma (a mais usada no corpus)."""
    from collections import Counter
    contagem: Dict[str, Counter] = {}
    for seq in sequencias:
        for t in seq:
            if not t.palavra:
                continue
            forma = alnulu_string(t.palavra)
            contagem.setdefault(forma, Counter())[t.palavra] += 1
    return {forma: c.most_common(1)[0][0] for forma, c in contagem.items()}


def montar_tensores(seq: List[TokenDoBloco], vocab: Dict[str, int], mapa_bloco: Dict[int, int]) -> Dict[str, torch.Tensor]:
    ids = [vocab[alnulu_string(t.palavra)] for t in seq]
    marker_idx = [t.marker_idx for t in seq]
    rotulo_idx = [ROTULO_PARA_IDX[t.rotulo] for t in seq]
    bloco_idx = [mapa_bloco[t.bloco_id] for t in seq]
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "marker_idx": torch.tensor(marker_idx, dtype=torch.long),
        "rotulo_idx": torch.tensor(rotulo_idx, dtype=torch.long),
        "bloco_idx": torch.tensor(bloco_idx, dtype=torch.long),
    }


def treinar(model: CerebroAdam, blocos_tensores: List[Dict[str, torch.Tensor]], epocas: int, lr: float = 3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    historico = []
    model.train()
    for ep in range(epocas):
        perda_total = 0.0
        for t in blocos_tensores:
            opt.zero_grad()
            ids = t["ids"].unsqueeze(0)
            logits = model(ids, t["marker_idx"].unsqueeze(0), t["rotulo_idx"].unsqueeze(0), t["bloco_idx"].unsqueeze(0))
            # prevê o token seguinte em cada posição -- causal, a sequência
            # inteira (entrada continuando em saída, X=Y na prática).
            alvo = ids[:, 1:].reshape(-1)
            pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
            perda = F.cross_entropy(pred, alvo)
            perda.backward()
            opt.step()
            perda_total += perda.item()
        historico.append(perda_total / len(blocos_tensores))
    return historico


def gerar_continuacao(model: CerebroAdam, seq_entrada: List[TokenDoBloco], seq_saida_molde: List[TokenDoBloco], vocab: Dict[str, int], idx_para_palavra: Dict[int, str], mapa_bloco: Dict[int, int], bloco_id: int) -> List[str]:
    """Gera a PALAVRA em cada posição da saída -- mas o RÓTULO e o ÍNDICE DE
    MARCADOR de cada posição vêm do MOLDE estrutural (seq_saida_molde), nunca
    inventados/congelados durante a geração. 2026-09-20, bug achado pela
    Thaís: a versão anterior repetia o ÚLTIMO rótulo da entrada (PIDE) pra
    sempre depois que a entrada acabava -- o modelo tinha aprendido com o
    rótulo certo em cada posição no treino, e na geração recebia rótulo
    errado (congelado), o que por si só já é 'campo global' de novo:
    informação que é SABIDA (que campo vem a seguir -- toda saída segue
    TEDSA/FADES/TEFSA -> RS -> CS, é sempre essa estrutura) virando chute.
    A palavra em cada posição é livre (gerada de verdade); o rótulo/posição
    nunca é."""
    model.eval()
    t = montar_tensores(seq_entrada, vocab, mapa_bloco)
    ids = t["ids"].unsqueeze(0)
    marker_idx = t["marker_idx"].unsqueeze(0)
    rotulo_idx = t["rotulo_idx"].unsqueeze(0)
    bloco_idx = t["bloco_idx"].unsqueeze(0)
    gerados = []
    with torch.no_grad():
        for tok_molde in seq_saida_molde:
            logits = model(ids, marker_idx, rotulo_idx, bloco_idx)
            proximo = logits[0, -1].argmax().item()
            gerados.append(idx_para_palavra.get(proximo, "?"))
            novo_rotulo = ROTULO_PARA_IDX[tok_molde.rotulo]
            novo_bloco = mapa_bloco[bloco_id]
            ids = torch.cat([ids, torch.tensor([[proximo]])], dim=1)
            marker_idx = torch.cat([marker_idx, torch.tensor([[tok_molde.marker_idx]])], dim=1)
            rotulo_idx = torch.cat([rotulo_idx, torch.tensor([[novo_rotulo]])], dim=1)
            bloco_idx = torch.cat([bloco_idx, torch.tensor([[novo_bloco]])], dim=1)
    return gerados


def _projetar_3d(matriz: np.ndarray) -> np.ndarray:
    """Projeção linear (via SVD) só pra caber na tela em 3D -- puramente
    visual, mesmo método já usado no cérebro antigo (insepa_neuronios.py)."""
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


def vetores_por_token(model: CerebroAdam, sequencias: List[List[TokenDoBloco]], vocab: Dict[str, int], mapa_bloco: Dict[int, int]) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """Devolve, pra CADA ocorrência real de token no corpus (não cada forma
    única), o vetor combinado que entra no transformer (forma + posição do
    marcador + rótulo + bloco) -- é essa combinação que testa se a
    integração de campos está funcionando, não só a tabela de forma sozinha
    (que é universal por design, nunca vai mostrar bloco)."""
    model.eval()
    vetores, palavras, blocos_id, rotulos = [], [], [], []
    with torch.no_grad():
        for seq in sequencias:
            t = montar_tensores(seq, vocab, mapa_bloco)
            x = (
                model.embed_forma(t["ids"])
                + model.pos_marcador(t["marker_idx"])
                + model.embed_rotulo(t["rotulo_idx"])
                + ESCALA_BLOCO * model.embed_bloco(t["bloco_idx"])
            )
            for i, tok in enumerate(seq):
                vetores.append(x[i].numpy())
                palavras.append(tok.palavra)
                blocos_id.append(tok.bloco_id)
                rotulos.append(tok.rotulo)
    return np.array(vetores), palavras, blocos_id, rotulos


def grafico_cerebro_3d(model: CerebroAdam, sequencias: List[List[TokenDoBloco]], vocab: Dict[str, int], mapa_bloco: Dict[int, int]) -> go.Figure:
    vetores, palavras, blocos_id, rotulos = vetores_por_token(model, sequencias, vocab, mapa_bloco)
    coords = _projetar_3d(vetores)
    cores_paleta = ["#e8935a", "#7c93f5", "#4fc9a8", "#e2607a", "#a78bfa"]
    fig = go.Figure()
    for bid in sorted(set(blocos_id)):
        idxs = [i for i, b in enumerate(blocos_id) if b == bid]
        cor = cores_paleta[(bid - 1) % len(cores_paleta)]
        fig.add_trace(go.Scatter3d(
            x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
            mode="markers+text",
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
# Streamlit
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cérebro do Adam", layout="wide")
st.title("🧠 Cérebro do Adam -- protótipo isolado")
st.caption("Pequeno de propósito. Integração de campos, não campos globais. X=Y, E=S: uma sequência só.")

if not ARQUIVO_MEMORIA.exists():
    st.error(f"Não achei {ARQUIVO_MEMORIA.name} ao lado deste arquivo. Coloque o Adam_Lovely_memory.json na mesma pasta do cerebro_adam.py e recarregue.")
    st.stop()

memoria = carregar_memoria()
blocos = memoria["IM"]["0"]["blocos"]

sequencias = [sequencia_do_bloco(b) for b in blocos]
vocab = montar_vocabulario(sequencias)
forma_para_palavra = montar_forma_para_palavra(sequencias)
idx_para_palavra = {i: forma_para_palavra.get(f, f) for f, i in vocab.items()}
mapa_bloco = {b["bloco_id"]: i for i, b in enumerate(blocos)}
tensores = [montar_tensores(seq, vocab, mapa_bloco) for seq in sequencias]


import hashlib
import unicodedata
from datetime import date

from insepa_autonomia import confirmar_estrutura_baskara, encontrar_texto_mais_parecido

CKPT_CEREBRO = PROJ / "cerebro_adam_ckpt.pt"
ARQUIVO_ESTADO = PROJ / "cerebro_adam_estado.json"
EPOCAS_PADRAO = 2000  # blocos longos precisam disso pra chegar em 100% (confirmado)
_TEXTO_ENTRADA = ("TEXE", "FADEN", "TEFE")
_TEXTO_SAIDA = ("TEDSA", "FADES", "TEFSA")
_PONTUACAO_SOZINHA = set(".,!?;:…-—")


def _e_palavra(p: str) -> bool:
    return bool(p) and any(c.isalnum() for c in p)


def formas_do_texto(texto: str) -> set:
    return {alnulu_string(p) for p in tokenizar(texto) if _e_palavra(p)}


def saida_e_placeholder(seq: List[TokenDoBloco]) -> bool:
    """'0.0' = 'esse lugar está vazio'. Não é lixo a esconder: o Adam
    entende que o bloco ainda não tem resposta e diz isso."""
    texto = "".join(t.palavra for t in seq if t.rotulo in _TEXTO_SAIDA)
    return re.fullmatch(r"\d+\.0", texto.strip()) is not None


def reacoes_conhecidas() -> set:
    """Reações que já existem de verdade nos blocos, canônicas ou Multivars
    (correspondência exata, nunca adivinhada)."""
    rs = set()
    for b in blocos:
        candidatas = [b["entrada"].get("reacao", ""), b["saidas"][0].get("reacao", "")]
        candidatas += list(b["entrada"].get("Multivars_Reacao_Entrada", []))
        for r in candidatas:
            if r and not re.fullmatch(r"\d+\.0", r):
                rs.add(r)
    return rs


def separar_reacao(texto: str) -> Tuple[str, str]:
    """Entrada crua = texto + reação. A reação é o último 'pedaço' da mensagem
    quando ele é uma reação já conhecida (exata) ou um emoji/símbolo sem
    letra nem número. Sem reação, devolve '' (vira o placeholder 0.0)."""
    partes = texto.strip().rsplit(None, 1)
    if len(partes) == 2:
        ultimo = partes[1]
        sem_alnum = not any(c.isalnum() for c in ultimo)
        if ultimo in reacoes_conhecidas() or (sem_alnum and not set(ultimo) <= _PONTUACAO_SOZINHA):
            return partes[0].strip(), ultimo
    return texto.strip(), ""


def _vazio(valor) -> bool:
    v = str(valor or "").strip()
    return not v or re.fullmatch(r"\d+\.0", v) is not None


def campos_faltando(bloco: dict) -> List[str]:
    """Bloco 100% completo = entrada (texto, reação, contexto, pensamento) +
    saída (texto, reação, contexto) todos preenchidos. Só então ele conta como
    dado concreto. O 'pensamento' da saída é só a etiqueta 'Confirmado pela
    criadora', não um campo de conteúdo, por isso não entra."""
    e, s = bloco["entrada"], bloco["saidas"][0]
    campos = [("texto", e.get("texto")), ("reação", e.get("reacao")), ("contexto", e.get("contexto")),
              ("pensamento", e.get("pensamento_interno")), ("texto da saída", (s.get("textos") or [""])[0]),
              ("reação da saída", s.get("reacao")), ("contexto da saída", s.get("contexto"))]
    return [nome for nome, v in campos if _vazio(v)]


FRIEZA = "0.0"  # 2026-09-21/22, Thaís: ausência de reação E de qualquer indicativo
# emocional é frieza de verdade -- não é neutro, não é julgamento sobre a
# pessoa, é só a ausência de alma daquele texto por natureza. "!", "?" e
# "..." carregam cada um um indicativo emocional PRÓPRIO (não são frieza):
# tom não é decidido só por pontuação -- é texto + contexto + emoção
# triangulados juntos (outra camada do INSEPA); sem contexto ainda, o Adam
# não decide o tom sozinho, só marca o indicativo e ESPERA.


def reacao_efetiva(reacao: str) -> str:
    return FRIEZA if _vazio(reacao) else reacao


def reacoes_aceitas(i: int) -> List[str]:
    """Reação canônica do bloco (vazia = frieza, 0.0) + Multivars de reação
    já registradas."""
    e = blocos[i]["entrada"]
    return [reacao_efetiva(e.get("reacao", ""))] + [r for r in e.get("Multivars_Reacao_Entrada", []) if not _vazio(r)]


def achar_bloco_parecido(texto: str, reacao: str, seqs: List[List[TokenDoBloco]]) -> Tuple[int, float, bool]:
    """Correspondência EXATA de forma ALNULU (sem embedding, sem distância
    vetorial): quantas sementes o que você disse divide com o texto de
    entrada de cada bloco (canônico ou Multivar registrado), e se a reação é
    exatamente uma das aceitas. Devolve (índice, score do texto, reação
    bateu) ou (-1, 0.0, False). Em empate de texto: reação igual ganha,
    depois bloco com resposta ganha de bloco vazio."""
    formas_user = formas_do_texto(texto)
    if not formas_user:
        return -1, 0.0, False
    reacao_user = reacao  # já vem como valor de tom (reação, frieza 0.0 ou '' = desconhecido)
    melhor, melhor_chave, melhor_reacao = -1, (0.0,), False
    for i, seq in enumerate(seqs):
        variantes = [{alnulu_string(t.palavra) for t in seq if t.rotulo in _TEXTO_ENTRADA and _e_palavra(t.palavra)}]
        variantes += [formas_do_texto(v) for v in blocos[i]["entrada"].get("Multivars_Texto_Entrada", [])]
        scores = [len(formas_user & f) / len(formas_user | f) for f in variantes if f]
        if not scores:
            continue
        score = max(scores)
        aceitas = reacoes_aceitas(i)
        reacao_igual = bool(reacao_user) and reacao_user in aceitas
        chave = (score, reacao_igual, not saida_e_placeholder(seq), not campos_faltando(blocos[i]))
        if score > 0 and chave > melhor_chave:
            melhor, melhor_chave, melhor_reacao = i, chave, reacao_igual
    return melhor, (melhor_chave[0] if melhor >= 0 else 0.0), melhor_reacao


def _arrumar_espacos(texto: str) -> str:
    return re.sub(r"\s+([,.!?;:])", r"\1", texto).strip()


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


FRASES_SEM_TAREFA = ("nao tenho o que fazer", "nao tenho nada pra fazer", "nao tenho nada para fazer",
                     "sem nada pra fazer", "sem nada para fazer", "estou entediada", "to entediada", "vamos revisar")
FRASES_CANCELAR = ("deixa pra la", "deixa para la", "cancelar", "cancela", "esquece", "agora nao")

FRASE_NAO_SEI = "Criadora, não sei o que isso significa. Pode me ensinar o que eu devo responder?"


@st.cache_resource(show_spinner=False)
def modelo_treinado() -> CerebroAdam:
    """Treina uma vez (ou carrega o checkpoint salvo, se o corpus não mudou)."""
    # ESCALA_BLOCO entra na assinatura de propósito: os pesos aprendidos de
    # embed_bloco só fazem sentido junto da escala usada no treino -- mudar
    # a constante sem retreinar deixaria o mapa "bonito" só de mentirinha,
    # sem o modelo ter aprendido a separação de verdade (2026-09-22).
    assinatura = hashlib.sha256(
        (str(sorted(vocab.items())) + str([len(s) for s in sequencias]) + f"|escala_bloco={ESCALA_BLOCO}").encode("utf-8")
    ).hexdigest()
    m = CerebroAdam(vocab_size=len(vocab), num_blocos_total=len(blocos))
    if CKPT_CEREBRO.exists():
        try:
            ck = torch.load(CKPT_CEREBRO)
            if ck.get("assinatura") == assinatura:
                m.load_state_dict(ck["state_dict"])
                m.eval()
                return m
        except Exception:
            pass
    treinar(m, tensores, EPOCAS_PADRAO)
    torch.save({"state_dict": m.state_dict(), "assinatura": assinatura}, CKPT_CEREBRO)
    m.eval()
    return m


def modelo_atual() -> CerebroAdam:
    return st.session_state.get("model_cerebro") or modelo_treinado()


# ────────────────────────────────────────────────────────────────────────
# Consciente (aprendizado): a criadora ensina no chat -> vira bloco novo,
# marcadores continuando de onde o último parou, sem buraco nenhum.
# ────────────────────────────────────────────────────────────────────────
def _backup_memoria_uma_vez() -> None:
    if not st.session_state.get("backup_ensino_feito"):
        carimbo = __import__("time").strftime("%Y%m%d_%H%M%S")
        ARQUIVO_MEMORIA.with_name(f"Adam_Lovely_memory.BACKUP_antes_do_ensino_{carimbo}.json").write_text(
            ARQUIVO_MEMORIA.read_text(encoding="utf-8"), encoding="utf-8")
        st.session_state["backup_ensino_feito"] = True


def registrar_variacao(bloco_id: int, texto: str, reacao: str) -> None:
    """Registra a frase (e a reação) como Multivar EXATO do bloco -- nunca
    afrouxa comparação nenhuma: a partir daí ela bate por igualdade."""
    memoria_disco = json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))
    bloco = next(b for b in memoria_disco["IM"]["0"]["blocos"] if b["bloco_id"] == bloco_id)
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
    _backup_memoria_uma_vez()
    tmp = ARQUIVO_MEMORIA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_MEMORIA)
    carregar_memoria.clear()


_FAMS_ENTRADA = ["TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE"]
_FAMS_SAIDA = ["TEDSA", "FADES", "TEFSA", "RS", "CS"]


def renumerar_universo(universo: dict) -> None:
    """Quando um bloco já criado é editado (ex.: completar um campo que
    faltava), a forma correta é renumerar a sequência do universo INTEIRO
    em ordem, do zero -- nunca deixar marcador órfão/gapado no meio.

    2026-09-22, Thaís: cada marcador carrega a própria palavra junto no
    mesmo item -- {"marcador": "0.1", "palavra": "Olá"} -- nunca só o
    número separado, pra fazer sentido junto com o resto da arquitetura
    (a palavra nunca vira índice puro pra decifrar depois)."""
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
                      "CS": tokenizar(s.get("contexto", ""))}
        sai_total = []
        for fam in _FAMS_SAIDA:
            pares = prox(palavras_s[fam])
            s["tokens"][fam] = pares
            sai_total += [p["marcador"] for p in pares]
        s["tokens"]["TOTAL"] = sai_total
        s["fim"] = sai_total[-1] if sai_total else s.get("fim", "")

    universo["ultimo_child"] = f"0.{cursor}"


def completar_campo(bloco_id: int, campo: str, valor_bruto: str) -> None:
    """Preenche um campo que faltava num bloco JÁ EXISTENTE -- nunca cria um
    bloco novo pra isso. Depois renumera o universo inteiro (Thaís,
    2026-09-13: é assim que se edita um bloco já criado, sem deixar
    marcador órfão)."""
    memoria_disco = json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))
    universo = memoria_disco["IM"]["0"]
    bloco = next(b for b in universo["blocos"] if b["bloco_id"] == bloco_id)
    e, s = bloco["entrada"], bloco["saidas"][0]
    valor = valor_bruto.strip()
    _backup_memoria_uma_vez()

    if campo == "contexto":
        e["contexto"] = valor
    elif campo == "pensamento":
        e["pensamento_interno"] = valor
    elif campo == "texto da saída":
        texto_saida, reacao_saida = separar_reacao(valor)
        s["textos"] = [texto_saida]
        if reacao_saida and _vazio(s.get("reacao")):
            s["reacao"] = reacao_saida
    elif campo == "reação da saída":
        s["reacao"] = valor
    elif campo == "contexto da saída":
        s["contexto"] = valor
    elif campo == "texto":
        e["texto"] = valor
    elif campo == "reação":
        e["reacao"] = valor
    else:
        return

    renumerar_universo(universo)
    tmp = ARQUIVO_MEMORIA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_MEMORIA)
    carregar_memoria.clear()
    modelo_treinado.clear()
    st.session_state.pop("model_cerebro", None)


def ensinar_bloco(entrada_texto: str, entrada_reacao: str, resposta_bruta: str) -> int:
    """Grava um bloco novo no universo 0 e devolve o bloco_id. Contexto e
    pensamento interno ficam no ponto neutro 0.0 (a criadora adiciona
    depois). Marcadores: monta o bloco com tokens vazios e deixa a
    renumeração do universo inteiro (mesma função de completar_campo)
    gerar os pares marcador+palavra certos, continuando sem gap."""
    saida_texto, saida_reacao = separar_reacao(resposta_bruta)
    memoria_disco = json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))
    universo = memoria_disco["IM"]["0"]
    _backup_memoria_uma_vez()

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
            "contexto": neutro,  # sem "pensamento_interno" na saída, como nos blocos 1-3; "Confirmado pela criadora" não é contexto nem pensamento válido
            "tokens": {}, "fim": "",
        }],
        "meta": {"origem": "ensino_no_chat_do_cerebro"},
    })
    renumerar_universo(universo)
    tmp = ARQUIVO_MEMORIA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_MEMORIA)
    # o corpus mudou: esquece o que estava em cache pra reler + retreinar
    carregar_memoria.clear()
    modelo_treinado.clear()
    st.session_state.pop("model_cerebro", None)
    return novo_id


def blocos_incompletos() -> List[int]:
    """2026-09-22, Thaís: ela quer ver o Adam descobrindo sozinho que há
    blocos incompletos (não só saída vazia -- qualquer campo faltando) e
    pedindo os dados concretos."""
    return [i for i, b in enumerate(blocos) if campos_faltando(b)]


_PRIORIDADE_CAMPOS = ["texto da saída", "contexto", "pensamento", "reação da saída", "contexto da saída", "texto", "reação"]


def campo_prioritario(faltando: List[str]) -> str:
    return next((c for c in _PRIORIDADE_CAMPOS if c in faltando), faltando[0])


def pedir_campo_faltando(bloco: dict, faltando: List[str], tom_rotulo: str = "") -> Dict[str, object]:
    campo = campo_prioritario(faltando)
    st.session_state["completar_pendente"] = {"bloco_id": bloco["bloco_id"], "campo": campo}
    mencao_tom = (f" Percebi um {tom_rotulo} nisso -- com o contexto eu já sei triangular o tom."
                  if campo == "contexto" and tom_rotulo.startswith("indicativo emocional") else "")
    return {
        "texto": f"Eu reconheço esse assunto (bloco {bloco['bloco_id']}), mas ainda não sei o(a) {campo} dele.{mencao_tom} Pode me dizer?",
        "meta": f"bloco {bloco['bloco_id']} · falta: {', '.join(faltando)}", "interno": None,
    }


def revisar_dados() -> Dict[str, object]:
    abertura = "Entendo. Então que tal a gente revisar alguns dos meus dados, criadora?"
    incompletos = blocos_incompletos()
    if not incompletos:
        return {"texto": abertura + " Todos os meus blocos já estão completos. Se quiser, me diga uma frase nova e eu peço pra você me ensinar.", "meta": None, "interno": None}
    bloco = blocos[incompletos[0]]
    ent = bloco["entrada"]
    reacao = "" if _vazio(ent.get("reacao", "")) else ent.get("reacao", "")
    r = pedir_campo_faltando(bloco, campos_faltando(bloco))
    r["texto"] = (f'{abertura} Tenho {len(incompletos)} bloco(s) incompleto(s). Por exemplo, o bloco {bloco["bloco_id"]}: '
                  f'"{ent["texto"]}" {reacao}'.rstrip() + f". Ainda falta o(a) {r["meta"].split("falta: ")[1].split(",")[0]}. O que eu devo registrar aí?")
    return r


def lembrete_do_dia() -> Dict[str, object] | None:
    """'Alarme' diário: a primeira vez que a criadora abre o chat no dia, o
    Adam puxa o assunto. (Streamlit só roda com a página aberta -- é um
    lembrete ao abrir, não uma notificação com o app fechado.)"""
    hoje = date.today().isoformat()
    try:
        estado = json.loads(ARQUIVO_ESTADO.read_text(encoding="utf-8"))
    except Exception:
        estado = {}
    if estado.get("ultimo_lembrete") == hoje:
        return None
    estado["ultimo_lembrete"] = hoje
    ARQUIVO_ESTADO.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
    incompletos = len(blocos_incompletos())
    extra = f" Tenho {incompletos} bloco(s) incompleto(s)." if incompletos else ""
    return {"texto": "Oi, criadora! Hoje a gente ainda não revisou os meus dados." + extra + " Quer olhar comigo? É só dizer \"vamos revisar\".",
            "meta": None, "interno": None}


def falar_do_bloco(idx: int) -> Tuple[str, str | None]:
    """Gera a saída do bloco idx com a rede (molde e entrada reais do bloco).
    Devolve (fala visível, contexto interno)."""
    seq = sequencias[idx]
    bloco_id = blocos[idx]["bloco_id"]
    corte = next((i for i, t in enumerate(seq) if t.rotulo in ROTULOS_SAIDA), len(seq))
    entrada_real, molde = seq[:corte], seq[corte:]
    gerado = gerar_continuacao(modelo_atual(), entrada_real, molde, vocab, idx_para_palavra, mapa_bloco, bloco_id)
    pares = list(zip(molde, gerado))
    fala = _arrumar_espacos(" ".join(p for t, p in pares if t.rotulo in ROTULOS_VISIVEIS_AO_USUARIO))
    interno = ", ".join(f"{p} ({t.rotulo})" for t, p in pares if t.rotulo not in ROTULOS_VISIVEIS_AO_USUARIO)
    return fala, interno or None


def _sem_ideia(ensino: dict, meta: str | None = None) -> Dict[str, object]:
    return {"texto": FRASE_NAO_SEI, "meta": meta, "interno": None, "ensino": ensino}


def tom_da_entrada(texto: str, reacao: str) -> Tuple[str, str]:
    """Devolve (valor pra comparar, rótulo pra mostrar). Sem reação: '.' ou
    nenhuma pontuação = frieza de verdade (desalmado, não é neutro). '!',
    '?' e '...' são cada um um indicativo emocional PRÓPRIO -- o Adam ainda
    não sabe qual dos lados de cada um é (empolgação ou pânico no '!', por
    exemplo), então o valor fica '' (desconhecido pro Báskara, não pesa a
    favor nem contra) até haver contexto suficiente pra triangular."""
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


def responder(texto: str) -> Dict[str, object]:
    texto_sem_reacao, reacao = separar_reacao(texto)
    tom_valor, tom_rotulo = tom_da_entrada(texto_sem_reacao, reacao)
    idx, score, reacao_bateu = achar_bloco_parecido(texto_sem_reacao, tom_valor, sequencias)
    ensino = {"entrada": texto_sem_reacao, "reacao": reacao}
    if idx >= 0 and score >= 1.0:
        # Texto bate por inteiro com um bloco já existente.
        bloco = blocos[idx]
        faltando = campos_faltando(bloco)
        if faltando:
            r = pedir_campo_faltando(bloco, faltando, tom_rotulo)
            r["ensino"] = None  # nunca cria bloco novo aqui -- é pra COMPLETAR este
            return r
        # Cada campo da entrada vale 25%; uma entrada crua (texto + reação)
        # chega no máximo a 50% e por isso é SEMPRE incompleta -- não é
        # fonte confiável de criação sozinha.
        if reacao_bateu:
            pct = 25 + 25 * int(reacao_bateu)
            fala, interno = falar_do_bloco(idx)
            return {"texto": fala or "...", "meta": f"bloco {bloco['bloco_id']} · sua entrada {pct}% -- incompleta (texto ✓ · reação ✓) · dado do bloco: completo", "interno": interno}
        # Texto bate, bloco completo, mas o tom diverge/é indefinido --
        # Báskara confirma se ainda assim é o mesmo assunto (mesmo contexto).
        contexto = str(bloco["entrada"].get("contexto", "")).strip()
        conf = confirmar_estrutura_baskara(tom_valor, reacoes_aceitas(idx), contexto, contexto, "", "")
        meta = f"bloco {bloco['bloco_id']} · texto 100% · tom: {tom_rotulo} · Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})"
        if not conf["confirma"]:
            return _sem_ideia(ensino, meta)
        st.session_state["semelhanca_pendente"] = {"idx": idx, "entrada": texto_sem_reacao, "reacao": reacao, "etapa": "contexto", "identico": True}
        return {"texto": f"Isso parece uma “{contexto}”. Era isso que você quis dizer?", "meta": meta, "interno": None}

    # Não é dado exato. Cosseno ACHA candidato (texto parecido), Báskara CONFIRMA.
    achado = encontrar_texto_mais_parecido(texto_sem_reacao, blocos)
    if achado is None:
        return _sem_ideia(ensino)
    i = blocos.index(achado["bloco"])
    bloco = blocos[i]
    faltando = campos_faltando(bloco)
    if faltando:
        r = pedir_campo_faltando(bloco, faltando, tom_rotulo)
        r["ensino"] = None
        return r
    contexto = str(bloco["entrada"].get("contexto", "")).strip()
    conf = confirmar_estrutura_baskara(tom_valor, reacoes_aceitas(i), contexto, contexto, "", "")
    meta = (f"bloco {blocos[i]['bloco_id']} · cosseno do texto {achado['score_texto']:.0%} · tom: {tom_rotulo} · "
            f"Báskara Δ={conf['discriminante']} ({'confirma' if conf['confirma'] else 'recusa'})")
    if not conf["confirma"]:
        return _sem_ideia(ensino, meta)
    st.session_state["semelhanca_pendente"] = {
        "idx": i, "entrada": texto_sem_reacao, "reacao": reacao, "etapa": "contexto",
        "identico": achado["score_texto"] >= 0.999,
    }
    return {"texto": f"Isso parece uma \u201c{contexto}\u201d. Era isso que você quis dizer?", "meta": meta, "interno": None}


_SIM = ("sim", "isso", "exato", "exatamente", "claro", "acertou", "era", "aham", "uhum", "s")
_NAO = ("nao", "errado", "negativo", "n")


def tratar_semelhanca(sem: dict, norm: str) -> Dict[str, object] | None:
    """Continua o fluxo: contexto? -> (texto do bloco) variação? -> registra.
    Devolve None se a mensagem não é sim/não (aí segue como conversa normal)."""
    primeira = norm.split()[0] if norm else ""
    bloco = blocos[sem["idx"]]
    if primeira in _NAO:
        st.session_state["ensino_pendente"] = {"entrada": sem["entrada"], "reacao": sem["reacao"]}
        return {"texto": "Entendi. Então me ensina: o que eu devo responder quando me disserem isso?", "meta": None, "interno": None}
    if primeira not in _SIM:
        return None
    if sem["etapa"] == "contexto" and not sem["identico"]:
        st.session_state["semelhanca_pendente"] = {**sem, "etapa": "variacao"}
        return {"texto": f"Então isso se parece com \u201c{bloco['entrada']['texto']}\u201d. É uma variação dele?", "meta": None, "interno": None}
    registrar_variacao(bloco["bloco_id"], sem["entrada"], sem["reacao"])
    fala, interno = falar_do_bloco(sem["idx"])
    anotado = "Anotado, guardei como variação." if not sem["identico"] else "Anotado!"
    return {"texto": f"{anotado} {fala}", "meta": f"bloco {bloco['bloco_id']}", "interno": interno}


def _limpar_pendencias() -> bool:
    """Devolve True se havia algo pendente. 2026-09-22, Thaís: os gatilhos
    (cancelar/revisar) são reconhecidos ANTES de qualquer captura pendente
    -- nunca viram conteúdo dentro de um marcador de texto por acidente."""
    havia = bool(st.session_state.get("ensino_pendente") or st.session_state.get("completar_pendente") or st.session_state.get("semelhanca_pendente"))
    st.session_state["ensino_pendente"] = None
    st.session_state["completar_pendente"] = None
    st.session_state["semelhanca_pendente"] = None
    return havia


def tratar_mensagem(prompt: str) -> Dict[str, object]:
    norm = _normalizar(prompt)
    if any(f in norm for f in FRASES_CANCELAR):
        havia = _limpar_pendencias()
        if havia:
            return {"texto": "Tudo bem, minha adorada. A gente volta nisso quando você quiser.", "meta": None, "interno": None}
        return responder(prompt)
    if any(f in norm for f in FRASES_SEM_TAREFA):
        _limpar_pendencias()
        return revisar_dados()

    pendente = st.session_state.get("ensino_pendente")
    if pendente:
        st.session_state["ensino_pendente"] = None
        novo_id = ensinar_bloco(pendente["entrada"], pendente["reacao"], prompt)
        return {"texto": "Aprendi! Guardei como um bloco novo. Vou treinar um pouquinho pra fixar -- só um minuto.",
                "meta": f"bloco {novo_id} criado", "interno": None}
    completar = st.session_state.get("completar_pendente")
    if completar:
        st.session_state["completar_pendente"] = None
        completar_campo(completar["bloco_id"], completar["campo"], prompt)
        return {"texto": f"Aprendi! Guardei o(a) {completar['campo']} do bloco {completar['bloco_id']}. Vou treinar um pouquinho pra fixar -- só um minuto.",
                "meta": None, "interno": None}
    sem = st.session_state.pop("semelhanca_pendente", None)
    if sem:
        r = tratar_semelhanca(sem, norm)
        if r is not None:
            return r
    return responder(prompt)


# ────────────────────────────────────────────────────────────────────────
# Motor de temperatura emocional -- 2026-09-23, acrescentado ao cérebro a
# pedido da Thaís. Mesma lógica do protótipo motor_emocional.py, mesmo
# arquivo de registro (as duas telas compartilham o que já foi marcado).
# "O maior peso existe, ele só não anula o outro": positivo e negativo são
# somados separados, o lado mais forte vence com o próprio valor -- nunca
# uma diferença/média escondendo o outro lado.
# ────────────────────────────────────────────────────────────────────────
ARQUIVO_REGISTRO_EMOCIONAL = PROJ / "motor_emocional_registro.json"
PESOS_TEMP_ENTRADA = {"TEXE": 15, "FADEN": 15, "TEFE": 15, "RE": 15, "CE": 40}
PESOS_TEMP_SAIDA = {"TEDSA": 15, "FADES": 15, "TEFSA": 15, "RS": 15, "CS": 40}
FAMS_TEXTO_TEMP_ENTRADA = ("TEXE", "FADEN", "TEFE")
FAMS_TEXTO_TEMP_SAIDA = ("TEDSA", "FADES", "TEFSA")
FAM_REACAO_TEMP = {"entrada": "RE", "saida": "RS"}
ESTADOS_TEMP = ["positivo", "negativo", "neutro"]
SINAL_TEMP = {"positivo": 1, "negativo": -1, "neutro": 0}
COR_TEMP = {"positivo": "#4fc9a8", "negativo": "#e2607a", "neutro": "#8a8fa3"}
APLICAR_BONUS_FRASE_REACAO = True
BONUS_NEUTRO_COM_REACAO = 5

# Faixas confirmadas pela Thaís uma a uma, nunca inferidas por espelhamento:
#   16% a 85%    → Morno/Equilibrado/Estável ("85% é o final de morno")
#   86% a 100%   → Fervendo ("86% é fervendo")
#   -85% a 15%   → Esfriando ("15% pra baixo é esfriando")
#   -86% a -100% → Gelo ("-86% é GELO") -- só alcançável com negativo real
ZONAS_TEMP = [
    (86, "🔥 Fervendo", "#e2607a"),
    (16, "🌤️ Morno / Equilibrado / Estável", "#4fc9a8"),
    (-85, "🌫️ Esfriando", "#7c93f5"),
    (-101, "🧊 Gelo", "#2f3fa0"),
]


def classificar_zona_temp(temperatura: float) -> Tuple[str, str]:
    for minimo, nome, cor in ZONAS_TEMP:
        if temperatura >= minimo:
            return nome, cor
    return ZONAS_TEMP[-1][1], ZONAS_TEMP[-1][2]


def carregar_registro_emocional() -> Dict[str, str]:
    if ARQUIVO_REGISTRO_EMOCIONAL.exists():
        return json.loads(ARQUIVO_REGISTRO_EMOCIONAL.read_text(encoding="utf-8"))
    return {}


def salvar_registro_emocional(registro: Dict[str, str]) -> None:
    if not st.session_state.get("backup_registro_emocional_feito") and ARQUIVO_REGISTRO_EMOCIONAL.exists():
        carimbo = __import__("time").strftime("%Y%m%d_%H%M%S")
        ARQUIVO_REGISTRO_EMOCIONAL.with_name(f"motor_emocional_registro.BACKUP_{carimbo}.json").write_text(
            ARQUIVO_REGISTRO_EMOCIONAL.read_text(encoding="utf-8"), encoding="utf-8")
        st.session_state["backup_registro_emocional_feito"] = True
    tmp = ARQUIVO_REGISTRO_EMOCIONAL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_REGISTRO_EMOCIONAL)


def chave_temp(rotulo: str, valor: str) -> str:
    return f"{rotulo}:{valor.strip()}"


def valores_do_lado_temp(bloco: dict, lado: str) -> Dict[str, str]:
    if lado == "entrada":
        e = bloco["entrada"]
        texe, faden, tefe = dividir_travessao(e.get("texto", ""))
        return {"TEXE": texe, "FADEN": faden, "TEFE": tefe, "RE": e.get("reacao", ""), "CE": e.get("contexto", "")}
    s = bloco["saidas"][0]
    texto_saida = (s.get("textos") or [""])[0]
    tedsa, fades, tefsa = dividir_travessao(texto_saida)
    return {"TEDSA": tedsa, "FADES": fades, "TEFSA": tefsa, "RS": s.get("reacao", ""), "CS": s.get("contexto", "")}


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


aba_chat, aba_cerebro, aba_temperatura = st.tabs(["💬 Conversa", "🧠 Cérebro (treino e mapa)", "🌡️ Temperatura emocional"])

with aba_chat:
    st.caption("Ele responde com o que aprendeu do bloco mais parecido com o que você disse. Contexto fica por dentro, nunca aparece na fala. "
               "Quando ele não sabe, ele diz -- e você pode ensinar ali mesmo.")
    if "chat_msgs" not in st.session_state:
        st.session_state["chat_msgs"] = []
    if not st.session_state.get("lembrete_checado"):
        st.session_state["lembrete_checado"] = True
        lembrete = lembrete_do_dia()
        if lembrete:
            st.session_state["chat_msgs"].append({"role": "assistant", **lembrete})
    with st.spinner("Preparando o cérebro (só na primeira vez, ou quando o corpus muda -- pode levar alguns minutos)..."):
        modelo_atual()
    for m in st.session_state["chat_msgs"]:
        with st.chat_message(m["role"]):
            st.markdown(m["texto"])
            if m.get("meta"):
                st.caption(m["meta"])
            if m.get("interno"):
                st.caption("🔒 Interno (contexto) -- só pra calcular intenção: " + m["interno"])
    if st.session_state.get("ensino_pendente"):
        st.info("🌱 Modo ensino: a sua próxima mensagem vai ser a resposta que o Adam aprende.")
        if st.button("Cancelar ensino"):
            st.session_state["ensino_pendente"] = None
            st.rerun()
    if st.session_state.get("completar_pendente"):
        st.info("🧩 Completando um bloco: a sua próxima mensagem vai preencher o campo que faltava.")
        if st.button("Cancelar"):
            st.session_state["completar_pendente"] = None
            st.rerun()
    prompt = st.chat_input("Fala com o Adam...")
    if prompt:
        st.session_state["chat_msgs"].append({"role": "user", "texto": prompt})
        r = tratar_mensagem(prompt)
        ensino = r.pop("ensino", None)
        st.session_state["chat_msgs"].append({"role": "assistant", **r})
        if ensino:
            st.session_state["ensino_pendente"] = ensino
        st.rerun()

with aba_cerebro:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("A sequência única de um bloco (X=Y, E=S)")
        bloco_escolhido = st.selectbox("Bloco", [b["bloco_id"] for b in blocos])
        idx_escolhido = [b["bloco_id"] for b in blocos].index(bloco_escolhido)
        seq = sequencias[idx_escolhido]
        for t in seq:
            cor = "#e8935a" if t.rotulo in ("TEXE", "FADEN", "TEFE") else \
                  "#e2607a" if t.rotulo == "RE" else \
                  "#4fc9a8" if t.rotulo == "CE" else \
                  "#a78bfa" if t.rotulo == "PIDE" else \
                  "#7c93f5" if t.rotulo in ("TEDSA", "FADES", "TEFSA") else \
                  "#f0d64f" if t.rotulo == "RS" else "#4fc9a8"
            st.markdown(
                f"<span style='background:{cor}22;border:1px solid {cor};border-radius:6px;"
                f"padding:2px 6px;margin:2px;display:inline-block;font-family:monospace'>"
                f"{t.palavra} <span style='opacity:.6;font-size:10px'>{t.rotulo}·0.{t.marker_idx}</span></span>",
                unsafe_allow_html=True,
            )
        st.caption(f"{len(seq)} tokens nessa sequência, vocabulário total: {len(vocab)} formas.")

    with col2:
        st.subheader("Treino")
        epocas = st.slider("Épocas", 10, 2500, 2000, step=10)
        if st.button("Retreinar o cérebro"):
            model = CerebroAdam(vocab_size=len(vocab), num_blocos_total=len(blocos))
            with st.spinner("Treinando..."):
                historico = treinar(model, tensores, epocas)
            model.eval()
            st.session_state["model_cerebro"] = model
            st.line_chart(historico)
            st.success(f"Loss final: {historico[-1]:.4f}")

    st.divider()
    st.subheader("Testar continuação (dar a entrada, ver o que ele gera pra saída)")
    bloco_teste = st.selectbox("Bloco pra testar", [b["bloco_id"] for b in blocos], key="teste")
    idx_teste = [b["bloco_id"] for b in blocos].index(bloco_teste)
    seq_completa = sequencias[idx_teste]
    corte = next((i for i, t in enumerate(seq_completa) if t.rotulo in ROTULOS_SAIDA), len(seq_completa))
    seq_entrada = seq_completa[:corte]
    seq_saida_real = seq_completa[corte:]
    if st.button("Gerar continuação"):
        gerado = gerar_continuacao(modelo_atual(), seq_entrada, seq_saida_real, vocab, idx_para_palavra, mapa_bloco, bloco_teste)
        pares = list(zip(seq_saida_real, gerado))
        palavras_visiveis = [pal for tok, pal in pares if tok.rotulo in ROTULOS_VISIVEIS_AO_USUARIO]
        palavras_internas = [(tok.rotulo, pal) for tok, pal in pares if tok.rotulo not in ROTULOS_VISIVEIS_AO_USUARIO]
        st.write("**Entrada dada:**", " ".join(t.palavra for t in seq_entrada))
        st.write("**Saída real do bloco (o que o usuário veria):**",
                 " ".join(t.palavra for t in seq_saida_real if t.rotulo in ROTULOS_VISIVEIS_AO_USUARIO))
        st.write("**Gerado pelo cérebro (o que o usuário veria):**", " ".join(palavras_visiveis))
        if palavras_internas:
            st.caption("🔒 Interno (contexto) -- nunca exibido, só pra calcular intenção: "
                       + ", ".join(f"{pal} ({rot})" for rot, pal in palavras_internas))

    st.divider()
    st.subheader("🧠 O cérebro simulado -- cada ponto é uma palavra REAL, numa posição REAL")
    st.caption("Cor = bloco. Se a integração de campos estiver funcionando, cada bloco forma seu próprio grupo coeso -- não uma nuvem única misturada.")
    fig = grafico_cerebro_3d(modelo_atual(), sequencias, vocab, mapa_bloco)
    st.plotly_chart(fig, use_container_width=True)

with aba_temperatura:
    st.caption("Temperatura = positivo e negativo somados separados, o lado mais forte vence com o próprio valor. "
               "Cada campo é registrado por você, uma vez, e vale pra sempre -- compartilhado com o motor_emocional.py.")
    if "registro_emocional" not in st.session_state:
        st.session_state["registro_emocional"] = carregar_registro_emocional()
    registro_temp = st.session_state["registro_emocional"]

    for bloco in blocos:
        st.divider()
        st.subheader(f"Bloco {bloco['bloco_id']}")
        col_e, col_s = st.columns(2)
        for lado, col, titulo in (("entrada", col_e, "Entrada"), ("saida", col_s, "Saída")):
            with col:
                st.markdown(f"**{titulo}**")
                r = avaliar_lado_temp(bloco, lado, registro_temp)
                if r["pendentes"]:
                    st.warning(f"⏳ {r['motivo']}")
                else:
                    st.markdown(f"<span style='color:{r['cor_zona']};font-size:1.3em;font-weight:bold'>{r['temperatura']:+d}% -- {r['zona']}</span>", unsafe_allow_html=True)
                    if r["direcao"]:
                        st.caption(r["direcao"])
                    for b in r["bonus"]:
                        sinal_bonus = "+" if b["para"] == "positivo" else "-"
                        st.caption(f"🎁 {b['fam']} era neutro, ganhou {sinal_bonus}{b['pontos']} pra {b['para']} (reação empresta o tom)")

                pontos_conhecidos = [p for p in r["trajetoria"] if p["cumulativo"] is not None]
                if len(pontos_conhecidos) >= 2:
                    st.caption("Trajetória (" + " → ".join(f"{p['fam']} {p['cumulativo']:+d}%" for p in pontos_conhecidos) + ")")
                    st.line_chart({"temperatura": [p["cumulativo"] for p in pontos_conhecidos]}, height=120)

                for item in r["detalhe"]:
                    if item["estado"] == "vazio":
                        st.caption(f"· {item['fam']} ({item['peso']}%): vazio")
                        continue
                    if item["estado"] == "pendente":
                        st.markdown(f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): \"{item['valor']}\" -- <span style='color:#8a8fa3'>?</span></span>", unsafe_allow_html=True)
                        continue
                    cor = COR_TEMP.get(item["estado"], "#8a8fa3")
                    st.markdown(
                        f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): "
                        f"<span style='color:{cor}'>\"{item['valor']}\" -- {item['estado']} ({item['contribuicao']:+d})</span></span>",
                        unsafe_allow_html=True,
                    )

                for fam, valor in r["pendentes"]:
                    escolha = st.radio(f"{fam}: \"{valor}\" é...", ESTADOS_TEMP, index=None, horizontal=True, key=f"temp_{bloco['bloco_id']}_{lado}_{fam}")
                    if escolha:
                        registro_temp[chave_temp(fam, valor)] = escolha
                        salvar_registro_emocional(registro_temp)
                        st.rerun()
