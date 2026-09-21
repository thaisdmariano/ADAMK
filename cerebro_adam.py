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
ESCALA_BLOCO = 4.0       # bloco domina o espaço -- cada bloco no seu espacinho


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

    # Entrada: TEXE/FADEN/TEFE (texto, dividido por travessão) + RE + CE + PIDE
    texe_txt, faden_txt, tefe_txt = dividir_travessao(entrada.get("texto", ""))
    partes_texto_e = [("TEXE", texe_txt, entrada["tokens"].get("TEXE", [])),
                       ("FADEN", faden_txt, entrada["tokens"].get("FADEN", [])),
                       ("TEFE", tefe_txt, entrada["tokens"].get("TEFE", []))]
    for rotulo, texto, marcadores in partes_texto_e:
        for palavra, marcador in zip(tokenizar(texto), marcadores):
            seq.append(TokenDoBloco(palavra, rotulo, filho(marcador), bloco_id))
    reacao_e = entrada.get("reacao", "")
    for marcador in entrada["tokens"].get("RE", []):
        seq.append(TokenDoBloco(reacao_e, "RE", filho(marcador), bloco_id))
    for palavra, marcador in zip(tokenizar(entrada.get("contexto", "")), entrada["tokens"].get("CE", [])):
        seq.append(TokenDoBloco(palavra, "CE", filho(marcador), bloco_id))
    for palavra, marcador in zip(tokenizar(entrada.get("pensamento_interno", "")), entrada["tokens"].get("PIDE", [])):
        seq.append(TokenDoBloco(palavra, "PIDE", filho(marcador), bloco_id))

    # Saída: TEDSA/FADES/TEFSA + RS + CS -- MESMA sequência, continuando.
    texto_saida = (saida.get("textos") or [""])[0]
    tedsa_txt, fades_txt, tefsa_txt = dividir_travessao(texto_saida)
    partes_texto_s = [("TEDSA", tedsa_txt, saida["tokens"].get("TEDSA", [])),
                       ("FADES", fades_txt, saida["tokens"].get("FADES", [])),
                       ("TEFSA", tefsa_txt, saida["tokens"].get("TEFSA", []))]
    for rotulo, texto, marcadores in partes_texto_s:
        for palavra, marcador in zip(tokenizar(texto), marcadores):
            seq.append(TokenDoBloco(palavra, rotulo, filho(marcador), bloco_id))
    reacao_s = saida.get("reacao", "")
    for marcador in saida["tokens"].get("RS", []):
        seq.append(TokenDoBloco(reacao_s, "RS", filho(marcador), bloco_id))
    for palavra, marcador in zip(tokenizar(saida.get("contexto", "")), saida["tokens"].get("CS", [])):
        seq.append(TokenDoBloco(palavra, "CS", filho(marcador), bloco_id))

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


FRIEZA = "0.0"  # 2026-09-21, Thaís: ausência de reação NÃO é "desconhecido" -- significa tom de frieza


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

FRASE_CONHECE_MAS_VAZIO = ("Eu tenho a impressão de que já conheço esse assunto, mas não tenho dados suficientes para afirmar. "
                           "Pode me ensinar mais sobre ele, minha adorada?")
FRASE_NAO_SEI = "Criadora, não sei o que isso significa. Pode me ensinar o que eu devo responder?"


@st.cache_resource(show_spinner=False)
def modelo_treinado() -> CerebroAdam:
    """Treina uma vez (ou carrega o checkpoint salvo, se o corpus não mudou)."""
    assinatura = hashlib.sha256(
        (str(sorted(vocab.items())) + str([len(s) for s in sequencias])).encode("utf-8")
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
def _filho(marcador: str) -> int:
    return int(marcador.split(".", 1)[1])


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


def ensinar_bloco(entrada_texto: str, entrada_reacao: str, resposta_bruta: str) -> int:
    """Grava um bloco novo no universo 0 e devolve o bloco_id. Contexto e
    pensamento interno ficam no ponto neutro 0.0 (a criadora adiciona depois)."""
    saida_texto, saida_reacao = separar_reacao(resposta_bruta)
    memoria_disco = json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))
    universo = memoria_disco["IM"]["0"]
    _backup_memoria_uma_vez()

    # último marcador REAL (nunca confia só no campo ultimo_child)
    ultimo = max([_filho(universo.get("ultimo_child", "0.0"))] +
                 [_filho(b["saidas"][0]["fim"]) for b in universo["blocos"] if b["saidas"][0].get("fim")])
    neutro = "0.0"
    reacao_e = entrada_reacao or neutro
    reacao_s = saida_reacao or neutro
    ctx_e, pens_e, ctx_s = neutro, neutro, neutro

    def gerar(n: int) -> List[str]:
        nonlocal ultimo
        marcas = [f"0.{ultimo + i + 1}" for i in range(n)]
        ultimo += n
        return marcas

    texe, faden, tefe = (tokenizar(p) for p in dividir_travessao(entrada_texto))
    ce, pide = tokenizar(ctx_e), tokenizar(pens_e)[:3]
    m_texe, m_faden, m_tefe = gerar(len(texe)), gerar(len(faden)), gerar(len(tefe))
    m_re, m_ce, m_pide = gerar(1), gerar(len(ce)), gerar(len(pide))
    ent_total = m_texe + m_faden + m_tefe + m_re + m_ce + m_pide

    tedsa, fades, tefsa = (tokenizar(p) for p in dividir_travessao(saida_texto))
    cs = tokenizar(ctx_s)
    m_tedsa, m_fades, m_tefsa = gerar(len(tedsa)), gerar(len(fades)), gerar(len(tefsa))
    m_rs, m_cs = gerar(1), gerar(len(cs))
    sai_total = m_tedsa + m_fades + m_tefsa + m_rs + m_cs

    novo_id = max([b["bloco_id"] for b in universo["blocos"]] + [0]) + 1
    universo["blocos"].append({
        "bloco_id": novo_id,
        "entrada": {
            "texto": entrada_texto, "Multivars_Texto_Entrada": [], "reacao": reacao_e, "Multivars_Reacao_Entrada": [],
            "contexto": ctx_e, "pensamento_interno": pens_e,
            "tokens": {"TEXE": m_texe, "FADEN": m_faden, "TEFE": m_tefe, "RE": m_re, "CE": m_ce, "PIDE": m_pide, "TOTAL": ent_total},
            "fim": ent_total[-1],
        },
        "saidas": [{
            "textos": [saida_texto], "Multivars_Texto_Saida": [], "reacao": reacao_s, "Multivars_Reacao_Saida": [],
            "contexto": ctx_s,  # sem "pensamento_interno" na saída, como nos blocos 1-3; "Confirmado pela criadora" não é contexto nem pensamento válido
            "tokens": {"TEDSA": m_tedsa, "FADES": m_fades, "TEFSA": m_tefsa, "RS": m_rs, "CS": m_cs, "TOTAL": sai_total},
            "fim": sai_total[-1],
        }],
        "meta": {"origem": "ensino_no_chat_do_cerebro"},
    })
    universo["ultimo_child"] = f"0.{ultimo}"
    tmp = ARQUIVO_MEMORIA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memoria_disco, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_MEMORIA)
    # o corpus mudou: esquece o que estava em cache pra reler + retreinar
    carregar_memoria.clear()
    modelo_treinado.clear()
    st.session_state.pop("model_cerebro", None)
    return novo_id


def blocos_vazios() -> List[int]:
    return [i for i, s in enumerate(sequencias) if saida_e_placeholder(s)]


def revisar_dados() -> Dict[str, object]:
    abertura = "Entendo. Então que tal a gente revisar alguns dos meus dados, criadora?"
    vazios = blocos_vazios()
    if not vazios:
        return {"texto": abertura + " Todos os meus blocos já têm resposta. Se quiser, me diga uma frase nova e eu peço pra você me ensinar como responder.", "meta": None, "interno": None}
    ent = blocos[vazios[0]]["entrada"]
    reacao = "" if re.fullmatch(r"\d+\.0", ent.get("reacao", "")) else ent.get("reacao", "")
    return {
        "texto": f'{abertura} Tenho {len(vazios)} bloco(s) com a resposta ainda vazia. Por exemplo: "{ent["texto"]}" {reacao}'.rstrip()
                 + ". O que eu devo responder quando me disserem isso?",
        "meta": None, "interno": None, "ensino": {"entrada": ent["texto"], "reacao": reacao},
    }


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
    vazios = len(blocos_vazios())
    extra = f" Tenho {vazios} bloco(s) com a resposta ainda vazia." if vazios else ""
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
    """Devolve (valor pra comparar, rótulo pra mostrar). Sem reação = frieza
    (0.0) -- exceto quando o texto termina em '!': aí o tom não é frieza e
    fica como indicador de aprendizado (valor '' = desconhecido, não pesa
    contra nenhum bloco)."""
    if not _vazio(reacao):
        return reacao, reacao
    if texto.rstrip().endswith("!"):
        return "", "indicador de aprendizado (!)"
    return FRIEZA, "frieza (sem reação)"


def responder(texto: str) -> Dict[str, object]:
    texto_sem_reacao, reacao = separar_reacao(texto)
    tom_valor, tom_rotulo = tom_da_entrada(texto_sem_reacao, reacao)
    idx, score, reacao_bateu = achar_bloco_parecido(texto_sem_reacao, tom_valor, sequencias)
    ensino = {"entrada": texto_sem_reacao, "reacao": reacao}
    if idx >= 0 and saida_e_placeholder(sequencias[idx]) and score >= 1.0:
        return {"texto": FRASE_CONHECE_MAS_VAZIO, "meta": f"bloco {blocos[idx]['bloco_id']} · resposta vazia", "interno": None, "ensino": ensino}
    # Cada campo da entrada vale 25%; uma entrada crua (texto + reação) chega
    # no máximo a 50% e por isso é SEMPRE incompleta -- não é fonte confiável
    # de criação sozinha.
    if idx >= 0 and score >= 1.0 and reacao_bateu and not campos_faltando(blocos[idx]):
        # contexto 100%: texto e reação exatos num bloco completo -> puxa o bloco
        pct = 25 * int(score >= 1.0) + 25 * int(reacao_bateu)
        fala, interno = falar_do_bloco(idx)
        return {"texto": fala or "...", "meta": f"bloco {blocos[idx]['bloco_id']} · sua entrada {pct}% -- incompleta (texto ✓ · reação ✓) · dado do bloco: completo", "interno": interno}

    # Não é dado exato. Cosseno ACHA candidato (texto parecido), Báskara CONFIRMA.
    achado = encontrar_texto_mais_parecido(texto_sem_reacao, blocos)
    if achado is None:
        return _sem_ideia(ensino)
    i = blocos.index(achado["bloco"])
    contexto = str(blocos[i]["entrada"].get("contexto", "")).strip()
    if _vazio(contexto) or saida_e_placeholder(sequencias[i]):
        return {"texto": FRASE_CONHECE_MAS_VAZIO, "meta": f"bloco {blocos[i]['bloco_id']} · ainda sem contexto ou resposta", "interno": None}
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


def tratar_mensagem(prompt: str) -> Dict[str, object]:
    norm = _normalizar(prompt)
    pendente = st.session_state.get("ensino_pendente")
    if pendente:
        st.session_state["ensino_pendente"] = None
        if any(f in norm for f in FRASES_CANCELAR):
            return {"texto": "Tudo bem, minha adorada. A gente volta nisso quando você quiser.", "meta": None, "interno": None}
        novo_id = ensinar_bloco(pendente["entrada"], pendente["reacao"], prompt)
        return {"texto": "Aprendi! Guardei como um bloco novo. Vou treinar um pouquinho pra fixar -- só um minuto.",
                "meta": f"bloco {novo_id} criado", "interno": None}
    sem = st.session_state.pop("semelhanca_pendente", None)
    if any(f in norm for f in FRASES_SEM_TAREFA):
        return revisar_dados()
    if sem:
        r = tratar_semelhanca(sem, norm)
        if r is not None:
            return r
    return responder(prompt)


aba_chat, aba_cerebro = st.tabs(["💬 Conversa", "🧠 Cérebro (treino e mapa)"])

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
