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
# Carregar corpus real (só leitura -- este protótipo nunca escreve de volta
# no arquivo real).
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

memoria = carregar_memoria()
blocos = memoria["IM"]["0"]["blocos"]

sequencias = [sequencia_do_bloco(b) for b in blocos]
vocab = montar_vocabulario(sequencias)
forma_para_palavra = montar_forma_para_palavra(sequencias)
idx_para_palavra = {i: forma_para_palavra.get(f, f) for f, i in vocab.items()}
mapa_bloco = {b["bloco_id"]: i for i, b in enumerate(blocos)}
tensores = [montar_tensores(seq, vocab, mapa_bloco) for seq in sequencias]

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
    epocas = st.slider("Épocas", 10, 500, 200, step=10)
    if st.button("Treinar o cérebro"):
        model = CerebroAdam(vocab_size=len(vocab), num_blocos_total=len(blocos))
        with st.spinner("Treinando..."):
            historico = treinar(model, tensores, epocas)
        st.session_state["model_cerebro"] = model
        st.line_chart(historico)
        st.success(f"Loss final: {historico[-1]:.4f}")

if "model_cerebro" in st.session_state:
    st.divider()
    st.subheader("Testar continuação (dar a entrada, ver o que ele gera pra saída)")
    bloco_teste = st.selectbox("Bloco pra testar", [b["bloco_id"] for b in blocos], key="teste")
    idx_teste = [b["bloco_id"] for b in blocos].index(bloco_teste)
    seq_completa = sequencias[idx_teste]
    # separa só a parte de ENTRADA como prompt (rótulos que terminam antes de TEDSA)
    corte = next((i for i, t in enumerate(seq_completa) if t.rotulo in ROTULOS_SAIDA), len(seq_completa))
    seq_entrada = seq_completa[:corte]
    seq_saida_real = seq_completa[corte:]
    if st.button("Gerar continuação"):
        gerado = gerar_continuacao(st.session_state["model_cerebro"], seq_entrada, seq_saida_real, vocab, idx_para_palavra, mapa_bloco, bloco_teste)
        # Contexto (CS) nunca aparece pro usuário -- é insumo interno pra
        # calcular intenção (e, no futuro, sentimento). Separa o que é
        # FALA de verdade (texto + reação) do que é cálculo interno.
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
    fig = grafico_cerebro_3d(st.session_state["model_cerebro"], sequencias, vocab, mapa_bloco)
    st.plotly_chart(fig, use_container_width=True)
