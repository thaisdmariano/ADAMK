#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA Camada 8 -- Neurônios e Aprendizado.

Diferente do insepa_brinquedo_cerebro.py (que é uma simulação didática com
dados de mentira), este módulo só lê o checkpoint REAL treinado
(`insepa_{dominio}.pt`, salvo pelo próprio lovely_test.py) e o histórico
REAL de perdas (`Adam_Lovely_historico_treino.json`, alimentado a cada
retreino automático que acontece quando você conversa com o Adam).

Não decide nada, não participa do chat -- só lê e desenha em 3D o que já
existe. A projeção pra 3D (via SVD/PCA) é usada SÓ pra caber na tela; nunca
é usada como comparação/registro de dado (isso continua sendo sempre por
igualdade exata, em outro lugar do sistema).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch

ARQUIVO_HISTORICO_TREINO = "Adam_Lovely_historico_treino.json"

# Nome da camada de embedding real (treinável) por campo, e o índice dentro
# da tupla salva por torch.save() em train() (lovely_test.py) onde mora o
# dicionário palavra->índice correspondente -- pra rotular cada ponto com a
# palavra real que ele representa, não um número solto.
CAMPOS_EMBEDDING: Dict[str, Tuple[str, int]] = {
    "Texto (E)": ("em_Eval.weight", 6),
    "Reação (RE)": ("em_REval.weight", 7),
    "Contexto (CE)": ("em_CEval.weight", 8),
    "Pensamento (PIDE)": ("em_PIDEval.weight", 9),
}


def ckpt_path(dominio: str) -> str:
    return f"insepa_{dominio}.pt"


def checkpoint_disponivel(dominio: str) -> bool:
    return os.path.exists(ckpt_path(dominio))


def carregar_checkpoint(dominio: str):
    """Carrega a tupla real salva por train() -- não reconstrói nada, só lê
    os pesos que o Adam usa de verdade pra responder no chat."""
    caminho = ckpt_path(dominio)
    if not os.path.exists(caminho):
        return None
    return torch.load(caminho, map_location="cpu", weights_only=False)


def carregar_historico_treino(caminho: str = ARQUIVO_HISTORICO_TREINO) -> List[dict]:
    if not os.path.exists(caminho):
        return []
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _projetar_3d(matriz: np.ndarray) -> np.ndarray:
    """Projeção linear (via SVD) só pra caber na tela em 3D -- puramente
    visual. Com 1 ou 2 pontos, os eixos que faltam ficam zerados (não dá
    pra extrair 3 componentes de menos de 3 pontos, e tá tudo bem: é
    esperado no início da vida do Adam, com poucas palavras registradas)."""
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
        preenchimento = np.zeros((n, 3 - k))
        projetado = np.hstack([projetado, preenchimento])
    return projetado


def grafico_3d_neuronios(dominio: str, campo_rotulo: str) -> Tuple[Optional[go.Figure], Optional[str]]:
    """3D real dos neurônios treinados (embeddings por valor) de um campo.
    Cada ponto é uma palavra/reação REAL que o Adam já tem registrada nesse
    universo -- não é dado inventado nem de brinquedo."""
    dados = carregar_checkpoint(dominio)
    if dados is None:
        return None, f"Nenhum checkpoint treinado ainda para o universo {dominio} -- converse com o Adam pelo menos uma vez pra gerar um."

    nome_camada, idx_val_to_idx = CAMPOS_EMBEDDING[campo_rotulo]
    state_dict = dados[0]
    val_to_idx = dados[idx_val_to_idx]
    idx_to_val = {v: k for k, v in val_to_idx.items()}

    if nome_camada not in state_dict:
        return None, f"Checkpoint não tem a camada '{nome_camada}' (formato antigo?)."

    peso = state_dict[nome_camada].detach().cpu().numpy()  # (num_valores_reais, EMBED_DIM)
    if peso.shape[0] == 0:
        return None, f"Nenhum valor registrado ainda em {campo_rotulo} neste universo."

    coords = _projetar_3d(peso)
    palavras = [str(idx_to_val.get(i, f"#{i}")) for i in range(peso.shape[0])]

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        text=palavras,
        textposition="top center",
        marker=dict(size=6, color=coords[:, 2], colorscale="Viridis", showscale=False),
        hovertext=palavras,
    )])
    fig.update_layout(
        scene=dict(xaxis_title="componente 1", yaxis_title="componente 2", zaxis_title="componente 3"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=550,
    )
    return fig, None


def grafico_3d_curva_aprendizado(historico: List[dict], dominio: Optional[str] = None) -> Optional[go.Figure]:
    """Curva de aprendizado real em 3D: eixo X é cada rodada de treino
    (cada vez que você conversou com o Adam e ele retreinou), eixo Y é a
    época dentro daquela rodada, eixo Z é a perda (loss) real medida."""
    rodadas = [r for r in historico if dominio is None or r.get("dominio") == dominio]
    if not rodadas:
        return None

    fig = go.Figure()
    for i, rodada in enumerate(rodadas):
        epocas = rodada.get("epocas", [])
        if not epocas:
            continue
        xs = [i] * len(epocas)
        ys = [e["epoch"] for e in epocas]
        zs = [e["val_loss"] for e in epocas]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            name=f"Conversa #{i + 1} ({rodada.get('n_blocos', '?')} blocos)",
            marker=dict(size=3),
            line=dict(width=4),
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Conversa / rodada de retreino",
            yaxis_title="Época dentro da rodada",
            zaxis_title="Perda (loss) real",
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=550,
        legend=dict(orientation="h"),
    )
    return fig
