#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAMK – Chatbot Insepa
Script completo com normalização de pontuação ajustada apenas para
os Separadores Genéricos (vírgula, ponto, ponto-e-vírgula e dois-pontos),
treino em PyTorch e inferência com opção de listar todos os parágrafos CBCS.
"""

import os
import json
import re
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List

ARQUIVO_MEMORIA = "adam_memoria.json"
CKPT             = "insepa_xy.pt"

# ────────────────────────────────────────────────────────────────────────────────
# Utilitários
# ────────────────────────────────────────────────────────────────────────────────
def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else txt)

def tokenizar(txt: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", txt, re.UNICODE)

def parse_text_reaction(raw: str, blocos: list[dict]) -> tuple[str, str]:
    """
    Separa texto e reação EXATAMENTE como estão no JSON.
    Tenta cada reação cadastrada (ordem decrescente de tamanho).
    """
    s = raw.strip()
    reactions = sorted(
        {b["entrada"]["reacao"] for b in blocos if b.get("entrada", {}).get("reacao")},
        key=len, reverse=True
    )
    for rea in reactions:
        if s.endswith(rea):
            txt = s[:-len(rea)].rstrip()
            return garantir_pontuacao(txt), rea
    return garantir_pontuacao(s), ""

def _saida_tokens_legacy_or_insepa(saida: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Compatibilidade legado (E/RE/CE) e atual (S/RS/CS).
    Retorna sempre (S, RS, CS).
    """
    t = saida.get("tokens", {})
    if "S" in t or "RS" in t or "CS" in t:
        return t.get("S", []), t.get("RS", []), t.get("CS", [])
    else:
        return t.get("E", []), t.get("RE", []), t.get("CE", [])

def xy_from_block_many(b: dict) -> list[tuple[list[float], list[float]]]:
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    X    = Ein + REin + CEin

    pares = []
    if "saidas" in b and b["saidas"]:
        for saida in b["saidas"]:
            S, RS, CS = _saida_tokens_legacy_or_insepa(saida)
            pares.append((X, [float(v) for v in (S + RS + CS)]))
    elif "saida" in b and b["saida"]:
        S, RS, CS = _saida_tokens_legacy_or_insepa(b["saida"])
        pares.append((X, [float(v) for v in (S + RS + CS)]))
    return pares

# ────────────────────────────────────────────────────────────────────────────────
# Normalização de texto (pipeline enxuto)
# ────────────────────────────────────────────────────────────────────────────────
Normalizers = List[Callable[[str], str]]

def normalize_collapse_spaces(txt: str) -> str:
    return re.sub(r'\s+', ' ', txt).strip()

def normalize_separators(txt: str) -> str:
    txt = re.sub(r'\s*([,.;:])\s*', r'\1 ', txt)
    return txt.strip()

NORMALIZE_PIPELINE: Normalizers = [
    normalize_collapse_spaces,
    normalize_separators,
]

def normalize(txt: str) -> str:
    for fn in NORMALIZE_PIPELINE:
        txt = fn(txt)
    return txt

# ────────────────────────────────────────────────────────────────────────────────
# Dataset e modelo
# ────────────────────────────────────────────────────────────────────────────────
class InsepaXY(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["maes"][dominio]["blocos"]
        self.pares = []
        for b in blocos:
            self.pares.extend(xy_from_block_many(b))
        if not self.pares:
            raise ValueError("Nenhum par (X,Y) encontrado.")
        self.max_x = max(len(x) for x, _ in self.pares)
        self.max_y = max(len(y) for _, y in self.pares)

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int):
        x, y = self.pares[idx]
        x_pad = x + [0.0] * (self.max_x - len(x))
        y_pad = y + [0.0] * (self.max_y - len(y))
        return torch.tensor(x_pad, dtype=torch.float32), torch.tensor(y_pad, dtype=torch.float32)

class InsepaReg(nn.Module):
    def __init__(self, xin: int, yout: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, yout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ────────────────────────────────────────────────────────────────────────────────
# Treino
# ────────────────────────────────────────────────────────────────────────────────
def train(memoria: dict, dominio: str) -> None:
    torch.manual_seed(42)
    ds        = InsepaXY(memoria, dominio)
    loader    = DataLoader(ds, batch_size=2, shuffle=True)
    model     = InsepaReg(ds.max_x, ds.max_y)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    print(f"🚀 Treinando domínio {dominio} ({len(ds)} pares X→Y)...")
    for ep in range(1, 101):
        total_loss = 0.0
        model.train()
        for X, Y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            print(f" Ep {ep:03d}/100  loss={total_loss/len(loader):.4f}")

    torch.save((model.state_dict(), ds.max_x, ds.max_y), CKPT)
    print(f"✅ Treino concluído. Checkpoint salvo em '{CKPT}'\n")

# ────────────────────────────────────────────────────────────────────────────────
# Inferência (playlist interativa com escadinha de blocos)
# ────────────────────────────────────────────────────────────────────────────────
def _montar_X_do_bloco(b: dict) -> list[float]:
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    return Ein + REin + CEin

def _montar_Y_da_saida(saida: dict) -> list[float]:
    S, RS, CS = _saida_tokens_legacy_or_insepa(saida)
    return [float(v) for v in (S + RS + CS)]

def _escolher_saida_por_modelo(model, max_x, max_y, bloco) -> dict:
    X     = _montar_X_do_bloco(bloco)
    X_pad = X + [0.0] * (max_x - len(X))
    with torch.no_grad():
        y_hat = model(torch.tensor([X_pad], dtype=torch.float32))[0].numpy()

    saidas = bloco.get("saidas") or ([bloco["saida"]] if "saida" in bloco else [])
    if not saidas:
        return None

    melhor, best_idx = float("inf"), None
    for i, s in enumerate(saidas):
        Y_pad = _montar_Y_da_saida(s) + [0.0] * (max_y - len(_montar_Y_da_saida(s)))
        dist  = sum((yh - yr) ** 2 for yh, yr in zip(y_hat, Y_pad))
        if dist < melhor:
            melhor, best_idx = dist, i

    return saidas[best_idx]

def _variacoes_da_saida(saida: dict) -> list[str]:
    if "textos" in saida and saida["textos"]:
        variacoes = saida["textos"][:]
    elif "texto" in saida:
        variacoes = [saida["texto"]]
    else:
        variacoes = ["[Sem texto registrado nesta saída]"]
    if saida.get("reacao"):
        variacoes = [f"{v} {saida['reacao']}" for v in variacoes]
    return variacoes

def infer(memoria: dict, dominio: str) -> None:
    if not os.path.exists(CKPT):
        print("⚠️ Checkpoint não encontrado. Treinando antes de inferir.")
        train(memoria, dominio)

    state, max_x, max_y = torch.load(CKPT)
    blocos = memoria["maes"].get(dominio, {}).get("blocos")
    if not blocos:
        print("❌ Universo não encontrado ou sem blocos.")
        return

    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    # ----> Aqui inserimos a opção de parágrafos
    while True:
        raw = input("\n👤 Entrada + Reação (ou 'p' para parágrafos CBCS): ").strip()
        if raw.lower() == "p":
            # gera e exibe todos os parágrafos
            parags = build_paragraphs_with_emojis(memoria, dominio, randomize=True)
            for seq, texto in parags.items():
                print(f"\nSequência {list(seq)}:\n{texto}\n")
            return

        # fluxo normal de inferência
        txt, rea = parse_text_reaction(raw, blocos)
        key = normalize(txt)

        bloco_atual = next(
            (b for b in blocos
             if normalize(b["entrada"]["texto"]) == key
             and b["entrada"].get("reacao", "") == rea),
            None
        )
        if not bloco_atual:
            print("❌ Entrada+reação não cadastrada neste universo.")
            continue

        # percorre variações do bloco atual
        while True:
            saida_sel = _escolher_saida_por_modelo(model, max_x, max_y, bloco_atual)
            if not saida_sel:
                print("⚠️ Bloco atual sem saídas.")
                return

            variacoes = _variacoes_da_saida(saida_sel)
            idx = 0
            while idx < len(variacoes):
                print(f"\n🤖 {variacoes[idx]}")
                idx += 1
                entrada = input("(Enter p/ próxima | texto p/ outro bloco) ").strip()
                if entrada:
                    novo_txt, novo_rea = parse_text_reaction(entrada, blocos)
                    key2 = normalize(novo_txt)
                    bloco_novo = next(
                        (b for b in blocos
                         if normalize(b["entrada"]["texto"]) == key2
                         and b["entrada"].get("reacao", "") == novo_rea),
                        None
                    )
                    if bloco_novo:
                        bloco_atual = bloco_novo
                        break
                    else:
                        print("❌ Não achei esse bloco. Continuo no atual.")
            else:
                print("\n😔 Sem mais variações. Fim da playlist.")
                return

# ────────────────────────────────────────────────────────────────────────────────
# A FUNÇÃO P DO MURO — VERSÃO APRIMORADA
# ────────────────────────────────────────────────────────────────────────────────
def build_paragraphs_with_emojis(memoria: dict, dominio: str, randomize: bool = True) -> dict:
    """
    Lê memoria["maes"][dominio], detecta 'cb'/'cbcs' no topo ou em um bloco
    e gera um parágrafo para cada combinação possível de bloco_id,
    concatenando texto + reação em linhas separadas.
    """
    universo  = memoria["maes"][dominio]
    blocos    = universo["blocos"]

    # detecta cb e cbcs no topo ou dentro de um bloco
    if "cb" in universo:
        cb_info   = universo["cb"]
        bids      = set(cb_info.get("bids", []))
        sequences = universo.get("cbcs", [])
    else:
        bloco_cb  = next((b for b in blocos if "CB" in b), {})
        cb_info   = bloco_cb.get("CB", {})
        bids      = set(cb_info.get("BIDS", []))
        sequences = cb_info.get("CBCS", [])

    resultados = {}
    for seq in sequences:
        # monta lista de listas de variações (texto + reação) para cada bloco na sequência
        listas_por_bloco = []
        for bid in seq:
            if bid not in bids:
                continue
            bloco = next(b for b in blocos if b["bloco_id"] == bid and b.get("open"))
            saida = bloco["saidas"][0]
            textos = saida.get("textos", [])
            reacao = saida.get("reacao", "").strip()
            # cria lista de "texto + <espaço> + reação" (ou só texto, se não houver reação)
            if reacao:
                listas_por_bloco.append([f"{t} {reacao}" for t in textos])
            else:
                listas_por_bloco.append(textos[:])

        # gera o produto cartesiano entre as listas de variações
        paragrafos = []
        for combo in itertools.product(*listas_por_bloco):
            paragrafos.append("\n".join(combo))

        # embaralha, se desejado
        if randomize:
            random.shuffle(paragrafos)

        # junta cada combinação num único texto, separado por 2 linhas em branco
        resultados[tuple(seq)] = "\n\n".join(paragrafos)

    return resultados

# ────────────────────────────────────────────────────────────────────────────────
# CLI multi-universo com menu principal
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump({"maes": {}}, f, ensure_ascii=False, indent=2)

    memoria = json.load(open(ARQUIVO_MEMORIA, "r", encoding="utf-8"))

    while True:
        print("\n=== Menu Principal ===")
        print("1) Treinar rede neural")
        print("2) Inferir com rede neural")
        print("3) Sair do programa")
        opc = input("Escolha uma opção (1/2/3): ").strip()

        if opc == "1":
            while True:
                dom = input("→ Índice-mãe p/ treinar (ou 'sair' p/ voltar): ").strip()
                if dom.lower() == "sair":
                    break
                if dom not in memoria["maes"]:
                    print(f"⚠️ Universo '{dom}' não existe.")
                    continue
                train(memoria, dom)

        elif opc == "2":
            while True:
                dom = input("→ Índice-mãe p/ inferir (ou 'sair' p/ voltar): ").strip()
                if dom.lower() == "sair":
                    break
                if dom not in memoria["maes"]:
                    print(f"⚠️ Universo '{dom}' não existe.")
                    continue
                infer(memoria, dom)
                break

        elif opc in ("3", "sair", "exit", "quit"):
            print("👋 Até mais!")
            break

        else:
            print("❌ Opção inválida. Tente 1, 2 ou 3.")
