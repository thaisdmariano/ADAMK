#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List

ARQUIVO_MEMORIA = "adam_memoria.json"
CKPT            = "insepa_xy.pt"

# ────────────────────────────────────────────────────────────────────────────────
# Utilitários
# ────────────────────────────────────────────────────────────────────────────────
def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else txt)

def parse_text_reaction(raw: str, blocos: list[dict]) -> tuple[str, str]:
    """
    Separa texto e reação (sem depender de espaços extras).
    Retorna (texto_com_ponto, reacção_sem_espaco_extra).
    """
    s = raw.strip()
    # extrai todas as reações cadastradas, já sem espaços nas bordas
    reactions = sorted(
        {b["entrada"].get("reacao","").strip() for b in blocos if b.get("entrada",{}).get("reacao")},
        key=len, reverse=True
    )
    for rea in reactions:
        if s.endswith(rea):
            txt = s[: -len(rea)].rstrip()
            return garantir_pontuacao(txt), rea
    # sem reação reconhecida
    return garantir_pontuacao(s), ""

def _saida_tokens_legacy_or_insepa(saida: dict) -> tuple[list[str], list[str], list[str]]:
    t = saida.get("tokens", {})
    if "S" in t or "RS" in t or "CS" in t:
        return t.get("S", []), t.get("RS", []), t.get("CS", [])
    return t.get("E", []), t.get("RE", []), t.get("CE", [])

def xy_from_block_many(b: dict) -> list[tuple[list[float], list[float]]]:
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    X    = Ein + REin + CEin

    pares = []
    saidas = b.get("saidas") or ([b["saida"]] if "saida" in b else [])
    for s in saidas:
        S, RS, CS = _saida_tokens_legacy_or_insepa(s)
        pares.append((X, [float(v) for v in (S + RS + CS)]))
    return pares

# ────────────────────────────────────────────────────────────────────────────────
# Normalização de texto
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
# Dataset e Modelo
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
# Inferência (playlist simples + CB conditional)
# ────────────────────────────────────────────────────────────────────────────────
def _montar_X_do_bloco(b: dict) -> list[float]:
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    return Ein + REin + CEin

def _escolher_saida_por_modelo(model, max_x, max_y, bloco) -> dict:
    X     = _montar_X_do_bloco(bloco)
    X_pad = X + [0.0] * (max_x - len(X))
    with torch.no_grad():
        y_hat = model(torch.tensor([X_pad], dtype=torch.float32))[0].numpy()

    saidas = bloco.get("saidas") or ([bloco["saida"]] if "saida" in bloco else [])
    melhor, best_idx = float("inf"), None
    for i, s in enumerate(saidas):
        S, RS, CS = _saida_tokens_legacy_or_insepa(s)
        Y_pad = [float(v) for v in (S + RS + CS)]
        Y_pad += [0.0] * (max_y - len(Y_pad))
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
        # reação já vem sem espaço extra
        variacoes = [f"{v} {saida['reacao'].strip()}" for v in variacoes]
    return variacoes

def infer(memoria: dict, dominio: str) -> None:
    if not os.path.exists(CKPT):
        print("⚠️ Checkpoint não encontrado. Treinando antes de inferir.")
        train(memoria, dominio)

    state, max_x, max_y = torch.load(CKPT)
    blocos = memoria["maes"].get(dominio, {}).get("blocos", [])
    if not blocos:
        print("❌ Universo não encontrado ou sem blocos.")
        return

    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    print("👋 Digite suas entradas (use '|' para combinar vários blocos, ou 'sair'):")
    while True:
        raw = input("👤 → ").strip()
        if raw.lower() in ("sair", "exit", "quit"):
            print("👋 Até mais!")
            break

        # passo 1: split por '|' e encontrar blocos
        partes = [p.strip() for p in re.split(r'\|+', raw) if p.strip()]
        blocos_encontrados = []
        for parte in partes:
            txt, rea = parse_text_reaction(parte, blocos)
            key = normalize(txt)
            # compara texto normalizado e reação sem espaços
            b = next((b for b in blocos
                      if normalize(b["entrada"]["texto"]) == key
                      and b["entrada"].get("reacao","").strip() == rea.strip()), None)
            if b:
                blocos_encontrados.append(b)

        if not blocos_encontrados:
            print("❌ Nenhum bloco encontrado. Tente novamente.")
            continue

        # passo 2: se >1 bloco → Modo CB
        if len(blocos_encontrados) > 1:
            print("🔗 Modo CB ativado (múltiplos blocos).")
            for bloco in blocos_encontrados:
                saida_sel = _escolher_saida_por_modelo(model, max_x, max_y, bloco)
                if saida_sel:
                    for v in _variacoes_da_saida(saida_sel):
                        print(f"🤖 {v}")
            print("\n--- Fim do combo. Pode digitar outra entrada ou 'sair'.")
            continue

        # passo 3: 1 único bloco → playlist simples
        bloco_atual = blocos_encontrados[0]
        print(f"🎯 Iniciando playlist simples para: "
              f"{bloco_atual['entrada']['texto']} {bloco_atual['entrada'].get('reacao','')}")
        while True:
            saida_sel = _escolher_saida_por_modelo(model, max_x, max_y, bloco_atual)
            if not saida_sel:
                print("⚠️ Bloco atual sem saídas. Voltando ao menu principal.")
                break

            for v in _variacoes_da_saida(saida_sel):
                print(f"\n🤖 {v}")

            nxt = input("(Enter p/ repetir variações | texto p/ mudar bloco | 'voltar' p/ menu) ").strip()
            if not nxt:
                continue
            if nxt.lower() == "voltar":
                break

            txt2, rea2 = parse_text_reaction(nxt, blocos)
            key2 = normalize(txt2)
            bloco_novo = next((b for b in blocos
                               if normalize(b["entrada"]["texto"]) == key2
                               and b["entrada"].get("reacao","").strip() == rea2.strip()), None)
            if bloco_novo:
                bloco_atual = bloco_novo
            else:
                print("❌ Bloco não encontrado. Continuando no atual.")
        # volta ao prompt principal

# ────────────────────────────────────────────────────────────────────────────────
# CLI Multi-Universo
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump({"maes": {}}, f, ensure_ascii=False, indent=2)

    memoria = json.load(open(ARQUIVO_MEMORIA, "r", encoding="utf-8"))

    while True:
        print("\n=== Menu Principal ===")
        print("1) Treinar rede neural")
        print("2) Iniciar inferência")
        print("3) Sair")
        opc = input("Escolha (1/2/3): ").strip()

        if opc == "1":
            dom = input("→ Índice-mãe para treinar: ").strip()
            if dom in memoria["maes"]:
                train(memoria, dom)
            else:
                print(f"❌ Universo '{dom}' não existe.")
        elif opc == "2":
            dom = input("→ Índice-mãe para inferir: ").strip()
            if dom in memoria["maes"]:
                infer(memoria, dom)
            else:
                print(f"❌ Universo '{dom}' não existe.")
        elif opc in ("3", "sair", "exit", "quit"):
            print("👋 Até a próxima!")
            break
        else:
            print("❌ Opção inválida. Tente 1, 2 ou 3.")