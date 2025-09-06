#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
O INSEPA: É um sistema de tokenização que permite que a leitura dos dados seja efetuada pela máquina.
Tal como um humano leria um livro. ACEITA: Pontuação, emojis, stopwords e tudo que outros sistemas rejeitam.
DELIMITA: todas as palavras com marcadores únicos baseados no índice mãe, que permitem variabilidade de dados sem cair em ambiguidades. [Índice Mãe 1, filhos 1.1, 1.2,1.3]
DIVIDE: Cada universo é treinado com base na mãe, portanto os dados jamais se generalizam (o quê é motivo de orgulho e não falha) [Índice Mãe 1 ≠ Índice Mãe 2]
ORGANIZA: os dados de índices filhos por blocos inseparizados que se dividem em:
  Entrada=Texto +reação+contexto+pensamento interno
  Saída=Multiplicidade de textos+reação+contexto+explicação+imersão
DISPARA RESPOSTAS COM BASE NOS MARCADORES ÚNICOS: Se o Total da Entrada X (E+RE+CE+PIDE) é:
  ["1.1", "1.2", ..., "1.9"]
ele sempre dispara o Total da entrada Y (S+RS+CS+EXDS+IME)
  ["1.10", ..., "1.37"]
PERMITE QUE A MÁQUINA FAÇA A DIFERENCIAÇÃO DE DADOS PRÉVIOS E OS RESULTADOS DELES:
  Texto+reação+contexto de entrada e de saída são 100% tokenizados,
  mas os campos de Pensamento Interno da entrada, e Explicação e Imersão da saída recebem apenas 3 marcadores únicos.
  [Texto= Oi. E: 0.1, 02 ≠ Explicação= Olha que legal. EXDS: 0.3, 0.4, 0.5]
MANTÉM A COERÊNCIA DE VARIABILIDADE: Cada Bloco possui um ID, cada ID pode ser combinado para gerar um Conjunto de Blocos que segue uma determinada sequência, e caso a sequência possua uma harmonia semântica ela recebe o selo de validação aceita, por meio de um sistema booleano.
  [B1+B2= CB > SDB > VA (True ou False)]
NÃO É: Feito com embbedings ou estatísticas globais.
FUNÇÃO:
  1. Evita a generalização de universos.
  2. Permite variabilidade de respostas mesmo sendo determinístico.
  3. Aumenta a segurança quanto aos dados que serão exibidos.
  4. Garante a integridade da rede neural e consequentemente da mente da IA.
"""

import os
import json
import re
import random
from itertools import product
from typing import Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ARQUIVO_MEMORIA = "adam_memoria.json"
CKPT            = "insepa_xy.pt"


# ────────────────────────────────────────────────────────────────────────────────
# Utilitários apenas faça com PIDE, EXDS E IME sejam calculados
# ────────────────────────────────────────────────────────────────────────────────

def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else txt)


def tokenizar(txt: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", txt, re.UNICODE)


def parse_text_reaction(raw: str, blocos: list[dict]) -> tuple[str, str]:
    s = raw.strip()
    reactions = sorted(
        {b["entrada"]["reacao"] for b in blocos
         if b.get("entrada", {}).get("reacao")},
        key=len, reverse=True
    )
    for rea in reactions:
        if s.endswith(rea):
            txt = s[:-len(rea)].rstrip()
            return garantir_pontuacao(txt), rea
    return garantir_pontuacao(s), ""


def _saida_tokens_legacy_or_insepa(saida: dict) -> tuple[list[str], list[str], list[str]]:
    t = saida.get("tokens", {})
    if "S" in t or "RS" in t or "CS" in t:
        return t.get("S", []), t.get("RS", []), t.get("CS", [])
    return t.get("E", []), t.get("RE", []), t.get("CE", [])


def xy_from_block_many(b: dict) -> list[tuple[list[float], list[float]]]:
    # Entrada agora inclui também tokens de PIDE
    t_in   = b["entrada"]["tokens"]
    Ein    = [float(v) for v in t_in.get("E", [])]
    REin   = [float(v) for v in t_in.get("RE", [])]
    CEin   = [float(v) for v in t_in.get("CE", [])]
    PIDEin = [float(v) for v in t_in.get("PIDE", [])]
    X      = Ein + REin + CEin + PIDEin

    pares = []
    # Saídas agora incluem também tokens EXDS e IME
    for saida in b.get("saidas") or ([b.get("saida")] if b.get("saida") else []):
        t_out = saida.get("tokens", {})
        S     = [float(v) for v in t_out.get("S", [])]
        RS    = [float(v) for v in t_out.get("RS", [])]
        CS    = [float(v) for v in t_out.get("CS", [])]
        EXDS  = [float(v) for v in t_out.get("EXDS", [])]
        IME   = [float(v) for v in t_out.get("IME", [])]
        Y     = S + RS + CS + EXDS + IME
        pares.append((X, Y))

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
# Dataset e modelo (usando Índice-Mãe 'IM' em vez de 'maes')
# ────────────────────────────────────────────────────────────────────────────────

class InsepaXY(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        # Agora busca blocos em memoria["IM"] (maiúsculo)
        blocos = memoria["IM"][dominio]["blocos"]
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
        return (
            torch.tensor(x_pad, dtype=torch.float32),
            torch.tensor(y_pad, dtype=torch.float32)
        )


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
# Inferência interativa PIDE, EXDS E IME agora calculados
# ────────────────────────────────────────────────────────────────────────────────

def _montar_X_do_bloco(b: dict) -> list[float]:
    t_in   = b["entrada"]["tokens"]
    Ein    = [float(v) for v in t_in.get("E", [])]
    REin   = [float(v) for v in t_in.get("RE", [])]
    CEin   = [float(v) for v in t_in.get("CE", [])]
    PIDEin = [float(v) for v in t_in.get("PIDE", [])]
    return Ein + REin + CEin + PIDEin

def _montar_Y_da_saida(saida: dict) -> list[float]:
    t_out = saida.get("tokens", {})
    S     = [float(v) for v in t_out.get("S", [])]
    RS    = [float(v) for v in t_out.get("RS", [])]
    CS    = [float(v) for v in t_out.get("CS", [])]
    EXDS  = [float(v) for v in t_out.get("EXDS", [])]
    IME   = [float(v) for v in t_out.get("IME", [])]
    return S + RS + CS + EXDS + IME

def _escolher_saida_por_modelo(model, max_x, max_y, bloco) -> dict:
    X     = _montar_X_do_bloco(bloco)
    X_pad = X + [0.0] * (max_x - len(X))
    with torch.no_grad():
        y_hat = model(torch.tensor([X_pad], dtype=torch.float32))[0].numpy()

    saidas = bloco.get("saidas") or ([bloco.get("saida")] if bloco.get("saida") else [])
    if not saidas:
        return None

    melhor, best_idx = float("inf"), None
    for i, s in enumerate(saidas):
        Y     = _montar_Y_da_saida(s)
        Y_pad = Y + [0.0] * (max_y - len(Y))
        dist  = sum((yh - yr) ** 2 for yh, yr in zip(y_hat, Y_pad))
        if dist < melhor:
            melhor, best_idx = dist, i

    return saidas[best_idx]

def _variacoes_da_saida(saida: dict) -> list[str]:
    """
    Gera variações de saída exibindo apenas S+RS.
    Oculta qualquer string idêntica ao contexto (CS) e mantém a reação.
    """
    if "textos" in saida and saida["textos"]:
        variacoes = saida["textos"][:]
    elif "texto" in saida and saida["texto"].strip():
        variacoes = [saida["texto"]]
    else:
        variacoes = ["[Sem texto registrado nesta saída]"]

    ctx = (saida.get("contexto") or "").strip()
    if ctx:
        variacoes = [v for v in variacoes if normalize(v) != normalize(ctx)]

    rea = (saida.get("reacao") or "").strip()
    if rea:
        variacoes = [f"{v} {rea}" for v in variacoes]

    return variacoes

def infer(memoria: dict, dominio: str) -> None:
    if not os.path.exists(CKPT):
        print("⚠️ Checkpoint não encontrado. Treinando antes de inferir.")
        train(memoria, dominio)

    state, max_x, max_y = torch.load(CKPT)
    blocos = memoria["IM"].get(dominio, {}).get("blocos", [])
    if not blocos:
        print("❌ Universo não encontrado ou sem blocos.")
        return

    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    # loop principal: repete até usuário digitar 'sair'
    while True:
        raw = input("👤 Entrada + Reação ('sair' p/ encerrar): ").strip()
        if raw.lower() == "sair":
            print("Inferência encerrada pelo usuário.")
            return

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

        # gera variações em loop infinito até trocar de bloco ou digitar 'sair'
        while True:
            saida_sel = _escolher_saida_por_modelo(model, max_x, max_y, bloco_atual)
            if not saida_sel:
                print("⚠️ Bloco atual sem saídas.")
                return

            variacoes = _variacoes_da_saida(saida_sel)
            idx = 0

            while True:
                print(f"\n🤖 {variacoes[idx]}")
                entrada = input("(Enter p/ próxima | texto p/ outro bloco | 'sair' p/ encerrar) ").strip()

                if entrada.lower() == "sair":
                    print("Inferência encerrada pelo usuário.")
                    return

                if entrada == "":
                    # wrap-around: continua indefinidamente
                    idx = (idx + 1) % len(variacoes)
                    continue

                # tentativa de mudar de bloco
                novo_txt, novo_rea = parse_text_reaction(entrada, blocos)
                bloco_novo = next(
                    (b for b in blocos
                     if normalize(b["entrada"]["texto"]) == normalize(novo_txt)
                     and b["entrada"].get("reacao", "") == novo_rea),
                    None
                )
                if bloco_novo:
                    bloco_atual = bloco_novo
                    break

                print("❌ Não achei esse bloco. Continuo no atual.")
            # fim do loop de variações
        # fim do loop de bloco atual
    # fim do loop principal
# ────────────────────────────────────────────────────────────────────────────────
# Construção de parágrafos com contexto/CS oculto via SDB (não mais CBCS)
# ────────────────────────────────────────────────────────────────────────────────

def _is_hidden_block(bloco: dict) -> bool:
    """
    Identifica blocos cujo texto e reação devem ficar ocultos:
      - exibir == False ou hide == True
      - tipo em {'contexto','context','ctx','consciente','cs'}
      - presença de chaves context/CONTEXT ou CS truthy
    """
    if bloco.get("exibir") is False or bloco.get("hide") is True:
        return True

    tipo = str(bloco.get("tipo", "")).strip().lower()
    if tipo in {"contexto", "context", "ctx", "consciente", "cs"}:
        return True

    for k in ("context", "CONTEXT", "CS"):
        if k in bloco and bool(bloco.get(k)):
            return True

    return False


def build_paragraphs_with_emojis(
    memoria: dict,
    dominio: str,
    randomize: bool = False
) -> dict[tuple, list[str]]:
    """
    Gera combinações de parágrafos para cada sequência em SDB:
      - considera CB > Status > SDB > VA > Validation
      - esconde texto/emoji em blocos de contexto/CS
      - filtra variações idênticas ao contexto
      - concatena textos visíveis e lista emojis no final
    """
    universo   = memoria["IM"][dominio]
    blocos     = universo["blocos"]

    cb_info    = universo.get("CB", {})
    # só segue se Status == "Disponível" e Validation == True
    if cb_info.get("Status", "").lower() != "disponível":
        return {}
    if not cb_info.get("Validation", True):
        return {}

    sequences  = cb_info.get("SDB", [])
    resultados = {}

    for seq in sequences:
        opcoes_por_bloco = []
        valido = True

        for bid in seq:
            bloco = next(
                (b for b in blocos if b.get("bloco_id") == bid and b.get("open")),
                None
            )
            if not bloco:
                valido = False
                break

            saida = bloco.get("saidas", [bloco.get("saida")])[0]
            if not saida:
                valido = False
                break

            textos       = saida.get("textos") or ([saida["texto"]] if saida.get("texto") else [])
            reacao       = (saida.get("reacao") or "").strip()
            contexto_txt = (saida.get("contexto") or "").strip()

            if contexto_txt:
                textos = [t for t in textos if normalize(t) != normalize(contexto_txt)]

            if randomize:
                random.shuffle(textos)

            if _is_hidden_block(bloco):
                opcoes_bloco = [("", "")]
            else:
                opcoes_bloco = [(t, reacao) for t in textos]

            if not opcoes_bloco:
                valido = False
                break

            opcoes_por_bloco.append(opcoes_bloco)

        if not valido or not opcoes_por_bloco:
            continue

        combinacoes = list(product(*opcoes_por_bloco))
        if randomize:
            random.shuffle(combinacoes)

        paragrafos = []
        for combo in combinacoes:
            partes  = [t for t, _ in combo if t.strip()]
            reacoes = [r for _, r in combo if r.strip()]
            texto   = " ".join(partes).strip()
            if reacoes:
                texto += f"\nEmojis: {''.join(reacoes)}"
            paragrafos.append(texto)

        resultados[tuple(seq)] = paragrafos

    return resultados


def criar_paragrafos_cli(memoria: dict):
    dom = input("→ Índice-mãe p/ gerar parágrafos (ou 'sair'): ").strip()
    if dom.lower() == "sair":
        return
    if dom not in memoria["IM"]:
        print(f"⚠️ Universo '{dom}' não existe.")
        return

    parags_por_seq = build_paragraphs_with_emojis(memoria, dom, randomize=False)
    for seq, paragrafos in parags_por_seq.items():
        for i, texto in enumerate(paragrafos, 1):
            print(f"\n► Variação {i}:\n{texto}\n")


# ────────────────────────────────────────────────────────────────────────────────
# CLI principal (agora usando IM em vez de 'maes', e SDB em vez de CBCS)
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump({"IM": {}}, f, ensure_ascii=False, indent=2)

    memoria = json.load(open(ARQUIVO_MEMORIA, "r", encoding="utf-8"))

    while True:
        print("\n=== Menu Principal ===")
        print("1) Treinar rede neural")
        print("2) Inferir com rede neural")
        print("3) Gerar parágrafos SDB")
        print("4) Sair do programa")
        opc = input("Escolha uma opção (1/2/3/4): ").strip()

        if opc == "1":
            while True:
                dom = input("→ Índice-mãe p/ treinar (ou 'sair' p/ voltar): ").strip()
                if dom.lower() == "sair":
                    break
                if dom not in memoria["IM"]:
                    print(f"⚠️ Universo '{dom}' não existe.")
                    continue
                train(memoria, dom)

        elif opc == "2":
            while True:
                dom = input("→ Índice-mãe p/ inferir (ou 'sair' p/ voltar): ").strip()
                if dom.lower() == "sair":
                    break
                if dom not in memoria["IM"]:
                    print(f"⚠️ Universo '{dom}' não existe.")
                    continue
                infer(memoria, dom)
                break

        elif opc == "3":
            criar_paragrafos_cli(memoria)

        elif opc in ("4", "sair", "exit", "quit"):
            print("👋 Até mais!")
            break

        else:
            print("❌ Opção inválida. Tente 1, 2, 3 ou 4.")



