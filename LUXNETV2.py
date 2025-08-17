import os
import json
import re
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
    """1) Colapsa múltiplos espaços em um só e remove espaços nas bordas."""
    return re.sub(r'\s+', ' ', txt).strip()

def normalize_separators(txt: str) -> str:
    """
    2) Normaliza vírgula e ponto:
       - remove espaços antes de ',' e '.'
       - garante um espaço após ',' e '.'
    """
    txt = re.sub(r'\s*([.,])\s*', r'\1 ', txt)
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
        if ep == 1 or ep % 10 == 0 or ep == 100:
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

    # 1) escolhe entrada+reação
    raw = input("👤 Entrada + Reação: ")
    txt, rea = parse_text_reaction(raw, blocos)
    key = normalize(txt)

    # 2) localiza bloco inicial com texto NORMALIZADO
    bloco_atual = next(
        (b for b in blocos
         if normalize(b["entrada"]["texto"]) == key
         and b["entrada"].get("reacao", "") == rea),
        None
    )
    if not bloco_atual:
        print("❌ Entrada+reação não cadastrada neste universo.")
        return

    # 3) playlist com escadinha
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
            entrada = input("(Enter p/ próxima | texto p/ outro bloco) ")
            if entrada.strip():
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

