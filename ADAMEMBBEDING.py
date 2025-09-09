#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import re as _re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE ARQUIVOS E CONSTANTES
# ────────────────────────────────────────────────────────────────────────────────

ARQUIVO_MEMORIA      = "adam_memoria.json"
ARQUIVO_INCONSCIENTE = "inconsciente.json"
EMBED_DIM            = 16

def ckpt_path(dominio: str) -> str:
    return f"insepa_{dominio}.pt"

# ────────────────────────────────────────────────────────────────────────────────
# I/O JSON
# ────────────────────────────────────────────────────────────────────────────────

def carregar_json(caminho: str, default: dict) -> dict:
    if not os.path.exists(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_json(caminho: str, data: dict) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────────────
# NORMALIZAÇÃO E PARSING DE TEXTO + REAÇÃO (ACEITA EMOJIS)
# ────────────────────────────────────────────────────────────────────────────────

def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else "")

def normalize_collapse_spaces(txt: str) -> str:
    return _re.sub(r'\s+', ' ', txt).strip()

def normalize_separators(txt: str) -> str:
    return _re.sub(r'\s*([,.;:])\s*', r'\1 ', txt).strip()

def normalize(txt: str) -> str:
    for fn in (normalize_collapse_spaces, normalize_separators):
        txt = fn(txt)
    return txt

def parse_text_reaction(raw: str, blocos: List[dict]) -> Tuple[str, str]:
    """
    Retorna (texto, reação), identificando no fim da string
    um dos valores b['entrada']['reacao'], preservando emojis.
    """
    s = raw.strip()
    reactions = sorted(
        {b["entrada"].get("reacao","") for b in blocos},
        key=len, reverse=True
    )
    for reac in reactions:
        if reac and s.endswith(reac):
            txt = s[:-len(reac)].rstrip()
            return garantir_pontuacao(txt), reac
    return garantir_pontuacao(s), ""

# ────────────────────────────────────────────────────────────────────────────────
# MONTAGEM DE VETORES X (entrada) e Y (saída)
# ────────────────────────────────────────────────────────────────────────────────

def _montar_X_do_bloco(b: dict) -> List[float]:
    t    = b["entrada"]["tokens"]
    vals = t.get("E", []) + t.get("RE", []) + t.get("CE", []) + t.get("PIDE", [])
    return [float(v) for v in vals]

def _montar_Y_da_saida(saida: dict) -> List[float]:
    t    = saida.get("tokens", {})
    vals = t.get("S", []) + t.get("RS", []) + t.get("CS", []) + t.get("EXDS", []) + t.get("IME", [])
    return [float(v) for v in vals]

# ────────────────────────────────────────────────────────────────────────────────
# DATASETS
# ────────────────────────────────────────────────────────────────────────────────

class InsepaXY(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["IM"][dominio]["blocos"]
        self.pares = []
        for b in blocos:
            X = _montar_X_do_bloco(b)
            saidas = b.get("saidas") or [b.get("saida")]
            for s in filter(None, saidas):
                Y = _montar_Y_da_saida(s)
                self.pares.append((X, Y))
        if not self.pares:
            raise ValueError("Nenhum par (X,Y) encontrado.")
        self.max_x = max(len(x) for x,_ in self.pares)
        self.max_y = max(len(y) for _,y in self.pares)

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int):
        x, y    = self.pares[idx]
        x_pad   = x + [0.0] * (self.max_x - len(x))
        y_pad   = y + [0.0] * (self.max_y - len(y))
        return torch.tensor(x_pad, dtype=torch.float32), torch.tensor(y_pad, dtype=torch.float32)

class TestXY(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["IM"][dominio]["blocos"]
        if not blocos:
            raise ValueError("Nenhum bloco para teste.")
        self.Xs = [_montar_X_do_bloco(b) for b in blocos]
        _, max_x, _ = torch.load(ckpt_path(dominio))
        self.max_x = max_x
        self.ids   = [b["bloco_id"] for b in blocos]

    def __len__(self) -> int:
        return len(self.Xs)

    def __getitem__(self, idx: int):
        x     = self.Xs[idx]
        x_pad = x + [0.0] * (self.max_x - len(x))
        return torch.tensor(x_pad, dtype=torch.float32), self.ids[idx]

# ────────────────────────────────────────────────────────────────────────────────
# MODELO COM EMBEDDING INSEPA
# ────────────────────────────────────────────────────────────────────────────────

class EmbeddingINSEPA(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = EMBED_DIM):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class InsepaReg(nn.Module):
    def __init__(self, xin: int, yout: int, hidden: int = 32, emb_dim: int = EMBED_DIM):
        super().__init__()
        self.embed = EmbeddingINSEPA(xin, emb_dim)
        self.net   = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, yout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))

# ────────────────────────────────────────────────────────────────────────────────
# TREINO COM EARLY STOPPING
# ────────────────────────────────────────────────────────────────────────────────

def train(memoria: dict, dominio: str, patience: int = 5) -> None:
    torch.manual_seed(42)
    ds     = InsepaXY(memoria, dominio)
    n      = len(ds)
    ckpt   = ckpt_path(dominio)
    loss_fn = nn.MSELoss()

    # Se poucos dados, treino simples
    if n < 2:
        loader    = DataLoader(ds, batch_size=1, shuffle=True)
        model     = InsepaReg(ds.max_x, ds.max_y)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        print(f"⚠️ Apenas {n} amostra — sem validação.")
        for ep in range(1, 101):
            total = 0.0
            model.train()
            for X, Y in loader:
                optimizer.zero_grad()
                loss_fn(model(X), Y).backward()
                optimizer.step()
                total += loss_fn(model(X), Y).item()
            if ep % 10 == 0:
                print(f" Ep{ep:03d}/100  loss={total/len(loader):.4f}")
        torch.save((model.state_dict(), ds.max_x, ds.max_y), ckpt)
        print("✅ Treino concluído.")
        return

    # Split 80/20
    idxs       = list(range(n))
    random.shuffle(idxs)
    val_size   = max(1, int(0.2 * n))
    val_idx    = idxs[:val_size]
    train_idx  = idxs[val_size:]

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=4, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=4)

    model     = InsepaReg(ds.max_x, ds.max_y)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    lr = optimizer.param_groups[0]['lr']
    print(f"🚀 Treinando domínio {dominio} — lr={lr:.0e} — {len(train_idx)} train / {len(val_idx)} val")

    best_val, wait, prev_val = float("inf"), 0, None
    for ep in range(1, 101):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(X), Y).backward()
            optimizer.step()

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                total_val += loss_fn(model(X), Y).item()

        val_loss = total_val / len(val_loader)
        rmse     = val_loss ** 0.5

        if prev_val is None:
            delta, rel, status = 0.0, 0.0, "— início —"
        else:
            delta = prev_val - val_loss
            rel   = delta / prev_val if prev_val > 0 else 0.0
            if rel >= 0.05:
                status = "🔥 Alta taxa de aprendizado!"
            elif rel >= 0.01:
                status = "👍 Boa taxa de aprendizado"
            elif rel > 0:
                status = "⚠️ Baixa taxa de aprendizado"
            else:
                status = "❌ Perda aumentou!"

        print(
            f" Ep{ep:03d} val_loss={val_loss:.4f} rmse={rmse:.4f} "
            f"Δloss={delta:.4f} ({rel*100:.1f}%) {status}"
        )
        prev_val = val_loss

        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            torch.save((model.state_dict(), ds.max_x, ds.max_y), ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping (patience={patience})")
                break

    print(f"✅ Treino concluído. best_val_loss={best_val:.4f}")

# ────────────────────────────────────────────────────────────────────────────────
# INFERÊNCIA INTERATIVA COM BUG-FIX DE NOVA ENTRADA
# ────────────────────────────────────────────────────────────────────────────────

def infer(memoria: dict, dominio: str) -> None:
    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        print("⚠️ Sem checkpoint — treinando primeiro.")
        train(memoria, dominio)

    state, max_x, max_y = torch.load(ckpt)
    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    prompt_base = "(Enter ↩ próxima | 'insight' | 'roleplay' | 'novo' | 'sair') ► "

    raw = None
    while True:
        # se raw for None, pedimos nova entrada+reação
        if raw is None:
            raw = input("Entrada+Reação ► ").strip()

        cmd = raw.lower()
        if cmd == "sair":
            print("👋 Até mais!"); return

        if cmd == "roleplay":
            bloco = random.choice(blocos)
        else:
            txt, reac = parse_text_reaction(raw, blocos)
            bloco = next((
                b for b in blocos
                if normalize(b["entrada"]["texto"]) == normalize(txt)
                and b["entrada"].get("reacao","") == reac
            ), None)
            if not bloco:
                print("❌ Entrada+reação não cadastrada.")
                raw = None
                continue

        # predição e escolha da melhor saída
        X     = _montar_X_do_bloco(bloco)
        Xp    = X + [0.0] * (max_x - len(X))
        with torch.no_grad():
            y_hat = model(torch.tensor([Xp], dtype=torch.float32))[0].tolist()

        saidas = bloco.get("saidas") or [bloco.get("saida")]
        saidas = [s for s in saidas if s]
        dists = []
        for s in saidas:
            Y   = _montar_Y_da_saida(s)
            Yp  = Y + [0.0] * (max_y - len(Y))
            dists.append(sum((yh - yr)**2 for yh,yr in zip(y_hat, Yp)))
        best_idx = min(range(len(dists)), key=lambda i: dists[i])
        saida    = saidas[best_idx]

        # prepara variações
        vars_txt = saida.get("textos") or [saida.get("texto","")]
        ctx      = saida.get("contexto","").strip()
        if ctx:
            vars_txt = [v for v in vars_txt if normalize(v) != normalize(ctx)]
        reac_s   = saida.get("reacao","").strip()
        if reac_s:
            vars_txt = [f"{v} {reac_s}" for v in vars_txt]

        idx = 0
        print(f"\n🤖 {vars_txt[idx]}")

        last_bloco = bloco
        last_saida = saida

        # loop pós-resposta: comandos ou nova entrada
        while True:
            sub = input(prompt_base).strip()

            # variação de resposta
            if sub == "":
                idx = (idx + 1) % len(vars_txt)
                print(f"\n🤖 {vars_txt[idx]}")
                continue

            sb = sub.lower()
            if sb == "sair":
                print("👋 Até mais!"); return
            if sb == "novo":
                raw = None
                break
            if sb == "roleplay":
                raw = "roleplay"
                break
            if sb == "insight":
                explic = last_saida.get("explicacao","").strip()
                if explic:
                    print(f"\n💡 Insight:\n  {explic}")
                else:
                    ent = last_bloco["entrada"]
                    txt = ent.get("texto","").strip()
                    re  = ent.get("reacao","").strip()
                    cx  = ent.get("contexto","").strip()
                    auto = (
                        f"Devido à expressão \"{txt}\", "
                        f"a reação emocional \"{re}\" "
                        f"e a breve noção do assunto \"{cx}\", "
                        f"concluo que esta é a melhor resposta."
                    )
                    print(f"\n💡 Insight:\n  {auto}")
                continue

            # QUALQUER OUTRA STRING → nova entrada+reação
            raw = sub
            break

        # volta ao loop externo com raw definido ou None
        continue

# ────────────────────────────────────────────────────────────────────────────────
# TESTE EM LOTE
# ────────────────────────────────────────────────────────────────────────────────

def test_model(memoria: dict, dominio: str) -> None:
    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        print("⚠️ Sem checkpoint — treine antes."); return

    state, max_x, max_y = torch.load(ckpt)
    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    print(f"📊 Teste em lote — Domínio {dominio} ({len(blocos)} blocos)")
    for b in blocos:
        X   = _montar_X_do_bloco(b)
        Xp  = torch.tensor([X + [0.0]*(max_x - len(X))], dtype=torch.float32)
        y_h = model(Xp)[0].detach().tolist()
        print(f"\n❏ Bloco_id={b['bloco_id']} Entrada: {b['entrada']['texto']} {b['entrada']['reacao']}")
        print(f"   Y_pred: {y_h}")

# ────────────────────────────────────────────────────────────────────────────────
# CLI PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────

def menu_principal() -> str:
    print("\n=== Menu Principal ===")
    print("1) Treinar rede neural")
    print("2) Inferir com rede neural")
    print("3) Testar em lote")
    print("4) Sair do programa")
    return input("Escolha uma opção (1/2/3/4): ").strip()

def prompt_dominio(action: str) -> str:
    return input(f"→ Índice-mãe p/ {action} (ou 'sair' p/ voltar): ").strip()

def main():
    memoria      = carregar_json(ARQUIVO_MEMORIA,      {"IM": {}})
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"conteudos": []})

    while True:
        opc = menu_principal()
        if opc == "1":
            dom = prompt_dominio("treinar")
            if dom.lower() == "sair":
                continue
            if dom in memoria["IM"]:
                print(f"\n🚀 Treinando domínio {dom}...")
                train(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")

        elif opc == "2":
            dom = prompt_dominio("inferir")
            if dom.lower() == "sair":
                continue
            if dom in memoria["IM"]:
                infer(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")

        elif opc == "3":
            dom = prompt_dominio("testar")
            if dom.lower() == "sair":
                continue
            if dom in memoria["IM"]:
                test_model(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")

        elif opc == "4":
            print("👋 Até mais!"); break
        else:
            print("❌ Opção inválida. Digite 1, 2, 3 ou 4.")

if __name__ == "__main__":
    main()