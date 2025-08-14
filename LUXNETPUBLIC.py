# -*- coding: utf-8 -*-
# Adam – CLI Multiversal com Estética Cósmica 🌌
# MODO PÚBLICO — SOMENTE INFERÊNCIA COM ESCADINHA

import os, json, re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ARQUIVO_MEMORIA = "adam_memoria.json"
CKPT            = "insepa_xy.pt"

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
    Separa texto e reação exatamente como estão no JSON.
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

def _saida_tokens_legacy_or_insepa(saida: dict):
    """
    Compat: legado (E/RE/CE) e atual (S/RS/CS). Retorna sempre (S, RS, CS).
    """
    t = saida.get("tokens", {})
    if "S" in t or "RS" in t or "CS" in t:
        S  = t.get("S", [])
        RS = t.get("RS", [])
        CS = t.get("CS", [])
    else:
        S  = t.get("E", [])
        RS = t.get("RE", [])
        CS = t.get("CE", [])
    return S, RS, CS

def xy_from_block_many(b: dict):
    """
    Gera múltiplos pares (X, Y) por bloco (uma amostra por saída).
    X = E_in + RE_in + CE_in
    Y = S_out + RS_out + CS_out
    """
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    X    = Ein + REin + CEin
    pares = []
    if "saidas" in b and b["saidas"]:
        for saida in b["saidas"]:
            S, RS, CS = _saida_tokens_legacy_or_insepa(saida)
            Y = [float(v) for v in (S + RS + CS)]
            pares.append((X, Y))
    elif "saida" in b and b["saida"]:
        S, RS, CS = _saida_tokens_legacy_or_insepa(b["saida"])
        Y = [float(v) for v in (S + RS + CS)]
        pares.append((X, Y))
    return pares

# ────────────────────────────────────────────────────────────────────────────────
# Dataset e modelo
# ────────────────────────────────────────────────────────────────────────────────
class InsepaXY(Dataset):
    """Pares (X, Y) com padding automático, cobrindo todas as saídas dos blocos."""
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["maes"][dominio]["blocos"]
        self.pares = []
        for b in blocos:
            self.pares.extend(xy_from_block_many(b))
        if not self.pares:
            raise ValueError("Nenhum par (X,Y) encontrado. Verifique se há saídas nos blocos.")
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
    """MLP simples X→Y."""
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
# Escadinha — helpers
# ────────────────────────────────────────────────────────────────────────────────
def _montar_X_do_bloco(b: dict) -> list[float]:
    Ein  = [float(v) for v in b["entrada"]["tokens"].get("E", [])]
    REin = [float(v) for v in b["entrada"]["tokens"].get("RE", [])]
    CEin = [float(v) for v in b["entrada"]["tokens"].get("CE", [])]
    return Ein + REin + CEin

def _montar_Y_da_saida(saida: dict) -> list[float]:
    S, RS, CS = _saida_tokens_legacy_or_insepa(saida)
    return [float(v) for v in (S + RS + CS)]

def _escolher_saida_por_modelo(model, max_x: int, max_y: int, bloco: dict) -> dict | None:
    X = _montar_X_do_bloco(bloco)
    X_pad = X + [0.0] * (max_x - len(X))
    with torch.no_grad():
        y_hat = model(torch.tensor([X_pad], dtype=torch.float32))[0].numpy()

    saidas = bloco.get("saidas") or ([bloco["saida"]] if "saida" in bloco else [])
    if not saidas:
        return None

    melhor, best_idx = float("inf"), None
    for i, s in enumerate(saidas):
        Y = _montar_Y_da_saida(s)
        Y_pad = Y + [0.0] * (max_y - len(Y))
        dist = sum((float(yh) - float(yr)) ** 2 for yh, yr in zip(y_hat, Y_pad))
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

# ────────────────────────────────────────────────────────────────────────────────
# Inferência (playlist interativa com escadinha de blocos)
# ────────────────────────────────────────────────────────────────────────────────
def infer(memoria: dict, dominio: str) -> None:
    if not os.path.exists(CKPT):
        print("⚠️ A criadora ainda não manifestou sua obra neste espaço.")
        return

    state, max_x, max_y = torch.load(CKPT)
    blocos = memoria["maes"][dominio]["blocos"]
    if not blocos:
        print("⚠️ Não há cenários prontos ainda. Mas estamos em construção.")
        return

    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    # Entrada inicial
    raw = input("👤 Digite sua mensagem e o emoji: \n ")
    txt, rea = parse_text_reaction(raw, blocos)

    # Localiza bloco inicial
    bloco_atual = next((b for b in blocos
                        if b.get("entrada", {}).get("texto") == txt
                        and b.get("entrada", {}).get("reacao", "") == rea), None)
    if bloco_atual is None:
        print("❌ Essa mensagem não pertence a este universo")
        return

    # Loop principal: playlist com escadinha entre blocos
    while True:
        saida_escolhida = _escolher_saida_por_modelo(model, max_x, max_y, bloco_atual)
        if not saida_escolhida:
            print("⚠️ Sem resposta.")
            return

        variacoes = _variacoes_da_saida(saida_escolhida)
        idx = 0
        acabou_sem_nova = False

        while True:
            # 1) Emite a próxima variação desta saída
            if idx < len(variacoes):
                print(f"\n🤖 {variacoes[idx]}")
                idx += 1
            else:
                # Acabaram as variações desta saída
                print("\nSinto muito o universo acaba aqui.\n Mas não desanime. Como o cosmos estamos sempre em expansão!")
                acabou_sem_nova = True
                break

            # 2) Espera Enter (próxima) ou nova entrada (possível mudança de bloco)
            entrada_user = input("Digite Enter ou uma nova mensagem com emoji ")
            if entrada_user.strip():
                novo_txt, novo_rea = parse_text_reaction(entrada_user, blocos)
                # Tenta localizar um novo bloco por texto+reação exatos
                bloco_novo = next((b for b in blocos
                                   if b.get("entrada", {}).get("texto") == novo_txt
                                   and b.get("entrada", {}).get("reacao", "") == novo_rea), None)
                if bloco_novo:
                    # Troca de bloco (escadinha) e reinicia o laço externo
                    bloco_atual = bloco_novo
                    break
                else:
                    print("❌ Essa interação não faz parte deste universo...")
            # Se só Enter, continua no mesmo bloco/saída/variações

        # Saiu do while interno por troca de bloco? Continua no while externo.
        # Saiu por fim de variações sem nova entrada? Encerramos geral.
        if acabou_sem_nova:
            break

# ────────────────────────────────────────────────────────────────────────────────
# CLI — somente inferência
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump({"maes": {}}, f, ensure_ascii=False, indent=2)

    with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
        memoria = json.load(f)

    # Lista universos
    if memoria.get("maes"):
        print("\n🌌 Universos disponíveis no Multiverso Lux:")
        for idx, mae in memoria["maes"].items():
            print(f"  • {mae.get('nome', idx)}   {idx}")

    dominio = input("\nDigite a chave de acesso do seu universo: ").strip()
    if dominio not in (memoria.get("maes") or {}):
        print(f"❌ Domínio '{dominio}' não encontrado em {ARQUIVO_MEMORIA}.")
    else:
        infer(memoria, dominio)