import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ARQUIVO_MEMORIA = "adam_memoria.json"
CKPT            = "insepa_xy.pt"


def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else txt + "."


def tokenizar(txt: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", txt, re.UNICODE)


def parse_text_reaction(raw: str, blocos: list[dict]) -> tuple[str, str]:
    """
    Separa texto e reação EXATAMENTE como estão no JSON.
    Tenta cada reação cadastrada (ordem decrescente de tamanho).
    """
    s = raw.strip()
    # coleta reações únicas e ordena por comprimento (maiores primeiro)
    reactions = sorted(
        {b["entrada"]["reacao"] for b in blocos if b["entrada"]["reacao"]},
        key=len, reverse=True
    )
    for rea in reactions:
        if s.endswith(rea):
            txt = s[:-len(rea)].rstrip()
            return garantir_pontuacao(txt), rea
    return garantir_pontuacao(s), ""


def xy_from_block(b: dict) -> tuple[list[float], list[float]]:
    """Retorna X e Y (vetores de floats) de um bloco INSEPA."""
    E_in   = [float(v) for v in b["entrada"]["tokens"]["E"]]
    RE_in  = [float(v) for v in b["entrada"]["tokens"]["RE"]]
    CE_in  = [float(v) for v in b["entrada"]["tokens"]["CE"]]
    X      = E_in + RE_in + CE_in

    E_out  = [float(v) for v in b["saida"]["tokens"]["E"]]
    RE_out = [float(v) for v in b["saida"]["tokens"]["RE"]]
    CE_out = [float(v) for v in b["saida"]["tokens"]["CE"]]
    Y      = E_out + RE_out + CE_out

    return X, Y


class InsepaXY(Dataset):
    """Dataset de pares (X, Y), com padding automático."""
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["maes"][dominio]["blocos"]
        self.pares = [xy_from_block(b) for b in blocos]
        self.max_x = max(len(x) for x, _ in self.pares)
        self.max_y = max(len(y) for _, y in self.pares)

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.pares[idx]
        x_pad = x + [0.0] * (self.max_x - len(x))
        y_pad = y + [0.0] * (self.max_y - len(y))
        return torch.tensor(x_pad, dtype=torch.float32), \
               torch.tensor(y_pad, dtype=torch.float32)


class InsepaReg(nn.Module):
    """MLP regressor X→Y."""
    def __init__(self, xin: int, yout: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, yout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(memoria: dict, dominio: str) -> None:
    blocos = memoria["maes"][dominio]["blocos"]
    ds     = InsepaXY(memoria, dominio)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model  = InsepaReg(ds.max_x, ds.max_y)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    print(f"🚀 Treinando domínio {dominio} ({len(ds)} blocos)...")
    for ep in range(1, 101):
        total_loss = 0.0
        model.train()
        for X, Y in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep == 1 or ep % 10 == 0:
            avg = total_loss / len(loader)
            print(f" Ep {ep:02d}/50  loss={avg:.4f}")

    torch.save((model.state_dict(), ds.max_x, ds.max_y), CKPT)
    print(f"✅ Treino concluído. Checkpoint salvo em '{CKPT}'\n")


def infer(memoria: dict, dominio: str) -> None:
    if not os.path.exists(CKPT):
        print("⚠️ Checkpoint não encontrado. Treinando antes de inferir.")
        train(memoria, dominio)

    state, max_x, max_y = torch.load(CKPT)
    blocos = memoria["maes"][dominio]["blocos"]

    model = InsepaReg(max_x, max_y)
    model.load_state_dict(state)
    model.eval()

    raw = input("👤 Entrada + Reação: ")
    txt, rea = parse_text_reaction(raw, blocos)

    # recupera X do bloco correspondente
    X = None
    for b in blocos:
        if txt == b["entrada"]["texto"] and rea == b["entrada"]["reacao"]:
            X, _ = xy_from_block(b)
            break

    if X is None:
        print("❌ Entrada+reação não cadastrada.")
        return

    # inferência e busca pelo Y real mais próximo
    X_pad = X + [0.0] * (max_x - len(X))
    with torch.no_grad():
        y_hat = model(torch.tensor([X_pad], dtype=torch.float32))[0].numpy()

    melhor, best_id = float("inf"), None
    for b in blocos:
        _, Yreal = xy_from_block(b)
        Y_pad    = Yreal + [0.0] * (max_y - len(Yreal))
        dist     = sum((yh - yr) ** 2 for yh, yr in zip(y_hat, Y_pad))
        if dist < melhor:
            melhor, best_id = dist, b["bloco_id"]

    resposta = next(b for b in blocos if b["bloco_id"] == best_id)["saida"]
    print(f"\n🤖 {resposta['texto']} {resposta['reacao']}")


if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump({"maes": {}}, f, ensure_ascii=False, indent=2)

    memoria = json.load(open(ARQUIVO_MEMORIA, "r", encoding="utf-8"))
    dominio = input("Domínio (índice-mãe): ").strip()

    print("\n1) Treinar rede neural   2) Inferir com rede neural")
    op = input("Opção: ").strip()
    if op == "1":
        train(memoria, dominio)
    else:
        infer(memoria, dominio)
