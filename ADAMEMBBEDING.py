#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import re as _re
from typing import List, Dict, Tuple

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
HIDDEN_DIM           = 64
PATIENCE             = 5
BATCH_SIZE           = 8
LR                   = 1e-3
EPOCHS               = 50

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
# NORMALIZAÇÃO E PARSING
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
# VOCABULÁRIOS POR CAMPO E RÓTULOS
# ────────────────────────────────────────────────────────────────────────────────

def build_field_vocabs(memoria: dict, dominio: str) -> Dict[str,Dict[str,int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets   = { "E": set(), "RE": set(), "CE": set(), "PIDE": set() }
    for b in blocos:
        t = b["entrada"]["tokens"]
        for f in sets:
            sets[f] |= set(t.get(f, []))
    return {
        f: {tok: i+1 for i, tok in enumerate(sorted(sets[f]))}
        for f in sets
    }

def build_label_vocabs(memoria: dict, dominio: str) -> Dict[str,Dict[str,int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets   = { "texto": set(), "emoji": set(), "ctx": set(),
               "exp": set(),   "ime": set() }
    for b in blocos:
        for s in b.get("saidas", []):
            for v in s.get("textos", []):
                sets["texto"].add(normalize(v))
            emo = s.get("reacao","")
            if emo:   sets["emoji"].add(emo)
            ctx = s.get("contexto","")
            if ctx:   sets["ctx"].add(normalize(ctx))
            exp = s.get("explicacao","")
            if exp:   sets["exp"].add(normalize(exp))
            ime = s.get("imersao","")
            if ime:   sets["ime"].add(normalize(ime))
    return {
        f: {tok: i for i, tok in enumerate(sorted(sets[f]))}
        for f in sets
    }

# ────────────────────────────────────────────────────────────────────────────────
# DATASET MULTI-FIELD COM VALOR, MÃE E POSIÇÃO
# ────────────────────────────────────────────────────────────────────────────────

class InsepaFieldDataset(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        fv = build_field_vocabs(memoria, dominio)
        lv = build_label_vocabs(memoria, dominio)
        self.v_E, self.v_RE, self.v_CE, self.v_PIDE = \
            fv["E"], fv["RE"], fv["CE"], fv["PIDE"]
        self.l_txt, self.l_emo, self.l_ctx, self.l_exp, self.l_ime = \
            lv["texto"], lv["emoji"], lv["ctx"], lv["exp"], lv["ime"]

        blocos = memoria["IM"][dominio]["blocos"]
        self.max_E    = max(len(b["entrada"]["tokens"].get("E",[]))    for b in blocos)
        self.max_RE   = max(len(b["entrada"]["tokens"].get("RE",[]))   for b in blocos)
        self.max_CE   = max(len(b["entrada"]["tokens"].get("CE",[]))   for b in blocos)
        self.max_PIDE = max(len(b["entrada"]["tokens"].get("PIDE",[])) for b in blocos)
        self.max_pos  = max(self.max_E, self.max_RE, self.max_CE, self.max_PIDE)

        # calcula mom_size = maior mãe + 1
        max_mom = 0
        for b in blocos:
            for tok in b["entrada"]["tokens"].get("TOTAL", []):
                m = int(tok.split(".",1)[0])
                if m > max_mom: max_mom = m
        self.mom_size = max_mom + 1

        self.pares: List[Tuple[Dict, Dict]] = []
        for b in blocos:
            E_ids    = [self.v_E[t]    for t in b["entrada"]["tokens"].get("E", [])]
            RE_ids   = [self.v_RE[t]   for t in b["entrada"]["tokens"].get("RE",[])]
            CE_ids   = [self.v_CE[t]   for t in b["entrada"]["tokens"].get("CE",[])]
            PIDE_ids = [self.v_PIDE[t] for t in b["entrada"]["tokens"].get("PIDE",[])]
            E_ids    += [0]*(self.max_E    - len(E_ids))
            RE_ids   += [0]*(self.max_RE   - len(RE_ids))
            CE_ids   += [0]*(self.max_CE   - len(CE_ids))
            PIDE_ids += [0]*(self.max_PIDE - len(PIDE_ids))

            # função para gerar valores, mães e posições
            def build_feats(lst, maxlen):
                vals = [float(tok) for tok in lst]
                moms = [int(tok.split(".",1)[0]) for tok in lst]
                pos  = list(range(len(lst)))
                pad = maxlen - len(lst)
                vals += [0.0]*pad
                moms += [0]*pad
                pos  += [0]*pad
                return vals, moms, pos

            E_vals, E_moms, E_pos     = build_feats(b["entrada"]["tokens"].get("E",[]),    self.max_E)
            RE_vals, RE_moms, RE_pos  = build_feats(b["entrada"]["tokens"].get("RE",[]),   self.max_RE)
            CE_vals, CE_moms, CE_pos  = build_feats(b["entrada"]["tokens"].get("CE",[]),   self.max_CE)
            PI_vals, PI_moms, PI_pos  = build_feats(b["entrada"]["tokens"].get("PIDE",[]), self.max_PIDE)

            for s in b.get("saidas", []):
                y = {
                    "texto": self.l_txt[normalize(s["textos"][0])],
                    "emoji": self.l_emo.get(s.get("reacao",""), 0),
                    "ctx":   self.l_ctx.get(normalize(s.get("contexto","")), 0),
                    "exp":   self.l_exp.get(normalize(s.get("explicacao","")), 0),
                    "ime":   self.l_ime.get(normalize(s.get("imersao","")), 0),
                }
                x = {
                    "E":      E_ids,    "E_val":  E_vals,  "E_mom":  E_moms,  "E_pos":  E_pos,
                    "RE":     RE_ids,   "RE_val": RE_vals, "RE_mom": RE_moms, "RE_pos": RE_pos,
                    "CE":     CE_ids,   "CE_val": CE_vals, "CE_mom": CE_moms, "CE_pos": CE_pos,
                    "PIDE":   PIDE_ids, "PIDE_val":PI_vals, "PIDE_mom":PI_moms, "PIDE_pos":PI_pos,
                }
                self.pares.append((x, y))

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int):
        x, y = self.pares[idx]
        x_t = {
            "E":      torch.tensor(x["E"],      dtype=torch.long),
            "E_val":  torch.tensor(x["E_val"],  dtype=torch.float32),
            "E_mom":  torch.tensor(x["E_mom"],  dtype=torch.long),
            "E_pos":  torch.tensor(x["E_pos"],  dtype=torch.long),

            "RE":     torch.tensor(x["RE"],     dtype=torch.long),
            "RE_val": torch.tensor(x["RE_val"], dtype=torch.float32),
            "RE_mom": torch.tensor(x["RE_mom"], dtype=torch.long),
            "RE_pos": torch.tensor(x["RE_pos"], dtype=torch.long),

            "CE":     torch.tensor(x["CE"],     dtype=torch.long),
            "CE_val": torch.tensor(x["CE_val"], dtype=torch.float32),
            "CE_mom": torch.tensor(x["CE_mom"], dtype=torch.long),
            "CE_pos": torch.tensor(x["CE_pos"], dtype=torch.long),

            "PIDE":     torch.tensor(x["PIDE"],     dtype=torch.long),
            "PIDE_val": torch.tensor(x["PIDE_val"], dtype=torch.float32),
            "PIDE_mom": torch.tensor(x["PIDE_mom"], dtype=torch.long),
            "PIDE_pos": torch.tensor(x["PIDE_pos"], dtype=torch.long),
        }
        y_t = {
            "texto": torch.tensor(y["texto"], dtype=torch.long),
            "emoji": torch.tensor(y["emoji"], dtype=torch.long),
            "ctx":   torch.tensor(y["ctx"],   dtype=torch.long),
            "exp":   torch.tensor(y["exp"],   dtype=torch.long),
            "ime":   torch.tensor(y["ime"],   dtype=torch.long),
        }
        return x_t, y_t

# ────────────────────────────────────────────────────────────────────────────────
# MODELO MULTI‐HEAD COM CATEGORIA, VALOR, MÃE E POSIÇÃO
# ────────────────────────────────────────────────────────────────────────────────

class AdamSegmentado(nn.Module):
    def __init__(self,
                 nE:int, nRE:int, nCE:int, nPIDE:int,
                 mom_size:int, max_pos:int,
                 n_txt:int, n_emo:int, n_ctx:int, n_exp:int, n_ime:int):
        super().__init__()
        # Embeddings por campo
        # E
        self.em_E      = nn.Embedding(nE+1,    EMBED_DIM, padding_idx=0)
        self.proj_Eval = nn.Linear(1, EMBED_DIM, bias=False)
        self.em_Emom   = nn.Embedding(mom_size, EMBED_DIM, padding_idx=0)
        self.em_Epos   = nn.Embedding(max_pos,  EMBED_DIM, padding_idx=0)
        # RE
        self.em_RE      = nn.Embedding(nRE+1,    EMBED_DIM, padding_idx=0)
        self.proj_REval = nn.Linear(1, EMBED_DIM, bias=False)
        self.em_REmom   = nn.Embedding(mom_size, EMBED_DIM, padding_idx=0)
        self.em_REpos   = nn.Embedding(max_pos,  EMBED_DIM, padding_idx=0)
        # CE
        self.em_CE      = nn.Embedding(nCE+1,    EMBED_DIM, padding_idx=0)
        self.proj_CEval = nn.Linear(1, EMBED_DIM, bias=False)
        self.em_CEmom   = nn.Embedding(mom_size, EMBED_DIM, padding_idx=0)
        self.em_CEpos   = nn.Embedding(max_pos,  EMBED_DIM, padding_idx=0)
        # PIDE
        self.em_PIDE      = nn.Embedding(nPIDE+1,    EMBED_DIM, padding_idx=0)
        self.proj_PIDEval = nn.Linear(1, EMBED_DIM, bias=False)
        self.em_PIDEmom   = nn.Embedding(mom_size,    EMBED_DIM, padding_idx=0)
        self.em_PIDEpos   = nn.Embedding(max_pos,     EMBED_DIM, padding_idx=0)

        total = EMBED_DIM * 4
        self.fc1 = nn.Linear(total, HIDDEN_DIM)
        self.act = nn.ReLU()

        # Cabeças de saída
        self.h_txt = nn.Linear(HIDDEN_DIM, n_txt)
        self.h_emo = nn.Linear(HIDDEN_DIM, n_emo)
        self.h_ctx = nn.Linear(HIDDEN_DIM, n_ctx)
        self.h_exp = nn.Linear(HIDDEN_DIM, n_exp)
        self.h_ime = nn.Linear(HIDDEN_DIM, n_ime)

    def forward(self, x: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        # Campo E
        eE_tok  = self.em_E(x["E"])
        eE_val  = self.proj_Eval(x["E_val"].unsqueeze(-1))
        eE_mom  = self.em_Emom(x["E_mom"])
        eE_pos  = self.em_Epos(x["E_pos"])
        eE      = (eE_tok + eE_val + eE_mom + eE_pos).mean(dim=1)
        # Campo RE
        eRE_tok = self.em_RE(x["RE"])
        eRE_val = self.proj_REval(x["RE_val"].unsqueeze(-1))
        eRE_mom = self.em_REmom(x["RE_mom"])
        eRE_pos = self.em_REpos(x["RE_pos"])
        eRE      = (eRE_tok + eRE_val + eRE_mom + eRE_pos).mean(dim=1)
        # Campo CE
        eCE_tok = self.em_CE(x["CE"])
        eCE_val = self.proj_CEval(x["CE_val"].unsqueeze(-1))
        eCE_mom = self.em_CEmom(x["CE_mom"])
        eCE_pos = self.em_CEpos(x["CE_pos"])
        eCE      = (eCE_tok + eCE_val + eCE_mom + eCE_pos).mean(dim=1)
        # Campo PIDE
        ePI_tok = self.em_PIDE(x["PIDE"])
        ePI_val = self.proj_PIDEval(x["PIDE_val"].unsqueeze(-1))
        ePI_mom = self.em_PIDEmom(x["PIDE_mom"])
        ePI_pos = self.em_PIDEpos(x["PIDE_pos"])
        ePIDE    = (ePI_tok + ePI_val + ePI_mom + ePI_pos).mean(dim=1)

        # Agrega e classifica
        h = torch.cat([eE, eRE, eCE, ePIDE], dim=1)
        h = self.act(self.fc1(h))
        return {
            "texto": self.h_txt(h),
            "emoji": self.h_emo(h),
            "ctx":   self.h_ctx(h),
            "exp":   self.h_exp(h),
            "ime":   self.h_ime(h),
        }

# ────────────────────────────────────────────────────────────────────────────────
# TREINO COM CrossEntropyLoss E EARLY STOPPING
# ────────────────────────────────────────────────────────────────────────────────

def train(memoria: dict, dominio: str) -> None:
    ds = InsepaFieldDataset(memoria, dominio)
    n  = len(ds)
    ckpt = ckpt_path(dominio)

    idxs = list(range(n))
    random.shuffle(idxs)
    vsz = max(1, int(0.2 * n))
    vidx, tidx = idxs[:vsz], idxs[vsz:]
    train_ld = DataLoader(Subset(ds, tidx), batch_size=BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(Subset(ds, vidx), batch_size=BATCH_SIZE)

    model = AdamSegmentado(
        nE=len(ds.v_E), nRE=len(ds.v_RE),
        nCE=len(ds.v_CE), nPIDE=len(ds.v_PIDE),
        mom_size=ds.mom_size, max_pos=ds.max_pos,
        n_txt=len(ds.l_txt), n_emo=len(ds.l_emo),
        n_ctx=len(ds.l_ctx), n_exp=len(ds.l_exp), n_ime=len(ds.l_ime)
    )
    opt = optim.Adam(model.parameters(), lr=LR)
    ce  = nn.CrossEntropyLoss()

    best, wait, prev_val = float("inf"), 0, None
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in train_ld:
            opt.zero_grad()
            out = model(x)
            loss = (
                ce(out["texto"], y["texto"]) +
                ce(out["emoji"], y["emoji"]) +
                ce(out["ctx"],   y["ctx"]) +
                ce(out["exp"],   y["exp"]) +
                ce(out["ime"],   y["ime"])
            )
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_ld:
                out = model(x)
                val_loss += (
                    ce(out["texto"], y["texto"]).item() +
                    ce(out["emoji"], y["emoji"]).item() +
                    ce(out["ctx"],   y["ctx"]).item() +
                    ce(out["exp"],   y["exp"]).item() +
                    ce(out["ime"],   y["ime"]).item()
                )
        val_loss /= len(val_ld)

        if prev_val is None or val_loss < best:
            best, wait = val_loss, 0
            torch.save((
                model.state_dict(),
                ds.max_E, ds.max_RE, ds.max_CE, ds.max_PIDE,
                ds.mom_size, ds.max_pos,
                ds.v_E, ds.v_RE, ds.v_CE, ds.v_PIDE,
                ds.l_txt, ds.l_emo, ds.l_ctx, ds.l_exp, ds.l_ime
            ), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        prev_val = val_loss

    print(f"✅ Treino concluído. best_val_loss={best:.4f}")
