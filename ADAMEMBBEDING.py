#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import re as _re
from typing import List, Dict, Tuple, Any, Optional

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
UNK = "<UNK>"
UNK_VAL = -1.0
N_GRAM = 2  # Tamanho do n-grama (2 para bigrams)

## INSEPA_TOKENIZER
def generate_ngrams(token: str, n: int) -> List[str]:
    """Gera n-gramas de caracteres de um token."""
    if len(token) < n:
        return [token]  # Se menor que n, retorna o token inteiro
    return [token[i:i+n] for i in range(len(token) - n + 1)]

def ckpt_path(dominio: str) -> str:
    return f"insepa_{dominio}.pt"

def Token(text: str) -> List[str]:
    """INSEPA tokenização: mantém palavras, pontuação, emojis, stopwords."""
    return _re.findall(r'\w+|[^\w\s]', text, _re.UNICODE)

def next_marker(prev: str) -> str:
    """Incrementa sem arredondar: 0.99 → 0.100"""
    mom, _, suf = prev.partition('.')
    if not mom.isdigit():
        raise ValueError(f"Marcador inválido: {prev!r}")
    return f"{mom}.{int(suf or '0') + 1}"

def generate_markers(start: str, count: int) -> List[str]:
    seq, cur = [], start
    for _ in range(count):
        cur = next_marker(cur)
        seq.append(cur)
    return seq

## INSEPA_UTILS
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
            return txt, reac
    return s, ""

def build_field_vocabs(memoria: dict, dominio: str) -> Dict[str,Dict[str,int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets   = { "E": set(), "RE": set(), "CE": set(), "PIDE": set() }
    for b in blocos:
        t = b["entrada"]["tokens"]
        for f in sets:
            for tok in t.get(f, []):
                ngrams = generate_ngrams(tok, N_GRAM)
                sets[f].update(ngrams)
    return {
        f: {ng: i+1 for i, ng in enumerate(sorted(sets[f]))}
        for f in sets
    }

def build_label_vocabs(memoria: dict, dominio: str) -> Dict[str,Dict[str,int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets   = { "texto": set(), "emoji": set(), "ctx": set() }
    for b in blocos:
        for s in b.get("saidas", []):
            for v in s.get("textos", []):
                sets["texto"].add(normalize(v))
            emo = s.get("reacao","")
            if emo:   sets["emoji"].add(emo)
            ctx = s.get("contexto","")
            if ctx:   sets["ctx"].add(normalize(ctx))
    return {
        f: {tok: i for i, tok in enumerate(sorted(sets[f]))}
        for f in sets
    }

## INSEPA_DATASET
class InsepaFieldDataset(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["IM"][dominio]["blocos"]
        inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
        self.ultimo_child_per_block = {}
        if dominio in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][dominio]:
            bloco_num = int(inconsciente["INCO"][dominio]["Blocos"]["Bloco_id"])
            saida_vals = [float(key) for key in inconsciente["INCO"][dominio]["Blocos"].get("SAÍDA", {}).keys()]
            if saida_vals:
                self.ultimo_child_per_block[bloco_num] = max(saida_vals)
            else:
                self.ultimo_child_per_block[bloco_num] = 0.50

        # Coletar tokens únicos por campo diretamente do JSON
        sets = {"E": set(), "RE": set(), "CE": set(), "PIDE": set()}
        for b in blocos:
            for f in sets:
                sets[f] |= set(b["entrada"]["tokens"].get(f, []))

        self.v_E = {tok: i+1 for i, tok in enumerate(sorted(sets["E"]))}
        self.v_RE = {tok: i+1 for i, tok in enumerate(sorted(sets["RE"]))}
        self.v_CE = {tok: i+1 for i, tok in enumerate(sorted(sets["CE"]))}
        self.v_PIDE = {tok: i+1 for i, tok in enumerate(sorted(sets["PIDE"]))}
        self.v_E[UNK] = len(self.v_E)
        self.v_RE[UNK] = len(self.v_RE)
        self.v_CE[UNK] = len(self.v_CE)
        self.v_PIDE[UNK] = len(self.v_PIDE)

        # val_to_idx por campo: tokens únicos como chaves
        self.val_to_idx_E = {tok: i for i, tok in enumerate(sorted(sets["E"]))}
        self.val_to_idx_RE = {tok: i for i, tok in enumerate(sorted(sets["RE"]))}
        self.val_to_idx_CE = {tok: i for i, tok in enumerate(sorted(sets["CE"]))}
        self.val_to_idx_PIDE = {tok: i for i, tok in enumerate(sorted(sets["PIDE"]))}

        self.max_E    = max(len(b["entrada"]["tokens"].get("E",[]))    for b in blocos)
        self.max_RE   = max(len(b["entrada"]["tokens"].get("RE",[]))   for b in blocos)
        self.max_CE   = max(len(b["entrada"]["tokens"].get("CE",[]))   for b in blocos)
        self.max_PIDE = max(len(b["entrada"]["tokens"].get("PIDE",[])) for b in blocos)
        self.max_pos  = max(self.max_E, self.max_RE, self.max_CE, self.max_PIDE)

        # Calcular max n-gramas por token
        all_tokens = set()
        for b in blocos:
            for field in ["E", "RE", "CE", "PIDE"]:
                all_tokens |= set(b["entrada"]["tokens"].get(field, []))
        self.max_ng = max(len(generate_ngrams(t, N_GRAM)) for t in all_tokens if t) if all_tokens else 1
        self.max_E_ng = self.max_E * self.max_ng
        self.max_RE_ng = self.max_RE * self.max_ng
        self.max_CE_ng = self.max_CE * self.max_ng
        self.max_PIDE_ng = self.max_PIDE * self.max_ng

        # calcula mom_size = maior mãe + 1
        max_mom = 0
        for b in blocos:
            for tok in b["entrada"]["tokens"].get("TOTAL", []):
                m = int(tok.split(".",1)[0])
                if m > max_mom: max_mom = m
        self.mom_size = max_mom + 1

        # valores únicos para posições fixas (não usado agora, mas manter compatibilidade)
        vals = {float(t) for t in all_tokens if t}
        sorted_vals = sorted(vals)
        self.val_to_idx = {v: i+1 for i, v in enumerate(sorted_vals)}  # índices de 1 em diante, 0 para padding
        self.num_vals = len(sorted_vals)

        # vocabulários de rótulos por bloco
        self.n_txt = max(len(b["saidas"][0]["textos"]) for b in blocos)
        self.n_emo = max(1, len(set(b["saidas"][0].get("reacao","") for b in blocos if b["saidas"][0].get("reacao"))))
        self.n_ctx = max(1, len(set(normalize(b["saidas"][0].get("contexto","")) for b in blocos if b["saidas"][0].get("contexto"))))

        self.pares: List[Tuple[Dict, Dict]] = []
        for b in blocos:
            bloco_id = b["bloco_id"]
            max_val = self.ultimo_child_per_block.get(bloco_id, 0.50)

            # Usar tokens fixos do JSON
            E_tokens = b["entrada"]["tokens"].get("E", [])
            RE_tokens = b["entrada"]["tokens"].get("RE", [])
            CE_tokens = b["entrada"]["tokens"].get("CE", [])
            PIDE_tokens = b["entrada"]["tokens"].get("PIDE", [])

            # Gerar n-gramas e ids
            E_ngrams = [generate_ngrams(t, N_GRAM) for t in E_tokens]
            RE_ngrams = [generate_ngrams(t, N_GRAM) for t in RE_tokens]
            CE_ngrams = [generate_ngrams(t, N_GRAM) for t in CE_tokens]
            PIDE_ngrams = [generate_ngrams(t, N_GRAM) for t in PIDE_tokens]

            E_ids = [self.v_E.get(ng, self.v_E.get(UNK, 0)) for nglist in E_ngrams for ng in nglist]
            RE_ids = [self.v_RE.get(ng, self.v_RE.get(UNK, 0)) for nglist in RE_ngrams for ng in nglist]
            CE_ids = [self.v_CE.get(ng, self.v_CE.get(UNK, 0)) for nglist in CE_ngrams for ng in nglist]
            PIDE_ids = [self.v_PIDE.get(ng, self.v_PIDE.get(UNK, 0)) for nglist in PIDE_ngrams for ng in nglist]

            E_ids    += [0]*(self.max_E_ng    - len(E_ids))
            RE_ids   += [0]*(self.max_RE_ng   - len(RE_ids))
            CE_ids   += [0]*(self.max_CE_ng   - len(CE_ids))
            PIDE_ids += [0]*(self.max_PIDE_ng - len(PIDE_ids))

            # índices de valores para embedding (mantém tokens)
            E_val_idxs    = [self.val_to_idx_E.get(t, 0) for t in E_tokens]
            RE_val_idxs   = [self.val_to_idx_RE.get(t, 0) for t in RE_tokens]
            CE_val_idxs   = [self.val_to_idx_CE.get(t, 0) for t in CE_tokens]
            PIDE_val_idxs = [self.val_to_idx_PIDE.get(t, 0) for t in PIDE_tokens]
            E_val_idxs    += [0]*(self.max_E    - len(E_val_idxs))
            RE_val_idxs   += [0]*(self.max_RE   - len(RE_val_idxs))
            CE_val_idxs   += [0]*(self.max_CE   - len(CE_val_idxs))
            PIDE_val_idxs += [0]*(self.max_PIDE - len(PIDE_val_idxs))

            # função para gerar valores, mães e posições
            def build_feats(tokens, maxlen):
                vals = [float(tok) for tok in tokens]
                moms = [int(tok.split(".",1)[0]) for tok in tokens]
                if vals:
                    min_v, max_v = min(vals), max(vals)
                    pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
                else:
                    pos = []
                pad = maxlen - len(tokens)
                vals += [0.0]*pad
                moms += [0]*pad
                pos  += [0.0]*pad
                return vals, moms, pos

            E_vals, E_moms, E_pos     = build_feats(E_tokens,    self.max_E)
            RE_vals, RE_moms, RE_pos  = build_feats(RE_tokens,   self.max_RE)
            CE_vals, CE_moms, CE_pos  = build_feats(CE_tokens,   self.max_CE)
            PI_vals, PI_moms, PI_pos  = build_feats(PIDE_tokens, self.max_PIDE)

            for s in b.get("saidas", []):
                # calcula pos_label = média dos valores dos tokens no bloco
                all_vals = []
                for tokens in [E_tokens, RE_tokens, CE_tokens, PIDE_tokens]:
                    all_vals.extend([float(t) for t in tokens])
                pos_label = sum(all_vals) / len(all_vals) if all_vals else 0.0

                y = {
                    "texto": b["saidas"][0]["textos"].index(s["textos"][0]),
                    "emoji": 0 if b["saidas"][0].get("reacao","") == s.get("reacao","") else 1,  # simplificar
                    "ctx":   0 if normalize(b["saidas"][0].get("contexto","")) == normalize(s.get("contexto","")) else 1,
                    "pos":   pos_label,
                }
                x = {
                    "E":      E_ids,    "E_val":  E_vals,  "E_mom":  E_moms,  "E_pos":  E_pos,  "E_val_idx": E_val_idxs,
                    "RE":     RE_ids,   "RE_val": RE_vals, "RE_mom": RE_moms, "RE_pos": RE_pos, "RE_val_idx": RE_val_idxs,
                    "CE":     CE_ids,   "CE_val": CE_vals, "CE_mom": CE_moms, "CE_pos": CE_pos, "CE_val_idx": CE_val_idxs,
                    "PIDE":   PIDE_ids, "PIDE_val":PI_vals, "PIDE_mom":PI_moms, "PIDE_pos":PI_pos, "PIDE_val_idx": PIDE_val_idxs,
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
            "E_val_idx": torch.tensor(x["E_val_idx"], dtype=torch.long),

            "RE":     torch.tensor(x["RE"],     dtype=torch.long),
            "RE_val": torch.tensor(x["RE_val"], dtype=torch.float32),
            "RE_mom": torch.tensor(x["RE_mom"], dtype=torch.long),
            "RE_pos": torch.tensor(x["RE_pos"], dtype=torch.long),
            "RE_val_idx": torch.tensor(x["RE_val_idx"], dtype=torch.long),

            "CE":     torch.tensor(x["CE"],     dtype=torch.long),
            "CE_val": torch.tensor(x["CE_val"], dtype=torch.float32),
            "CE_mom": torch.tensor(x["CE_mom"], dtype=torch.long),
            "CE_pos": torch.tensor(x["CE_pos"], dtype=torch.long),
            "CE_val_idx": torch.tensor(x["CE_val_idx"], dtype=torch.long),

            "PIDE":     torch.tensor(x["PIDE"],     dtype=torch.long),
            "PIDE_val": torch.tensor(x["PIDE_val"], dtype=torch.float32),
            "PIDE_mom": torch.tensor(x["PIDE_mom"], dtype=torch.long),
            "PIDE_pos": torch.tensor(x["PIDE_pos"], dtype=torch.long),
            "PIDE_val_idx": torch.tensor(x["PIDE_val_idx"], dtype=torch.long),
        }
        y_t = {
            "texto": torch.tensor(y["texto"], dtype=torch.long),
            "emoji": torch.tensor(y["emoji"], dtype=torch.long),
            "ctx":   torch.tensor(y["ctx"],   dtype=torch.long),
            "pos":   torch.tensor(y["pos"],   dtype=torch.float32),
        }
        return x_t, y_t

## INSEPA_MODEL
class AdamSegmentado(nn.Module):
    def __init__(self,
                 nE:int, nRE:int, nCE:int, nPIDE:int,
                 mom_size:int,
                 num_vals_E:int, num_vals_RE:int, num_vals_CE:int, num_vals_PIDE:int,
                 n_txt:int, n_emo:int, n_ctx:int,
                 max_E:int, max_RE:int, max_CE:int, max_PIDE:int, max_ng:int):
        super().__init__()
        # Embeddings por valor, separados por campo (treináveis)
        self.em_Eval = nn.Embedding(num_vals_E, EMBED_DIM)
        self.em_REval = nn.Embedding(num_vals_RE, EMBED_DIM)
        self.em_CEval = nn.Embedding(num_vals_CE, EMBED_DIM)
        self.em_PIDEval = nn.Embedding(num_vals_PIDE, EMBED_DIM)

        # Embeddings para tokens, mães e projeções de posição
        self.em_E = nn.Embedding(nE, EMBED_DIM)
        self.em_RE = nn.Embedding(nRE, EMBED_DIM)
        self.em_CE = nn.Embedding(nCE, EMBED_DIM)
        self.em_PIDE = nn.Embedding(nPIDE, EMBED_DIM)
        self.em_Emom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_REmom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_CEmom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_PIDEmom = nn.Embedding(mom_size, EMBED_DIM)
        self.proj_Epos = nn.Linear(1, EMBED_DIM)
        self.proj_REpos = nn.Linear(1, EMBED_DIM)
        self.proj_CEpos = nn.Linear(1, EMBED_DIM)
        self.proj_PIDEpos = nn.Linear(1, EMBED_DIM)

        self.max_E = max_E
        self.max_RE = max_RE
        self.max_CE = max_CE
        self.max_PIDE = max_PIDE
        self.max_ng = max_ng

        total = EMBED_DIM * 4
        self.fc1 = nn.Linear(total, HIDDEN_DIM)
        self.act = nn.ReLU()

        # Cabeças de saída
        self.h_txt = nn.Linear(HIDDEN_DIM, n_txt)
        self.h_emo = nn.Linear(HIDDEN_DIM, n_emo)
        self.h_ctx = nn.Linear(HIDDEN_DIM, n_ctx)
        self.h_pos = nn.Linear(HIDDEN_DIM, 1)  # Nova cabeça para posição (regressão)

    def forward(self, x: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        batch = x["E"].shape[0]
        # Campo E
        eE_tok = self.em_E(x["E"]).view(batch, self.max_E, self.max_ng, EMBED_DIM).mean(dim=2)
        eE_val = self.em_Eval(x["E_val_idx"])
        eE_mom = self.em_Emom(x["E_mom"])
        eE_pos = self.proj_Epos(x["E_val"].unsqueeze(-1))
        eE = (eE_tok + eE_val + eE_mom + eE_pos).mean(dim=1)
        # Campo RE
        eRE_tok = self.em_RE(x["RE"]).view(batch, self.max_RE, self.max_ng, EMBED_DIM).mean(dim=2)
        eRE_val = self.em_REval(x["RE_val_idx"])
        eRE_mom = self.em_REmom(x["RE_mom"])
        eRE_pos = self.proj_REpos(x["RE_val"].unsqueeze(-1))
        eRE = (eRE_tok + eRE_val + eRE_mom + eRE_pos).mean(dim=1)
        # Campo CE
        eCE_tok = self.em_CE(x["CE"]).view(batch, self.max_CE, self.max_ng, EMBED_DIM).mean(dim=2)
        eCE_val = self.em_CEval(x["CE_val_idx"])
        eCE_mom = self.em_CEmom(x["CE_mom"])
        eCE_pos = self.proj_CEpos(x["CE_val"].unsqueeze(-1))
        eCE = (eCE_tok + eCE_val + eCE_mom + eCE_pos).mean(dim=1)
        # Campo PIDE
        ePI_tok = self.em_PIDE(x["PIDE"]).view(batch, self.max_PIDE, self.max_ng, EMBED_DIM).mean(dim=2)
        ePI_val = self.em_PIDEval(x["PIDE_val_idx"])
        ePI_mom = self.em_PIDEmom(x["PIDE_mom"])
        ePI_pos = self.proj_PIDEpos(x["PIDE_val"].unsqueeze(-1))
        ePIDE = (ePI_tok + ePI_val + ePI_mom + ePI_pos).mean(dim=1)

        # Agrega e classifica
        h = torch.cat([eE, eRE, eCE, ePIDE], dim=1)
        h = self.act(self.fc1(h))
        return {
            "texto": self.h_txt(h),
            "emoji": self.h_emo(h),
            "ctx":   self.h_ctx(h),
            "pos":   self.h_pos(h),  # Posição como valor numérico
        }

## INSEPA_TRAIN
def train(memoria: dict, dominio: str) -> None:
    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)
    
    ds = InsepaFieldDataset(memoria, dominio)
    n  = len(ds)
    ckpt = ckpt_path(dominio)

    idxs = list(range(n))
    random.shuffle(idxs)
    vsz = min(max(1, int(0.2 * n)), n - 1)  # garantir pelo menos 1 para treino
    vidx, tidx = idxs[:vsz], idxs[vsz:]
    if not tidx:  # se tidx vazio, usar todos para treino, sem val
        tidx = idxs
        vidx = []
    train_ld = DataLoader(Subset(ds, tidx), batch_size=min(BATCH_SIZE, len(tidx)), shuffle=True)
    val_ld = DataLoader(Subset(ds, vidx), batch_size=min(BATCH_SIZE, len(vidx))) if vidx else None

    model = AdamSegmentado(
        nE=len(ds.v_E), nRE=len(ds.v_RE),
        nCE=len(ds.v_CE), nPIDE=len(ds.v_PIDE),
        mom_size=ds.mom_size,
        num_vals_E=len(ds.val_to_idx_E), num_vals_RE=len(ds.val_to_idx_RE),
        num_vals_CE=len(ds.val_to_idx_CE), num_vals_PIDE=len(ds.val_to_idx_PIDE),
        n_txt=ds.n_txt, n_emo=ds.n_emo,
        n_ctx=ds.n_ctx,
        max_E=ds.max_E, max_RE=ds.max_RE, max_CE=ds.max_CE, max_PIDE=ds.max_PIDE, max_ng=ds.max_ng
    )
    opt = optim.Adam(model.parameters(), lr=LR)
    ce  = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

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
                mse(out["pos"],  y["pos"])
            )
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        if val_ld:
            with torch.no_grad():
                for x, y in val_ld:
                    out = model(x)
                    val_loss += (
                        ce(out["texto"], y["texto"]).item() +
                        ce(out["emoji"], y["emoji"]).item() +
                        ce(out["ctx"],   y["ctx"]).item() +
                        mse(out["pos"],  y["pos"]).item()
                    )
            val_loss /= len(val_ld)
        else:
            val_loss = float("inf")  # sem validação, usar inf para não salvar

        if prev_val is None or val_loss < best:
            best, wait = val_loss, 0
            torch.save((
                model.state_dict(),
                ds.max_E, ds.max_RE, ds.max_CE, ds.max_PIDE,
                ds.mom_size, ds.val_to_idx_E, ds.val_to_idx_RE, ds.val_to_idx_CE, ds.val_to_idx_PIDE,
                ds.v_E, ds.v_RE, ds.v_CE, ds.v_PIDE,
                ds.n_txt, ds.n_emo, ds.n_ctx,
                ds.max_ng
            ), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        prev_val = val_loss

    print(f"✅ Treino concluído. best_val_loss={best:.4f}")

def infer(memoria: dict, dominio: str) -> None:
    """
    - <Enter> cicla variações de texto no bloco atual.
    - 'insight' exibe explicação manual ou insight automático.
    - Qualquer outro texto + emoji dispara inferência normal.
    - 'sair' encerra.
    """
    import os, torch, random
    # parse_text_reaction, normalize, ckpt_path, train, AdamSegmentado já disponíveis

    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        print("⚠️ Sem checkpoint — treine primeiro.")
        train(memoria, dominio)
        return

    (state,
     maxE, maxRE, maxCE, maxPIDE,
     mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
     vE, vRE, vCE, vPIDE,
     n_txt, n_emo, n_ctx,
     max_ng
    ) = torch.load(ckpt)

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE),
        nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        n_txt=n_txt, n_emo=n_emo,
        n_ctx=n_ctx,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    model.load_state_dict(state)
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][dominio]:
        bloco_num = int(inconsciente["INCO"][dominio]["Blocos"]["Bloco_id"])
        saida_vals = [float(key) for key in inconsciente["INCO"][dominio]["Blocos"].get("SAÍDA", {}).keys()]
        if saida_vals:
            ultimo_child_per_block[bloco_num] = max(saida_vals)
        else:
            ultimo_child_per_block[bloco_num] = 0.50
    prompt = "(Enter ↩ variação | insight | sair) ► "

    def featurize(field: str, bloco: dict, max_len: int, vocab: dict, val_to_idx: dict, max_ng: int):
        tokens = bloco["entrada"]["tokens"].get(field, [])
        ngrams_list = [generate_ngrams(t, N_GRAM) for t in tokens]
        ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
        val_idxs = [val_to_idx.get(t, 0) for t in tokens]
        vals = [float(t) for t in tokens]
        moms = [int(t.split(".",1)[0]) for t in tokens]
        if vals:
            min_v, max_v = min(vals), max(vals)
            pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
        else:
            pos = []
        pad_ids = (max_len * max_ng) - len(ids)
        pad_vals = max_len - len(tokens)
        ids += [0]*pad_ids
        val_idxs += [0]*pad_vals
        vals += [0.0]*pad_vals
        moms += [0]*pad_vals
        pos += [0.0]*pad_vals
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([val_idxs], dtype=torch.long),
            torch.tensor([vals], dtype=torch.float32),
            torch.tensor([moms], dtype=torch.long),
            torch.tensor([pos], dtype=torch.float32),
        )

    raw = None

    while True:
        # pede entrada+reação se raw for None
        if raw is None:
            raw = input("Entrada+Reação ► ").strip()
        cmd = raw.lower()

        if cmd == "sair":
            print("👋 Até mais!")
            return

        # parseia entrada+reação normal
        txt, reac = parse_text_reaction(raw, blocos)
        bloco = next(
            (b for b in blocos
             if normalize(b["entrada"]["texto"]) == normalize(txt)
                and b["entrada"].get("reacao","") == reac),
            None
        )
        if bloco is None:
            print("❌ Entrada+reação não cadastrada.")
            raw = None
            continue

        # dados para insight
        ep_txt  = bloco["entrada"]["texto"]
        ep_reac = bloco["entrada"].get("reacao","")
        contexto= bloco["saidas"][0].get("contexto","")
        emoji   = bloco["saidas"][0].get("reacao","")

        # preparo e forward
        max_val = ultimo_child_per_block.get(bloco["bloco_id"], 0.50)
        E_ids, E_val_idxs, E_val, E_mom, E_pos  = featurize("E",    bloco, maxE,   vE, val_to_idx_E, max_ng)
        RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos = featurize("RE",   bloco, maxRE,  vRE, val_to_idx_RE, max_ng)
        CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE",   bloco, maxCE,  vCE, val_to_idx_CE, max_ng)
        PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", bloco, maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

        x = {
            "E":    E_ids,    "E_val":  E_val,   "E_mom":  E_mom,   "E_pos":  E_pos,   "E_val_idx": E_val_idxs,
            "RE":   RE_ids,   "RE_val": RE_val,  "RE_mom": RE_mom,  "RE_pos": RE_pos,  "RE_val_idx": RE_val_idxs,
            "CE":   CE_ids,   "CE_val": CE_val,  "CE_mom": CE_mom,  "CE_pos": CE_pos,  "CE_val_idx": CE_val_idxs,
            "PIDE": PI_ids,   "PIDE_val":PI_val, "PIDE_mom":PI_mom, "PIDE_pos": PI_pos, "PIDE_val_idx": PI_val_idxs,
        }

        with torch.no_grad():
            out = model(x)

        texts     = bloco["saidas"][0]["textos"]
        variation = 0

        # exibe fala + emoji
        chosen = texts[variation]
        print(f"\n🤖 {chosen} {emoji}")

        # loop de variações e comandos
        while True:
            cmd2 = input(prompt).strip().lower()

            if cmd2 == "":
                # cicla variações de texto
                variation = (variation + 1) % len(texts)
                chosen   = texts[variation]
                print(f"\n🤖 {chosen} {emoji}")
                continue

            if cmd2 == "insight":
                print(
                    f"💡 De acordo com a expressão “{ep_txt}”, "
                    f"a reação “{ep_reac}” e o contexto “{contexto}”, "
                    f"conclui que “{chosen} {emoji}” é a resposta mais adequada."
                )
                continue

            if cmd2 == "sair":
                print("👋 Até mais!")
                return

            # qualquer outro texto é nova entrada+reação
            raw = cmd2
            break

        # volta ao topo para próxima inferência (raw definido ou None)

def test_model(memoria: dict, dominio: str) -> None:
    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        print("⚠️ Sem checkpoint — treine primeiro."); return

    (state,
     maxE, maxRE, maxCE, maxPIDE,
     mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
     vE, vRE, vCE, vPIDE,
     n_txt, n_emo, n_ctx,
     max_ng
    ) = torch.load(ckpt)

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE), nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        n_txt=n_txt, n_emo=n_emo,
        n_ctx=n_ctx,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    model.load_state_dict(state)
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][dominio]:
        bloco_num = int(inconsciente["INCO"][dominio]["Blocos"]["Bloco_id"])
        saida_vals = [float(key) for key in inconsciente["INCO"][dominio]["Blocos"].get("SAÍDA", {}).keys()]
        if saida_vals:
            ultimo_child_per_block[bloco_num] = max(saida_vals)
        else:
            ultimo_child_per_block[bloco_num] = 0.50
    print(f"📊 Teste em lote — Domínio {dominio} ({len(blocos)} blocos)")

    # Inicializar acumuladores para métricas
    total_samples = 0
    acc_txt = 0.0
    acc_emo = 0.0
    acc_ctx = 0.0
    mse_pos = 0.0

    for b in blocos:
        max_val = ultimo_child_per_block.get(b["bloco_id"], 0.50)
        def featurize(field, max_len, vocab, val_to_idx, max_ng):
            tokens = b["entrada"]["tokens"].get(field, [])
            ngrams_list = [generate_ngrams(t, N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
            val_idxs = [val_to_idx.get(t, 0) for t in tokens]
            vals = [float(t) for t in tokens]
            moms = [int(t.split(".",1)[0]) for t in tokens]
            if vals:
                min_v, max_v = min(vals), max(vals)
                pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
            else:
                pos = []
            pad_ids = (max_len * max_ng) - len(ids)
            pad_vals = max_len - len(tokens)
            ids += [0]*pad_ids
            val_idxs += [0]*pad_vals
            vals += [0.0]*pad_vals
            moms += [0]*pad_vals
            pos += [0.0]*pad_vals
            return (
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([val_idxs], dtype=torch.long),
                torch.tensor([vals], dtype=torch.float32),
                torch.tensor([moms], dtype=torch.long),
                torch.tensor([pos], dtype=torch.float32),
            )

        E_ids, E_val_idxs, E_val, E_mom, E_pos   = featurize("E",    maxE,   vE, val_to_idx_E, max_ng)
        RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos  = featurize("RE",   maxRE,  vRE, val_to_idx_RE, max_ng)
        CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE",   maxCE,  vCE, val_to_idx_CE, max_ng)
        PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

        x = {
            "E":    E_ids,   "E_val":  E_val,   "E_mom":  E_mom,   "E_pos":  E_pos,   "E_val_idx": E_val_idxs,
            "RE":   RE_ids,  "RE_val": RE_val,  "RE_mom": RE_mom,  "RE_pos": RE_pos,  "RE_val_idx": RE_val_idxs,
            "CE":   CE_ids,  "CE_val": CE_val,  "CE_mom": CE_mom,  "CE_pos": CE_pos,  "CE_val_idx": CE_val_idxs,
            "PIDE": PI_ids,  "PIDE_val":PI_val, "PIDE_mom":PI_mom, "PIDE_pos": PI_pos, "PIDE_val_idx": PI_val_idxs,
        }

        with torch.no_grad():
            out = model(x)

        # Calcular métricas para este bloco
        pred_txt = out["texto"].argmax(dim=1).item()
        pred_emo = out["emoji"].argmax(dim=1).item()
        pred_ctx = out["ctx"].argmax(dim=1).item()
        pred_pos = out["pos"].item()

        true_texts = [normalize(t) for t in b["saidas"][0]["textos"]]
        pred_text = b["saidas"][0]["textos"][pred_txt] if pred_txt < len(b["saidas"][0]["textos"]) else "N/A"
        true_emo = b["saidas"][0].get("reacao", "")
        true_ctx = b["saidas"][0].get("contexto", "")
        # Para pos, usar a média dos valores dos tokens do bloco
        all_vals = []
        for field in ["E", "RE", "CE", "PIDE"]:
            tokens = b["entrada"]["tokens"].get(field, [])
            all_vals.extend([float(t) for t in tokens])
        true_pos = sum(all_vals) / len(all_vals) if all_vals else 0.0

        # Acurácias (comparar índices)
        acc_txt_block = 1 if normalize(pred_text) in true_texts else 0
        acc_emo_block = 1 if pred_emo == 0 else 0  # 0 correto
        acc_ctx_block = 1 if pred_ctx == 0 else 0
        mse_pos_block = (pred_pos - true_pos) ** 2

        acc_txt += acc_txt_block
        acc_emo += acc_emo_block
        acc_ctx += acc_ctx_block
        mse_pos += mse_pos_block

        total_samples += 1

        # Coletar valores únicos no bloco
        block_vals = set()
        for field in ["E", "RE", "CE", "PIDE"]:
            block_vals.update(float(t) for t in b["entrada"]["tokens"].get(field, []) if t)

        print(f"\n❏ Bloco_id={b['bloco_id']} Entrada: {b['entrada']['texto']} {b['entrada']['reacao']}")
        print(f"   Texto pred: {pred_text} | True: {true_texts}")
        print(f"   Emoji pred: {true_emo if pred_emo == 0 else 'Outro'} | True: {true_emo}")
        print(f"   Contexto pred: {true_ctx if pred_ctx == 0 else 'Outro'} | True: {true_ctx}")
        print(f"   Posição pred: {pred_pos:.4f} | True: {true_pos:.4f}")
        print(f"   Acurácia Texto: {acc_txt_block:.1f}")
        print(f"   Acurácia Emoji: {acc_emo_block:.1f}")
        print(f"   Acurácia Contexto: {acc_ctx_block:.1f}")
        print(f"   MSE Posição: {mse_pos_block:.4f}")

    # Calcular médias
    if total_samples > 0:
        acc_txt /= total_samples
        acc_emo /= total_samples
        acc_ctx /= total_samples
        mse_pos /= total_samples

        print("\n📈 Métricas Gerais:")
        print(f"Acurácia Texto: {acc_txt:.2%}")
        print(f"Acurácia Emoji: {acc_emo:.2%}")
        print(f"Acurácia Contexto: {acc_ctx:.2%}")
        print(f"MSE Posição: {mse_pos:.4f}")
    else:
        print("Nenhum bloco para testar.")

## INSEPA_CLI
def menu_principal() -> str:
    print("\n=== Menu Principal ===")
    print("1) Gerenciar IMs (criar novo ou gerar bloco)")
    print("2) Treinar rede neural")
    print("3) Inferir com rede neural")
    print("4) Testar em lote")
    print("5) Sair do programa")
    return input("Escolha uma opção (1/2/3/4/5): ").strip()

def prompt_dominio(action: str, memoria: dict) -> str:
    """Lista IMs disponíveis e permite escolher um para a ação especificada."""
    ims = list(memoria.get("IM", {}).keys())
    if not ims:
        print("❌ Nenhum IM encontrado. Crie um primeiro.")
        return ""
    
    print(f"\n--- Escolher IM para {action} ---")
    print("IMs disponíveis:")
    for im_id in ims:
        nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
        num_blocos = len(memoria["IM"][im_id].get("blocos", []))
        print(f"- {im_id}: {nome} ({num_blocos} blocos)")
    
    while True:
        dom = input(f"→ Digite o ID do IM para {action} (ou 'sair' para voltar): ").strip()
        if dom.lower() == "sair":
            return ""
        if dom in ims:
            return dom
        print("❌ IM não encontrado. Tente novamente.")

def create_new_im(memoria: dict) -> None:
    """Cria um novo IM (Índice Mãe) vazio."""
    im_id = input("→ Índice mãe para o novo IM: ").strip()
    if not im_id.isdigit():
        print("❌ Índice mãe deve ser um número.")
        return
    im_id = int(im_id)
    if str(im_id) in memoria.get("IM", {}):
        print(f"❌ IM {im_id} já existe.")
        return
    nome = input("→ Nome do IM (opcional): ").strip() or f"IM_{im_id}"
    memoria.setdefault("IM", {})[str(im_id)] = {
        "nome": nome,
        "ultimo_child": f"{im_id}.0",
        "blocos": []
    }
    salvar_json(ARQUIVO_MEMORIA, memoria)
    print(f"✅ IM {im_id} criado: {nome}")

def submenu_im(memoria: dict) -> None:
    while True:
        print("\n--- Gerenciar IMs ---")
        print("1) Criar novo IM")
        print("2) Gerar bloco a partir de template INSEPA")
        print("3) Apagar bloco")
        print("4) Apagar IM")
        print("5) Alimentar vars dos tokens")
        print("6) Voltar ao menu principal")
        sub_opc = input("Escolha (1/2/3/4/5/6): ").strip()
        if sub_opc == "1":
            create_new_im(memoria)
        elif sub_opc == "2":
            # Listar IMs existentes
            ims = list(memoria.get("IM", {}).keys())
            if not ims:
                print("❌ Nenhum IM encontrado. Crie um primeiro.")
                continue
            print("IMs disponíveis:")
            for im_id in ims:
                nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
                print(f"- {im_id}: {nome}")
            im_escolhido = input("Digite o ID do IM: ").strip()
            if im_escolhido not in ims:
                print("❌ IM não encontrado.")
                continue
            print(f"Gerando bloco no IM {im_escolhido}...")
            print("Cole seu bloco template INSEPA (sem 'Índice mãe:') e pressione Enter em branco para finalizar:")
            lines: List[str] = []
            while True:
                line = input().rstrip()
                if not line:
                    break
                lines.append(line)
            # Adicionar Índice mãe ao template
            template_text = f"Índice mãe: {im_escolhido}\n" + "\n".join(lines)
            try:
                generate_block_from_template(memoria, template_text)
            except Exception as e:
                print(f"❌ Erro ao gerar bloco: {e}")
        elif sub_opc == "3":
            # Apagar bloco
            ims = list(memoria.get("IM", {}).keys())
            if not ims:
                print("❌ Nenhum IM encontrado.")
                continue
            print("IMs disponíveis:")
            for im_id in ims:
                nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
                print(f"- {im_id}: {nome}")
            im_escolhido = input("Digite o ID do IM: ").strip()
            if im_escolhido not in ims:
                print("❌ IM não encontrado.")
                continue
            universo = memoria["IM"][im_escolhido]
            blocos = universo.get("blocos", [])
            if not blocos:
                print("❌ Nenhum bloco neste IM.")
                continue
            print("Blocos no IM:")
            for b in blocos:
                bid = b["bloco_id"]
                texto = b["entrada"]["texto"]
                print(f"- ID {bid}: {texto}")
            bid_apagar = input("Digite o ID do bloco para apagar: ").strip()
            try:
                bid_int = int(bid_apagar)
                bloco = next((b for b in blocos if b["bloco_id"] == bid_int), None)
                if bloco:
                    blocos.remove(bloco)
                    # Renumerar blocos sequencialmente
                    for i, b in enumerate(blocos, 1):
                        b["bloco_id"] = i
                    # Recalcular ultimo_child
                    if blocos:
                        universo["ultimo_child"] = max(b["saidas"][0]["fim"] for b in blocos)
                    else:
                        universo["ultimo_child"] = f"{im_escolhido}.0"
                    salvar_json(ARQUIVO_MEMORIA, memoria)

                    # Remover do inconsciente.json
                    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
                    if im_escolhido in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][im_escolhido] and inconsciente["INCO"][im_escolhido]["Blocos"]["Bloco_id"] == str(bid_int):
                        del inconsciente["INCO"][im_escolhido]["Blocos"]
                    # Recalcular Ultimo child
                    if blocos:
                        saida_vals = []
                        for b in blocos:
                            if im_escolhido in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][im_escolhido]:
                                saida_vals.extend(float(k) for k in inconsciente["INCO"][im_escolhido]["Blocos"].get("SAÍDA", {}).keys())
                        if saida_vals:
                            inconsciente["INCO"][im_escolhido]["Ultimo child"] = str(max(saida_vals))
                        else:
                            inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                    else:
                        if im_escolhido in inconsciente.get("INCO", {}):
                            inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

                    print(f"✅ Bloco {bid_int} apagado. Blocos renumerados e ultimo_child ajustado.")
                else:
                    print("❌ Bloco não encontrado.")
            except ValueError:
                print("❌ ID inválido.")
        elif sub_opc == "4":
            # Apagar IM
            ims = list(memoria.get("IM", {}).keys())
            if not ims:
                print("❌ Nenhum IM encontrado.")
                continue
            print("IMs disponíveis:")
            for im_id in ims:
                nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
                print(f"- {im_id}: {nome}")
            im_apagar = input("Digite o ID do IM: ").strip()
            if im_apagar not in ims:
                print("❌ IM não encontrado.")
                continue
            confirm = input(f"Tem certeza que quer apagar o IM {im_apagar} e todos os seus blocos? (sim/não): ").strip().lower()
            if confirm == "sim":
                # Coletar blocos para remover do inconsciente
                blocos_a_remover = [f"Bloco_{b['bloco_id']}" for b in memoria["IM"][im_apagar]["blocos"]]
                del memoria["IM"][im_apagar]
                salvar_json(ARQUIVO_MEMORIA, memoria)

                # Remover do inconsciente.json
                inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
                if im_apagar in inconsciente.get("INCO", {}):
                    del inconsciente["INCO"][im_apagar]
                salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

                print(f"✅ IM {im_apagar} apagado.")
            else:
                print("Cancelado.")
        elif sub_opc == "5":
            # Alimentar vars dos tokens
            ims = list(memoria.get("IM", {}).keys())
            if not ims:
                print("❌ Nenhum IM encontrado.")
                continue
            print("IMs disponíveis:")
            for im_id in ims:
                nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
                print(f"- {im_id}: {nome}")
            im_escolhido = input("Digite o ID do IM: ").strip()
            if im_escolhido not in ims:
                print("❌ IM não encontrado.")
                continue
            inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
            if im_escolhido not in inconsciente.get("INCO", {}) or "Blocos" not in inconsciente["INCO"][im_escolhido]:
                print("❌ Nenhum bloco no inconsciente para este IM.")
                continue
            bloco = inconsciente["INCO"][im_escolhido]["Blocos"]
            print(f"Editando vars do bloco {bloco['Bloco_id']} no IM {im_escolhido}")
            print("Escolha o campo: 1) Entrada 2) SAÍDA")
            campo_opc = input("Escolha (1/2): ").strip()
            campo = "Entrada" if campo_opc == "1" else "SAÍDA" if campo_opc == "2" else None
            if not campo:
                print("❌ Opção inválida.")
                continue
            tokens = bloco.get(campo, {})
            if not tokens:
                print(f"❌ Nenhum token em {campo}.")
                continue
            print(f"Tokens em {campo}:")
            for marker, data in tokens.items():
                print(f"- {marker}: {data['token']} | vars: {data['vars']}")
            marker_edit = input("Digite o marcador do token para editar vars (ex: 0.1): ").strip()
            if marker_edit not in tokens:
                print("❌ Marcador não encontrado.")
                continue
            current_vars = tokens[marker_edit]["vars"]
            print(f"Vars atuais: {current_vars}")
            new_vars_str = input("Digite os novos vars separados por vírgula (ex: 0.1,0.2): ").strip()
            new_vars = [v.strip() for v in new_vars_str.split(",") if v.strip()]
            if not new_vars:
                print("❌ Vars inválidos.")
                continue
            tokens[marker_edit]["vars"] = new_vars
            salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
            print(f"✅ Vars atualizados para {marker_edit}: {new_vars}")
        elif sub_opc == "6":
            break
        else:
            print("❌ Opção inválida.")

def atualizar_inconsciente_para_im(memoria: dict, im_id: str) -> None:
    """Atualiza o inconsciente.json com os dados do IM especificado."""
    if im_id not in memoria["IM"]:
        print(f"❌ IM {im_id} não encontrado.")
        return
    
    universo = memoria["IM"][im_id]
    blocos = universo.get("blocos", [])
    
    if not blocos:
        print(f"❌ IM {im_id} não tem blocos.")
        return
    
    # Pega o último bloco criado (ou o primeiro se houver apenas um)
    bloco = blocos[-1]  # Último bloco
    
    # Carregar inconsciente atual
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    
    # Preparar dados para o IM específico
    im_data = {
        "NOME": universo["nome"],
        "Ultimo child": universo["ultimo_child"],
        "Blocos": {
            "Bloco_id": str(bloco["bloco_id"]),
            "Entrada": {},
            "SAÍDA": {}
        }
    }
    
    # Coletar todos os tokens da entrada
    entrada_tokens = []
    saida_tokens = []
    
    for field in ["E", "RE", "CE", "PIDE"]:
        tokens = bloco["entrada"]["tokens"].get(field, [])
        entrada_tokens.extend(tokens)
    
    for saida in bloco["saidas"]:
        for field in ["S", "RS", "CS"]:
            tokens = saida["tokens"].get(field, [])
            saida_tokens.extend(tokens)
    
    # Preencher tokens de entrada
    for marker in entrada_tokens:
        token_encontrado = None
        for field in ["E", "RE", "CE", "PIDE"]:
            tokens_field = bloco["entrada"]["tokens"].get(field, [])
            if marker in tokens_field:
                idx = tokens_field.index(marker)
                if field == "E":
                    token_encontrado = bloco["entrada"]["texto"].split()[idx] if idx < len(bloco["entrada"]["texto"].split()) else marker
                elif field == "RE":
                    token_encontrado = bloco["entrada"]["reacao"]
                elif field == "CE":
                    token_encontrado = bloco["entrada"]["contexto"].split()[idx] if idx < len(bloco["entrada"]["contexto"].split()) else marker
                elif field == "PIDE":
                    token_encontrado = marker
                break
        
        if token_encontrado:
            im_data["Blocos"]["Entrada"][marker] = {
                "token": token_encontrado,
                "vars": ["0.0"]
            }
    
    # Preencher tokens de saída
    for marker in saida_tokens:
        token_encontrado = None
        for saida in bloco["saidas"]:
            for field in ["S", "RS", "CS"]:
                tokens_field = saida["tokens"].get(field, [])
                if marker in tokens_field:
                    if field == "S":
                        all_words = []
                        for texto in saida["textos"]:
                            all_words.extend(texto.split())
                        if tokens_field.index(marker) < len(all_words):
                            token_encontrado = all_words[tokens_field.index(marker)]
                        else:
                            token_encontrado = marker
                    elif field == "RS":
                        token_encontrado = saida["reacao"]
                    elif field == "CS":
                        contexto_words = saida["contexto"].split()
                        idx = tokens_field.index(marker)
                        if idx < len(contexto_words):
                            token_encontrado = contexto_words[idx]
                        else:
                            token_encontrado = marker
                    break
            if token_encontrado:
                break
        
        if token_encontrado:
            im_data["Blocos"]["SAÍDA"][marker] = {
                "token": token_encontrado,
                "vars": ["0.0"]
            }
    
    # Atualizar apenas o IM específico no inconsciente
    inconsciente.setdefault("INCO", {})[im_id] = im_data
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
    print(f"✅ Inconsciente atualizado para IM {im_id} ({universo['nome']})")

def parse_template(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "indice_mae": None,
        "nome": "",
        "entrada": {"texto":"","reacao":"","contexto":"","pensamento_interno":""},
        "saida":   {"textos":[], "reacao":"","contexto":""}
    }
    # quebra linhas que tiveram vários campos na mesma linha
    expanded: List[str] = []
    for raw in lines:
        tmp = _re.sub(
            r'(Índice mãe:|Nome:|Entrada:|Reação:|Contexto:|Pensamento Interno:|Saída:|\d+\.)',
            r'\n\1',
            raw
        )
        expanded += [l.strip() for l in tmp.split('\n') if l.strip()]

    section: Optional[str] = None
    for line in expanded:

        # Índice mãe e Nome
        if line.startswith("Índice mãe:"):
            tpl["indice_mae"] = int(line.split(":",1)[1].strip())
            continue
        if line.startswith("Nome:"):
            tpl["nome"] = line.split(":",1)[1].strip()
            continue

        # Entrada inline
        m_ent = _re.match(r'^Entrada:\s*(.+)', line)
        if m_ent:
            tpl["entrada"]["texto"] = m_ent.group(1).strip()
            section = "entrada"
            continue

        # Seções
        if line.startswith("Entrada:"):
            section = "entrada"
            continue
        if _re.match(r'^sa[íi]da\s*:', line, _re.IGNORECASE):
            section = "saida_textos"
            continue

        # Campos de entrada
        if section == "entrada":
            if line.startswith("Texto:"):
                tpl["entrada"]["texto"] = line.split(":",1)[1].strip()
            elif _re.match(r'^(reacao|reação)\s*:', line, _re.IGNORECASE):
                tpl["entrada"]["reacao"] = line.split(":",1)[1].strip()
            elif _re.match(r'^contexto\s*:', line, _re.IGNORECASE):
                tpl["entrada"]["contexto"] = line.split(":",1)[1].strip()
            elif _re.match(r'^pensamento\s+interno\s*:', line, _re.IGNORECASE):
                tpl["entrada"]["pensamento_interno"] = line.split(":",1)[1].strip()

        # Linhas de saída
        if section == "saida_textos":
            m = _re.match(r'^\d+\.\s*(.+)', line)
            if m:
                tpl["saida"]["textos"].append(m.group(1).strip())
            else:
                section = "saida_meta"

        # Campos de meta-saída (parse sempre se section == "saida_meta")
        if section == "saida_meta":
            if _re.match(r'^(reacao|reação)\s*:', line, _re.IGNORECASE):
                tpl["saida"]["reacao"] = line.split(":",1)[1].strip()
            elif _re.match(r'^contexto\s*:', line, _re.IGNORECASE):
                tpl["saida"]["contexto"] = line.split(":",1)[1].strip()

    if tpl["indice_mae"] is None:
        raise ValueError("'Índice mãe' não encontrado no template.")
    return tpl

def generate_block_from_template(memoria: dict, template_text: str) -> None:
    """Gera bloco INSEPA a partir de template colado e adiciona ao adam_memoria.json."""
    lines = template_text.splitlines()
    tpl = parse_template(lines)
    mom = str(tpl["indice_mae"])
    universo = memoria["IM"].get(mom)

    # Cria IM se não existir
    if universo is None:
        universo = {
            "nome": tpl["nome"] or f"IM_{mom}",
            "ultimo_child": f"{mom}.0",
            "blocos": []
        }
        memoria["IM"][mom] = universo

    last = universo["ultimo_child"]

    # Tokenização
    E    = Token(tpl["entrada"]["texto"])
    RE   = [tpl["entrada"]["reacao"]] if tpl["entrada"]["reacao"] else []
    CE   = Token(tpl["entrada"]["contexto"])
    # Limpar e dividir pensamento_interno
    pensamento_limpo = tpl["entrada"]["pensamento_interno"].strip('"')
    tpl["entrada"]["pensamento_interno"] = pensamento_limpo  # Salvar limpo no adam_memoria
    partes = pensamento_limpo.split('.')[:3]  # Dividir em até 3 sequências por '.'
    PIDE_full = []
    for parte in partes:
        PIDE_full.extend(Token(parte.strip()))
    PIDE_limited = PIDE_full[:3]

    S: List[str] = []
    for t in tpl["saida"]["textos"]:
        S += Token(t)
    RS   = [tpl["saida"]["reacao"]] if tpl["saida"]["reacao"] else []
    CS   = Token(tpl["saida"]["contexto"])

    total_ent = len(E) + len(RE) + len(CE) + len(PIDE_limited)
    total_out = len(S) + len(RS) + len(CS)

    markers    = generate_markers(last, total_ent + total_out)
    ent_marks  = markers[:total_ent]
    out_marks  = markers[total_ent:] 

    fim_ent = ent_marks[-1] if ent_marks else last
    fim_out = out_marks[-1] if out_marks else fim_ent

    # Subdivide
    idx  = 0
    E_m   = ent_marks[idx: idx+len(E)];    idx += len(E)
    RE_m  = ent_marks[idx: idx+len(RE)];   idx += len(RE)
    CE_m  = ent_marks[idx: idx+len(CE)];   idx += len(CE)
    PIDE_m= ent_marks[idx: idx+len(PIDE_limited)]

    jdx  = 0
    S_m   = out_marks[jdx: jdx+len(S)];    jdx += len(S)
    RS_m  = out_marks[jdx: jdx+len(RS)];   jdx += len(RS)
    CS_m  = out_marks[jdx: jdx+len(CS)]

    next_id = max((b["bloco_id"] for b in universo["blocos"]), default=0) + 1

    new_block: Dict[str, Any] = {
        "bloco_id": next_id,
        "entrada": {
            **tpl["entrada"],
            "tokens": {
                "E":    E_m,
                "RE":   RE_m,
                "CE":   CE_m,
                "PIDE": PIDE_m,
                "TOTAL": ent_marks
            },
            "fim": fim_ent,
            "alnulu": len(tpl["entrada"]["texto"])
        },
        "saidas": [{
            **tpl["saida"],
            "tokens": {
                "S":     S_m,
                "RS":    RS_m,
                "CS":    CS_m,
                "TOTAL": out_marks
            },
            "fim":    fim_out
        }],
        "open": True
    }

    universo["blocos"].append(new_block)
    universo["ultimo_child"] = fim_out

    salvar_json(ARQUIVO_MEMORIA, memoria)

    # Atualizar inconsciente.json com o novo bloco
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    all_ent_tokens = E + RE + CE + PIDE_full
    all_out_tokens = S + RS + CS
    ent_marks_inco = ent_marks[:]
    if len(PIDE_full) > 3:
        extra_count = len(PIDE_full) - 3
        extra_marks = generate_markers(ent_marks[-1], extra_count)
        ent_marks_inco.extend(extra_marks)
    
    # Preparar dados para o IM específico
    im_data = {
        "NOME": universo["nome"],
        "Ultimo child": fim_out,
        "Blocos": {
            "Bloco_id": str(next_id),
            "Entrada": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(ent_marks_inco, all_ent_tokens)},
            "SAÍDA": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(out_marks, all_out_tokens)}
        }
    }
    
    # Atualizar apenas o IM específico no inconsciente
    inconsciente.setdefault("INCO", {})[mom] = im_data
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

    print(f"✅ Bloco adicionado ao domínio {mom}. Último marker: {fim_out}")

def main():
    memoria      = carregar_json(ARQUIVO_MEMORIA,      {"IM": {}})
    inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"conteudos": []})

    while True:
        opc = menu_principal()
        if opc == "1":
            submenu_im(memoria)
        elif opc == "2":
            dom = prompt_dominio("treinar", memoria)
            if not dom: continue
            if dom in memoria["IM"]:
                train(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")
        elif opc == "3":
            dom = prompt_dominio("inferir", memoria)
            if not dom: continue
            if dom in memoria["IM"]:
                infer(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")
        elif opc == "4":
            dom = prompt_dominio("testar", memoria)
            if not dom: continue
            if dom in memoria["IM"]:
                test_model(memoria, dom)
            else:
                print(f"❌ Domínio '{dom}' não encontrado.")
        elif opc == "5":
            print("👋 Até mais!")
            break
        else:
            print("❌ Opção inválida. Digite 1, 2, 3, 4 ou 5.")

if __name__ == "__main__":
    main()
