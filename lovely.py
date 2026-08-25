#!/usr/bin/env python3
# -*- coding: utf-8 -*-.

import os
import shutil
import sys
import json
import random
import re as _re
import hashlib
import uuid
import subprocess
import copy
from typing import List, Dict, Tuple, Any, Optional
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import streamlit as st

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

def _start_mongo_service_if_available() -> None:
    """Tenta iniciar o serviço MongoDB no Windows quando estiver instalado mas parado."""
    if os.name != "nt":
        return
    try:
        result = subprocess.run(["sc", "query", "MongoDB"], capture_output=True, text=True)
        output = result.stdout + result.stderr
        if "STATE" in output and "STOPPED" in output:
            subprocess.run(["sc", "start", "MongoDB"], capture_output=True, text=True)
    except Exception:
        pass

# Opcional: persistência em MongoDB em vez de JSON local.
# Configure MONGO_URI e MONGO_DB por variável de ambiente (local) OU por
# Secrets do Streamlit Cloud (deploy) -- variável de ambiente sempre vence se
# as duas existirem, pra não surpreender quem já testa localmente.
def _mongo_config(chave: str, padrao: str) -> str:
    if os.environ.get(chave):
        return os.environ[chave]
    try:
        if chave in st.secrets:
            return st.secrets[chave]
    except Exception:
        pass  # sem secrets.toml (normal rodando só localmente) -- segue pro padrão
    return padrao


try:
    from pymongo import MongoClient
    from gridfs import GridFS
    MONGO_URI = _mongo_config("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = _mongo_config("MONGO_DB", "adam_lovely")
    _mongo_enabled = False
    _mongo_error = None

    # On Windows, try to start MongoDB service if installed but not started.
    _start_mongo_service_if_available()

    # Tenta conectar ao Mongo; se falhar, desativa silenciosamente
    try:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        # Força conexão para capturar erros de rede/auth
        _mongo_client.admin.command('ping')
        _mongo_db = _mongo_client[MONGO_DB]
        _mongo_fs = GridFS(_mongo_db)
        _mongo_enabled = True
    except Exception as e:
        _mongo_error = str(e)
except ImportError:
    _mongo_enabled = False
    _mongo_error = "pymongo não instalado"


try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

import sys
import time

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE ARQUIVOS E CONSTANTES
# ────────────────────────────────────────────────────────────────────────────────

ARQUIVO_MEMORIA = "Adam_Lovely_memory.json"
ARQUIVO_INCONSCIENTE = "Adam_Lovely_unconscious.json"
ARQUIVO_OPINIOES = "Adam_Lovely_opinioes.json"  # opiniões (texto+emoção) pendentes de revisão da criadora
EMBED_DIM = 64
HIDDEN_DIM = 64
PATIENCE = 5
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
UNK = "<UNK>"
UNK_VAL = -1.0
N_GRAM = 8  # Tamanho do n-grama (8 para 8-grams)
CORPUS_SIMILARITY_THRESHOLD = 0.70  # Só persiste blocos aprendidos autonomamente com >= 70% de fidelidade ao corpus
RAIZ_COERENCIA_THRESHOLD = 0.15  # Contexto da saída do bloco precisa ter ao menos essa afinidade com o contexto buscado (mesma raiz)
IMITACAO_ASSUNTO_MINIMO = 0.70  # Abaixo disso, o assunto é considerado divergente e não pode emprestar contexto/pensamento

SENHA_ADMIN = "ADAM123"  # Senha para acesso total (Gerenciar IMs + painel completo)
SENHA_CRIAR_BLOCOS = "TMADAM123"  # Senha para criar blocos via Cérbero (sem acesso ao painel)


## INSEPA_TOKENIZER
def generate_ngrams(token: str, n: int) -> List[str]:
    """Gera n-gramas de caracteres de um token."""
    if len(token) < n:
        return [token]  # Se menor que n, retorna o token inteiro
    return [token[i:i + n] for i in range(len(token) - n + 1)]


def ckpt_path(dominio: str) -> str:
    return f"insepa_{dominio}.pt"


def ensure_checkpoint_exists(dominio: str) -> bool:
    """Garante que o checkpoint existe, carregando do Mongo se não estiver em disco."""
    ckpt = ckpt_path(dominio)
    if os.path.exists(ckpt):
        return True
    
    # Tentar carregar do Mongo
    checkpoint_bytes = _mongo_load_checkpoint(dominio)
    if checkpoint_bytes:
        try:
            with open(ckpt, "wb") as f:
                f.write(checkpoint_bytes)
            return True
        except Exception as e:
            print(f"⚠️ Erro ao restaurar checkpoint do Mongo: {e}")
    
    return False


def Token_with_vars(text: str) -> List[str]:
    """INSEPA tokenização mantendo [vars: ...] intactos."""
    import re
    
    # Primeiro, encontrar todos os [vars: ...] e substituir temporariamente
    vars_pattern = r'\[vars:[^\]]*\]'
    vars_placeholders = []
    var_counter = 0
    
    def replace_vars(match):
        nonlocal var_counter
        placeholder = f"__VARS_PLACEHOLDER_{var_counter}__"
        vars_placeholders.append((placeholder, match.group(0)))
        var_counter += 1
        return placeholder
    
    # Substituir [vars: ...] por placeholders
    text_with_placeholders = _re.sub(vars_pattern, replace_vars, text)
    
    # Tokenizar normalmente
    tokens = _re.findall(r'\w+|[^\w\s]', text_with_placeholders, _re.UNICODE)
    
    # Restaurar os [vars: ...] DENTRO dos tokens (não apenas tokens que são placeholders inteiros)
    for placeholder, original in vars_placeholders:
        tokens = [t.replace(placeholder, original) for t in tokens]
    
    return tokens


def Token(text: str) -> List[str]:
    """INSEPA tokenização básica sem preservar vars."""
    return _re.findall(r'\w+|[^\w\s]', text, _re.UNICODE)


_PONTUACAO_FRASE = set('.,!?;:')


def split_frases_por_pontuacao(texto: str) -> List[str]:
    """Quebra um texto em frases nos limites de pontuação (o INSEPA já trata cada
    pontuação como um token próprio, então o limite é natural).

    Só frases inteiras devem ser usadas para casar 'pensamento' — uma frase
    carrega seu contexto junto; uma palavra solta do léxico inconsciente pode
    ser reaproveitada num contexto errado.
    """
    if not texto:
        return []
    frases: List[str] = []
    atual: List[str] = []
    for tok in Token(texto):
        if tok in _PONTUACAO_FRASE:
            if atual:
                frases.append(' '.join(atual))
                atual = []
        else:
            atual.append(tok)
    if atual:
        frases.append(' '.join(atual))
    return [f for f in frases if f.strip()]


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
def _mongo_load(key: str, default: dict) -> dict:
    """Carrega estado do MongoDB (coleção única).

    O documento é salvo com _id="singleton" e todos os campos do estado são
    armazenados no nível superior (mais prático para inspeção no MongoDB).
    """
    if not _mongo_enabled:
        return default
    col = _mongo_db[key]
    doc = col.find_one({"_id": "singleton"})
    if not doc:
        doc = {"_id": "singleton"}
        doc.update(default)
        col.replace_one({"_id": "singleton"}, doc, upsert=True)
        return default
    # Remover _id antes de voltar
    return {k: v for k, v in doc.items() if k != "_id"}


def _mongo_save(key: str, data: dict) -> None:
    """Salva estado no MongoDB (coleção única)."""
    if not _mongo_enabled:
        return
    # Não armazenamos a payload dentro de "data" para manter o JSON tokenizado
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict) and len(data) == 1:
        data = data["data"]

    col = _mongo_db[key]
    doc = {"_id": "singleton"}
    if isinstance(data, dict):
        doc.update(data)
    else:
        # Caso incomum: armazenar em campo genérico
        doc["data"] = data
    col.replace_one({"_id": "singleton"}, doc, upsert=True)


def _mongo_save_checkpoint(dominio: str, checkpoint_data: bytes) -> None:
    """Salva checkpoint (arquivo .pt) no MongoDB usando GridFS."""
    if not _mongo_enabled:
        return
    try:
        filename = f"checkpoint_{dominio}.pt"
        # Remove arquivo antigo se existir
        old_file = _mongo_db.fs.files.find_one({"filename": filename})
        if old_file:
            _mongo_fs.delete(old_file["_id"])
        # Salva novo checkpoint
        _mongo_fs.put(checkpoint_data, filename=filename)
    except Exception as e:
        print(f"⚠️ Erro ao salvar checkpoint em Mongo: {e}")


def _mongo_load_checkpoint(dominio: str) -> Optional[bytes]:
    """Carrega checkpoint (arquivo .pt) do MongoDB usando GridFS."""
    if not _mongo_enabled:
        return None
    try:
        filename = f"checkpoint_{dominio}.pt"
        grid_out = _mongo_fs.find_one({"filename": filename})
        if grid_out:
            return grid_out.read()
    except Exception as e:
        print(f"⚠️ Erro ao carregar checkpoint do Mongo: {e}")
    return None


def carregar_json(caminho: str, default: dict) -> dict:
    """Carrega estado do disco (JSON) ou do Mongo (se habilitado)."""
    def _unwrap(d: dict) -> dict:
        # Algumas versões anteriores armazenavam a payload dentro de uma chave "data".
        # Isso garante compatibilidade para não quebrar o formato esperado pelo app.
        if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict) and len(d) == 1:
            return d["data"]
        return d

    mongo_keys = {ARQUIVO_MEMORIA: "memoria", ARQUIVO_INCONSCIENTE: "inconsciente", ARQUIVO_OPINIOES: "opinioes"}
    if _mongo_enabled and caminho in mongo_keys:
        data = _mongo_load(mongo_keys[caminho], default)
        data = _unwrap(data)
        if caminho == ARQUIVO_MEMORIA:
            st.session_state.memoria = data
        elif caminho == ARQUIVO_INCONSCIENTE:
            st.session_state.inconsciente = data
        return data

    if not os.path.exists(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _unwrap(data)
    # Sempre atualizar session_state
    if caminho == ARQUIVO_MEMORIA:
        st.session_state.memoria = data
    elif caminho == ARQUIVO_INCONSCIENTE:
        st.session_state.inconsciente = data
    return data


def backup_json(caminho: str) -> None:
    """Faz um backup do JSON antes de sobrescrever.

    Os backups ficam em `backup/` com timestamp. Isso evita perda de dados
    caso algo grave aconteça durante a gravação.
    """
    if not os.path.exists(caminho):
        return
    try:
        os.makedirs("backup", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(caminho)
        backup_path = os.path.join("backup", f"{base}.{timestamp}.bak")
        shutil.copy2(caminho, backup_path)
    except Exception:
        # Não falhar a gravação por conta do backup
        pass


def salvar_json(caminho: str, data: dict) -> None:
    """Salva estado no disco (JSON) e no Mongo (se habilitado)."""
    if caminho == ARQUIVO_MEMORIA:
        st.session_state.memoria = data
    elif caminho == ARQUIVO_INCONSCIENTE:
        st.session_state.inconsciente = data

    mongo_keys = {ARQUIVO_MEMORIA: "memoria", ARQUIVO_INCONSCIENTE: "inconsciente", ARQUIVO_OPINIOES: "opinioes"}
    if _mongo_enabled and caminho in mongo_keys:
        _mongo_save(mongo_keys[caminho], data)

    # Backup antes de sobrescrever (evitar perda acidental)
    backup_json(caminho)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dict_hash(data: dict) -> str:
    """Retorna hash determinístico de um dict para detectar alterações."""
    dumped = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(dumped.encode('utf-8')).hexdigest()


def auto_save_state() -> None:
    """Salva automaticamente memória e inconsciente em disco quando habilitado.

    O Streamlit rerun é acionado em cada interação, então essa função garante que
    os dados em `st.session_state` sejam persistidos nos JSONs em cada rerun.

    Essa função evita gravações redundantes comparando hashes do estado.
    """
    if not st.session_state.get("auto_save", True):
        return

    memoria = st.session_state.get("memoria", {})
    inconsciente = st.session_state.get("inconsciente", {})

    # Somente salvar se houve mudança real
    mem_hash = dict_hash(memoria)
    inco_hash = dict_hash(inconsciente)

    if st.session_state.get("_last_mem_hash") != mem_hash:
        salvar_json(ARQUIVO_MEMORIA, memoria)
        st.session_state["_last_mem_hash"] = mem_hash

    if st.session_state.get("_last_inco_hash") != inco_hash:
        salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
        st.session_state["_last_inco_hash"] = inco_hash


def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else "")


def normalize_collapse_spaces(txt: str) -> str:
    return _re.sub(r'\s+', ' ', txt).strip()


def normalize_separators(txt: str) -> str:
    return _re.sub(r'\s*([,.;:])\s*', '', txt).strip()


def normalize(txt: str) -> str:
    for fn in (normalize_collapse_spaces, normalize_separators):
        txt = fn(txt)
    return txt.lower()


def variar_texto(texto: str, bloco: dict, dominio: str, tipo: str = 'saida', inconsciente: dict = None) -> str:
    """Varia o texto substituindo tokens por suas variações aleatórias baseadas nas vars do inconsciente, evitando repetições de palavras já usadas."""
    if bloco is None:
        return texto
    if inconsciente is None:
        inconsciente = st.session_state.inconsciente
    tokens = Token(texto)
    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return texto
    campo = 'Entrada' if tipo == 'entrada' else 'SAÍDA'
    variado = []
    for tok in tokens:
        marcador = None
        for m, data in bloco_inco[campo].items():
            if data["token"] == tok:
                marcador = m
                break
        if marcador:
            valid_vars = [v for v in data["vars"] if v != "0.0"]
            all_options = valid_vars + ([tok] if tok not in variado else [])
            if all_options:
                attempts = 0
                while attempts < 10:
                    chosen_var = random.choice(all_options)
                    if chosen_var not in variado:
                        break
                    attempts += 1
                else:
                    chosen_var = tok  # fallback to tok even if repeated, to avoid breaking text
            else:
                chosen_var = tok
        else:
            chosen_var = tok
        variado.append(chosen_var)
    return ' '.join(variado)


def variar_texto_rag(bloco, dominio, variations_from_blocks):
    inconsciente = st.session_state.inconsciente
    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return "Erro: bloco não encontrado no inconsciente."
    # Coletar vars inconscientes
    unconscious_vars = {}
    for data in bloco_inco["Entrada"].values():
        token = data["token"]
        if isinstance(token, list):
            token = str(token)
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            unconscious_vars[token] = vars_list
    for data in bloco_inco["SAÍDA"].values():
        token = data["token"]
        if isinstance(token, list):
            token = str(token)
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            unconscious_vars[token] = vars_list
    # Agora, para cada variation in variations_from_blocks, aplicar variações
    varied_texts = []
    for variation in variations_from_blocks:
        varied = variation
        for token, vars_list in unconscious_vars.items():
            if token in variation:
                # Escolher uma var aleatória
                chosen_var = random.choice(vars_list)
                varied = varied.replace(token, chosen_var)
        varied_texts.append(varied)
    # Escolher uma das varied_texts
    chosen = random.choice(varied_texts) if varied_texts else "Resposta variada vazia."
    return chosen


def get_variations_for_tokens(im_id: str, bloco_id: int, campo: str, markers: List[str]) -> List[str]:
    """Obtém variações de tokens para marcadores específicos."""
    inconsciente = st.session_state.inconsciente
    bloco_inco = next((b for b in inconsciente["INCO"][im_id]["Blocos"] if b["Bloco_id"] == str(bloco_id)), None)
    if bloco_inco:
        variations = set()
        for marker in markers:
            if marker in bloco_inco[campo]:
                data = bloco_inco[campo][marker]
                variations.add(normalize(data["token"]))
                for var in data.get("vars", []):
                    variations.add(normalize(var))
        return list(variations)
    return []





def build_field_vocabs(memoria: dict, dominio: str) -> Dict[str, Dict[str, int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets = {"E": set(), "RE": set(), "CE": set(), "PIDE": set()}
    for b in blocos:
        t = b["entrada"]["tokens"]
        for f in sets:
            for tok in t.get(f, []):
                ngrams = generate_ngrams(tok, N_GRAM)
                sets[f].update(ngrams)
    return {
        f: {ng: i + 1 for i, ng in enumerate(sorted(sets[f]))}
        for f in sets
    }


def build_label_vocabs(memoria: dict, dominio: str) -> Dict[str, Dict[str, int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets = {"texto": set(), "emoji": set(), "ctx": set()}
    for b in blocos:
        for s in b.get("saidas", []):
            for v in s.get("textos", []):
                sets["texto"].add(normalize(v))
            emo = s.get("reacao", "")
            if emo:   sets["emoji"].add(emo)
            ctx = s.get("contexto", "")
            if ctx:   sets["ctx"].add(normalize(ctx))
    return {
        f: {tok: i for i, tok in enumerate(sorted(sets[f]))}
        for f in sets
    }


## INSEPA_DATASET
class InsepaFieldDataset(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["IM"][dominio]["blocos"]
        inconsciente = st.session_state.inconsciente
        self.ultimo_child_per_block = {}
        if dominio in inconsciente.get("INCO", {}):
            blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
            for bloco in blocos_inco:
                bloco_num = int(bloco["Bloco_id"])
                saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
                if saida_vals:
                    self.ultimo_child_per_block[bloco_num] = max(saida_vals)
                else:
                    self.ultimo_child_per_block[bloco_num] = 0.50

        # Coletar PALAVRAS únicas por campo (nunca marcadores) -- o modelo aprende
        # a forma da palavra via ALNULU, não a posição dela. Camada 1 garante que
        # marcador nunca se repete, então n-gramar marcador nunca generalizaria.
        sets = {"E": set(), "RE": set(), "CE": set(), "PIDE": set()}
        for b in blocos:
            for f in sets:
                sets[f] |= set(palavras_do_campo(b, f))

        # Palavras diferentes podem colidir na mesma string ALNULU (ex.: emojis,
        # que a tabela do ALNULU não cobre e todos viram zero) -- por isso o
        # vocabulário nasce das strings ALNULU já deduplicadas, nunca enumerando
        # a lista de palavras originais direto (senão o índice passa do tamanho
        # real do dicionário e o embedding estoura).
        def vocab_de_formas(palavras: set) -> dict:
            formas = sorted({alnulu_string(p) for p in palavras})
            v = {forma: i + 1 for i, forma in enumerate(formas)}
            v[UNK] = len(v)
            return v

        self.v_E = vocab_de_formas(sets["E"])
        self.v_RE = vocab_de_formas(sets["RE"])
        self.v_CE = vocab_de_formas(sets["CE"])
        self.v_PIDE = vocab_de_formas(sets["PIDE"])

        # val_to_idx por campo: a PALAVRA em si (embedding por palavra exata),
        # complementar ao n-grama de forma acima (que generaliza pra parecidas).
        self.val_to_idx_E = {tok: i for i, tok in enumerate(sorted(sets["E"]))}
        self.val_to_idx_RE = {tok: i for i, tok in enumerate(sorted(sets["RE"]))}
        self.val_to_idx_CE = {tok: i for i, tok in enumerate(sorted(sets["CE"]))}
        self.val_to_idx_PIDE = {tok: i for i, tok in enumerate(sorted(sets["PIDE"]))}

        self.max_E = max((len(palavras_do_campo(b, "E")) for b in blocos), default=1) or 1
        self.max_RE = max((len(palavras_do_campo(b, "RE")) for b in blocos), default=1) or 1
        self.max_CE = max((len(palavras_do_campo(b, "CE")) for b in blocos), default=1) or 1
        self.max_PIDE = max((len(palavras_do_campo(b, "PIDE")) for b in blocos), default=1) or 1
        self.max_pos = max(self.max_E, self.max_RE, self.max_CE, self.max_PIDE)

        # Calcular max n-gramas por palavra (agora sobre a string ALNULU da palavra)
        all_tokens = set()
        for b in blocos:
            for field in ["E", "RE", "CE", "PIDE"]:
                all_tokens |= set(palavras_do_campo(b, field))
        all_alnulu_strs = {alnulu_string(t) for t in all_tokens if t}
        self.max_ng = max((len(generate_ngrams(s, N_GRAM)) for s in all_alnulu_strs), default=1) or 1
        self.max_E_ng = self.max_E * self.max_ng
        self.max_RE_ng = self.max_RE * self.max_ng
        self.max_CE_ng = self.max_CE * self.max_ng
        self.max_PIDE_ng = self.max_PIDE * self.max_ng

        # calcula mom_size = maior mãe + 1
        max_mom = 0
        for b in blocos:
            for tok in b["entrada"]["tokens"].get("TOTAL", []):
                m = int(tok.split(".", 1)[0])
                if m > max_mom: max_mom = m
        self.mom_size = max_mom + 1

        # (self.val_to_idx/self.num_vals removidos: eram um resquício não usado por
        # nada em forward(), e assumiam que os tokens eram números de marcador --
        # agora são palavras, então float(palavra) quebrava sem necessidade.)

        # vocabulários de rótulos por bloco
        self.max_S = max(len(b["saidas"][0]["tokens"].get("S", [])) for b in blocos)
        self.max_RS = max(len(b["saidas"][0]["tokens"].get("RS", [])) for b in blocos)
        self.max_CS = max(len(b["saidas"][0]["tokens"].get("CS", [])) for b in blocos)
        self.max_out_len = max(len(b["saidas"][0]["tokens"].get("TOTAL", [])) for b in blocos) if blocos else 1

        # Vocabulário de marcadores de saída
        all_out_markers = set()
        for b in blocos:
            all_out_markers.update(b["saidas"][0]["tokens"].get("TOTAL", []))
        self.all_out_markers = list(all_out_markers)
        
        # Detectar formato dos marcadores
        if all_out_markers and all(len(m.split()) == 1 for m in all_out_markers):
            # Checkpoint antigo: apenas floats, recriar vocabulário dos blocos
            self.out_vocab = {}
            for b in blocos:
                for saida in b["saidas"]:
                    for texto in saida["textos"]:
                        for token in Token(texto):
                            if token not in self.out_vocab:
                                self.out_vocab[token] = len(self.out_vocab) * 0.001
                    reac = saida.get("reacao", "")
                    if reac and reac not in self.out_vocab:
                        self.out_vocab[reac] = len(self.out_vocab) * 0.001
                    ctx = saida.get("contexto", "")
                    for token in Token(ctx):
                        if token not in self.out_vocab:
                            self.out_vocab[token] = len(self.out_vocab) * 0.001
            self.idx_to_txt = {v: k for k, v in self.out_vocab.items()}
        else:
            # Novo formato: "float word"
            self.out_vocab = {m.split()[1]: float(m.split()[0]) for m in all_out_markers}
            self.idx_to_txt = {float(m.split()[0]): m.split()[1] for m in all_out_markers}
        
        self.pad_token = "<PAD>"
        self.out_vocab[self.pad_token] = -1.0

        dominio_int = int(dominio) if str(dominio).isdigit() else 0

        self.pares: List[Tuple[Dict, Dict]] = []
        for b in blocos:
            # Palavras de verdade (nunca marcadores) -- é a FORMA delas (via ALNULU)
            # que o modelo aprende a reconhecer.
            E_tokens = palavras_do_campo(b, "E")
            RE_tokens = palavras_do_campo(b, "RE")
            CE_tokens = palavras_do_campo(b, "CE")
            PIDE_tokens = palavras_do_campo(b, "PIDE")

            # Gerar n-gramas e ids a partir da string ALNULU de cada palavra
            E_ngrams = [generate_ngrams(alnulu_string(t), N_GRAM) for t in E_tokens]
            RE_ngrams = [generate_ngrams(alnulu_string(t), N_GRAM) for t in RE_tokens]
            CE_ngrams = [generate_ngrams(alnulu_string(t), N_GRAM) for t in CE_tokens]
            PIDE_ngrams = [generate_ngrams(alnulu_string(t), N_GRAM) for t in PIDE_tokens]

            E_ids = [self.v_E.get(ng, self.v_E.get(UNK, 0)) for nglist in E_ngrams for ng in nglist]
            RE_ids = [self.v_RE.get(ng, self.v_RE.get(UNK, 0)) for nglist in RE_ngrams for ng in nglist]
            CE_ids = [self.v_CE.get(ng, self.v_CE.get(UNK, 0)) for nglist in CE_ngrams for ng in nglist]
            PIDE_ids = [self.v_PIDE.get(ng, self.v_PIDE.get(UNK, 0)) for nglist in PIDE_ngrams for ng in nglist]

            E_ids += [0] * (self.max_E_ng - len(E_ids))
            RE_ids += [0] * (self.max_RE_ng - len(RE_ids))
            CE_ids += [0] * (self.max_CE_ng - len(CE_ids))
            PIDE_ids += [0] * (self.max_PIDE_ng - len(PIDE_ids))

            # índices de valores para embedding (pela palavra exata, não pelo marcador)
            E_val_idxs = [self.val_to_idx_E.get(t, 0) for t in E_tokens]
            RE_val_idxs = [self.val_to_idx_RE.get(t, 0) for t in RE_tokens]
            CE_val_idxs = [self.val_to_idx_CE.get(t, 0) for t in CE_tokens]
            PIDE_val_idxs = [self.val_to_idx_PIDE.get(t, 0) for t in PIDE_tokens]
            E_val_idxs += [0] * (self.max_E - len(E_val_idxs))
            RE_val_idxs += [0] * (self.max_RE - len(RE_val_idxs))
            CE_val_idxs += [0] * (self.max_CE - len(CE_val_idxs))
            PIDE_val_idxs += [0] * (self.max_PIDE - len(PIDE_val_idxs))

            # mãe = universo do bloco (constante); posição = índice relativo na
            # sequência de palavras -- nunca mais derivado do valor do marcador.
            def build_feats(tokens, maxlen):
                n = len(tokens)
                vals = [(i / (n - 1) if n > 1 else 0.0) for i in range(n)]
                moms = [dominio_int] * n
                pad = maxlen - n
                vals += [0.0] * pad
                moms += [0] * pad
                return vals, moms

            E_vals, E_moms = build_feats(E_tokens, self.max_E)
            RE_vals, RE_moms = build_feats(RE_tokens, self.max_RE)
            CE_vals, CE_moms = build_feats(CE_tokens, self.max_CE)
            PI_vals, PI_moms = build_feats(PIDE_tokens, self.max_PIDE)

            out_total = b["saidas"][0]["tokens"].get("TOTAL", [])
            out_ids = [self.out_vocab.get(m, self.out_vocab[self.pad_token]) for m in out_total]
            y = out_ids
            # "_pos" fica zerado -- não é consumido em forward() (só E_val/E_val_idx/
            # E_mom/E são), mantido só pra não quebrar quem espera a chave existir.
            x = {
                "E": E_ids, "E_val": E_vals, "E_mom": E_moms, "E_pos": [0.0] * self.max_E, "E_val_idx": E_val_idxs,
                "RE": RE_ids, "RE_val": RE_vals, "RE_mom": RE_moms, "RE_pos": [0.0] * self.max_RE, "RE_val_idx": RE_val_idxs,
                "CE": CE_ids, "CE_val": CE_vals, "CE_mom": CE_moms, "CE_pos": [0.0] * self.max_CE, "CE_val_idx": CE_val_idxs,
                "PIDE": PIDE_ids, "PIDE_val": PI_vals, "PIDE_mom": PI_moms, "PIDE_pos": [0.0] * self.max_PIDE,
                "PIDE_val_idx": PIDE_val_idxs,
            }
            self.pares.append((x, y))

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int):
        x, y = self.pares[idx]
        x_t = {
            "E": torch.tensor(x["E"], dtype=torch.long),
            "E_val": torch.tensor(x["E_val"], dtype=torch.float32),
            "E_mom": torch.tensor(x["E_mom"], dtype=torch.long),
            "E_pos": torch.tensor(x["E_pos"], dtype=torch.long),
            "E_val_idx": torch.tensor(x["E_val_idx"], dtype=torch.long),

            "RE": torch.tensor(x["RE"], dtype=torch.long),
            "RE_val": torch.tensor(x["RE_val"], dtype=torch.float32),
            "RE_mom": torch.tensor(x["RE_mom"], dtype=torch.long),
            "RE_pos": torch.tensor(x["RE_pos"], dtype=torch.long),
            "RE_val_idx": torch.tensor(x["RE_val_idx"], dtype=torch.long),

            "CE": torch.tensor(x["CE"], dtype=torch.long),
            "CE_val": torch.tensor(x["CE_val"], dtype=torch.float32),
            "CE_mom": torch.tensor(x["CE_mom"], dtype=torch.long),
            "CE_pos": torch.tensor(x["CE_pos"], dtype=torch.long),
            "CE_val_idx": torch.tensor(x["CE_val_idx"], dtype=torch.long),

            "PIDE": torch.tensor(x["PIDE"], dtype=torch.long),
            "PIDE_val": torch.tensor(x["PIDE_val"], dtype=torch.float32),
            "PIDE_mom": torch.tensor(x["PIDE_mom"], dtype=torch.long),
            "PIDE_pos": torch.tensor(x["PIDE_pos"], dtype=torch.long),
            "PIDE_val_idx": torch.tensor(x["PIDE_val_idx"], dtype=torch.long),
        }
        y_t = torch.tensor(y + [self.out_vocab[self.pad_token]] * (self.max_out_len - len(y)), dtype=torch.float32)
        return x_t, y_t


## INSEPA_MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        super(SimpleGPT, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=0.1),
            num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embed(tgt)
        tgt_emb = self.pos_enc(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        out = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)


class AdamSegmentado(nn.Module):
    def __init__(self,
                 nE: int, nRE: int, nCE: int, nPIDE: int,
                 mom_size: int,
                 num_vals_E: int, num_vals_RE: int, num_vals_CE: int, num_vals_PIDE: int,
                 out_vocab_size: int, max_out_len: int,
                 max_E: int, max_RE: int, max_CE: int, max_PIDE: int, max_ng: int):
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
        self.max_out_len = max_out_len

        # Transformer Encoder melhorado com atenção multi-head
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=HIDDEN_DIM * 2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Módulo de raciocínio para PIDE (pensamentos internos) - Autoencoder Não Supervisionado
        self.encoder_pide = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4)  # Codificação comprimida
        )
        self.decoder_pide = nn.Sequential(
            nn.Linear(HIDDEN_DIM // 4, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, EMBED_DIM)  # Reconstrução
        )

        self.fc1 = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.act = nn.ReLU()

        # Decoder para geração de saída
        self.proj_value = nn.Linear(1, EMBED_DIM)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=HIDDEN_DIM * 2, dropout=0.1)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.out_head = nn.Linear(EMBED_DIM, 1)

        # Mapeamento de floats para índices para GPT
        self.float_to_idx = {float(i / (out_vocab_size - 1)): i for i in range(out_vocab_size)}
        self.idx_to_float = {i: float(i / (out_vocab_size - 1)) for i in range(out_vocab_size)}
        self.gpt = SimpleGPT(vocab_size=out_vocab_size, embed_dim=EMBED_DIM, num_heads=8, num_layers=2, max_len=max_out_len)

        # Para decodificação
        self.v_txt = None  # Vocabulário de saída (dicionário token -> id)
        self.idx_to_txt = None  # Mapeamento id -> token

    def forward(self, x: Dict[str, torch.Tensor], tgt: torch.Tensor = None, start_value: float = 0.26) -> Dict[str, torch.Tensor]:
        batch = x["E"].shape[0]
        # Campo E
        seq_len_E = x["E"].shape[1]
        expected_E = self.max_E * self.max_ng
        if seq_len_E < expected_E:
            pad_E = torch.zeros(batch, expected_E - seq_len_E, dtype=torch.long, device=x["E"].device)
            x_E = torch.cat([x["E"], pad_E], dim=1)
        elif seq_len_E > expected_E:
            x_E = x["E"][:, :expected_E]
        else:
            x_E = x["E"]
        eE_tok = self.em_E(x_E).view(batch, self.max_E, self.max_ng, EMBED_DIM).mean(dim=2)
        eE_val = self.em_Eval(x["E_val_idx"])
        eE_mom = self.em_Emom(x["E_mom"])
        eE_pos = self.proj_Epos(x["E_val"].unsqueeze(-1))
        eE = (eE_tok + eE_val + eE_mom + eE_pos).mean(dim=1)
        # Campo RE
        seq_len_RE = x["RE"].shape[1]
        expected_RE = self.max_RE * self.max_ng
        if seq_len_RE < expected_RE:
            pad_RE = torch.zeros(batch, expected_RE - seq_len_RE, dtype=torch.long, device=x["RE"].device)
            x_RE = torch.cat([x["RE"], pad_RE], dim=1)
        elif seq_len_RE > expected_RE:
            x_RE = x["RE"][:, :expected_RE]
        else:
            x_RE = x["RE"]
        eRE_tok = self.em_RE(x_RE).view(batch, self.max_RE, self.max_ng, EMBED_DIM).mean(dim=2)
        eRE_val = self.em_REval(x["RE_val_idx"])
        eRE_mom = self.em_REmom(x["RE_mom"])
        eRE_pos = self.proj_REpos(x["RE_val"].unsqueeze(-1))
        eRE = (eRE_tok + eRE_val + eRE_mom + eRE_pos).mean(dim=1)
        # Campo CE
        seq_len_CE = x["CE"].shape[1]
        expected_CE = self.max_CE * self.max_ng
        if seq_len_CE < expected_CE:
            pad_CE = torch.zeros(batch, expected_CE - seq_len_CE, dtype=torch.long, device=x["CE"].device)
            x_CE = torch.cat([x["CE"], pad_CE], dim=1)
        elif seq_len_CE > expected_CE:
            x_CE = x["CE"][:, :expected_CE]
        else:
            x_CE = x["CE"]
        eCE_tok = self.em_CE(x_CE).view(batch, self.max_CE, self.max_ng, EMBED_DIM).mean(dim=2)
        eCE_val = self.em_CEval(x["CE_val_idx"])
        eCE_mom = self.em_CEmom(x["CE_mom"])
        eCE_pos = self.proj_CEpos(x["CE_val"].unsqueeze(-1))
        eCE = (eCE_tok + eCE_val + eCE_mom + eCE_pos).mean(dim=1)
        # Campo PIDE
        seq_len_PIDE = x["PIDE"].shape[1]
        expected_PIDE = self.max_PIDE * self.max_ng
        if seq_len_PIDE < expected_PIDE:
            pad_PIDE = torch.zeros(batch, expected_PIDE - seq_len_PIDE, dtype=torch.long, device=x["PIDE"].device)
            x_PIDE = torch.cat([x["PIDE"], pad_PIDE], dim=1)
        elif seq_len_PIDE > expected_PIDE:
            x_PIDE = x["PIDE"][:, :expected_PIDE]
        else:
            x_PIDE = x["PIDE"]
        ePI_tok = self.em_PIDE(x_PIDE).view(batch, self.max_PIDE, self.max_ng, EMBED_DIM).mean(dim=2)
        ePI_val = self.em_PIDEval(x["PIDE_val_idx"])
        ePI_mom = self.em_PIDEmom(x["PIDE_mom"])
        ePI_pos = self.proj_PIDEpos(x["PIDE_val"].unsqueeze(-1))
        ePIDE_raw = (ePI_tok + ePI_val + ePI_mom + ePI_pos).mean(dim=1)
        ePIDE_encoded = self.encoder_pide(ePIDE_raw)  # Codificação comprimida
        ePIDE_recon = self.decoder_pide(ePIDE_encoded)  # Reconstrução
        ePIDE = ePIDE_raw  # Usar embedding raw no transformer

        # Agrega e classifica com transformer melhorado
        seq = torch.stack([eE, eRE, eCE, ePIDE], dim=1)  # (batch, 4, EMBED_DIM)
        seq = seq.permute(1, 0, 2)  # (4, batch, EMBED_DIM)
        transformed = self.transformer(seq)  # (4, batch, EMBED_DIM)
        transformed = transformed.permute(1, 0, 2)  # (batch, 4, EMBED_DIM)
        h = transformed.mean(dim=1)  # (batch, EMBED_DIM)
        h = self.act(self.fc1(h))

        # Decoder para geração usando GPT
        if tgt is not None:
            # Converter tgt floats para índices
            tgt_indices = torch.tensor([self.float_to_idx.get(float(val), 0) for val in tgt.flatten()], dtype=torch.long, device=tgt.device).view(tgt.shape)
            tgt_emb = self.gpt.embed(tgt_indices).permute(1, 0, 2)  # (max_out_len, batch, embed_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            memory = h.unsqueeze(0)  # (1, batch, EMBED_DIM)
            out_dec = self.gpt.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
            logits_indices = self.gpt.fc_out(out_dec)  # (max_out_len, batch, vocab_size)
            # Converter logits de índices para floats
            pred_indices = logits_indices.argmax(dim=-1)  # (max_out_len, batch)
            pred_floats = torch.tensor([[self.idx_to_float[int(idx)] for idx in seq] for seq in pred_indices.t()], dtype=torch.float32, device=tgt.device).t()
            logits = pred_floats  # (max_out_len, batch)
        else:
            # Geração autoregressiva usando GPT
            generated = []
            current_idx = self.float_to_idx.get(start_value, 0)
            current_tensor = torch.full((batch, 1), current_idx, dtype=torch.long, device=h.device)
            for _ in range(self.max_out_len):
                tgt_emb = self.gpt.embed(current_tensor).permute(1, 0, 2)  # (1, batch, embed_dim)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_tensor.size(1)).to(current_tensor.device)
                memory = h.unsqueeze(0)
                out_dec = self.gpt.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
                next_logits = self.gpt.fc_out(out_dec[-1])  # último token (batch, vocab_size)
                next_idx = next_logits.argmax(dim=-1)  # (batch,)
                generated.append(self.idx_to_float[int(next_idx[0])])  # assumir batch=1
                current_tensor = torch.cat([current_tensor, next_idx.unsqueeze(1)], dim=1)
            logits = torch.tensor(generated, dtype=torch.float32, device=h.device).unsqueeze(1).repeat(1, batch)

        return {
            "out": logits,
            "recon_pide": ePIDE_recon,  # Reconstrução para perda não supervisionada
            "pide_raw": ePIDE_raw  # Embedding original do PIDE para comparar
        }

    def decode_tokens(self, generated_ids: torch.Tensor, bloco: dict, dominio: str, inconsciente: dict = None) -> list:
        """Decodifica IDs de tokens gerados para uma lista de respostas únicas usando matching de sequências de floats."""
        if bloco is None:
            # Usar vocabulário do modelo para geração autônoma
            generated_seq = [val.item() for val in generated_ids.flatten() if val.item() != -1.0]
            response_tokens = []
            for val in generated_seq:
                idx = self.float_to_idx.get(val, 0)
                token = self.idx_to_txt.get(idx, UNK)
                response_tokens.append(token)
            response_text = ' '.join(response_tokens)
            return [response_text]
        if inconsciente is None:
            inconsciente = st.session_state.inconsciente
        if not hasattr(self, 'idx_to_txt'):
            return ["Vocabulário não carregado."]
        
        # Inverter idx_to_txt para word_to_idx
        word_to_idx = {v: k for k, v in self.idx_to_txt.items()}
        
        # Obter as respostas possíveis do bloco
        textos = bloco["saidas"][0]["textos"]
        reacao = bloco["saidas"][0].get("reacao", "")
        respostas_possiveis = [texto + (" " + reacao if reacao else "") for texto in textos]
        
        # Definir sequências esperadas para cada resposta
        sequencias_esperadas = {}
        for resp in respostas_possiveis:
            tokens = Token(resp)
            seq = []
            for token in tokens:
                if token in word_to_idx:
                    seq.append(word_to_idx[token])
                else:
                    # Fallback para um valor padrão
                    seq.append(0.26)
            sequencias_esperadas[resp] = seq
        
        # Sequência gerada
        generated_seq = [val.item() for val in generated_ids.flatten() if val.item() != -1.0]
        
        # Encontrar a resposta com a sequência mais próxima
        best_resp = None
        best_dist = float('inf')
        for resp, seq_exp in sequencias_esperadas.items():
            # Comparar sequências (assumindo mesmo tamanho ou truncar)
            min_len = min(len(generated_seq), len(seq_exp))
            dist = sum(abs(generated_seq[i] - seq_exp[i]) for i in range(min_len))
            if len(generated_seq) != len(seq_exp):
                dist += abs(len(generated_seq) - len(seq_exp)) * 0.1  # penalidade por diferença de tamanho
            if dist < best_dist:
                best_dist = dist
                best_resp = resp
        
        if best_resp:
            # Aplicar variações inconscientes
            unique_responses = set()
            attempts = 0
            while len(unique_responses) < 3 and attempts < 20:
                varied = variar_texto(best_resp, bloco, dominio, 'saida', inconsciente)
                unique_responses.add(varied)
                attempts += 1
            responses = list(unique_responses)
            return responses if responses else [best_resp]
        else:
            return ["Sequência não reconhecida."]


## INSEPA_TRAIN
def train(memoria: dict, dominio: str) -> None:
    try:
        # Atualizar inconsciente para o IM selecionado
        atualizar_inconsciente_para_im(memoria, dominio)

        ds = InsepaFieldDataset(memoria, dominio)
        n = len(ds)
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
            out_vocab_size=len(ds.out_vocab), max_out_len=ds.max_out_len,
            max_E=ds.max_E, max_RE=ds.max_RE, max_CE=ds.max_CE, max_PIDE=ds.max_PIDE, max_ng=ds.max_ng
        )
        opt = optim.Adam(model.parameters(), lr=LR)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        best, wait, prev_val = float("inf"), 0, None
        progress_bar = st.progress(0)
        status_text = st.empty()
        side_bar = st.sidebar.container()
        side_progress = side_bar.progress(0)
        side_status = side_bar.empty()
        for ep in range(1, EPOCHS + 1):
            model.train()
            for x, y in train_ld:
                opt.zero_grad()
                out = model(x, y)
                loss = (
                        mse(out["out"].reshape(-1), y.view(-1)) +
                        mse(out["recon_pide"], out["pide_raw"])  # Perda não supervisionada para PIDE
                )
                loss.backward()
                opt.step()

            model.eval()
            val_loss = 0.0
            if val_ld:
                with torch.no_grad():
                    for x, y in val_ld:
                        out = model(x, y)
                        val_loss += (
                                mse(out["out"].reshape(-1), y.view(-1)).item() +
                                mse(out["recon_pide"], out["pide_raw"]).item()  # Perda não supervisionada
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
                    len(ds.out_vocab), ds.max_out_len,
                    ds.max_ng,
                    ds.out_vocab, ds.all_out_markers, ds.idx_to_txt  # Adicionar vS, all_out_markers e idx_to_txt
                ), ckpt)
                # Salvar checkpoint em Mongo também (para persistência em cloud)
                try:
                    with open(ckpt, "rb") as f:
                        checkpoint_bytes = f.read()
                    _mongo_save_checkpoint(dominio, checkpoint_bytes)
                except Exception as e:
                    pass  # Não falhar treino por Mongo
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
            prev_val = val_loss
            progress_bar.progress(ep / EPOCHS)
            side_progress.progress(ep / EPOCHS)
            status_msg = f"Época {ep}/{EPOCHS}, Val Loss: {val_loss:.4f}"
            status_text.text(status_msg)
            side_status.text(status_msg)

        st.success(f"✅ Treino concluído. best_val_loss={best:.4f}")
        
        # Salvar backup do JSON usado para treinamento
        backup_memoria = f"backup/Adam_Lovely_memory_backup_{dominio}_{int(time.time())}.json"
        salvar_json(backup_memoria, memoria)
        st.info(f"📁 Backup do JSON salvo como: {backup_memoria}")
    except Exception as e:
        st.error(f"❌ Treino falhou: {e}")
        st.exception(e)


def fine_tune_model(memoria: dict, dominio: str, new_data: List[Tuple[Dict, Dict]]) -> None:
    """Fine-tuning incremental com novos dados de interação."""
    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint para fine-tuning.")
        return

    data = torch.load(ckpt)
    if len(data) == 18:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         out_vocab_size, max_out_len,
         max_ng,
         vS
        ) = data
        n_txt, n_emo, n_ctx = out_vocab_size, 1, 1  # defaults for old model
    elif len(data) == 17:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, n_emo, n_ctx,
         max_ng
        ) = data
        out_vocab_size = n_txt
        max_out_len = 10  # default
        # Recriar vocabulário de saída
        blocos = memoria["IM"][dominio]["blocos"]
        vS = {}
        for b in blocos:
            for saida in b["saidas"]:
                for texto in saida["textos"]:
                    for token in Token(texto):
                        vS[token] = vS.get(token, len(vS))
                reac = saida.get("reacao", "")
                if reac:
                    vS[reac] = vS.get(reac, len(vS))
                ctx = saida.get("contexto", "")
                for token in Token(ctx):
                    vS[token] = vS.get(token, len(vS))
    else:
        raise ValueError(f"Checkpoint has {len(data)} values, expected 17 or 18")

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE),
        nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        out_vocab_size=out_vocab_size, max_out_len=max_out_len,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    model.load_state_dict(state)
    model.v_txt = vS
    model.idx_to_txt = {v: k for k, v in vS.items()}
    opt = optim.Adam(model.parameters(), lr=LR * 0.1)  # LR menor para fine-tuning
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Criar dataset com novos dados
    class TempDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    temp_ds = TempDataset(new_data)
    temp_ld = DataLoader(temp_ds, batch_size=1, shuffle=True)

    model.train()
    for ep in range(5):  # Poucas épocas para fine-tuning
        for x, y in temp_ld:
            opt.zero_grad()
            out = model(x)
            loss = (
                    ce(out["texto"], y["texto"]) +
                    ce(out["emoji"], y["emoji"]) +
                    ce(out["ctx"], y["ctx"]) +
                    mse(out["pos"], y["pos"]) +
                    mse(out["recon_pide"], out["pide_raw"])  # Perda não supervisionada
            )
            loss.backward()
            opt.step()

    # Salvar modelo atualizado
    torch.save((
        model.state_dict(),
        maxE, maxRE, maxCE, maxPIDE,
        mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
        vE, vRE, vCE, vPIDE,
        out_vocab_size, max_out_len,
        max_ng
    ), ckpt)
    # Salvar checkpoint em Mongo também (para persistência em cloud)
    try:
        with open(ckpt, "rb") as f:
            checkpoint_bytes = f.read()
        _mongo_save_checkpoint(dominio, checkpoint_bytes)
    except Exception as e:
        pass  # Não falhar fine-tuning por Mongo
    st.info("🔄 Modelo fine-tunado com nova interação.")


def generate_insight(bloco, chosen=None):
    # Usar o contexto de ENTRADA (CE) para explicar a seleção de resposta.
    if bloco["entrada"].get("texto") and bloco["entrada"].get("reacao"):
        ep_txt = bloco["entrada"]["texto"]
        ep_reac = bloco["entrada"]["reacao"]
        # Usar exclusivamente o contexto de entrada (CE); o contexto de saída (CS) não deve ser usado aqui.
        contexto = bloco["entrada"].get("contexto", "")
        if chosen is None:
            chosen = bloco["saidas"][0]["textos"][0]
        emoji = bloco["saidas"][0].get("reacao", "")
        return f"Baseado na entrada '{ep_txt}', reação '{ep_reac}' e contexto '{contexto}', conclui que '{chosen} {emoji}' é a resposta mais adequada."
    return None


def gerar_reflexao(conversa_blocos: List[dict], dominio: str) -> str:
    """Gera uma reflexão interna baseada no histórico de blocos."""
    if len(conversa_blocos) < 2:
        return None
    
    # Analisar padrões: emoções, contextos (usar contexto de entrada - CE)
    emocoes = [b["entrada"].get("reacao", "") for b in conversa_blocos]
    contextos = [b["entrada"].get("contexto", "") for b in conversa_blocos]
    
    emocao_comum = max(set(emocoes), key=emocoes.count) if emocoes else ""
    contexto_comum = max(set(contextos), key=contextos.count) if contextos else ""
    
    reflexoes = [
        f"Observo que as interações recentes envolvem principalmente a emoção '{emocao_comum}', sugerindo um padrão emocional consistente.",
        f"O contexto '{contexto_comum}' aparece frequentemente, indicando temas recorrentes na conversa.",
        f"Com base nas últimas {len(conversa_blocos)} interações, estou aprendendo a adaptar minhas respostas para melhor refletir o fluxo emocional.",
        f"Minha 'mente' está evoluindo: de respostas isoladas para um entendimento mais coeso das emoções e contextos."
    ]
    
    return random.choice(reflexoes)


def fine_tune_online(memoria: dict, dominio: str, bloco_id: str, response: str) -> None:
    """Fine-tuning online com um bloco específico baseado no like."""
    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint para fine-tuning online.")
        return

    data = torch.load(ckpt)
    if len(data) == 18:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         out_vocab_size, max_out_len,
         max_ng,
         vS
        ) = data
        n_txt, n_emo, n_ctx = out_vocab_size, 1, 1  # defaults for old model
    elif len(data) == 17:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, n_emo, n_ctx,
         max_ng
        ) = data
        out_vocab_size = n_txt
        max_out_len = 10  # default
        # Recriar vocabulário de saída
        blocos = memoria["IM"][dominio]["blocos"]
        vS = {}
        for b in blocos:
            for saida in b["saidas"]:
                for texto in saida["textos"]:
                    for token in Token(texto):
                        vS[token] = vS.get(token, len(vS))
                reac = saida.get("reacao", "")
                if reac:
                    vS[reac] = vS.get(reac, len(vS))
                ctx = saida.get("contexto", "")
                for token in Token(ctx):
                    vS[token] = vS.get(token, len(vS))
    else:
        raise ValueError(f"Checkpoint has {len(data)} values, expected 17 or 18")

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE),
        nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        out_vocab_size=out_vocab_size, max_out_len=max_out_len,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    model.load_state_dict(state)
    model.v_txt = vS
    model.idx_to_txt = {v: k for k, v in vS.items()}
    opt = optim.Adam(model.parameters(), lr=LR * 0.01)  # LR ainda menor para online
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Criar dataset com o bloco específico
    ds_temp = InsepaFieldDataset(memoria, dominio)
    # Filtrar para o bloco_id
    indices = [i for i, (x, y) in enumerate(ds_temp) if ds_temp.pares[i][0]['E'].shape[0] > 0]  # Aproximado, ajustar se necessário
    # Para simplicidade, treinar com todos os dados por 1-2 épocas rápidas
    temp_ld = DataLoader(ds_temp, batch_size=1, shuffle=True)

    model.train()
    for ep in range(2):  # Poucas épocas para ajuste rápido
        for x, y in temp_ld:
            opt.zero_grad()
            out = model(x)
            loss = (
                    ce(out["texto"], y["texto"]) +
                    ce(out["emoji"], y["emoji"]) +
                    ce(out["ctx"], y["ctx"]) +
                    mse(out["pos"], y["pos"]) +
                    mse(out["recon_pide"], out["pide_raw"])  # Perda não supervisionada
            )
            loss.backward()
            opt.step()

    # Salvar modelo atualizado
    torch.save((
        model.state_dict(),
        maxE, maxRE, maxCE, maxPIDE,
        mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
        vE, vRE, vCE, vPIDE,
        out_vocab_size, max_out_len,
        max_ng
    ), ckpt)
    # Salvar checkpoint em Mongo também (para persistência em cloud)
    try:
        with open(ckpt, "rb") as f:
            checkpoint_bytes = f.read()
        _mongo_save_checkpoint(dominio, checkpoint_bytes)
    except Exception as e:
        pass  # Não falhar aprendizado por Mongo


def calcular_similaridade(bloco: dict, txt: str, reac: str, contexto: str, thought: str, dominio: str) -> float:
    """Calcula similaridade entre input e bloco baseado em texto, reação, contexto e pensamento."""
    txt_tokens = set(Token(normalize(txt)))
    reac_tokens = set(Token(normalize(reac)))
    ctx_tokens = set(Token(normalize(contexto)))
    thought_tokens = set(Token(normalize(thought)))
    
    bloco_txt = set(Token(normalize(bloco["entrada"]["texto"])))
    bloco_reac = set(Token(normalize(bloco["entrada"].get("reacao", ""))))
    bloco_ctx = set(Token(normalize(bloco["entrada"].get("contexto", ""))))
    bloco_thought = set(Token(normalize(bloco["entrada"].get("pensamento_interno", ""))))
    
    txt_sim = len(txt_tokens & bloco_txt) / len(txt_tokens | bloco_txt) if txt_tokens or bloco_txt else 0
    reac_sim = len(reac_tokens & bloco_reac) / len(reac_tokens | bloco_reac) if reac_tokens or bloco_reac else 0
    ctx_sim = len(ctx_tokens & bloco_ctx) / len(ctx_tokens | bloco_ctx) if ctx_tokens or bloco_ctx else 0
    thought_sim = len(thought_tokens & bloco_thought) / len(thought_tokens | bloco_thought) if thought_tokens or bloco_thought else 0
    
    # Peso: contexto 0.3, reação 0.3, pensamento 0.3, texto 0.1
    return 0.3 * ctx_sim + 0.3 * reac_sim + 0.3 * thought_sim + 0.1 * txt_sim


def corpus_similarity_score(entrada_texto: str, entrada_reacao: str, entrada_contexto: str, entrada_pensamento: str, dominio: str) -> float:
    """Maior similaridade entre a entrada e qualquer bloco já existente no corpus do domínio.

    Usado como portão de confiança: só o que estiver suficientemente fiel ao
    corpus (ver CORPUS_SIMILARITY_THRESHOLD) deve ser persistido como aprendizado autônomo.
    """
    memoria = st.session_state.get("memoria", {})
    blocos = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
    if not blocos:
        return 0.0
    return max(
        calcular_similaridade(b, entrada_texto, entrada_reacao, entrada_contexto, entrada_pensamento, dominio)
        for b in blocos
    )


## ALNULU_ENCODING
def alnulu_encode(texto: str) -> List[float]:
    """ALNULU encoding: converte texto em valores numéricos para similaridade."""
    mapa = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':-10,'K':11,'L':12,'M':-13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':-22,'W':23,'X':24,'Y':-25,'Z':26,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'.':2,'!':3,'?':4,',':1,';':1,':':1,'-':1}
    equiv = {'Á':'A','À':'A','Â':'A','Ã':'A','Ä':'A','È':'E','Ê':'E','É':'E','Ì':'I','Î':'I','Í':'I','Ó':'O','Ò':'O','Ô':'O','Õ':'O','Ö':'O','Ú':'U','Ù':'U','Û':'U','Ü':'U','Ç':'C','Ñ':'N','4':'A','3':'E','1':'I','0':'O','5':'S','7':'T','2':'Z'}
    return [float(mapa.get(equiv.get(char.upper(), char.upper()), 0.0)) for char in texto]


def alnulu_string(palavra: str) -> str:
    """Representação em string do ALNULU de uma palavra -- é isso (a FORMA da
    palavra), e não o número do marcador, que deve virar n-grama pra alimentar
    o treino do modelo. Palavras parecidas na forma (acento/leet-speak, que o
    ALNULU já normaliza) geram strings parecidas, então o modelo generaliza por
    forma -- nunca por posição, que nunca se repete (Camada 1)."""
    return "-".join(str(int(v)) for v in alnulu_encode(palavra))


def palavras_do_campo(bloco: dict, campo: str) -> List[str]:
    """Tokeniza o conteúdo REAL de um campo do bloco (texto/reação/contexto/
    pensamento, entrada ou saída) -- nunca o marcador. É isso que o modelo deve
    aprender a reconhecer; o marcador continua só marcando posição/universo."""
    entrada = bloco.get("entrada", {})
    saida = (bloco.get("saidas") or [{}])[0]
    fonte = {
        "E": entrada.get("texto", ""),
        "RE": entrada.get("reacao", ""),
        "CE": entrada.get("contexto", ""),
        "PIDE": entrada.get("pensamento_interno", ""),
        "S": (saida.get("textos") or [""])[0],
        "RS": saida.get("reacao", ""),
        "CS": saida.get("contexto", ""),
    }
    return Token(fonte.get(campo, "")) if fonte.get(campo) else []


def load_inconsciente_lexicon(inconsciente: dict) -> dict:
    """Carrega o léxico a partir do objeto `inconsciente` (JSON) e garante estrutura.
    Retorna dicionário com chaves: 'universos', 'universo0', 'dominios', 'global'."""
    lex = {"universos": {}, "universo0": {}, "dominios": {}, "global": {}}
    if not inconsciente or not isinstance(inconsciente, dict):
        return lex
    # If the inconsciente already contains a lexicon structure, merge it
    inc_lex = inconsciente.get("lexicon") or inconsciente.get("lexico") or {}
    # universos específicos
    for u_name, u_map in (inc_lex.get("universos") or {}).items():
        lex["universos"][u_name.lower()] = {k.lower(): v for k, v in (u_map or {}).items()}
    # universo0 (raiz)
    for k, v in (inc_lex.get("universo0") or {}).items():
        lex["universo0"][k.lower()] = v
    # dominios
    for d_name, d_map in (inc_lex.get("dominios") or {}).items():
        lex["dominios"][d_name.lower()] = {k.lower(): v for k, v in (d_map or {}).items()}
    # global
    for k, v in (inc_lex.get("global") or {}).items():
        lex["global"][k.lower()] = v
    return lex


def apply_domain_lexicon(text: str, universo: str = None, dominio: str = None) -> str:
    """Aplica mapeamento léxico consultando apenas o léxico do `inconsciente`.
    Prioridade:
    1) léxico do universo específico (st.session_state.inconsciente_lexicon['universos'])
    2) universo0 (raiz)
    3) léxico por domínio
    4) global
    Retorna a forma semântica sem alterar os dados originais.
    """
    if not text:
        return text
    lexicon = None
    try:
        lexicon = st.session_state.get("inconsciente_lexicon")
    except Exception:
        lexicon = None
    if not lexicon:
        return text
    universo_key = (universo or "").lower()
    dominio_key = (dominio or universo or "").lower()

    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip(".,;:!?()[]\"'")
        replaced = None
        # universo específico
        u_map = lexicon.get("universos", {}).get(universo_key, {})
        if key in u_map:
            replaced = u_map[key]
        # universo0
        if replaced is None:
            u0_map = lexicon.get("universo0", {})
            if key in u0_map:
                replaced = u0_map[key]
        # domínio
        if replaced is None:
            d_map = lexicon.get("dominios", {}).get(dominio_key, {})
            if key in d_map:
                replaced = d_map[key]
        # global
        if replaced is None:
            g_map = lexicon.get("global", {})
            if key in g_map:
                replaced = g_map[key]
        out.append(replaced if replaced is not None else w)
    return " ".join(out)


def alnulu_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcula similaridade entre dois vetores ALNULU usando similaridade por cosseno
    com uma leve penalidade por diferença de comprimento. Retorna valor em [0,1]."""
    import numpy as _np
    if not vec1 or not vec2:
        return 0.0
    a = _np.array(vec1, dtype=_np.float32)
    b = _np.array(vec2, dtype=_np.float32)
    # pad para o mesmo comprimento com zeros
    if a.size < b.size:
        a = _np.pad(a, (0, b.size - a.size))
    elif b.size < a.size:
        b = _np.pad(b, (0, a.size - b.size))
    norm_a = _np.linalg.norm(a)
    norm_b = _np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos = float(_np.dot(a, b) / (norm_a * norm_b))
    # normalizar para 0..1 (cos varia -1..1)
    sim = max(0.0, (cos + 1.0) / 2.0)
    # leve penalidade por diferença de comprimento relativa (até 0.1)
    len_penalty = abs(len(vec1) - len(vec2)) / max(len(vec1), len(vec2)) * 0.1
    return max(0.0, sim - len_penalty)


## INSEPA_CAMADA_5 (Aprendizado Seguro)
# O ALNULU serve só para o Adam "ver" a forma de uma palavra (transformá-la em número).
# Reconhecer não é uma questão de distância/aproximação -- é o INSEPA que decide se algo
# é conhecido, por registro exato: é um token já usado, ou é uma var/multivar já
# registrada em algum bloco. IA comum trabalha com embeddings e similaridade de vetor;
# o Adam trabalha com algarismos sequenciais e posição exata. Sem meio-termo por forma.
def classificar_dado(texto: str, emocao: str, contexto: str, pensamento: str) -> str:
    """Opinião: só texto+emoção (o que qualquer usuário pode produzir). Dado concreto:
    todos os quatro campos -- e só a criadora pode preencher contexto/pensamento."""
    if texto.strip() and emocao.strip() and contexto.strip() and pensamento.strip():
        return "dado concreto"
    if texto.strip() and emocao.strip():
        return "opinião"
    return "incompleto"


def carregar_opinioes() -> List[dict]:
    """Opiniões (texto+emoção) registradas por qualquer usuário, pendentes de
    revisão da criadora -- contexto e pensamento_interno ficam None até lá."""
    data = carregar_json(ARQUIVO_OPINIOES, {"opinioes": []})
    return data.get("opinioes", [])


def salvar_opinioes(opinioes: List[dict]) -> None:
    """Salva no Mongo (se habilitado) e no arquivo local, igual memória/inconsciente --
    sem isso, opiniões pendentes sumiriam a cada reinício do container na nuvem."""
    salvar_json(ARQUIVO_OPINIOES, {"opinioes": opinioes})


def registrar_opiniao_pendente(texto: str, emocao: str, dominio: str, candidato: Optional[dict] = None, origem: str = "reflexao", palpite_rede_neural: Optional[str] = None) -> dict:
    """Registra uma opinião pendente. Só a criadora pode depois preencher
    contexto/pensamento e convertê-la em dado concreto (ver submenu_opinioes).

    `candidato` (opcional): o melhor bloco parecido que o Adam já tentou (ver
    melhor_candidato_fraco), pra criadora ver o que ele "achou que era" e
    decidir se reforça a ligação (vira var) ou se é assunto realmente novo.

    `origem`: "reflexao" (padrão, veio de uma entrada real que não bateu) ou
    "semeadura" (o Adam propôs sozinho, fora de uma conversa, clonando uma
    raiz do universo 0 -- ver propor_semeadura_universo).

    `palpite_rede_neural` (opcional): o que a rede neural treinada gerou pra
    essa entrada (ver gerar_palpite_rede_neural) -- um sinal ainda mais fraco
    que o candidato, nunca mostrado como resposta, só guardado pra criadora
    avaliar junto com o resto."""
    opinioes = carregar_opinioes()
    ja_pendente = next(
        (o for o in opinioes if o.get("status") == "pendente" and o.get("dominio") == dominio
         and normalize(o.get("texto", "")) == normalize(texto) and o.get("emocao") == emocao),
        None,
    )
    if ja_pendente:
        return ja_pendente

    placeholder = f"{dominio}.0"  # ponto neutro do universo -- nunca usado pela sequência normal (que começa em .1)
    emprestado = candidato.get("emprestado", False) if candidato else False
    if emprestado:
        # Empréstimo/semeadura: contexto e pensamento JÁ são conhecidos (vieram do
        # universo 0) -- guardar o placeholder aqui seria perder a própria informação
        # que foi emprestada. Ainda fica "pendente": emprestar não é a mesma coisa
        # que a criadora confirmar.
        contexto_inicial = candidato["bloco"]["entrada"].get("contexto", placeholder)
        pensamento_inicial = candidato["bloco"]["entrada"].get("pensamento_interno", placeholder)
    else:
        contexto_inicial = placeholder
        pensamento_inicial = placeholder

    opiniao = {
        "id": str(uuid.uuid4()),
        "dominio": dominio,
        "texto": texto,
        "emocao": emocao,
        "contexto": contexto_inicial,
        "pensamento_interno": pensamento_inicial,
        "status": "pendente",
        "origem": origem,
        "bloco_candidato_id": candidato["bloco"]["bloco_id"] if candidato else None,
        "bloco_candidato_score": candidato["score"] if candidato else None,
        "bloco_candidato_origem_im": candidato.get("origem_im", dominio) if candidato else None,
        "bloco_candidato_emprestado": emprestado,
        "palpite_rede_neural": palpite_rede_neural,
    }
    opinioes.append(opiniao)
    salvar_opinioes(opinioes)
    return opiniao


def propor_semeadura_universo(dominio_alvo: str) -> List[dict]:
    """Fora de uma conversa real (ex.: rotina automatizada/harness): o Adam procura
    raízes (pensamento_interno) do universo 0 que ainda não têm equivalente no
    universo derivado `dominio_alvo`, e PROPÕE -- nunca confirma sozinho -- um bloco
    novo pra esse universo, clonando texto+emoção+contexto do universo 0 como ponto
    de partida (ver melhor_candidato_fraco para a mesma lógica de empréstimo, aqui
    aplicada proativamente em vez de reativa a uma entrada real).

    Só funciona nessa direção (0 -> derivado); nunca planta dentro do próprio
    universo 0. Idempotente: não repropõe uma raiz já proposta antes (pendente,
    confirmada ou descartada), pra não spammar a mesma sugestão toda noite."""
    if dominio_alvo == "0":
        return []
    memoria = st.session_state.get("memoria", {})
    blocos_raiz = memoria.get("IM", {}).get("0", {}).get("blocos", [])
    blocos_alvo = memoria.get("IM", {}).get(dominio_alvo, {}).get("blocos", [])
    if not blocos_raiz:
        return []

    ja_propostas = {
        o.get("bloco_candidato_id")
        for o in carregar_opinioes()
        if o.get("dominio") == dominio_alvo and o.get("origem") == "semeadura"
    }

    propostas = []
    for bloco0 in blocos_raiz:
        bloco_id0 = bloco0.get("bloco_id")
        if bloco_id0 in ja_propostas:
            continue
        pensamento0 = bloco0.get("entrada", {}).get("pensamento_interno", "")
        if not pensamento0:
            continue
        ja_tem_equivalente = any(
            alnulu_token_similarity(pensamento0, b.get("entrada", {}).get("pensamento_interno", "")) >= 1.0
            for b in blocos_alvo
        )
        if ja_tem_equivalente:
            continue
        candidato = {"bloco": bloco0, "score": 1.0, "origem_im": "0", "emprestado": True}
        opiniao = registrar_opiniao_pendente(
            bloco0["entrada"]["texto"], bloco0["entrada"].get("reacao", ""), dominio_alvo, candidato, origem="semeadura"
        )
        propostas.append(opiniao)
    return propostas


def alnulu_token_similarity(txt1: str, txt2: str) -> float:
    """Similaridade ALNULU que respeita o INSEPA: compara os TOKENS (a unidade que
    o INSEPA já cataloga), usando o valor ALNULU (Alfabeto Numérico de Lux) de
    cada token como identidade — em vez de jogar o texto inteiro num único vetor
    e medir ângulo de cosseno, o que força uma normalidade entre os valores do
    alfabeto que não existe de verdade.

    Dois tokens só "batem" se tiverem a mesma sequência exata de valores ALNULU
    (o que já resolve variações de acento/leet-speak, porque alnulu_encode
    normaliza isso letra a letra antes de comparar).
    """
    tokens1 = Token(txt1 or "")
    tokens2 = Token(txt2 or "")
    if not tokens1 or not tokens2:
        return 0.0
    assinaturas1 = {tuple(alnulu_encode(t)) for t in tokens1}
    assinaturas2 = {tuple(alnulu_encode(t)) for t in tokens2}
    uniao = assinaturas1 | assinaturas2
    if not uniao:
        return 0.0
    return len(assinaturas1 & assinaturas2) / len(uniao)


def build_alnulu_cache(memoria: dict) -> dict:
    """Precomputa vetores ALNULU para cada bloco e armazena em cache para acelerar buscas."""
    cache = {}
    for dominio, universo in memoria.get("IM", {}).items():
        for bloco in universo.get("blocos", []):
            bloco_id = str(bloco.get("bloco_id", id(bloco)))
            # aplicar léxico de domínio antes de codificar para reduzir ambiguidade
            txt_proc = apply_domain_lexicon(bloco["entrada"]["texto"], dominio)
            reac_proc = apply_domain_lexicon(bloco["entrada"].get("reacao", ""), dominio)
            ctx_proc = apply_domain_lexicon(bloco["entrada"].get("contexto", ""), dominio)
            thought_proc = apply_domain_lexicon(bloco["entrada"].get("pensamento_interno", ""), dominio)
            cache[bloco_id] = {
                # manter texto original (preservando marcadores)
                "txt_original": bloco["entrada"]["texto"],
                "reac_original": bloco["entrada"].get("reacao", ""),
                "ctx_original": bloco["entrada"].get("contexto", ""),
                "thought_original": bloco["entrada"].get("pensamento_interno", ""),
                # forma semântica (após léxico) e vetores semânticos
                "txt_sem_text": txt_proc,
                "reac_sem_text": reac_proc,
                "ctx_sem_text": ctx_proc,
                "thought_sem_text": thought_proc,
                "txt": alnulu_encode(txt_proc),
                "reac": alnulu_encode(reac_proc),
                "ctx": alnulu_encode(ctx_proc),
                "thought": alnulu_encode(thought_proc),
            }
    return cache


def caracteristicas_persona(dominio: str) -> Dict[str, str]:
    """Características registradas da persona de um universo (nome, gênero, e no
    futuro cabelos/olhos/pele/estilo/gostos...) -- é por meio delas que o Adam
    reconhece que uma entrada é sobre/para ele, mesmo que o texto do bloco use
    outra forma de endereçá-lo (ex.: apelido usado num bloco vs o nome oficial)."""
    im_data = st.session_state.get("memoria", {}).get("IM", {}).get(dominio, {})
    campos_ignorados = {"blocos", "ultimo_child"}
    return {
        k: v for k, v in im_data.items()
        if k not in campos_ignorados and isinstance(v, str) and v.strip()
    }


def remover_termos_identidade(texto: str, dominio: str) -> str:
    """Tira da comparação os termos de identidade da persona (nome, gênero...):
    eles identificam QUEM está sendo endereçado, não SOBRE O QUE é a frase -- não
    devem contar a favor nem contra a similaridade de assunto."""
    termos = set(caracteristicas_persona(dominio).values())
    if not termos:
        return texto
    tokens = [t for t in Token(texto) if normalize(t) not in {normalize(v) for v in termos}]
    return " ".join(tokens)


def melhor_candidato_fraco(txt: str, dominio: str) -> Optional[dict]:
    """Quando nada bate o piso de confiança da imitação, o Adam não deve desistir
    direto -- deve refletir antes de agir: pega o melhor candidato que existir
    (mesmo fraco), pra reproduzir a resposta dele e PERGUNTAR se o contexto
    corresponde, em vez de responder com confiança ou fingir que não sabe nada.

    É livre pra errar o bloco -- só a criadora, ao reforçar, é que consolida a
    ligação de verdade (ver submenu_opinioes).

    O universo 0 (o "Big Bang", identidade real do Adam) pode emprestar
    pensamento pra universos derivados quando texto+emoção baterem de
    verdade -- mas nunca o contrário: um universo derivado (história/ficção)
    jamais empresta pra outro derivado nem de volta pro 0, pra não misturar
    a identidade real do Adam com as historinhas que ele conta."""
    memoria = st.session_state.get("memoria", {})
    if not txt:
        return None
    txt_sem_identidade = remover_termos_identidade(txt, dominio)

    pools = [(dominio, memoria.get("IM", {}).get(dominio, {}).get("blocos", []))]
    if dominio != "0":
        pools.append(("0", memoria.get("IM", {}).get("0", {}).get("blocos", [])))

    melhor_bloco, melhor_sim, melhor_origem = None, 0.0, dominio
    for origem_im, blocos in pools:
        for b in blocos:
            entrada = b.get("entrada", {})
            textos_variantes = [entrada.get("texto", "")] + entrada.get("Multivars_Texto_Entrada", [])
            sim = max((alnulu_token_similarity(txt_sem_identidade, v) for v in textos_variantes if v), default=0.0)
            if sim > melhor_sim:
                melhor_sim, melhor_bloco, melhor_origem = sim, b, origem_im
    if melhor_bloco is None or melhor_sim <= 0.0:
        return None
    return {"bloco": melhor_bloco, "score": melhor_sim, "origem_im": melhor_origem, "emprestado": melhor_origem != dominio}


def gerar_palpite_rede_neural(txt: str, reac: str, dominio: str) -> Optional[str]:
    """Terceiro sinal, mais fraco que tudo: quando nem match exato nem
    melhor_candidato_fraco acham nada, tenta a rede neural treinada (Camada 5
    corrigida -- ela aprende a FORMA da palavra via ALNULU, nunca o número do
    marcador). Só um palpite bruto de geração livre (decode_tokens com
    bloco=None); nunca é mostrado como resposta -- só alimenta a opinião
    pendente pra criadora avaliar. Qualquer falha (sem checkpoint, formato
    incompatível, etc.) retorna None silenciosamente -- nunca derruba o chat."""
    if not txt:
        return None
    try:
        import torch
        ckpt = ckpt_path(dominio)
        if not os.path.exists(ckpt):
            return None
        data = torch.load(ckpt)
        if len(data) != 20:
            return None  # só o formato mais novo -- os antigos não têm garantia de vocabulário por palavra
        (state, maxE, maxRE, maxCE, maxPIDE, mom_size,
         val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE, out_vocab_size, max_out_len, max_ng,
         vS, all_out_markers, idx_to_txt) = data

        model = AdamSegmentado(
            nE=len(vE), nRE=len(vRE), nCE=len(vCE), nPIDE=len(vPIDE), mom_size=mom_size,
            num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
            num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
            out_vocab_size=out_vocab_size, max_out_len=max_out_len,
            max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng,
        )
        model.load_state_dict(state)
        model.v_txt = vS
        model.idx_to_txt = idx_to_txt
        model.eval()

        dominio_int = int(dominio) if str(dominio).isdigit() else 0

        def feat(tokens, max_len, vocab, val_to_idx):
            ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
            val_idxs = [val_to_idx.get(t, 0) for t in tokens]
            n = len(tokens)
            vals = [(i / (n - 1) if n > 1 else 0.0) for i in range(n)]
            moms = [dominio_int] * n
            pos = [0.0] * n
            ids += [0] * ((max_len * max_ng) - len(ids))
            val_idxs += [0] * (max_len - n)
            vals += [0.0] * (max_len - n)
            moms += [0] * (max_len - n)
            pos += [0.0] * (max_len - n)
            return (
                torch.tensor([ids], dtype=torch.long), torch.tensor([val_idxs], dtype=torch.long),
                torch.tensor([vals], dtype=torch.float32), torch.tensor([moms], dtype=torch.long),
                torch.tensor([pos], dtype=torch.float32),
            )

        # Entrada nova: só texto/reação existem de verdade -- contexto/pensamento
        # ainda não foram inferidos, então entram vazios (nunca inventados aqui).
        E_ids, E_val_idx, E_val, E_mom, E_pos = feat(Token(txt), maxE, vE, val_to_idx_E)
        RE_ids, RE_val_idx, RE_val, RE_mom, RE_pos = feat(Token(reac) if reac else [], maxRE, vRE, val_to_idx_RE)
        CE_ids, CE_val_idx, CE_val, CE_mom, CE_pos = feat([], maxCE, vCE, val_to_idx_CE)
        PI_ids, PI_val_idx, PI_val, PI_mom, PI_pos = feat([], maxPIDE, vPIDE, val_to_idx_PIDE)

        x = {
            "E": E_ids, "E_val": E_val, "E_mom": E_mom, "E_val_idx": E_val_idx,
            "RE": RE_ids, "RE_val": RE_val, "RE_mom": RE_mom, "RE_val_idx": RE_val_idx,
            "CE": CE_ids, "CE_val": CE_val, "CE_mom": CE_mom, "CE_val_idx": CE_val_idx,
            "PIDE": PI_ids, "PIDE_val": PI_val, "PIDE_mom": PI_mom, "PIDE_val_idx": PI_val_idx,
        }
        with torch.no_grad():
            out = model(x)
        # out["out"] é (batch, max_out_len, 1) -- regressão (MSE no treino), um
        # float por posição, não logits de classe. decode_tokens já espera
        # exatamente esses floats direto (via float_to_idx), sem argmax nenhum.
        generated_ids = out["out"][0].squeeze(-1)
        respostas = model.decode_tokens(generated_ids, None, dominio)
        texto_gerado = respostas[0].strip() if respostas else ""
        return texto_gerado or None
    except Exception:
        # Qualquer coisa dando errado aqui é só "sem palpite" -- nunca derruba o chat.
        return None


def infer_context_thought_by_imitation(txt: str, reac: str, dominio: str) -> Tuple[str, str]:
    """Imita o corpus: encontra o bloco cujo texto+emoção mais se aproxima da entrada
    atual e empresta o contexto/pensamento desse bloco como estimativa.

    Esse é o primeiro estágio do aprendizado do Adam: reproduzir o encadeamento já
    presente no corpus (texto + emoção → contexto → pensamento) antes de poder criar
    contexto/pensamento livremente por conta própria.
    """
    memoria = st.session_state.get("memoria", {})
    blocos = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
    if not blocos or not txt:
        return "", ""

    MARGEM_MESMO_ASSUNTO = 0.08
    # Nome/gênero da persona identificam QUEM está sendo endereçado, não do que se
    # trata a frase -- tirar isso da comparação evita que uma saudação com o nome
    # da persona perca pontos de assunto só por não repetir o texto dos blocos.
    txt_sem_identidade = remover_termos_identidade(txt, dominio)
    candidatos = []
    for b in blocos:
        entrada = b.get("entrada", {})

        # Texto e emoção têm Vars/Multivars (várias formas de superfície para o
        # mesmo sentido) — comparar só a string canônica subestima o match.
        # Usa ALNULU por TOKEN (não o texto inteiro num vetor de cosseno): frases-modelo
        # como "Você gosta de X?" são quase idênticas em nível de caractere não importa
        # o X, então o cosseno não distingue assunto — o token que muda sim.
        textos_variantes = [entrada.get("texto", "")] + entrada.get("Multivars_Texto_Entrada", [])
        txt_sim = max((alnulu_token_similarity(txt_sem_identidade, v) for v in textos_variantes if v), default=0.0)

        reac_variantes = [entrada.get("reacao", "")]
        re_markers = entrada.get("tokens", {}).get("RE", [])
        if re_markers:
            try:
                reac_variantes += get_variations_for_tokens(dominio, b.get("bloco_id"), "RE", re_markers)
            except (KeyError, TypeError):
                pass
        reac_sim = 1.0 if reac and any(normalize(reac) == normalize(v) for v in reac_variantes if v) else 0.0

        candidatos.append((txt_sim, reac_sim, b))

    if not candidatos:
        return "", ""

    # O texto decide o assunto. A emoção só desempata ENTRE blocos cujo assunto já
    # está próximo do melhor — ela nunca pode sozinha roubar o assunto de um bloco
    # de tema diferente só por coincidir (ex.: ":D" não pode fazer "poderes" vencer
    # "lua" quando a pergunta era sobre a noite).
    melhor_txt_sim = max(c[0] for c in candidatos)

    # Contexto e pensamento_interno não têm Vars/Multivars: são únicos por raiz/situação
    # e NUNCA devem ser compartilhados entre assuntos divergentes (lua e noite são
    # divergentes por definição, mesmo sendo o candidato mais próximo disponível).
    # Sem um assunto genuinamente próximo, não há o que imitar — melhor admitir que
    # não sabe do que emprestar uma raiz de um tema diferente.
    if melhor_txt_sim < IMITACAO_ASSUNTO_MINIMO:
        return "", ""

    mesmo_assunto = [c for c in candidatos if c[0] >= melhor_txt_sim - MARGEM_MESMO_ASSUNTO]
    melhor_txt_sim_local, melhor_reac_sim, melhor_bloco = max(mesmo_assunto, key=lambda c: (c[1], c[0]))

    return melhor_bloco["entrada"].get("contexto", ""), melhor_bloco["entrada"].get("pensamento_interno", "")


def retrieve_similar_blocks_alnulu(txt: str, reac: str, contexto: str, thought: str, dominio: str, top_k=3) -> List[Tuple[float, dict]]:
    """Busca blocos similares usando ALNULU para identidade e similaridade, priorizando contexto e emoção."""
    memoria = st.session_state.memoria
    if dominio not in memoria["IM"]:
        return []
    blocos = memoria["IM"][dominio]["blocos"]
    
    # Pré-processar com léxico do domínio e gerar vetores
    txt_proc = apply_domain_lexicon(txt, dominio)
    reac_proc = apply_domain_lexicon(reac, dominio)
    ctx_proc = apply_domain_lexicon(contexto, dominio)
    thought_proc = apply_domain_lexicon(thought, dominio)
    txt_vec = alnulu_encode(txt_proc)
    reac_vec = alnulu_encode(reac_proc)
    ctx_vec = alnulu_encode(ctx_proc)
    thought_vec = alnulu_encode(thought_proc)
    
    similarities = []
    for bloco in blocos:
        # Encode bloco (usar cache para evitar recálculos caros)
        bloco_id = str(bloco.get("bloco_id", id(bloco)))
        bloco_cache = st.session_state.get("alnulu_cache", {}).get(bloco_id)
        if bloco_cache:
            bloco_txt_vec = bloco_cache["txt"]
            bloco_reac_vec = bloco_cache["reac"]
            bloco_ctx_vec = bloco_cache["ctx"]
            bloco_thought_vec = bloco_cache["thought"]
        else:
            # aplicar léxico do domínio ao bloco antes de codificar
            bloco_txt = apply_domain_lexicon(bloco["entrada"]["texto"], dominio)
            bloco_reac = apply_domain_lexicon(bloco["entrada"].get("reacao", ""), dominio)
            bloco_ctx = apply_domain_lexicon(bloco["entrada"].get("contexto", ""), dominio)
            bloco_thought = apply_domain_lexicon(bloco["entrada"].get("pensamento_interno", ""), dominio)
            bloco_txt_vec = alnulu_encode(bloco_txt)
            bloco_reac_vec = alnulu_encode(bloco_reac)
            bloco_ctx_vec = alnulu_encode(bloco_ctx)
            bloco_thought_vec = alnulu_encode(bloco_thought)
        
        # Similaridade por campo, toda via ALNULU por token (não cosseno no texto inteiro)
        txt_sim = semantic_text_similarity(txt, bloco["entrada"]["texto"], dominio)
        reac_sim = alnulu_token_similarity(reac_proc, apply_domain_lexicon(bloco["entrada"].get("reacao", ""), dominio))
        ctx_sim = alnulu_token_similarity(ctx_proc, apply_domain_lexicon(bloco["entrada"].get("contexto", ""), dominio))
        thought_sim = key_phrase_similarity(thought, bloco["entrada"].get("pensamento_interno", ""), dominio)
        
        # Similaridade por campo: contexto e pensamento lideram (0.6 juntos), texto e emoção têm
        # significado em conjunto e pesam igual entre si (0.4 juntos, sem um dominar o outro)
        overall_sim = 0.2 * txt_sim + 0.2 * reac_sim + 0.35 * ctx_sim + 0.25 * thought_sim
        
        # Bônus por concretude: se bloco tem contexto e pensamento, +0.1
        concretude_bonus = 0.1 if bloco["entrada"].get("contexto") and bloco["entrada"].get("pensamento_interno") else 0.0
        overall_sim = min(1.0, overall_sim + concretude_bonus)
        
        similarities.append((overall_sim, bloco))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]


def raiz_coerente(ctx_busca: str, bloco: dict, dominio: str, limiar: float = RAIZ_COERENCIA_THRESHOLD) -> bool:
    """Verifica se o contexto de SAÍDA do bloco (saidas[0].contexto) pertence à mesma
    raiz do contexto usado na busca (o da entrada).

    Um bloco pode bater bem pela entrada e mesmo assim ter uma saída mal
    encadeada (ex.: dado herdado de fora do corpus) — esse portão rejeita
    blocos assim mesmo que a entrada combine, em vez de aceitar cegamente
    qualquer saída de um bloco cuja entrada bateu.
    """
    saidas = bloco.get("saidas") or []
    saida_ctx = saidas[0].get("contexto", "") if saidas else ""
    if not saida_ctx or not ctx_busca:
        return True  # sem dado suficiente para negar — não bloqueia por ausência
    return semantic_text_similarity(ctx_busca, saida_ctx, dominio) >= limiar


def explain_similarity_match(txt: str, reac: str, contexto: str, thought: str, bloco: dict, dominio: str) -> Dict[str, Any]:
    """Recalcula, para UM bloco já escolhido, a mesma decomposição de similaridade
    usada em retrieve_similar_blocks_alnulu — só para exibir como métrica
    explicável no próprio chat (não influencia o matching em si)."""
    entrada = bloco.get("entrada", {})
    txt_proc = apply_domain_lexicon(txt, dominio)
    reac_proc = apply_domain_lexicon(reac, dominio)
    ctx_proc = apply_domain_lexicon(contexto, dominio)
    bloco_reac_proc = apply_domain_lexicon(entrada.get("reacao", ""), dominio)
    bloco_ctx_proc = apply_domain_lexicon(entrada.get("contexto", ""), dominio)

    txt_sim = semantic_text_similarity(txt, entrada.get("texto", ""), dominio)
    reac_sim = alnulu_token_similarity(reac_proc, bloco_reac_proc)
    ctx_sim = alnulu_token_similarity(ctx_proc, bloco_ctx_proc)
    thought_sim = key_phrase_similarity(thought, entrada.get("pensamento_interno", ""), dominio)
    overall_sim = 0.2 * txt_sim + 0.2 * reac_sim + 0.35 * ctx_sim + 0.25 * thought_sim
    concretude_bonus = 0.1 if entrada.get("contexto") and entrada.get("pensamento_interno") else 0.0
    overall_sim = min(1.0, overall_sim + concretude_bonus)

    return {
        "bloco_id": bloco.get("bloco_id"),
        "txt_sim": txt_sim,
        "reac_sim": reac_sim,
        "ctx_sim": ctx_sim,
        "thought_sim": thought_sim,
        "concretude_bonus": concretude_bonus,
        "overall_sim": overall_sim,
        "contexto_imitado": contexto,
        "pensamento_imitado": thought,
        "bloco_contexto": entrada.get("contexto", ""),
        "bloco_pensamento": entrada.get("pensamento_interno", "").strip('"'),
    }


def montar_resposta_por_confianca(resposta_texto: str, resposta_reacao: str, metrics: Dict[str, Any]) -> str:
    """Decide quanto da resposta mostrar com base na fidelidade de CONTEXTO e
    PENSAMENTO — os campos sem Vars/Multivars, portanto o fingerprint confiável
    da situação — em vez da igualdade literal de texto/reação, que varia por
    natureza e por isso quase nunca bate 100% mesmo quando o sentido é o mesmo.
    """
    fingerprint = min(metrics["ctx_sim"], metrics["thought_sim"])
    if fingerprint >= 0.85:
        corpo = resposta_texto
    elif fingerprint >= 0.5:
        palavras = Token(resposta_texto)
        metade = max(1, len(palavras) // 2)
        corpo = ' '.join(palavras[:metade])
    else:
        corpo = resposta_texto.split()[0] if resposta_texto.split() else resposta_texto
    return corpo + (" " + resposta_reacao if resposta_reacao else "")


def _confianca_em_palavras(score: float) -> str:
    if score >= 0.85:
        return "tenho quase certeza disso"
    if score >= 0.65:
        return "acho que é bem por aí"
    if score >= 0.5:
        return "não tenho tanta certeza, mas arrisco"
    return "tô mais no chute, sendo sincero"


def narrar_raciocinio(m: Dict[str, Any]) -> str:
    """Traduz uma métrica de similaridade numa fala do próprio Adam, em primeira
    pessoa, em vez de uma lista de números."""
    contexto = (m.get("contexto_imitado") or "").strip()
    pensamento_bloco = (m.get("bloco_pensamento") or "").strip()
    frase = []
    if contexto:
        frase.append(f"Isso me lembrou de \"{contexto}\".")
    if pensamento_bloco:
        frase.append(f"Veio à mente: {pensamento_bloco}")
    frase.append(f"E {_confianca_em_palavras(m['overall_sim'])} ({m['overall_sim']:.0%} de familiaridade com algo que eu já vivi).")
    return " ".join(frase)


def render_match_metrics(metrics) -> None:
    """Mostra, dentro do próprio chat, o Adam narrando em primeira pessoa como
    chegou na resposta — em vez de uma lista de métricas cruas.

    Aceita uma narrativa única (resposta de uma parte só) ou uma lista
    (resposta combinada, com um raciocínio por parte da frase).
    """
    if not metrics:
        return
    lista = metrics if isinstance(metrics, list) else [metrics]
    if not lista:
        return
    with st.expander("💭 Como eu cheguei nisso"):
        for i, m in enumerate(lista):
            if len(lista) > 1:
                st.markdown(f"**Sobre a parte {i + 1}:**")
            st.markdown(narrar_raciocinio(m))


def similaridade_palavras(txt1: str, txt2: str) -> float:
    """Calcula similaridade baseada em interseção de palavras tokenizadas."""
    set1 = set(Token(txt1.lower()))
    set2 = set(Token(txt2.lower()))
    return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0.0


def semantic_text_similarity(txt1: str, txt2: str, dominio: str = "") -> float:
    """Similaridade ALNULU por token (ver alnulu_token_similarity) — respeita o
    INSEPA em vez de forçar o texto inteiro num vetor e medir cosseno."""
    if not txt1 and not txt2:
        return 0.0
    txt1_proc = apply_domain_lexicon(txt1, dominio)
    txt2_proc = apply_domain_lexicon(txt2, dominio)
    return alnulu_token_similarity(txt1_proc, txt2_proc)


def key_phrase_similarity(thought1: str, thought2: str, dominio: str = "") -> float:
    """Similaridade entre dois campos de 'pensamento_interno', comparando por
    FRASES (quebradas na pontuação), não por palavras soltas.

    Uma frase carrega seu contexto junto; por isso comparamos a melhor
    correspondência entre qualquer par de frases dos dois textos, em vez de
    tratar o pensamento como um único bloco ou como palavras isoladas do
    léxico inconsciente.
    """
    frases1 = split_frases_por_pontuacao(thought1)
    frases2 = split_frases_por_pontuacao(thought2)
    if not frases1 or not frases2:
        return semantic_text_similarity(thought1, thought2, dominio)
    return max(
        semantic_text_similarity(f1, f2, dominio)
        for f1 in frases1
        for f2 in frases2
    )


def _has_meaningful_value(value: Any) -> bool:
    """Retorna True quando um campo contém conteúdo real e não um placeholder genérico."""
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = normalize(value)
        if not cleaned:
            return False
        generic_tokens = {
            "nao definido",
            "não definido",
            "nao definida",
            "não definida",
            "sem contexto",
            "sem pensamento",
            "sem informacao",
            "sem informação",
            "n/a",
            "none",
            "null",
            "dado sem exatidão ou similaridade.",
        }
        return cleaned not in generic_tokens
    return True


def build_semantic_signature(bloco: Optional[dict]) -> Dict[str, Any]:
    """Constrói uma assinatura semântica do bloco usando texto, vars e multivars do INSEPA."""
    if not isinstance(bloco, dict):
        return {"text": "", "variants": [], "multivars": [], "tokens": []}

    entrada = bloco.get("entrada", {}) or {}
    saidas = bloco.get("saidas") or []
    saida0 = saidas[0] if saidas else {}

    variants = []
    multivars = []

    for field in ("texto", "reacao", "contexto", "pensamento_interno"):
        value = entrada.get(field, "")
        if isinstance(value, str):
            variants.extend(_extract_variants_from_text(value))

    for value in entrada.get("Multivars_Texto_Entrada", []) or []:
        if isinstance(value, str):
            multivars.append(normalize(value))

    for value in saida0.get("Multivars_Texto_Saida", []) or []:
        if isinstance(value, str):
            multivars.append(normalize(value))

    for value in saida0.get("textos", []) or []:
        if isinstance(value, str):
            variants.extend(_extract_variants_from_text(value))

    return {
        "text": normalize(entrada.get("texto", "")),
        "variants": sorted({normalize(v) for v in variants if normalize(v)}),
        "multivars": sorted({normalize(v) for v in multivars if normalize(v)}),
        "tokens": [normalize(token) for token in Token(entrada.get("texto", "")) if normalize(token)],
    }


def _extract_variants_from_text(value: str) -> List[str]:
    """Extrai palavras e variações de uma string com sintaxe [vars: ...]."""
    if not isinstance(value, str):
        return []

    variants = []
    for token in Token(value):
        palavra, vars_list = parse_bloco_template_with_vars(token)
        base = normalize(palavra)
        if base:
            variants.append(base)
        for var in vars_list:
            var_norm = normalize(var)
            if var_norm:
                variants.append(var_norm)
    return variants


def grounding_score(bloco: Optional[dict]) -> float:
    """Calcula quanto um bloco está fundamentado por campos concretos de entrada/saída."""
    if not isinstance(bloco, dict):
        return 0.0

    entrada = bloco.get("entrada", {}) or {}
    score = 0.0
    if _has_meaningful_value(entrada.get("texto")):
        score += 0.25
    if _has_meaningful_value(entrada.get("reacao")):
        score += 0.20
    if _has_meaningful_value(entrada.get("contexto")):
        score += 0.25
    if _has_meaningful_value(entrada.get("pensamento_interno")):
        score += 0.30

    saidas = bloco.get("saidas") or []
    if saidas:
        saida0 = saidas[0] or {}
        if _has_meaningful_value(saida0.get("reacao")):
            score += 0.05
        if saida0.get("textos"):
            score += 0.05
        if _has_meaningful_value(saida0.get("contexto")):
            score += 0.05

    return min(1.0, score)


def is_concrete_block(bloco: Optional[dict], threshold: float = 0.55) -> bool:
    """Classifica blocos como concretos quando há fundamentação real em contexto/pensamento."""
    if not isinstance(bloco, dict):
        return False

    entrada = bloco.get("entrada", {}) or {}
    has_context = _has_meaningful_value(entrada.get("contexto"))
    has_thought = _has_meaningful_value(entrada.get("pensamento_interno"))
    if not has_context and not has_thought:
        return False

    return grounding_score(bloco) >= threshold


def parse_quoted_response(prompt: str) -> str:
    """Parseia resposta, extraindo apenas o conteúdo entre aspas duplas se presente, senão retorna o prompt limpo."""
    match = _re.search(r'"([^"]*)"', prompt)
    if match:
        return match.group(1).strip()
    else:
        return prompt.strip()


def select_structured_autonomous_candidate(entrada_texto: str, entrada_reacao: str, entrada_contexto: str, entrada_pensamento: str, dominio: str, memoria: dict) -> Optional[dict]:
    """Seleciona um bloco base para geração autônoma apenas quando o bloco está bem alinhado com a estrutura INSEPA."""
    if not memoria or not isinstance(memoria, dict):
        return None
    universo = memoria.get("IM", {}).get(dominio, {})
    blocos = universo.get("blocos") or []
    if not blocos:
        return None

    candidates = []
    for bloco in blocos:
        if not isinstance(bloco, dict):
            continue
        entrada = bloco.get("entrada", {}) or {}
        if not entrada.get("texto"):
            continue

        text_match = semantic_text_similarity(entrada_texto, entrada.get("texto", ""), dominio)
        reaction_match = 1.0 if normalize(entrada_reacao or "") and normalize(entrada.get("reacao", "") or "") and normalize(entrada_reacao) == normalize(entrada.get("reacao", "")) else 0.0
        context_match = 1.0 if normalize(entrada_contexto) and normalize(entrada.get("contexto", "") or "") and normalize(entrada_contexto) == normalize(entrada.get("contexto", "")) else 0.0
        thought_match = 1.0 if normalize(entrada_pensamento) and normalize(entrada.get("pensamento_interno", "") or "") and normalize(entrada_pensamento) == normalize(entrada.get("pensamento_interno", "")) else 0.0
        signature = build_semantic_signature(bloco)
        variant_overlap = len(set(signature["variants"]) & set(_extract_variants_from_text(entrada_texto))) / max(1, len(set(signature["variants"]) | set(_extract_variants_from_text(entrada_texto))))
        multivar_overlap = len(set(signature["multivars"]) & set([normalize(v) for v in (entrada.get("Multivars_Texto_Entrada", []) or [])])) / max(1, len(set(signature["multivars"]) | set([normalize(v) for v in (entrada.get("Multivars_Texto_Entrada", []) or [])])))
        grounding = grounding_score(bloco)
        concrete = is_concrete_block(bloco)

        if not concrete:
            continue

        if text_match < 0.25 and reaction_match < 0.5 and context_match < 0.5 and thought_match < 0.5 and variant_overlap < 0.2 and multivar_overlap < 0.2:
            continue

        score = (
            0.35 * text_match
            + 0.20 * reaction_match
            + 0.15 * context_match
            + 0.10 * thought_match
            + 0.10 * variant_overlap
            + 0.10 * multivar_overlap
            + 0.10 * grounding
        )
        candidates.append((score, bloco))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_block = candidates[0]
    _, best_block = candidates[0]
    best_variant_signal = 0.0
    best_multivar_signal = 0.0
    for score, block in candidates:
        signature = build_semantic_signature(block)
        input_variants = set(_extract_variants_from_text(entrada_texto))
        block_variants = set(signature["variants"])
        block_multivars = set(signature["multivars"])
        variant_overlap = len(block_variants & input_variants) / max(1, len(block_variants | input_variants))
        multivar_overlap = len(block_multivars & set([normalize(v) for v in (block.get("entrada", {}).get("Multivars_Texto_Entrada", []) or [])])) / max(1, len(block_multivars | set([normalize(v) for v in (block.get("entrada", {}).get("Multivars_Texto_Entrada", []) or [])])))
        best_variant_signal = max(best_variant_signal, variant_overlap)
        best_multivar_signal = max(best_multivar_signal, multivar_overlap)

    strong_semantic_signal = best_variant_signal >= 0.25 or best_multivar_signal >= 0.25
    if best_score < 0.25 and not strong_semantic_signal:
        return None
    return best_block


def parse_text_reaction(prompt: str) -> tuple[str, str]:
    """Parseia o prompt para separar texto e reação, assumindo que a reação é a última palavra se for curta ou não alfanumérica."""
    words = prompt.split()
    if not words:
        return prompt, ""
    last = words[-1]
    if len(last) <= 3 or not last.isalnum():
        txt = ' '.join(words[:-1])
        reac = last
        return txt, reac
    else:
        return prompt, ""


def parse_bloco_template_with_vars(texto: str) -> tuple[str, list]:
    """Parseia formato: 'palavra[vars: var1, var2]' e extrai a palavra base + lista de vars.
    
    Exemplo: 'Olá[vars: Oi, Oie]' → ('Olá', ['Oi', 'Oie'])
    """
    import re
    
    # Padrão: palavra[vars: item1, item2, ...]
    match = _re.search(r'(\S+?)\s*\[vars:\s*(.*?)\]', texto, _re.IGNORECASE)
    
    if match:
        palavra = match.group(1)
        vars_str = match.group(2)
        vars_list = [v.strip() for v in vars_str.split(',') if v.strip()]
        return palavra, vars_list
    
    # Se não encontrar [vars:], retorna só a palavra/texto
    return texto.strip(), []


def clean_vars_syntax(texto: str) -> str:
    """Remove a sintaxe [vars: ...] de um texto, devolvendo apenas a parte limpa.
    
    Exemplo: 'Olá[vars: Oi, Oie]' → 'Olá'
             'Olá' → 'Olá'
    """
    # Usar regex para remover [vars: ...]
    cleaned = _re.sub(r'\s*\[vars:\s*.*?\]', '', texto, flags=_re.IGNORECASE)
    return cleaned.strip()


def extract_vars_from_tokens(tokens: List[str]) -> Dict[str, List[str]]:
    """Extrai vars de tokens que contêm [vars: ...]"""
    vars_dict = {}
    for token in tokens:
        palavra, vars_list = parse_bloco_template_with_vars(token)
        if vars_list:
            vars_dict[token] = vars_list
    return vars_dict


def create_bloco_template_example() -> str:
    """Gera template de exemplo para criar novos blocos com vars."""
    return """
╔════════════════════════════════════════════════════════╗
║         TEMPLATE PARA NOVO BLOCO INSEPA               ║
╚════════════════════════════════════════════════════════╝

Entrada: Olá[vars: Oi, Oie]
Multivars: E aí ?
Reação: :)[vars: Sorriso]
Multivars: Rosto sorridente
Contexto: Saudação cotidiana e formal
Pensamento Interno: Pelo que vejo o usuário está feliz, vou responder a altura

Saída: Olá[vars: Saudações]
Multivars: Saudações usuário querido
Reação: :D[vars: Feliz]
Multivars: Rosto feliz
Contexto: Cumprimento cotidiano

═══════════════════════════════════════════════════════

COMO USAR:
✓ Entrada: e Saída: são OBRIGATÓRIAS
✓ Palavra[vars: alternativa1, alternativa2] = palavra com variações
✓ Multivars: frase completa = frase inteira opcional
✓ Reação pode ter [vars: ...] e Multivars (palavras, emojis ou frases)
✓ Deixe em branco se não quiser adicionar um campo
✓ Contexto, Pensamento Interno são opcionais
✓ IM ID opcional (padrão: IM 1) - use "Índice mãe: 2" para outros IMs
✓ IM ID opcional (padrão: IM 1) - use "Índice mãe: 2" para outros IMs

EXEMPLO COM MÚLTIPLOS BLOCOS (separe por ═════):
Entrada: Oi
Saída: Olá!
═══════════════════════════════════════════════════════
Entrada: Tudo bem?
Saída: Tudo bem sim! E você?
"""




def infer(memoria: dict, dominio: str) -> None:
    """
    Interface de chat inovadora para inferência.
    """
    import os, torch, random
    # parse_text_reaction, normalize, ckpt_path, train, AdamSegmentado já disponíveis

    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    # Garantir que checkpoint existe (carregar do Mongo se necessário)
    ensure_checkpoint_exists(dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint — treine primeiro.")
        train(memoria, dominio)
        return

    data = torch.load(ckpt)
    if len(data) == 20:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         out_vocab_size, max_out_len,
         max_ng,
         vS, all_out_markers, idx_to_txt
        ) = data
        n_txt, n_emo, n_ctx = out_vocab_size, 1, 1  # defaults for old model
    elif len(data) == 19:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         out_vocab_size, max_out_len,
         max_ng,
         vS, all_out_markers
        ) = data
        n_txt, n_emo, n_ctx = out_vocab_size, 1, 1  # defaults for old model
        # Recriar idx_to_txt
        idx_to_txt = {v: k for k, v in vS.items()}
    elif len(data) == 18:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         out_vocab_size, max_out_len,
         max_ng,
         vS
        ) = data
        n_txt, n_emo, n_ctx = out_vocab_size, 1, 1  # defaults for old model
        # Recriar all_out_markers e idx_to_txt
        ds_temp = InsepaFieldDataset(memoria, dominio)
        all_out_markers = ds_temp.all_out_markers
        idx_to_txt = ds_temp.idx_to_txt
    elif len(data) == 17:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, n_emo, n_ctx,
         max_ng
        ) = data
        out_vocab_size = n_txt
        max_out_len = 10  # default
        max_ng = 3  # default
        # Recriar vocabulário de saída
        blocos = memoria["IM"][dominio]["blocos"]
        vS = {}
        for b in blocos:
            for saida in b["saidas"]:
                for texto in saida["textos"]:
                    for token in Token(texto):
                        vS[token] = vS.get(token, len(vS))
                reac = saida.get("reacao", "")
                if reac:
                    vS[reac] = vS.get(reac, len(vS))
                ctx = saida.get("contexto", "")
                for token in Token(ctx):
                    vS[token] = vS.get(token, len(vS))
        # Recriar all_out_markers e idx_to_txt
        ds_temp = InsepaFieldDataset(memoria, dominio)
        all_out_markers = ds_temp.all_out_markers
        idx_to_txt = ds_temp.idx_to_txt
    else:
        raise ValueError(f"Checkpoint has {len(data)} values, expected 17, 18, 19 or 20")

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE),
        nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        out_vocab_size=out_vocab_size, max_out_len=max_out_len,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    try:
        model.load_state_dict(state)
        model.v_txt = vS
        model.idx_to_txt = idx_to_txt
        model.all_out_markers = all_out_markers
    except RuntimeError as e:
        st.warning(f"⚠️ Checkpoint incompatível devido a mudanças na arquitetura: {e}. Retreinando...")
        train(memoria, dominio)
        return
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = st.session_state.inconsciente
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
        for bloco in blocos_inco:
            bloco_num = int(bloco["Bloco_id"])
            saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
            if saida_vals:
                ultimo_child_per_block[bloco_num] = max(saida_vals)
            else:
                ultimo_child_per_block[bloco_num] = 0.50



    # Mostrar nome do IM
    nome_im = memoria["IM"][dominio].get("nome", f"IM_{dominio}")
    genero = memoria["IM"][dominio].get("genero", "feminino")
    voz = memoria["IM"][dominio].get("voz", None)
    st.write(f"**Conversando com: {nome_im}**")

    # Inicializar histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "variation" not in st.session_state:
        st.session_state.variation = 0
    if "current_bloco" not in st.session_state:
        st.session_state.current_bloco = None
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "conversa_blocos" not in st.session_state:
        st.session_state.conversa_blocos = []

    x = None  # Inicializar x para evitar UnboundLocalError

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metrics"):
                render_match_metrics(message["metrics"])

    # Mostrar áudio se existir
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3')

    def featurize(tokens: List[str], max_len: int, vocab: dict, val_to_idx: dict, max_ng: int, dominio_int: int):
        # Palavras de verdade (via ALNULU), nunca marcador -- mesma lógica do
        # treino (InsepaFieldDataset/test_model), pra bater com o que o modelo
        # aprendeu. `tokens` já vem em texto puro (Token(txt)/Token(reac)...).
        ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
        ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
        val_idxs = [val_to_idx.get(t, 0) for t in tokens]
        n = len(tokens)
        vals = [(i / (n - 1) if n > 1 else 0.0) for i in range(n)]
        moms = [dominio_int] * n
        pos = [0.0] * n
        pad_ids = (max_len * max_ng) - len(ids)
        pad_vals = max_len - n
        ids += [0] * pad_ids
        val_idxs += [0] * pad_vals
        vals += [0.0] * pad_vals
        moms += [0] * pad_vals
        pos += [0.0] * pad_vals
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([val_idxs], dtype=torch.long),
            torch.tensor([vals], dtype=torch.float32),
            torch.tensor([moms], dtype=torch.long),
            torch.tensor([pos], dtype=torch.float32),
        )

    # Entrada do usuário
    if prompt := st.chat_input("Digite sua mensagem + reação (ex: Olá 😊)"):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        cmd = prompt.lower().strip()
        s = prompt.strip()
        txt, reac = parse_text_reaction(s)

        if cmd == "sair":
            st.session_state.messages.append({"role": "assistant", "content": "👋 Até mais!"})
            with st.chat_message("assistant"):
                st.markdown("👋 Até mais!")
            st.session_state.messages = []
            st.session_state.variation = 0
            st.session_state.current_bloco = None
            st.session_state.conversa_blocos = []
            return

        if should_run_harness_admin_automation():
            harness_result = run_harness_admin_step(dominio, prompt)
            st.session_state["last_harness_result"] = harness_result or {}
            if st.session_state.get("admin", False):
                if harness_result and harness_result.get("action") == "learned":
                    st.info(f"🛠️ Harness: aprendi um bloco novo para '{harness_result.get('learned_block_id')}'.")
                elif harness_result and harness_result.get("action") == "blocked":
                    st.warning("🛡️ Harness: bloco abstrato bloqueado; autorização do adm é necessária para aprender.")
                elif harness_result and harness_result.get("action") == "error":
                    st.error(f"⚠️ Harness falhou: {harness_result.get('error')}")

        if should_run_autonomous_mode():
            auto_result = try_autonomous_learning_from_prompt(txt, reac, "", "", dominio)
            if auto_result:
                st.session_state["last_autonomous_result"] = auto_result
                if auto_result.get("action") == "opiniao_registrada" and st.session_state.get("admin", False):
                    st.caption(f"🤖 Modo autônomo: fidelidade de {auto_result.get('score', 0):.0%} ao corpus — registrado como opinião, aguardando reforço da criadora.")
                elif auto_result.get("action") == "skipped_low_similarity" and st.session_state.get("admin", False):
                    st.caption(f"🤖 Modo autônomo: fidelidade de {auto_result.get('score', 0):.0%} ao corpus — abaixo do limiar, nada foi salvo.")

        if cmd == "reiniciar":
            st.session_state.messages.append({"role": "assistant", "content": "🔄 Conversa reiniciada. Histórico limpo."})
            with st.chat_message("assistant"):
                st.markdown("🔄 Conversa reiniciada. Histórico limpo.")
            st.session_state.messages = []
            st.session_state.variation = 0
            st.session_state.current_bloco = None
            st.session_state.conversa_blocos = []
            return

        if cmd == "insight" and st.session_state.current_bloco:
            bloco = st.session_state.current_bloco
            ep_txt = bloco["entrada"]["texto"]
            ep_reac = bloco["entrada"].get("reacao", "")
            # Usar o contexto da intenção (entrada), não o contexto da resposta
            contexto = bloco["entrada"].get("contexto", "")
            emoji = bloco["saidas"][0].get("reacao", "")
            texts = bloco["saidas"][0]["textos"]
            chosen = texts[st.session_state.variation]
            insight_msg = f"💡 De acordo com a expressão “{ep_txt}”, a reação “{ep_reac}” e o contexto “{contexto}”, conclui que “{chosen} {emoji}” é a resposta mais adequada."
            st.session_state.messages.append({"role": "assistant", "content": insight_msg})
            with st.chat_message("assistant"):
                st.markdown(insight_msg)
            # Armazenar a última resposta para like
            st.session_state.last_response = insight_msg
            st.session_state.last_bloco_id = str(bloco["bloco_id"])
            st.rerun()

        # Parse entrada usando INSEPA: encontrar bloco por reação no final, extrair txt/reac, matching por texto (incluindo vars e multivars) e reação
        s = prompt.strip()
        # Usar parse_text_reaction para definir reac inicialmente
        txt, reac = parse_text_reaction(s)
        match_metrics = None  # Métricas de similaridade, preenchidas só quando o match não é exato
        bloco = None
        for b in blocos:
            bloco_reac = b["entrada"].get("reacao", "").lower().strip()
            if bloco_reac and s.lower().strip().endswith(bloco_reac):
                reac = bloco_reac
                txt = s[:-len(bloco_reac)].rstrip()
                # Coletar textos possíveis: base, multivars, variações com vars
                textos_possiveis = [b["entrada"]["texto"]] + b["entrada"].get("Multivars_Texto_Entrada", []) + [variar_texto(b["entrada"]["texto"], b, dominio, 'entrada')]
                # Matching normalizado por texto e reação do bloco
                if any(normalize(t) == normalize(txt) for t in textos_possiveis) and reac == bloco_reac:
                    bloco = b
                    break

        # Mostrar passos como no teste
        # st.write("### 1. Match Exato")  # Removido para chat limpo
        if bloco:
            # st.success(f"✅ Match exato encontrado: '{txt} {reac}'")  # Removido
            pass
        else:
            # st.warning(f"❌ Nenhum match exato para '{txt} {reac}'")  # Removido
            pass

        # Se não encontrou match exato, tentar similaridade ALNULU
        # st.write("### 2. Similaridade ALNULU")  # Removido para chat limpo
        if bloco is None and txt and reac:
            # Dividir input em partes baseadas em reações encontradas, como no teste
            partes = []
            remaining = s  # Usar o input original para incluir reações
            while remaining:
                found = False
                for b in blocos:
                    bloco_reac = b["entrada"].get("reacao", "").strip()
                    if bloco_reac and len(bloco_reac) > 1 and bloco_reac in remaining:
                        idx = remaining.find(bloco_reac)
                        if idx > 0:
                            parte = remaining[:idx + len(bloco_reac)].strip()
                            partes.append(parte)
                            remaining = remaining[idx + len(bloco_reac):].strip().lstrip(".,!? ")
                            found = True
                            break
                if not found:
                    if remaining.strip():
                        partes.append(remaining.strip())
                    break
            if not partes:
                partes = [s]
            # Não adicionar reac global
            
            respostas_combinadas = []
            metrics_combinadas = []
            for parte in partes:
                # Usar parse_text_reaction para cada parte
                parte_clean, parte_reac = parse_text_reaction(parte)

                ctx_imitado, pensamento_imitado = infer_context_thought_by_imitation(parte_clean, parte_reac, dominio)
                sem_raiz_proxima = not ctx_imitado and not pensamento_imitado
                similares = [] if sem_raiz_proxima else retrieve_similar_blocks_alnulu(parte_clean, parte_reac, ctx_imitado, pensamento_imitado, dominio, top_k=1)
                if sem_raiz_proxima or not similares:
                    # Nenhum assunto no corpus é próximo o bastante -- melhor admitir do que inventar.
                    resposta = "Estou alucinando... Vamos aprender juntos?"
                else:
                    sim_score, bloco_sim = similares[0]
                    if (sim_score < 0.5 and not is_concrete_block(bloco_sim)) or not raiz_coerente(ctx_imitado, bloco_sim, dominio):
                        # Blocos sem fundamentação, ou cuja saída não pertence à mesma raiz do contexto buscado, não devem gerar resposta confiante.
                        resposta = "Estou alucinando... Vamos aprender juntos?"
                    else:
                        resposta_texto = bloco_sim['saidas'][0]['textos'][0]
                        resposta_reacao = bloco_sim['saidas'][0].get('reacao', '')
                        metrics_parte = explain_similarity_match(parte_clean, parte_reac, ctx_imitado, pensamento_imitado, bloco_sim, dominio)
                        resposta = montar_resposta_por_confianca(resposta_texto, resposta_reacao, metrics_parte)
                        metrics_combinadas.append(metrics_parte)
                respostas_combinadas.append(resposta)

            if respostas_combinadas:
                response = ' '.join(respostas_combinadas)
                bloco = "combined"
                match_metrics = metrics_combinadas
                # Se todas as partes alucinaram, ativar Cerbero
                if all(r == "Estou alucinando... Vamos aprender juntos?" for r in respostas_combinadas):
                    bloco = None
                    match_metrics = None
        elif bloco is None:
            # st.write("### 2. Similaridade ALNULU")  # Removido
            ctx_imitado, pensamento_imitado = infer_context_thought_by_imitation(txt, reac, dominio)
            # Sem raiz próxima no corpus (assunto divergente), não há o que imitar/buscar.
            similares = [] if (not ctx_imitado and not pensamento_imitado) else retrieve_similar_blocks_alnulu(txt, reac, ctx_imitado, pensamento_imitado, dominio, top_k=1)
            if similares:
                sim_score, bloco_sim = similares[0]
                if (sim_score < 0.5 and not is_concrete_block(bloco_sim)) or not raiz_coerente(ctx_imitado, bloco_sim, dominio):
                    # Blocos sem fundamentação, ou cuja saída não pertence à mesma raiz do contexto buscado, não devem gerar resposta confiante.
                    response = "Estou alucinando... Vamos aprender juntos?"
                    bloco = None  # Para ativar Cerbero
                else:
                    resposta_texto = bloco_sim['saidas'][0]['textos'][0]
                    resposta_reacao = bloco_sim['saidas'][0].get('reacao', '')
                    metrics_atual = explain_similarity_match(txt, reac, ctx_imitado, pensamento_imitado, bloco_sim, dominio)
                    resposta = montar_resposta_por_confianca(resposta_texto, resposta_reacao, metrics_atual)
                    response = resposta
                    # Detectar alucinação interna: se resposta base é "A" ou "O", ativar Cerbero
                    if resposta_texto.strip() in ["A", "O"]:
                        bloco = None  # Tratar como não encontrado para aprendizado
                    else:
                        bloco = bloco_sim
                        match_metrics = metrics_atual
            else:
                bloco = None

        if bloco and bloco != "combined":
            # Determinar se é match exato ou similar
            is_exato = any(normalize(t) == normalize(txt) for t in [bloco["entrada"]["texto"]] + bloco["entrada"].get("Multivars_Texto_Entrada", []) + [variar_texto(bloco["entrada"]["texto"], bloco, dominio, 'entrada')]) and reac == bloco["entrada"].get("reacao", "")
            resposta_texto = bloco['saidas'][0]['textos'][0]
            resposta_reacao = bloco['saidas'][0].get('reacao', '')
            if is_exato:
                # Match exato: resposta completa
                response = resposta_texto + (" " + resposta_reacao if resposta_reacao else "")
            else:
                # Similar: resposta completa
                response = resposta_texto + (" " + resposta_reacao if resposta_reacao else "")
            # Aplicar variação se disponível
            variations_from_blocks = bloco["saidas"][0]["textos"] + bloco["saidas"][0].get("Multivars_Texto_Saida", [])
            resposta_variada = variar_texto_rag(bloco, dominio, variations_from_blocks)
            if resposta_variada:
                response = resposta_variada + (" " + resposta_reacao if resposta_reacao else "")
            else:
                response = response
            # Detectar alucinação interna: se resposta base é "A" ou "O", ativar Cerbero
            if resposta_texto.strip() in ["A", "O"]:
                bloco = None  # Tratar como não encontrado para aprendizado
            st.session_state.messages.append({"role": "assistant", "content": response, "metrics": match_metrics})
            with st.chat_message("assistant"):
                st.markdown(response)
                if match_metrics:
                    render_match_metrics(match_metrics)
            # Armazenar a última resposta para like
            st.session_state.last_response = response
            st.session_state.last_bloco_id = str(bloco["bloco_id"])
            # Definir chosen para TTS
            chosen = response
            # Adicionar bloco ao histórico se novo
            if bloco not in st.session_state.conversa_blocos:
                st.session_state.conversa_blocos.append(bloco)
            st.session_state.current_bloco = bloco
            st.session_state.last_valid = True
            st.rerun()

        elif bloco == "combined":
            # Resposta combinada já definida
            st.session_state.messages.append({"role": "assistant", "content": response, "metrics": match_metrics})
            with st.chat_message("assistant"):
                st.markdown(response)
                if match_metrics:
                    render_match_metrics(match_metrics)
            st.session_state.last_response = response
            st.rerun()

        # Detectar bloco de fallback e ativar Cerbero
        if isinstance(bloco, dict) and bloco["entrada"]["texto"] == "Dado sem padrão" and bloco["entrada"].get("reacao") == "Não definida" and bloco["entrada"].get("contexto") == "Dado sem exatidão ou similaridade.":
            bloco = None  # Tratar como não encontrado para ativar aprendizado

        # Se nenhum bloco encontrado: ninguém além da criadora define contexto/pensamento.
        # Qualquer usuário pode deixar uma OPINIÃO (texto+emoção); vira dado concreto só
        # quando a criadora confirmar via Cérbero (já protegido por SENHA_CRIAR_BLOCOS).
        if bloco is None:
            if "cerbero_step" not in st.session_state:
                txt, reac = parse_text_reaction(s)
                st.session_state.cerbero_step = "verify_credentials"
                st.session_state.new_input = txt
                st.session_state.new_reac = reac

                # Reflexão antes de desistir: mesmo sem bater o piso de confiança, o
                # Adam pode ter um candidato fraco -- ele é livre pra errar o bloco,
                # desde que reproduza a resposta e PERGUNTE se o contexto corresponde,
                # em vez de responder com confiança ou fingir que não sabe de nada.
                candidato = melhor_candidato_fraco(txt, dominio)

                if candidato:
                    registrar_opiniao_pendente(txt, reac, dominio, candidato)
                    bloco_c = candidato["bloco"]
                    resposta_tentativa = bloco_c["saidas"][0]["textos"][0]
                    ai_msg = (
                        f'🤔 Não sei o que isso significa. Mas me traz memórias de algo parecido. '
                        f'Se eu estiver certo, a resposta seria: "{resposta_tentativa}"\n\n'
                        f'Entendo sua opinião. Mas preciso de confirmações, antes de tratar sua '
                        f'resposta como concreta.\n\nPor favor, confirme suas credenciais '
                        f'se você é a criadora:'
                    )
                else:
                    # Nada de aproximar por forma/distância quando não há candidato nenhum:
                    # é simplesmente desconhecido. O ALNULU só deixa o Adam "ver" a forma; é
                    # o INSEPA (registro exato) que decide se algo é reconhecido. Terceiro
                    # sinal, mais fraco ainda: a rede neural treinada -- nunca vira resposta,
                    # só engrossa a opinião registrada pra criadora avaliar.
                    palpite = gerar_palpite_rede_neural(txt, reac, dominio)
                    registrar_opiniao_pendente(txt, reac, dominio, candidato, palpite_rede_neural=palpite)
                    ai_msg = ('🔍 Não sei o que isso significa, e não me traz memória nenhuma de '
                              'algo parecido.\n\nEntendo sua opinião. Mas preciso de confirmações, '
                              'antes de tratar sua resposta como concreta.\n\nPor favor, confirme '
                              'suas credenciais se você é a criadora:')
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "verify_credentials":
                credential = prompt.strip()
                if credential == SENHA_CRIAR_BLOCOS:
                    st.session_state.cerbero_step = "collect_text_confirmation"
                    ai_msg = f'Fantástico! Olá TM. Pode por favor confirmar? "{st.session_state.new_input}" é um texto correto?'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    ai_msg = f'Desculpe. Seu acesso não é permitido por questões de segurança. Sinta-se a vontade para entrar em outros universos, ou aguarde por nossas atualizações. Até mais!'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    # Reset para recusar o aprendizado
                    for key in ["cerbero_step", "new_input", "new_reac"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            elif st.session_state.cerbero_step == "collect_text_confirmation":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y", "correto", "certo", "ok"]:
                    st.session_state.cerbero_step = "collect_reaction_confirmation"
                    ai_msg = f'Maravilhoso! Então eu presumo que "{st.session_state.new_reac}" seja uma reação. Correto?'
                elif confirmation in ["não", "nao", "n", "no"]:
                    st.session_state.cerbero_step = "edit_text_input"
                    ai_msg = 'Sem problemas -- qual é o texto correto?'
                else:
                    # Nem confirmação nem negação clara -- nunca resetar tudo por isso,
                    # só pedir de novo (a criadora não deveria ter que recomeçar do zero
                    # só porque a resposta não bateu com "sim"/"não" exatamente).
                    ai_msg = f'Não entendi. "{st.session_state.new_input}" é o texto correto? Responda "sim" ou "não".'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "edit_text_input":
                st.session_state.new_input = parse_quoted_response(prompt)
                st.session_state.cerbero_step = "collect_reaction_confirmation"
                ai_msg = f'Certo! Então eu presumo que "{st.session_state.new_reac}" seja uma reação. Correto?'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "collect_reaction_confirmation":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y", "correto", "certo", "ok"]:
                    st.session_state.cerbero_step = "collect_context"
                    ai_msg = f'Incrível! Qual é o contexto ou situação em que "{st.session_state.new_input}" com emoção "{st.session_state.new_reac}" se aplica?'
                elif confirmation in ["não", "nao", "n", "no"]:
                    st.session_state.cerbero_step = "edit_reacao_input"
                    ai_msg = 'Sem problemas -- qual é a reação correta?'
                else:
                    ai_msg = f'Não entendi. "{st.session_state.new_reac}" é a reação correta? Responda "sim" ou "não".'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "edit_reacao_input":
                _, st.session_state.new_reac = parse_text_reaction(prompt)
                if not st.session_state.new_reac:
                    st.session_state.new_reac = prompt.strip()
                st.session_state.cerbero_step = "collect_context"
                ai_msg = f'Certo! Qual é o contexto ou situação em que "{st.session_state.new_input}" com emoção "{st.session_state.new_reac}" se aplica?'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "collect_context":
                contexto = parse_quoted_response(prompt)
                if '"' in prompt:
                    st.session_state.new_contexto = contexto
                    st.session_state.cerbero_step = "collect_thought"
                    ai_msg = f'Ótimo! Agora que temos a expressão "{st.session_state.new_input}" ligada à emoção "{st.session_state.new_reac}" e o contexto "{st.session_state.new_contexto}". O quê devo pensar a respeito do assunto?'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.temp_contexto = contexto
                    st.session_state.cerbero_step = "confirm_context"
                    ai_msg = f'O contexto é "{contexto}"? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "confirm_context":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.new_contexto = st.session_state.temp_contexto
                    st.session_state.cerbero_step = "collect_thought"
                    ai_msg = f'Perfeito! O quê devo pensar sobre "{st.session_state.new_input}" que é ligado à emoção "{st.session_state.new_reac}" no contexto "{st.session_state.new_contexto}"?'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.cerbero_step = "edit_context"
                    ai_msg = f'Ok, digite o contexto correto:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "edit_context":
                contexto = parse_quoted_response(prompt)
                st.session_state.new_contexto = contexto
                st.session_state.cerbero_step = "collect_thought"
                ai_msg = f'Ótimo! Agora que temos a expressão "{st.session_state.new_input}" ligada à emoção "{st.session_state.new_reac}" e o contexto "{st.session_state.new_contexto}". O quê devo pensar a respeito do assunto?'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "collect_thought":
                pensamento = parse_quoted_response(prompt)
                if '"' in prompt:
                    st.session_state.new_pensamento = pensamento
                    st.session_state.cerbero_step = "ask_add_entrada_phrase"
                    ai_msg = f'"{pensamento}". Quer adicionar uma frase alternativa para entrada? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.temp_pensamento = pensamento
                    st.session_state.cerbero_step = "confirm_thought"
                    ai_msg = f'O pensamento é "{pensamento}"? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "confirm_thought":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.new_pensamento = st.session_state.temp_pensamento
                    st.session_state.cerbero_step = "ask_add_entrada_phrase"
                    ai_msg = f' Quer adicionar uma frase alternativa para entrada? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.cerbero_step = "edit_thought"
                    ai_msg = f'Ok, digite o pensamento correto:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "edit_thought":
                pensamento = parse_quoted_response(prompt)
                st.session_state.new_pensamento = pensamento
                st.session_state.cerbero_step = "ask_add_entrada_phrase"
                ai_msg = f' "{pensamento}". Quer adicionar uma frase alternativa para entrada? Responda "sim" ou "não".'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "ask_add_entrada_phrase":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.cerbero_step = "collect_entrada_phrase"
                    ai_msg = f'Ok, digite a frase alternativa para entrada:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.new_multivars_entrada = []
                    st.session_state.cerbero_step = "generate_dynamic_response"
                    # Gerar proposta dinâmica
                    proposta_autonoma = generate_autonomous_block(st.session_state.new_input, st.session_state.new_reac, st.session_state.new_contexto, st.session_state.new_pensamento, dominio, memoria, st.session_state.get("new_multivars_entrada", []), [])
                    # Parsear a saída da proposta para obter a resposta dinâmica
                    saida_texto = proposta_autonoma.split("1. ")[1].split("\n")[0].strip() if "1. " in proposta_autonoma else "Resposta dinâmica gerada."
                    ai_msg = f'De acordo com a minha reflexão sobre o contexto, a emoção e o texto que me enviou, cheguei a conclusão de que "{saida_texto}" é a ideal. Está de acordo? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "collect_entrada_phrase":
                frase = parse_quoted_response(prompt)
                if '"' in prompt:
                    st.session_state.new_multivars_entrada = [frase]
                    st.session_state.cerbero_step = "generate_dynamic_response"
                    # Gerar proposta dinâmica
                    proposta_autonoma = generate_autonomous_block(st.session_state.new_input, st.session_state.new_reac, st.session_state.new_contexto, st.session_state.new_pensamento, dominio, memoria, st.session_state.get("new_multivars_entrada", []), [])
                    # Parsear a saída da proposta para obter a resposta dinâmica
                    saida_texto = proposta_autonoma.split("1. ")[1].split("\n")[0].strip() if "1. " in proposta_autonoma else "Resposta dinâmica gerada."
                    ai_msg = f' "{frase}". De acordo com a minha reflexão, "{saida_texto}" é a ideal. Está de acordo? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.temp_frase_entrada = frase
                    st.session_state.cerbero_step = "confirm_entrada_phrase"
                    ai_msg = f'A frase alternativa para entrada é "{frase}"? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "confirm_entrada_phrase":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.new_multivars_entrada = [st.session_state.temp_frase_entrada]
                    st.session_state.cerbero_step = "generate_dynamic_response"
                    # Gerar proposta dinâmica
                    proposta_autonoma = generate_autonomous_block(st.session_state.new_input, st.session_state.new_reac, st.session_state.new_contexto, st.session_state.new_pensamento, dominio, memoria, st.session_state.get("new_multivars_entrada", []), [])
                    # Parsear a saída da proposta para obter a resposta dinâmica
                    saida_texto = proposta_autonoma.split("1. ")[1].split("\n")[0].strip() if "1. " in proposta_autonoma else "Resposta dinâmica gerada."
                    ai_msg = f'De acordo com a minha reflexão sobre o contexto, a emoção e o texto que me enviou, cheguei a conclusão de que "{saida_texto}" é a ideal. Está de acordo? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.cerbero_step = "edit_entrada_phrase"
                    ai_msg = f'Ok, digite a frase alternativa correta:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "edit_entrada_phrase":
                frase = parse_quoted_response(prompt)
                st.session_state.new_multivars_entrada = [frase]
                st.session_state.cerbero_step = "generate_dynamic_response"
                # Gerar proposta dinâmica
                proposta_autonoma = generate_autonomous_block(st.session_state.new_input, st.session_state.new_reac, st.session_state.new_contexto, st.session_state.new_pensamento, dominio, memoria, st.session_state.get("new_multivars_entrada", []), [])
                # Parsear a saída da proposta para obter a resposta dinâmica
                saida_texto = proposta_autonoma.split("1. ")[1].split("\n")[0].strip() if "1. " in proposta_autonoma else "Resposta dinâmica gerada."
                ai_msg = f'✅ Frase corrigida: "{frase}". De acordo com a minha reflexão, "{saida_texto}" é a ideal. Está de acordo? Responda "sim" ou "não".'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "generate_dynamic_response":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    # Aceitar e criar o bloco
                    proposta_autonoma = generate_autonomous_block(st.session_state.new_input, st.session_state.new_reac, st.session_state.new_contexto, st.session_state.new_pensamento, dominio, memoria, st.session_state.get("new_multivars_entrada", []), [])
                    try:
                        generate_block_from_template(memoria, proposta_autonoma)
                        # Agora que o bloco foi criado, encontrar o bloco recém-criado e gerar resposta
                        blocos = memoria["IM"][dominio]["blocos"]
                        bloco_novo = blocos[-1]  # Último bloco adicionado
                        variations_from_blocks = bloco_novo["saidas"][0]["textos"] + bloco_novo["saidas"][0].get("Multivars_Texto_Saida", [])
                        response = variar_texto_rag(bloco_novo, dominio, variations_from_blocks)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Armazenar a última resposta para like
                        st.session_state.last_response = response
                        st.session_state.last_bloco_id = str(bloco_novo["bloco_id"])
                        # Definir chosen para TTS
                        chosen = response
                        # Adicionar bloco_novo ao histórico se novo
                        if bloco_novo not in st.session_state.conversa_blocos:
                            st.session_state.conversa_blocos.append(bloco_novo)
                        st.session_state.current_bloco = bloco_novo
                        st.session_state.last_valid = True
                        # Reset Cérbero
                        for key in ["cerbero_step", "new_input", "new_reac", "new_pensamento"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    except Exception as e:
                        ai_msg = f"❌ Erro ao criar bloco autônomo: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg)
                        st.rerun()
                else:
                    st.session_state.cerbero_step = "ask_new_output"
                    ai_msg = f'Ok, digite uma nova saída ideal:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "ask_new_output":
                nova_saida = parse_quoted_response(prompt)
                if '"' in prompt:
                    st.session_state.new_saida_custom = nova_saida
                    st.session_state.cerbero_step = "ask_add_saida_phrase"
                    ai_msg = f'✅ Nova saída coletada: "{nova_saida}". Quer acrescentar uma frase alternativa para saída? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.temp_nova_saida = nova_saida
                    st.session_state.cerbero_step = "confirm_new_output"
                    ai_msg = f'A nova saída é "{nova_saida}"? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "confirm_new_output":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.new_saida_custom = st.session_state.temp_nova_saida
                    st.session_state.cerbero_step = "ask_add_saida_phrase"
                    ai_msg = f'Quer acrescentar uma frase alternativa para saída? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.cerbero_step = "edit_new_output"
                    ai_msg = f'Ok, digite a nova saída correta:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "edit_new_output":
                nova_saida = parse_quoted_response(prompt)
                st.session_state.new_saida_custom = nova_saida
                st.session_state.cerbero_step = "ask_add_saida_phrase"
                ai_msg = f'✅ Saída corrigida: "{nova_saida}". Quer acrescentar uma frase alternativa para saída? Responda "sim" ou "não".'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "ask_add_saida_phrase":
                confirmation = parse_quoted_response(prompt).lower().strip()
                if confirmation in ["sim", "s", "yes", "y"]:
                    st.session_state.cerbero_step = "collect_saida_phrase"
                    ai_msg = f'Ok, digite a frase alternativa para saída:'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    st.session_state.new_multivars_saida = []
                    # Criar bloco com nova saída
                    proposta_custom = f"""Índice mãe: {dominio}

Entrada: {st.session_state.new_input}

Reação: {st.session_state.new_reac}

Contexto: {st.session_state.new_contexto}

Pensamento Interno: {st.session_state.new_pensamento}

Saída:

1. {st.session_state.new_saida_custom}

Reação: 🤖

Contexto: Resposta customizada
"""
                    try:
                        generate_block_from_template(memoria, proposta_custom)
                        # Agora que o bloco foi criado, encontrar o bloco recém-criado e gerar resposta
                        blocos = memoria["IM"][dominio]["blocos"]
                        bloco_novo = blocos[-1]  # Último bloco adicionado
                        variations_from_blocks = bloco_novo["saidas"][0]["textos"] + bloco_novo["saidas"][0].get("Multivars_Texto_Saida", [])
                        response = variar_texto_rag(bloco_novo, dominio, variations_from_blocks)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Armazenar a última resposta para like
                        st.session_state.last_response = response
                        st.session_state.last_bloco_id = str(bloco_novo["bloco_id"])
                        # Definir chosen para TTS
                        chosen = response
                        # Adicionar bloco_novo ao histórico se novo
                        if bloco_novo not in st.session_state.conversa_blocos:
                            st.session_state.conversa_blocos.append(bloco_novo)
                        st.session_state.current_bloco = bloco_novo
                        st.session_state.last_valid = True
                        # Reset Cérbero
                        for key in ["cerbero_step", "new_input", "new_reac", "new_pensamento", "new_saida_custom"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    except Exception as e:
                        ai_msg = f"❌ Erro ao criar bloco custom: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg)
                        st.rerun()
            elif st.session_state.cerbero_step == "collect_saida_phrase":
                frase_saida = parse_quoted_response(prompt)
                if frase_saida:
                    if '"' in prompt:
                        st.session_state.new_multivars_saida = [frase_saida]
                        # Criar bloco com nova saída e multivars
                        proposta_custom = f"""Índice mãe: {dominio}

Entrada: {st.session_state.new_input}

Reação: {st.session_state.new_reac}

Contexto: {st.session_state.new_contexto}

Pensamento Interno: {st.session_state.new_pensamento}

Multivars_Texto_Entrada: {" | ".join(st.session_state.get("new_multivars_entrada", []))}

Saída:

1. {st.session_state.new_saida_custom}

Multivars_Texto_Saida: {" | ".join(st.session_state.new_multivars_saida)}

Reação: 🤖

Contexto: Resposta customizada com multivars
"""
                        try:
                            generate_block_from_template(memoria, proposta_custom)
                            # Agora que o bloco foi criado, encontrar o bloco recém-criado e gerar resposta
                            blocos = memoria["IM"][dominio]["blocos"]
                            bloco_novo = blocos[-1]  # Último bloco adicionado
                            variations_from_blocks = bloco_novo["saidas"][0]["textos"] + bloco_novo["saidas"][0].get("Multivars_Texto_Saida", [])
                            response = variar_texto_rag(bloco_novo, dominio, variations_from_blocks)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            # Armazenar a última resposta para like
                            st.session_state.last_response = response
                            st.session_state.last_bloco_id = str(bloco_novo["bloco_id"])
                            # Definir chosen para TTS
                            chosen = response
                            # Adicionar bloco_novo ao histórico se novo
                            if bloco_novo not in st.session_state.conversa_blocos:
                                st.session_state.conversa_blocos.append(bloco_novo)
                            st.session_state.current_bloco = bloco_novo
                            st.session_state.last_valid = True
                            # Reset Cérbero
                            for key in ["cerbero_step", "new_input", "new_reac", "new_pensamento", "new_saida_custom"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                        except Exception as e:
                            ai_msg = f"❌ Erro ao criar bloco custom com multivars: {e}"
                            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                            with st.chat_message("assistant"):
                                st.markdown(ai_msg)
                            st.rerun()
                    else:
                        st.session_state.new_multivars_saida = [frase_saida]
                        st.session_state.cerbero_step = "confirm_saida_phrase"
                        ai_msg = f'A frase alternativa para saída é "{frase_saida}". Está correto? Responda "sim" ou "não".'
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg)
                        st.rerun()
                else:
                    st.session_state.cerbero_step = "edit_saida_phrase"
                    ai_msg = f'❌ Erro de parsing. Digite a frase alternativa para saída sem aspas.'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "confirm_saida_phrase":
                if prompt.lower().strip() in ["sim", "s", "yes", "y", "correto", "ok"]:
                    # Criar bloco com nova saída e multivars
                    proposta_custom = f"""Índice mãe: {dominio}

Entrada: {st.session_state.new_input}

Reação: {st.session_state.new_reac}

Contexto: {st.session_state.new_contexto}

Pensamento Interno: {st.session_state.new_pensamento}

Multivars_Texto_Entrada: {" | ".join(st.session_state.get("new_multivars_entrada", []))}

Saída:

1. {st.session_state.new_saida_custom}

Multivars_Texto_Saida: {" | ".join(st.session_state.new_multivars_saida)}

Reação: 🤖

Contexto: Resposta customizada com multivars
"""
                    try:
                        generate_block_from_template(memoria, proposta_custom)
                        # Agora que o bloco foi criado, encontrar o bloco recém-criado e gerar resposta
                        blocos = memoria["IM"][dominio]["blocos"]
                        bloco_novo = blocos[-1]  # Último bloco adicionado
                        variations_from_blocks = bloco_novo["saidas"][0]["textos"] + bloco_novo["saidas"][0].get("Multivars_Texto_Saida", [])
                        response = variar_texto_rag(bloco_novo, dominio, variations_from_blocks)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Armazenar a última resposta para like
                        st.session_state.last_response = response
                        st.session_state.last_bloco_id = str(bloco_novo["bloco_id"])
                        # Definir chosen para TTS
                        chosen = response
                        # Adicionar bloco_novo ao histórico se novo
                        if bloco_novo not in st.session_state.conversa_blocos:
                            st.session_state.conversa_blocos.append(bloco_novo)
                        st.session_state.current_bloco = bloco_novo
                        st.session_state.last_valid = True
                        # Reset Cérbero
                        for key in ["cerbero_step", "new_input", "new_reac", "new_pensamento", "new_saida_custom"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    except Exception as e:
                        ai_msg = f"❌ Erro ao criar bloco custom com multivars: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg)
                        st.rerun()
                elif prompt.lower().strip() in ["não", "n", "no", "nao", "errado", "incorreto"]:
                    st.session_state.cerbero_step = "edit_saida_phrase"
                    ai_msg = f'Ok, vamos corrigir. Digite a frase alternativa para saída sem aspas.'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                else:
                    ai_msg = f'Por favor, responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
            elif st.session_state.cerbero_step == "edit_saida_phrase":
                frase_saida = parse_quoted_response(prompt)
                if frase_saida:
                    st.session_state.new_multivars_saida = [frase_saida]
                    # Criar bloco com nova saída e multivars
                    proposta_custom = f"""Índice mãe: {dominio}

Entrada: {st.session_state.new_input}

Reação: {st.session_state.new_reac}

Contexto: {st.session_state.new_contexto}

Pensamento Interno: {st.session_state.new_pensamento}

Multivars_Texto_Entrada: {" | ".join(st.session_state.get("new_multivars_entrada", []))}

Saída:

1. {st.session_state.new_saida_custom}

Multivars_Texto_Saida: {" | ".join(st.session_state.new_multivars_saida)}

Reação: 🤖

Contexto: Resposta customizada com multivars
"""
                    try:
                        generate_block_from_template(memoria, proposta_custom)
                        # Agora que o bloco foi criado, encontrar o bloco recém-criado e gerar resposta
                        blocos = memoria["IM"][dominio]["blocos"]
                        bloco_novo = blocos[-1]  # Último bloco adicionado
                        variations_from_blocks = bloco_novo["saidas"][0]["textos"] + bloco_novo["saidas"][0].get("Multivars_Texto_Saida", [])
                        response = variar_texto_rag(bloco_novo, dominio, variations_from_blocks)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Armazenar a última resposta para like
                        st.session_state.last_response = response
                        st.session_state.last_bloco_id = str(bloco_novo["bloco_id"])
                        # Definir chosen para TTS
                        chosen = response
                        # Adicionar bloco_novo ao histórico se novo
                        if bloco_novo not in st.session_state.conversa_blocos:
                            st.session_state.conversa_blocos.append(bloco_novo)
                        st.session_state.current_bloco = bloco_novo
                        st.session_state.last_valid = True
                        # Reset Cérbero
                        for key in ["cerbero_step", "new_input", "new_reac", "new_pensamento", "new_saida_custom"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    except Exception as e:
                        ai_msg = f"❌ Erro ao criar bloco custom com multivars: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg)
                        st.rerun()
                else:
                    ai_msg = f'❌ Ainda erro. Tente novamente sem aspas.'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()


    # Botão Enter para gerar variações se há bloco atual e última entrada foi válida
    if st.session_state.current_bloco and st.session_state.last_valid:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Enter", key="enter_button"):
                if x is not None:
                    with torch.no_grad():
                        out = model(x)  # tgt=None para geração autoregressiva

                    # out["out"] é (batch, max_out_len, 1) -- regressão (MSE no treino),
                    # um float por posição, não logits de classe. decode_tokens já
                    # espera exatamente esses floats direto (via float_to_idx).
                    generated_ids = out["out"][0].squeeze(-1)

                    # Decodificar IDs para texto usando o vocabulário do modelo
                    generated_responses = model.decode_tokens(generated_ids, bloco, dominio)
                    generated_text = generated_responses[0] if generated_responses else ""

                    # Aplicar variações inconscientes para criatividade na saída
                    bloco = st.session_state.current_bloco
                    if generated_text:
                        response = variar_texto(generated_text, bloco, dominio)
                        # Adicionar frase extra de Multivars_Texto_Saida se disponível
                        multivars_saida = bloco["saidas"][0].get("Multivars_Texto_Saida", [])
                        if multivars_saida:
                            extra = random.choice(multivars_saida)
                            response += " " + extra
                        # Aplicar variações ao response completo
                        response = variar_texto(response, bloco, dominio)
                    else:
                        response = "Resposta gerada vazia."

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # Armazenar a última resposta para like
                    st.session_state.last_response = generated_text
                    st.session_state.last_bloco_id = str(st.session_state.current_bloco["bloco_id"])
                    st.session_state.last_button = "Enter"
                    # Incrementar variação para próxima vez (removido, agora random)
                    # Generate speech - sistema híbrido: Edge TTS para premium, gTTS para leves, pyttsx3 para outras
                    chosen = response
                    if TTS_AVAILABLE and voz:
                        try:
                            if voz.startswith('edge-') and EDGE_TTS_AVAILABLE:
                                # Usar Edge TTS para vozes premium
                                voice_name = voz.split('-', 1)[1]
                                
                                edge_voice_map = {
                                    'pt-br': 'pt-BR-FranciscaNeural',  # Feminina
                                    'pt-pt': 'pt-PT-RaquelNeural',     # Feminina
                                    'en': 'en-US-AriaNeural',          # Feminina
                                    'en-us': 'en-US-AriaNeural',       # Feminina
                                    'en-gb': 'en-GB-SoniaNeural',      # Feminina
                                    'es': 'es-ES-ElviraNeural',        # Feminina
                                    'es-us': 'es-US-PalomaNeural',     # Feminina
                                    'fr': 'fr-FR-DeniseNeural',        # Feminina
                                    'de': 'de-DE-KatjaNeural',         # Feminina
                                    'it': 'it-IT-ElsaNeural',          # Feminina
                                    'ja': 'ja-JP-NanamiNeural',        # Feminina
                                    'ko': 'ko-KR-SunHiNeural',         # Feminina
                                    'ru': 'ru-RU-SvetlanaNeural',      # Feminina
                                    'ar': 'ar-SA-ZariyahNeural',       # Feminina
                                    'hi': 'hi-IN-SwaraNeural',         # Feminina
                                    'female': 'en-US-AriaNeural',      # Feminina
                                    'pt-br-male': 'pt-BR-AntonioNeural',    # Masculina
                                    'en-male': 'en-US-AndrewNeural',        # Masculina
                                    'es-male': 'es-ES-AlvaroNeural',        # Masculina
                                    'fr-male': 'fr-FR-HenriNeural',         # Masculina
                                    'de-male': 'de-DE-ConradNeural',        # Masculina
                                    'it-male': 'it-IT-DiegoNeural',         # Masculina
                                    'ja-male': 'ja-JP-KeitaNeural',         # Masculina
                                    'ko-male': 'ko-KR-InJoonNeural',        # Masculina
                                    'ru-male': 'ru-RU-DmitryNeural',        # Masculina
                                    'ar-male': 'ar-SA-HamedNeural',         # Masculina
                                    'hi-male': 'hi-IN-MadhurNeural',        # Masculina
                                    'male': 'en-US-ZiraNeural',             # Masculina (nota: Zira é feminino, mas usado como padrão masculino)
                                }
                                
                                selected_voice = edge_voice_map.get(voice_name, 'en-US-AriaNeural')
                                
                                import asyncio
                                import io
                                
                                async def generate_edge_audio():
                                    communicate = edge_tts.Communicate(chosen, selected_voice)
                                    audio_data = b""
                                    async for chunk in communicate.stream():
                                        if chunk["type"] == "audio":
                                            audio_data += chunk["data"]
                                    return audio_data
                                
                                audio_bytes = asyncio.run(generate_edge_audio())
                                
                                if audio_bytes and len(audio_bytes) > 0:
                                    st.session_state.last_audio = audio_bytes
                                    st.audio(st.session_state.last_audio, format='audio/mp3')
                                    st.success(f"🎵 Áudio gerado com Edge TTS '{selected_voice}': {len(audio_bytes)} bytes")
                                else:
                                    st.error("❌ Falha ao gerar arquivo de áudio com Edge TTS.")
                            elif voz.startswith('gtts-') and GTTS_AVAILABLE:
                                # Usar gTTS para vozes leves
                                lang_code = voz.split('-', 1)[1]
                                
                                lang_map = {
                                    'pt-br': 'pt-br', 'pt-pt': 'pt-pt', 'en': 'en', 'en-us': 'en', 'en-gb': 'en',
                                    'es': 'es', 'es-us': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'ja': 'ja',
                                    'ko': 'ko', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
                                }
                                
                                if lang_code in lang_map:
                                    from gtts import gTTS
                                    import io
                                    
                                    tts = gTTS(text=chosen, lang=lang_map[lang_code], slow=False)
                                    audio_buffer = io.BytesIO()
                                    tts.write_to_fp(audio_buffer)
                                    audio_buffer.seek(0)
                                    audio_bytes = audio_buffer.read()
                                    
                                    if audio_bytes and len(audio_bytes) > 0:
                                        st.session_state.last_audio = audio_bytes
                                        st.audio(st.session_state.last_audio, format='audio/mp3')
                                        st.success(f"🎵 Áudio gerado com gTTS '{lang_code}': {len(audio_bytes)} bytes")
                                    else:
                                        st.error("❌ Falha ao gerar arquivo de áudio com gTTS.")
                                else:
                                    st.warning(f"Idioma '{lang_code}' não suportado pelo gTTS.")
                            else:
                                # Usar pyttsx3 para outras vozes
                                try:
                                    import pyttsx3
                                    engine = pyttsx3.init()
                                    voices = engine.getProperty('voices')
                                    if voz.startswith('tortoise-'):
                                        voice_name = voz.split('-', 1)[1]
                                        if 'emma' in voice_name.lower() or 'female' in voice_name.lower():
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                        else:
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                    else:
                                        selected_voice = next((v for v in voices if v.name == voz), voices[0] if voices else None)

                                    if not selected_voice and not voz.startswith('tortoise-'):
                                        if genero == "masculino":
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                        elif genero == "feminino":
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                        else:
                                            selected_voice = random.choice(voices) if voices else None

                                    if selected_voice:
                                        engine.setProperty('voice', selected_voice.id)
                                        engine.setProperty('rate', 180)
                                        engine.setProperty('volume', 0.9)
                                        engine.say(chosen)
                                        engine.runAndWait()
                                        st.success(f"🎵 Áudio reproduzido com sucesso! (Voz: {selected_voice.name})")
                                    else:
                                        st.warning("⚠️ Nenhuma voz do sistema encontrada.")
                                except RuntimeError:
                                    st.warning("⚠️ pyttsx3 não disponível neste ambiente. TTS pulado.")

                        except Exception as e:
                            import traceback
                            st.error(f"Erro ao reproduzir áudio: {str(e)}")
                            st.error("Detalhes do erro:")
                            st.code(traceback.format_exc())
                            st.warning("TTS falhou, mas a conversa continua normalmente.")
        with col2:
            if st.button("💡 Insight", key="insight_button"):
                bloco = st.session_state.current_bloco
                insight_msg = generate_insight(bloco, st.session_state.get("last_response"))
                if insight_msg:
                    st.session_state.messages.append({"role": "assistant", "content": insight_msg})
                    with st.chat_message("assistant"):
                        st.markdown(insight_msg)
                    # Armazenar a última resposta para like (mas like só para Enter)
                    st.session_state.last_response = insight_msg
                    st.session_state.last_bloco_id = str(bloco["bloco_id"])
                    st.session_state.last_button = "Insight"
                    # Generate speech - sistema híbrido
                    if TTS_AVAILABLE:
                        try:
                            if voz and voz.startswith('edge-') and EDGE_TTS_AVAILABLE:
                                # Usar Edge TTS para vozes premium
                                voice_name = voz.split('-', 1)[1]
                                
                                edge_voice_map = {
                                    'pt-br': 'pt-BR-FranciscaNeural',  # Feminina
                                    'pt-pt': 'pt-PT-RaquelNeural',     # Feminina
                                    'en': 'en-US-AriaNeural',          # Feminina
                                    'en-us': 'en-US-AriaNeural',       # Feminina
                                    'en-gb': 'en-GB-SoniaNeural',      # Feminina
                                    'es': 'es-ES-ElviraNeural',        # Feminina
                                    'es-us': 'es-US-PalomaNeural',     # Feminina
                                    'fr': 'fr-FR-DeniseNeural',        # Feminina
                                    'de': 'de-DE-KatjaNeural',         # Feminina
                                    'it': 'it-IT-ElsaNeural',          # Feminina
                                    'ja': 'ja-JP-NanamiNeural',        # Feminina
                                    'ko': 'ko-KR-SunHiNeural',         # Feminina
                                    'ru': 'ru-RU-SvetlanaNeural',      # Feminina
                                    'ar': 'ar-SA-ZariyahNeural',       # Feminina
                                    'hi': 'hi-IN-SwaraNeural',         # Feminina
                                    'female': 'en-US-AriaNeural',      # Feminina
                                    'pt-br-male': 'pt-BR-AntonioNeural',    # Masculina
                                    'en-male': 'en-US-AndrewNeural',        # Masculina
                                    'es-male': 'es-ES-AlvaroNeural',        # Masculina
                                    'fr-male': 'fr-FR-HenriNeural',         # Masculina
                                    'de-male': 'de-DE-ConradNeural',        # Masculina
                                    'it-male': 'it-IT-DiegoNeural',         # Masculina
                                    'ja-male': 'ja-JP-KeitaNeural',         # Masculina
                                    'ko-male': 'ko-KR-InJoonNeural',        # Masculina
                                    'ru-male': 'ru-RU-DmitryNeural',        # Masculina
                                    'ar-male': 'ar-SA-HamedNeural',         # Masculina
                                    'hi-male': 'hi-IN-MadhurNeural',        # Masculina
                                    'male': 'en-US-ZiraNeural',             # Masculina (nota: Zira é feminino, mas usado como padrão masculino)
                                }
                                
                                selected_voice = edge_voice_map.get(voice_name, 'en-US-AriaNeural')
                                
                                import asyncio
                                import io
                                
                                async def generate_edge_audio():
                                    communicate = edge_tts.Communicate(insight_msg, selected_voice)
                                    audio_data = b""
                                    async for chunk in communicate.stream():
                                        if chunk["type"] == "audio":
                                            audio_data += chunk["data"]
                                    return audio_data
                                
                                audio_bytes = asyncio.run(generate_edge_audio())
                                
                                if audio_bytes and len(audio_bytes) > 0:
                                    st.session_state.last_audio = audio_bytes
                                    st.audio(st.session_state.last_audio, format='audio/mp3')
                                    st.success(f"🎵 Áudio gerado com Edge TTS '{selected_voice}': {len(audio_bytes)} bytes")
                                else:
                                    st.error("❌ Falha ao gerar arquivo de áudio com Edge TTS.")
                            elif voz and voz.startswith('gtts-') and GTTS_AVAILABLE:
                                # Usar gTTS para vozes leves
                                lang_code = voz.split('-', 1)[1]
                                
                                lang_map = {
                                    'pt-br': 'pt-br', 'pt-pt': 'pt-pt', 'en': 'en', 'en-us': 'en', 'en-gb': 'en',
                                    'es': 'es', 'es-us': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'ja': 'ja',
                                    'ko': 'ko', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
                                }
                                
                                if lang_code in lang_map:
                                    from gtts import gTTS
                                    import io
                                    
                                    tts = gTTS(text=insight_msg, lang=lang_map[lang_code], slow=False)
                                    audio_buffer = io.BytesIO()
                                    tts.write_to_fp(audio_buffer)
                                    audio_buffer.seek(0)
                                    audio_bytes = audio_buffer.read()
                                    
                                    if audio_bytes and len(audio_bytes) > 0:
                                        st.session_state.last_audio = audio_bytes
                                        st.audio(st.session_state.last_audio, format='audio/mp3')
                                        st.success(f"🎵 Áudio gerado com gTTS '{lang_code}': {len(audio_bytes)} bytes")
                                    else:
                                        st.error("❌ Falha ao gerar arquivo de áudio com gTTS.")
                                else:
                                    st.warning(f"Idioma '{lang_code}' não suportado pelo gTTS.")
                            else:
                                # Usar pyttsx3 para outras vozes
                                try:
                                    import pyttsx3
                                    engine = pyttsx3.init()
                                    voices = engine.getProperty('voices')
                                    if voz.startswith('tortoise-'):
                                        voice_name = voz.split('-', 1)[1]
                                        if 'emma' in voice_name.lower() or 'female' in voice_name.lower():
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                        else:
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                    else:
                                        selected_voice = next((v for v in voices if v.name == voz), voices[0] if voices else None)

                                    if not selected_voice and not voz.startswith('tortoise-'):
                                        if genero == "masculino":
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                        elif genero == "feminino":
                                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                        else:
                                            selected_voice = random.choice(voices) if voices else None

                                    if selected_voice:
                                        engine.setProperty('voice', selected_voice.id)
                                        engine.setProperty('rate', 180)
                                        engine.setProperty('volume', 0.9)
                                        engine.say(insight_msg)
                                        engine.runAndWait()
                                        st.success(f"🎵 Áudio reproduzido com sucesso! (Voz: {selected_voice.name})")
                                    else:
                                        st.warning("⚠️ Nenhuma voz do sistema encontrada.")
                                except RuntimeError:
                                    st.warning("⚠️ pyttsx3 não disponível neste ambiente. TTS pulado.")
                        except Exception as e:
                            import traceback
                            st.error(f"Erro ao reproduzir áudio: {str(e)}")
                            st.error("Detalhes do erro:")
                            st.code(traceback.format_exc())
                            st.warning("TTS falhou, mas a conversa continua normalmente.")

        st.rerun()

    # Botão de Like se há última resposta do Enter
    if "last_response" in st.session_state and "last_bloco_id" in st.session_state and st.session_state.get("last_button") == "Enter":
        if st.button("👍 Like na última resposta"):
            bloco_id = st.session_state.last_bloco_id
            response = st.session_state.last_response
            if bloco_id not in st.session_state.likes:
                st.session_state.likes[bloco_id] = {}
            if response not in st.session_state.likes[bloco_id]:
                st.session_state.likes[bloco_id][response] = 0
            st.session_state.likes[bloco_id][response] += 1
            reinforcement = record_human_reinforcement(bloco_id, response, liked=True)
            st.success(f"👍 Curtido! Agora '{response}' tem mais chances de aparecer.")
            
            # Fine-tuning imediato com o like
            fine_tune_online(memoria, dominio, bloco_id, response)
            st.success(f"✅ Modelo ajustado com o feedback! Confiança do bloco: {reinforcement['reinforcement']['confidence_score']:.2f}.")
            st.rerun()


def record_human_reinforcement(bloco_id: str, response: str, liked: bool = True, human_name: str = "") -> dict:
    """Registra reforço humano para um bloco e atualiza sua confiança percebida.

    A regra de peso é:
    - 66% para o que o Adam aprende com a criadora Thaís D' Mariano, especialmente dados fundamentados com E+RE+CE+PIDE;
    - 12% para opiniões de usuários, quando os dados forem abstratos sem fundamento em E+RE;
    - 23% para a autonomia do Adam decidir sozinho, mas somente após aprender com a criadora.
    """
    if not bloco_id:
        return {"block_id": None, "reinforcement": {"human_likes": 0, "human_dislikes": 0, "confidence_score": 0.0}, "status": "rejected"}

    creator_names = {
        "thais d' mariano",
        "thais d mariano",
        "thais mariano",
        "thaís d' mariano",
        "thaís d mariano",
        "thaís mariano",
    }
    normalized_name = (human_name or "").strip().lower()
    is_creator = (
        not normalized_name
        or normalized_name in creator_names
        or normalized_name.startswith("thais")
        or normalized_name.startswith("thaís")
    )

    if "reinforcements" not in st.session_state:
        st.session_state.reinforcements = {}

    reinforcement = st.session_state.reinforcements.setdefault(
        bloco_id,
        {
            "human_likes": 0,
            "human_dislikes": 0,
            "responses": {},
            "confidence_score": 0.5,
            "creator": "Thaís D' Mariano",
            "weights": {"creator": 0.66, "other_humans": 0.12, "autonomy": 0.23},
            "source": "creator" if is_creator else "other_human",
        },
    )

    if liked:
        reinforcement["human_likes"] += 1
    else:
        reinforcement["human_dislikes"] += 1

    if response:
        reinforcement["responses"][response] = reinforcement["responses"].get(response, 0) + 1

    if is_creator:
        reinforcement["confidence_score"] = min(1.0, 0.66 + 0.12 * reinforcement["human_likes"] + 0.03)
        reinforcement["source"] = "creator"
        status = "accepted"
    else:
        reinforcement["confidence_score"] = min(1.0, 0.12 + 0.02 * reinforcement["human_likes"])
        reinforcement["source"] = "other_human"
        status = "accepted"

    reinforcement["autonomy_weight"] = 0.23
    reinforcement["autonomy_ready"] = bool(reinforcement["human_likes"] > 0 and is_creator)

    memoria = st.session_state.get("memoria", {})
    for dominio, universo in memoria.get("IM", {}).items():
        for bloco in universo.get("blocos", []):
            if str(bloco.get("bloco_id")) == str(bloco_id):
                bloco["reinforcement"] = reinforcement
                break
        else:
            continue
        break

    st.session_state.memoria = memoria
    auto_save_state()
    return {
        "block_id": bloco_id,
        "reinforcement": copy.deepcopy(reinforcement),
        "status": status,
    }


def weighted_choice(variations, bloco_id):
    """Escolhe uma variação com pesos baseados em likes."""
    if bloco_id not in st.session_state.likes:
        st.session_state.likes[bloco_id] = {}
    weights = []
    for var in variations:
        count = st.session_state.likes[bloco_id].get(var, 0)
        weights.append(max(1, count + 1))  # mínimo 1 para não zerar
    return random.choices(variations, weights=weights, k=1)[0]


def get_bloco_from_text(entrada: str, dominio: str) -> dict:
    memoria = st.session_state.memoria
    if dominio not in memoria["IM"]:
        return None
    blocos = memoria["IM"][dominio]["blocos"]
    for bloco in blocos:
        if bloco["entrada"]["texto"] == entrada:
            return bloco
    return None


def get_unconscious_vars_for_block(bloco: dict, dominio: str) -> dict:
    inconsciente = st.session_state.inconsciente
    if dominio not in inconsciente.get("INCO", {}):
        return {}
    blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
    bloco_inco = next((b for b in blocos_inco if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return {}
    vars_dict = {}
    for data in bloco_inco["Entrada"].values():
        token = data["token"]
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            vars_dict[token] = vars_list
    for data in bloco_inco["SAÍDA"].values():
        token = data["token"]
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            vars_dict[token] = vars_list
    return vars_dict


def should_run_harness_admin_automation() -> bool:
    """Retorna True quando o admin habilitou o assistente automático do harness no chat."""
    return bool(st.session_state.get("admin", False) and st.session_state.get("harness_auto_admin", True))


def should_run_autonomous_mode() -> bool:
    """Retorna True quando o modo autônomo do Adam está habilitado para o admin."""
    if st.session_state.get("autonomous_mode", True) is False:
        return False
    return bool(st.session_state.get("admin", False) or st.session_state.get("autonomous_mode", True))


def try_autonomous_learning_from_prompt(entrada_texto: str, entrada_reacao: str, entrada_contexto: str, entrada_pensamento: str, dominio: str) -> Optional[dict]:
    """Tenta criar um bloco novo e aprender autonomamente a partir do prompt do usuário."""
    if not should_run_autonomous_mode():
        return None
    if not st.session_state.get("auto_learn", True):
        return None
    if not entrada_texto and not entrada_reacao:
        return None

    memoria = st.session_state.get("memoria", {})
    if not isinstance(memoria, dict):
        return None

    # Portão de fidelidade ao corpus: só vale a pena refletir sobre o que já reconhece
    # como suficientemente próximo dos padrões que já existem (>= CORPUS_SIMILARITY_THRESHOLD).
    score = corpus_similarity_score(
        entrada_texto,
        entrada_reacao,
        entrada_contexto or "",
        entrada_pensamento or "",
        dominio,
    )
    if score < CORPUS_SIMILARITY_THRESHOLD:
        return {"action": "skipped_low_similarity", "score": score}

    # Mesmo com boa fidelidade ao corpus, ninguém além da criadora define contexto e
    # pensamento -- isso nunca mais persiste um bloco sozinho. Vira opinião com o
    # candidato mais provável anexado, pra criadora reforçar como var/multivar ou
    # como bloco novo de verdade (ver submenu_opinioes).
    candidato = melhor_candidato_fraco(entrada_texto, dominio)
    registrar_opiniao_pendente(entrada_texto, entrada_reacao, dominio, candidato)
    return {"action": "opiniao_registrada", "score": score, "candidato": candidato}


def run_harness_admin_step(dominio: str, raw_input: str) -> Optional[dict]:
    """Executa o harness como assistente automático no fluxo do chat do Streamlit."""
    if not should_run_harness_admin_automation():
        return None

    try:
        import adam_harness as harness_module
        lovely_module = harness_module.load_lovely_module()
        harness_module.ensure_runtime_state(lovely_module)
        return harness_module.run_learning_cycle(
            lovely_module,
            dominio,
            raw_input,
            learn_on_failure=True,
            allow_abstract_learning=st.session_state.get("admin_override_learning", False),
        )
    except Exception as exc:
        st.session_state["harness_last_error"] = str(exc)
        return {"action": "error", "error": str(exc)}


def generate_cartesian_responses(texts: list, unconscious_vars_dict: dict) -> list:
    # Para frases "Olá [A] [B]."
    a_words = set()
    b_words = set()
    for text in texts:
        tokens = Token(text)
        if len(tokens) > 1:
            a_words.add(tokens[1])
        if len(tokens) > 2:
            b_words.add(tokens[2])
    a_variations = list(a_words)
    for word in a_words:
        if word in unconscious_vars_dict:
            a_variations.extend(unconscious_vars_dict[word])
    b_variations = list(b_words)
    for word in b_words:
        if word in unconscious_vars_dict:
            b_variations.extend(unconscious_vars_dict[word])
    b_variations = [""] + b_variations  # incluir vazio
    combinations = []
    for a in a_variations:
        for b in b_variations:
            if b:
                combinations.append(f"Olá {a} {b}.")
            else:
                combinations.append(f"Olá {a}.")
    return combinations


def atualizar_inconsciente_para_im(memoria: dict, dominio: str) -> None:
    """Atualiza o inconsciente para o IM selecionado."""
    if dominio not in memoria["IM"]:
        return
    im_data = memoria["IM"][dominio]
    inconsciente = st.session_state.inconsciente
    if dominio not in inconsciente.get("INCO", {}):
        inconsciente.setdefault("INCO", {})[dominio] = {
            "NOME": im_data.get("nome", f"IM_{dominio}"),
            "Ultimo child": im_data.get("ultimo_child", f"{dominio}.0"),
            "Blocos": []
        }
        salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def test_model(memoria: dict, dominio: str) -> None:
    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint — treine primeiro.");
        return

    data = torch.load(ckpt)
    if len(data) == 20:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, max_out_len, max_ng,
         vS, all_out_markers, idx_to_txt) = data
    elif len(data) == 19:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, max_out_len, max_ng,
         vS, all_out_markers) = data
        idx_to_txt = {v: k for k, v in vS.items()}
    elif len(data) == 18:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, max_out_len, max_ng,
         vS) = data
        all_out_markers = None
        idx_to_txt = {v: k for k, v in vS.items()}
    elif len(data) == 17:
        (state,
         maxE, maxRE, maxCE, maxPIDE,
         mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE,
         n_txt, max_out_len, max_ng) = data
        # Recriar vS para compatibilidade
        from dataset import Dataset
        ds = Dataset(memoria, dominio)
        vS = ds.out_vocab
        all_out_markers = None
        idx_to_txt = {v: k for k, v in vS.items()}
    else:
        raise ValueError(f"Checkpoint inválido: esperado 17, 18, 19 ou 20 valores, encontrado {len(data)}")

    out_vocab_size = n_txt
    n_emo = 1
    n_ctx = 1

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE), nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        out_vocab_size=out_vocab_size, max_out_len=max_out_len,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    try:
        model.load_state_dict(state)
        model.v_txt = vS
        model.idx_to_txt = idx_to_txt
        if all_out_markers:
            model.all_out_markers = all_out_markers
    except RuntimeError as e:
        st.warning(f"⚠️ Checkpoint incompatível devido a mudanças na arquitetura: {e}. Treine primeiro.")
        return
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = st.session_state.inconsciente
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
        for bloco in blocos_inco:
            bloco_num = int(bloco["Bloco_id"])
            saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
            if saida_vals:
                ultimo_child_per_block[bloco_num] = max(saida_vals)
            else:
                ultimo_child_per_block[bloco_num] = 0.50
    st.write(f"📊 Teste em lote — Domínio {dominio} ({len(blocos)} blocos)")

    # Inicializar acumuladores para métricas
    total_samples = 0
    acc_txt = 0.0
    acc_emo = 0.0
    acc_ctx = 0.0
    mse_pos = 0.0

    for b in blocos:
        max_val = ultimo_child_per_block.get(b["bloco_id"], 0.50)

        dominio_int = int(dominio) if str(dominio).isdigit() else 0

        def featurize(field, max_len, vocab, val_to_idx, max_ng):
            # Palavra de verdade (via ALNULU), nunca o marcador -- mesma lógica
            # do InsepaFieldDataset, pra não repetir o erro de treinar em cima
            # de números de posição que nunca se repetem (Camada 1).
            tokens = palavras_do_campo(b, field)
            ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
            val_idxs = [val_to_idx.get(t, 0) for t in tokens]
            n = len(tokens)
            vals = [(i / (n - 1) if n > 1 else 0.0) for i in range(n)]
            moms = [dominio_int] * n
            pos = [0.0] * n
            pad_ids = (max_len * max_ng) - len(ids)
            pad_vals = max_len - n
            ids += [0] * pad_ids
            val_idxs += [0] * pad_vals
            vals += [0.0] * pad_vals
            moms += [0] * pad_vals
            pos += [0.0] * pad_vals
            return (
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([val_idxs], dtype=torch.long),
                torch.tensor([vals], dtype=torch.float32),
                torch.tensor([moms], dtype=torch.long),
                torch.tensor([pos], dtype=torch.float32),
            )

        E_ids, E_val_idxs, E_val, E_mom, E_pos = featurize("E", maxE, vE, val_to_idx_E, max_ng)
        RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos = featurize("RE", maxRE, vRE, val_to_idx_RE, max_ng)
        CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE", maxCE, vCE, val_to_idx_CE, max_ng)
        PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

        x = {
            "E": E_ids, "E_val": E_val, "E_mom": E_mom, "E_pos": E_pos, "E_val_idx": E_val_idxs,
            "RE": RE_ids, "RE_val": RE_val, "RE_mom": RE_mom, "RE_pos": RE_pos, "RE_val_idx": RE_val_idxs,
            "CE": CE_ids, "CE_val": CE_val, "CE_mom": CE_mom, "CE_pos": CE_pos, "CE_val_idx": CE_val_idxs,
            "PIDE": PI_ids, "PIDE_val": PI_val, "PIDE_mom": PI_mom, "PIDE_pos": PI_pos, "PIDE_val_idx": PI_val_idxs,
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

        if st.session_state.get("admin", False):
            st.write(f"\n❏ Bloco_id={b['bloco_id']} Entrada: {b['entrada']['texto']} {b['entrada']['reacao']}")
            st.write(f"   Texto pred: {pred_text} | True: {true_texts}")
            st.write(f"   Emoji pred: {true_emo if pred_emo == 0 else 'Outro'} | True: {true_emo}")
            st.write(f"   Contexto pred: {true_ctx if pred_ctx == 0 else 'Outro'} | True: {true_ctx}")
            st.write(f"   Posição pred: {pred_pos:.4f} | True: {true_pos:.4f}")
            st.write(f"   Acurácia Texto: {acc_txt_block:.1f}")
            st.write(f"   Acurácia Emoji: {acc_emo_block:.1f}")
            st.write(f"   Acurácia Contexto: {acc_ctx_block:.1f}")
            st.write(f"   MSE Posição: {mse_pos_block:.4f}")

    # Calcular médias
    if total_samples > 0:
        acc_txt /= total_samples
        acc_emo /= total_samples
        acc_ctx /= total_samples
        mse_pos /= total_samples

        if not st.session_state.get("admin", False):
            st.info("📋 Detalhes dos testes disponíveis apenas para administradores. As métricas gerais são exibidas abaixo.")

        st.write("\n📈 Métricas Gerais:")
        st.write(f"Acurácia Texto: {acc_txt:.2%}")
        st.write(f"Acurácia Emoji: {acc_emo:.2%}")
        st.write(f"Acurácia Contexto: {acc_ctx:.2%}")
        st.write(f"MSE Posição: {mse_pos:.4f}")
    else:
        st.write("Nenhum bloco para testar.")


## INSEPA_CLI
def prompt_dominio(action: str, memoria: dict) -> str:
    """Lista IMs disponíveis e permite escolher um para a ação especificada.

    O IM "0" é o princípio de tudo (o "Big Bang" do Adam) -- só acessível em
    modo administrador. Os universos derivados (1, 2, 3...) que o Adam for
    criando com o tempo ficam abertos pra qualquer um."""
    ims = list(memoria.get("IM", {}).keys())
    if not st.session_state.get("admin", False):
        ims = [im for im in ims if im != "0"]
    if not ims:
        st.error("❌ Nenhum IM encontrado. Crie um primeiro.")
        return ""

    if action == "conversar":
        st.write("Selecione o Universo para conversar:")
    else:
        st.write(f"\n--- Escolher IM para {action} ---")
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            num_blocos = len(memoria["IM"][im_id].get("blocos", []))
            st.write(f"- {im_id}: {nome} ({num_blocos} blocos)")

    dom = st.selectbox(f"Escolha o Universo para {action}", ims, key=f"dominio_{action}")
    return dom


def create_new_im(memoria: dict) -> None:
    """Cria um novo IM (Índice Mãe) vazio."""
    im_id = st.text_input("Índice mãe para o novo IM:", key="new_im_id")
    if not im_id.isdigit():
        st.error("❌ Índice mãe deve ser um número.")
        return
    im_id = int(im_id)
    if str(im_id) in memoria.get("IM", {}):
        st.error(f"❌ IM {im_id} já existe.")
        return
    nome = st.text_input("Nome do IM (opcional):", key="new_im_name") or f"IM_{im_id}"
    genero = st.selectbox("Gênero do IM:", ["masculino", "feminino", "não binário", "outro"], key="new_im_genero")
    voz = None
    if TTS_AVAILABLE:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            voice_options = [v.name for v in voices if v]
        except RuntimeError:
            voice_options = []
        gtts_voices = ['gtts-pt-br', 'gtts-pt-pt', 'gtts-en', 'gtts-en-us', 'gtts-en-gb', 'gtts-es', 'gtts-es-us', 'gtts-fr', 'gtts-de', 'gtts-it', 'gtts-ja', 'gtts-ko', 'gtts-ru', 'gtts-ar', 'gtts-hi']
        coqui_voices = ['tts_models/pt/cv/vits', 'tts_models/en/ljspeech/tacotron2-DDC_ph']
        voice_options += gtts_voices + [f"coqui-{cv}" for cv in coqui_voices]
        voz = st.selectbox("Voz preferida (opcional):", ["Automático"] + voice_options, key="new_im_voz")
        if voz == "Automático":
            voz = None
    if st.button("Criar IM"):
        im_data = {
            "nome": nome,
            "genero": genero,
            "ultimo_child": f"{im_id}.0",
            "blocos": []
        }
        if voz:
            im_data["voz"] = voz
        memoria.setdefault("IM", {})[str(im_id)] = im_data
        salvar_json(ARQUIVO_MEMORIA, memoria)
        st.success(f"✅ IM {im_id} criado: {nome} ({genero})" + (f" - Voz: {voz}" if voz else ""))


def submenu_im(memoria: dict, inconsciente: dict) -> None:
    st.subheader("🛠️ Gerenciar IMs e Blocos")
    st.write("Áudio disponível. Ouça a voz do personagem escolhido agora")
    st.write(f"gTTS: {GTTS_AVAILABLE}")
    st.write(f"Python executable: {sys.executable}")
    sub_opc = st.selectbox("Escolha uma opção:", [
        "📋 Visualizar IMs e Blocos",
        "➕ Criar novo IM",
        "🔧 Gerar bloco a partir de template INSEPA",
        "🗑️ Apagar bloco",
        "🚮 Apagar IM",
        "⚙️ Alimentar vars dos tokens",
        "✏️ Editar nomes de IMs",
        "💾 Backup JSON",
        "⬅️ Voltar ao menu principal"
    ], key="submenu_im")

    if sub_opc == "📋 Visualizar IMs e Blocos":
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.info("Nenhum IM encontrado. Crie um primeiro.")
            return
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            genero = memoria["IM"][im_id].get("genero", "não definido")
            voz = memoria["IM"][im_id].get("voz", None)
            num_blocos = len(memoria["IM"][im_id].get("blocos", []))
            with st.expander(f"📁 IM {im_id}: {nome} ({genero})" + (f" - Voz: {voz}" if voz else "") + f" ({num_blocos} blocos)"):
                blocos = memoria["IM"][im_id].get("blocos", [])
                if blocos:
                    # Tabela de Entrada
                    data_entrada = [
                        {
                            "ID": b["bloco_id"],
                            "Entrada": b["entrada"]["texto"],
                            "Reação": b["entrada"].get("reacao", ""),
                            "Contexto": b["entrada"].get("contexto", ""),
                            "Pensamento Interno": b["entrada"].get("pensamento_interno", "")
                        } for b in blocos
                    ]
                    st.subheader("📥 Entradas dos Blocos")
                    st.dataframe(data_entrada, use_container_width=True, column_config={
                        "Entrada": st.column_config.TextColumn("Entrada", width=None),
                        "Reação": st.column_config.TextColumn("Reação", width=None),
                        "Contexto": st.column_config.TextColumn("Contexto", width=None),
                        "Pensamento Interno": st.column_config.TextColumn("Pensamento Interno", width=None)
                    })
                    
                    # Multivars de Entrada
                    st.subheader("🔄 Multivars de Entrada (Frases Completas)")
                    multivars_entrada_data = [
                        {
                            "ID": b["bloco_id"],
                            "Multivars_Texto_Entrada": "\n".join(b["entrada"].get("Multivars_Texto_Entrada", [])) or "Nenhum"
                        } for b in blocos
                    ]
                    st.dataframe(multivars_entrada_data, use_container_width=True, column_config={
                        "Multivars_Texto_Entrada": st.column_config.TextColumn("Multivars_Texto_Entrada", width=None)
                    })
                    
                    # Tabela de Saída
                    data_saida = [
                        {
                            "ID": b["bloco_id"],
                            "Saídas": "\n".join(b["saidas"][0]["textos"]),
                            "Reação": b["saidas"][0].get("reacao", ""),
                            "Contexto": b["saidas"][0].get("contexto", "")
                        } for b in blocos
                    ]
                    st.subheader("📤 Saídas dos Blocos")
                    st.dataframe(data_saida, use_container_width=True, column_config={
                        "Saídas": st.column_config.TextColumn("Saídas", width=None),
                        "Reação": st.column_config.TextColumn("Reação", width=None),
                        "Contexto": st.column_config.TextColumn("Contexto", width=None)
                    })
                    
                    # Multivars de Saída
                    st.subheader("🔄 Multivars de Saída (Frases Completas)")
                    multivars_saida_data = [
                        {
                            "ID": b["bloco_id"],
                            "Multivars_Texto_Saida": "\n".join(b["saidas"][0].get("Multivars_Texto_Saida", [])) or "Nenhum"
                        } for b in blocos
                    ]
                    st.dataframe(multivars_saida_data, use_container_width=True, column_config={
                        "Multivars_Texto_Saida": st.column_config.TextColumn("Multivars_Texto_Saida", width=None)
                    })
                    
                    # Lista de vozes disponíveis
                    if TTS_AVAILABLE:
                        st.subheader("🎤 Vozes Disponíveis para TTS")
                        
                        st.write("**Vozes do Google Text-to-Speech (gTTS):**")
                        if GTTS_AVAILABLE:
                            gtts_voices = [
                                "gtts-pt-br (Português Brasil)", "gtts-pt-pt (Português Portugal)", 
                                "gtts-en (Inglês)", "gtts-en-us (Inglês EUA)", "gtts-en-gb (Inglês GB)",
                                "gtts-es (Espanhol)", "gtts-es-us (Espanhol EUA)", "gtts-fr (Francês)",
                                "gtts-de (Alemão)", "gtts-it (Italiano)", "gtts-ja (Japonês)",
                                "gtts-ko (Coreano)", "gtts-ru (Russo)", "gtts-ar (Árabe)", "gtts-hi (Hindi)"
                            ]
                            for voice in gtts_voices:
                                st.write(f"- {voice}")
                        else:
                            st.write("- gTTS não disponível")
                        
                        st.write("**Vozes do Edge TTS (Microsoft Edge - Premium):**")
                        if EDGE_TTS_AVAILABLE:
                            edge_voices = [
                                "edge-pt-br (Português Brasil - Francisca)", "edge-pt-br-male (Português Brasil - Antonio)",
                                "edge-en (Inglês EUA - Aria)", "edge-en-male (Inglês EUA - Andrew)",
                                "edge-es (Espanhol - Elvira)", "edge-fr (Francês - Denise)",
                                "edge-de (Alemão - Katja)", "edge-it (Italiano - Elsa)",
                                "edge-ja (Japonês - Nanami)", "edge-ko (Coreano - SunHi)",
                                "edge-ru (Russo - Svetlana)", "edge-ar (Árabe - Zariyah)",
                                "edge-hi (Hindi - Hemant)"
                            ]
                            for voice in edge_voices:
                                st.write(f"- {voice}")
                        else:
                            st.write("- Edge TTS não disponível")
                        
                        # Alterar voz do IM
                        st.subheader("🎤 Alterar Voz do IM")
                        voz_atual = memoria["IM"][im_id].get("voz", None)
                        
                        # Mapeamento de códigos para nomes descritivos
                        code_to_name = {
                            # Google TTS
                            "gtts-pt-br": "Google TTS - Português Brasil",
                            "gtts-pt-pt": "Google TTS - Português Portugal",
                            "gtts-en": "Google TTS - Inglês",
                            "gtts-en-us": "Google TTS - Inglês (EUA)",
                            "gtts-en-gb": "Google TTS - Inglês (GB)",
                            "gtts-es": "Google TTS - Espanhol",
                            "gtts-es-us": "Google TTS - Espanhol (EUA)",
                            "gtts-fr": "Google TTS - Francês",
                            "gtts-de": "Google TTS - Alemão",
                            "gtts-it": "Google TTS - Italiano",
                            "gtts-ja": "Google TTS - Japonês",
                            "gtts-ko": "Google TTS - Coreano",
                            "gtts-ru": "Google TTS - Russo",
                            "gtts-ar": "Google TTS - Árabe",
                            "gtts-hi": "Google TTS - Hindi",
                            # Edge TTS
                            "edge-pt-br": "Edge TTS - Português Brasil (Francisca - Feminina)",
                            "edge-pt-br-male": "Edge TTS - Português Brasil (Antônio - Masculino)",
                            "edge-en": "Edge TTS - Inglês (Jenny - Feminina)",
                            "edge-en-male": "Edge TTS - Inglês (Guy - Masculino)",
                            "edge-es": "Edge TTS - Espanhol (Helena - Feminina)",
                            "edge-fr": "Edge TTS - Francês (Denise - Feminina)",
                            "edge-de": "Edge TTS - Alemão (Katja - Feminina)",
                            "edge-it": "Edge TTS - Italiano (Elsa - Feminina)",
                            "edge-ja": "Edge TTS - Japonês (Nanami - Feminina)",
                            "edge-ko": "Edge TTS - Coreano (SunHi - Feminina)",
                            "edge-ru": "Edge TTS - Russo (Svetlana - Feminina)",
                            "edge-ar": "Edge TTS - Árabe (Hoda - Feminina)",
                            "edge-hi": "Edge TTS - Hindi (Hemant - Masculino)"
                        }
                        name_to_code = {v: k for k, v in code_to_name.items()}
                        
                        # Opções de voz: Automático e vozes com nomes descritivos
                        voz_options = ["Automático"]
                        
                        # Adicionar vozes do gTTS (Google Text-to-Speech)
                        if GTTS_AVAILABLE:
                            gtts_voices = [
                                "gtts-pt-br", "gtts-pt-pt", "gtts-en", "gtts-en-us", "gtts-en-gb", 
                                "gtts-es", "gtts-es-us", "gtts-fr", "gtts-de", "gtts-it", "gtts-ja", 
                                "gtts-ko", "gtts-ru", "gtts-ar", "gtts-hi"
                            ]
                            voz_options.extend([code_to_name[code] for code in gtts_voices if code in code_to_name])
                        
                        # Adicionar vozes do Edge TTS (Microsoft Edge)
                        if EDGE_TTS_AVAILABLE:
                            edge_voices = [
                                "edge-pt-br", "edge-pt-br-male", "edge-en", "edge-en-male", 
                                "edge-es", "edge-fr", "edge-de", "edge-it", "edge-ja", 
                                "edge-ko", "edge-ru", "edge-ar", "edge-hi"
                            ]
                            voz_options.extend([code_to_name[code] for code in edge_voices if code in code_to_name])
                        
                        default_index = 0
                        if voz_atual and voz_atual in code_to_name:
                            voz_nome_atual = code_to_name[voz_atual]
                            if voz_nome_atual in voz_options:
                                default_index = voz_options.index(voz_nome_atual)
                        voz_selecionada = st.selectbox("Selecione uma voz:", voz_options, index=default_index, key=f"voz_{im_id}")
                        if st.button("Salvar Voz", key=f"save_voz_{im_id}"):
                            if voz_selecionada == "Automático":
                                memoria["IM"][im_id].pop("voz", None)
                            else:
                                voz_code = name_to_code[voz_selecionada]
                                memoria["IM"][im_id]["voz"] = voz_code
                            salvar_json(ARQUIVO_MEMORIA, memoria)
                            st.success(f"✅ Voz do IM {im_id} atualizada para {voz_selecionada}!")
                            st.rerun()
                        
                        st.info("💡 **Sistema TTS Otimizado!** Edge TTS para vozes premium, gTTS para vozes leves e pyttsx3 como fallback. Sem Tortoise para melhor performance!")
                    
                    # Submenu para editar blocos
                    bloco_options = {f"ID {b['bloco_id']}: {b['entrada']['texto']}": b for b in blocos}
                    bloco_selecionado = st.selectbox("Selecione o bloco para editar:", list(bloco_options.keys()), key=f"edit_{im_id}")
                    bloco = bloco_options[bloco_selecionado]
                    with st.form(f"edit_bloco_{im_id}_{bloco['bloco_id']}"):
                        st.subheader("Editar Bloco")
                        entrada_texto = st.text_area("Entrada:", bloco["entrada"]["texto"], height=100)
                        multivars_entrada = st.text_area("Multivars Entrada (uma por linha):", "\n".join(bloco["entrada"].get("Multivars_Texto_Entrada", [])), height=100)
                        entrada_reacao = st.text_input("Reação (Entrada):", bloco["entrada"].get("reacao", ""))
                        entrada_contexto = st.text_area("Contexto (Entrada):", bloco["entrada"].get("contexto", ""), height=100)
                        entrada_pensamento = st.text_area("Pensamento Interno:", bloco["entrada"].get("pensamento_interno", ""), height=100)
                        saida_textos = st.text_area("Saída:", "\n".join(bloco["saidas"][0]["textos"]), height=150)
                        multivars_saida = st.text_area("Multivars Saída (uma por linha):", "\n".join(bloco["saidas"][0].get("Multivars_Texto_Saida", [])), height=100)
                        saida_reacao = st.text_input("Reação (Saída):", bloco["saidas"][0].get("reacao", ""))
                        saida_contexto = st.text_area("Contexto (Saída):", bloco["saidas"][0].get("contexto", ""), height=100)
                        if st.form_submit_button("Salvar Edições"):
                            bloco["entrada"] = {
                                "texto": entrada_texto,
                                "Multivars_Texto_Entrada": [m.strip() for m in multivars_entrada.split("\n") if m.strip()],
                                "reacao": entrada_reacao,
                                "contexto": entrada_contexto,
                                "pensamento_interno": entrada_pensamento,
                                "tokens": bloco["entrada"]["tokens"],
                                "fim": bloco["entrada"]["fim"],
                                "alnulu": bloco["entrada"]["alnulu"]
                            }
                            bloco["saidas"][0] = {
                                "textos": saida_textos.split("\n"),
                                "Multivars_Texto_Saida": [m.strip() for m in multivars_saida.split("\n") if m.strip()],
                                "reacao": saida_reacao,
                                "contexto": saida_contexto,
                                "tokens": bloco["saidas"][0]["tokens"],
                                "fim": bloco["saidas"][0]["fim"]
                            }
                            salvar_json(ARQUIVO_MEMORIA, memoria)
                            recalcular_marcadores_im(memoria, im_id)
                            st.success("Bloco editado com sucesso!")
                else:
                    st.write("Nenhum bloco.")
    elif sub_opc == "➕ Criar novo IM":
        create_new_im(memoria)
    elif sub_opc == "🔧 Gerar bloco a partir de template INSEPA":
        # Listar IMs existentes
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado. Crie um primeiro.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_gerar")
        
        # Mostrar template de exemplo
        with st.expander("📋 Ver template de exemplo"):
            template_example = create_bloco_template_example()
            st.code(template_example, language="text")
            st.info("💡 Copie o template acima e preencha com seus dados. Use [vars: ...] para variações de palavras e 'Multivars:' para frases completas.")
        
        st.write(f"Gerando blocos no IM {im_escolhido} (ou em outros se especificado no template)...")
        st.write("Cole seus blocos templates INSEPA separados por --- (ou copie o template do exemplo acima):")
        template_text = st.text_area("Templates:", key="template_text", height=300)
        if st.button("Gerar Blocos"):
            blocks = template_text.split("---")
            generated_count = 0
            for block in blocks:
                block = block.strip()
                if block:
                    if not block.startswith("Índice mãe:"):
                        block = f"Índice mãe: {im_escolhido}\n" + block
                    try:
                        generate_block_from_template(memoria, block)
                        generated_count += 1
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar bloco: {e}")
            if generated_count > 0:
                salvar_json(ARQUIVO_MEMORIA, memoria)
                st.success(f"✅ {generated_count} bloco(s) gerado(s) com sucesso!")
    elif sub_opc == "🗑️ Apagar bloco":
        # Apagar bloco
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_apagar_bloco")
        universo = memoria["IM"][im_escolhido]
        blocos = universo.get("blocos", [])
        if not blocos:
            st.error("❌ Nenhum bloco neste IM.")
            return
        st.write("Blocos no IM:")
        bloco_options = [f"ID {b['bloco_id']}: {b['entrada']['texto']}" for b in blocos]
        bloco_selecionado = st.selectbox("Selecione o bloco para apagar:", bloco_options, key="bloco_apagar")
        bid_apagar = bloco_selecionado.split(":")[0].split()[1]
        if st.button("Apagar Bloco"):
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
                    inconsciente = st.session_state.inconsciente
                    if im_escolhido in inconsciente.get("INCO", {}):
                        blocos_inco = inconsciente["INCO"][im_escolhido].get("Blocos", [])
                        inconsciente["INCO"][im_escolhido]["Blocos"] = [b for b in blocos_inco if b["Bloco_id"] != str(bid_int)]
                        # Recalcular Ultimo child se necessário
                        if blocos:
                            saida_vals = []
                            for b in blocos:
                                if im_escolhido in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][im_escolhido]:
                                    bloco_inco = next((bi for bi in inconsciente["INCO"][im_escolhido]["Blocos"] if bi["Bloco_id"] == str(b["bloco_id"])), None)
                                    if bloco_inco:
                                        saida_vals.extend(float(k) for k in bloco_inco.get("SAÍDA", {}).keys())
                            if saida_vals:
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = str(max(saida_vals))
                            else:
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                        else:
                            if im_escolhido in inconsciente.get("INCO", {}):
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                    st.success(f"✅ Bloco {bid_int} apagado. Blocos renumerados e ultimo_child ajustado.")
                else:
                    st.error("❌ Bloco não encontrado.")
            except ValueError:
                st.error("❌ ID inválido.")
    elif sub_opc == "🚮 Apagar IM":
        # Apagar IM
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_apagar = st.selectbox("Digite o ID do IM:", ims, key="im_apagar")
        confirm = st.checkbox("Tem certeza que quer apagar o IM e todos os seus blocos?")
        if confirm and st.button("Apagar IM"):
            # Coletar blocos para remover do inconsciente
            blocos_a_remover = [f"Bloco_{b['bloco_id']}" for b in memoria["IM"][im_apagar]["blocos"]]
            del memoria["IM"][im_apagar]
            salvar_json(ARQUIVO_MEMORIA, memoria)

            # Remover do inconsciente.json
            inconsciente = st.session_state.inconsciente
            if im_apagar in inconsciente.get("INCO", {}):
                del inconsciente["INCO"][im_apagar]
            salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

            st.success(f"✅ IM {im_apagar} apagado.")
    elif sub_opc == "⚙️ Alimentar vars dos tokens":
        # Alimentar vars dos tokens
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_vars")
        inconsciente = st.session_state.inconsciente
        st.write(f"DEBUG: Inconsciente carregado. Chaves INCO: {list(inconsciente.get('INCO', {}).keys())}")
        if im_escolhido not in inconsciente.get("INCO", {}):
            st.error("❌ Nenhum bloco no inconsciente para este IM.")
            return
        im_data = inconsciente["INCO"][im_escolhido]
        st.write(f"DEBUG: IM {im_escolhido} tem {len(im_data.get('Blocos', []))} blocos")
        blocos = im_data.get("Blocos", [])
        if not blocos:
            st.error("❌ Nenhum bloco no inconsciente para este IM.")
            return
        st.write(f"Blocos do IM {im_escolhido}:")
        for bloco in blocos:
            with st.expander(f"Bloco {bloco['Bloco_id']}"):
                st.subheader("Entrada")
                st.write(f"DEBUG: Bloco tem {len(bloco['Entrada'])} entradas")
                for marker, data in bloco["Entrada"].items():
                    vars_list = data.get('vars', [])
                    st.write(f"{marker}: {data['token']} | vars: {vars_list}")
                    if vars_list and any(v != "0.0" for v in vars_list):
                        st.write(f"  ✅ Vars não vazias encontradas: {vars_list}")
                st.subheader("SAÍDA")
                st.write(f"DEBUG: Bloco tem {len(bloco['SAÍDA'])} saídas")
                for marker, data in bloco["SAÍDA"].items():
                    vars_list = data.get('vars', [])
                    st.write(f"{marker}: {data['token']} | vars: {vars_list}")
                    if vars_list and any(v != "0.0" for v in vars_list):
                        st.write(f"  ✅ Vars não vazias encontradas: {vars_list}")
        # Editar vars
        bloco_ids = [b["Bloco_id"] for b in blocos]
        bloco_edit = st.selectbox("Escolha o bloco para editar:", bloco_ids, key="bloco_edit")
        bloco = next((b for b in blocos if b["Bloco_id"] == bloco_edit), None)
        if bloco:
            campo_opc = st.selectbox("Escolha o campo:", ["Entrada", "SAÍDA"], key="campo_edit")
            campo = bloco[campo_opc]
            markers = list(campo.keys())
            marker_edit = st.selectbox("Escolha o marcador:", markers, key="marker_edit")
            current_vars = campo[marker_edit]["vars"]
            st.write(f"Vars atuais: {current_vars}")
            new_vars_str = st.text_input("Digite os novos vars separados por vírgula (ex: 0.1,0.2):", key="new_vars_edit")
            if st.button("Atualizar Vars"):
                new_vars = [v.strip() for v in new_vars_str.split(",") if v.strip()]
                new_vars = sorted(list(set(new_vars)))  # Remover duplicatas e ordenar
                if not new_vars:
                    st.error("❌ Vars inválidos.")
                    return
                campo[marker_edit]["vars"] = new_vars
                salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                st.success(f"✅ Vars atualizados para {marker_edit}: {new_vars}")

            # Gerar vars automaticamente com dicionário de sinônimos
            token = campo[marker_edit]["token"]
            word_to_search = new_vars_str.strip().split(',')[0].strip() if new_vars_str.strip() else token
            if st.button("Gerar Vars com Dicionário", key="gerar_vars_dict"):
                try:
                    import re
                    import unidecode
                    st.write(f"Buscando sinônimos para a palavra: '{word_to_search}'")
                    clean_token = unidecode.unidecode(word_to_search.lower())
                    url = f"https://www.sinonimos.com.br/{clean_token}"
                    st.write(f"URL consultada: {url}")
                    content = ""
                    try:
                        # Tentar com requests primeiro
                        import requests
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                        response = requests.get(url, headers=headers)
                        st.write(f"Status da resposta (requests): {response.status_code}")
                        if response.status_code == 200:
                            content = response.text
                    except ImportError:
                        st.warning("Requests não disponível, tentando Selenium...")
                    
                    if not content:
                        # Fallback para Selenium
                        from selenium import webdriver
                        from selenium.webdriver.chrome.options import Options
                        options = Options()
                        options.add_argument("--headless")
                        options.add_argument("--no-sandbox")
                        options.add_argument("--disable-dev-shm-usage")
                        driver = webdriver.Chrome(options=options)
                        driver.get(url)
                        content = driver.page_source
                        driver.quit()
                        st.write("Conteúdo obtido via Selenium.")
                    
                    candidates = []
                    if content:
                        syn_links = _re.findall(r'<a href="https://www\.sinonimos\.com\.br/[^"]+">([^<]+)</a>', content)
                        candidates = [s for s in syn_links if s.lower() != word_to_search.lower() and len(s) > 1][:5]
                    
                    if not candidates:
                        # Tentar com Selenium se requests não encontrou
                        st.write("Tentando com Selenium...")
                        try:
                            from selenium import webdriver
                            from selenium.webdriver.chrome.options import Options
                            options = Options()
                            options.add_argument("--headless")
                            options.add_argument("--no-sandbox")
                            options.add_argument("--disable-dev-shm-usage")
                            driver = webdriver.Chrome(options=options)
                            driver.get(url)
                            content = driver.page_source
                            driver.quit()
                            st.write("Conteúdo obtido via Selenium.")
                            syn_links = _re.findall(r'<a href="https://www\.sinonimos\.com\.br/[^"]+">([^<]+)</a>', content)
                            candidates = [s for s in syn_links if s.lower() != word_to_search.lower() and len(s) > 1][:5]
                        except Exception as e:
                            st.error(f"❌ Erro com Selenium: {e}")
                    
                    if candidates:
                        st.write(f"Sugestões geradas: {candidates}")
                        selected = st.multiselect("Selecione as vars para adicionar:", candidates, key=f"select_{marker_edit}")
                        if st.button("Adicionar Selecionadas", key=f"add_{marker_edit}"):
                            current_vars = campo[marker_edit]["vars"]
                            new_vars = list(set(current_vars + selected))  # Evitar duplicatas
                            campo[marker_edit]["vars"] = new_vars
                            salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                            st.success(f"✅ Vars adicionadas: {selected}")
                    else:
                        st.warning("⚠️ Nenhuma variação válida encontrada.")
                except ImportError as e:
                    if 'unidecode' in str(e):
                        st.error("❌ Biblioteca 'unidecode' não instalada. Instale com: pip install unidecode")
                    elif 'selenium' in str(e):
                        st.error("❌ Biblioteca 'selenium' não instalada. Instale com: pip install selenium")
                    else:
                        st.error(f"❌ Erro de import: {e}")
                except Exception as e:
                    st.error(f"❌ Erro ao buscar: {e}")
    elif sub_opc == "✏️ Editar nomes de IMs":
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.info("Nenhum IM encontrado.")
            return
        st.write("Edite os nomes dos IMs:")
        for im_id in ims:
            current_name = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            new_name = st.text_input(f"Nome do IM {im_id}:", value=current_name, key=f"name_{im_id}")
            if st.button(f"Salvar nome para IM {im_id}", key=f"save_name_{im_id}"):
                memoria["IM"][im_id]["nome"] = new_name
                salvar_json(ARQUIVO_MEMORIA, memoria)
                st.success(f"Nome do IM {im_id} atualizado para '{new_name}'!")
                st.rerun()
            
    elif sub_opc == "💾 Backup JSON":
        submenu_backup(memoria, inconsciente)
    
    elif sub_opc == "⬅️ Voltar ao menu principal":
        st.session_state.menu = "principal"


def generate_block_from_template(memoria: dict, template: str) -> None:
    # Parsing mais robusto para textos grandes
    im_id = None
    entrada_texto = ""
    entrada_reacao = ""
    entrada_contexto = ""
    entrada_pensamento = ""
    entrada_multivars = []
    entrada_multivars_reacao = []
    saidas_textos = []
    saida_reacao = ""
    saida_contexto = ""
    saida_multivars = []
    saida_multivars_reacao = []
    
    # Encontrar seções
    entrada_start = template.find("Entrada:")
    saida_start = template.find("Saída:")
    
    if entrada_start == -1 or saida_start == -1:
        raise ValueError("Template inválido: seções 'Entrada:' e 'Saída:' são obrigatórias")
    
    # Se não encontrou IM ID, usar IM 1 como padrão
    if not im_id:
        im_id = "1"
    
    # Extrair entrada
    entrada_section = template[entrada_start:saida_start].strip()
    entrada_lines = entrada_section.split('\n')
    
    # Se a primeira linha começa com "Entrada:", capturar o texto
    if entrada_lines and entrada_lines[0].startswith("Entrada:"):
        entrada_texto = entrada_lines[0].split(":", 1)[1].strip()
        entrada_lines = entrada_lines[1:]
    
    current_field = None
    for line in entrada_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Reação:"):
            entrada_reacao = line.split(":", 1)[1].strip()
            current_field = "reacao"
        elif line.startswith("Contexto:"):
            entrada_contexto = line.split(":", 1)[1].strip()
            current_field = "contexto"
        elif line.startswith("Pensamento Interno:"):
            entrada_pensamento = line.split(":", 1)[1].strip()
            current_field = "pensamento"
        elif line.startswith("Multivars:"):
            if current_field == "reacao":
                entrada_multivars_reacao.append(line.split(":", 1)[1].strip())
            else:
                entrada_multivars.append(line.split(":", 1)[1].strip())
            current_field = "multivars"
        elif current_field == "reacao":
            entrada_reacao += " " + line
        elif current_field == "contexto":
            entrada_contexto += " " + line
        elif current_field == "pensamento":
            entrada_pensamento += " " + line
        elif current_field == "multivars":
            entrada_multivars.extend([v.strip() for v in line.split("|") if v.strip()])
        else:
            # Assume que é continuação do texto de entrada
            if entrada_texto:
                entrada_texto += " " + line
            else:
                entrada_texto = line
    
    # Extrair saída
    saida_section = template[saida_start:].strip()
    saida_lines = saida_section.split('\n')
    
    # Se a primeira linha começa com "Saída:", capturar o texto
    if saida_lines and saida_lines[0].startswith("Saída:"):
        saidas_textos.append(saida_lines[0].split(":", 1)[1].strip())
        saida_lines = saida_lines[1:]
    
    current_field = None
    for line in saida_lines:
        line = line.strip()
        if not line:
            continue
        # Ignorar linhas de separação visual
        if line == "═" * len(line) and len(line) > 5:
            continue
        if line.startswith("Reação:"):
            saida_reacao = line.split(":", 1)[1].strip()
            current_field = "reacao"
        elif line.startswith("Contexto:"):
            saida_contexto = line.split(":", 1)[1].strip()
            current_field = "contexto"
        elif line.startswith("Multivars:"):
            if current_field == "reacao":
                saida_multivars_reacao.append(line.split(":", 1)[1].strip())
            else:
                saida_multivars.append(line.split(":", 1)[1].strip())
            current_field = "multivars"
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            texto = line.split(".", 1)[1].strip()
            saidas_textos.append(texto)
            current_field = "texto"
        elif current_field == "reacao":
            saida_reacao += " " + line
        elif current_field == "contexto":
            saida_contexto += " " + line
        elif current_field == "texto":
            saidas_textos[-1] += " " + line
    
    if not entrada_texto or not saidas_textos:
        raise ValueError("Template inválido: texto de entrada e textos de saída são obrigatórios")
    
    # Resto do código permanece o mesmo
    if im_id not in memoria["IM"]:
        memoria["IM"][im_id] = {"nome": f"IM_{im_id}", "genero": "não binário", "ultimo_child": f"{im_id}.0", "blocos": []}
    universo = memoria["IM"][im_id]
    blocos = universo["blocos"]
    next_id = len(blocos) + 1
    # Limpar sintaxe de vars antes de salvar no memoria ← variações vão pro inconsciente
    entrada_texto_limpa = clean_vars_syntax(entrada_texto)
    entrada_reacao_limpa = clean_vars_syntax(entrada_reacao)
    saidas_textos_limpos = [clean_vars_syntax(t) for t in saidas_textos]
    saida_reacao_limpa = clean_vars_syntax(saida_reacao)
    
    bloco = {
        "bloco_id": next_id,
        "entrada": {
            "texto": entrada_texto_limpa,
            "Multivars_Texto_Entrada": entrada_multivars,
            "reacao": entrada_reacao_limpa,
            "Multivars_Reacao_Entrada": entrada_multivars_reacao,
            "contexto": entrada_contexto,
            "pensamento_interno": entrada_pensamento,
            "tokens": {},
            "fim": "",
            "alnulu": len(entrada_texto_limpa)
        },
        "saidas": [{
            "textos": saidas_textos_limpos,
            "Multivars_Texto_Saida": saida_multivars,
            "reacao": saida_reacao_limpa,
            "Multivars_Reacao_Saida": saida_multivars_reacao,
            "contexto": saida_contexto,
            "tokens": {},
            "fim": ""
        }],
        "open": True
    }
    blocos.append(bloco)
    current_last = universo["ultimo_child"]
    E = Token_with_vars(entrada_texto)
    # Combinar tokens consecutivos palavra + [vars: ...]
    combined_E = []
    i = 0
    while i < len(E):
        if i + 1 < len(E) and E[i+1].startswith('[vars:'):
            combined_E.append(E[i] + E[i+1])
            i += 2
        else:
            combined_E.append(E[i])
            i += 1
    E = combined_E
    
    RE = [entrada_reacao] if entrada_reacao else []
    combined_RE = []
    i = 0
    while i < len(RE):
        if i + 1 < len(RE) and RE[i+1].startswith('[vars:'):
            combined_RE.append(RE[i] + RE[i+1])
            i += 2
        else:
            combined_RE.append(RE[i])
            i += 1
    RE = combined_RE
    
    CE = Token(entrada_contexto)
    pensamento_limpo = entrada_pensamento.strip('"')
    partes = pensamento_limpo.split('.')[:3]
    PIDE_full = []
    for parte in partes:
        PIDE_full.extend(Token(parte.strip()))
    PIDE_limited = PIDE_full[:3]
    S = []
    for t in saidas_textos:
        tokens = Token_with_vars(t)
        combined_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i+1].startswith('[vars:'):
                combined_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                combined_tokens.append(tokens[i])
                i += 1
        S += combined_tokens
    RS = [saida_reacao] if saida_reacao else []
    combined_RS = []
    i = 0
    while i < len(RS):
        if i + 1 < len(RS) and RS[i+1].startswith('[vars:'):
            combined_RS.append(RS[i] + RS[i+1])
            i += 2
        else:
            combined_RS.append(RS[i])
            i += 1
    RS = combined_RS
    CS = Token(saida_contexto)
    entrada_tokens = E + RE + CE + PIDE_full
    saida_tokens = S + RS + CS
    ent_marks_inco = generate_markers(current_last, len(entrada_tokens))
    out_marks = generate_markers(ent_marks_inco[-1], len(saida_tokens))
    fim_ent = ent_marks_inco[-1]
    fim_out = out_marks[-1]
    ent_marks = ent_marks_inco[:len(E) + len(RE) + len(CE) + len(PIDE_limited)]
    idx = 0
    E_m = ent_marks[idx: idx + len(E)]; idx += len(E)
    RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
    CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
    PIDE_m = ent_marks[idx: idx + len(PIDE_limited)]
    jdx = 0
    S_m = out_marks[jdx: jdx + len(S)]; jdx += len(S)
    RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
    CS_m = out_marks[jdx: jdx + len(CS)]
    bloco["entrada"]["tokens"] = {
        "E": E_m,
        "RE": RE_m,
        "CE": CE_m,
        "PIDE": PIDE_m,
        "TOTAL": ent_marks_inco
    }
    bloco["entrada"]["fim"] = fim_ent
    bloco["saidas"][0]["tokens"] = {
        "S": S_m,
        "RS": RS_m,
        "CS": CS_m,
        "TOTAL": out_marks
    }
    bloco["saidas"][0]["fim"] = fim_out
    universo["ultimo_child"] = fim_out
    salvar_json(ARQUIVO_MEMORIA, memoria)
    inconsciente = st.session_state.inconsciente
    bloco_data = {
        "Bloco_id": str(next_id),
        "Entrada": {},
        "SAÍDA": {}
    }
    
    # Processar vars da entrada
    entrada_vars_dict = extract_vars_from_tokens(entrada_tokens)
    
    # Integrar alimentador de vars do inconsciente (usar último bloco como referência)
    if im_id in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][im_id].get("Blocos", [])
        if blocos_inco:
            ultimo_bloco = blocos_inco[-1]  # Usar último bloco como referência
            unconscious_vars = get_unconscious_vars_for_block({"bloco_id": ultimo_bloco["Bloco_id"]}, im_id)
            # Adicionar vars do inconsciente para tokens que não têm vars no template
            for token in entrada_tokens:
                if token not in entrada_vars_dict and token in unconscious_vars:
                    entrada_vars_dict[token] = unconscious_vars[token]
    
    for m, t in zip(ent_marks_inco, entrada_tokens):
        # Se o token tem vars, usar o token limpo como chave
        palavra, vars_list = parse_bloco_template_with_vars(t)
        if vars_list:
            bloco_data["Entrada"][m] = {"token": palavra, "vars": vars_list}
        else:
            bloco_data["Entrada"][m] = {"token": t, "vars": entrada_vars_dict.get(t, ["0.0"])}
    
    # Processar vars da saída
    saida_vars_dict = extract_vars_from_tokens(saida_tokens)
    
    # Integrar alimentador de vars do inconsciente para saída
    if im_id in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][im_id].get("Blocos", [])
        if blocos_inco:
            ultimo_bloco = blocos_inco[-1]
            unconscious_vars = get_unconscious_vars_for_block({"bloco_id": ultimo_bloco["Bloco_id"]}, im_id)
            for token in saida_tokens:
                if token not in saida_vars_dict and token in unconscious_vars:
                    saida_vars_dict[token] = unconscious_vars[token]
    
    for m, t in zip(out_marks, saida_tokens):
        palavra, vars_list = parse_bloco_template_with_vars(t)
        if vars_list:
            bloco_data["SAÍDA"][m] = {"token": palavra, "vars": vars_list}
        else:
            bloco_data["SAÍDA"][m] = {"token": t, "vars": saida_vars_dict.get(t, ["0.0"])}
    if im_id in inconsciente.get("INCO", {}):
        inconsciente["INCO"][im_id]["Blocos"].append(bloco_data)
        inconsciente["INCO"][im_id]["Ultimo child"] = fim_out
    else:
        inconsciente["INCO"][im_id] = {
            "NOME": universo["nome"],
            "Ultimo child": fim_out,
            "Blocos": [bloco_data]
        }
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def recalcular_marcadores_im(memoria: dict, im_id: str) -> None:
    """Recalcula marcadores e tokens para todos os blocos do IM após edição."""
    universo = memoria["IM"][im_id]
    blocos = universo["blocos"]
    if not blocos:
        universo["ultimo_child"] = f"{im_id}.0"
        salvar_json(ARQUIVO_MEMORIA, memoria)
        return

    # Ordenar blocos por id
    blocos.sort(key=lambda b: b["bloco_id"])
    current_last = f"{im_id}.0"

    for bloco in blocos:
        # Retokenizar entrada
        E = Token(bloco["entrada"]["texto"])
        RE = [bloco["entrada"]["reacao"]] if bloco["entrada"]["reacao"] else []
        CE = Token(bloco["entrada"]["contexto"])
        pensamento_limpo = bloco["entrada"]["pensamento_interno"].strip('"')
        partes = pensamento_limpo.split('.')[:3]
        PIDE_full = []
        for parte in partes:
            PIDE_full.extend(Token(parte.strip()))
        PIDE_limited = PIDE_full[:3]

        # Saída
        S = []
        for t in bloco["saidas"][0]["textos"]:
            S += Token(t)
        RS = [bloco["saidas"][0]["reacao"]] if bloco["saidas"][0]["reacao"] else []
        CS = Token(bloco["saidas"][0]["contexto"])

        # Calcular tokens completos
        entrada_tokens = E + RE + CE + PIDE_full
        saida_tokens = S + RS + CS

        # Gerar marcadores alinhados sem sobreposição
        ent_marks_inco = generate_markers(current_last, len(entrada_tokens))
        out_marks = generate_markers(ent_marks_inco[-1], len(saida_tokens))

        fim_ent = ent_marks_inco[-1]
        fim_out = out_marks[-1]

        # Para compatibilidade, ent_marks é o limitado
        ent_marks = ent_marks_inco[:len(E) + len(RE) + len(CE) + len(PIDE_limited)]

        # Subdivide
        idx = 0
        E_m = ent_marks[idx: idx + len(E)]; idx += len(E)
        RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
        CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
        PIDE_m = ent_marks[idx: idx + len(PIDE_limited)]

        jdx = 0
        S_m = out_marks[jdx: jdx + len(S)]; jdx += len(S)
        RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
        CS_m = out_marks[jdx: jdx + len(CS)]

        # Atualizar bloco existente
        bloco["entrada"]["tokens"] = {
            "E": E_m,
            "RE": RE_m,
            "CE": CE_m,
            "PIDE": PIDE_m,
            "TOTAL": ent_marks_inco
        }
        bloco["entrada"]["fim"] = fim_ent
        bloco["entrada"]["alnulu"] = len(bloco["entrada"]["texto"])

        bloco["saidas"][0]["tokens"] = {
            "S": S_m,
            "RS": RS_m,
            "CS": CS_m,
            "TOTAL": out_marks
        }
        bloco["saidas"][0]["fim"] = fim_out

        current_last = fim_out

    universo["ultimo_child"] = current_last
    salvar_json(ARQUIVO_MEMORIA, memoria)

    # Atualizar inconsciente - recriar baseado nos blocos, preservando vars existentes por token
    inconsciente = st.session_state.inconsciente
    # Carregar vars existentes por token
    existing_vars_by_token = {}
    if im_id in inconsciente.get("INCO", {}):
        for bloco_inco in inconsciente["INCO"][im_id].get("Blocos", []):
            bloco_id = bloco_inco["Bloco_id"]
            existing_vars_by_token[bloco_id] = {
                "Entrada": {data["token"]: data["vars"] for data in bloco_inco["Entrada"].values()},
                "SAÍDA": {data["token"]: data["vars"] for data in bloco_inco["SAÍDA"].values()}
            }
    
    if im_id not in inconsciente.get("INCO", {}):
        inconsciente.setdefault("INCO", {})[im_id] = {
            "NOME": universo["nome"],
            "Ultimo child": universo["ultimo_child"],
            "Blocos": []
        }
    im_data = inconsciente["INCO"][im_id]
    im_data["Ultimo child"] = universo["ultimo_child"]
    im_data["Blocos"] = []
    for bloco in blocos:
        # Retokenizar para obter tokens atuais
        E = Token(bloco["entrada"]["texto"])
        RE = [bloco["entrada"]["reacao"]] if bloco["entrada"]["reacao"] else []
        CE = Token(bloco["entrada"]["contexto"])
        pensamento_limpo = bloco["entrada"]["pensamento_interno"].strip('"')
        partes = pensamento_limpo.split('.')[:3]
        PIDE_full = []
        for parte in partes:
            PIDE_full.extend(Token(parte.strip()))
        
        S = []
        for t in bloco["saidas"][0]["textos"]:
            S += Token(t)
        RS = [bloco["saidas"][0]["reacao"]] if bloco["saidas"][0]["reacao"] else []
        CS = Token(bloco["saidas"][0]["contexto"])
        
        entrada_tokens_list = E + RE + CE + PIDE_full
        saida_tokens_list = S + RS + CS
        
        entrada_tokens = bloco["entrada"]["tokens"]["TOTAL"]  # marcadores
        saida_tokens = bloco["saidas"][0]["tokens"]["TOTAL"]  # marcadores
        
        # Usar vars existentes por token
        bloco_id = str(bloco["bloco_id"])
        entrada_vars_by_token = existing_vars_by_token.get(bloco_id, {}).get("Entrada", {})
        saida_vars_by_token = existing_vars_by_token.get(bloco_id, {}).get("SAÍDA", {})
        
        bloco_data = {
            "Bloco_id": bloco_id,
            "Entrada": {m: {"token": t, "vars": entrada_vars_by_token.get(t, ["0.0"])} for m, t in zip(entrada_tokens, entrada_tokens_list)},
            "SAÍDA": {m: {"token": t, "vars": saida_vars_by_token.get(t, ["0.0"])} for m, t in zip(saida_tokens, saida_tokens_list)}
        }
        im_data["Blocos"].append(bloco_data)
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def submenu_opinioes(memoria: dict) -> None:
    """Só a criadora chega aqui (a rota já exige admin). Opiniões (texto+emoção,
    registradas por qualquer usuário quando o Adam não reconhece a entrada) só viram
    dado concreto quando ela preenche contexto e pensamento_interno aqui."""
    st.subheader("💬 Opiniões pendentes")
    st.caption(
        "Texto + emoção que usuários trouxeram e o Adam não reconheceu. Só viram dado "
        "concreto quando você confirma o contexto e o pensamento por trás delas."
    )
    opinioes = carregar_opinioes()
    pendentes = [o for o in opinioes if o.get("status") == "pendente"]

    if not pendentes:
        st.write("Nenhuma opinião pendente no momento.")
        return

    for opiniao in pendentes:
        eh_semeadura = opiniao.get("origem") == "semeadura"
        titulo = f"{'🌱 ' if eh_semeadura else ''}\"{opiniao['texto']}\" {opiniao['emocao']} — IM {opiniao['dominio']}"
        with st.expander(titulo):
            candidato_id = opiniao.get("bloco_candidato_id")
            bloco_tentativa = None
            emprestado = opiniao.get("bloco_candidato_emprestado", False)
            if not candidato_id:
                st.info("🔍 Nenhum candidato -- nem registro exato, nem raiz emprestável, nem reflexão fraca.")
                st.caption(
                    "📖 Codex, Camada 5: sem token/var/multivar batendo, é simplesmente desconhecido -- "
                    "não existe aproximação por forma que valha como reconhecimento."
                )
                palpite_rn = opiniao.get("palpite_rede_neural")
                if palpite_rn:
                    st.warning(f"🧠 Palpite da rede neural (o sinal mais fraco de todos, geração livre): \"{palpite_rn}\"")
                    st.caption("Nunca foi mostrado como resposta pro usuário -- só um rascunho pra você avaliar.")
            if candidato_id:
                origem_im = opiniao.get("bloco_candidato_origem_im") or opiniao["dominio"]
                bloco_tentativa = next(
                    (b for b in memoria.get("IM", {}).get(origem_im, {}).get("blocos", [])
                     if str(b.get("bloco_id")) == str(candidato_id)),
                    None,
                )
                if bloco_tentativa:
                    if eh_semeadura:
                        st.warning(
                            f"🌱 O Adam propôs isso SOZINHO, fora de uma conversa real (semeadura automática) -- "
                            f"raiz do universo 0 (pensamento \"{bloco_tentativa['entrada'].get('pensamento_interno', '')}\") "
                            f"sem equivalente ainda no universo {opiniao['dominio']}. Texto, emoção e contexto vêm "
                            f"clonados do universo 0 só como ponto de partida -- edite a voz pro universo {opiniao['dominio']} "
                            f"antes de confirmar, ou descarte se não fizer sentido aqui."
                        )
                        st.caption(
                            "📖 Codex, Camada 1: só o universo 0 empresta pra derivados, nunca o contrário -- "
                            "e o vínculo fica registrado aqui nos metadados da opinião, não no marcador em si."
                        )
                    elif emprestado:
                        st.info(
                            f"🌱 Emprestado do universo 0 (score {opiniao.get('bloco_candidato_score', 0):.0%}): "
                            f"\"{bloco_tentativa['entrada']['texto']}\" — pensamento \"{bloco_tentativa['entrada'].get('pensamento_interno', '')}\". "
                            f"Contexto e pensamento já vêm preenchidos como ponto de partida -- edite ou confirme como preferir."
                        )
                        st.caption(
                            "📖 Codex, Camada 3: o pensamento é a raiz compartilhada -- bater na raiz é o que "
                            "justifica emprestar contexto/saída, mesmo com texto/reação diferentes entre universos."
                        )
                    else:
                        st.info(
                            f"🤔 O Adam achou parecido (score {opiniao.get('bloco_candidato_score', 0):.0%}) com "
                            f"\"{bloco_tentativa['entrada']['texto']}\" — contexto \"{bloco_tentativa['entrada'].get('contexto', '')}\". "
                            f"Se você reforçar, considere cadastrar \"{opiniao['texto']}\" como var/multivar desse bloco "
                            f"em vez de criar um bloco novo do zero."
                        )
                        st.caption(
                            "📖 Codex, Camada 5: score baixo não é reconhecimento por registro exato -- é só um "
                            "candidato fraco, reflexão antes de agir, nunca aproximação assumida como fato."
                        )
            contexto_padrao = bloco_tentativa["entrada"].get("contexto", "") if (bloco_tentativa and emprestado) else ""
            pensamento_padrao = bloco_tentativa["entrada"].get("pensamento_interno", "") if (bloco_tentativa and emprestado) else ""
            # Empréstimo empresta o bloco inteiro, não só contexto/pensamento -- a saída
            # também vem junto como sugestão pra confirmar, nunca precisa ser digitada do zero.
            saida_padrao = bloco_tentativa["saidas"][0] if (bloco_tentativa and emprestado and bloco_tentativa.get("saidas")) else {}

            if eh_semeadura:
                st.caption("Clonado do universo 0 -- ajuste a voz pro universo derivado antes de confirmar:")
                texto_novo_input = st.text_input("Texto (entrada)", value=opiniao["texto"], key=f"op_texto_{opiniao['id']}")
                emocao_novo_input = st.text_input("Reação (entrada)", value=opiniao["emocao"], key=f"op_emocao_{opiniao['id']}")
            else:
                texto_novo_input, emocao_novo_input = opiniao["texto"], opiniao["emocao"]

            contexto_novo = st.text_input("Contexto", value=contexto_padrao, key=f"op_ctx_{opiniao['id']}")
            pensamento_novo = st.text_input("Pensamento interno", value=pensamento_padrao, key=f"op_pide_{opiniao['id']}")
            saida_texto_novo = st.text_input("Texto de saída (a resposta do Adam)", value=(saida_padrao.get("textos", [""])[0] if saida_padrao else ""), key=f"op_saida_{opiniao['id']}")
            saida_reacao_novo = st.text_input("Reação de saída", value=saida_padrao.get("reacao", ""), key=f"op_saida_reac_{opiniao['id']}")

            st.caption("A reação também tem vars (uma palavra/expressão por linha) e multivars (uma frase por linha) -- pra ser reconhecida em outras formas, igual o texto.")
            col_re1, col_re2 = st.columns(2)
            entrada_re_vars_txt = col_re1.text_area("Vars da reação (entrada)", value="", placeholder="estrela no quadrado\ndespedida", key=f"op_re_vars_{opiniao['id']}")
            entrada_re_multivars_txt = col_re2.text_area("Multivars da reação (entrada)", value="", placeholder="Se despedindo", key=f"op_re_multivars_{opiniao['id']}")

            st.caption("Pode deixar em branco o que ainda não souber -- vira o ponto neutro do universo, sem travar a confirmação.")
            col_confirmar, col_descartar = st.columns(2)
            if col_confirmar.button("✅ Confirmar como dado concreto", key=f"op_confirmar_{opiniao['id']}"):
                if True:  # nada aqui é exigência -- o que faltar vira o ponto neutro, nunca bloqueia
                    dominio = opiniao["dominio"]
                    placeholder = f"{dominio}.0"
                    contexto_novo = contexto_novo.strip() or placeholder
                    pensamento_novo = pensamento_novo.strip() or placeholder
                    saida_texto_novo = saida_texto_novo.strip() or placeholder

                    memoria.setdefault("IM", {}).setdefault(dominio, {"nome": f"IM_{dominio}", "ultimo_child": f"{dominio}.0", "blocos": []})
                    universo = memoria["IM"][dominio]
                    saida_contexto_texto = "Confirmado pela criadora a partir de uma opinião"
                    next_id = len(universo["blocos"]) + 1

                    entrada_re_vars = [v.strip() for v in entrada_re_vars_txt.split("\n") if v.strip()]
                    entrada_re_multivars = [v.strip() for v in entrada_re_multivars_txt.split("\n") if v.strip()]

                    novo_bloco = {
                        "bloco_id": next_id,
                        "entrada": {
                            "texto": texto_novo_input,
                            "Multivars_Texto_Entrada": [],
                            "reacao": emocao_novo_input,
                            "Multivars_Reacao_Entrada": entrada_re_multivars,
                            "contexto": contexto_novo,
                            "pensamento_interno": pensamento_novo,
                            "tokens": {},
                            "fim": "",
                        },
                        "saidas": [{
                            "textos": [saida_texto_novo],
                            "Multivars_Texto_Saida": [],
                            "reacao": saida_reacao_novo,
                            "Multivars_Reacao_Saida": [],
                            "contexto": saida_contexto_texto,
                            "tokens": {},
                            "fim": "",
                        }],
                        "meta": {"origem": "opiniao_confirmada", "opiniao_id": opiniao["id"]},
                    }

                    # Camada 1: marcador é local do universo (dominio.X) -- a ligação com
                    # o universo 0, quando o dado foi emprestado, fica registrada no meta/
                    # bloco_candidato_origem_im do bloco, não em compartilhar o mesmo prefixo.
                    E = Token(texto_novo_input)
                    RE = Token(emocao_novo_input) if emocao_novo_input else []
                    CE = Token(contexto_novo)
                    PIDE = Token(pensamento_novo)
                    S = Token(saida_texto_novo)
                    RS = Token(saida_reacao_novo) if saida_reacao_novo else []
                    CS = Token(saida_contexto_texto)

                    current_last = universo["ultimo_child"]
                    ent_marks = generate_markers(current_last, len(E) + len(RE) + len(CE) + len(PIDE))
                    out_marks = generate_markers(ent_marks[-1], len(S) + len(RS) + len(CS))

                    idx = 0
                    E_m = ent_marks[idx: idx + len(E)]; idx += len(E)
                    RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
                    CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
                    PIDE_m = ent_marks[idx: idx + len(PIDE)]
                    jdx = 0
                    S_m = out_marks[jdx: jdx + len(S)]; jdx += len(S)
                    RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
                    CS_m = out_marks[jdx: jdx + len(CS)]

                    # PIDE na memória só guarda 3 marcadores CHAVE (ponteiro pra onde a
                    # informação completa vive) -- o detalhe token a token completo do
                    # pensamento fica no inconsciente (ver bloco_inco["Entrada"] abaixo),
                    # nunca duplicado aqui. fim/TOTAL continuam reservando o range inteiro.
                    novo_bloco["entrada"]["tokens"] = {"E": E_m, "RE": RE_m, "CE": CE_m, "PIDE": PIDE_m[:3], "TOTAL": ent_marks}
                    novo_bloco["entrada"]["fim"] = ent_marks[-1]
                    novo_bloco["saidas"][0]["tokens"] = {"S": S_m, "RS": RS_m, "CS": CS_m, "TOTAL": out_marks}
                    novo_bloco["saidas"][0]["fim"] = out_marks[-1]

                    universo["blocos"].append(novo_bloco)
                    universo["ultimo_child"] = out_marks[-1]

                    # Dado concreto confirmado pela criadora sempre persiste na hora --
                    # nunca deve depender do checkbox de auto-save (que é opcional/
                    # desligável). Grava no Mongo + arquivo local direto, sem precisar
                    # de terminal nem de mexer no Mongo manualmente depois.
                    st.session_state.memoria = memoria
                    salvar_json(ARQUIVO_MEMORIA, memoria)
                    st.session_state.alnulu_cache = build_alnulu_cache(memoria)

                    # Espelha no inconsciente pra manter Bloco_id/marcadores consistentes entre os dois arquivos
                    inconsciente = st.session_state.inconsciente
                    inconsciente.setdefault("INCO", {}).setdefault(dominio, {"NOME": universo.get("nome", f"IM_{dominio}"), "Ultimo child": current_last, "Blocos": []})
                    # A reação tem vars igual o texto -- as que a criadora cadastrou acima
                    # ficam nos marcadores RE em vez do placeholder "0.0" genérico.
                    vars_re_registradas = entrada_re_vars if entrada_re_vars else ["0.0"]
                    entrada_inco = {m: {"token": t, "vars": ["0.0"]} for m, t in zip(E_m + CE_m + PIDE_m, E + CE + PIDE)}
                    for m, t in zip(RE_m, RE):
                        entrada_inco[m] = {"token": t, "vars": vars_re_registradas}
                    bloco_inco = {
                        "Bloco_id": str(next_id),
                        "Entrada": entrada_inco,
                        "SAÍDA": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(S_m + RS_m + CS_m, S + RS + CS)},
                    }
                    inconsciente["INCO"][dominio]["Blocos"].append(bloco_inco)
                    inconsciente["INCO"][dominio]["Ultimo child"] = out_marks[-1]
                    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

                    opiniao["status"] = "confirmada"
                    opiniao["contexto"] = contexto_novo
                    opiniao["pensamento_interno"] = pensamento_novo
                    salvar_opinioes(opinioes)
                    st.success(f"✅ Virou o bloco #{novo_bloco['bloco_id']} (marcadores {ent_marks[0]}…{out_marks[-1]}).")
                    st.rerun()
            if col_descartar.button("🗑️ Descartar", key=f"op_descartar_{opiniao['id']}"):
                opiniao["status"] = "descartada"
                salvar_opinioes(opinioes)
                st.rerun()


def harness_find_exact_match(dominio: str, txt: str, reac: str) -> Optional[dict]:
    """Mesmo registro exato do adam_harness.py, só que direto contra o estado
    já carregado do app -- sem precisar reimportar o módulo."""
    memoria = st.session_state.get("memoria", {})
    blocos = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
    for b in blocos:
        textos_possiveis = (
            [b["entrada"]["texto"]]
            + b["entrada"].get("Multivars_Texto_Entrada", [])
            + [variar_texto(b["entrada"]["texto"], b, dominio, "entrada")]
        )
        if any(normalize(t) == normalize(txt) for t in textos_possiveis) and reac == b["entrada"].get("reacao", ""):
            return b
    return None


def harness_run_case(dominio: str, raw_input: str) -> Dict[str, Any]:
    """A mesma lógica do adam_harness.py (match exato -> reflexão -> opinião),
    rodando dentro do próprio app, pra criadora testar direto do navegador."""
    txt, reac = parse_text_reaction(raw_input)
    resultado: Dict[str, Any] = {"dominio": dominio, "input": raw_input, "tokens": Token(raw_input)}

    bloco_exato = harness_find_exact_match(dominio, txt, reac)
    if bloco_exato:
        resultado.update({"status": "match_exato", "bloco_id": bloco_exato["bloco_id"], "resposta": bloco_exato["saidas"][0]["textos"][0]})
        return resultado

    candidato = melhor_candidato_fraco(txt, dominio)
    if candidato:
        b = candidato["bloco"]
        resultado.update({
            "status": "reflexao",
            "candidato_bloco_id": b["bloco_id"],
            "candidato_score": candidato["score"],
            "resposta_tentativa": b["saidas"][0]["textos"][0],
        })
    else:
        resultado["status"] = "desconhecido"

    opiniao = registrar_opiniao_pendente(txt, reac, dominio, candidato)
    resultado["opiniao_id"] = opiniao["id"]
    return resultado


def submenu_harness(memoria: dict) -> None:
    """Versão ao vivo do adam_harness.py, direto no app -- pra criadora testar
    sem precisar do terminal. Mesma regra: registro exato ou reflexão, nunca
    aproximação, e nada vira dado concreto sozinho."""
    st.subheader("🧪 Harness ao vivo")
    st.caption(
        "Roda os mesmos casos de teste do adam_harness.py (mais qualquer frase que "
        "você quiser testar), dentro do próprio app. Nada é confirmado sozinho -- "
        "tudo que não bate exato vira opinião pendente, igual sempre."
    )

    casos_padrao = [
        ("0", "Olá 😊"),
        ("0", "Como você aprende coisas novas? 🤔"),
        ("0", "Boa noite 🌃"),
        ("0", "Qual é a sua cor favorita? 🎨"),
    ]

    st.markdown("**Testar uma frase sua:**")
    col_dom, col_txt = st.columns([1, 3])
    dominio_teste = col_dom.text_input("IM", value="0", key="harness_dominio")
    frase_teste = col_txt.text_input("Frase (texto + emoji de reação)", placeholder="Boa noite 🌃", key="harness_frase")
    if st.button("▶️ Testar essa frase"):
        if frase_teste.strip():
            r = harness_run_case(dominio_teste.strip() or "0", frase_teste)
            st.session_state.setdefault("harness_resultados_avulsos", []).insert(0, r)

    if st.button("▶️ Rodar todos os casos padrão"):
        st.session_state["harness_resultados_padrao"] = [harness_run_case(dominio, raw) for dominio, raw in casos_padrao]
        universos_derivados = [im for im in memoria.get("IM", {}).keys() if im != "0"]
        st.session_state["harness_semeadura"] = {
            dominio: propor_semeadura_universo(dominio) for dominio in universos_derivados
        }
        st.session_state["harness_semeadura"] = {k: v for k, v in st.session_state["harness_semeadura"].items() if v}

    def mostrar_resultado(r: dict) -> None:
        with st.container(border=True):
            st.write(f"**{r['input']}** — IM {r['dominio']}")
            st.caption(f"tokens INSEPA: {r['tokens']}")
            if r["status"] == "match_exato":
                st.success(f"✅ Match exato -- bloco #{r['bloco_id']}: \"{r['resposta']}\"")
            elif r["status"] == "reflexao":
                st.warning(
                    f"🤔 Candidato fraco (score {r['candidato_score']:.0%}) -- bloco #{r['candidato_bloco_id']}: "
                    f"\"{r['resposta_tentativa']}\". Registrado como opinião #{r['opiniao_id']}, nada confirmado sozinho."
                )
            else:
                st.info(f"🔍 Desconhecido. Registrado como opinião #{r['opiniao_id']}, aguardando você.")

    if st.session_state.get("harness_resultados_padrao"):
        st.markdown("**Casos padrão:**")
        for r in st.session_state["harness_resultados_padrao"]:
            mostrar_resultado(r)

    if st.session_state.get("harness_semeadura"):
        st.markdown("**Semeadura noturna (universo 0 → derivados):**")
        for dominio, propostas in st.session_state["harness_semeadura"].items():
            st.info(f"Universo {dominio}: {len(propostas)} raiz(es) proposta(s) -- aguardando revisão.")

    if st.session_state.get("harness_resultados_avulsos"):
        st.markdown("**Seus testes avulsos:**")
        for r in st.session_state["harness_resultados_avulsos"]:
            mostrar_resultado(r)

    st.caption("Reveja tudo que virou opinião em '💬 Opiniões pendentes' (admin).")


def submenu_codex() -> None:
    """O Codex ADAM INSEPA, direto dentro do lovely.py -- pra que o próprio Adam
    (e a criadora) tenham, junto do resto do painel admin, a referência de como
    ele mesmo funciona: as camadas do INSEPA, a terminologia, e como tudo se
    conecta ao código real. Espelha o Codex do CDF.py (a fonte original, onde
    a referência continua evoluindo camada por camada)."""
    st.header("📖 Codex ADAM INSEPA")
    st.markdown(
        "Referência viva de como o próprio Adam funciona -- as camadas do INSEPA "
        "explicadas, pra ele (e pra você) refletir sobre a própria existência, não só "
        "pra recriar blocos."
    )

    st.subheader("Terminologia")
    st.markdown(
        "- **IM (Índice Mãe)**: o número antes do ponto num marcador (ex.: o `0` de `0.25`). "
        "É o mesmo para todos os blocos de um universo/domínio — todo IM é 'mãe' dos seus IFs.\n"
        "- **IF (Índice Filho)**: o número depois do ponto (`0.1`, `0.25`, `0.26`...). "
        "Cada IF é único dentro daquele IM e nunca se repete, mesmo entre blocos diferentes.\n"
        "- **Vars**: variações de PALAVRA (texto) ou de emoji (reação) que significam a mesma coisa.\n"
        "- **Multivars**: variações de FRASE inteira (texto ou reação) que significam a mesma coisa.\n"
        "- **Pensamento interno**: a raiz/tronco de um bloco — de onde nascem o ramo entrada "
        "e o ramo saída (Camada 3).\n"
        "- **Opinião**: um dado com só texto + reação — o que qualquer usuário pode "
        "produzir. Ainda não confirmado.\n"
        "- **Dado concreto**: um dado com texto + reação + contexto + pensamento interno, "
        "todos presentes — e contexto/pensamento só podem ser definidos pela **criadora**, "
        "nunca por qualquer usuário (mesma senha de admin/criação de blocos deste painel).\n"
        "- **Placeholder `{IM}.0`**: marcador reservado (nunca usado pela sequência normal, "
        "que começa em `.1`) que sinaliza \"contexto/pensamento ainda não confirmados\" "
        "direto na estrutura do dado, sem precisar de uma flag separada.\n"
        "- **`Adam_Lovely_unconscious.json` (o inconsciente)**: é a Camada 1 crua e só ela — "
        "puro marcador → token + vars, sem nenhuma classificação (sem reação/contexto/"
        "pensamento interpretados). `Adam_Lovely_memory.json` (o consciente) é o pacote "
        "completo: os mesmos marcadores, mas já com Camada 2 e 3 aplicadas (reação, "
        "contexto e pensamento_interno presentes). O inconsciente é o \"dado bruto\"; o "
        "consciente é o \"dado com sentido\"."
    )

    st.subheader("As camadas")
    st.markdown(
        "1. **Marcadores Únicos** — cada token ganha um marcador que nunca se repete fora "
        "daquela entrada. A mesma palavra em frases diferentes é um símbolo diferente — "
        "isso impede o sistema de confundir significados. Cada universo tem sua própria "
        "sequência local (0.X, 1.X, 2.X...) -- só o universo 0 pode emprestar pensamento "
        "pros derivados, nunca o contrário, e a ligação fica registrada nos metadados do "
        "bloco, não nos dígitos do marcador.\n"
        "2. **Classificação** — não inventa nada, só classifica o que o texto já carrega: "
        "toda entrada tem reação (mesmo que seja frieza) e contexto, explícitos ou ocultos "
        "(via var).\n"
        "3. **Integração** — o pensamento interno é a raiz; dele nascem o ramo entrada e o "
        "ramo saída, que ressoam entre si por compartilharem essa raiz, não por seus "
        "contextos se parecerem.\n"
        "4. **Hashrização** — a chave é a junção dos marcadores de TEXTO + REAÇÃO (explícita "
        "ou oculta por var — é aí que entra a similaridade da Camada 2). Contexto fica de "
        "fora da chave, ele é inferido depois por ressonância (Camada 3), não por hash. "
        "Só a chave exata (texto+reação) destranca a porta pra saída correspondente. A "
        "sequência de marcadores nunca reinicia nem reusa posição dentro de um IM — "
        "só cresce, o que garante que a chave seja sempre única no histórico real.\n"
        "5. **Aprendizado Seguro** — o ALNULU só deixa o Adam VER a forma de uma palavra; "
        "reconhecer é o INSEPA que decide, por **registro exato** (token conhecido, ou "
        "var/multivar já cadastrada num bloco) — nunca por aproximação/distância. IA comum "
        "usa embeddings e similaridade de vetor; o Adam usa algarismos sequenciais e posição "
        "exata. Sem token/var/multivar batendo, é simplesmente desconhecido. Qualquer usuário "
        "pode produzir uma opinião (só texto+reação). **Só a criadora pode definir contexto e "
        "pensamento** — nunca qualquer usuário — convertendo a opinião em dado concreto. Ao "
        "salvar uma opinião, texto (E) e reação (RE) ganham marcadores reais; contexto (CE) e "
        "pensamento (PIDE) ficam travados no placeholder `{IM}.0` até a criadora confirmar.\n"
        "6. **Espelhamento e Análise** — em vez de um score único, compara a entrada nova "
        "contra um bloco de referência eixo por eixo (reação, texto, contexto, pensamento), "
        "sempre por igualdade EXATA. Texto batendo com reação divergindo é sinal de tom/"
        "sarcasmo (pela própria Camada 4 já não é a mesma chave). Contexto divergindo é "
        "assunto genuinamente diferente. Pensamento batendo é necessário mas nunca "
        "suficiente — a criadora sempre revisa antes de virar dado concreto. É assim que o "
        "Adam \"aprende com a interação\" sem nunca aproximar: cada diagnóstico que ela "
        "resolve vira registro exato novo."
    )

    st.subheader("Como isso se conecta ao lovely.py")
    st.markdown(
        "- **ALNULU** foi feito pra alimentar o treino do modelo PyTorch e pra comparar "
        "FORMA/ortografia (Camada 5) — usá-lo como medida de significado (cosseno sobre "
        "o texto inteiro) foi um erro corrigido: gerava falso-positivo grosseiro "
        "(\"feliz\"/\"triste\" batendo 89%).\n"
        "- **Vars/Multivars** existem pra texto E reação, entrada e saída. Contexto e "
        "pensamento nunca têm var — por isso são o fingerprint confiável, e por isso "
        "assuntos divergentes (\"lua\"/\"noite\") nunca devem ser aproximados.\n"
        "- O **bloco** em `Adam_Lovely_memory.json` é literalmente a árvore da Camada 3: "
        "um pensamento, um ramo de entrada, um ramo de saída.\n"
        "- **Empréstimo/semeadura**: o universo 0 pode propor -- nunca confirmar sozinho -- "
        "que um universo derivado herde uma raiz sua (texto+reação+contexto), quando o "
        "pensamento coincide. Vira opinião pendente, sempre esperando a criadora."
    )

    st.caption(
        "Princípio geral: o processo do INSEPA deve ser respeitado camada por camada, "
        "não substituído por atalhos de similaridade genérica."
    )


def submenu_estatisticas(memoria: dict) -> None:
    st.subheader("📊 Estatísticas do Sistema INSEPA")
    
    # Número de IMs
    num_ims = len(memoria.get("IM", {}))
    st.metric("Número de IMs", num_ims)
    
    if num_ims > 0:
        # Dados para gráficos
        im_names = []
        num_blocos = []
        total_blocos = 0
        for im_id, im_data in memoria["IM"].items():
            nome = im_data.get("nome", f"IM_{im_id}")
            blocos = len(im_data.get("blocos", []))
            im_names.append(nome)
            num_blocos.append(blocos)
            total_blocos += blocos
        
        st.metric("Total de Blocos", total_blocos)
        
        # Gráfico de barras: Blocos por IM
        import pandas as pd
        df_blocos = pd.DataFrame({"IM": im_names, "Blocos": num_blocos})
        st.bar_chart(df_blocos.set_index("IM"))
        
        # Estatísticas adicionais
        if total_blocos > 0:
            avg_blocos = total_blocos / num_ims
            st.metric("Média de Blocos por IM", f"{avg_blocos:.1f}")
            
            # Distribuição de reações
            reacoes = {}
            for im_data in memoria["IM"].values():
                for bloco in im_data.get("blocos", []):
                    reac = bloco["entrada"].get("reacao", "")
                    if reac:
                        reacoes[reac] = reacoes.get(reac, 0) + 1
            
            if reacoes:
                df_reacoes = pd.DataFrame(list(reacoes.items()), columns=["Reação", "Contagem"])
                st.subheader("Distribuição de Reações de Entrada")
                st.bar_chart(df_reacoes.set_index("Reação"))
        
        # Verificar se há checkpoints treinados
        import os
        ckpts = [f for f in os.listdir(".") if f.startswith("insepa_") and f.endswith(".pt")]
        st.metric("Modelos Treinados", len(ckpts))
        if ckpts:
            st.write("Modelos disponíveis:")
            for ckpt in ckpts:
                dom = ckpt.replace("insepa_", "").replace(".pt", "")
                st.write(f"- {dom}")
    else:
        st.info("Nenhum IM criado ainda.")


def submenu_backup(memoria: dict, inconsciente: dict) -> None:
    st.subheader("💾 Backup dos JSONs")
    st.write("Aqui você pode visualizar e baixar cópias dos JSONs de memória e inconsciente.")
    
    # Backup da Memória
    st.subheader("📄 JSON de Memória (Adam_Lovely_memory.json)")
    memoria_json = json.dumps(memoria, ensure_ascii=False, indent=2)
    st.code(memoria_json, language="json")
    st.download_button(
        label="📥 Baixar JSON de Memória",
        data=memoria_json,
        file_name="Adam_Lovely_memory.json",
        mime="application/json",
        key="download_memoria"
    )
    
    # Backup do Inconsciente
    st.subheader("🧠 JSON do Inconsciente (Adam_Lovely_unconscious.json)")
    inconsciente_json = json.dumps(inconsciente, ensure_ascii=False, indent=2)
    st.code(inconsciente_json, language="json")
    st.download_button(
        label="📥 Baixar JSON do Inconsciente",
        data=inconsciente_json,
        file_name="Adam_Lovely_unconscious.json",
        mime="application/json",
        key="download_inconsciente"
    )
    
    st.info("💡 Use esses backups para restaurar dados ou para deploy. Os arquivos são salvos com timestamp após treinamentos automáticos.")
    
    # Opções de Restauração e Manutenção
    st.subheader("🔧 Restauração e Manutenção")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Limpar Cache Streamlit"):
            # Limpar caches do Streamlit
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache do Streamlit limpo!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reiniciar Sessão"):
            # Limpar session_state e recarregar
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Sessão reiniciada! Recarregando...")
            st.rerun()
    
    with col3:
        if st.button("💾 Fazer Backup Manual"):
            # Salvar backups manuais com timestamp na pasta backup
            import time
            timestamp = int(time.time())
            backup_memoria_file = f"backup/Adam_Lovely_memory_manual_backup_{timestamp}.json"
            backup_inconsciente_file = f"backup/Adam_Lovely_unconscious_manual_backup_{timestamp}.json"
            salvar_json(backup_memoria_file, memoria)
            salvar_json(backup_inconsciente_file, inconsciente)
            st.success(f"✅ Backups manuais salvos na pasta backup: {backup_memoria_file} e {backup_inconsciente_file}")
    
    st.warning("⚠️ **Atenção:** 'Reiniciar Sessão' limpa todos os dados não salvos. Faça backup antes!")


def submenu_testar_adam(memoria: dict, inconsciente: dict) -> None:
    st.subheader("🧪 Testar Adam Afiado com ALNULU")
    st.write("Teste prático do sistema refinado: match exato, similaridade ALNULU, reflexão e geração autônoma.")
    
    dominio = prompt_dominio("testar", memoria)
    if not dominio:
        return
    
    st.write(f"Testando no domínio: {dominio}")
    
    # Teste 1: Match Exato
    st.subheader("1. Match Exato")
    txt_exato = st.text_input("Digite input exato (ex: Olá Sr. Vampiro ^^):", key="txt_exato")
    if txt_exato:
        txt_exato_norm = normalize(txt_exato)
        blocos = memoria["IM"][dominio]["blocos"]
        bloco_exato = None
        for b in blocos:
            bloco_texto = b["entrada"]["texto"]
            bloco_reac = b["entrada"].get("reacao", "") or ""
            if normalize(f"{bloco_texto} {bloco_reac}".strip()) == txt_exato_norm:
                bloco_exato = b
                break
            if normalize(bloco_texto) == txt_exato_norm and not bloco_reac:
                bloco_exato = b
                break
        if bloco_exato:
            resposta = bloco_exato['saidas'][0]['textos'][0] if bloco_exato['saidas'] else "(sem saída definida)"
            st.success(f"✅ Match exato: '{txt_exato}' → '{resposta}'")
        else:
            st.warning(f"❌ Nenhum match exato para '{txt_exato}'")
    
    # Teste 2: Similaridade ALNULU
    st.subheader("2. Similaridade ALNULU")
    txt_sim = st.text_input("Digite input para similaridade (ex: Oiee ^^):", key="txt_sim")
    if txt_sim:
        # Parsing inicial para reação global se aplicável
        full_input = txt_sim
        reac_sim = ""
        blocos = memoria["IM"][dominio]["blocos"]
        for b in blocos:
            bloco_reac = b["entrada"].get("reacao", "").lower().strip()
            if bloco_reac and full_input.lower().strip().endswith(bloco_reac):
                reac_sim = bloco_reac
                txt_sim = full_input[:-len(bloco_reac)].rstrip()
                break
        ctx_sim = ""  # Ainda opcional
        
        # Dividir input em partes baseadas em reações encontradas
        partes = []
        remaining = txt_sim
        while remaining:
            found = False
            for b in blocos:
                bloco_reac = b["entrada"].get("reacao", "").strip()
                if bloco_reac and len(bloco_reac) > 1 and bloco_reac in remaining:  # Só dividir em reações com mais de 1 char
                    idx = remaining.find(bloco_reac)
                    if idx > 0:
                        parte = remaining[:idx + len(bloco_reac)].strip()
                        partes.append(parte)
                        remaining = remaining[idx + len(bloco_reac):].strip().lstrip(".,!? ")
                        found = True
                        break
            if not found:
                if remaining.strip():
                    partes.append(remaining.strip())
                break
        if not partes:
            partes = [txt_sim]
        # Se há reac_sim, adicionar à última parte
        if reac_sim and partes:
            partes[-1] += " " + reac_sim
        
        # Para cada parte, fazer análise isolada
        respostas_combinadas = []
        for i, parte in enumerate(partes, 1):
            st.subheader(f"Bloco {i}: '{parte}'")
            # Extrair reação da parte
            reac_parte = ""
            for b in blocos:
                bloco_reac = b["entrada"].get("reacao", "").lower().strip()
                if bloco_reac and parte.lower().strip().endswith(bloco_reac):
                    reac_parte = bloco_reac
                    parte_clean = parte[:-len(bloco_reac)].rstrip()
                    break
            else:
                parte_clean = parte
            
            # Buscar similar para esta parte
            similares = retrieve_similar_blocks_alnulu(parte_clean, reac_parte, ctx_sim, "", dominio, top_k=1)
            if similares:
                sim_score, bloco_sim = similares[0]
                grounding = grounding_score(bloco_sim)
                concrete = is_concrete_block(bloco_sim)
                st.info(f"🔍 Melhor match (score: {sim_score:.2f}, grounding: {grounding:.2f}): '{bloco_sim['entrada']['texto']} {bloco_sim['entrada'].get('reacao', '')}'")
                # Reflexão para esta parte
                has_reac = bool(reac_parte.strip())
                has_ctx = bool(ctx_sim.strip())
                if concrete:
                    reflexao = "Isso é um conhecimento concreto: tem texto, reação, contexto e/ou pensamento internamente fundamentados."
                elif has_reac and has_ctx:
                    reflexao = "Isso é um conhecimento concreto: tem texto, reação, contexto e significado."
                else:
                    reflexao = "Isso é uma opinião ou bloco abstrato: só tem texto (e talvez reação), baseado em similaridade."
                st.write(f"Reflexão: {reflexao}")
                # Resposta sugerida para esta parte
                resposta_texto = bloco_sim['saidas'][0]['textos'][0]
                resposta_reacao = bloco_sim['saidas'][0].get('reacao', '')
                if sim_score < 0.5 and not concrete:
                    st.error(f"🚨 Alucinação detectada! Score baixo ({sim_score:.2f}) e bloco pouco fundamentado. Ativando aprendizado...")
                    resposta = "Estou alucinando... Vamos aprender juntos?"
                else:
                    texto_exato = normalize(parte_clean) == normalize(bloco_sim['entrada']['texto'])
                    reacao_exata = reac_parte == bloco_sim['entrada'].get('reacao', '')
                    if reacao_exata:
                        resposta = resposta_texto + (" " + resposta_reacao if resposta_reacao else "")
                    elif texto_exato:
                        palavras_resposta = Token(resposta_texto)
                        metade = max(1, len(palavras_resposta) // 2)
                        resposta = ' '.join(palavras_resposta[:metade]) + (" " + resposta_reacao if resposta_reacao else "")
                    else:
                        primeira_palavra = resposta_texto.split()[0] if resposta_texto.split() else resposta_texto
                        resposta = primeira_palavra + (" " + resposta_reacao if resposta_reacao else "")
                st.write(f"Resposta sugerida: {resposta}")
                respostas_combinadas.append(resposta)
            else:
                st.warning(f"❌ Nenhum bloco similar encontrado para '{parte}'.")
        
        if respostas_combinadas:
            st.write(f"Resposta combinada: {' '.join(respostas_combinadas)}")
    
    # Teste 3: Geração Autônoma
    st.subheader("3. Geração Autônoma")
    txt_auto = st.text_input("Texto para autonomia:", key="txt_auto")
    reac_auto = st.text_input("Reação:", key="reac_auto")
    ctx_auto = st.text_input("Contexto:", key="ctx_auto")
    thought_auto = st.text_input("Pensamento:", key="thought_auto")
    if st.button("Gerar Autônomo"):
        proposta = generate_autonomous_block(txt_auto, reac_auto, ctx_auto, thought_auto, dominio, memoria)
        st.code(proposta, language="text")
    
    # Teste 4: Encoding ALNULU
    st.subheader("4. Encoding ALNULU")
    word1 = st.text_input("Palavra 1:", key="word1")
    word2 = st.text_input("Palavra 2:", key="word2")
    if word1 and word2:
        vec1 = alnulu_encode(word1)
        vec2 = alnulu_encode(word2)
        sim = alnulu_similarity(vec1, vec2)
        st.write(f"Vetor '{word1}': {vec1}")
        st.write(f"Vetor '{word2}': {vec2}")
        st.write(f"Similaridade: {sim:.2f}")
    
    st.success("🎉 Teste concluído! Adam afiado na prática. :3 <3")


def generate_autonomous_block(entrada_texto: str, entrada_reacao: str, entrada_contexto: str, entrada_pensamento: str, dominio: str, memoria: dict, entrada_multivars: list = None, multivars_saida: list = None) -> str:
    """Gera um bloco INSEPA automaticamente usando o modelo treinado para autonomia real, não hardcoded."""
    # Garantir que checkpoint existe (carregar do Mongo se necessário)
    ensure_checkpoint_exists(dominio)
    
    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        # Fallback para hardcoded se não há modelo
        return f"""Índice mãe: {dominio}

Entrada: {entrada_texto}

Reação: {entrada_reacao}

Contexto: {entrada_contexto}

Pensamento Interno: {entrada_pensamento}

Saída:

1. Modelo não treinado ainda. Treine primeiro para autonomia completa.

Reação: 🤖

Contexto: Fallback autônomo
"""

    # Carregar modelo para geração autônoma
    data = torch.load(ckpt)
    if len(data) == 20:
        (state, maxE, maxRE, maxCE, maxPIDE, mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE, n_txt, max_out_len, max_ng, vS, all_out_markers, idx_to_txt) = data
    elif len(data) == 19:
        (state, maxE, maxRE, maxCE, maxPIDE, mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
         vE, vRE, vCE, vPIDE, n_txt, max_out_len, max_ng, vS, all_out_markers) = data
        idx_to_txt = {v: k for k, v in vS.items()}
    else:
        return f"""Índice mãe: {dominio}

Entrada: {entrada_texto}

Reação: {entrada_reacao}

Contexto: {entrada_contexto}

Pensamento Interno: {entrada_pensamento}

Saída:

1. Checkpoint incompatível.

Reação: ❌

Contexto: Erro de carregamento
"""

    out_vocab_size = n_txt
    n_emo = 1
    n_ctx = 1

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE), nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        out_vocab_size=out_vocab_size, max_out_len=max_out_len,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    model.load_state_dict(state)
    model.v_txt = vS
    model.idx_to_txt = idx_to_txt
    if all_out_markers:
        model.all_out_markers = all_out_markers
    model.eval()

    # Preparar entrada para o modelo (usar entrada atual como base)
    universo = memoria["IM"][dominio]
    blocos = universo["blocos"]
    if not blocos:
        return f"""Índice mãe: {dominio}

Entrada: {entrada_texto}

Reação: {entrada_reacao}

Contexto: {entrada_contexto}

Pensamento Interno: {entrada_pensamento}

Saída:

1. Nenhum bloco existente para basear autonomia.

Reação: 📭

Contexto: Base vazia
"""

    # Selecionar um bloco base apenas se ele estiver bem alinhado com a estrutura INSEPA.
    bloco_base = select_structured_autonomous_candidate(entrada_texto, entrada_reacao, entrada_contexto, entrada_pensamento, dominio, memoria)
    if bloco_base is None:
        return f"""Índice mãe: {dominio}

Entrada: {entrada_texto}

Reação: {entrada_reacao}

Contexto: {entrada_contexto}

Pensamento Interno: {entrada_pensamento}

Saída:

1. Bloco base não encontrado ou insuficientemente fundamentado para autonomia segura.

Reação: 🧠

Contexto: Falha de alinhamento INSEPA
"""

    # Preparar “contexto” para geração autônoma usando blocos existentes
    similares = retrieve_similar_blocks_alnulu(entrada_texto, entrada_reacao, entrada_contexto, entrada_pensamento, dominio, top_k=3)
    exemplos = []
    for score, b in similares:
        if not is_concrete_block(b):
            continue
        resp = b["saidas"][0]["textos"][0] if b["saidas"] and b["saidas"][0].get("textos") else ""
        pens = b["entrada"].get("pensamento_interno", "")
        exemplos.append(f"{b['entrada']['texto']} => {resp} (pensamento: {pens})")

    # Injetar “consciência” (orientação) no pensamento interno para guiar geração
    if exemplos:
        guia = " | ".join(exemplos[:3])
        entrada_pensamento = f"{entrada_pensamento} | Baseado nos exemplos: {guia}" if entrada_pensamento else f"Baseado nos exemplos: {guia}"

    # Calcular start_value baseado no fim das saídas do bloco base
    fim_saida_ultimo = float(bloco_base["saidas"][0]["fim"])
    start_value = fim_saida_ultimo + 0.01

    # Featurizar entrada autônoma como um bloco INSEPA completo: texto + reação + contexto + pensamento interno.
    texto_base = normalize(entrada_texto) or ""
    reacao_base = normalize(entrada_reacao) or ""
    contexto_base = normalize(entrada_contexto) or ""
    pensamento_base = normalize(entrada_pensamento) or ""

    texto_tokens = Token(texto_base)
    reacao_tokens = Token(reacao_base)
    contexto_tokens = Token(contexto_base)
    pensamento_tokens = Token(pensamento_base)

    E = texto_tokens + reacao_tokens
    RE = [entrada_reacao] if entrada_reacao else []
    CE = contexto_tokens + pensamento_tokens
    pensamento_limpo = entrada_pensamento.strip('"')
    partes = pensamento_limpo.split('.')[:3]
    PIDE_full = []
    for parte in partes:
        PIDE_full.extend(Token(parte.strip()))
    PIDE_limited = PIDE_full[:3]

    # Usar dimensões do modelo treinado (do checkpoint)
    # maxE, maxRE, maxCE, maxPIDE já carregados do checkpoint

    # Pad/truncate
    def pad_list(lst, max_len):
        return (lst + [0] * max_len)[:max_len]

    E_ids = pad_list([vE.get(ng, vE.get(UNK, 0)) for t in E for ng in generate_ngrams(t, N_GRAM)], maxE * max_ng)
    E_val_idxs = pad_list([val_to_idx_E.get(t, 0) for t in E], maxE)
    E_vals = pad_list([0.0] * len(E), maxE)  # Placeholder, since tokens are strings
    E_moms = pad_list([0] * len(E), maxE)  # Placeholder
    E_pos = pad_list([0.0] * len(E), maxE)  # Simplificado

    RE_ids = pad_list([vRE.get(ng, vRE.get(UNK, 0)) for t in RE for ng in generate_ngrams(t, N_GRAM)], maxRE * max_ng)
    RE_val_idxs = pad_list([val_to_idx_RE.get(t, 0) for t in RE], maxRE)
    RE_vals = pad_list([0.0] * len(RE), maxRE)  # Placeholder
    RE_moms = pad_list([0] * len(RE), maxRE)
    RE_pos = pad_list([0.0] * len(RE), maxRE)

    CE_ids = pad_list([vCE.get(ng, vCE.get(UNK, 0)) for t in CE for ng in generate_ngrams(t, N_GRAM)], maxCE * max_ng)
    CE_val_idxs = pad_list([val_to_idx_CE.get(t, 0) for t in CE], maxCE)
    CE_vals = pad_list([0.0] * len(CE), maxCE)  # Placeholder
    CE_moms = pad_list([0] * len(CE), maxCE)
    CE_pos = pad_list([0.0] * len(CE), maxCE)

    PI_ids = pad_list([vPIDE.get(ng, vPIDE.get(UNK, 0)) for t in PIDE_limited for ng in generate_ngrams(t, N_GRAM)], maxPIDE * max_ng)
    PI_val_idxs = pad_list([val_to_idx_PIDE.get(t, 0) for t in PIDE_limited], maxPIDE)
    PI_vals = pad_list([0.0] * len(PIDE_limited), maxPIDE)  # Placeholder
    PI_moms = pad_list([0] * len(PIDE_limited), maxPIDE)
    PI_pos = pad_list([0.0] * len(PIDE_limited), maxPIDE)

    x = {
        "E": torch.tensor([E_ids], dtype=torch.long),
        "E_val": torch.tensor([E_vals], dtype=torch.float32),
        "E_mom": torch.tensor([E_moms], dtype=torch.long),
        "E_pos": torch.tensor([E_pos], dtype=torch.float32),
        "E_val_idx": torch.tensor([E_val_idxs], dtype=torch.long),
        "RE": torch.tensor([RE_ids], dtype=torch.long),
        "RE_val": torch.tensor([RE_vals], dtype=torch.float32),
        "RE_mom": torch.tensor([RE_moms], dtype=torch.long),
        "RE_pos": torch.tensor([RE_pos], dtype=torch.float32),
        "RE_val_idx": torch.tensor([RE_val_idxs], dtype=torch.long),
        "CE": torch.tensor([CE_ids], dtype=torch.long),
        "CE_val": torch.tensor([CE_vals], dtype=torch.float32),
        "CE_mom": torch.tensor([CE_moms], dtype=torch.long),
        "CE_pos": torch.tensor([CE_pos], dtype=torch.float32),
        "CE_val_idx": torch.tensor([CE_val_idxs], dtype=torch.long),
        "PIDE": torch.tensor([PI_ids], dtype=torch.long),
        "PIDE_val": torch.tensor([PI_vals], dtype=torch.float32),
        "PIDE_mom": torch.tensor([PI_moms], dtype=torch.long),
        "PIDE_pos": torch.tensor([PI_pos], dtype=torch.float32),
        "PIDE_val_idx": torch.tensor([PI_val_idxs], dtype=torch.long),
    }

    with torch.no_grad():
        out = model(x, start_value=start_value)

    # Gerar resposta autônoma
    generated_logits = out["out"][0]
    generated_ids = generated_logits

    # Decodificar para autonomia (usar bloco=None)
    generated_responses = model.decode_tokens(generated_ids, None, dominio)
    saida_texto = generated_responses[0] if generated_responses else "Resposta autônoma gerada."

    # Reação e contexto autônomos (simplificados)
    saida_reacao = "🤖"  # Autônomo
    saida_contexto = "Resposta gerada autonomamente pelo modelo treinado"

    template = f"""Índice mãe: {dominio}

Entrada: {entrada_texto}

Reação: {entrada_reacao}

Contexto: {entrada_contexto}

Pensamento Interno: {entrada_pensamento}

Saída:

1. {saida_texto}

Reação: {saida_reacao}

Contexto: {saida_contexto}
"""
    if entrada_multivars:
        template = template.replace("Pensamento Interno: {entrada_pensamento}", f"Pensamento Interno: {entrada_pensamento}\n\nMultivars_Texto_Entrada: {' | '.join(entrada_multivars)}")
    if multivars_saida:
        template = template.replace("Reação: {saida_reacao}", f"Reação: {saida_reacao}\n\nMultivars_Texto_Saida: {' | '.join(multivars_saida)}")
    return template


def parse_autonomous_block_template(template: str) -> dict:
    """Parseia o texto gerado por generate_autonomous_block em campos estruturados."""
    header, _, body = template.partition("Saída:")

    def get_field(name: str, text: str) -> str:
        m = _re.search(rf"{_re.escape(name)}:\s*(.*)", text)
        return m.group(1).strip() if m else ""

    def get_multivars(name: str, text: str) -> list[str]:
        raw = get_field(name, text)
        if not raw:
            return []
        return [item.strip() for item in _re.split(r"\s*\|\s*", raw) if item.strip()]

    entrada_texto = get_field("Entrada", header)
    entrada_reacao = get_field("Reação", header)
    entrada_contexto = get_field("Contexto", header)
    entrada_pensamento = get_field("Pensamento Interno", header)
    entrada_multivars = get_multivars("Multivars_Texto_Entrada", header)

    saida_texto = ""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("1."):
            saida_texto = line[2:].strip()
            break

    saida_reacao = get_field("Reação", body)
    saida_contexto = get_field("Contexto", body)
    saida_multivars = get_multivars("Multivars_Texto_Saida", body)

    return {
        "entrada_texto": entrada_texto,
        "entrada_reacao": entrada_reacao,
        "entrada_contexto": entrada_contexto,
        "entrada_pensamento": entrada_pensamento,
        "entrada_multivars": entrada_multivars,
        "saida_texto": saida_texto,
        "saida_reacao": saida_reacao,
        "saida_contexto": saida_contexto,
        "saida_multivars": saida_multivars,
        "template": template,
    }


def auto_learn_and_add_block(entrada_texto: str, entrada_reacao: str, entrada_contexto: str, entrada_pensamento: str, dominio: str, default_pensamento: str = "") -> tuple[dict, str, str, str]:
    """Gera um bloco automaticamente e adiciona à memória como novo conhecimento.

    O pensamento interno (persona) pode ser fornecido diretamente ou herdado do
    `st.session_state.default_pensamento`.
    """
    # Injetar persona / pensamento interno se não fornecido explicitamente
    if not entrada_pensamento:
        entrada_pensamento = default_pensamento or st.session_state.get("default_pensamento", "")

    memoria = st.session_state.memoria
    # Garantir existência de universo (IM)
    if dominio not in memoria.get("IM", {}):
        memoria.setdefault("IM", {})[dominio] = {"blocos": []}

    proposta = generate_autonomous_block(entrada_texto, entrada_reacao, entrada_contexto, entrada_pensamento, dominio, memoria)
    parsed = parse_autonomous_block_template(proposta)

    bloco = {
        "bloco_id": str(uuid.uuid4()),
        "entrada": {
            "texto": entrada_texto,
            "Multivars_Texto_Entrada": parsed.get("entrada_multivars", []),
            "reacao": entrada_reacao,
            "Multivars_Reacao_Entrada": [],
            "contexto": entrada_contexto or "Aprendizado autônomo",
            "pensamento_interno": entrada_pensamento or default_pensamento or "Autocorreção do Adam",
        },
        "saidas": [
            {
                "textos": [parsed["saida_texto"] or entrada_texto],
                "Multivars_Texto_Saida": parsed.get("saida_multivars", []),
                "reacao": parsed["saida_reacao"] or "🤖",
                "Multivars_Reacao_Saida": [],
                "contexto": parsed["saida_contexto"] or "Aprendido autonomamente",
            }
        ],
    }

    if not is_concrete_block(bloco):
        bloco["entrada"]["contexto"] = entrada_contexto or "Aprendizado autônomo com correção"
        bloco["entrada"]["pensamento_interno"] = entrada_pensamento or default_pensamento or "Autocorreção do Adam"
        bloco["saidas"][0]["contexto"] = f"Correção autônoma: {bloco['entrada']['contexto']}"
        bloco["saidas"][0]["reacao"] = parsed.get("saida_reacao") or "🤖"
        bloco["meta"] = {"self_correction": True, "source": "autonomous_learning"}

    memoria["IM"][dominio]["blocos"].append(bloco)
    st.session_state.memoria = memoria
    auto_save_state()

    # Recriar cache de ALNULU com o novo bloco
    st.session_state.alnulu_cache = build_alnulu_cache(memoria)

    return bloco, parsed["saida_texto"], parsed["saida_reacao"], parsed["saida_contexto"]


def get_learning_status_summary() -> dict:
    """Resumo do estado de aprendizagem do Adam com os pesos 66/12/23."""
    reinforcements = st.session_state.get("reinforcements", {})
    if not reinforcements:
        return {
            "creator_weight": 0.66,
            "user_weight": 0.12,
            "autonomy_weight": 0.23,
            "autonomy_ready": False,
            "blocks": 0,
        }

    blocks = 0
    autonomy_ready = False
    for reinforcement in reinforcements.values():
        blocks += 1
        if reinforcement.get("human_likes", 0) > 0 and reinforcement.get("creator") == "Thaís D' Mariano":
            autonomy_ready = True

    return {
        "creator_weight": 0.66,
        "user_weight": 0.12,
        "autonomy_weight": 0.23,
        "autonomy_ready": autonomy_ready,
        "blocks": blocks,
    }


def main():
    st.set_page_config(layout="wide")
    # Para deploy no Streamlit Cloud ou similar:
    # 1. Faça upload do código para um repositório Git (GitHub).
    # 2. Vá para share.streamlit.io, conecte o repo e deploy.
    # 3. Para persistência, os dados ficam em session_state; arquivos JSON são backups locais.
    # Nota: Treinamento de IA pode ser lento na nuvem gratuita; considere recursos pagos se necessário.

    tema_claro = st.sidebar.checkbox("☀️ Tema claro", value=st.session_state.get("tema_claro", False), key="tema_claro")

    # CSS harmonizado, com cobertura pra widgets que o tema custom antigo deixava
    # escapar no branco padrão do Streamlit (code/json, dataframe, expander, alertas,
    # mensagens do chat, métricas) -- por isso o claro/escuro cobre tudo, não só o fundo.
    if tema_claro:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
            color: #1e1b2e;
        }
        .stButton>button {
            background: white;
            color: #1e1b2e;
            border: 1px solid #764ba2;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stButton>button:hover { background: #ede9fe; color: #1e1b2e; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stMarkdown li,
        .stMarkdown, label, .stCaption { color: #1e1b2e !important; }
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            background: white; color: #1e1b2e; border: 1px solid #764ba2;
        }
        .stDataFrame, [data-testid="stExpander"], [data-testid="stMetric"],
        .stCodeBlock, [data-testid="stJson"], .stAlert, [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.9) !important; color: #1e1b2e !important;
        }
        .stSidebar { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px); }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 50'%3E%3Ctext fill='rgba(255,255,255,0.1)' font-size='20' x='150' y='25' text-anchor='middle'%3E🤖 Adam 😊 Amor 💜 INSEPA 🌟%3C/text%3E%3C/svg%3E");
            background-repeat: repeat;
            background-size: 300px 50px;
        }
        .stButton>button {
            background: black;
            color: white;
            border: 1px solid white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background: #333;
            color: white;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stMarkdown li,
        .stMarkdown, label, .stCaption { color: white !important; }
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid white;
        }
        .stDataFrame, [data-testid="stExpander"], [data-testid="stMetric"],
        .stCodeBlock, [data-testid="stJson"], .stAlert, [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.1) !important; color: white !important;
        }
        .stSidebar {
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🤖 Adam Lovely AI - Sistema INSEPA")
    st.markdown("### Interface de Chat com IA Avançada")
    if st.session_state.get("admin", False):
        status = get_learning_status_summary()
        st.info(
            f"🧠 Peso do aprendizado: criadora 66% | usuários 12% | autonomia do Adam 23%. "
            f"Autonomia pronta: {'sim' if status['autonomy_ready'] else 'não'} | blocos com reforço: {status['blocks']}"
        )
        try:
            import torch
            torch_ver = torch.__version__
        except Exception:
            torch_ver = "não instalado"
        st.caption(f"Python em uso: {sys.executable} | Python {sys.version.split()[0]} | PyTorch {torch_ver}")

    # Informações de persistência (JSON local e opcionalmente MongoDB)
    if _mongo_enabled:
        st.success(f"✅ MongoDB ativado (DB: {MONGO_DB})")
    else:
        st.info("🔒 MongoDB não ativado. Use MONGO_URI+MONGO_DB para habilitar.")
        if _mongo_error:
            st.warning(f"(erro Mongo: {_mongo_error})")

    # Inicializar dados em session_state para persistência na nuvem
    # Sempre recarregar dados do arquivo para garantir sincronização
    st.session_state.memoria = carregar_json(ARQUIVO_MEMORIA, {"IM": {}})
    st.session_state.inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    # Carregar léxico derivado do inconsciente (universo0 / universos específicos / dominios)
    st.session_state.inconsciente_lexicon = load_inconsciente_lexicon(st.session_state.inconsciente)
    # Salva imediatamente ao entrar no site (garante persistência mesmo sem interação).
    auto_save_state()
    if "likes" not in st.session_state:
        st.session_state.likes = {}  # {bloco_id: {variacao: count}}

    # Cache para acelerar busca de similaridade ALNULU (evita recálculos caros a cada pergunta)
    if "alnulu_cache" not in st.session_state or st.session_state.get("_memoria_hash") != hash(json.dumps(st.session_state.memoria, sort_keys=True)):
        st.session_state._memoria_hash = hash(json.dumps(st.session_state.memoria, sort_keys=True))
        st.session_state.alnulu_cache = build_alnulu_cache(st.session_state.memoria)

    memoria = st.session_state.memoria
    inconsciente = st.session_state.inconsciente

    # Menu no canto esquerdo
    with st.sidebar:
        st.header("Menu")

        # Acesso público: apenas conversar e estatísticas
        if st.button("💬 Conversar"):
            st.session_state.menu = "conversar"
        if st.button("📊 Estatísticas"):
            st.session_state.menu = "estatisticas"
        if st.button("❌ Sair"):
            st.write("👋 Até mais!")
            st.stop()

        # Auto-save opcional: grava JSONs sempre que a interface é atualizada.
        # Só isso é inofensivo o bastante pra ficar público -- o resto controla
        # aprendizado/autonomia e é coisa só da criadora decidir.
        st.checkbox("💾 Auto-save (gravar JSON automaticamente)", value=st.session_state.get("auto_save", True), key="auto_save")

        # Acesso administrativo: tudo concentrado em Gerenciar IMs e Testes
        if st.session_state.get("admin", False):
            st.checkbox("🎓 Auto-aprender (criar blocos automaticamente)", value=st.session_state.get("auto_learn", True), key="auto_learn")
            st.checkbox("🤖 Harness automático no chat (admin)", value=st.session_state.get("harness_auto_admin", True), key="harness_auto_admin")
            st.checkbox("🧠 Modo autônomo do Adam (admin)", value=st.session_state.get("autonomous_mode", True), key="autonomous_mode")
            st.checkbox("🛡️ Autorizar aprendizado abstrato (admin)", value=st.session_state.get("admin_override_learning", False), key="admin_override_learning")

            if st.button("🏗️ Gerenciar IMs (admin)"):
                st.session_state.menu = "gerenciar"
            if st.button("🧪 Testes ALNULU (admin)"):
                st.session_state.menu = "testes"
            if st.button("💬 Opiniões pendentes (admin)"):
                st.session_state.menu = "opinioes"
            if st.button("🧪 Harness ao vivo (admin)"):
                st.session_state.menu = "harness"
            if st.button("📖 Codex ADAM INSEPA (admin)"):
                st.session_state.menu = "codex"

        # Modo Administrador
        with st.expander("🔐 Modo Administrador"):
            senha_input = st.text_input("Digite a senha:", type="password", key="admin_senha")
            if st.button("Entrar"):
                if senha_input == SENHA_ADMIN:
                    st.session_state.admin = True
                    st.success("✅ Acesso administrativo concedido!")
                    st.session_state.menu = "conversar"
                else:
                    st.error("❌ Senha incorreta.")

            if st.session_state.get("admin", False):
                st.text_area(
                    "💭 Pensamento interno padrão (persona) - admin",
                    value=st.session_state.get("default_pensamento", ""),
                    key="default_pensamento",
                    help="Este pensamento interno é usado apenas para geração automática de blocos e só deve ser alterado por administrador.",
                    height=100,
                )
                st.warning("⚠️ Reset limpa toda a sessão (histórico, mensagens, estados)")
                confirmar_reset = st.checkbox("Confirmo que desejo limpar a sessão", key="confirm_reset")
                if st.button("🧹 Resetar interface (limpar sessão)"):
                    if confirmar_reset:
                        st.session_state.clear()
                        try:
                            st.cache_data.clear()
                        except Exception:
                            pass
                        try:
                            st.cache_resource.clear()
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error("Marque a confirmação antes de resetar.")

                with st.expander("🧨 Hard reset (apagar dados e checkpoints)"):
                    st.error("Apaga memória, inconsciente, checkpoints e backups. Irreversível.")
                    confirmar_hard = st.checkbox("Confirmo que desejo apagar TODOS os dados", key="confirm_hard_reset")
                    confirmar_texto = st.text_input("Digite APAGAR para confirmar", key="confirm_hard_reset_text")
                    if st.button("🔥 Apagar tudo (hard reset)"):
                        if confirmar_hard and confirmar_texto.strip().upper() == "APAGAR":
                            erros = []
                            for alvo in [ARQUIVO_MEMORIA, ARQUIVO_INCONSCIENTE]:
                                if os.path.exists(alvo):
                                    try:
                                        os.remove(alvo)
                                    except Exception as e:
                                        erros.append(f"Falha ao remover {alvo}: {e}")
                            # Remover checkpoints insepa_*.pt
                            try:
                                for f in os.listdir('.'):
                                    if f.startswith('insepa_') and f.endswith('.pt') and os.path.isfile(f):
                                        try:
                                            os.remove(f)
                                        except Exception as e:
                                            erros.append(f"Falha ao remover {f}: {e}")
                            except Exception as e:
                                erros.append(f"Falha ao listar checkpoints: {e}")
                            # Remover backups
                            if os.path.exists('backup'):
                                try:
                                    shutil.rmtree('backup', ignore_errors=True)
                                except Exception as e:
                                    erros.append(f"Falha ao remover backup/: {e}")

                            # Remover config/cache do Streamlit no perfil do usuário
                            streamlit_home = os.path.join(os.path.expanduser('~'), '.streamlit')
                            if os.path.exists(streamlit_home):
                                try:
                                    shutil.rmtree(streamlit_home, ignore_errors=True)
                                except Exception as e:
                                    erros.append(f"Falha ao remover {streamlit_home}: {e}")

                            st.session_state.clear()
                            try:
                                st.cache_data.clear()
                            except Exception:
                                pass
                            try:
                                st.cache_resource.clear()
                            except Exception:
                                pass

                            if erros:
                                st.warning("Hard reset concluído com avisos:\n" + "\n".join(erros))
                            else:
                                st.success("Hard reset concluído. Dados, checkpoints e backups removidos.")
                            st.rerun()
                        else:
                            st.error("Marque a confirmação e digite APAGAR para prosseguir.")

    if "menu" not in st.session_state:
        st.session_state.menu = "conversar"

    # Garantir que usuários não-admin fiquem apenas em conversar/estatísticas
    if not st.session_state.get("admin", False) and st.session_state.menu not in ("conversar", "estatisticas"):
        st.session_state.menu = "conversar"

    if st.session_state.menu == "gerenciar":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar o Gerenciador de IMs.")
            return
        submenu_im(memoria, inconsciente)

        # Ações avançadas concentradas aqui para admins
        with st.expander("⚙️ Treino, Testes e Evolução (admin)"):
            acao = st.selectbox("Escolha a ação:", ["Treinar", "Testar", "Testar Adam Afiado", "Evoluir IA"], key="admin_acao")
            dom = prompt_dominio(acao.lower(), memoria)
            if dom and st.button(f"Executar {acao}", key="btn_admin_acao"):
                if dom not in memoria.get("IM", {}):
                    st.error(f"❌ Domínio '{dom}' não encontrado.")
                else:
                    if acao == "Treinar":
                        train(memoria, dom)
                    elif acao == "Testar":
                        test_model(memoria, dom)
                    elif acao == "Testar Adam Afiado":
                        submenu_testar_adam(memoria, inconsciente)
                    elif acao == "Evoluir IA":
                        submenu_testar_adam(memoria, inconsciente)
    elif st.session_state.menu == "testes":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar Testes ALNULU.")
            return
        submenu_testar_adam(memoria, inconsciente)
    elif st.session_state.menu == "opinioes":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar Opiniões pendentes.")
            return
        submenu_opinioes(memoria)
    elif st.session_state.menu == "harness":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar o Harness ao vivo.")
            return
        submenu_harness(memoria)
    elif st.session_state.menu == "codex":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar o Codex.")
            return
        submenu_codex()
    elif st.session_state.menu == "conversar":
        st.write("Áudio disponível. Ouça a voz do personagem escolhido agora!")
        dom = prompt_dominio("conversar", memoria)
        if dom:
            if dom in memoria["IM"]:
                infer(memoria, dom)
            else:
                st.error(f"❌ Domínio '{dom}' não encontrado.")
    elif st.session_state.menu == "estatisticas":
        submenu_estatisticas(memoria)

    # Auto-save (se habilitado) em cada rerun (inclui entrada no site)
    auto_save_state()


if __name__ == "__main__":
    main()
