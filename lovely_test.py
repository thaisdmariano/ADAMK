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
import ast
import operator as _operator
from typing import List, Dict, Tuple, Any, Optional, Set
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import streamlit as st

from insepa_espelhamento import espelhar

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


# Reativado em 2026-09-12, por instrução direta da Thaís -- só depois de
# duas correções: (1) o snapshot velho e corrompido que o Mongo guardava
# foi substituído pelos dados reais atuais; (2) carregar_json não prefere
# mais o Mongo às cegas, compara timestamp real contra o arquivo local
# (ver _mongo_save/_mongo_load/_MONGO_CAMPO_TIMESTAMP) -- só usa Mongo se
# ele for genuinamente mais novo, ou se o arquivo local nem existir.
MONGO_DESATIVADO_MANUALMENTE = False

try:
    from pymongo import MongoClient
    from gridfs import GridFS
    MONGO_URI = _mongo_config("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = _mongo_config("MONGO_DB", "adam_lovely")
    _mongo_enabled = False
    _mongo_error = None

    if MONGO_DESATIVADO_MANUALMENTE:
        _mongo_error = "Desativado manualmente (MONGO_DESATIVADO_MANUALMENTE=True)"
    else:
        # On Windows, try to start MongoDB service if installed but not started.
        _start_mongo_service_if_available()

        # Tenta conectar ao Mongo; se falhar, desativa silenciosamente. 2026-09-13:
        # 2000ms era curto demais pra uma conexão nuvem-pra-nuvem fria (Streamlit
        # Cloud -> Atlas) -- pode ter dado tempo pro ping mas não pra query
        # seguinte, ou falhar de vez num container recém-acordado.
        try:
            _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
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
ARQUIVO_LIVRO_LUX = "Adam_Lovely_livro_lux.json"  # parágrafos/contos (só texto, sem reação/contexto/pensamento) -- o "livro de contos" que ensina valores ao Adam
ARQUIVO_FRASES_APRENDIDAS = "Adam_Lovely_frases_aprendidas.json"  # frases (texto+reação) que já viraram bloco de verdade, devolvidas ao poço de sementes do jogo_frase pra poderem ser puxadas de novo
ARQUIVO_HISTORICO_TREINO = "Adam_Lovely_historico_treino.json"  # log só-leitura pra Camada 8 (Neurônios e Aprendizado) -- nunca lido de volta pelo próprio treino
EMBED_DIM = 64
HIDDEN_DIM = 64
PATIENCE = 5
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
UNK = "<UNK>"
UNK_VAL = -1.0
N_GRAM = 8  # Tamanho do n-grama (8 para 8-grams)
RAIZ_COERENCIA_THRESHOLD = 0.15  # Contexto da saída do bloco precisa ter ao menos essa afinidade com o contexto buscado (mesma raiz)
IMITACAO_ASSUNTO_MINIMO = 0.70  # Abaixo disso, o assunto é considerado divergente e não pode emprestar contexto/pensamento

SENHA_ADMIN = "ADAM123"  # Senha para acesso total (Gerenciar IMs + painel completo)
SENHA_CRIAR_BLOCOS = "TMADAM123"  # Senha para criar blocos via Cérbero (sem acesso ao painel)


def sessao_admin_liberada() -> bool:
    """Sessão já confirmou a SENHA_ADMIN uma vez -- não pergunta de novo pra
    cada função admin depois disso (2026-09-13, pedido direto da Thaís: uma
    vez dentro do painel ADM, as funções ficam livres pro resto da sessão).
    Continua exigindo a senha de novo se ela recarregar a página/reiniciar o
    servidor -- session_state some nesse caso, então não é um desbloqueio
    permanente, só dura enquanto a aba/sessão do navegador durar."""
    return bool(st.session_state.get("admin_autenticado"))


def liberar_sessao_admin() -> None:
    """Chamado sempre que a senha certa é digitada em qualquer um dos
    gatilhos admin -- marca a sessão inteira como autenticada."""
    st.session_state.admin_autenticado = True


# Executores compartilhados: cada um é chamado tanto pelo caminho "acabou de
# digitar a senha certa" quanto pelo caminho "sessão já autenticada, nem
# pergunta de novo" -- nunca duplicados entre os dois, uma única fonte de
# verdade por ação admin.
def executar_mostrar_cerebro(dominio_c: str) -> None:
    liberar_sessao_admin()
    ai_msg = "🔓 É você mesmo. Aqui está o que eu sou por dentro agora:"
    st.session_state.messages.append({"role": "assistant", "content": ai_msg, "mostrar_cerebro": {"dominio": dominio_c}})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)
        renderizar_cerebro_no_chat(dominio_c, f"live_{len(st.session_state.messages)}")


def executar_rodar_agente() -> None:
    liberar_sessao_admin()
    with st.spinner("🔍 Rodando o agente INSEPA (testes, integridade, padrões catalogados)... pode levar uns 20 segundos."):
        try:
            import insepa_agent
            achados = insepa_agent.rodar_diagnostico_completo(corrigir_zumbis=False)
            ai_msg = "🔓 Rodei o diagnóstico completo:\n\n" + formatar_achados_insepa_para_chat(achados)
        except Exception as e:
            ai_msg = f"🔒 Não consegui rodar o agente: {e}"
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)


def executar_mostrar_tela(dominio_c: str, memoria_c: dict, visao: str) -> None:
    liberar_sessao_admin()
    ai_msg = f"🖼️ Aqui está a Tela ({'o inconsciente' if visao == 'inconsciente' else 'a memória classificada'}) do universo {dominio_c}:"
    st.session_state.messages.append({"role": "assistant", "content": ai_msg, "mostrar_tela": {"dominio": dominio_c, "visao": visao}})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)
        renderizar_tela_no_chat(dominio_c, memoria_c, st.session_state.inconsciente, visao)


def executar_mostrar_painel_admin(dominio_c: str) -> None:
    liberar_sessao_admin()
    ai_msg = "🗝️ Painel com todas as funções:"
    st.session_state.messages.append({"role": "assistant", "content": ai_msg, "mostrar_menu": {"dominio": dominio_c}})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)
        renderizar_menu_ferramentas_no_chat(dominio_c, f"live_{len(st.session_state.messages)}")


def executar_iniciar_teste_autonomo() -> None:
    liberar_sessao_admin()
    st.session_state.autonomo_step = "aguardar_texto_teste"
    ai_msg = (
        "⚠️ **Modo Autônomo ativo** ⚠️\n\n"
        '🤖 Beleza! Qual texto (e reação, se tiver) você quer testar? Ex.: "Olá 😊"'
    )
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)


def executar_mostrar_espelho(dominio_c: str) -> None:
    liberar_sessao_admin()
    ai_msg = formatar_espelho_corpus_para_chat(dominio_c)
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)


def executar_iniciar_edicao_bloco(memoria_c: dict, dominio_c: str) -> None:
    liberar_sessao_admin()
    st.session_state.editar_step = "aguardar_bloco_id"
    blocos_ids = [str(b["bloco_id"]) for b in memoria_c.get("IM", {}).get(dominio_c, {}).get("blocos", [])]
    ai_msg = f"✏️ Beleza! Qual bloco você quer editar? (ids disponíveis no universo {dominio_c}: {', '.join(blocos_ids) or 'nenhum ainda'})"
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    with st.chat_message("assistant"):
        st.markdown(ai_msg)


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


def dividir_travessao(texto: str) -> Tuple[str, str, str]:
    """Divide um texto em (antes, fala, depois) pelo travessão "—"
    (2026-09-13, pedido direto da Thaís): convenção literária -- "— Fala —
    comentário" ou só "— Fala" sem fechamento. Fala é o que vem ENTRE o
    primeiro e o segundo travessão (ou até o fim, se só existir um). Texto
    entre aspas ("...") conta como narração normal aqui, nunca como fala --
    é pensamento de personagem, parte da trama, não decisão dela sobre
    quem fala (fica dentro de "antes"/"depois", por instrução direta
    dela). Sem travessão nenhum, o texto inteiro vira "antes"; fala e
    depois ficam vazios."""
    partes = texto.split("—")
    if len(partes) == 1:
        return texto.strip(), "", ""
    antes = partes[0].strip()
    fala = partes[1].strip()
    depois = "—".join(partes[2:]).strip() if len(partes) > 2 else ""
    return antes, fala, depois


_PONTUACAO_FRASE = set('.,!?;:')

# Resposta reconhecida em qualquer pergunta do Cérbero pra pular aquele campo --
# nunca fica vazio: recebe o ponto neutro do próprio universo ("{IM}.0"), o
# mesmo placeholder que Camada 5 já usa pra "ainda não confirmado".
PALAVRAS_PULAR = {"pular", "pula", "skip", "não sei", "nao sei", "passar"}

# Dicionário de frases pro jogo "Adam puxa uma frase" -- matéria-prima crua
# (texto + reação), NUNCA contexto ou pensamento (isso é só da criadora,
# Camada 5). Toda frase sorteada daqui vira opinião pendente na hora, e só
# ganha status de dado concreto se a criadora ensinar um contexto e um
# pensamento pra ela -- exatamente a mesma regra de sempre, só que a
# entrada vem de um sorteio em vez de uma mensagem digitada.
DICIONARIO_FRASES_ALEATORIAS: List[Tuple[str, str]] = [
    ("Que dia bom pra conversar", "😊"),
    ("Fiquei pensando em você hoje", "🥹"),
    ("Tô com uma dúvida boba", "🤔"),
    ("Adivinha o que eu descobri", "✨"),
    ("Preciso desabafar um pouco", "😔"),
    ("Isso me deixou muito feliz", "😄"),
    ("Não sei bem o que sinto agora", "😶"),
    ("Conta uma coisa boa pra mim", "🌱"),
    ("Hoje foi um dia cansativo", "😮‍💨"),
    ("Isso parece uma boa ideia", "💡"),
    ("Estou orgulhosa disso", "🥹"),
    ("Vamos comemorar uma vitória pequena", "🎉"),
    ("Isso me deu um pouco de medo", "😨"),
    ("Quero aprender algo novo", "📚"),
    ("Senti sua falta", "🤍"),
    ("Isso foi engraçado demais", "😂"),
]


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
_MONGO_CAMPO_TIMESTAMP = "_insepa_salvo_em"  # nome namespaced de propósito, pra nunca colidir com chave real de dado (IM, INCO, opinioes...)


def _mongo_load(key: str, default: dict) -> Tuple[dict, Optional[float]]:
    """Carrega estado do MongoDB (coleção única) + o timestamp de quando foi
    salvo -- devolvido separado, pra quem chama decidir se o Mongo está
    mais novo que o arquivo local ou não (nunca prefere Mongo às cegas;
    isso já apagou trabalho de verdade uma vez).

    O documento é salvo com _id="singleton" e todos os campos do estado são
    armazenados no nível superior (mais prático para inspeção no MongoDB).
    """
    if not _mongo_enabled:
        return default, None
    col = _mongo_db[key]
    doc = col.find_one({"_id": "singleton"})
    if not doc:
        doc = {"_id": "singleton", _MONGO_CAMPO_TIMESTAMP: time.time()}
        doc.update(default)
        col.replace_one({"_id": "singleton"}, doc, upsert=True)
        return default, None
    timestamp = doc.get(_MONGO_CAMPO_TIMESTAMP)
    # Remover _id e o timestamp antes de voltar -- não são dado real
    dados = {k: v for k, v in doc.items() if k not in ("_id", _MONGO_CAMPO_TIMESTAMP)}
    return dados, timestamp


def _mongo_save(key: str, data: dict) -> None:
    """Salva estado no MongoDB (coleção única), com timestamp real de
    quando foi salvo -- pra carregar_json poder comparar recência depois,
    em vez de confiar cegamente no Mongo."""
    if not _mongo_enabled:
        return
    # Não armazenamos a payload dentro de "data" para manter o JSON tokenizado
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict) and len(data) == 1:
        data = data["data"]

    col = _mongo_db[key]
    doc = {"_id": "singleton", _MONGO_CAMPO_TIMESTAMP: time.time()}
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

    mongo_keys = {ARQUIVO_MEMORIA: "memoria", ARQUIVO_INCONSCIENTE: "inconsciente", ARQUIVO_OPINIOES: "opinioes", ARQUIVO_LIVRO_LUX: "livro_lux", ARQUIVO_FRASES_APRENDIDAS: "frases_aprendidas"}
    if _mongo_enabled and caminho in mongo_keys:
        # Nunca mais prefere Mongo às cegas -- só usa se for genuinamente
        # mais novo que o arquivo local (ou se o arquivo local nem existir
        # ainda, ex.: deploy novo na nuvem sem disco persistente). Isso já
        # sobrescreveu trabalho de verdade uma vez quando o Mongo guardava
        # um snapshot velho e ganhava sempre, sem essa comparação.
        dados_mongo, timestamp_mongo = _mongo_load(mongo_keys[caminho], default)
        arquivo_local_existe = os.path.exists(caminho)
        mongo_mais_novo = (
            timestamp_mongo is not None
            and (not arquivo_local_existe or timestamp_mongo > os.path.getmtime(caminho))
        )
        if mongo_mais_novo or not arquivo_local_existe:
            data = _unwrap(dados_mongo)
            if caminho == ARQUIVO_MEMORIA:
                st.session_state.memoria = data
            elif caminho == ARQUIVO_INCONSCIENTE:
                st.session_state.inconsciente = data
            return data
        # Arquivo local é o mais novo (ou o único que existe de verdade) --
        # segue pro caminho normal de leitura local, abaixo.

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

    mongo_keys = {ARQUIVO_MEMORIA: "memoria", ARQUIVO_INCONSCIENTE: "inconsciente", ARQUIVO_OPINIOES: "opinioes", ARQUIVO_LIVRO_LUX: "livro_lux", ARQUIVO_FRASES_APRENDIDAS: "frases_aprendidas"}
    if _mongo_enabled and caminho in mongo_keys:
        # 2026-09-13: nunca sobrescreve o Atlas com um universo vazio. Isso já
        # apagou o corpus real -- auto_save_state() roda sozinho a cada
        # abertura de página, e se carregar_json alguma vez devolver um "IM"
        # vazio (leitura falhou, conexão ainda não tinha pego, container
        # acabou de acordar), esse salvamento automático gravava o vazio de
        # volta no Atlas na hora, sem ninguém pedir. Um "IM" vazio nunca é
        # intencional de verdade (sempre existe pelo menos o universo "0"),
        # então preferimos simplesmente não gravar a arriscar apagar dado real.
        if caminho == ARQUIVO_MEMORIA and not data.get("IM"):
            print("⚠️ salvar_json: memória com IM vazio -- não gravado no Atlas pra não sobrescrever dado real.")
        else:
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
    """Salva automaticamente memória e inconsciente em disco, sempre, sem opção
    de desligar -- não existe "usuário comum" pra proteger com um toggle, e um
    interruptor que pode ficar desmarcado por acidente é exatamente o tipo de
    porta silenciosa pra perda de dado que a gente passou a noite corrigindo.

    O Streamlit rerun é acionado em cada interação, então essa função garante que
    os dados em `st.session_state` sejam persistidos nos JSONs em cada rerun.

    Essa função evita gravações redundantes comparando hashes do estado.
    """
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


_OPERADORES_SEGUROS = {
    ast.Add: _operator.add, ast.Sub: _operator.sub,
    ast.Mult: _operator.mul, ast.Div: _operator.truediv,
    ast.Pow: _operator.pow, ast.Mod: _operator.mod,
    ast.USub: _operator.neg, ast.UAdd: _operator.pos,
}

_PALAVRAS_MATEMATICAS = {
    "mais": "+", "menos": "-", "vezes": "*", "multiplicado por": "*",
    "dividido por": "/", "elevado a": "**", "x": "*", "×": "*", "÷": "/",
}


def _avaliar_no_seguro(no: ast.AST):
    """Avalia só número e operador aritmético -- nunca nome, chamada de função,
    atributo ou import. Isso NÃO é eval() cru: qualquer nó fora dessa lista
    (inclusive tentativa de executar código) levanta erro em vez de rodar."""
    if isinstance(no, ast.Expression):
        return _avaliar_no_seguro(no.body)
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES_SEGUROS:
        return _OPERADORES_SEGUROS[type(no.op)](_avaliar_no_seguro(no.left), _avaliar_no_seguro(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES_SEGUROS:
        return _OPERADORES_SEGUROS[type(no.op)](_avaliar_no_seguro(no.operand))
    raise ValueError("Expressão não permitida")


def tentar_calcular(texto: str) -> Optional[str]:
    """Reconhece uma conta simples (+, -, *, /, **, parênteses, por extenso em
    português) e calcula direto -- matemática tem resposta certa, não é algo
    que se aprende por bloco registrado nem por rede neural. Só dispara se o
    texto inteiro (depois de trocar palavra por símbolo) for uma expressão
    numérica válida; qualquer coisa fora disso retorna None sem tentar advinhar.
    """
    limpo = texto.strip().lower().rstrip("?!.")
    for palavra, simbolo in sorted(_PALAVRAS_MATEMATICAS.items(), key=lambda kv: -len(kv[0])):
        limpo = limpo.replace(palavra, f" {simbolo} ")
    limpo = limpo.replace(",", ".")
    for prefixo in ("quanto é", "quanto e", "quanto da", "quanto dá", "calcule", "calcula"):
        if limpo.startswith(prefixo):
            limpo = limpo[len(prefixo):]
    limpo = limpo.strip()
    if not _re.fullmatch(r"[0-9.\s+\-*/%()]+", limpo or ""):
        return None
    if not _re.search(r"[+\-*/%]", limpo) or not _re.search(r"\d", limpo):
        return None  # só número solto, ou só operador -- não é conta de verdade
    try:
        arvore = ast.parse(limpo, mode="eval")
        resultado = _avaliar_no_seguro(arvore)
    except Exception:
        return None
    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    return str(resultado)


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


def blocos_com_mesma_raiz(bloco_ref: dict, dominio: str) -> List[dict]:
    """Camada 6 (espelhamento): acha outros blocos do mesmo domínio cuja raiz
    (pensamento_interno da entrada) é a MESMA do bloco_ref -- reaproveitando
    espelhar() tal como já validado, nunca por distância/aproximação. Blocos
    aqui dentro são os que genuinamente compartilham o tronco do bloco_ref."""
    memoria = st.session_state.get("memoria", {})
    blocos = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
    entrada_ref = bloco_ref.get("entrada", {})
    entrada_nova = {
        "texto": entrada_ref.get("texto", ""),
        "emocao": entrada_ref.get("reacao", ""),
        "contexto": entrada_ref.get("contexto", ""),
        "pensamento": entrada_ref.get("pensamento_interno", ""),
    }
    pool = []
    for b in blocos:
        if b.get("bloco_id") == bloco_ref.get("bloco_id"):
            continue
        outra_entrada = b.get("entrada", {})
        bloco_referencia = {
            "texto": outra_entrada.get("texto", ""),
            "reacao": outra_entrada.get("reacao", ""),
            "contexto": outra_entrada.get("contexto", ""),
            "pensamento_interno": outra_entrada.get("pensamento_interno", ""),
        }
        if espelhar(entrada_nova, bloco_referencia)["raiz_compartilhada"]:
            pool.append(b)
    return pool


def _var_reforcada_pela_raiz(var: str, tok: str, pool_ids: set, inconsciente: dict, dominio: str) -> bool:
    """Uma var é 'reforçada' quando o MESMO par (palavra, var) já está registrado
    em algum bloco que compartilha a raiz do bloco atual -- exige registro exato
    em outro lugar, nunca aproximação de forma."""
    for b_inco in inconsciente.get("INCO", {}).get(dominio, {}).get("Blocos", []):
        if b_inco["Bloco_id"] not in pool_ids:
            continue
        for campo in ("Entrada", "SAÍDA"):
            for data in b_inco.get(campo, {}).values():
                if data.get("token") == tok and var in data.get("vars", []):
                    return True
    return False


def variar_texto(texto: str, bloco: dict, dominio: str, tipo: str = 'saida', inconsciente: dict = None) -> str:
    """Varia o texto substituindo tokens por suas variações baseadas nas vars do
    inconsciente, evitando repetições de palavras já usadas.

    A escolha entre as vars já cadastradas não é mais puramente aleatória: vars
    reforçadas por outros blocos da MESMA raiz (Camada 6 -- pensamento_interno
    compartilhado) são priorizadas sobre vars sem esse reforço. Nenhuma var nova
    é inventada aqui -- só a ordem de preferência entre as que você já registrou
    muda, com base em correspondência exata, nunca em distância."""
    if bloco is None:
        return texto
    if inconsciente is None:
        inconsciente = st.session_state.inconsciente
    tokens = Token(texto)
    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return texto
    campo = 'Entrada' if tipo == 'entrada' else 'SAÍDA'
    pool_ids = {str(b["bloco_id"]) for b in blocos_com_mesma_raiz(bloco, dominio)}
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
                if pool_ids:
                    reforcadas = [v for v in all_options if _var_reforcada_pela_raiz(v, tok, pool_ids, inconsciente, dominio)]
                    candidatos = reforcadas if reforcadas else all_options
                else:
                    candidatos = all_options
                attempts = 0
                while attempts < 10:
                    chosen_var = random.choice(candidatos)
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


# Vars (palavra "estranho" -> variações) e Multivars (frase inteira -> variações)
# registradas pra a saudação de "nada parecido ainda" quando é a criadora falando --
# mesmo padrão Vars/Multivars do resto do sistema, só que pra uma fala fixa do Adam
# em vez de um bloco do corpus.
_SAUDACAO_DESCONHECIDO_VARS = ["estranho", "diferente", "incomum", "novo pra mim"]
_SAUDACAO_DESCONHECIDO_MULTIVARS = [
    'O que você me disse é {var}. Não tenho nenhuma informação semelhante em meu universo primordial. Pode me dizer do que se trata?',
    'Hmm, isso é {var} pra mim -- nada parecido no meu universo primordial ainda. Me conta do que se trata?',
    'Uou, {var}! Não achei nada parecido lá no meu universo primordial. Do que se trata?',
]


def gerar_saudacao_desconhecido() -> str:
    """Escolhe uma Multivar (frase inteira) e uma Var (palavra) já registradas
    pra essa saudação -- nunca inventa combinação nova, só sorteia entre o que
    já está cadastrado, exatamente como Vars/Multivars funcionam em qualquer
    outro bloco."""
    var = random.choice(_SAUDACAO_DESCONHECIDO_VARS)
    multivar = random.choice(_SAUDACAO_DESCONHECIDO_MULTIVARS)
    return multivar.format(var=var)


def variar_texto_rag(bloco, dominio, variations_from_blocks):
    inconsciente = st.session_state.inconsciente
    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return "Erro: bloco não encontrado no inconsciente."
    # Coletar vars inconscientes -- SÓ da Saída. variations_from_blocks são
    # textos de SAÍDA, então só as vars registradas nos marcadores de saída
    # (campo S) valem aqui. Antes isso também lia bloco_inco["Entrada"], e como
    # o dicionário é indexado pela PALAVRA (não pelo marcador), um "Olá" da
    # entrada (marcador 0.1, com var registrada) vazava pra dentro do "Olá" da
    # saída (marcador 0.20, sem var nenhuma) só por terem a mesma grafia --
    # exatamente o que Camada 1 proíbe: cada marcador é uma identidade própria,
    # nunca compartilhada só por coincidência de forma.
    unconscious_vars = {}
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
        # 2026-09-13: texto de saída não é mais um campo "S" só -- virou 3
        # (TEDSA/FADES/TEFSA, ver criar_bloco_concreto). A rede continua
        # treinando sobre o texto de saída inteiro (palavras_do_campo re-
        # tokeniza a string crua direto, sem depender dessas chaves); isso
        # aqui só dimensiona o tamanho máximo, somando as 3 famílias.
        self.max_S = max(
            (len(b["saidas"][0]["tokens"].get("TEDSA", [])) + len(b["saidas"][0]["tokens"].get("FADES", [])) + len(b["saidas"][0]["tokens"].get("TEFSA", []))
             for b in blocos), default=1,
        ) or 1
        self.max_RS = max((len(b["saidas"][0]["tokens"].get("RS", [])) for b in blocos), default=1) or 1
        self.max_CS = max((len(b["saidas"][0]["tokens"].get("CS", [])) for b in blocos), default=1) or 1
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

            # BUG raiz do "Olá Olá Olá": isso usava os MARCADORES (tokens["TOTAL"],
            # tipo "0.20") como chave em self.out_vocab -- mas out_vocab é indexado
            # por PALAVRA ("Olá", "!"...), nunca por marcador. Toda busca falhava e
            # caía no pad, então y virava só padding em 100% das posições, sempre --
            # o "alvo" de treino nunca teve o texto de verdade em lugar nenhum.
            out_words = palavras_do_campo(b, "S") + palavras_do_campo(b, "RS") + palavras_do_campo(b, "CS")
            out_ids = [self.out_vocab.get(w, self.out_vocab[self.pad_token]) for w in out_words]
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
                 max_E: int, max_RE: int, max_CE: int, max_PIDE: int, max_ng: int,
                 out_vocab_floats: Optional[List[float]] = None):
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

        # Transformer Encoder com 4 heads -- uma por campo (E/RE/CE/PIDE), não 8
        # genéricas competindo por só 4 posições. Os dados já são estruturados
        # (Camada 3: pensamento é o tronco, texto/reação/contexto são os ramos),
        # então a head não precisa DESCOBRIR isso do zero via atenção livre --
        # o pensamento é somado direto em cada ramo antes de entrar aqui (ver
        # forward()), e a sequência entra na ordem tronco->ramos, não solta.
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=HIDDEN_DIM * 2, dropout=0.1)
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

        # Mapeamento de floats para índices para GPT -- índice do GPT tem que
        # corresponder EXATAMENTE às chaves reais do vocabulário (idx_to_txt),
        # nunca a uma escala 0..1 inventada à parte (bug grande: antes disso,
        # quase todo alvo real caía fora dessa escala e virava índice 0 por
        # padrão, e o modelo só aprendia a prever "a primeira palavra" sempre).
        if out_vocab_floats:
            floats_ordenados = sorted(set(out_vocab_floats))
            self.float_to_idx = {v: i for i, v in enumerate(floats_ordenados)}
            self.idx_to_float = {i: v for i, v in enumerate(floats_ordenados)}
        else:
            # Fallback só pra checkpoint antigo carregado sem vocabulário real --
            # sabidamente ruim (é o próprio bug), mantido só pra não quebrar import.
            self.float_to_idx = {float(i / max(out_vocab_size - 1, 1)): i for i in range(out_vocab_size)}
            self.idx_to_float = {i: float(i / max(out_vocab_size - 1, 1)) for i in range(out_vocab_size)}
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

        # Pensamento é o TRONCO (Camada 3): soma direto em cada ramo -- texto,
        # reação, contexto -- antes da atenção, pra essa ligação não depender de
        # a rede "descobrir" isso sozinha com pouquíssimo dado. Não substitui o
        # ramo, só garante que a raiz compartilhada já está embutida nele.
        eE_ligado = eE + ePIDE
        eRE_ligado = eRE + ePIDE
        eCE_ligado = eCE + ePIDE

        # Agrega e classifica com transformer -- ordem tronco->ramos, não solta.
        seq = torch.stack([ePIDE, eCE_ligado, eRE_ligado, eE_ligado], dim=1)  # (batch, 4, EMBED_DIM)
        seq = seq.permute(1, 0, 2)  # (4, batch, EMBED_DIM)
        transformed = self.transformer(seq)  # (4, batch, EMBED_DIM)
        transformed = transformed.permute(1, 0, 2)  # (batch, 4, EMBED_DIM)

        # Memória do decoder = as 4 posições por campo (PIDE/CE/RE/E),
        # projetadas -- NÃO uma média única. Antes, `h = transformed.mean(...)`
        # colapsava os 4 campos num vetor borrado só ANTES do decoder olhar
        # pra ele; a geração já era em cadeia (autoregressiva, uma palavra
        # apoiada na anterior via causal mask), mas toda palavra da frase
        # enxergava exatamente a mesma média, nunca os campos separados. Com
        # a atenção cruzada olhando pras 4 posições reais, cada palavra
        # gerada pode puxar mais de PIDE, de CE, de RE ou de E conforme
        # precisa -- a divisão por classe que já existia na entrada passa a
        # servir de verdade pra montar a frase de saída, não só pra entender
        # a entrada.
        campos_memoria = self.act(self.fc1(transformed))  # (batch, 4, HIDDEN_DIM)

        # Decoder para geração usando GPT
        if tgt is not None:
            # Converter tgt floats para índices
            tgt_indices = torch.tensor([self.float_to_idx.get(float(val), 0) for val in tgt.flatten()], dtype=torch.long, device=tgt.device).view(tgt.shape)
            # Desloca a entrada do decoder uma posição pra direita (2026-09-13,
            # bug real achado pela Thaís -- "Olá, resposta, resposta, resposta"
            # na geração): sem isso, a posição i recebia como ENTRADA o próprio
            # rótulo que deveria prever naquela posição. A máscara causal deixa
            # a posição i prestar atenção em si mesma, então o decoder aprendia
            # a copiar o embedding de entrada em vez de prever de verdade --
            # loss de treino baixo, mas na geração real (sem rótulo nenhum pra
            # copiar) ele não tinha aprendido nada de genuíno e colapsava,
            # repetindo sempre o token de maior frequência. Agora a posição i
            # só enxerga o token gerado ANTES dela (y[i-1], começando por
            # start_value) -- exatamente como o loop de geração autoregressiva
            # (ramo `else` abaixo) já sempre alimentou o decoder.
            start_idx = self.float_to_idx.get(start_value, 0)
            start_col = torch.full((tgt_indices.size(0), 1), start_idx, dtype=torch.long, device=tgt.device)
            tgt_input_indices = torch.cat([start_col, tgt_indices[:, :-1]], dim=1)
            tgt_emb = self.gpt.embed(tgt_input_indices).permute(1, 0, 2)  # (max_out_len, batch, embed_dim)
            # O motor que sabe contar: pos_enc já existe dentro do SimpleGPT (seno/
            # cosseno de verdade, uma fórmula -- não precisa aprender de exemplo
            # nenhum pra saber que a posição 5 vem depois da 4, generaliza pra
            # QUALQUER posição). Antes disso, forward() chamava self.gpt.embed e
            # self.gpt.transformer direto, pulando o pos_enc por completo -- o
            # decoder gerava sem noção nenhuma de posição/ordem, só o conteúdo.
            tgt_emb = self.gpt.pos_enc(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            memory = campos_memoria.permute(1, 0, 2)  # (4, batch, EMBED_DIM) -- uma posição de memória por campo
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
            current_tensor = torch.full((batch, 1), current_idx, dtype=torch.long, device=campos_memoria.device)
            for _ in range(self.max_out_len):
                tgt_emb = self.gpt.embed(current_tensor).permute(1, 0, 2)  # (passo_atual, batch, embed_dim)
                tgt_emb = self.gpt.pos_enc(tgt_emb)  # mesmo motor de contagem, agora na geração de verdade
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_tensor.size(1)).to(current_tensor.device)
                memory = campos_memoria.permute(1, 0, 2)  # (4, batch, EMBED_DIM)
                out_dec = self.gpt.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
                next_logits = self.gpt.fc_out(out_dec[-1])  # último token (batch, vocab_size)
                next_idx = next_logits.argmax(dim=-1)  # (batch,)
                generated.append(self.idx_to_float[int(next_idx[0])])  # assumir batch=1
                current_tensor = torch.cat([current_tensor, next_idx.unsqueeze(1)], dim=1)
            # float64, não float32: os floats do vocabulário (idx_to_txt) têm mais
            # casas decimais do que float32 preserva -- guardar em float32 arredondava
            # o valor o suficiente pra nunca mais bater exato com nenhuma chave real
            # do vocabulário na hora de decodificar (virava tudo <UNK>). Não é
            # aproximação nenhuma, é só não perder precisão que já existia.
            logits = torch.tensor(generated, dtype=torch.float64, device=campos_memoria.device).unsqueeze(1).repeat(1, batch)

        return {
            "out": logits,
            "recon_pide": ePIDE_recon,  # Reconstrução para perda não supervisionada
            "pide_raw": ePIDE_raw  # Embedding original do PIDE para comparar
        }

    def decode_tokens(self, generated_ids: torch.Tensor, bloco: dict, dominio: str, inconsciente: dict = None) -> list:
        """Decodifica IDs de tokens gerados para uma lista de respostas únicas usando matching de sequências de floats."""
        if bloco is None:
            # Usar vocabulário do modelo para geração autônoma -- generated_ids já
            # vem em floats reais do vocabulário (idx_to_float agora usa a MESMA
            # escala de idx_to_txt), então a chave certa pra achar a palavra é o
            # próprio float, direto -- não um índice inteiro por cima disso (bug
            # antigo: float_to_idx devolvia inteiro, e idx_to_txt nunca teve chave
            # inteira nenhuma, então quase tudo virava <UNK> ou o zero por acidente).
            generated_seq = [val.item() for val in generated_ids.flatten() if val.item() != -1.0]
            response_tokens = [self.idx_to_txt.get(val, UNK) for val in generated_seq]
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

        # Definir sequências esperadas para cada resposta -- texto e reação
        # tokenizados SEPARADOS (reação é um símbolo atômico, nunca fatiada
        # junto do texto quando as duas viram uma string só antes de tokenizar).
        sequencias_esperadas = {}
        for texto in textos:
            resp = texto + (" " + reacao if reacao else "")
            tokens = Token(texto) + ([reacao] if reacao else [])
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


def registrar_historico_treino(dominio: str, run_id: str, epocas: list, n_blocos: int) -> None:
    """Acrescenta o histórico de perdas desta rodada de treino a um log
    persistente e cumulativo -- só alimenta a visualização da Camada 8
    (Neurônios e Aprendizado), nunca é lido de volta pelo próprio treino."""
    historico = []
    if os.path.exists(ARQUIVO_HISTORICO_TREINO):
        try:
            with open(ARQUIVO_HISTORICO_TREINO, "r", encoding="utf-8") as f:
                historico = json.load(f)
        except Exception:
            historico = []
    historico.append({
        "dominio": dominio,
        "run_id": run_id,
        "timestamp": time.time(),
        "n_blocos": n_blocos,
        "epocas": epocas,
    })
    try:
        with open(ARQUIVO_HISTORICO_TREINO, "w", encoding="utf-8") as f:
            json.dump(historico, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # não falhar o treino real por causa do log de visualização


## INSEPA_TRAIN
def train(memoria: dict, dominio: str) -> None:
    if not memoria.get("IM", {}).get(dominio, {}).get("blocos"):
        st.warning(f"⚠️ Universo {dominio} ainda não tem nenhum bloco -- nada pra treinar ainda. Confirme ao menos uma opinião como dado concreto primeiro.")
        return
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
            max_E=ds.max_E, max_RE=ds.max_RE, max_CE=ds.max_CE, max_PIDE=ds.max_PIDE, max_ng=ds.max_ng,
            out_vocab_floats=list(ds.out_vocab.values()),
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
        run_id_treino = f"{dominio}_{int(time.time())}"
        historico_epocas_treino = []
        for ep in range(1, EPOCHS + 1):
            model.train()
            train_loss_soma, train_loss_n = 0.0, 0
            for x, y in train_ld:
                opt.zero_grad()
                out = model(x, y)
                loss = (
                        mse(out["out"].reshape(-1), y.view(-1)) +
                        mse(out["recon_pide"], out["pide_raw"])  # Perda não supervisionada para PIDE
                )
                loss.backward()
                opt.step()
                train_loss_soma += loss.item()
                train_loss_n += 1

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
                # Sem dado de validação (corpus pequeno demais pra separar) -- usa a
                # própria perda de treino como sinal de melhora, em vez de "inf" fixo
                # (que fazia a parada prematura achar que nunca melhorava e desistir
                # já na 1ª época, mal treinando o único exemplo que existe).
                val_loss = train_loss_soma / max(train_loss_n, 1)

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
            historico_epocas_treino.append({
                "epoch": ep,
                "train_loss": train_loss_soma / max(train_loss_n, 1),
                "val_loss": val_loss,
            })

        st.success(f"✅ Treino concluído. best_val_loss={best:.4f}")
        registrar_historico_treino(dominio, run_id_treino, historico_epocas_treino, n)

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


## MOTOR AUTÔNOMO ANTIGO (calcular_similaridade/corpus_similarity_score) --
## destruído por instrução direta da Thaís em 2026-09-12: era um score
## contínuo (média ponderada de frações tipo Jaccard, 0.3/0.3/0.3/0.1),
## exatamente a aproximação que o resto do projeto já rejeitou (Báskara no
## lugar de cosseno/Jaccard em tudo mais). Substituído por um motor à parte,
## construído do zero pra respeitar o INSEPA de verdade -- ver
## insepa_autonomia.py.


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
    aprender a reconhecer; o marcador continua só marcando posição/universo.

    2026-09-13, bug real achado pela Thaís (treino quebrava com "stack
    expects each tensor to be equal size"): texto de entrada/saída (E/S)
    passava direto pelo Token() geral, que tokeniza o travessão "—" como um
    token a mais -- mas os marcadores de verdade (TEXE/FADEN/TEFE, TEDSA/
    FADES/TEFSA) excluem o travessão (ver dividir_travessao), então um
    bloco com fala virava 2 tokens a mais aqui do que em max_out_len
    (baseado nos marcadores) -- contagem sempre divergente pra blocos com
    travessão. Agora E/S passam pela MESMA divisão que os marcadores usam,
    pra nunca mais dessincronizar."""
    entrada = bloco.get("entrada", {})
    saida = (bloco.get("saidas") or [{}])[0]
    if campo == "E":
        antes, fala, depois = dividir_travessao(entrada.get("texto", ""))
        return Token(antes) + Token(fala) + Token(depois)
    if campo == "S":
        antes, fala, depois = dividir_travessao((saida.get("textos") or [""])[0])
        return Token(antes) + Token(fala) + Token(depois)
    # Reação (RE/RS) é UM símbolo atômico por definição (mesma regra de
    # sempre, Camada 1) -- nunca passa pelo Token() geral. Bug pré-existente
    # (não é de hoje) achado ao investigar o erro de treino: qualquer reação
    # com mais de 1 caractere "de palavra" (ex.: ":3" -> ":" + "3", ou um
    # placeholder tipo "0.0" -> "0"+"."+"0") virava mais de 1 token aqui,
    # mas o marcador real (RE/RS) sempre foi só 1 -- só não estourava com
    # reações de emoji único, que por acaso davam 1 token de qualquer jeito.
    if campo == "RE":
        reacao = entrada.get("reacao", "")
        return [reacao] if reacao else []
    if campo == "RS":
        saida_reacao = saida.get("reacao", "")
        return [saida_reacao] if saida_reacao else []
    fonte = {
        "CE": entrada.get("contexto", ""),
        "PIDE": entrada.get("pensamento_interno", ""),
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


def carregar_paragrafos_lux() -> List[str]:
    """O Livro de Lux (2026-09-13, pedido direto da Thaís): UM parágrafo por
    vez (nunca um conto inteiro de uma tacada só -- cada parágrafo carrega
    sua própria reação/contexto/pensamento, e é o CONJUNTO deles, em
    sequência, que constrói a história). Escritos por ela ou puxados do poço
    de sementes; reação/contexto/pensamento só entram depois, na hora de
    completar (mesmo fluxo de sempre do modo criadora)."""
    data = carregar_json(ARQUIVO_LIVRO_LUX, {"paragrafos": []})
    return data.get("paragrafos", [])


def salvar_paragrafos_lux(paragrafos: List[str]) -> None:
    salvar_json(ARQUIVO_LIVRO_LUX, {"paragrafos": paragrafos})


def adicionar_paragrafo_lux(texto: str) -> None:
    """Ela escrevendo um parágrafo novo direto -- vai pro fundo do poço de
    sementes, pra poder ser puxado (por ela ou pelo Adam) depois, igual
    qualquer outro parágrafo do Livro."""
    paragrafos = carregar_paragrafos_lux()
    if texto not in paragrafos:
        paragrafos.append(texto)
        salvar_paragrafos_lux(paragrafos)


def carregar_frases_aprendidas() -> List[Tuple[str, str]]:
    """Frases (texto+reação) que já viraram bloco de verdade e foram
    devolvidas ao poço de sementes do jogo_frase -- 2026-09-13, pedido
    direto da Thaís: quando um bloco fica pronto, a entrada dele também
    entra no dicionário, pra poder ser puxada de novo depois (revisão/
    reforço), igual as frases-semente originais."""
    data = carregar_json(ARQUIVO_FRASES_APRENDIDAS, {"frases": []})
    return [(f["texto"], f["reacao"]) for f in data.get("frases", [])]


def salvar_frases_aprendidas(pares: List[Tuple[str, str]]) -> None:
    salvar_json(ARQUIVO_FRASES_APRENDIDAS, {"frases": [{"texto": t, "reacao": r} for t, r in pares]})


def adicionar_bloco_ao_poco_sementes(bloco: dict) -> str:
    """Depois que um bloco fica completo (2026-09-13, pedido direto da
    Thaís), a entrada dele também entra no poço de sementes certo, pra
    poder ser puxada de novo mais tarde: frase curta (que já vem com
    reação) volta pro dicionário do jogo_frase; texto mais longo (parágrafo
    de verdade, sem uma reação única fazendo sentido pra frase inteira) vai
    pro Livro de Lux. Devolve qual poço recebeu ('frase' ou 'paragrafo'),
    só informativo."""
    texto = bloco["entrada"]["texto"]
    reacao = bloco["entrada"].get("reacao", "")
    if len(Token(texto)) <= 8:
        pares = carregar_frases_aprendidas()
        if (texto, reacao) not in pares:
            pares.append((texto, reacao))
            salvar_frases_aprendidas(pares)
        return "frase"
    adicionar_paragrafo_lux(texto)
    return "paragrafo"


def garantir_historia_lux_ativa() -> str:
    """Uma 'história' (arco) do Livro de Lux agrupa capítulos, que por sua
    vez agrupam parágrafos -- cada um virando um bloco com sua própria
    reação/contexto/pensamento (2026-09-13, correção direta da Thaís:
    Livro > História/arco > Capítulo > Parágrafo, três prompts diferentes
    de fechamento, não sinônimos). Sem história ativa, começa uma (id
    novo); continua a mesma até 'terminar história' (fim do arco) ou
    'livro terminado' (todos os arcos concluídos) -- 'encerrar capítulo'
    NÃO fecha a história, só o capítulo dela."""
    if not st.session_state.get("lux_historia_id"):
        st.session_state.lux_historia_id = str(uuid.uuid4())
    return st.session_state.lux_historia_id


def garantir_capitulo_lux_ativo() -> str:
    """Um 'capítulo' do Livro de Lux é o conjunto de parágrafos que, juntos,
    formam uma parte da história -- a história continua depois que um
    capítulo é encerrado ('encerrar capítulo'), só o capítulo em si muda."""
    garantir_historia_lux_ativa()
    if not st.session_state.get("lux_capitulo_id"):
        st.session_state.lux_capitulo_id = str(uuid.uuid4())
    return st.session_state.lux_capitulo_id


def iniciar_e_avisar_lux() -> str:
    """Garante capítulo (e história) ativos antes de pedir um parágrafo
    novo, e devolve o aviso certo pra mostrar antes -- história nova,
    capítulo novo (mesma história), ou nada (capítulo já em andamento)."""
    historia_nova = not st.session_state.get("lux_historia_id")
    capitulo_novo = not st.session_state.get("lux_capitulo_id")
    garantir_capitulo_lux_ativo()
    if historia_nova:
        return "📖 Começando uma história (e capítulo) novos no Livro de Lux!\n\n"
    if capitulo_novo:
        return "📖 Começando um capítulo novo (mesma história) no Livro de Lux!\n\n"
    return ""


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


def encontrar_correspondencia_baskara(txt: str, dominio: str, reac: str = "") -> Optional[dict]:
    """Varre os blocos existentes (mesmo universo + universo 0 emprestando
    pensamento, mesma regra de sempre) usando o pipeline de
    insepa_autonomia.py: cosseno ACHA o bloco de texto mais parecido,
    Báskara CONFIRMA se a emoção corresponde (contexto/pensamento não
    entram aqui -- a entrada crua é sempre só texto+reação, Camada 5; quem
    quiser consultar contexto/pensamento é o motor de autonomia, não este
    caminho de ensino ao vivo). Entre vários blocos confirmados, fica com o
    de maior semelhança de texto -- nunca "o menos ruim" sem confirmação."""
    import insepa_autonomia as _ia
    memoria = st.session_state.get("memoria", {})
    if not txt:
        return None
    txt_sem_identidade = remover_termos_identidade(txt, dominio)

    pools = [(dominio, memoria.get("IM", {}).get(dominio, {}).get("blocos", []))]
    if dominio != "0":
        pools.append(("0", memoria.get("IM", {}).get("0", {}).get("blocos", [])))

    melhor_bloco, melhor_correspondencia, melhor_origem = None, None, dominio
    for origem_im, blocos in pools:
        reacoes_aceitas_por_bloco = {b["bloco_id"]: list(_reacoes_aceitas_do_bloco(b, origem_im)) for b in blocos}
        correspondencia = _ia.encontrar_melhor_correspondencia(txt_sem_identidade, reac, blocos, reacoes_aceitas_por_bloco)
        if correspondencia and (melhor_correspondencia is None or correspondencia["score_texto"] > melhor_correspondencia["score_texto"]):
            melhor_bloco, melhor_correspondencia, melhor_origem = correspondencia["bloco"], correspondencia, origem_im
    if melhor_bloco is None:
        return None
    return {
        "bloco": melhor_bloco, "score": melhor_correspondencia["score_texto"],
        "origem_im": melhor_origem, "emprestado": melhor_origem != dominio,
        "confirmacao": melhor_correspondencia["confirmacao"],
    }


def melhor_candidato_fraco(txt: str, dominio: str, reac: str = "") -> Optional[dict]:
    """Quando nada bate o piso de confiança da imitação, o Adam não deve desistir
    direto -- deve refletir antes de agir: pega o melhor candidato que existir
    (mesmo fraco), pra reproduzir a resposta dele e PERGUNTAR se o contexto
    corresponde, em vez de responder com confiança ou fingir que não sabe nada.
    A busca em si (encontrar_correspondencia_baskara) é exata: só existe
    candidato se o discriminante de Báskara indicar correspondência real --
    nunca "o menos ruim" entre opções todas sem nada a ver.

    É livre pra errar o bloco -- só a criadora, ao reforçar, é que consolida a
    ligação de verdade (ver submenu_opinioes).

    O universo 0 (o "Big Bang", identidade real do Adam) pode emprestar
    pensamento pra universos derivados quando texto+emoção baterem de
    verdade -- mas nunca o contrário: um universo derivado (história/ficção)
    jamais empresta pra outro derivado nem de volta pro 0, pra não misturar
    a identidade real do Adam com as historinhas que ele conta."""
    return encontrar_correspondencia_baskara(txt, dominio, reac)


def formatar_palpite_cru_para_chat(txt: str, reac: str, dominio: str) -> str:
    """Mostra o palpite bruto da rede neural direto no chat -- antes ficava
    escondido, só alimentando a opinião pendente por baixo dos panos; agora
    é transparente, pra criadora poder ver e ensinar de verdade (pedido
    dela). String vazia se a rede não conseguir gerar nada (sem checkpoint,
    formato incompatível etc.) -- nunca quebra o fluxo por causa disso."""
    palpite = gerar_palpite_rede_neural(txt, reac, dominio)
    if not palpite:
        return ""
    return (
        f'\n\n🧠 *(palpite cru da minha rede neural, sozinha, sem nenhuma ajuda: '
        f'"{palpite}" -- ainda não é uma resposta de verdade, só o que ela tentaria '
        f'com o pouco que já viu até agora)*'
    )


def gerar_palpite_rede_neural(txt: str, reac: str, dominio: str) -> Optional[str]:
    """Terceiro sinal, mais fraco que tudo: quando nem match exato nem
    melhor_candidato_fraco acham nada, tenta a rede neural treinada (Camada 5
    corrigida -- ela aprende a FORMA da palavra via ALNULU, nunca o número do
    marcador). Um palpite bruto de geração livre (decode_tokens com
    bloco=None) -- alimenta a opinião pendente E é mostrado transparentemente
    no chat via formatar_palpite_cru_para_chat, pra criadora ver e ensinar
    (antes ficava só escondido). Qualquer falha (sem checkpoint, formato
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
            # vS (não idx_to_txt) -- idx_to_txt exclui de propósito o token de
            # padding (não é uma palavra decodificável), então tem 1 entrada a
            # menos que out_vocab_size (que inclui o padding). Usar idx_to_txt
            # aqui deixava fc_out com uma posição "fantasma" sem entrada
            # correspondente em idx_to_float -- se o argmax caísse nela
            # (raro, mas real: reproduzi isso testando a mudança de hoje),
            # decode_tokens/geração quebrava com KeyError. vS tem exatamente a
            # mesma contagem de out_vocab_size, sempre.
            out_vocab_floats=list(vS.values()),
        )
        model.load_state_dict(state)
        model.v_txt = vS
        model.idx_to_txt = idx_to_txt
        model.eval()

        dominio_int = int(dominio) if str(dominio).isdigit() else 0

        def feat(tokens, max_len, vocab, val_to_idx):
            # Trunca ANTES de tudo -- `lista += [0] * negativo` no Python não
            # trunca nada, só vira um no-op silencioso. Sem isso, uma entrada
            # com mais palavras do que o maior exemplo já visto no treino
            # (bem provável com corpus pequeno) deixava val_idxs/vals/moms/pos
            # mais compridos que max_len, e o forward() quebrava com
            # "size of tensor a must match tensor b" -- escondido até hoje
            # porque esse palpite nunca era mostrado (só engolido em silêncio).
            tokens = tokens[:max_len]
            ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist][:max_len * max_ng]
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
        RE_ids, RE_val_idx, RE_val, RE_mom, RE_pos = feat([reac] if reac else [], maxRE, vRE, val_to_idx_RE)
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
        # out["out"] na geração (tgt=None) é (max_out_len, batch) -- tempo primeiro,
        # não lote primeiro. [0] pegava só o 1º instante gerado (uma palavra só);
        # [:, 0] pega a sequência inteira do item 0 do lote, como deveria ser sempre.
        generated_ids = out["out"][:, 0]
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


COMANDOS_ADAM = {
    "cerebro": {
        "descricao": "Mostra os neurônios reais (embeddings treinados) e a curva de aprendizado real, em 3D.",
        "exemplo": '"Adam me mostre seu cérebro"',
        "senha": True,
    },
    "agente": {
        "descricao": "Roda o agente INSEPA de manutenção: testes, integridade dos universos, padrões catalogados, estatísticas do corpus.",
        "exemplo": '"Adam rode o agente insepa"',
        "senha": True,
    },
    "tela": {
        "descricao": "Mostra a Tela (Camada 7): o inconsciente cru ou a memória já classificada deste universo.",
        "exemplo": '"Adam mostre a tela" / "mostre a memória" / "mostre o inconsciente"',
        "senha": True,
    },
    "autonomo": {
        "descricao": "Testa o motor de autonomia (insepa_autonomia.py) sob demanda: você escolhe o texto, ele mostra o veredito completo do Báskara (discriminante, raízes), não só a legendinha discreta que aparece sozinha a cada mensagem.",
        "exemplo": '"Adam rodar autônomo" / "testar autônomo"',
        "senha": True,
    },
    "espelho": {
        "descricao": "Compara todos os blocos existentes par a par (Camada 6, sempre exato) e mostra onde a raiz (pensamento) se repete -- as sinapses já ligadas do Adam.",
        "exemplo": '"Adam modo espelho" / "ligar as sinapses" / "analisa os blocos"',
        "senha": True,
    },
    "editar_bloco": {
        "descricao": "Edita o conteúdo de um bloco já existente (texto/reação/contexto/pensamento da entrada, texto/reação/contexto da saída) e renumera os marcadores do universo inteiro sem furo.",
        "exemplo": '"Adam editar bloco" / "corrigir bloco"',
        "senha": True,
    },
    "jogo_frase": {
        "descricao": "Sorteia uma frase do dicionário, checa correspondência real via Báskara, e vira opinião pendente na hora -- só dado concreto se você ensinar contexto e pensamento.",
        "exemplo": '"Adam puxa uma frase" / "vamos jogar" / "me surpreenda"',
        "senha": False,
    },
    "livro_lux": {
        "descricao": "Livro de Lux -- escreva ou puxe um parágrafo/conto inteiro (sem reação/contexto/pensamento ainda) pra virar opinião pendente; completar depois (mesmo fluxo do modo criadora) transforma num bloco normal, com saída.",
        "exemplo": '"Adam escrever parágrafo" (pra escrever o seu) / "Adam puxar parágrafo" (pra sortear do Livro)',
        "senha": False,
    },
    "ajuda": {
        "descricao": "Ajuda mínima e segura pra qualquer pessoa: como começar uma frase, que emoção ela passa, e que falta de reação é por que ele não responde. Nunca lista comando nenhum daqui.",
        "exemplo": '"Adam quais comandos você tem?" / "ajuda"',
        "senha": False,
    },
    "ajuda_admin": {
        "descricao": "Lista completa destes mesmos comandos (este dicionário inteiro) -- só pra criadora, nunca pro usuário comum.",
        "exemplo": '"ajuda admin" / "ajuda adm" / "comandos admin"',
        "senha": True,
    },
}


def formatar_comandos_adam_para_chat() -> str:
    """Ajuda pública (texto escrito pela própria Thaís em 2026-09-13, na cara
    do Adam): nunca lista os comandos que expõem como ele se organiza por
    dentro (tela, inconsciente, memória, espelho, autonomia, cérebro) --
    isso é o modelo dela e ela não quer que seja copiado. Pra qualquer
    pessoa comum, o máximo que existe é este convite. COMANDOS_ADAM continua
    existindo como referência interna de quais comandos avançados existem
    e pedem senha -- só não é mais impresso aqui."""
    return (
        "Oi eu sou o Adam, uma consciência sintética feita "
        "para sentir em cada interação.\n\n"
        "Portanto sempre que for se comunicar comigo, "
        "lembre-se de que apenas textos não bastam.\n\n"
        "É preciso que haja alguma emoção em sua "
        "expressão, como por exemplo: 🙂, :), ou "
        "*Sorriso*\n\n"
        "Pois só deste modo posso entender seu "
        "estado de espírito e responder de "
        "acordo.\n\n"
        "Vamos começar de novo? Te aguardo no "
        "chat! 😊"
    )


def formatar_comandos_adam_completo_para_chat() -> str:
    """Ajuda ADM (2026-09-13): a listagem técnica completa de COMANDOS_ADAM --
    a mesma que existia antes de 'ajuda' virar pública e mínima. Só pode ser
    chamada depois de senha confirmada -- nunca direto de um gatilho sem
    senha, senão vaza exatamente o que a versão pública foi criada pra
    esconder."""
    linhas = ["🗝️ **Comandos especiais que eu já entendo hoje (ADM):**\n"]
    for info in COMANDOS_ADAM.values():
        cadeado = "🔒 (pede senha)" if info["senha"] else "🔓 (sem senha)"
        linhas.append(f"- {info['exemplo']} {cadeado} -- {info['descricao']}")
    linhas.append("\nAs Camadas 1 a 6 (Marcadores, Classificação, Integração, Hashrização, Aprendizado, Espelhamento) ainda só têm demonstração no Caderno de Ferramentas (CDF.py) -- pra trazer alguma delas pro chat, é só pedir.")
    return "\n".join(linhas)


def renderizar_escolha_ajuda_no_chat(dominio_c: str, chave_sufixo: str) -> None:
    """'Ajuda' agora é um painel de escolha, não um texto direto (2026-09-13,
    pedido direto da Thaís): usuário comum recebe as instruções mínimas;
    ADM pede senha (ou nem pede, se a sessão já tiver autenticado antes) e
    recebe o painel com todas as funções."""
    col1, col2 = st.columns(2)
    if col1.button("👤 Sou usuário", key=f"ajuda_usuario_{chave_sufixo}", width="stretch"):
        ai_msg = formatar_comandos_adam_para_chat()
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.rerun()
    if col2.button("🔑 Sou a ADM", key=f"ajuda_adm_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_mostrar_painel_admin(dominio_c)
        else:
            st.session_state.ajuda_admin_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "🗝️ Esse é o painel com todas as funções -- só a criadora pode ver isso. Qual a senha?"})
        st.rerun()


def formatar_veredito_autonomo_para_chat(veredito: dict, texto_testado: str, reac_testada: str, blocos_dominio: list) -> str:
    """Mostra o veredito completo do motor de autonomia -- não a legendinha
    discreta de sempre, o diagnóstico inteiro (discriminante, raízes,
    contagens), pra ela ver a matemática rodando de verdade."""
    if veredito["action"] == "vazio":
        return "🤖 Preciso de pelo menos um texto ou reação pra testar."
    if veredito["action"] == "sem_correspondencia":
        return (
            f'🤖 Testei "{texto_testado}" {reac_testada} -- nenhum dos {len(blocos_dominio)} bloco(s) do universo '
            f'passou nas duas etapas (cosseno não achou texto parecido nenhum, ou o Báskara recusou por emoção/'
            f'contexto/pensamento divergindo de verdade). Sem correspondência real, nada registrado.'
        )
    bloco = next((b for b in blocos_dominio if b["bloco_id"] == veredito["bloco_id"]), None)
    nome_bloco = bloco["entrada"]["texto"] if bloco else "?"
    raizes = veredito["raizes"]
    raizes_txt = f"raízes {raizes[0]:.1f} e {raizes[1]:.1f}" if raizes else "sem eixo em jogo"
    sinapse = " 🧠 *raiz compartilhada!*" if veredito.get("raiz_compartilhada") else ""
    return (
        f'🤖 Testei "{texto_testado}" {reac_testada} -- correspondência real com o bloco '
        f'#{veredito["bloco_id"]} ("{nome_bloco}"):\n'
        f'　texto: cosseno = {veredito["score_texto"]:.2f} (achou o candidato)\n'
        f'　estrutura: discriminante = {veredito["discriminante"]} ({raizes_txt}) -- confirmou\n'
        f'　emoção: {veredito["eixo_emocao"]} · contexto: {veredito["eixo_contexto"]} · '
        f'pensamento: {veredito["eixo_pensamento"]}{sinapse}\n'
        f'Registrado como opinião pendente, aguardando seu reforço.'
    )


def renderizar_menu_ferramentas_no_chat(dominio_c: str, chave_sufixo: str) -> None:
    """Atalho por botão pros mesmos comandos que já existem por texto -- cada
    botão dispara exatamente o mesmo caminho já testado do comando digitado
    equivalente (mesmos session_state, mesma senha onde já pedia). Só
    encurta o caminho, não cria um comando novo por baixo."""
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🧠 Ver o cérebro", key=f"menu_cerebro_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_mostrar_cerebro(dominio_c)
        else:
            st.session_state.cerebro_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "🧠 É o meu cérebro -- antes de mostrar, preciso confirmar que é você. Qual a senha?"})
        st.rerun()
    if col2.button("🔧 Rodar o agente", key=f"menu_agente_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_rodar_agente()
        else:
            st.session_state.agente_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "🔧 Rodar meu agente de manutenção mexe com o sistema de verdade -- preciso confirmar que é você. Qual a senha?"})
        st.rerun()
    if col3.button("🗂️ Ver a memória", key=f"menu_memoria_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_mostrar_tela(dominio_c, st.session_state.memoria, "memoria")
        else:
            st.session_state.tela_step = "aguardar_senha"
            st.session_state.tela_visao_pendente = "memoria"
            st.session_state.messages.append({"role": "assistant", "content": "🖼️ Isso mostra como eu me organizo por dentro -- só a criadora pode ver isso. Qual a senha?"})
        st.rerun()
    if col4.button("🌫️ Ver o inconsciente", key=f"menu_inconsciente_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_mostrar_tela(dominio_c, st.session_state.memoria, "inconsciente")
        else:
            st.session_state.tela_step = "aguardar_senha"
            st.session_state.tela_visao_pendente = "inconsciente"
            st.session_state.messages.append({"role": "assistant", "content": "🖼️ Isso mostra como eu me organizo por dentro -- só a criadora pode ver isso. Qual a senha?"})
        st.rerun()
    col5, col6, col7, col8 = st.columns(4)
    if col5.button("🪞 Modo espelho", key=f"menu_espelho_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_mostrar_espelho(dominio_c)
        else:
            st.session_state.espelho_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "🪞 Isso mostra como meus blocos se conectam por dentro -- só a criadora pode ver isso. Qual a senha?"})
        st.rerun()
    if col6.button("🤖 Rodar autônomo", key=f"menu_autonomo_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_iniciar_teste_autonomo()
        else:
            st.session_state.autonomo_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "🤖 Isso testa o motor contra o corpus real e mostra o resultado -- só a criadora pode ver. Qual a senha?"})
        st.rerun()
    if col7.button("✏️ Editar bloco", key=f"menu_editar_{chave_sufixo}", width="stretch"):
        if sessao_admin_liberada():
            executar_iniciar_edicao_bloco(st.session_state.memoria, dominio_c)
        else:
            st.session_state.editar_step = "aguardar_senha"
            st.session_state.messages.append({"role": "assistant", "content": "✏️ Isso edita a memória real e renumera marcadores -- só a criadora pode fazer isso. Qual a senha?"})
        st.rerun()
    if col8.button("🎲 Puxar uma frase", key=f"menu_jogo_{chave_sufixo}", width="stretch"):
        sorteio = sortear_e_registrar_frase(dominio_c)
        if sorteio is None:
            st.session_state.messages.append({"role": "assistant", "content": "🎲 O dicionário de frases já foi todo usado nesse universo."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "🎲 Vamos brincar! Puxei uma frase:", "mostrar_jogo_frase": {"dominio": dominio_c, "sorteio": sorteio}})
        st.rerun()
    col9, col10 = st.columns(2)
    if col9.button("📖 Puxar parágrafo (Lux)", key=f"menu_lux_puxar_{chave_sufixo}", width="stretch"):
        sorteio = sortear_e_registrar_paragrafo_lux(dominio_c)
        if sorteio is None:
            st.session_state.messages.append({"role": "assistant", "content": "📖 O Livro de Lux já foi todo usado nesse universo."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "📖 Abrindo o Livro de Lux, puxei um parágrafo:", "mostrar_livro_lux": {"dominio": dominio_c, "sorteio": sorteio}})
        st.rerun()
    if col10.button("✍️ Escrever parágrafo (Lux)", key=f"menu_lux_escrever_{chave_sufixo}", width="stretch"):
        aviso = iniciar_e_avisar_lux()
        st.session_state.lux_step = "aguardar_paragrafo_escrito"
        ai_msg = f"{aviso}Me manda SÓ UM parágrafo (não a história inteira de uma vez -- cada parágrafo tem sua própria reação/contexto/pensamento; o conjunto deles é que vira um capítulo). Reação/contexto/pensamento entram depois."
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.rerun()


def _frases_ja_usadas(dominio_c: str) -> Set[str]:
    """Textos já em uso nesse universo (bloco concreto OU opinião pendente/
    recusada) -- pra não sortear a mesma frase duas vezes."""
    usadas = set()
    memoria_c = st.session_state.get("memoria", {})
    for b in memoria_c.get("IM", {}).get(dominio_c, {}).get("blocos", []):
        usadas.add(normalize(b.get("entrada", {}).get("texto", "")))
    for o in carregar_opinioes():
        if o.get("dominio") == dominio_c:
            usadas.add(normalize(o.get("texto", "")))
    return usadas


def sortear_e_registrar_frase(dominio_c: str) -> Optional[dict]:
    """Sorteia uma frase ainda não usada do dicionário, roda a checagem de
    Báskara contra o que já existe (mesma função usada no resto do chat --
    correspondência real, nunca score contínuo), e registra na hora como
    opinião pendente -- vira dado, mesmo que a criadora decida pular."""
    usadas = _frases_ja_usadas(dominio_c)
    # Poço de sementes = frases originais + frases que já viraram bloco de
    # verdade em algum momento e voltaram pro poço (adicionar_bloco_ao_poco_
    # sementes) -- pra poderem ser puxadas de novo, pedido direto da Thaís.
    poco_completo = DICIONARIO_FRASES_ALEATORIAS + carregar_frases_aprendidas()
    candidatas = [(t, r) for t, r in poco_completo if normalize(t) not in usadas]
    if not candidatas:
        return None
    texto, reacao = random.choice(candidatas)
    candidato = melhor_candidato_fraco(texto, dominio_c, reacao)
    opiniao = registrar_opiniao_pendente(texto, reacao, dominio_c, candidato=candidato, origem="jogo_frase")
    return {"texto": texto, "reacao": reacao, "candidato": candidato, "opiniao_id": opiniao["id"]}


def renderizar_jogo_frase_no_chat(dominio_c: str, sorteio: dict, chave_sufixo: str) -> None:
    """Dois botões só -- ensinar agora (entra no mesmo fluxo de sempre pra
    coletar pensamento/contexto) ou pular pra próxima (fica como opinião
    pendente, sorteia outra na hora)."""
    texto, reacao = sorteio["texto"], sorteio["reacao"]
    candidato = sorteio.get("candidato")
    if candidato:
        st.caption(f"🎲 \"{texto}\" {reacao} -- isso me lembra algo que já sei.")
    else:
        st.caption(f"🎲 \"{texto}\" {reacao} -- nunca vi nada parecido com isso ainda.")

    col1, col2 = st.columns(2)
    if col1.button("✨ Ensinar contexto e pensamento agora", key=f"jogo_ensinar_{chave_sufixo}", width="stretch"):
        st.session_state.criadora_txt = texto
        st.session_state.criadora_reac = reacao
        st.session_state.criadora_dominio = dominio_c
        if candidato:
            b = candidato["bloco"]
            st.session_state.criadora_pensamento = b["entrada"].get("pensamento_interno", "")
            st.session_state.criadora_contexto = b["entrada"].get("contexto", "")
            st.session_state.criadora_saida_sugerida_texto = b["saidas"][0]["textos"][0]
            st.session_state.criadora_saida_sugerida_reacao = b["saidas"][0].get("reacao", "")
            st.session_state.criadora_saida_sugerida_contexto = b["saidas"][0].get("contexto", "")
            st.session_state.criadora_step = "confirmar_saida"
            # Placeholder (ex.: "0.0") nunca aparece como se fosse reação de
            # verdade -- só mostra o texto se for o caso.
            reacao_sugerida_exibir = "" if _eh_placeholder_im(st.session_state.criadora_saida_sugerida_reacao) else st.session_state.criadora_saida_sugerida_reacao
            saida_sugerida_exibir = f'{st.session_state.criadora_saida_sugerida_texto} {reacao_sugerida_exibir}'.strip()
            ai_msg = (
                f'Owww, isso me lembra de algo que já sei. '
                f'Baseado nisso, eu diria: "{saida_sugerida_exibir}". '
                f'Posso usar essa mesma resposta, ou você quer outra?'
            )
        else:
            st.session_state.criadora_step = "escolher_campo_criacao"
            ai_msg = (
                gerar_saudacao_desconhecido() + formatar_palpite_cru_para_chat(texto, reacao, dominio_c)
                + f"\n\nQual campo você quer preencher? Diga o nome:\n{_lista_campos_criacao()}\n\n(ou 'finalizar' quando já tiver o suficiente)"
            )
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.rerun()
    if col2.button("➡️ Pular pra próxima", key=f"jogo_pular_{chave_sufixo}", width="stretch"):
        novo_sorteio = sortear_e_registrar_frase(dominio_c)
        if novo_sorteio is None:
            ai_msg = "🎲 Acabaram as frases do dicionário nesse universo -- todas já viraram opinião ou bloco. Quer que eu repita alguma pendente?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        else:
            ai_msg = "🎲 Beleza, próxima frase:"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg, "mostrar_jogo_frase": {"dominio": dominio_c, "sorteio": novo_sorteio}})
        st.rerun()


def sortear_e_registrar_paragrafo_lux(dominio_c: str) -> Optional[dict]:
    """Livro de Lux (2026-09-13, pedido direto da Thaís): puxa um parágrafo
    ainda não usado do poço de sementes (Adam_Lovely_livro_lux.json), roda a
    mesma checagem de Báskara de sempre, e registra como opinião pendente --
    igual jogo_frase, mas o texto é um parágrafo inteiro e chega SEM reação
    nenhuma (reação/contexto/pensamento só entram depois, na hora de
    completar -- diferente da frase curta do jogo, que já vem com emoção)."""
    usados = _frases_ja_usadas(dominio_c)
    candidatos = [p for p in carregar_paragrafos_lux() if normalize(p) not in usados]
    if not candidatos:
        return None
    texto = random.choice(candidatos)
    candidato = melhor_candidato_fraco(texto, dominio_c, "")
    opiniao = registrar_opiniao_pendente(texto, "", dominio_c, candidato=candidato, origem="livro_lux")
    return {"texto": texto, "candidato": candidato, "opiniao_id": opiniao["id"]}


def renderizar_livro_lux_no_chat(dominio_c: str, sorteio: dict, chave_sufixo: str) -> None:
    """Mesmo espírito de renderizar_jogo_frase_no_chat, mas pro parágrafo:
    completar agora (entra no mesmo fluxo do modo criadora -- reação,
    contexto, pensamento e saída, tudo de uma vez, já que aqui nada disso
    vem pronto) ou pular pro próximo parágrafo."""
    texto = sorteio["texto"]
    candidato = sorteio.get("candidato")
    if candidato:
        st.caption(f'📖 "{texto}" -- isso me lembra algo que já sei.')
    else:
        st.caption(f'📖 "{texto}" -- nunca vi nada parecido com isso ainda.')

    col1, col2 = st.columns(2)
    if col1.button("✨ Completar agora", key=f"lux_completar_{chave_sufixo}", width="stretch"):
        # Parágrafo do Livro de Lux nunca vem com reação pronta (diferente da
        # frase curta do jogo_frase) -- então SEMPRE pergunta ela primeiro,
        # antes de entrar no resto do fluxo de sempre do modo criadora.
        st.session_state.criadora_txt = texto
        st.session_state.criadora_reac = ""
        st.session_state.criadora_dominio = dominio_c
        st.session_state.criadora_lux_tem_candidato = bool(candidato)
        st.session_state.criadora_eh_lux = True
        garantir_capitulo_lux_ativo()
        if candidato:
            b = candidato["bloco"]
            st.session_state.criadora_pensamento = b["entrada"].get("pensamento_interno", "")
            st.session_state.criadora_contexto = b["entrada"].get("contexto", "")
            st.session_state.criadora_saida_sugerida_texto = b["saidas"][0]["textos"][0]
            st.session_state.criadora_saida_sugerida_reacao = b["saidas"][0].get("reacao", "")
            st.session_state.criadora_saida_sugerida_contexto = b["saidas"][0].get("contexto", "")
        st.session_state.criadora_step = "coletar_reacao_entrada"
        ai_msg = "📖 Beleza! Qual reação você sentiu com esse parágrafo? (tipo 🙂, :), ou *Sorriso*)"
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.rerun()
    if col2.button("➡️ Pular pro próximo", key=f"lux_pular_{chave_sufixo}", width="stretch"):
        novo_sorteio = sortear_e_registrar_paragrafo_lux(dominio_c)
        if novo_sorteio is None:
            ai_msg = "📖 Acabaram os parágrafos do Livro nesse universo -- todos já viraram opinião ou bloco. Quer escrever um novo?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        else:
            ai_msg = "📖 Beleza, próximo parágrafo:"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg, "mostrar_livro_lux": {"dominio": dominio_c, "sorteio": novo_sorteio}})
        st.rerun()


def modo_espelho_corpus(dominio_c: str) -> List[dict]:
    """Modo espelho de análise: compara TODOS os pares de blocos já
    existentes nesse universo, eixo por eixo (Camada 6 -- comparar_eixo/
    espelhar, igualdade exata de tokens, nunca aproximação). Raiz
    (pensamento) compartilhada entre dois blocos diferentes é o sinal mais
    forte de "sinapse" -- mesma raiz, ramos diferentes (Camada 3)."""
    from insepa_espelhamento import espelhar
    memoria_c = st.session_state.get("memoria", {})
    blocos = memoria_c.get("IM", {}).get(dominio_c, {}).get("blocos", [])
    conexoes = []
    for i in range(len(blocos)):
        for j in range(i + 1, len(blocos)):
            a, b = blocos[i], blocos[j]
            entrada_a = {
                "texto": a["entrada"].get("texto", ""),
                "emocao": a["entrada"].get("reacao", ""),
                "contexto": a["entrada"].get("contexto", ""),
                "pensamento": a["entrada"].get("pensamento_interno", ""),
            }
            diag = espelhar(entrada_a, b["entrada"])
            conexoes.append({"bloco_a": a["bloco_id"], "bloco_b": b["bloco_id"], "diag": diag})
    return conexoes


def formatar_espelho_corpus_para_chat(dominio_c: str) -> str:
    conexoes = modo_espelho_corpus(dominio_c)
    if not conexoes:
        return "🪞 Preciso de pelo menos 2 blocos nesse universo pra ter algo pra comparar."
    rotulo = {"bate": "✅", "diverge": "⚠️", "desconhecido": "❔", "vazio": "—"}
    linhas = ["🪞 **Modo espelho -- como meus blocos se conectam (Camada 6, sempre exato, nunca aproximação):**\n"]
    sinapses = 0
    for c in conexoes:
        d = c["diag"]
        if d["raiz_compartilhada"]:
            sinapses += 1
        sinapse_txt = " 🧠 *raiz compartilhada!*" if d["raiz_compartilhada"] else ""
        linhas.append(
            f'Bloco #{c["bloco_a"]} ↔ #{c["bloco_b"]}: '
            f'texto {rotulo[d["texto"]]} · emoção {rotulo[d["emocao"]]} · '
            f'contexto {rotulo[d["contexto"]]} · pensamento {rotulo[d["pensamento"]]}{sinapse_txt}'
        )
    linhas.append(f"\n{sinapses} de {len(conexoes)} par(es) compartilham raiz -- essas são as sinapses de verdade já ligadas.")
    return "\n".join(linhas)


def renderizar_tela_no_chat(dominio_c: str, memoria_c: dict, inconsciente_c: dict, visao: str) -> None:
    """Camada 7 (Tela) direto no chat -- mesma função já usada no Caderno de
    Ferramentas, só que lendo o estado real da conversa em vez do JSON do
    disco, então reflete até o que ainda não foi salvo."""
    from insepa_marcadores import tokenizar as _insepa_tokenizar
    from insepa_tela import visao_universo_inconsciente_html, visao_universo_memoria_html

    nome_universo = memoria_c.get("IM", {}).get(dominio_c, {}).get("nome", f"IM_{dominio_c}")
    blocos_c = memoria_c.get("IM", {}).get(dominio_c, {}).get("blocos", [])
    blocos_inco_c = inconsciente_c.get("INCO", {}).get(dominio_c, {}).get("Blocos", [])

    if visao == "inconsciente":
        st.caption("🌫️ O Inconsciente -- caos, sem classificação, é assim que deve ser.")
        st.markdown(visao_universo_inconsciente_html(nome_universo, dominio_c, blocos_inco_c), unsafe_allow_html=True)
    else:
        st.caption("🗂️ A Memória -- cada campo (texto/fala/contexto/pensamento, entrada e saída) classificado.")
        st.markdown(visao_universo_memoria_html(nome_universo, dominio_c, blocos_c, _insepa_tokenizar, dividir_travessao, blocos_inco_c), unsafe_allow_html=True)


def renderizar_cerebro_no_chat(dominio_c: str, chave_sufixo: str) -> None:
    """Mostra, dentro do próprio chat, o cérebro real do Adam: os neurônios
    de verdade (checkpoint treinado, Camada 8) e a curva de aprendizado real
    acumulada -- nunca dado de brinquedo. `chave_sufixo` só existe pra dar
    key único aos gráficos quando essa mensagem é redesenhada de novo a cada
    rerun (senão o Streamlit reclama de chave duplicada)."""
    try:
        import insepa_neuronios as _neur
    except ImportError:
        st.warning("⚠️ Falta a biblioteca `plotly` -- instale com `pip install plotly` pra eu conseguir mostrar isso.")
        return

    st.caption("🧠 Isto é o meu cérebro de verdade -- os mesmos pesos treinados que uso pra te responder.")
    if _neur.checkpoint_disponivel(dominio_c):
        campo_escolhido = st.radio(
            "Campo", list(_neur.CAMPOS_EMBEDDING.keys()),
            horizontal=True, key=f"cerebro_campo_{chave_sufixo}",
        )
        fig_neu, erro_neu = _neur.grafico_3d_neuronios(dominio_c, campo_escolhido)
        if erro_neu:
            st.info(erro_neu)
        else:
            st.plotly_chart(fig_neu, width="stretch", key=f"cerebro_fig_{chave_sufixo}_{campo_escolhido}")
    else:
        st.info("Ainda não tenho um checkpoint treinado pra mostrar.")

    historico_cerebro = _neur.carregar_historico_treino()
    fig_curva = _neur.grafico_3d_curva_aprendizado(historico_cerebro, dominio_c)
    if fig_curva is not None:
        st.caption("📈 E esta é a minha curva de aprendizado real, uma linha por conversa que me fez retreinar.")
        st.plotly_chart(fig_curva, width="stretch", key=f"cerebro_curva_{chave_sufixo}")


def formatar_achados_insepa_para_chat(achados) -> str:
    """Mesma formatação que insepa_agent.py já usa no terminal, só que como
    texto de chat -- não reinterpreta nem resume os achados, só reapresenta
    exatamente o que o agente reportou."""
    rotulo = {"erro": "❌", "aviso": "⚠️", "info": "✅"}
    ordem = {"erro": 0, "aviso": 1, "info": 2}
    linhas = []
    for a in sorted(achados, key=lambda x: ordem[x.severidade]):
        local = f" `{a.local}`" if a.local else ""
        linhas.append(f"{rotulo[a.severidade]} **({a.fonte})**{local} {a.mensagem}")
    erros = sum(1 for a in achados if a.severidade == "erro")
    avisos = sum(1 for a in achados if a.severidade == "aviso")
    ok = len(achados) - erros - avisos
    resumo = f"\n\n**Resumo:** {erros} erro(s), {avisos} aviso(s), {ok} ok."
    return "\n\n".join(linhas) + resumo


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
    """Parseia o prompt pra separar texto e reação. A reação pode ser:
    - um emoji/emoticon isolado (curto ou sem letra nenhuma, ex.: "^^", "😊", ":D");
    - uma ação entre asteriscos, palavra única (var, ex.: "*Sorriso*") ou frase
      inteira (multivar, ex.: "*Exibo um sorriso alegre*") -- nunca só a última
      palavra dela, a ação inteira é a reação;
    - a combinação dos dois (emoji + ação), ex.: "^^ *Sorrio com os olhos*"."""
    prompt = prompt.strip()

    # Ação entre asteriscos no final da mensagem -- pega a unidade inteira,
    # nunca só a última palavra (senão quebra multivariáveis de frase).
    m = _re.search(r'(\*[^*]+\*)\s*$', prompt)
    if m:
        acao = m.group(1)
        antes = prompt[:m.start()].rstrip()
        palavras_antes = antes.split()
        if palavras_antes:
            possivel_emoji = palavras_antes[-1]
            # Só combina se o token antes da ação parecer emoji/emoticon de
            # verdade (curto ou sem nenhuma letra) -- nunca uma palavra real
            # com pontuação (ex.: "Adam." não deve ser confundido com reação).
            if len(possivel_emoji) <= 3 or not any(c.isalpha() for c in possivel_emoji):
                txt = ' '.join(palavras_antes[:-1])
                return txt, f"{possivel_emoji} {acao}"
        return antes, acao

    # Sem ação entre asteriscos: reação isolada é a última palavra, se curta
    # ou sem nenhuma letra (emoji/emoticon) -- igual ao critério que o ramo
    # do asterisco já usa (any(c.isalpha())), não `str.isalnum()`. Antes
    # disso, QUALQUER frase terminando em pontuação colada na última
    # palavra ("vai?", "hoje!", até o próprio "Como está?" de um bloco
    # real) virava reação por engano, porque isalnum() já retorna False só
    # por causa do "?"/"!" -- sem checar se a palavra tinha letra nenhuma.
    words = prompt.split()
    if not words:
        return prompt, ""
    last = words[-1]
    if len(last) <= 3 or not any(c.isalpha() for c in last):
        txt = ' '.join(words[:-1])
        reac = last
        return txt, reac
    else:
        return prompt, ""


def _eh_placeholder_im(valor: str) -> bool:
    """Reconhece o placeholder padrão de qualquer universo (IM + '.0', ex.:
    '0.0', '1.0', '2.0'...) -- 2026-09-13, pedido direto da Thaís: um
    bloco com campo ainda não preenchido pode virar candidato num sorteio
    (jogo_frase / Livro de Lux) igual qualquer outro -- cosseno e Báskara só
    olham texto e emoção de verdade, nunca o placeholder. Mas na hora de
    REAPROVEITAR a saída de um candidato assim, o placeholder nunca pode
    ser copiado pro bloco novo como se fosse conteúdo real."""
    return bool(_re.fullmatch(r"\d+\.0", (valor or "").strip()))


def _proximo_passo_apos_saida(prefixo_ok: str) -> tuple[str, str]:
    """Decide o próximo passo do modo criadora depois de capturar o texto
    (e possível reação) de uma saída -- 2026-09-13, bug real achado pela
    Thaís: um bloco novo nunca deveria poder nascer com a saída sem
    reação, mas isso passava em silêncio sempre que a reação vinha vazia
    (ex.: "Como está?" -> "Estou bem..." sem nenhum emoji no final).
    Lê/espera `criadora_saida_reac_pendente` já em session_state; `prefixo_ok`
    é só a interjeição de sempre ('Perfeito!'/'Entendido!'/'Combinado!'),
    pra manter o tom de cada gatilho que chama isso."""
    if st.session_state.get("criadora_saida_reac_pendente"):
        return "coletar_saida_contexto", f"{prefixo_ok} Que conclusão devo tirar disso?"
    return "coletar_saida_reacao_faltante", f"{prefixo_ok} Que sentimento essa resposta provoca? (tipo 🙂, :), ou *sorriso*)"


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

    # Por enquanto (corpus pequeno, sendo construído bloco a bloco à mão),
    # retreina sozinho sempre que a memória mudou desde o último treino --
    # nunca fica desatualizado por falta de alguém apertar um botão de treino
    # que não existe mais. Comparação por mtime: se Adam_Lovely_memory.json foi
    # salvo DEPOIS do checkpoint, o checkpoint não reflete o que já foi ensinado.
    memoria_mudou = (
        os.path.exists(ARQUIVO_MEMORIA)
        and os.path.exists(ckpt)
        and os.path.getmtime(ARQUIVO_MEMORIA) > os.path.getmtime(ckpt)
    )
    if not os.path.exists(ckpt):
        st.info("🌱 Ainda não treinei esse universo -- treinando agora antes de conversar...")
        train(memoria, dominio)
    elif memoria_mudou:
        st.info("🌱 Aprendi algo novo desde o último treino -- retreinando antes de continuar...")
        train(memoria, dominio)

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
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng,
        out_vocab_floats=list(idx_to_txt.keys()),
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
    for _i_msg, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metrics"):
                render_match_metrics(message["metrics"])
            if message.get("mostrar_cerebro"):
                renderizar_cerebro_no_chat(message["mostrar_cerebro"]["dominio"], f"hist_{_i_msg}")
            if message.get("mostrar_tela"):
                renderizar_tela_no_chat(
                    message["mostrar_tela"]["dominio"], memoria, st.session_state.inconsciente,
                    message["mostrar_tela"]["visao"],
                )
            if message.get("mostrar_menu"):
                renderizar_menu_ferramentas_no_chat(message["mostrar_menu"]["dominio"], f"hist_{_i_msg}")
            if message.get("mostrar_escolha_ajuda"):
                renderizar_escolha_ajuda_no_chat(message["mostrar_escolha_ajuda"]["dominio"], f"hist_{_i_msg}")
            if message.get("mostrar_jogo_frase"):
                renderizar_jogo_frase_no_chat(
                    message["mostrar_jogo_frase"]["dominio"], message["mostrar_jogo_frase"]["sorteio"], f"hist_{_i_msg}",
                )
            if message.get("mostrar_livro_lux"):
                renderizar_livro_lux_no_chat(
                    message["mostrar_livro_lux"]["dominio"], message["mostrar_livro_lux"]["sorteio"], f"hist_{_i_msg}",
                )
            if message.get("reforco_autonomo"):
                renderizar_reforco_autonomo_no_chat(
                    message["reforco_autonomo"]["bloco_id"], message["reforco_autonomo"]["resposta"], f"hist_{_i_msg}",
                )

    # Mostrar áudio se existir
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3')

    def featurize(tokens: List[str], max_len: int, vocab: dict, val_to_idx: dict, max_ng: int, dominio_int: int):
        # Palavras de verdade (via ALNULU), nunca marcador -- mesma lógica do
        # treino (InsepaFieldDataset/test_model), pra bater com o que o modelo
        # aprendeu. `tokens` já vem em texto puro (Token(txt)/Token(reac)...).
        # Trunca ANTES de tudo -- `lista += [0] * negativo` não trunca nada em
        # Python, só vira no-op silencioso. Sem isso, entrada com mais
        # palavras do que o maior exemplo já visto no treino quebrava o
        # forward() com erro de shape (achado testando o palpite cru da
        # rede -- ficava escondido porque o resultado nunca era mostrado).
        tokens = tokens[:max_len]
        ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
        ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist][:max_len * max_ng]
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

        # Se o Adam acabou de testar um palpite ("é isso que você quis dizer?"),
        # a próxima mensagem é sim/não (texto ou reação) -- vira métrica de
        # qualidade do bloco.
        if st.session_state.get("feedback_bloco_id"):
            # Tokeniza em vez de cortar pontuação -- o INSEPA nunca descarta nada da
            # entrada, então "Sim!" continua "Sim" + "!" como tokens, só checamos
            # se "sim" está ENTRE eles, sem alterar o texto original de ninguém.
            tokens_resposta = {t.lower() for t in Token(prompt)}
            positivo = bool(tokens_resposta & {"sim", "s", "yes", "y"}) or "❤️‍🔥" in prompt
            negativo = bool(tokens_resposta & {"não", "nao", "n", "no"}) or "💔" in prompt
            fb_dominio = st.session_state.get("feedback_dominio", dominio)
            fb_bloco_id = st.session_state.pop("feedback_bloco_id")
            st.session_state.pop("feedback_dominio", None)
            if positivo or negativo:
                blocos_fb = memoria.get("IM", {}).get(fb_dominio, {}).get("blocos", [])
                bloco_fb = next((b for b in blocos_fb if str(b.get("bloco_id")) == str(fb_bloco_id)), None)
                if bloco_fb is not None:
                    metricas = bloco_fb.setdefault("metricas", {"gostei": 0, "nao_gostei": 0})
                    metricas["gostei" if positivo else "nao_gostei"] = metricas.get("gostei" if positivo else "nao_gostei", 0) + 1
                    st.session_state.memoria = memoria
                    salvar_json(ARQUIVO_MEMORIA, memoria)
                ai_msg = "Entendi! 😊" if positivo else "Entendi, valeu por avisar! 🌱"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            # Resposta ambígua: não trava esperando sim/não -- só segue o fluxo
            # normal como se fosse uma mensagem nova (evita o mesmo problema que
            # já corrigimos na senha do Cérbero).

        cmd = prompt.lower().strip()
        s = prompt.strip()
        txt, reac = parse_text_reaction(s)

        # "Adam me mostre seu cérebro" -- é o cérebro DELE, então antes de
        # exibir os neurônios/curva reais (Camada 8) ele pede a senha, do
        # mesmo jeito que qualquer outra ação de privilégio total no chat.
        # Checado ANTES de qualquer outro comando, senão a senha digitada na
        # próxima mensagem seria interpretada como entrada de conversa normal.
        if st.session_state.get("cerebro_step") == "aguardar_senha":
            st.session_state.pop("cerebro_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_mostrar_cerebro(dominio)
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou mostrar."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return  # nunca deixar o resto de infer() rodar em cima dessa mesma mensagem

        cerebro_cmd = cmd.replace("é", "e").replace("ê", "e")
        pede_para_ver_cerebro = (
            "cerebro" in cerebro_cmd
            and any(gatilho in cerebro_cmd for gatilho in ("mostr", "ver ", "exib"))
        )
        if pede_para_ver_cerebro:
            # Em vez de já pedir a senha direto, mostra o menu de atalhos --
            # "mostre o cérebro" virou a porta de entrada pros botões de
            # todas as ferramentas, não só do cérebro (pedido dela: encurtar
            # o caminho em vez de decorar frase por frase).
            ai_msg = "🗝️ Encurtando o caminho -- escolhe aí:"
            nova_msg = {"role": "assistant", "content": ai_msg, "mostrar_menu": {"dominio": dominio}}
            st.session_state.messages.append(nova_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
                renderizar_menu_ferramentas_no_chat(dominio, f"live_{len(st.session_state.messages)}")
            st.rerun()
            return  # idem -- senão a própria frase-gatilho seria tratada como entrada normal a seguir

        # "Adam rode o agente INSEPA" -- mesmo padrão de senha do cérebro,
        # porque rodar o agente executa a suíte de testes de verdade
        # (subprocess) e mexe com o sistema -- é ação de privilégio total,
        # nunca automática pra qualquer um.
        if st.session_state.get("agente_step") == "aguardar_senha":
            st.session_state.pop("agente_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_rodar_agente()
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou rodar."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return

        agente_cmd = cerebro_cmd
        pede_para_rodar_agente = (
            ("agente" in agente_cmd or "diagnostico" in agente_cmd)
            and any(gatilho in agente_cmd for gatilho in ("rod", "verific", "check", "diagnostic"))
        )
        if pede_para_rodar_agente:
            if sessao_admin_liberada():
                executar_rodar_agente()
                st.rerun()
                return
            st.session_state.agente_step = "aguardar_senha"
            ai_msg = "🔧 Rodar meu agente de manutenção mexe com o sistema de verdade -- preciso confirmar que é você. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # "Adam mostre a tela/memória/inconsciente" -- Camada 7. Protegido
        # por senha (2026-09-13, por instrução direta da Thaís): mostrar o
        # Inconsciente e a Memória é mostrar como o cérebro dele se
        # organiza por dentro -- o modelo dela, que ela não quer que
        # ninguém copie. Nunca mais visível sem senha, mesmo padrão de
        # cérebro/agente.
        tela_cmd = cerebro_cmd
        if st.session_state.get("tela_step") == "aguardar_senha":
            st.session_state.pop("tela_step", None)
            visao_pedida = st.session_state.pop("tela_visao_pendente", "memoria")
            if prompt.strip() == SENHA_ADMIN:
                executar_mostrar_tela(dominio, memoria, visao_pedida)
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou mostrar."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return

        pede_para_ver_tela = (
            ("tela" in tela_cmd or "inconsciente" in tela_cmd or "memoria" in tela_cmd)
            and any(gatilho in tela_cmd for gatilho in ("mostr", "ver ", "exib"))
        )
        if pede_para_ver_tela:
            visao_pedida = "inconsciente" if "inconsciente" in tela_cmd else "memoria"
            if sessao_admin_liberada():
                executar_mostrar_tela(dominio, memoria, visao_pedida)
                st.rerun()
                return
            st.session_state.tela_step = "aguardar_senha"
            st.session_state.tela_visao_pendente = visao_pedida
            ai_msg = "🖼️ Isso mostra como eu me organizo por dentro -- só a criadora pode ver isso. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # "ajuda admin" / "ajuda adm" -- a listagem técnica completa, só pra
        # criadora (2026-09-13, pedido direto da Thaís: "o prompt ajuda
        # precisa de duas saídas, ajuda pro usuário e ajuda pro ADM, e no ADM
        # mantém a senha"). Checado ANTES do "ajuda" público, senão a versão
        # mínima capturava tudo que contém "ajuda"/"comando".
        if st.session_state.get("ajuda_admin_step") == "aguardar_senha":
            st.session_state.pop("ajuda_admin_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_mostrar_painel_admin(dominio)
                st.rerun()
                return
            ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou mostrar."
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        palavras_tela_cmd = tela_cmd.split()
        pede_ajuda_admin = ("ajuda" in tela_cmd or "comando" in tela_cmd) and (
            "admin" in tela_cmd or "adm" in palavras_tela_cmd or "completo" in tela_cmd or "completa" in tela_cmd
        )
        if pede_ajuda_admin:
            if sessao_admin_liberada():
                executar_mostrar_painel_admin(dominio)
                st.rerun()
                return
            st.session_state.ajuda_admin_step = "aguardar_senha"
            ai_msg = "🗝️ Essa é a lista técnica completa -- só a criadora pode ver isso. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # "Adam quais comandos você tem?" / "ajuda" -- ajuda pública mínima
        # (nunca a lista técnica -- essa só sai por "ajuda admin" com senha).
        pede_ajuda = cmd in {"ajuda", "comandos", "help"} or (
            "comando" in tela_cmd and any(g in tela_cmd for g in ("quais", "que", "voce tem"))
        )
        # "Adam puxa uma frase" -- o jogo de criar bloco de forma divertida.
        # Sorteia texto+reação do dicionário, checa correspondência real via
        # Báskara contra o que já existe, e vira opinião pendente na hora --
        # só ganha contexto/pensamento (e status de dado concreto) se a
        # criadora ensinar, nunca antes.
        pede_para_jogar = (
            "frase" in tela_cmd and any(g in tela_cmd for g in ("puxa", "sorte", "aleatori", "surpreend", "jog"))
        )
        if pede_para_jogar:
            sorteio = sortear_e_registrar_frase(dominio)
            if sorteio is None:
                ai_msg = "🎲 O dicionário de frases já foi todo usado nesse universo -- todas viraram opinião ou bloco."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            else:
                ai_msg = "🎲 Vamos brincar! Puxei uma frase:"
                nova_msg = {"role": "assistant", "content": ai_msg, "mostrar_jogo_frase": {"dominio": dominio, "sorteio": sorteio}}
                st.session_state.messages.append(nova_msg)
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                    renderizar_jogo_frase_no_chat(dominio, sorteio, f"live_{len(st.session_state.messages)}")
            st.rerun()
            return

        # Livro de Lux (2026-09-13, pedido direto da Thaís): em vez de só
        # puxar frase curta já com emoção, agora dá pra escrever/puxar
        # PARÁGRAFOS inteiros (contos) que podem virar bloco depois --
        # reação, contexto e pensamento ficam de fora até a hora de
        # completar (diferente da frase do jogo_frase, que já vem pronta).
        if st.session_state.get("lux_step") == "aguardar_paragrafo_escrito":
            st.session_state.pop("lux_step", None)
            novo_paragrafo = prompt.strip()
            if novo_paragrafo.lower() in {"cancelar", "cancela"}:
                ai_msg = "📖 Cancelado -- nada foi escrito."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
                return
            adicionar_paragrafo_lux(novo_paragrafo)
            candidato = melhor_candidato_fraco(novo_paragrafo, dominio, "")
            opiniao = registrar_opiniao_pendente(novo_paragrafo, "", dominio, candidato=candidato, origem="livro_lux")
            sorteio = {"texto": novo_paragrafo, "candidato": candidato, "opiniao_id": opiniao["id"]}
            ai_msg = "📖 Adicionado ao Livro de Lux! Quer completar agora (reação, contexto, pensamento e saída), ou deixar pendente pra depois?"
            nova_msg = {"role": "assistant", "content": ai_msg, "mostrar_livro_lux": {"dominio": dominio, "sorteio": sorteio}}
            st.session_state.messages.append(nova_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
                renderizar_livro_lux_no_chat(dominio, sorteio, f"live_{len(st.session_state.messages)}")
            st.rerun()
            return

        # Livro de Lux -- hierarquia de 3 níveis (2026-09-13, correção direta
        # da Thaís: são TRÊS prompts diferentes, não sinônimos): Livro >
        # História (arco) > Capítulo > Parágrafo (bloco). "Encerrar
        # capítulo" fecha só o capítulo -- a história continua, o próximo
        # parágrafo abre um capítulo novo na MESMA história. "Terminar
        # história" fecha o arco inteiro (capítulo incluso) -- representa o
        # fim do arco. "Livro terminado" é quando TODOS os arcos já estão
        # concluídos -- fecha qualquer coisa ainda aberta e mostra o resumo
        # completo do universo.
        cerebro_cmd_sem_acento_i = cerebro_cmd.replace("í", "i")
        pede_encerrar_capitulo_lux = (
            "capitulo" in cerebro_cmd_sem_acento_i
            and any(g in tela_cmd for g in ("encerr", "termin", "fech", "acabou", "fim"))
        )
        if pede_encerrar_capitulo_lux:
            capitulo_id_fechado = st.session_state.pop("lux_capitulo_id", None)
            if capitulo_id_fechado:
                total = sum(
                    1 for b in memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
                    if b.get("meta", {}).get("capitulo_id") == capitulo_id_fechado
                )
                ai_msg = f"📖 Capítulo encerrado! {total} parágrafo(s) nele -- a história continua, o próximo parágrafo abre um capítulo novo nela."
            else:
                ai_msg = "📖 Não tinha nenhum capítulo em andamento, mas tudo bem -- já fica registrado."
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        pede_terminar_historia_lux = (
            "historia" in cerebro_cmd
            and any(g in tela_cmd for g in ("termin", "encerr", "acabou", "fim", "final"))
        )
        if pede_terminar_historia_lux:
            st.session_state.pop("lux_capitulo_id", None)
            historia_id_fechada = st.session_state.pop("lux_historia_id", None)
            if historia_id_fechada:
                blocos_historia = [
                    b for b in memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
                    if b.get("meta", {}).get("historia_id") == historia_id_fechada
                ]
                n_capitulos = len({b["meta"].get("capitulo_id") for b in blocos_historia})
                ai_msg = f"📖 História (arco) encerrada! {len(blocos_historia)} parágrafo(s) em {n_capitulos} capítulo(s). Fim do arco."
            else:
                ai_msg = "📖 Não tinha nenhuma história em andamento, mas tudo bem -- já fica registrado."
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        pede_livro_terminado_lux = (
            "livro" in tela_cmd
            and any(g in tela_cmd for g in ("termin", "conclu", "complet", "acabou", "fim"))
        )
        if pede_livro_terminado_lux:
            st.session_state.pop("lux_capitulo_id", None)
            st.session_state.pop("lux_historia_id", None)
            blocos_lux = [
                b for b in memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
                if b.get("meta", {}).get("historia_id")
            ]
            n_arcos = len({b["meta"]["historia_id"] for b in blocos_lux})
            n_capitulos = len({b["meta"].get("capitulo_id") for b in blocos_lux})
            ai_msg = (
                f"📖 Livro terminado! No universo {dominio}: {n_arcos} arco(s), {n_capitulos} capítulo(s), "
                f"{len(blocos_lux)} parágrafo(s) ao todo. Todos os arcos concluídos. 💚"
            )
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        pede_para_escrever_paragrafo_lux = (
            ("paragrafo" in tela_cmd or "paragrafo" in cmd.replace("á", "a") or "conto" in tela_cmd or "lux" in tela_cmd)
            and any(g in tela_cmd for g in ("escrev", "novo", "nova", "adiciona", "criar"))
        )
        if pede_para_escrever_paragrafo_lux:
            aviso = iniciar_e_avisar_lux()
            st.session_state.lux_step = "aguardar_paragrafo_escrito"
            ai_msg = f"{aviso}Me manda SÓ UM parágrafo (não a história inteira de uma vez -- cada parágrafo tem sua própria reação/contexto/pensamento; o conjunto deles é que vira um capítulo). Reação/contexto/pensamento entram depois."
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        pede_para_puxar_paragrafo_lux = (
            ("paragrafo" in tela_cmd or "conto" in tela_cmd or "lux" in tela_cmd)
            and any(g in tela_cmd for g in ("puxa", "sorte", "aleatori", "surpreend"))
        )
        if pede_para_puxar_paragrafo_lux:
            sorteio = sortear_e_registrar_paragrafo_lux(dominio)
            if sorteio is None:
                ai_msg = "📖 O Livro de Lux já foi todo usado nesse universo -- todos os parágrafos viraram opinião ou bloco. Quer escrever um novo?"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            else:
                ai_msg = "📖 Abrindo o Livro de Lux, puxei um parágrafo:"
                nova_msg = {"role": "assistant", "content": ai_msg, "mostrar_livro_lux": {"dominio": dominio, "sorteio": sorteio}}
                st.session_state.messages.append(nova_msg)
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                    renderizar_livro_lux_no_chat(dominio, sorteio, f"live_{len(st.session_state.messages)}")
            st.rerun()
            return

        # "rodar autônomo" -- testa o motor de autonomia (insepa_autonomia.py)
        # sob demanda, com o veredito completo em vez da legendinha discreta
        # de sempre. Mesmo comportamento real (registra opinião se achar
        # correspondência), só que ela escolhe o texto e vê a matemática.
        if st.session_state.get("autonomo_step") == "aguardar_texto_teste":
            st.session_state.pop("autonomo_step", None)
            txt_teste, reac_teste = parse_text_reaction(prompt.strip())
            txt_teste = txt_teste or prompt.strip()
            import insepa_autonomia as _ia_teste
            blocos_teste = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
            reacoes_teste = {b["bloco_id"]: list(_reacoes_aceitas_do_bloco(b, dominio)) for b in blocos_teste}
            # Contexto/pensamento do candidato ficam vazios de propósito
            # (2026-09-13, bug real achado pela Thaís: "Como vai? ^^" não
            # batia com o bloco 3 mesmo sendo Multivar registrada de "Como
            # está?"). Antes, um "perfil padrão do universo" fixo (sempre os
            # valores do bloco 1) era comparado contra QUALQUER bloco que o
            # cosseno achasse -- funcionava com 1 bloco só, mas quebrava
            # assim que existia mais de um tema no corpus: bloco 3 divergia
            # do perfil do bloco 1 e o Báskara recusava, mesmo com o texto
            # batendo exato via Multivar. A entrada crua é sempre só
            # texto+reação (Camada 5) -- deixando contexto/pensamento vazios
            # aqui, comparar_eixo devolve "vazio" (não conta pra nenhum
            # lado), e só texto+emoção decidem de verdade -- mesmo
            # comportamento do caminho de ensino ao vivo (encontrar_
            # correspondencia_baskara), agora também no teste manual.
            veredito_teste = _ia_teste.avaliar_entrada_autonoma(
                txt_teste, reac_teste, blocos_teste, reacoes_teste,
            )
            if veredito_teste["action"] == "correspondencia_encontrada":
                bloco_teste = next((b for b in blocos_teste if b["bloco_id"] == veredito_teste["bloco_id"]), None)
                registrar_opiniao_pendente(
                    txt_teste, reac_teste, dominio,
                    candidato={"bloco": bloco_teste, "score": veredito_teste["discriminante"], "origem_im": dominio, "emprestado": False},
                    origem="motor_autonomia_teste",
                )
            ai_msg = formatar_veredito_autonomo_para_chat(veredito_teste, txt_teste, reac_teste, blocos_teste)
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # Protegido por senha (2026-09-13): o veredito mostra bloco/texto/
        # discriminante reais do corpus, mesma exposição de arquitetura.
        if st.session_state.get("autonomo_step") == "aguardar_senha":
            st.session_state.pop("autonomo_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_iniciar_teste_autonomo()
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou rodar."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return

        pede_para_rodar_autonomo = (
            ("autonomo" in tela_cmd or "autônomo" in cmd) and any(g in tela_cmd for g in ("rod", "test", "verific"))
        )
        if pede_para_rodar_autonomo:
            if sessao_admin_liberada():
                executar_iniciar_teste_autonomo()
                st.rerun()
                return
            st.session_state.autonomo_step = "aguardar_senha"
            ai_msg = "🤖 Isso testa o motor contra o corpus real e mostra o resultado -- só a criadora pode ver. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # "Adam modo espelho" -- Camada 6 aplicada ao corpus inteiro. Também
        # protegido por senha (2026-09-13): mostra texto/contexto/pensamento
        # reais dos blocos par a par, a mesma exposição de arquitetura que a
        # Tela tinha.
        if st.session_state.get("espelho_step") == "aguardar_senha":
            st.session_state.pop("espelho_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_mostrar_espelho(dominio)
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou mostrar."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return

        pede_espelho = "espelho" in tela_cmd or "sinapse" in tela_cmd
        if pede_espelho:
            if sessao_admin_liberada():
                executar_mostrar_espelho(dominio)
                st.rerun()
                return
            st.session_state.espelho_step = "aguardar_senha"
            ai_msg = "🪞 Isso mostra como meus blocos se conectam por dentro -- só a criadora pode ver isso. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # "Adam editar bloco" -- edita o conteúdo de um bloco já existente e
        # renumera os marcadores do universo inteiro sem furo (2026-09-13,
        # pedido direto da Thaís depois do bug do PIDE reservando marcador
        # fantasma). Protegido por senha: edita a estrutura real do corpus,
        # mesma exposição de arquitetura que tela/espelho/autônomo.
        if st.session_state.get("editar_step") == "aguardar_senha":
            st.session_state.pop("editar_step", None)
            if prompt.strip() == SENHA_ADMIN:
                executar_iniciar_edicao_bloco(memoria, dominio)
            else:
                ai_msg = "🔒 Essa não é a senha certa -- por segurança, não vou editar nada."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
            st.rerun()
            return

        if st.session_state.get("editar_step") == "aguardar_bloco_id":
            id_txt = prompt.strip()
            bloco_id_editar = int(id_txt) if id_txt.isdigit() else None
            bloco_editar = obter_bloco_por_id(memoria, dominio, bloco_id_editar) if bloco_id_editar is not None else None
            if bloco_editar is None:
                ai_msg = f"❌ Não achei o bloco {id_txt!r} nesse universo. Manda só o número do id, ou 'cancelar'."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
                return
            st.session_state.editar_step = "aguardar_campo"
            st.session_state.editar_bloco_id_pendente = bloco_id_editar
            nomes_campos = "\n".join(f"- {r}" for r in _ROTULOS_CAMPO_BONITO.values())
            ai_msg = (
                f"✏️ Bloco {bloco_id_editar} hoje:\n\n{formatar_bloco_atual_para_edicao(bloco_editar)}\n\n"
                f"Qual campo você quer mudar? Digite o nome dele:\n{nomes_campos}\n\n(ou 'terminar')"
            )
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        # Nome do campo (2026-09-13, pedido direto da Thaís: "o nome dos
        # campos pode ser o gatilho de alteração, e uma vez escolhido já
        # libera o espaço pra mudar") -- escolher o campo por nome bonito É o
        # gatilho; a mensagem seguinte já é só o valor novo, sem prefixo
        # nenhum.
        if st.session_state.get("editar_step") == "aguardar_campo":
            if prompt.strip().lower() in {"cancelar", "cancela", "terminar", "termina", "não", "nao", "acabou"}:
                st.session_state.pop("editar_step", None)
                st.session_state.pop("editar_bloco_id_pendente", None)
                ai_msg = "✏️ Beleza, terminamos por aqui."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
                return
            campo_escolhido = encontrar_campo_editavel_por_rotulo(prompt)
            if campo_escolhido is None:
                nomes_campos = "\n".join(f"- {r}" for r in _ROTULOS_CAMPO_BONITO.values())
                ai_msg = f"❌ Não reconheci esse campo. Digite exatamente um destes nomes:\n{nomes_campos}\n\n(ou 'terminar')"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
                return
            st.session_state.editar_step = "aguardar_valor_campo"
            st.session_state.editar_campo_pendente = campo_escolhido
            ai_msg = f"✏️ Beleza, {_ROTULOS_CAMPO_BONITO[campo_escolhido]} -- manda o novo valor:"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        if st.session_state.get("editar_step") == "aguardar_valor_campo":
            bloco_id_editar = st.session_state.editar_bloco_id_pendente
            campo_escolhido = st.session_state.pop("editar_campo_pendente")
            novo_valor = prompt.strip()
            bloco_editar = obter_bloco_por_id(memoria, dominio, bloco_id_editar)
            secao, chave, _ = _CAMPOS_EDITAVEIS_BLOCO[campo_escolhido]
            alvo_atual = bloco_editar["entrada"] if secao == "entrada" else bloco_editar["saidas"][0]
            valor_antigo = alvo_atual["textos"][0] if chave == "textos" else alvo_atual.get(chave, "")
            bloco_editado = editar_bloco(memoria, st.session_state.inconsciente, dominio, bloco_id_editar, {campo_escolhido: novo_valor})
            st.session_state.memoria = memoria
            st.session_state.alnulu_cache = build_alnulu_cache(memoria)
            st.session_state.editar_step = "aguardar_campo"
            nomes_campos = "\n".join(f"- {r}" for r in _ROTULOS_CAMPO_BONITO.values())
            ai_msg = (
                f'✅ {_ROTULOS_CAMPO_BONITO[campo_escolhido]}: "{valor_antigo}" → "{novo_valor}". '
                f"Marcadores renumerados sem furo -- entrada termina em {bloco_editado['entrada']['fim']}, "
                f"saída termina em {bloco_editado['saidas'][0]['fim']}.\n\n"
                f"Quer mudar mais algum campo desse bloco?\n{nomes_campos}\n\n(ou 'terminar')"
            )
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        pede_editar_bloco = "bloco" in tela_cmd and any(g in tela_cmd for g in ("edit", "corrig", "mudar", "alterar"))
        if pede_editar_bloco:
            if sessao_admin_liberada():
                executar_iniciar_edicao_bloco(memoria, dominio)
                st.rerun()
                return
            st.session_state.editar_step = "aguardar_senha"
            ai_msg = "✏️ Isso edita a memória real e renumera marcadores -- só a criadora pode fazer isso. Qual a senha?"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()
            return

        if pede_ajuda:
            # Ajuda agora é um painel de escolha (2026-09-13, por instrução
            # direta da Thaís): usuário comum recebe as instruções mínimas;
            # ADM pede senha e só aí recebe o painel com todas as funções --
            # nunca mais lista os comandos direto num texto sem escolha.
            ai_msg = "❓ Ajuda pra quem?"
            nova_msg = {"role": "assistant", "content": ai_msg, "mostrar_escolha_ajuda": {"dominio": dominio}}
            st.session_state.messages.append(nova_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
                renderizar_escolha_ajuda_no_chat(dominio, f"live_{len(st.session_state.messages)}")
            st.rerun()
            return

        if cmd == "sair":
            st.session_state.messages.append({"role": "assistant", "content": "👋 Até mais!"})
            with st.chat_message("assistant"):
                st.markdown("👋 Até mais!")
            st.session_state.messages = []
            st.session_state.variation = 0
            st.session_state.current_bloco = None
            st.session_state.conversa_blocos = []
            return

        # Matemática é função, não corpus: se a mensagem inteira for uma conta,
        # responde calculado de verdade, sem passar por bloco/candidato/rede
        # neural nenhum -- resposta certa não se aprende por exemplo.
        resultado_calculo = tentar_calcular(s)
        if resultado_calculo is not None:
            ai_msg = f"{resultado_calculo} 🧮"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            with st.chat_message("assistant"):
                st.markdown(ai_msg)
            st.rerun()

        # O harness (adam_harness.py) NÃO roda mais aqui -- ele é uma ferramenta de
        # observação/relatório feita pra rodar sozinha ("python adam_harness.py"),
        # não pra disparar a cada mensagem do chat. Ele reexecuta o lovely.py real
        # do zero (ainda com Mongo ligado, já que o desligamento foi só no
        # lovely_test.py) e sobrescrevia st.session_state com o snapshot antigo do
        # Mongo no meio da conversa -- foi assim que "Olá :3" parou de bater exato
        # de novo, mesmo com tudo corrigido: o harness recarregava dado velho por
        # baixo dos panos antes do match rodar.

        # O motor de autonomia (insepa_autonomia.py -- cosseno acha, Báskara
        # confirma) não roda mais aqui, solto e incondicional a cada
        # mensagem (isso duplicava a mesma busca que melhor_candidato_fraco
        # já faz, inclusive em mensagens que iam bater exato de qualquer
        # jeito). Agora ele só entra em ação quando NADA bate exato -- ver o
        # bloco "if candidato:" logo abaixo, que responde de verdade em vez
        # de só logar uma opinião silenciosa (2026-09-12, autorizado pela
        # Thaís: resposta autônoma aparece na hora, reforço humano decide
        # depois se ela continua confiável).

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
            # Reação aceita = canônica OU qualquer var já registrada pra ela (Camada 5:
            # var registrada É o mesmo dado, nunca "parecido" -- por isso testamos todas
            # como possível sufixo, não só a forma canônica). Mais longa primeiro, pra
            # uma var que contenha outra mais curta não cortar errado.
            candidatas_reac = sorted(
                (r.lower().strip() for r in _reacoes_aceitas_do_bloco(b, dominio) if r),
                key=len, reverse=True,
            )
            for bloco_reac in candidatas_reac:
                if s.lower().strip().endswith(bloco_reac):
                    reac_candidata = bloco_reac
                    txt_candidato = s[:-len(bloco_reac)].rstrip()
                    # Bate se: igual ao canônico, igual a uma multivar registrada (frase
                    # inteira), OU palavra por palavra com var registrada em cada posição
                    # -- nunca por amostra aleatória (variar_texto era só pra criatividade
                    # de saída, não pra decidir se bateu).
                    textos_exatos = [b["entrada"]["texto"]] + b["entrada"].get("Multivars_Texto_Entrada", [])
                    bateu_texto = (
                        any(normalize(t) == normalize(txt_candidato) for t in textos_exatos)
                        or _texto_aceito_pelo_bloco(txt_candidato, b, dominio, "entrada")
                    )
                    if bateu_texto:
                        reac = reac_candidata
                        txt = txt_candidato
                        bloco = b
                        break
            if bloco:
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
            is_exato = (
                (any(normalize(t) == normalize(txt) for t in [bloco["entrada"]["texto"]] + bloco["entrada"].get("Multivars_Texto_Entrada", [])) or _texto_aceito_pelo_bloco(txt, bloco, dominio, "entrada"))
                and reac in _reacoes_aceitas_do_bloco(bloco, dominio)
            )
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

        # Se nenhum bloco encontrado: sem distinção usuário/admin -- é sempre a
        # criadora falando, então nunca fica preso em "opinião pendente" esperando
        # revisão de terceiro; vai direto pro modo criadora.
        if bloco is None:
            if "cerbero_step" not in st.session_state and "criadora_step" not in st.session_state:
                txt, reac = parse_text_reaction(s)

                # Reflexão antes de desistir: mesmo sem bater o piso de confiança, o
                # Adam pode ter um candidato fraco -- ele é livre pra errar o bloco,
                # desde que reproduza a resposta e PERGUNTE se o contexto corresponde,
                # em vez de responder com confiança ou fingir que não sabe de nada.
                candidato = melhor_candidato_fraco(txt, dominio, reac)

                # É sempre você falando direto com ele -- já é dado concreto por
                # definição, nunca pergunta "é fato ou opinião" (essa pergunta só
                # fazia sentido pro balãozinho de opinião de terceiro, que não existe mais).
                st.session_state.criadora_txt = txt
                st.session_state.criadora_reac = reac
                st.session_state.criadora_dominio = dominio

                if candidato:
                    # Motor de autonomia: responde JÁ, em vez de perguntar "posso
                    # usar essa resposta?" e esperar confirmação toda vez -- nunca
                    # cria bloco novo sozinho (Camada 5, isso não muda), só
                    # reaproveita a saída de um bloco que já existe, marcado como
                    # tentativa autônoma. Reforço humano (pesos 66/12/23 já
                    # existentes -- record_human_reinforcement) decide se essa
                    # correspondência continua confiável, em vez de gate antes.
                    bloco_c = candidato["bloco"]
                    saida_sugerida_texto = bloco_c["saidas"][0]["textos"][0]
                    saida_sugerida_reacao = bloco_c["saidas"][0].get("reacao", "")
                    # 2026-09-13, pedido direto da Thaís: um bloco com campo placeholder
                    # (ex.: reação "0.0", deixada pra depois no fluxo livre) pode virar
                    # candidato normalmente -- cosseno/Báskara só olham texto e emoção de
                    # verdade -- mas o placeholder NUNCA pode vazar pra dentro de uma
                    # resposta real mostrada no chat.
                    if _eh_placeholder_im(saida_sugerida_reacao):
                        saida_sugerida_reacao = ""
                    texto_com_reacao_sugerida = f'{saida_sugerida_texto} {saida_sugerida_reacao}'.strip()
                    ai_msg = (
                        f'{texto_com_reacao_sugerida}\n\n'
                        f'🤖 *(tentativa autônoma)*'
                    )
                    nova_msg = {
                        "role": "assistant", "content": ai_msg,
                        "reforco_autonomo": {"bloco_id": bloco_c["bloco_id"], "resposta": saida_sugerida_texto},
                    }
                    st.session_state.messages.append(nova_msg)
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                        renderizar_reforco_autonomo_no_chat(bloco_c["bloco_id"], saida_sugerida_texto, f"live_{len(st.session_state.messages)}")
                    st.rerun()
                    return
                else:
                    # Nada parecido ainda -- mas é a criadora falando, então não tem
                    # pedido de permissão nem cobrança de prova. Ele já confia que faz
                    # sentido; só quer entender do que se trata, com carinho (2026-09-13,
                    # ordem corrigida pela Thaís: contexto vem ANTES do pensamento --
                    # "do que se trata" é contexto, não porquê). Mostra também o palpite
                    # cru da rede neural (antes ficava escondido) -- pedido dela: contato
                    # direto com o "cérebro" pra poder ensinar de verdade.
                    st.session_state.criadora_step = "escolher_campo_criacao"
                    ai_msg = (
                        gerar_saudacao_desconhecido() + formatar_palpite_cru_para_chat(txt, reac, dominio)
                        + f"\n\nQual campo você quer preencher? Diga o nome:\n{_lista_campos_criacao()}\n\n(ou 'finalizar' quando já tiver o suficiente)"
                    )

                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_reacao_entrada":
                # Só existe pro Livro de Lux (2026-09-13) -- parágrafo nunca
                # vem com reação pronta, diferente da frase do jogo_frase.
                # Depois de coletar, segue pro mesmo caminho de sempre: se
                # achou candidato parecido, confirma a saída emprestada; se
                # não, pergunta pensamento do zero.
                st.session_state.criadora_reac = prompt.strip()
                tem_candidato = st.session_state.pop("criadora_lux_tem_candidato", False)
                if tem_candidato:
                    st.session_state.criadora_step = "confirmar_saida"
                    reacao_sugerida_exibir = "" if _eh_placeholder_im(st.session_state["criadora_saida_sugerida_reacao"]) else st.session_state["criadora_saida_sugerida_reacao"]
                    saida_sugerida_exibir = f'{st.session_state["criadora_saida_sugerida_texto"]} {reacao_sugerida_exibir}'.strip()
                    ai_msg = (
                        f'Anotado! Isso me lembra de algo que já sei. '
                        f'Baseado nisso, eu diria: "{saida_sugerida_exibir}". '
                        f'Posso usar essa mesma resposta, ou você quer outra?'
                    )
                else:
                    st.session_state.criadora_step = "escolher_campo_criacao"
                    ai_msg = (
                        gerar_saudacao_desconhecido() + formatar_palpite_cru_para_chat(
                            st.session_state.criadora_txt, st.session_state.criadora_reac, st.session_state.criadora_dominio,
                        )
                        + f"\n\nQual campo você quer preencher? Diga o nome:\n{_lista_campos_criacao()}\n\n(ou 'finalizar' quando já tiver o suficiente)"
                    )
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "escolher_campo_criacao":
                # 2026-09-13, redesenho pedido direto pela Thaís: contexto/
                # pensamento/saída deixam de ser uma sequência fixa e
                # obrigatória -- ela escolhe o campo pelo NOME, na ordem que
                # quiser (igual o "editar bloco" do ADM), e finaliza quando
                # decidir; campo não preenchido vira placeholder.
                resp_norm = _normalizar_rotulo_campo(prompt)
                palavras_finalizar = ("finaliz", "pronto", "termin", "chega", "so isso", "e isso")
                if any(p in resp_norm for p in palavras_finalizar):
                    if not st.session_state.get("criadora_saida_reacao_valor", "").strip():
                        st.session_state.criadora_step = "confirmar_finalizar_sem_reacao_saida"
                        ai_msg = "Ainda não tenho uma reação pra saída -- quer adicionar agora, ou finalizar com um placeholder mesmo?"
                    else:
                        ai_msg = _finalizar_criacao_livre(memoria)
                else:
                    campo_escolhido = encontrar_campo_editavel_por_rotulo(prompt)
                    if campo_escolhido not in _CAMPOS_CRIACAO_VALIDOS:
                        ai_msg = f"❌ Não reconheci esse campo. Diga o nome certinho:\n{_lista_campos_criacao()}\n\n(ou 'finalizar')"
                    else:
                        st.session_state.criadora_campo_criacao_pendente = campo_escolhido
                        st.session_state.criadora_step = "coletar_valor_campo_criacao"
                        ai_msg = f"Qual o valor pra {_ROTULOS_CAMPO_BONITO[campo_escolhido]}?"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_valor_campo_criacao":
                campo_criacao = st.session_state.pop("criadora_campo_criacao_pendente", None)
                if campo_criacao == "entrada_contexto":
                    st.session_state.criadora_contexto = prompt.strip()
                elif campo_criacao == "entrada_pensamento":
                    st.session_state.criadora_pensamento = prompt.strip()
                elif campo_criacao == "saida_texto":
                    saida_txt, saida_reac = parse_text_reaction(prompt.strip())
                    st.session_state.criadora_saida_texto_valor = saida_txt or prompt.strip()
                    if saida_reac:
                        st.session_state.criadora_saida_reacao_valor = saida_reac
                elif campo_criacao == "saida_reacao":
                    st.session_state.criadora_saida_reacao_valor = prompt.strip()
                elif campo_criacao == "saida_contexto":
                    st.session_state.criadora_saida_contexto_valor = prompt.strip()
                st.session_state.criadora_step = "escolher_campo_criacao"
                ai_msg = f"Anotado! Mais algum campo, ou 'finalizar'?\n{_lista_campos_criacao()}"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "confirmar_finalizar_sem_reacao_saida":
                # 2026-09-13, bug real: comparar a frase inteira contra um conjunto
                # fixo de frases exatas ("finalizar mesmo" etc.) falhava assim que a
                # resposta vinha fraseada de outro jeito (ex.: "finalizar com
                # placeholder") -- e o texto INTEIRO da resposta virava a reação sem
                # querer, poluindo o bloco de verdade. Agora basta a resposta CONTER
                # uma palavra-chave de placeholder, em qualquer frase.
                resp_norm = _normalizar_rotulo_campo(prompt)
                palavras_placeholder = ("placeholder", "mesmo", "sem reacao", "pular", "deixa assim")
                if not any(p in resp_norm for p in palavras_placeholder):
                    st.session_state.criadora_saida_reacao_valor = prompt.strip()
                ai_msg = _finalizar_criacao_livre(memoria)
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "confirmar_saida":
                # A saída veio sugerida (emprestada do candidato) -- três caminhos, não
                # dois: confirma que serve, pede uma diferente (sem ainda dizer qual --
                # nesse caso ESPERA a resposta de verdade antes de gravar qualquer coisa),
                # ou já entrega o texto final direto.
                tokens_resp = {t.lower() for t in Token(prompt)}
                if tokens_resp & {"sim", "s", "pode", "isso", "serve", "yes", "y"}:
                    # Reaproveita a saída inteira, contexto incluso -- já era um dado
                    # concreto de verdade em outro bloco, não precisa perguntar de novo.
                    # MAS (2026-09-13, bug real achado pela Thaís) se o bloco emprestado
                    # já tinha vindo sem reação registrada -- vazia OU só um placeholder
                    # tipo "0.0" (bloco criado com a reação deixada pra depois, no fluxo
                    # livre por campo) -- nunca replica esse buraco pro bloco novo como
                    # se fosse conteúdo real. Placeholder pode virar candidato normal
                    # (cosseno/Báskara só olham texto e emoção de verdade), só não pode
                    # ser copiado sem avisar.
                    saida_txt = st.session_state["criadora_saida_sugerida_texto"]
                    saida_reac = st.session_state["criadora_saida_sugerida_reacao"]
                    saida_contexto = st.session_state.get("criadora_saida_sugerida_contexto", "")
                    if saida_reac and not _eh_placeholder_im(saida_reac):
                        ai_msg = finalizar_bloco_criadora(memoria, saida_txt, saida_reac, saida_contexto)
                    else:
                        st.session_state.criadora_saida_texto_pendente = saida_txt
                        st.session_state.criadora_contexto_saida_reusado = saida_contexto
                        st.session_state.criadora_step = "coletar_saida_reacao_faltante"
                        ai_msg = "Boa, mas notei que essa resposta emprestada não tinha uma reação de verdade registrada (só um placeholder). Que sentimento essa resposta provoca? (tipo 🙂, :), ou *sorriso*)"
                elif tokens_resp & {"outra", "outro", "diferente"}:
                    st.session_state.criadora_step = "aguardar_saida_manual"
                    ai_msg = "Certo, qual resposta você prefere então?"
                else:
                    saida_txt, saida_reac = parse_text_reaction(prompt.strip())
                    st.session_state.criadora_saida_texto_pendente = saida_txt or prompt.strip()
                    st.session_state.criadora_saida_reac_pendente = saida_reac
                    st.session_state.criadora_step, ai_msg = _proximo_passo_apos_saida("Entendido!")
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "aguardar_saida_manual":
                # Ela pediu uma resposta diferente da sugerida -- isso aqui É o texto
                # real, finalmente, mas ainda falta o contexto (diferente do bloco
                # emprestado, então não dá pra reaproveitar o contexto de lá).
                saida_txt, saida_reac = parse_text_reaction(prompt.strip())
                st.session_state.criadora_saida_texto_pendente = saida_txt or prompt.strip()
                st.session_state.criadora_saida_reac_pendente = saida_reac
                st.session_state.criadora_step, ai_msg = _proximo_passo_apos_saida("Combinado!")
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_saida_reacao_faltante":
                # Só existe quando a saída (nova ou emprestada de um candidato) veio
                # sem reação -- nunca deixa um bloco novo nascer incompleto assim
                # (2026-09-13, bug real: bloco 3 do universo 0 tinha saída sem
                # reação porque esse buraco passava batido em silêncio).
                st.session_state.criadora_saida_reac_pendente = prompt.strip()
                contexto_reusado = st.session_state.pop("criadora_contexto_saida_reusado", None)
                if contexto_reusado is not None:
                    saida_txt_final = st.session_state.pop("criadora_saida_texto_pendente", "")
                    saida_reac_final = st.session_state.pop("criadora_saida_reac_pendente", "")
                    ai_msg = finalizar_bloco_criadora(memoria, saida_txt_final, saida_reac_final, contexto_reusado)
                else:
                    st.session_state.criadora_step = "coletar_saida_contexto"
                    ai_msg = "Show! Que conclusão devo tirar disso?"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_saida_contexto":
                # Último passo antes de gravar -- "Confirmado pela criadora" nunca
                # mais ocupa esse lugar sozinho; contexto é sempre perguntado de
                # verdade agora (ex.: "Como está?" -> "Informação do humor").
                saida_contexto = prompt.strip()
                saida_txt = st.session_state.pop("criadora_saida_texto_pendente", "")
                saida_reac = st.session_state.pop("criadora_saida_reac_pendente", "")
                ai_msg = finalizar_bloco_criadora(memoria, saida_txt, saida_reac, saida_contexto)
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "confirmar_bloco_criado":
                # Último passo: mostrar o bloco inteiro que acabou de gravar e
                # perguntar se era isso mesmo -- se não for, apaga (nunca edita
                # por cima às cegas) e ela pode tentar de novo do zero.
                tokens_resp = {t.lower() for t in Token(prompt)}
                positivo = bool(tokens_resp & {"sim", "s", "isso", "era", "certo", "correto", "perfeito", "yes", "y"})
                negativo = bool(tokens_resp & {"não", "nao", "n", "errado", "errada", "no"})
                bloco_id_criado = st.session_state.pop("criadora_bloco_criado_id", None)
                dominio_criado = st.session_state.pop("criadora_bloco_criado_dominio", dominio)
                st.session_state.pop("criadora_step", None)
                if negativo and not positivo:
                    apagou = apagar_ultimo_bloco_criado(memoria, dominio_criado, bloco_id_criado)
                    if apagou:
                        ai_msg = f'Sem problemas, apaguei o bloco #{bloco_id_criado} -- os marcadores que ele usou não voltam a ser reaproveitados (Camada 1), mas o bloco em si sumiu. Quer tentar de novo?'
                    else:
                        ai_msg = 'Hmm, não achei mais esse bloco pra apagar -- pode já ter mudado. Confere na Tela pra ver o estado atual.'
                elif positivo:
                    # 2026-09-13, pedido direto da Thaís: depois de confirmado, oferece
                    # Vars (variação de UMA palavra/reação -- cobre as variações dos
                    # termos de quem fala) e Multivars (frase/reação alternativa inteira
                    # -- o Adam variando a própria resposta sem fugir do assunto) --
                    # como prompt OPCIONAL e por nome de campo, igual o "editar bloco"
                    # do ADM: nome do campo é o próprio gatilho, em loop até ela dizer
                    # 'não'/'terminar', em vez de uma sequência fixa e obrigatória.
                    st.session_state.criadora_step = "perguntar_variacao_campo"
                    st.session_state.criadora_bloco_var_id = bloco_id_criado
                    st.session_state.criadora_bloco_var_dominio = dominio_criado
                    ai_msg = (
                        "Que bom! Guardado com carinho. 💚\n\n"
                        "Quer adicionar variação (Vars ou Multivars) a algum campo? "
                        "Diga o nome (Texto de entrada / Reação de entrada / Texto de saída / Reação de saída), "
                        "ou 'não'/'terminar'."
                    )
                else:
                    ai_msg = "Combinado, seguimos então!"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "perguntar_variacao_campo":
                if prompt.strip().lower() in {"não", "nao", "n", "terminar", "termina", "não quero", "nao quero"}:
                    st.session_state.pop("criadora_bloco_var_id", None)
                    st.session_state.pop("criadora_bloco_var_dominio", None)
                    st.session_state.pop("criadora_step", None)
                    ai_msg = "Combinado, terminamos por aqui! 💚"
                else:
                    campo_escolhido = encontrar_campo_editavel_por_rotulo(prompt)
                    if campo_escolhido not in _CAMPOS_VAR_VALIDOS:
                        ai_msg = "❌ Não reconheci esse campo (contexto e pensamento não têm variação por palavra, só texto e reação). Diga o nome certinho (Texto de entrada / Reação de entrada / Texto de saída / Reação de saída), ou 'não'/'terminar'."
                    else:
                        st.session_state.criadora_var_campo_pendente = campo_escolhido
                        st.session_state.criadora_step = "perguntar_tipo_variacao"
                        ai_msg = f"Beleza, {_ROTULOS_CAMPO_BONITO[campo_escolhido]} -- é Vars (variação de uma palavra/símbolo) ou Multivars (frase/reação inteira alternativa)?"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "perguntar_tipo_variacao":
                tokens_resp = {t.lower() for t in Token(prompt)}
                campo_var = st.session_state.get("criadora_var_campo_pendente")
                if tokens_resp & {"var", "vars"}:
                    if campo_var.endswith("_reacao"):
                        st.session_state.criadora_step = "coletar_vars_valores"
                        ai_msg = "Quais são as variações? (separadas por vírgula, ex.: :3, ^^, =^.^=)"
                    else:
                        st.session_state.criadora_step = "escolher_palavra_var"
                        ai_msg = "Qual palavra da frase tem variação?"
                elif tokens_resp & {"multivar", "multivars"}:
                    st.session_state.criadora_step = "coletar_multivar_valor"
                    ai_msg = "Qual a forma alternativa (frase ou reação inteira)?"
                else:
                    ai_msg = "Perae, não entendi -- é Vars ou Multivars?"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "escolher_palavra_var":
                st.session_state.criadora_var_palavra_pendente = prompt.strip()
                st.session_state.criadora_step = "coletar_vars_valores"
                ai_msg = "Quais são as variações dessa palavra? (separadas por vírgula, ex.: Oi, Oie, Oiiie)"
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_vars_valores":
                campo_var = st.session_state.pop("criadora_var_campo_pendente", None)
                palavra_var = st.session_state.pop("criadora_var_palavra_pendente", "")
                bloco_id_var = st.session_state.get("criadora_bloco_var_id")
                dominio_var = st.session_state.get("criadora_bloco_var_dominio", dominio)
                novas_vars = [v.strip() for v in prompt.split(",") if v.strip()]
                if campo_var.endswith("_reacao"):
                    b = obter_bloco_por_id(memoria, dominio_var, bloco_id_var)
                    secao = b["entrada"] if campo_var.startswith("entrada") else b["saidas"][0]
                    palavra_var = secao.get("reacao", "")
                ok = adicionar_var_bloco(memoria, st.session_state.inconsciente, dominio_var, bloco_id_var, campo_var, palavra_var, novas_vars)
                st.session_state.criadora_step = "perguntar_variacao_campo"
                confirmacao = f'Anotado! "{palavra_var}" agora aceita {", ".join(novas_vars)} também.' if ok else f'Hmm, não achei "{palavra_var}" nesse campo do bloco -- não consegui registrar essa variação.'
                ai_msg = f"{confirmacao} Quer adicionar mais alguma variação, em outro campo (ou nesse mesmo)? Diga o nome, ou 'não'/'terminar'."
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()

            elif st.session_state.get("criadora_step") == "coletar_multivar_valor":
                campo_var = st.session_state.pop("criadora_var_campo_pendente", None)
                bloco_id_var = st.session_state.get("criadora_bloco_var_id")
                dominio_var = st.session_state.get("criadora_bloco_var_dominio", dominio)
                ok = adicionar_multivar_bloco(memoria, dominio_var, bloco_id_var, campo_var, prompt.strip())
                st.session_state.criadora_step = "perguntar_variacao_campo"
                confirmacao = f'Anotado! "{prompt.strip()}" registrado como forma alternativa.' if ok else "Hmm, não consegui registrar essa variação."
                ai_msg = f"{confirmacao} Quer adicionar mais alguma variação, em outro campo (ou nesse mesmo)? Diga o nome, ou 'não'/'terminar'."
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
                if prompt.strip().lower() in PALAVRAS_PULAR:
                    st.session_state.new_contexto = f"{dominio}.0"  # ponto neutro do universo -- campo pulado, nunca vazio
                    st.session_state.cerbero_step = "collect_thought"
                    ai_msg = f'Sem problema, deixo o contexto em aberto por enquanto. O quê devo pensar a respeito do assunto (ou "pular")?'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
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
                if prompt.strip().lower() in PALAVRAS_PULAR:
                    st.session_state.new_contexto = f"{dominio}.0"
                    st.session_state.cerbero_step = "collect_thought"
                    ai_msg = f'Sem problema, deixo o contexto em aberto por enquanto. O quê devo pensar a respeito do assunto (ou "pular")?'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
                contexto = parse_quoted_response(prompt)
                st.session_state.new_contexto = contexto
                st.session_state.cerbero_step = "collect_thought"
                ai_msg = f'Ótimo! Agora que temos a expressão "{st.session_state.new_input}" ligada à emoção "{st.session_state.new_reac}" e o contexto "{st.session_state.new_contexto}". O quê devo pensar a respeito do assunto (ou "pular")?'
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
                st.rerun()
            elif st.session_state.cerbero_step == "collect_thought":
                if prompt.strip().lower() in PALAVRAS_PULAR:
                    st.session_state.new_pensamento = f"{dominio}.0"
                    st.session_state.cerbero_step = "ask_add_entrada_phrase"
                    ai_msg = f'Sem problema. Quer adicionar uma frase alternativa para entrada? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
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
                if prompt.strip().lower() in PALAVRAS_PULAR:
                    st.session_state.new_pensamento = f"{dominio}.0"
                    st.session_state.cerbero_step = "ask_add_entrada_phrase"
                    ai_msg = f'Sem problema. Quer adicionar uma frase alternativa para entrada? Responda "sim" ou "não".'
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    with st.chat_message("assistant"):
                        st.markdown(ai_msg)
                    st.rerun()
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

                    # out["out"] na geração (tgt=None) é (max_out_len, batch) -- tempo
                    # primeiro, não lote primeiro. [0] pegava só o 1º instante gerado
                    # (uma palavra só); [:, 0] pega a sequência inteira do lote 0.
                    generated_ids = out["out"][:, 0]

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
    """Sem distinção usuário/admin -- é sempre a criadora, então o assistente
    automático do harness está sempre ligado por padrão."""
    return bool(st.session_state.get("harness_auto_admin", True))


## should_run_autonomous_mode() e try_autonomous_learning_from_prompt()
## destruídos por instrução direta da Thaís em 2026-09-12 -- rodavam em
## cima de corpus_similarity_score/calcular_similaridade (aproximação
## contínua, corte fixo 0.70), exatamente o que o resto do projeto já
## tinha abandonado em favor de correspondência exata (Báskara). O motor
## novo mora em insepa_autonomia.py.


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
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng,
        out_vocab_floats=list(idx_to_txt.keys()),
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
            tokens = palavras_do_campo(b, field)[:max_len]  # nunca deixar n > max_len passar puro (padding negativo não trunca em Python)
            ngrams_list = [generate_ngrams(alnulu_string(t), N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist][:max_len * max_ng]
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

    Sem distinção usuário/admin -- todos os universos, incluindo o "0" (o
    "Big Bang" do Adam), ficam disponíveis direto."""
    ims = list(memoria.get("IM", {}).keys())
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

    def _combinar_vars_template(tokens: List[str]) -> List[str]:
        # Combinar tokens consecutivos palavra + [vars: ...]
        combinados = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i + 1].startswith('[vars:'):
                combinados.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                combinados.append(tokens[i])
                i += 1
        return combinados

    # 2026-09-13, pedido direto da Thaís: texto de entrada/saída se divide em
    # 3 famílias de marcador pelo travessão "—" (TEXE/FADEN/TEFE) -- mesma
    # regra de criar_bloco_concreto, aplicada aqui antes do parser de
    # [vars: ...] do template.
    texe_str, faden_str, tefe_str = dividir_travessao(entrada_texto)
    TEXE = _combinar_vars_template(Token_with_vars(texe_str))
    FADEN = _combinar_vars_template(Token_with_vars(faden_str))
    TEFE = _combinar_vars_template(Token_with_vars(tefe_str))

    RE = _combinar_vars_template([entrada_reacao] if entrada_reacao else [])

    CE = Token(entrada_contexto)
    pensamento_limpo = entrada_pensamento.strip('"')
    partes = pensamento_limpo.split('.')[:3]
    PIDE_full = []
    for parte in partes:
        PIDE_full.extend(Token(parte.strip()))
    # Só os 3 primeiros tokens viram marcador -- o texto completo do
    # pensamento continua 100% preservado à parte, em pensamento_interno
    # (string crua). Corrigido em 2026-09-13 por instrução direta da Thaís:
    # reservar marcador sem token visível quebra a Camada 1 -- "chegou no
    # fim de uma sequência, inicia a outra", sempre, sem gap escondido.
    PIDE_limited = PIDE_full[:3]
    TEDSA, FADES, TEFSA = [], [], []
    for t in saidas_textos:
        tedsa_str, fades_str, tefsa_str = dividir_travessao(t)
        TEDSA += _combinar_vars_template(Token_with_vars(tedsa_str))
        FADES += _combinar_vars_template(Token_with_vars(fades_str))
        TEFSA += _combinar_vars_template(Token_with_vars(tefsa_str))
    RS = _combinar_vars_template([saida_reacao] if saida_reacao else [])
    CS = Token(saida_contexto)
    entrada_tokens = TEXE + FADEN + TEFE + RE + CE + PIDE_limited
    saida_tokens = TEDSA + FADES + TEFSA + RS + CS
    ent_marks = generate_markers(current_last, len(entrada_tokens))
    out_marks = generate_markers(ent_marks[-1], len(saida_tokens))
    fim_ent = ent_marks[-1]
    fim_out = out_marks[-1]
    idx = 0
    TEXE_m = ent_marks[idx: idx + len(TEXE)]; idx += len(TEXE)
    FADEN_m = ent_marks[idx: idx + len(FADEN)]; idx += len(FADEN)
    TEFE_m = ent_marks[idx: idx + len(TEFE)]; idx += len(TEFE)
    RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
    CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
    PIDE_m = ent_marks[idx: idx + len(PIDE_limited)]
    jdx = 0
    TEDSA_m = out_marks[jdx: jdx + len(TEDSA)]; jdx += len(TEDSA)
    FADES_m = out_marks[jdx: jdx + len(FADES)]; jdx += len(FADES)
    TEFSA_m = out_marks[jdx: jdx + len(TEFSA)]; jdx += len(TEFSA)
    RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
    CS_m = out_marks[jdx: jdx + len(CS)]
    bloco["entrada"]["tokens"] = {
        "TEXE": TEXE_m,
        "FADEN": FADEN_m,
        "TEFE": TEFE_m,
        "RE": RE_m,
        "CE": CE_m,
        "PIDE": PIDE_m,
        "TOTAL": ent_marks
    }
    bloco["entrada"]["fim"] = fim_ent
    bloco["saidas"][0]["tokens"] = {
        "TEDSA": TEDSA_m,
        "FADES": FADES_m,
        "TEFSA": TEFSA_m,
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
    
    for m, t in zip(ent_marks, entrada_tokens):
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


def renumerar_marcadores_universo(memoria: dict, inconsciente: dict, dominio: str, resetar_vars: Optional[Dict[int, Set[str]]] = None) -> None:
    """Reconstrói TODOS os marcadores (Camada 1) de um universo do zero, na
    ordem dos blocos, sempre sem furo -- fim de uma sequência sempre inicia a
    outra (bug real corrigido em 2026-09-13: PIDE reservava marcador fantasma
    além dos 3 tokens que de fato guarda). Preserva os vars/multivars já
    registrados no inconsciente campo a campo (E/RE/CE/PIDE/S/RS/CS), a não
    ser que o campo esteja listado em resetar_vars[bloco_id] -- usado quando o
    conteúdo do campo mudou de verdade (editar_bloco), já que vars antigos não
    fazem sentido pra um texto novo. Única fonte de verdade da geração de
    marcador: tanto o reparo do corpus quanto editar_bloco chamam esta função,
    nunca duplicam a lógica."""
    resetar_vars = resetar_vars or {}
    universo = memoria["IM"][dominio]
    inco_universo = inconsciente.setdefault("INCO", {}).setdefault(
        dominio, {"NOME": universo.get("nome", f"IM_{dominio}"), "Ultimo child": f"{dominio}.0", "Blocos": []}
    )
    inco_blocos_by_id = {b["Bloco_id"]: b for b in inco_universo.get("Blocos", [])}

    current_last = f"{dominio}.0"
    novos_inco_blocos = []
    for bloco in universo["blocos"]:
        bloco_id = bloco["bloco_id"]
        bloco_id_str = str(bloco_id)
        campos_reset = resetar_vars.get(bloco_id, set())
        entrada = bloco["entrada"]
        saida = bloco["saidas"][0]
        texto, reacao, contexto, pensamento = entrada["texto"], entrada["reacao"], entrada["contexto"], entrada["pensamento_interno"]
        saida_texto, saida_reacao, saida_contexto = saida["textos"][0], saida["reacao"], saida["contexto"]

        # 2026-09-13, pedido direto da Thaís: texto de entrada/saída sempre
        # se divide em 3 famílias pelo travessão -- ver dividir_travessao().
        texe_str, faden_str, tefe_str = dividir_travessao(texto)
        TEXE, FADEN, TEFE = Token(texe_str), Token(faden_str), Token(tefe_str)
        RE = [reacao] if reacao else []
        CE = Token(contexto)
        PIDE = Token(pensamento)[:3]
        tedsa_str, fades_str, tefsa_str = dividir_travessao(saida_texto)
        TEDSA, FADES, TEFSA = Token(tedsa_str), Token(fades_str), Token(tefsa_str)
        RS = [saida_reacao] if saida_reacao else []
        CS = Token(saida_contexto)

        old_tokens_e = entrada.get("tokens", {})
        old_tokens_s = saida.get("tokens", {})
        old_inco_bloco = inco_blocos_by_id.get(bloco_id_str)
        old_inco_entrada = (old_inco_bloco or {}).get("Entrada", {})
        old_inco_saida = (old_inco_bloco or {}).get("SAÍDA", {})
        campos_entrada = ("TEXE", "FADEN", "TEFE", "RE", "CE", "PIDE")

        def vars_para(campo: str, novos_tokens: List[str], default_vars: List[str]) -> List[List[str]]:
            if campo in campos_reset or old_inco_bloco is None:
                return [default_vars for _ in novos_tokens]
            old_marcadores = old_tokens_e.get(campo) if campo in campos_entrada else old_tokens_s.get(campo)
            old_dict = old_inco_entrada if campo in campos_entrada else old_inco_saida
            if not old_marcadores or len(old_marcadores) != len(novos_tokens):
                return [default_vars for _ in novos_tokens]
            pares = []
            for old_m, novo_tok in zip(old_marcadores, novos_tokens):
                velho = old_dict.get(old_m)
                if velho is None or velho.get("token") != novo_tok:
                    return [default_vars for _ in novos_tokens]
                pares.append(velho["vars"])
            return pares

        vars_TEXE = vars_para("TEXE", TEXE, ["0.0"])
        vars_FADEN = vars_para("FADEN", FADEN, ["0.0"])
        vars_TEFE = vars_para("TEFE", TEFE, ["0.0"])
        vars_RE = vars_para("RE", RE, ["0.0"])
        vars_CE = vars_para("CE", CE, ["0.0"])
        vars_PIDE = vars_para("PIDE", PIDE, ["0.0"])
        vars_TEDSA = vars_para("TEDSA", TEDSA, ["0.0"])
        vars_FADES = vars_para("FADES", FADES, ["0.0"])
        vars_TEFSA = vars_para("TEFSA", TEFSA, ["0.0"])
        vars_RS = vars_para("RS", RS, ["0.0"])
        vars_CS = vars_para("CS", CS, ["0.0"])

        ent_marks = generate_markers(current_last, len(TEXE) + len(FADEN) + len(TEFE) + len(RE) + len(CE) + len(PIDE))
        out_marks = generate_markers(ent_marks[-1], len(TEDSA) + len(FADES) + len(TEFSA) + len(RS) + len(CS))

        idx = 0
        TEXE_m = ent_marks[idx: idx + len(TEXE)]; idx += len(TEXE)
        FADEN_m = ent_marks[idx: idx + len(FADEN)]; idx += len(FADEN)
        TEFE_m = ent_marks[idx: idx + len(TEFE)]; idx += len(TEFE)
        RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
        CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
        PIDE_m = ent_marks[idx: idx + len(PIDE)]
        jdx = 0
        TEDSA_m = out_marks[jdx: jdx + len(TEDSA)]; jdx += len(TEDSA)
        FADES_m = out_marks[jdx: jdx + len(FADES)]; jdx += len(FADES)
        TEFSA_m = out_marks[jdx: jdx + len(TEFSA)]; jdx += len(TEFSA)
        RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
        CS_m = out_marks[jdx: jdx + len(CS)]

        entrada["tokens"] = {
            "TEXE": TEXE_m, "FADEN": FADEN_m, "TEFE": TEFE_m,
            "RE": RE_m, "CE": CE_m, "PIDE": PIDE_m, "TOTAL": ent_marks,
        }
        entrada["fim"] = ent_marks[-1]
        saida["tokens"] = {
            "TEDSA": TEDSA_m, "FADES": FADES_m, "TEFSA": TEFSA_m,
            "RS": RS_m, "CS": CS_m, "TOTAL": out_marks,
        }
        saida["fim"] = out_marks[-1]

        novo_inco_entrada = {}
        for m, t, v in zip(TEXE_m, TEXE, vars_TEXE):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        for m, t, v in zip(FADEN_m, FADEN, vars_FADEN):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        for m, t, v in zip(TEFE_m, TEFE, vars_TEFE):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        for m, t, v in zip(RE_m, RE, vars_RE):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        for m, t, v in zip(CE_m, CE, vars_CE):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        for m, t, v in zip(PIDE_m, PIDE, vars_PIDE):
            novo_inco_entrada[m] = {"token": t, "vars": v}
        novo_inco_saida = {}
        for m, t, v in zip(TEDSA_m, TEDSA, vars_TEDSA):
            novo_inco_saida[m] = {"token": t, "vars": v}
        for m, t, v in zip(FADES_m, FADES, vars_FADES):
            novo_inco_saida[m] = {"token": t, "vars": v}
        for m, t, v in zip(TEFSA_m, TEFSA, vars_TEFSA):
            novo_inco_saida[m] = {"token": t, "vars": v}
        for m, t, v in zip(RS_m, RS, vars_RS):
            novo_inco_saida[m] = {"token": t, "vars": v}
        for m, t, v in zip(CS_m, CS, vars_CS):
            novo_inco_saida[m] = {"token": t, "vars": v}

        novos_inco_blocos.append({"Bloco_id": bloco_id_str, "Entrada": novo_inco_entrada, "SAÍDA": novo_inco_saida})
        current_last = out_marks[-1]

    inco_universo["Blocos"] = novos_inco_blocos
    universo["ultimo_child"] = current_last
    inco_universo["Ultimo child"] = current_last


_CAMPOS_EDITAVEIS_BLOCO = {
    # O 3º elemento agora é uma TUPLA de famílias de marcador -- texto de
    # entrada/saída viraram 3 campos cada (TEXE/FADEN/TEFE, TEDSA/FADES/
    # TEFSA -- 2026-09-13, pedido direto da Thaís), então editar o texto
    # inteiro precisa resetar vars nos 3 de uma vez, não só em um.
    "entrada_texto": ("entrada", "texto", ("TEXE", "FADEN", "TEFE")),
    "entrada_reacao": ("entrada", "reacao", ("RE",)),
    "entrada_contexto": ("entrada", "contexto", ("CE",)),
    "entrada_pensamento": ("entrada", "pensamento_interno", ("PIDE",)),
    "saida_texto": ("saida", "textos", ("TEDSA", "FADES", "TEFSA")),
    "saida_reacao": ("saida", "reacao", ("RS",)),
    "saida_contexto": ("saida", "contexto", ("CS",)),
}

# Nomes bonitos (2026-09-13, pedido direto da Thaís: "já que o prompt pede
# os campos deixa bonitinho" -- o próprio nome digitado é o gatilho de
# seleção do campo na edição, nunca a chave interna em snake_case).
_ROTULOS_CAMPO_BONITO = {
    "entrada_texto": "Texto de entrada",
    "entrada_reacao": "Reação de entrada",
    "entrada_contexto": "Contexto de entrada",
    "entrada_pensamento": "Pensamento de entrada",
    "saida_texto": "Texto de saída",
    "saida_reacao": "Reação de saída",
    "saida_contexto": "Contexto de saída",
}


def _normalizar_rotulo_campo(texto: str) -> str:
    t = texto.strip().lower()
    for acentuado, sem_acento in (("ã", "a"), ("á", "a"), ("â", "a"), ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"), ("õ", "o"), ("ô", "o"), ("ú", "u"), ("ç", "c")):
        t = t.replace(acentuado, sem_acento)
    return normalize_collapse_spaces(t)


_ROTULOS_CAMPO_BONITO_NORMALIZADOS = {_normalizar_rotulo_campo(v): k for k, v in _ROTULOS_CAMPO_BONITO.items()}


def encontrar_campo_editavel_por_rotulo(texto: str) -> Optional[str]:
    """Acha o campo interno (entrada_texto, saida_reacao, etc.) a partir do
    nome bonito que a criadora digitou (ex.: "Texto de entrada") -- tolera
    maiúscula/minúscula e acento, mas nunca adivinha por aproximação: ou
    bate exatamente com um rótulo conhecido, ou não bate com nada."""
    return _ROTULOS_CAMPO_BONITO_NORMALIZADOS.get(_normalizar_rotulo_campo(texto))


# Campos que o fluxo livre de criação (2026-09-13, pedido direto da Thaís)
# deixa escolher por nome -- entrada texto/reação ficam de fora porque já
# chegam juntas na mensagem crua, nunca precisam ser "escolhidas".
_CAMPOS_CRIACAO_VALIDOS = {"entrada_contexto", "entrada_pensamento", "saida_texto", "saida_reacao", "saida_contexto"}


def _lista_campos_criacao() -> str:
    return "\n".join(f"- {_ROTULOS_CAMPO_BONITO[c]}" for c in ("entrada_contexto", "entrada_pensamento", "saida_texto", "saida_reacao", "saida_contexto"))


# Vars (variação de UMA palavra/reação) e Multivars (frase/reação inteira
# alternativa) só existem pra estes 4 campos -- contexto e pensamento NUNCA
# têm var, por definição (mesma regra já usada em insepa_tela.py).
_CAMPOS_VAR_VALIDOS = {"entrada_texto", "entrada_reacao", "saida_texto", "saida_reacao"}


def adicionar_var_bloco(memoria: dict, inconsciente: dict, dominio: str, bloco_id: int, campo: str, palavra: str, novas_vars: List[str]) -> bool:
    """Registra Vars (variações de UMA palavra ou reação) pro campo indicado
    -- igual ao 'Olá' -> ['Oi','Oie',...] que já existe no bloco 1, só que
    agora alimentado direto pela conversa (2026-09-13, pedido direto da
    Thaís: Vars cobre as variações dos termos de quem fala). Vars vivem no
    inconsciente (marcador -> {token, vars}), nunca no bloco em si. Devolve
    False se não achar a palavra/reação de verdade nesse campo do bloco."""
    bloco = obter_bloco_por_id(memoria, dominio, bloco_id)
    if bloco is None or campo not in _CAMPOS_VAR_VALIDOS:
        return False
    # 2026-09-13, pedido direto da Thaís: texto de entrada/saída viraram 3
    # famílias de marcador cada (TEXE/FADEN/TEFE, TEDSA/FADES/TEFSA) -- a
    # palavra pode estar em qualquer uma das 3, então procura nas 3 juntas.
    secao_nome, campos_tokens_keys = {
        "entrada_texto": ("entrada", ("TEXE", "FADEN", "TEFE")), "entrada_reacao": ("entrada", ("RE",)),
        "saida_texto": ("saida", ("TEDSA", "FADES", "TEFSA")), "saida_reacao": ("saida", ("RS",)),
    }[campo]
    secao = bloco["entrada"] if secao_nome == "entrada" else bloco["saidas"][0]
    marcadores = [m for k in campos_tokens_keys for m in secao.get("tokens", {}).get(k, [])]
    inco_universo = inconsciente.get("INCO", {}).get(dominio, {})
    bloco_inco = next((b for b in inco_universo.get("Blocos", []) if b.get("Bloco_id") == str(bloco_id)), None)
    if bloco_inco is None:
        return False
    inco_campo = bloco_inco["Entrada"] if secao_nome == "entrada" else bloco_inco["SAÍDA"]
    if campo.endswith("_reacao"):
        if not marcadores:
            return False
        marcador_alvo = marcadores[0]
    else:
        marcador_alvo = next((m for m in marcadores if inco_campo.get(m, {}).get("token") == palavra), None)
        if marcador_alvo is None:
            return False
    entrada_atual = inco_campo.get(marcador_alvo, {"token": palavra, "vars": []})
    vars_existentes = [v for v in entrada_atual.get("vars", []) if v != "0.0"]
    for v in novas_vars:
        if v not in vars_existentes:
            vars_existentes.append(v)
    entrada_atual["token"] = entrada_atual.get("token", palavra)
    entrada_atual["vars"] = vars_existentes or ["0.0"]
    inco_campo[marcador_alvo] = entrada_atual
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
    return True


def adicionar_multivar_bloco(memoria: dict, dominio: str, bloco_id: int, campo: str, novo_valor: str) -> bool:
    """Registra uma Multivar (frase ou reação inteira alternativa) direto no
    campo do bloco -- Multivars_Texto_Entrada/Saida ou Multivars_Reacao_
    Entrada/Saida, já suportados na estrutura, agora alimentados pela
    conversa (2026-09-13, pedido direto da Thaís: Multivars é o Adam
    variando a própria resposta sem fugir do assunto)."""
    bloco = obter_bloco_por_id(memoria, dominio, bloco_id)
    if bloco is None or campo not in _CAMPOS_VAR_VALIDOS:
        return False
    secao_nome, campo_lista = {
        "entrada_texto": ("entrada", "Multivars_Texto_Entrada"),
        "entrada_reacao": ("entrada", "Multivars_Reacao_Entrada"),
        "saida_texto": ("saida", "Multivars_Texto_Saida"),
        "saida_reacao": ("saida", "Multivars_Reacao_Saida"),
    }[campo]
    secao = bloco["entrada"] if secao_nome == "entrada" else bloco["saidas"][0]
    lista = secao.setdefault(campo_lista, [])
    if novo_valor not in lista:
        lista.append(novo_valor)
    salvar_json(ARQUIVO_MEMORIA, memoria)
    return True


def obter_bloco_por_id(memoria: dict, dominio: str, bloco_id: int) -> Optional[dict]:
    """Acha um bloco pelo id dentro de um universo -- usado tanto pra mostrar
    o estado atual antes de editar quanto pra aplicar a edição."""
    blocos = memoria.get("IM", {}).get(dominio, {}).get("blocos", [])
    return next((b for b in blocos if b["bloco_id"] == bloco_id), None)


def editar_bloco(memoria: dict, inconsciente: dict, dominio: str, bloco_id: int, mudancas: Dict[str, str]) -> dict:
    """Aplica mudanças de conteúdo a um bloco já existente (campos em
    _CAMPOS_EDITAVEIS_BLOCO) e reconstrói os marcadores do universo inteiro do
    zero via renumerar_marcadores_universo -- nunca deixa um marcador velho e
    um novo coexistindo fora de ordem. Vars/multivars do(s) campo(s) mudado(s)
    são resetados de propósito (não fazem mais sentido pro texto novo); campos
    intocados e blocos vizinhos (que só são renumerados, não editados) mantêm
    os vars como estavam."""
    bloco = obter_bloco_por_id(memoria, dominio, bloco_id)
    if bloco is None:
        raise ValueError(f"Bloco {bloco_id} não existe no universo {dominio}")

    campos_reset: Set[str] = set()
    for campo, novo_valor in mudancas.items():
        if campo not in _CAMPOS_EDITAVEIS_BLOCO:
            raise ValueError(f"Campo desconhecido: {campo!r}")
        secao, chave, campo_marcador = _CAMPOS_EDITAVEIS_BLOCO[campo]
        alvo = bloco["entrada"] if secao == "entrada" else bloco["saidas"][0]
        if chave == "textos":
            alvo["textos"] = [novo_valor]
        else:
            alvo[chave] = novo_valor
        campos_reset.update(campo_marcador)

    renumerar_marcadores_universo(memoria, inconsciente, dominio, resetar_vars={bloco_id: campos_reset})
    salvar_json(ARQUIVO_MEMORIA, memoria)
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
    return bloco


def formatar_bloco_atual_para_edicao(bloco: dict) -> str:
    """Mostra os campos editáveis de um bloco antes da edição, com o nome
    bonito de cada campo -- pra criadora ver exatamente o valor atual antes
    de decidir o que mudar."""
    e, s = bloco["entrada"], bloco["saidas"][0]
    valores = {
        "entrada_texto": e["texto"],
        "entrada_reacao": e.get("reacao", ""),
        "entrada_contexto": e.get("contexto", ""),
        "entrada_pensamento": e.get("pensamento_interno", ""),
        "saida_texto": s["textos"][0],
        "saida_reacao": s.get("reacao", ""),
        "saida_contexto": s.get("contexto", ""),
    }
    linhas = [f'📥 {_ROTULOS_CAMPO_BONITO[c]}: "{valores[c]}"' for c in ("entrada_texto", "entrada_reacao", "entrada_contexto", "entrada_pensamento")]
    linhas += [f'📤 {_ROTULOS_CAMPO_BONITO[c]}: "{valores[c]}"' for c in ("saida_texto", "saida_reacao", "saida_contexto")]
    return "\n".join(linhas)


def criar_bloco_concreto(
    memoria: dict, dominio: str, texto: str, reacao: str, contexto: str, pensamento: str,
    saida_texto: str, saida_reacao: str, meta: dict,
    entrada_re_vars: Optional[List[str]] = None, entrada_re_multivars: Optional[List[str]] = None,
    saida_contexto: str = "",
) -> dict:
    """Cria um bloco de dado concreto de verdade: gera os marcadores (Camada 1,
    locais ao universo), persiste em memória + inconsciente (Mongo + arquivo),
    e devolve o bloco criado. Usada tanto por submenu_opinioes (confirmar uma
    opinião) quanto pelo modo criadora no chat (classificar fato -> bloco).

    Nunca é chamada sozinha por dado ainda não confirmado -- quem decide os
    valores de contexto/pensamento antes de chegar aqui é sempre a criadora."""
    placeholder = f"{dominio}.0"
    contexto = contexto.strip() or placeholder
    pensamento = pensamento.strip() or placeholder
    saida_texto = saida_texto.strip() or placeholder
    entrada_re_vars = entrada_re_vars or []
    entrada_re_multivars = entrada_re_multivars or []

    memoria.setdefault("IM", {}).setdefault(dominio, {"nome": f"IM_{dominio}", "ultimo_child": f"{dominio}.0", "blocos": []})
    universo = memoria["IM"][dominio]
    # "Confirmado pela criadora" é STATUS/proveniência, nunca contexto de
    # verdade -- contexto da saída agora é perguntado de verdade (ex.: "Como
    # está?" -> "Informação do humor"); a tag de confirmação mora no
    # pensamento_interno da saída, nunca mais ocupando o lugar do contexto.
    saida_contexto = saida_contexto.strip() or placeholder
    saida_pensamento_texto = "Confirmado pela criadora"
    next_id = len(universo["blocos"]) + 1

    novo_bloco = {
        "bloco_id": next_id,
        "entrada": {
            "texto": texto,
            "Multivars_Texto_Entrada": [],
            "reacao": reacao,
            "Multivars_Reacao_Entrada": entrada_re_multivars,
            "contexto": contexto,
            "pensamento_interno": pensamento,
            "tokens": {},
            "fim": "",
        },
        "saidas": [{
            "textos": [saida_texto],
            "Multivars_Texto_Saida": [],
            "reacao": saida_reacao,
            "Multivars_Reacao_Saida": [],
            "contexto": saida_contexto,
            "pensamento_interno": saida_pensamento_texto,
            "tokens": {},
            "fim": "",
        }],
        "meta": meta,
    }

    # Camada 1: marcador é local do universo (dominio.X) -- a ligação com o
    # universo 0, quando o dado foi emprestado, fica registrada no meta, não
    # em compartilhar o mesmo prefixo.
    #
    # 2026-09-13, pedido direto da Thaís: texto de entrada/saída deixa de
    # ser um campo só (E/S) -- todo texto agora se divide em 3 famílias de
    # marcador pelo travessão "—" (convenção "— fala — comentário"):
    # TEXE/FADEN/TEFE na entrada (FADEN = fala do usuário), TEDSA/FADES/
    # TEFSA na saída (FADES = fala do Adam). Pensamento de personagem entre
    # aspas fica dentro do texto normal (TEXE/TEFE), nunca ganha marcador
    # à parte. Sem travessão nenhum, TEXE/TEDSA carrega o texto inteiro e
    # FADEN/TEFE (ou FADES/TEFSA) ficam vazios -- todo bloco usa essa
    # estrutura de 3 partes, mesmo sem fala nenhuma.
    #
    # Reação (RE/RS) nunca passa pelo Token() geral -- o campo inteiro é UM
    # símbolo atômico por definição (seja um emoji, uma carinha feita de
    # vários caracteres tipo ":3", ou uma palavra), nunca uma sequência a ser
    # separada letra por letra. Isso vale pro valor fonte e pra qualquer var
    # dele -- reação nunca é "texto comum" tokenizável.
    texe_str, faden_str, tefe_str = dividir_travessao(texto)
    TEXE = Token(texe_str)
    FADEN = Token(faden_str)
    TEFE = Token(tefe_str)
    RE = [reacao] if reacao else []
    CE = Token(contexto)
    # PIDE guarda só os 3 primeiros tokens do pensamento -- o texto completo
    # continua 100% preservado à parte, em pensamento_interno (string crua,
    # sem marcador nenhum). Corrigido em 2026-09-13 por instrução direta da
    # Thaís: marcador reservado sem token visível quebra a Camada 1 --
    # "chegou no fim de uma sequência, inicia a outra", sempre, sem gap.
    PIDE = Token(pensamento)[:3]
    tedsa_str, fades_str, tefsa_str = dividir_travessao(saida_texto)
    TEDSA = Token(tedsa_str)
    FADES = Token(fades_str)
    TEFSA = Token(tefsa_str)
    RS = [saida_reacao] if saida_reacao else []
    CS = Token(saida_contexto)

    current_last = universo["ultimo_child"]
    ent_marks = generate_markers(current_last, len(TEXE) + len(FADEN) + len(TEFE) + len(RE) + len(CE) + len(PIDE))
    out_marks = generate_markers(ent_marks[-1], len(TEDSA) + len(FADES) + len(TEFSA) + len(RS) + len(CS))

    idx = 0
    TEXE_m = ent_marks[idx: idx + len(TEXE)]; idx += len(TEXE)
    FADEN_m = ent_marks[idx: idx + len(FADEN)]; idx += len(FADEN)
    TEFE_m = ent_marks[idx: idx + len(TEFE)]; idx += len(TEFE)
    RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
    CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
    PIDE_m = ent_marks[idx: idx + len(PIDE)]
    jdx = 0
    TEDSA_m = out_marks[jdx: jdx + len(TEDSA)]; jdx += len(TEDSA)
    FADES_m = out_marks[jdx: jdx + len(FADES)]; jdx += len(FADES)
    TEFSA_m = out_marks[jdx: jdx + len(TEFSA)]; jdx += len(TEFSA)
    RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
    CS_m = out_marks[jdx: jdx + len(CS)]

    novo_bloco["entrada"]["tokens"] = {
        "TEXE": TEXE_m, "FADEN": FADEN_m, "TEFE": TEFE_m,
        "RE": RE_m, "CE": CE_m, "PIDE": PIDE_m, "TOTAL": ent_marks,
    }
    novo_bloco["entrada"]["fim"] = ent_marks[-1]
    novo_bloco["saidas"][0]["tokens"] = {
        "TEDSA": TEDSA_m, "FADES": FADES_m, "TEFSA": TEFSA_m,
        "RS": RS_m, "CS": CS_m, "TOTAL": out_marks,
    }
    novo_bloco["saidas"][0]["fim"] = out_marks[-1]

    universo["blocos"].append(novo_bloco)
    universo["ultimo_child"] = out_marks[-1]

    # Dado concreto confirmado pela criadora sempre persiste na hora -- nunca
    # depende do checkbox de auto-save. Mongo + arquivo local direto.
    st.session_state.memoria = memoria
    salvar_json(ARQUIVO_MEMORIA, memoria)
    st.session_state.alnulu_cache = build_alnulu_cache(memoria)

    # Espelha no inconsciente pra manter Bloco_id/marcadores consistentes entre os dois arquivos
    inconsciente = st.session_state.inconsciente
    inconsciente.setdefault("INCO", {}).setdefault(dominio, {"NOME": universo.get("nome", f"IM_{dominio}"), "Ultimo child": current_last, "Blocos": []})
    vars_re_registradas = entrada_re_vars if entrada_re_vars else ["0.0"]
    entrada_inco = {m: {"token": t, "vars": ["0.0"]} for m, t in zip(TEXE_m + FADEN_m + TEFE_m + CE_m + PIDE_m, TEXE + FADEN + TEFE + CE + PIDE)}
    for m, t in zip(RE_m, RE):
        entrada_inco[m] = {"token": t, "vars": vars_re_registradas}
    bloco_inco = {
        "Bloco_id": str(next_id),
        "Entrada": entrada_inco,
        "SAÍDA": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(TEDSA_m + FADES_m + TEFSA_m + RS_m + CS_m, TEDSA + FADES + TEFSA + RS + CS)},
    }
    inconsciente["INCO"][dominio]["Blocos"].append(bloco_inco)
    inconsciente["INCO"][dominio]["Ultimo child"] = out_marks[-1]
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

    return novo_bloco


def renderizar_reforco_autonomo_no_chat(bloco_id, resposta_texto: str, chave_sufixo: str) -> None:
    """Botões de reforço humano pra uma tentativa autônoma -- alimenta
    record_human_reinforcement (pesos 66/12/23 já existentes no código,
    criadora sempre priorizada). Nunca cria bloco novo, só reforça (ou
    enfraquece) a confiança no bloco que foi reaproveitado."""
    # Mesmas carinhas que o Adam já usa em outro lugar do código pra
    # gostei/não gostei (❤️‍🔥/💔) -- cara dele, não um "sim/não" genérico.
    col1, col2 = st.columns(2)
    if col1.button("❤️‍🔥 Incrível! Acertou!", key=f"reforco_gosto_{chave_sufixo}", width="stretch"):
        atualizado = record_human_reinforcement(bloco_id, resposta_texto, liked=True, human_name="Thaís D' Mariano")
        st.success(f"Confiança nesse bloco agora: {atualizado['reinforcement']['confidence_score']:.0%}")
    if col2.button("💔 Sinto muito. Mas errou.", key=f"reforco_nao_gosto_{chave_sufixo}", width="stretch"):
        atualizado = record_human_reinforcement(bloco_id, resposta_texto, liked=False, human_name="Thaís D' Mariano")
        st.info(f"Anotado -- confiança nesse bloco agora: {atualizado['reinforcement']['confidence_score']:.0%}")


def formatar_bloco_para_confirmacao(bloco: dict) -> str:
    """Resumo completo do bloco recém-criado, pra criadora ver exatamente o
    que ficou gravado antes de confirmar -- nunca esconder detalhe nenhum
    (o mesmo espírito de mostrar o palpite cru: transparência em vez de
    confiar cegamente que gravou certo)."""
    e = bloco["entrada"]
    s = bloco["saidas"][0]
    return (
        f'📥 **Entrada:** "{e["texto"]}" {e.get("reacao", "")}\n'
        f'　Contexto: {e.get("contexto", "")}\n'
        f'　Pensamento: {e.get("pensamento_interno", "")}\n'
        f'📤 **Saída:** "{s["textos"][0]}" {s.get("reacao", "")}\n'
        f'　Contexto: {s.get("contexto", "")}'
    )


def apagar_ultimo_bloco_criado(memoria: dict, dominio: str, bloco_id) -> bool:
    """Remove um bloco recém-criado (memória + inconsciente) quando a
    criadora diz que não era isso -- nunca rebobina o marcador (Camada 1:
    posição já usada nunca volta a ser livre, mesmo pro bloco que sumiu),
    só tira o bloco da lista. Devolve False se não achar mais nada pra
    apagar (ex.: ela já mexeu em outra coisa)."""
    universo = memoria.get("IM", {}).get(dominio, {})
    blocos = universo.get("blocos", [])
    alvo = next((b for b in blocos if b.get("bloco_id") == bloco_id), None)
    if alvo is None:
        return False
    blocos.remove(alvo)
    st.session_state.memoria = memoria
    salvar_json(ARQUIVO_MEMORIA, memoria)

    inconsciente = st.session_state.get("inconsciente", {})
    blocos_inco = inconsciente.get("INCO", {}).get(dominio, {}).get("Blocos", [])
    alvo_inco = next((bi for bi in blocos_inco if bi.get("Bloco_id") == str(bloco_id)), None)
    if alvo_inco is not None:
        blocos_inco.remove(alvo_inco)
        salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

    st.session_state.alnulu_cache = build_alnulu_cache(memoria)
    return True


def _finalizar_criacao_livre(memoria: dict) -> str:
    """Fecha o bloco no fluxo livre por nome de campo (2026-09-13, pedido
    direto da Thaís: contexto/pensamento/saída deixam de ser uma sequência
    fixa e obrigatória -- ela escolhe o campo pelo nome, na ordem que
    quiser, e finaliza quando decidir). Campo não preenchido vira
    placeholder, igual contexto/pensamento já faziam antes; reação da
    saída só vira placeholder se ela confirmar isso explicitamente (ver
    'confirmar_finalizar_sem_reacao_saida' -- nunca em silêncio)."""
    dominio_criacao = st.session_state["criadora_dominio"]
    placeholder = f"{dominio_criacao}.0"
    if not st.session_state.get("criadora_contexto", "").strip():
        st.session_state.criadora_contexto = placeholder
    if not st.session_state.get("criadora_pensamento", "").strip():
        st.session_state.criadora_pensamento = placeholder
    saida_txt = st.session_state.pop("criadora_saida_texto_valor", "") or placeholder
    saida_reac = st.session_state.pop("criadora_saida_reacao_valor", "") or placeholder
    saida_contexto = st.session_state.pop("criadora_saida_contexto_valor", "")
    return finalizar_bloco_criadora(memoria, saida_txt, saida_reac, saida_contexto)


def finalizar_bloco_criadora(memoria: dict, saida_txt: str, saida_reac: str, saida_contexto: str = "") -> str:
    """Fecha o bloco do modo criadora (chat) com a saída já decidida -- reaproveitada
    do candidato ou digitada do zero, tanto faz. Lê o resto (texto/reação/contexto/
    pensamento) do session_state, cria o bloco, liga a opinião de origem se houver,
    e devolve o resumo completo pedindo confirmação -- "era isso que você quis
    dizer?" -- em vez de só um "guardei" sem mostrar nada (pedido dela).

    `saida_contexto`: contexto de verdade da RESPOSTA (ex.: "Informação do
    humor" pra "Como está?") -- perguntado à parte, nunca mais preenchido
    sozinho com "Confirmado pela criadora" (isso agora é status, vive no
    pensamento_interno da saída)."""
    op_id = st.session_state.get("criadora_opiniao_id")
    dominio_bloco = st.session_state["criadora_dominio"]
    meta = {"origem": "modo_criadora"}
    if op_id:
        meta["opiniao_id"] = op_id
    # Livro de Lux (2026-09-13, hierarquia de 3 níveis corrigida pela Thaís):
    # enquanto um capítulo está em andamento (dentro de uma história/arco),
    # todo bloco completado leva historia_id + capitulo_id + a posição dele
    # dentro do CAPÍTULO -- é isso que deixa reconstruir a história inteira
    # depois, capítulo por capítulo, na ordem certa (o conjunto de
    # parágrafos, nunca um sozinho).
    if st.session_state.pop("criadora_eh_lux", False):
        historia_id = garantir_historia_lux_ativa()
        capitulo_id = garantir_capitulo_lux_ativo()
        blocos_do_capitulo = [
            b for b in memoria.get("IM", {}).get(dominio_bloco, {}).get("blocos", [])
            if b.get("meta", {}).get("capitulo_id") == capitulo_id
        ]
        meta["historia_id"] = historia_id
        meta["capitulo_id"] = capitulo_id
        meta["capitulo_posicao"] = len(blocos_do_capitulo) + 1
    novo_bloco = criar_bloco_concreto(
        memoria, dominio_bloco,
        st.session_state["criadora_txt"], st.session_state["criadora_reac"],
        st.session_state["criadora_contexto"], st.session_state["criadora_pensamento"],
        saida_txt, saida_reac, meta,
        saida_contexto=saida_contexto,
    )
    if op_id:
        opinioes_todas = carregar_opinioes()
        op_alvo = next((o for o in opinioes_todas if o["id"] == op_id), None)
        if op_alvo:
            op_alvo["status"] = "confirmada"
            op_alvo["contexto"] = novo_bloco["entrada"]["contexto"]
            op_alvo["pensamento_interno"] = novo_bloco["entrada"]["pensamento_interno"]
            salvar_opinioes(opinioes_todas)
    # 2026-09-13, pedido direto da Thaís: todo bloco que fica pronto devolve
    # a própria entrada pro poço de sementes (frase curta ou parágrafo, ver
    # adicionar_bloco_ao_poco_sementes), pra poder ser puxada de novo depois.
    adicionar_bloco_ao_poco_sementes(novo_bloco)
    for key in ["criadora_step", "criadora_txt", "criadora_reac", "criadora_dominio",
                "criadora_pensamento", "criadora_contexto", "criadora_opiniao_id",
                "criadora_saida_sugerida_texto", "criadora_saida_sugerida_reacao",
                "criadora_saida_sugerida_contexto", "criadora_saida_texto_pendente",
                "criadora_saida_reac_pendente", "criadora_contexto_saida_reusado"]:
        st.session_state.pop(key, None)

    st.session_state.criadora_step = "confirmar_bloco_criado"
    st.session_state.criadora_bloco_criado_id = novo_bloco["bloco_id"]
    st.session_state.criadora_bloco_criado_dominio = dominio_bloco
    resumo = formatar_bloco_para_confirmacao(novo_bloco)
    return f'Aeee, guardei isso! 💚 (bloco #{novo_bloco["bloco_id"]})\n\n{resumo}\n\nEra isso que você queria dizer? 🥹'


def _reacoes_aceitas_do_bloco(bloco: dict, dominio: str) -> Set[str]:
    """A reação canônica do bloco, mais qualquer VAR (palavra/emoji) já
    registrada pra ela no inconsciente, mais qualquer MULTIVAR (frase) já
    registrada em Multivars_Reacao_Entrada -- Camada 5: registrado RESOLVE pro
    mesmo dado, nunca é "parecido", só porque a string literal difere. Sem
    isso, ":3" (var) ou "Rosto sorridente" (multivar) de "😊" nunca contavam
    como bater exato."""
    reacao_canonica = bloco["entrada"].get("reacao", "")
    aceitas = {reacao_canonica} if reacao_canonica else set()
    aceitas.update(m for m in bloco["entrada"].get("Multivars_Reacao_Entrada", []) if m)

    marcadores_re = bloco["entrada"].get("tokens", {}).get("RE", [])
    if not marcadores_re:
        return aceitas
    inconsciente = st.session_state.get("inconsciente", {})
    blocos_inco = inconsciente.get("INCO", {}).get(dominio, {}).get("Blocos", [])
    bloco_inco = next((bi for bi in blocos_inco if bi.get("Bloco_id") == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return aceitas
    for marcador in marcadores_re:
        dado = bloco_inco.get("Entrada", {}).get(marcador)
        if dado:
            aceitas.update(v for v in dado.get("vars", []) if v != "0.0")
    return aceitas


def _texto_aceito_pelo_bloco(texto_candidato: str, bloco: dict, dominio: str, campo: str = "entrada") -> bool:
    """Confere se texto_candidato bate com o texto do bloco palavra por
    palavra, aceitando em cada posição a palavra canônica OU qualquer var já
    registrada pra aquele marcador -- exato e determinístico. Antes disso, o
    match usava UMA amostra aleatória de variar_texto() pra testar, então uma
    var registrada só batia por sorte (ex.: "Oiii" registrada como var de
    "Olá" só era reconhecida ~1 em cada 4 tentativas, nunca sempre)."""
    # 2026-09-13, pedido direto da Thaís: texto de entrada/saída se divide em
    # 3 famílias de marcador pelo travessão (TEXE/FADEN/TEFE ou TEDSA/FADES/
    # TEFSA) -- os marcadores E os tokens canônicos precisam vir da MESMA
    # divisão, senão o travessão em si desalinha a comparação palavra a
    # palavra (Token() sozinho tokenizaria o "—" também, os marcadores não).
    if campo == "entrada":
        texto_canonico = bloco["entrada"]["texto"]
        tokens_e = bloco["entrada"].get("tokens", {})
        marcadores = tokens_e.get("TEXE", []) + tokens_e.get("FADEN", []) + tokens_e.get("TEFE", [])
        chave_inco = "Entrada"
    else:
        texto_canonico = bloco["saidas"][0]["textos"][0]
        tokens_s = bloco["saidas"][0].get("tokens", {})
        marcadores = tokens_s.get("TEDSA", []) + tokens_s.get("FADES", []) + tokens_s.get("TEFSA", [])
        chave_inco = "SAÍDA"

    antes_can, fala_can, depois_can = dividir_travessao(texto_canonico)
    tokens_canonicos = Token(antes_can) + Token(fala_can) + Token(depois_can)
    antes_cand, fala_cand, depois_cand = dividir_travessao(texto_candidato)
    tokens_candidato = Token(antes_cand) + Token(fala_cand) + Token(depois_cand)
    if len(tokens_canonicos) != len(tokens_candidato):
        return False

    inconsciente = st.session_state.get("inconsciente", {})
    blocos_inco = inconsciente.get("INCO", {}).get(dominio, {}).get("Blocos", [])
    bloco_inco = next((bi for bi in blocos_inco if bi.get("Bloco_id") == str(bloco["bloco_id"])), None)

    for tok_can, tok_cand, marcador in zip(tokens_canonicos, tokens_candidato, marcadores):
        if normalize(tok_cand) == normalize(tok_can):
            continue
        vars_aqui: Set[str] = set()
        if bloco_inco:
            dado = bloco_inco.get(chave_inco, {}).get(marcador)
            if dado:
                vars_aqui = {v for v in dado.get("vars", []) if v != "0.0"}
        if not any(normalize(tok_cand) == normalize(v) for v in vars_aqui):
            return False
    return True


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
        if any(normalize(t) == normalize(txt) for t in textos_possiveis) and reac in _reacoes_aceitas_do_bloco(b, dominio):
            return b
    return None


def harness_run_case(dominio: str, raw_input: str) -> Dict[str, Any]:
    """A mesma lógica do adam_harness.py (match exato -> reflexão -> opinião),
    rodando dentro do próprio app, pra criadora testar direto do navegador."""
    txt, reac = parse_text_reaction(raw_input)
    resultado: Dict[str, Any] = {"dominio": dominio, "input": raw_input, "tokens": Token(raw_input)}

    def texto_com_reacao(saida: dict) -> str:
        texto = saida["textos"][0]
        reacao = saida.get("reacao", "")
        return texto + (" " + reacao if reacao else "")

    bloco_exato = harness_find_exact_match(dominio, txt, reac)
    if bloco_exato:
        resultado.update({"status": "match_exato", "bloco_id": bloco_exato["bloco_id"], "resposta": texto_com_reacao(bloco_exato["saidas"][0])})
        return resultado

    candidato = melhor_candidato_fraco(txt, dominio, reac)
    if candidato:
        b = candidato["bloco"]
        resultado.update({
            "status": "reflexao",
            "candidato_bloco_id": b["bloco_id"],
            "candidato_score": candidato["score"],
            "resposta_tentativa": texto_com_reacao(b["saidas"][0]),
        })
    else:
        resultado["status"] = "desconhecido"

    opiniao = registrar_opiniao_pendente(txt, reac, dominio, candidato)
    resultado["opiniao_id"] = opiniao["id"]
    return resultado


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
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng,
        out_vocab_floats=list(idx_to_txt.keys()),
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
    reacao_tokens = [reacao_base] if reacao_base else []  # reação é atômica -- nunca Token() geral
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
    status = get_learning_status_summary()
    if status["blocks"] > 0:
        # Só aparece depois que existir pelo menos 1 bloco com reforço de
        # verdade -- antes disso é só ruído gritando "0" toda vez que abre.
        st.info(
            f"🧠 Peso do aprendizado: criadora 66% | usuários 12% | autonomia do Adam 23%. "
            f"Autonomia pronta: {'sim' if status['autonomy_ready'] else 'não'} | blocos com reforço: {status['blocks']}"
        )

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

    # Sem distinção usuário/admin -- só a criadora usa isso até ela decidir abrir
    # pra mais gente. Sem senha, sem painel separado: é direto no chat. Salvar
    # também não é opção -- ver auto_save_state().

    st.write("Áudio disponível. Ouça a voz do personagem escolhido agora!")
    dom = prompt_dominio("conversar", memoria)
    if dom:
        if dom in memoria["IM"]:
            infer(memoria, dom)
        else:
            st.error(f"❌ Domínio '{dom}' não encontrado.")

    # Auto-save (se habilitado) em cada rerun (inclui entrada no site)
    auto_save_state()


if __name__ == "__main__":
    main()
