import streamlit as st
import json
import os
import re

# =============================================
# Passo 1: Criar o cabeçalho#
# [PASSO-1] Criar o cabeçalho
# =============================================
# Cabeçalho mínimo (sem CRUD, sem parser)
st.set_page_config(page_title="INSEPA — IDA/IM (CRUD)", layout="wide")
st.title("INSEPA — IDA/IM (CRUD: renomear)")

st.info("App mínimo carregado. Próximo passo: definir exatamente o que entra no cabeçalho.")

# ===== FIM do PASSO 1 =====


# =============================================
# Passo 2: CRUD de IM (renomear)#
# [PASSO-2] Renomear IM em inconsciente.json
# =============================================

# Caminho do arquivo (pasta pai de app_streamlit)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_OUT_IDA = os.path.join(_BASE_DIR, "inconsciente.json")
_OUT_UM = os.path.join(_BASE_DIR, "memoria.json")


# Carregar JSON simples (sem cache)
def _reload_data():
    """Recarrega os dados globais _ida_data e _um_data dos arquivos JSON no diretório pai.

    Tenta carregar inconsciente.json e memoria.json. Se não existirem, inicializa com estruturas vazias.
    Garante que as estruturas tenham as chaves mínimas {"IDA": {"IM": {}}} e {"UM": {}}.
    """
    global _ida_data, _um_data
    try:
        if os.path.exists(_OUT_IDA):
            with open(_OUT_IDA, "r", encoding="utf-8") as f:
                _ida_data = json.load(f) or {}
        else:
            _ida_data = {}
    except Exception:
        _ida_data = {}

    try:
        if os.path.exists(_OUT_UM):
            with open(_OUT_UM, "r", encoding="utf-8") as f:
                _um_data = json.load(f) or {}
        else:
            _um_data = {}
    except Exception:
        _um_data = {}

    # Garantir estrutura mínima {"IDA": {"IM": {}}}
    if not isinstance(_ida_data, dict):
        _ida_data = {}
    _ida_data.setdefault("IDA", {}).setdefault("IM", {})

    if not isinstance(_um_data, dict):
        _um_data = {}
    _um_data.setdefault("UM", {})


_reload_data()

st.subheader("IDA/IM — CRUD de IMs")
_ims = _ida_data["IDA"].get("IM", {})
_im_keys = sorted(_ims.keys(), key=lambda k: int(k) if str(k).isdigit() else float('inf'))

# Criar novo IM
st.markdown("#### Criar novo IM")
novo_im_key = st.text_input("Chave do novo IM (ex.: 1)", key="novo_im_key")
novo_im_nome = st.text_input("Nome do novo IM", key="novo_im_nome")
if st.button("Criar novo IM", type="primary", key="btn_criar_im"):
    try:
        if novo_im_key.strip() and novo_im_nome.strip():
            if novo_im_key in _ims:
                st.error(f"IM {novo_im_key} já existe.")
            else:
                _ida_data["IDA"]["IM"][novo_im_key] = {"nome": novo_im_nome.strip(), "blocos": []}
                _um_data["UM"][novo_im_key] = {"Universo Mãe": novo_im_nome.strip(), "blocos": []}
                os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                with open(_OUT_IDA, "w", encoding="utf-8") as f:
                    json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                with open(_OUT_UM, "w", encoding="utf-8") as f:
                    json.dump(_um_data, f, ensure_ascii=False, indent=2)
                st.success(f"IM {novo_im_key} criado com sucesso.")
                _reload_data()
                st.rerun()  # Recarregar para atualizar a lista
        else:
            st.error("Preencha a chave e o nome do IM.")
    except Exception as e:
        st.error(f"Falha ao criar IM: {e}")

# Renomear IM existente
st.markdown("#### Renomear IM existente")
if not _im_keys:
    st.info("Nenhum IM encontrado no inconsciente.json.")
else:
    sel_im = st.selectbox("Selecione o IM para renomear", options=_im_keys, key="sel_im")
    cur_nome = (_ims.get(sel_im) or {}).get("nome", "")
    st.caption(f"Chave selecionada: {sel_im}")
    novo_nome = st.text_input("Novo nome do IM", value=cur_nome, key=f"novo_nome_{sel_im}")
    if st.button("Salvar novo nome", type="primary", key=f"btn_salvar_{sel_im}"):
        try:
            _ida_data["IDA"]["IM"].setdefault(sel_im, {})
            _ida_data["IDA"]["IM"][sel_im]["nome"] = (novo_nome or "").strip()
            _um_data["UM"].setdefault(sel_im, {})
            _um_data["UM"][sel_im]["Universo Mãe"] = (novo_nome or "").strip()
            os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
            with open(_OUT_IDA, "w", encoding="utf-8") as f:
                json.dump(_ida_data, f, ensure_ascii=False, indent=2)
            with open(_OUT_UM, "w", encoding="utf-8") as f:
                json.dump(_um_data, f, ensure_ascii=False, indent=2)
            st.success("Nome atualizado com sucesso.")
        except Exception as e:
            st.error(f"Falha ao salvar inconsciente.json: {e}")

# Deletar IM
st.markdown("#### Deletar IM")
if _im_keys:
    del_im = st.selectbox("Selecione o IM para deletar", options=_im_keys, key="del_im")
    if st.button("Deletar IM selecionado", type="secondary", key=f"btn_deletar_{del_im}"):
        try:
            confirm = st.checkbox(f"Confirmar exclusão do IM {del_im}?", key=f"confirm_del_{del_im}")
            if confirm:
                if del_im in _ida_data["IDA"]["IM"]:
                    del _ida_data["IDA"]["IM"][del_im]
                if del_im in _um_data["UM"]:
                    del _um_data["UM"][del_im]
                os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                with open(_OUT_IDA, "w", encoding="utf-8") as f:
                    json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                with open(_OUT_UM, "w", encoding="utf-8") as f:
                    json.dump(_um_data, f, ensure_ascii=False, indent=2)
                st.success(f"IM {del_im} deletado com sucesso.")
                _reload_data()
                st.rerun()
            else:
                st.warning("Marque a confirmação para deletar.")
        except Exception as e:
            st.error(f"Falha ao deletar IM: {e}")
else:
    st.info("Nenhum IM para deletar.")

# Limpar blocos de IM
st.markdown("#### Limpar blocos de IM")
if _im_keys:
    clear_im = st.selectbox("Selecione o IM para limpar blocos", options=_im_keys, key="clear_im")
    if st.button("Limpar todos os blocos do IM selecionado", type="secondary", key=f"btn_clear_{clear_im}"):
        try:
            confirm_clear = st.checkbox(f"Confirmar limpeza de todos os blocos do IM {clear_im}?",
                                        key=f"confirm_clear_{clear_im}")
            if confirm_clear:
                if clear_im in _ida_data["IDA"]["IM"]:
                    _ida_data["IDA"]["IM"][clear_im]["blocos"] = []
                if clear_im in _um_data["UM"]:
                    _um_data["UM"][clear_im]["blocos"] = []
                os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                with open(_OUT_IDA, "w", encoding="utf-8") as f:
                    json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                with open(_OUT_UM, "w", encoding="utf-8") as f:
                    json.dump(_um_data, f, ensure_ascii=False, indent=2)
                st.success(f"Todos os blocos do IM {clear_im} foram limpos.")
                _reload_data()
            else:
                st.warning("Marque a confirmação para limpar.")
        except Exception as e:
            st.error(f"Falha ao limpar blocos: {e}")
else:
    st.info("Nenhum IM para limpar blocos.")


# ===== FIM do PASSO 2 =====


# =============================================
# Funções comuns para parsing e extração (refatoradas para reduzir redundância)
# =============================================

def _extract_text_parts(text: str, re_s: str = ""):
    """Extrai partes de texto de entrada/saída: antes do travessão (texe/texis), fala até o último ponto seguido de maiúscula (faden/fs), resto após (tefie/texfs).

    Args:
        text (str): Texto da entrada ou saída no formato "antes—fala. resto".
        re_s (str): Reação de saída, usada como delimitador para TEXFS.

    Returns:
        tuple: (antes, fala, resto) onde antes é texto inicial, fala é fala até último . antes de maiúscula, resto é texto final.
    """
    text = (text or "").strip()
    if not text:
        return "", "", ""
    idx = text.find("—")
    if idx < 0:
        return text, "", ""
    before = text[:idx].strip()
    after = text[idx + 1:].lstrip()
    if not after:
        return before, "", ""
    # Se há reação, dividir TEXFS após a reação
    if re_s and re_s in after:
        fala = after.split(re_s, 1)[0].strip()
        rest = after.split(re_s, 1)[1].strip()
    else:
        # Dividir after em sentenças baseadas em pontuação
        sentences = re.split(r'(?<=[.!?])\s+', after)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            fala = sentences[0]
            rest = ' '.join(sentences[1:]) if len(sentences) > 1 else ""
        else:
            fala = after.strip()
            rest = ""
    return before, fala, rest


def _parse_tpl_sections(raw: str):
    """Parseia o texto TPL bruto em seções estruturadas para processamento INSEPA.

    Divide o texto em componentes: entrada (TEXE, FADEN, TEFIE), reação, contexto,
    pensamento interno (PIDE), saída (TEXIS, FS, TEXFS), reação saída, contexto saída.

    Args:
        raw (str): Texto TPL bruto com marcadores de seção.

    Returns:
        dict: Dicionário com chaves 'entrada', 're_e', 'ce_e', 'pide', 'saida', 're_s', 'cs'.
              Cada valor é uma string com o texto correspondente.
    """
    s = (raw or "").strip() + "\n<END>:\n"

    def grab(name_pat: str, next_pats: list, start_pos: int = 0):
        nxt = "|".join([rf"^\s*{p}\s*:" for p in next_pats] + [r"^\s*<END>\s*:"])
        m = re.search(rf"(?ims)^\s*{name_pat}\s*:(.*?)(?={nxt})", s[start_pos:])
        if not m:
            return "", start_pos
        content = (m.group(1) or "").strip()
        end_pos = start_pos + m.end()
        return content, end_pos

    pos = 0
    entrada, pos = grab(r"Entrada", ["Reação", "Contexto", "Pensamento interno", "Saída"], pos)
    re_e, pos = grab(r"Reação", ["Contexto", "Pensamento interno", "Saída"], pos)
    ce_e, pos = grab(r"Contexto", ["Pensamento interno", "Saída"], pos)
    pide, pos = grab(r"Pensamento\s+interno", ["Saída"], pos)
    saida, pos = grab(r"Saída", ["Reação", "Contexto"], pos)
    re_s, pos = grab(r"Reação", ["Contexto"], pos)
    cs, pos = grab(r"Contexto", ["<END>"], pos)

    return {
        "entrada": entrada,
        "re_e": re_e,
        "ce_e": ce_e,
        "pide": pide,
        "saida": saida,
        "re_s": re_s,
        "cs": cs,
    }


def _scan_tokens(text: str):
    """Tokeniza o texto em palavras, pontuação e marcadores especiais.

    Args:
        text (str): Texto a ser tokenizado.

    Returns:
        list: Lista de dicionários com 'kind' ('word', 'punct', 'marker') e 'val'.
    """
    tokens = []
    if not text:
        return tokens
    pattern = r"(\[(?:Muden|Mudsa|Ade|Adsa)\]|\[(?:CAE|CAS)\s*:[^\]]+\])|([\wÀ-ÖØ-öø-ÿ']+(?:-[\wÀ-ÖØ-öø-ÿ']+)*)|([\.!?,;:—])"
    for m in re.finditer(pattern, text):
        if m.group(1):
            tokens.append({"kind": "marker", "val": m.group(1)})
        elif m.group(2):
            tokens.append({"kind": "word", "val": m.group(2)})
        else:
            tokens.append({"kind": "punct", "val": m.group(3)})
    return tokens


def _emit_tokens(
        scan,
        im_key: str,
        start_idx: int,
        dado: str,
        attr_key: str = None,
        span_marker: str = None,
        action_marker: str = None,
        in_span: bool = False,
        in_action_span: bool = False,
):
    """Emite tokens estruturados com IDs únicos para um campo específico.

    Args:
        scan (list): Lista de tokens do _scan_tokens.
        im_key (str): Chave do Índice Mãe.
        start_idx (int): Índice inicial para IDs.
        dado (str): Tipo de dado (ex.: 'TEXE').
        attr_key (str, optional): Chave para atributos (ex.: 'cae').
        span_marker (str, optional): Marcador para spans multivariações.
        action_marker (str, optional): Marcador para ações.
        in_span (bool): Se está em span.
        in_action_span (bool): Se está em action span.

    Returns:
        tuple: (out, idx, ids, spans, action_spans, in_span, in_action_span)
    """
    out = []
    ids = []
    spans = []
    action_spans = []
    idx = start_idx
    current_attr_out = None
    current_attr_norm = None
    span_tokens = []
    span_ids = []
    action_tokens = []
    action_ids = []

    i = 0
    while i < len(scan):
        tk = scan[i]
        if tk['kind'] == 'marker':
            s = tk['val']
            if re.match(r"\[(?:CAE|CAS)\s*:[^\]]+\]", s, flags=re.IGNORECASE):
                label_raw = s.split(":", 1)[1][:-1].strip()
                label_clean = re.sub(r"[\s\.:,;!?]+$", "", label_raw).strip()
                label_norm = label_clean.lower()
                if current_attr_norm == label_norm:
                    current_attr_out = None
                    current_attr_norm = None
                else:
                    current_attr_out = label_clean
                    current_attr_norm = label_norm
            elif span_marker and s.lower() == span_marker.lower():
                in_span = not in_span
                if not in_span and span_tokens:
                    spans.append({"Tokens": " ".join(span_tokens).strip(), "ids": span_ids[:]})
                    span_tokens, span_ids = [], []
            elif action_marker and s.lower() == action_marker.lower():
                in_action_span = not in_action_span
                if not in_action_span and action_tokens:
                    action_spans.append({"Tokens": " ".join(action_tokens).strip(), "ids": action_ids[:]})
                    action_tokens, action_ids = [], []
            i += 1
            continue

        idx += 1
        safe_im = re.sub(r"[^0-9]", "", str(im_key or "0"))
        tid = f"{safe_im}.{idx}"
        obj = {dado: tid, "t": tk['val'], "vars": ["0.0"]}
        if current_attr_out and attr_key:
            clean_attr = current_attr_out.lstrip(": ").strip()
            obj[attr_key] = [clean_attr]
        out.append(obj)
        if in_span:
            span_tokens.append(tk['val'])
            span_ids.append(tid)
        if in_action_span:
            action_tokens.append(tk['val'])
            action_ids.append(tid)
        i += 1

    if in_span and span_tokens:
        spans.append({"Tokens": " ".join(span_tokens).strip(), "ids": span_ids[:]})
    if in_action_span and action_tokens:
        action_spans.append({"Tokens": " ".join(action_tokens).strip(), "ids": action_ids[:]})

    return out, idx, ids, spans, action_spans, in_span, in_action_span


def _split_sentences(text: str):
    """Divide o texto em sentenças baseado em pontuação.

    Args:
        text (str): Texto a dividir.

    Returns:
        list: Lista de sentenças.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [p for p in parts if p]


def _build_um_from_attrs(tokens, attr_key: str, fonte: str, dado_key: str):
    """Constrói entradas de características para UM a partir de tokens com atributos.

    Args:
        tokens (list): Lista de tokens.
        attr_key (str): Chave do atributo (ex.: 'cae').
        fonte (str): Fonte (ex.: 'Texto de entrada').
        dado_key (str): Chave do dado (ex.: 'TEXE').

    Returns:
        list: Lista de entradas de características.
    """
    entries = []
    cur = None
    cur_tokens = []
    cur_ids = []
    for t in tokens or []:
        if attr_key in t:
            label = (t[attr_key][0] or "").strip()
            label = label[2:].strip() if label.startswith(":") else label
            tid = t.get(dado_key)
            if cur == label:
                cur_tokens.append(t['t'])
                if tid: cur_ids.append(tid)
            else:
                if cur and cur_tokens:
                    entries.append({attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key,
                                    "Tokens": " ".join(cur_tokens).strip(), dado_key: cur_ids[:]})
                cur = label
                cur_tokens = [t['t']]
                cur_ids = [tid] if tid else []
        else:
            if cur and cur_tokens:
                entries.append(
                    {attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key, "Tokens": " ".join(cur_tokens).strip(),
                     dado_key: cur_ids[:]})
            cur, cur_tokens, cur_ids = None, [], []
    if cur and cur_tokens:
        entries.append({attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key, "Tokens": " ".join(cur_tokens).strip(),
                        dado_key: cur_ids[:]})
    return entries


def _sanitize_ida_um(ida_block, um_block, im_key: str):
    """Sanitiza e completa os blocos IDA e UM com marcadores neutros e listas totais.

    Args:
        ida_block (dict): Bloco IDA.
        um_block (dict): Bloco UM.
        im_key (str): Chave do IM.

    Returns:
        tuple: (ida_block, um_block) sanitizados.
    """
    safe_im = re.sub(r"[^0-9]", "", str(im_key or "0"))
    neutral = f"{safe_im}.0"

    def ensure_neutral_list(objs, key_name):
        if objs:
            return objs
        return [{key_name: neutral, 't': '', 'vars': ['0.0']}]

    def ensure_neutral_ids(ids):
        return ids if ids else [neutral]

    ent_src = ida_block.get("Entrada", {})
    ent = {}
    ent["Texto Inicial DE ENTRADA"] = ensure_neutral_list(ent_src.get("Texto Inicial DE ENTRADA"), 'TEXE')
    ent["Fala DE ENTRADA"] = ensure_neutral_list(ent_src.get("Fala DE ENTRADA"), 'FADEN')
    ent["Texto Final de Entrada"] = ensure_neutral_list(ent_src.get("Texto Final de Entrada"), 'TEFIE')
    ent["Reação"] = ent_src.get("Reação") or [{"RE": neutral, "t": "", "vars": ["0.0"]}]
    ent["Contexto"] = ensure_neutral_list(ent_src.get("Contexto"), 'CE')
    ida_block["Entrada"] = ent

    sai_src = ida_block.get("Saída", {})
    sai = {}
    sai["Texto Inicial de SAÍDA"] = ensure_neutral_list(sai_src.get("Texto Inicial de SAÍDA"), 'TEXIS')
    sai["Fala de Saída"] = ensure_neutral_list(sai_src.get("Fala de Saída"), 'FS')
    sai["Texto Final de Saída"] = ensure_neutral_list(sai_src.get("Texto Final de Saída"), 'TEXFS')
    sai["Reação de Saída"] = sai_src.get("Reação de Saída") or [{"RS": neutral, "t": "", "vars": ["0.0"]}]
    sai["Contexto de Saída"] = ensure_neutral_list(sai_src.get("Contexto de Saída"), 'CS')
    ida_block["Saída"] = sai

    if not ida_block.get("Pensamento Interno"):
        ida_block["Pensamento Interno"] = [{"PIDE": neutral, "t": ""}]

    def ids_from(objs, key):
        res = []
        for o in objs or []:
            v = o.get(key)
            if v: res.append(v)
        return res

    total_e = [
        *ids_from(ent.get("Texto Inicial DE ENTRADA"), 'TEXE'),
        *ids_from(ent.get("Fala DE ENTRADA"), 'FADEN'),
        *ids_from(ent.get("Texto Final de Entrada"), 'TEFIE'),
        *[x.get('RE') for x in (ent.get("Reação") or []) if x.get('RE')],
        *ids_from(ent.get("Contexto"), 'CE'),
        *[x.get('PIDE') for x in (ida_block.get("Pensamento Interno") or []) if x.get('PIDE')],
    ]
    total_s = [
        *ids_from(sai.get("Texto Inicial de SAÍDA"), 'TEXIS'),
        *ids_from(sai.get("Fala de Saída"), 'FS'),
        *ids_from(sai.get("Texto Final de Saída"), 'TEXFS'),
        *[x.get('RS') for x in (sai.get("Reação de Saída") or []) if x.get('RS')],
        *ids_from(sai.get("Contexto de Saída"), 'CS'),
    ]

    def _id_num(s):
        m = re.search(r"\.([0-9]+)$", s or "")
        return int(m.group(1)) if m else 0

    ida_block["Total de Entrada"] = (sorted(total_e, key=_id_num) if total_e else [neutral])
    ida_block["Total de Saída"] = (sorted(total_s, key=_id_num) if total_s else [neutral])

    tot_e_for_um = ida_block.get("Total de Entrada", [])
    tot_s_for_um = ida_block.get("Total de Saída", [])

    try:
        sai_ordered = {}
        sai_section = ida_block.get("Saída", {})
        sai_ordered["Texto Inicial de SAÍDA"] = sai_section.get("Texto Inicial de SAÍDA", [])
        sai_ordered["Fala de Saída"] = sai_section.get("Fala de Saída", [])
        sai_ordered["Texto Final de Saída"] = sai_section.get("Texto Final de Saída", [])
        sai_ordered["Reação de Saída"] = sai_section.get("Reação de Saída", [])
        sai_ordered["Contexto de Saída"] = sai_section.get("Contexto de Saída", [])
        sai_ordered["Total de Saída"] = ida_block.get("Total de Saída", [])
        ida_ordered = {}
        ida_ordered["Entrada"] = ida_block.get("Entrada", {})
        ida_ordered["Pensamento Interno"] = ida_block.get("Pensamento Interno", [])
        ida_ordered["Total de Entrada"] = ida_block.get("Total de Entrada", [])
        ida_ordered["Saída"] = sai_ordered
        ida_block = ida_ordered
    except Exception:
        pass

    if um_block is not None:
        um_block["Total de Entrada"] = um_block.get("Total de Entrada") or tot_e_for_um
        um_block["Total de Saída"] = um_block.get("Total de Saída") or tot_s_for_um
        um_block["ultimo_child_entrada"] = um_block["Total de Entrada"][-1] if um_block.get(
            "Total de Entrada") else neutral
        um_block["ultimo_child_saida"] = um_block["Total de Saída"][-1] if um_block.get("Total de Saída") else neutral
    return ida_block, um_block


def _get_max_idx_in_im(im_data, im_key):
    """Retorna o maior índice usado no IM (ex.: 120 para 0.120).

    Args:
        im_data (dict): Dados do IM.
        im_key (str): Chave do IM.

    Returns:
        int: Maior idx.
    """
    max_idx = 0
    for bloco in im_data.get("blocos", []):
        for section in ["Entrada", "Saída"]:
            if section in bloco:
                for sub in bloco[section].values():
                    if isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, dict) and "vars" in item:
                                key = list(item.keys())[0]
                                if "." in key:
                                    parts = key.split(".")
                                    if len(parts) == 2 and parts[0] == str(im_key):
                                        try:
                                            idx = int(parts[1])
                                            max_idx = max(max_idx, idx)
                                        except ValueError:
                                            pass
    return max_idx


def _build_ida_um_from_tpl(secs: dict, im_key: str, start_idx: int = 0):
    """Constrói blocos IDA e UM a partir de seções TPL parseadas.

    Args:
        secs (dict): Seções parseadas do TPL.
        im_key (str): Chave do IM.
        start_idx (int): Índice inicial.

    Returns:
        tuple: (ida_block, um_block, idx) onde idx é o próximo índice.
    """

    def _clean_text(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"\[(?:Muden|Mudsa|Ade|Adsa)\]", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\[(?:CAE|CAS)\s*:[^\]]+\]", "", s, flags=re.IGNORECASE)
        s = _normalize_punct(s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _normalize_punct(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"\s+([,\.;:!?])", r"\1", s)
        s = re.sub(r"([,\.;:!?])(\S)", r"\1 \2", s)
        s = re.sub(r"\s+(\)|\]|\}|['\"])", r"\1", s)
        s = re.sub(r"^[\t ]+", "", s, flags=re.MULTILINE)
        return s

    def _collect_action_texts(raw: str, action_marker: str):
        texts = []
        scan = _scan_tokens(raw or "")
        in_act = False
        buf = []
        for tk in scan:
            if tk.get('kind') == 'marker' and tk.get('val', '').lower() == action_marker.lower():
                in_act = not in_act
                if not in_act and buf:
                    texts.append(" ".join(buf).strip())
                    buf = []
                continue
            if in_act and tk.get('kind') != 'marker':
                buf.append(tk.get('val', ''))
        if in_act and buf:
            texts.append(" ".join(buf).strip())
        return texts

    ent_before, ent_fala, ent_rest = _extract_text_parts(secs.get('entrada', ''), "")
    texe_scan = _scan_tokens(ent_before)
    faden_scan = _scan_tokens(ent_fala)
    tefie_scan = _scan_tokens(ent_rest)

    saida_before, saida_fala, saida_rest = _extract_text_parts(secs.get('saida', ''), secs.get('re_s', ''))
    texis_scan = _scan_tokens(saida_before)
    fs_scan = _scan_tokens(saida_fala)
    texfs_scan = _scan_tokens(saida_rest)

    ctx_e_scan = _scan_tokens(secs.get('ce_e', ''))
    ctx_s_scan = _scan_tokens(secs.get('cs', ''))
    re_e_val = (secs.get('re_e', '') or '').strip()
    re_s_val = (secs.get('re_s', '') or '').strip()

    im_key_clean = re.sub(r"[^0-9]", "", str(im_key or "0"))

    idx = start_idx
    TEXE, idx, ids_texe, mv_texe, act_texe, in_muden, in_ade = _emit_tokens(
        texe_scan, im_key_clean, idx, 'TEXE', 'cae', span_marker='[Muden]', action_marker='[Ade]', in_span=False,
        in_action_span=False
    )
    FADEN, idx, ids_faden, mv_faden, act_faden, in_muden, in_ade = _emit_tokens(
        faden_scan, im_key_clean, idx, 'FADEN', 'cae', span_marker='[Muden]', action_marker='[Ade]', in_span=in_muden,
        in_action_span=in_ade
    )
    ade_texts = []
    for s in (act_texe or []) + (act_faden or []):
        if s.get('Tokens'):
            ade_texts.append(s['Tokens'])
    ade_texts.extend([t for t in _collect_action_texts(ent_rest, '[Ade]') if t])
    ade_texts.extend([t for t in _collect_action_texts(secs.get('ce_e', ''), '[Ade]') if t])
    RE_list = []
    if re_e_val or ade_texts:
        idx += 1
        safe_im = im_key_clean
        re_t = re_e_val.strip() if re_e_val else ""
        vars_list = (ade_texts if ade_texts else ["0.0"])
        RE_list.append({"RE": f"{safe_im}.{idx}", "t": re_t, "vars": vars_list})
    TEFIE, idx, ids_tefie, mv_tefie, act_tefie, _, _ = _emit_tokens(
        tefie_scan, im_key_clean, idx, 'TEFIE', 'cae', span_marker='[Muden]', action_marker='[Ade]', in_span=in_muden,
        in_action_span=in_ade
    )
    CE_e, idx, ids_ce_e, _, act_ce_e, _, _ = _emit_tokens(
        ctx_e_scan, im_key_clean, idx, 'CE', None, span_marker='[Muden]', action_marker='[Ade]', in_span=False,
        in_action_span=False
    )

    PIDE = []
    for s in _split_sentences(secs.get('pide', '')):
        s2 = s.strip()
        if not s2:
            continue
        idx += 1
        safe_im = im_key_clean
        tid = f"{safe_im}.{idx}"
        PIDE.append({"PIDE": tid, "t": s2})

    TEXIS, idx, ids_texis, mv_texis, act_texis, in_mudsa, in_adsa = _emit_tokens(
        texis_scan, im_key_clean, idx, 'TEXIS', 'cas', span_marker='[Mudsa]', action_marker='[Adsa]', in_span=False,
        in_action_span=False
    )
    FS, idx, ids_fs, mv_fs, act_fs, in_mudsa, in_adsa = _emit_tokens(
        fs_scan, im_key_clean, idx, 'FS', 'cas', span_marker='[Mudsa]', action_marker='[Adsa]', in_span=in_mudsa,
        in_action_span=in_adsa
    )
    adsa_texts = []
    for s in (act_texis or []) + (act_fs or []):
        if s.get('Tokens'):
            adsa_texts.append(s['Tokens'])
    adsa_texts.extend([t for t in _collect_action_texts(saida_rest, '[Adsa]') if t])
    adsa_texts.extend([t for t in _collect_action_texts(secs.get('cs', ''), '[Adsa]') if t])
    RS_list = []
    if re_s_val or adsa_texts:
        idx += 1
        safe_im = im_key_clean
        rs_t = re_s_val.strip() if re_s_val else ""
        vars_list_s = (adsa_texts if adsa_texts else ["0.0"])
        RS_list.append({"RS": f"{safe_im}.{idx}", "t": rs_t, "vars": vars_list_s})
    TEXFS, idx, ids_texfs, mv_texfs, act_texfs, _, _ = _emit_tokens(
        texfs_scan, im_key_clean, idx, 'TEXFS', 'cas', span_marker='[Mudsa]', action_marker='[Adsa]', in_span=in_mudsa,
        in_action_span=in_adsa
    )
    CS, idx, ids_cs, _, act_cs, _, _ = _emit_tokens(
        ctx_s_scan, im_key_clean, idx, 'CS', None, span_marker='[Mudsa]', action_marker='[Adsa]', in_span=False,
        in_action_span=False
    )

    ida_block = {
        "Entrada": {
            "Texto Inicial DE ENTRADA": TEXE,
            "Fala DE ENTRADA": FADEN,
            "Texto Final de Entrada": TEFIE,
            "Reação": RE_list,
            "Contexto": CE_e,
        },
        "Pensamento Interno": PIDE,
        "Saída": {
            "Texto Inicial de SAÍDA": TEXIS,
            "Fala de Saída": FS,
            "Reação de Saída": RS_list,
            "Texto Final de Saída": TEXFS,
            "Contexto de Saída": CS,
        },
    }

    um_block = {
        "Características de Entrada": [],
        "Lista de Multivariações de Entrada": [],
        "Características de Saída": [],
        "Lista de Multivariações de Saída": [],
    }
    um_block["Características de Entrada"] += _build_um_from_attrs(TEXE, 'cae', 'Texto de entrada', 'TEXE')
    um_block["Características de Entrada"] += _build_um_from_attrs(FADEN, 'cae', 'Fala de entrada', 'FADEN')
    um_block["Características de Entrada"] += _build_um_from_attrs(TEFIE, 'cae', 'Texto de entrada', 'TEFIE')
    um_block["Características de Saída"] += _build_um_from_attrs(FS, 'cas', 'Fala de saída', 'FS')
    um_block["Características de Saída"] += _build_um_from_attrs(TEXFS, 'cas', 'Texto de saída', 'TEXFS')

    def push_mv(spans, fonte, dado):
        for s in spans or []:
            raw_tokens = (s.get('Tokens') or '').strip()
            if not raw_tokens:
                continue
            try:
                clean_tokens = _normalize_punct(raw_tokens)
            except Exception:
                clean_tokens = raw_tokens
            clean_tokens = re.sub(r"^[\s,\.;:!?—\-]+", "", clean_tokens)
            clean_tokens = re.sub(r"[\s,\.;:!?—\-]+$", "", clean_tokens)
            if not clean_tokens:
                continue
            um_block[f"Lista de Multivariações de {fonte}"].append({
                "Fonte": ("Texto de entrada" if fonte == "Entrada" and dado in ("TEXE", "TEFIE") else
                          "Fala de entrada" if fonte == "Entrada" else
                          "Texto de saída" if dado in ("TEXIS", "TEXFS") else "Fala de saída"),
                "Dado": dado,
                "Tokens": clean_tokens,
                "Multivars:": ["0.0"],
                **{dado: s['ids']}
            })

    push_mv(mv_texe, 'Entrada', 'TEXE')
    push_mv(mv_faden, 'Entrada', 'FADEN')
    push_mv(mv_tefie, 'Entrada', 'TEFIE')
    push_mv(mv_texis, 'Saída', 'TEXIS')
    push_mv(mv_fs, 'Saída', 'FS')
    push_mv(mv_texfs, 'Saída', 'TEXFS')

    clean_texe = _clean_text(ent_before)
    clean_faden = _clean_text(ent_fala)
    clean_tefie = _clean_text(ent_rest)
    clean_ctx_e = _clean_text(secs.get('ce_e', ''))
    clean_texis = _clean_text(saida_before)
    clean_fs = _clean_text(saida_fala)
    clean_texfs = _clean_text(saida_rest)
    clean_ctx_s = _clean_text(secs.get('cs', ''))
    pide_lines = [p.strip() for p in _split_sentences(secs.get('pide', '')) if p and p.strip()]

    has_entrada_text = any([clean_texe, clean_faden, clean_tefie])
    has_saida_text = any([clean_texis, clean_fs, clean_texfs])

    entrada_field = {}
    if clean_texe:
        entrada_field["Texto Inicial de Entrada"] = clean_texe
    if clean_faden:
        entrada_field["Fala de Entrada"] = clean_faden
    if clean_tefie:
        entrada_field["Texto Final de Entrada"] = clean_tefie

    pide_field = "\n".join(pide_lines).strip() if pide_lines else ""

    saida_field = {}
    if clean_texis:
        saida_field["Texto Inicial de Saída"] = clean_texis
    if clean_fs:
        saida_field["Fala de Saída"] = clean_fs
    if clean_texfs:
        saida_field["Texto Final de Saída"] = clean_texfs

    bloco_por_campo = {
        "Entrada": entrada_field,
        "Reação": re_e_val,
        "Contexto": clean_ctx_e,
        "Pensamento Interno": pide_field,
        "Saída": saida_field,
        "Reação (Saída)": re_s_val,
        "Contexto (Saída)": clean_ctx_s,
    }

    clean_lines = []
    if has_entrada_text or re_e_val or clean_ctx_e:
        clean_lines.append("Entrada:")
        if clean_texe:
            clean_lines.append(clean_texe)
        if clean_faden:
            clean_lines.append(f"—{clean_faden}")
        if re_e_val:
            clean_lines.append(f"Reação: {re_e_val}")
        if clean_tefie:
            clean_lines.append(clean_tefie)
        if clean_ctx_e:
            clean_lines.append(f"Contexto: {clean_ctx_e}")
        clean_lines.append("")

    if pide_lines:
        clean_lines.append("PIDE:")
        clean_lines.extend(pide_lines)
        clean_lines.append("")

    if has_saida_text or re_s_val or clean_ctx_s:
        clean_lines.append("Saída:")
        if clean_texis:
            clean_lines.append(clean_texis)
        if clean_fs:
            clean_lines.append(f"—{clean_fs}")
        if clean_texfs:
            clean_lines.append(clean_texfs)
        if re_s_val:
            clean_lines.append(f"Reação: {re_s_val}")
        if clean_ctx_s:
            clean_lines.append(f"Contexto: {clean_ctx_s}")

    def _squeeze_blank_lines(lines):
        out = []
        last_blank = False
        for ln in lines:
            is_blank = (ln.strip() == "")
            if is_blank and last_blank:
                continue
            out.append(ln)
            last_blank = is_blank
        while out and out[-1].strip() == "":
            out.pop()
        return out

    clean_block_text = "\n".join(_squeeze_blank_lines(clean_lines))

    ida_block, um_block = _sanitize_ida_um(ida_block, um_block, im_key)

    total_completo = (ida_block.get("Total de Entrada", []) or []) + (ida_block.get("Total de Saída", []) or [])
    fonte_list = []
    if clean_texe:
        fonte_list.append("Texto Inicial de Entrada")
    if clean_faden:
        fonte_list.append("Fala de Entrada")
    if clean_tefie:
        fonte_list.append("Texto Final de Entrada")
    if re_e_val:
        fonte_list.append("Reação de Entrada")
    if clean_ctx_e:
        fonte_list.append("Contexto de Entrada")
    if pide_lines:
        fonte_list.append("PIDE")
    if clean_texis:
        fonte_list.append("Texto Inicial de Saída")
    if clean_fs:
        fonte_list.append("Fala de Saída")
    if clean_texfs:
        fonte_list.append("Texto Final de Saída")
    if re_s_val:
        fonte_list.append("Reação de Saída")
    if clean_ctx_s:
        fonte_list.append("Contexto de Saída")

    # Construir o UM no formato original
    ordered_um = {
        "Bloco por campo": bloco_por_campo,
        "Fonte": fonte_list,
        "Dados extraídos": ["TEXE", "FADEN", "TEFIE", "RE", "CE", "PIDE", "TEXIS", "FS", "TEXFS", "RS", "CS"],
        "Total de bloco completo": sorted(total_completo, key=lambda s: int(s.split('.')[-1]) if '.' in s else 0),
        "Características de Entrada": um_block.get("Características de Entrada", []),
        "Lista de Multivariações de Entrada": um_block.get("Lista de Multivariações de Entrada", []),
        "Total de Entrada": ida_block.get("Total de Entrada", []),
        "ultimo_child_entrada": ida_block.get("Total de Entrada", [])[-1] if ida_block.get(
            "Total de Entrada") else None,
        "Características de Saída": um_block.get("Características de Saída", []),
        "Lista de Multivariações de Saída": um_block.get("Lista de Multivariações de Saída", []),
        "Total de Saída": ida_block.get("Total de Saída", []),
        "ultimo_child_saida": ida_block.get("Total de Saída", [])[-1] if ida_block.get("Total de Saída") else None,
    }

    # Adicionar textos por campo ao IDA para blocos por campos
    ida_block["Texto Inicial de Entrada (texto)"] = clean_texe
    ida_block["Fala de Entrada (texto)"] = clean_faden
    ida_block["Texto Final de Entrada (texto)"] = clean_tefie
    ida_block["Reação de Entrada (texto)"] = re_e_val
    ida_block["Contexto de Entrada (texto)"] = clean_ctx_e
    ida_block["Pensamento Interno (texto)"] = "\n".join(pide_lines).strip()
    ida_block["Texto Inicial de Saída (texto)"] = clean_texis
    ida_block["Fala de Saída (texto)"] = clean_fs
    ida_block["Texto Final de Saída (texto)"] = clean_texfs
    ida_block["Reação de Saída (texto)"] = re_s_val
    ida_block["Contexto de Saída (texto)"] = clean_ctx_s

    return ida_block, ordered_um, idx


def _extract_text_from_tokens(token_list):
    """Extrai texto de uma lista de tokens (dicts com 't')."""
    if isinstance(token_list, list):
        return ' '.join([item.get('t', '') for item in token_list if isinstance(item, dict)])
    return ''


def _extract_pide_text(pide_list):
    """Extrai texto de PIDE (lista de strings ou dicts)."""
    if isinstance(pide_list, list):
        texts = []
        for item in pide_list:
            if isinstance(item, dict):
                texts.append(item.get('t', ''))
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(texts)
    return ''


def _display_narrative_edit(bloco_id, texe, faden, tefie, reacao, contexto, pide, texis, fs, texfs, reacao_s,
                            contexto_s, im_key):
    """Exibe interface de edição narrativa para um bloco, com text_areas grandes e botões para salvar/cancelar/deletar/reduzir a neutro.

    Args:
        bloco_id: ID do bloco.
        texe, faden, tefie, reacao, contexto, pide, texis, fs, texfs, reacao_s, contexto_s: Textos atuais dos campos.
        im_key: Chave do IM.
    """
    edit_key = f"edit_narr_{bloco_id}"
    save_key = f"save_narr_{bloco_id}"
    delete_key = f"delete_narr_{bloco_id}"
    neutral_key = f"neutral_narr_{bloco_id}"

    # Construir textos narrativos
    entrada_narr = f"{texe} —{faden}. {tefie}"
    saida_narr = f"{texis} —{fs}. {texfs}"

    # Inicializar session_state para edição
    if f"narr_entrada_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_entrada_{bloco_id}"] = entrada_narr
    if f"narr_reacao_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_reacao_{bloco_id}"] = reacao
    if f"narr_contexto_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_contexto_{bloco_id}"] = contexto
    if f"narr_pide_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_pide_{bloco_id}"] = pide
    if f"narr_saida_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_saida_{bloco_id}"] = saida_narr
    if f"narr_reacao_s_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_reacao_s_{bloco_id}"] = reacao_s
    if f"narr_contexto_s_{bloco_id}" not in st.session_state:
        st.session_state[f"narr_contexto_s_{bloco_id}"] = contexto_s

    st.markdown(f"**Bloco {bloco_id}:**")

    if st.session_state.get(edit_key, False):
        # Modo edição
        st.text_area("Entrada:", value=st.session_state[f"narr_entrada_{bloco_id}"], height=100,
                     key=f"ta_entrada_{bloco_id}")
        st.text_input("Reação:", value=st.session_state[f"narr_reacao_{bloco_id}"], key=f"ti_reacao_{bloco_id}")
        st.text_area("Contexto:", value=st.session_state[f"narr_contexto_{bloco_id}"], height=50,
                     key=f"ta_contexto_{bloco_id}")
        st.text_area("Pensamento Interno:", value=st.session_state[f"narr_pide_{bloco_id}"], height=100,
                     key=f"ta_pide_{bloco_id}")
        st.text_area("Saída:", value=st.session_state[f"narr_saida_{bloco_id}"], height=100, key=f"ta_saida_{bloco_id}")
        st.text_input("Reação (Saída):", value=st.session_state[f"narr_reacao_s_{bloco_id}"],
                      key=f"ti_reacao_s_{bloco_id}")
        st.text_area("Contexto (Saída):", value=st.session_state[f"narr_contexto_s_{bloco_id}"], height=50,
                     key=f"ta_contexto_s_{bloco_id}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("SALVAR", key=save_key):
                # Salvar valores editados
                entrada_edit = st.session_state[f"ta_entrada_{bloco_id}"].strip()
                reacao_edit = st.session_state[f"ti_reacao_{bloco_id}"].strip()
                contexto_edit = st.session_state[f"ta_contexto_{bloco_id}"].strip()
                pide_edit = st.session_state[f"ta_pide_{bloco_id}"].strip()
                saida_edit = st.session_state[f"ta_saida_{bloco_id}"].strip()
                reacao_s_edit = st.session_state[f"ti_reacao_s_{bloco_id}"].strip()
                contexto_s_edit = st.session_state[f"ta_contexto_s_{bloco_id}"].strip()

                # Parsear entrada: antes de —, texe; entre — e ., fala; depois ., tefie
                if "—" in entrada_edit:
                    parts = entrada_edit.split("—", 1)
                    texe_edit = parts[0].strip()
                    rest = parts[1]
                    if "." in rest:
                        fala_parts = rest.split(".", 1)
                        faden_edit = fala_parts[0].strip()
                        tefie_edit = fala_parts[1].strip()
                    else:
                        faden_edit = rest.strip()
                        tefie_edit = ""
                else:
                    texe_edit = entrada_edit
                    faden_edit = ""
                    tefie_edit = ""

                # Similar para saída
                if "—" in saida_edit:
                    parts = saida_edit.split("—", 1)
                    texis_edit = parts[0].strip()
                    rest = parts[1]
                    if "." in rest:
                        fs_parts = rest.split(".", 1)
                        fs_edit = fs_parts[0].strip()
                        texfs_edit = fs_parts[1].strip()
                    else:
                        fs_edit = rest.strip()
                        texfs_edit = ""
                else:
                    texis_edit = saida_edit
                    fs_edit = ""
                    texfs_edit = ""

                # Atualizar session_state
                st.session_state[f"value_{bloco_id}_texe"] = texe_edit
                st.session_state[f"value_{bloco_id}_fala"] = faden_edit
                st.session_state[f"value_{bloco_id}_tefie"] = tefie_edit
                st.session_state[f"value_{bloco_id}_reacao"] = reacao_edit
                st.session_state[f"value_{bloco_id}_contexto"] = contexto_edit
                st.session_state[f"value_{bloco_id}_pide"] = pide_edit
                st.session_state[f"value_{bloco_id}_texis"] = texis_edit
                st.session_state[f"value_{bloco_id}_fala_s"] = fs_edit
                st.session_state[f"value_{bloco_id}_texfs"] = texfs_edit
                st.session_state[f"value_{bloco_id}_reacao_s"] = reacao_s_edit
                st.session_state[f"value_{bloco_id}_contexto_s"] = contexto_s_edit

                st.session_state[edit_key] = False
                st.success(f"Bloco {bloco_id} salvo!")
        with col2:
            if st.button("CANCELAR", key=f"cancel_narr_{bloco_id}"):
                st.session_state[edit_key] = False
        with col3:
            if st.button("DELETAR", key=delete_key):
                st.session_state[f"delete_{bloco_id}"] = True
                st.warning(f"Bloco {bloco_id} marcado para deletar. Confirme abaixo.")
        with col4:
            if st.button("Reduzir a Neutro", key=neutral_key):
                st.session_state[f"narr_entrada_{bloco_id}"] = ""
                st.session_state[f"narr_reacao_{bloco_id}"] = ""
                st.session_state[f"narr_contexto_{bloco_id}"] = ""
                st.session_state[f"narr_pide_{bloco_id}"] = ""
                st.session_state[f"narr_saida_{bloco_id}"] = ""
                st.session_state[f"narr_reacao_s_{bloco_id}"] = ""
                st.session_state[f"narr_contexto_s_{bloco_id}"] = ""
                st.info(f"Bloco {bloco_id} reduzido a neutro!")
    else:
        # Modo visualização
        st.write(f"**Entrada:** {entrada_narr}")
        st.write(f"**Reação:** {reacao}")
        st.write(f"**Contexto:** {contexto}")
        st.write(f"**Pensamento Interno:** {pide}")
        st.write(f"**Saída:** {saida_narr}")
        st.write(f"**Reação (Saída):** {reacao_s}")
        st.write(f"**Contexto (Saída):** {contexto_s}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("EDITAR", key=f"btn_edit_narr_{bloco_id}"):
                st.session_state[edit_key] = True
        with col2:
            if st.button("DELETAR", key=delete_key):
                st.session_state[f"delete_{bloco_id}"] = True
                st.warning(f"Bloco {bloco_id} marcado para deletar. Confirme abaixo.")
        with col3:
            if st.button("Reduzir a Neutro", key=neutral_key):
                st.session_state[f"narr_entrada_{bloco_id}"] = ""
                st.session_state[f"narr_reacao_{bloco_id}"] = ""
                st.session_state[f"narr_contexto_{bloco_id}"] = ""
                st.session_state[f"narr_pide_{bloco_id}"] = ""
                st.session_state[f"narr_saida_{bloco_id}"] = ""
                st.session_state[f"narr_reacao_s_{bloco_id}"] = ""
                st.session_state[f"narr_contexto_s_{bloco_id}"] = ""
                st.info(f"Bloco {bloco_id} reduzido a neutro!")
        with col4:
            if st.button("SALVAR", key=save_key):
                st.info(f"Bloco {bloco_id} pronto para salvar no lote.")


def _display_editable_field(label, current_value, bloco_id, field_key, im_key):
    """Exibe um campo editável com valor atual e botões para editar/deletar/salvar/reduzir a neutro.

    Args:
        label (str): Rótulo do campo.
        current_value (str): Valor atual do campo.
        bloco_id: ID do bloco.
        field_key (str): Chave do campo.
        im_key: Chave do IM.
    """
    edit_key = f"edit_{bloco_id}_{field_key}"
    delete_key = f"delete_{bloco_id}_{field_key}"
    neutral_key = f"neutral_{bloco_id}_{field_key}"
    save_key = f"save_{bloco_id}_{field_key}"
    cancel_key = f"cancel_{bloco_id}_{field_key}"

    if st.session_state.get(edit_key, False):
        # Modo edição: mostrar text_area e botões salvar/cancelar
        st.write(f"**Editando {label}:** (Valor atual: {current_value[:50]}...)")  # Debug
        new_value = st.text_area(f"{label}", value=current_value, key=f"ta_{bloco_id}_{field_key}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Salvar", key=save_key):
                # Salvar o novo valor
                st.session_state[f"value_{bloco_id}_{field_key}"] = new_value.strip()
                st.session_state[edit_key] = False
                st.success(f"{label} salvo!")
        with col2:
            if st.button("Cancelar", key=cancel_key):
                st.session_state[edit_key] = False
    else:
        # Modo visualização: mostrar valor e botões
        st.write(
            f"**{label}:** {current_value[:100]}..." if len(current_value) > 100 else f"**{label}:** {current_value}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("EDITAR", key=f"btn_edit_{bloco_id}_{field_key}"):
                st.session_state[edit_key] = True
        with col2:
            if st.button("DELETAR", key=delete_key):
                st.session_state[f"value_{bloco_id}_{field_key}"] = ""
                st.success(f"{label} deletado (limpo)!")
        with col3:
            if st.button("Reduzir a Neutro", key=neutral_key):
                st.session_state[f"value_{bloco_id}_{field_key}"] = ""
                st.info(f"{label} reduzido a neutro!")
        with col4:
            if st.button("SALVAR", key=f"btn_save_{bloco_id}_{field_key}"):
                # Aqui, talvez salvar individualmente, mas por enquanto, apenas confirma
                st.info(f"{label} pronto para salvar no bloco.")  # Função para reconstruir secs de um bloco


def _rebuild_secs_from_fields(entrada, reacao, contexto, pide, saida, reacao_s, contexto_s, bloco_id):
    """Reconstrói seções TPL a partir dos valores editados em session_state.

    Args:
        entrada, reacao, contexto, pide, saida, reacao_s, contexto_s: Valores padrão (usados se não em session_state).
        bloco_id: ID do bloco.

    Returns:
        dict: Seções reconstruídas no formato TPL.
    """
    # Usar valores de session_state se existirem
    entrada_texe = st.session_state.get(f"value_{bloco_id}_texe", entrada.get('Texto Inicial de Entrada', ''))
    entrada_fala = st.session_state.get(f"value_{bloco_id}_fala", entrada.get('Fala de Entrada', ''))
    entrada_tefie = st.session_state.get(f"value_{bloco_id}_tefie", entrada.get('Texto Final de Entrada', ''))
    reacao_val = st.session_state.get(f"value_{bloco_id}_reacao", reacao)
    contexto_val = st.session_state.get(f"value_{bloco_id}_contexto", contexto)
    pide_val = st.session_state.get(f"value_{bloco_id}_pide", pide)
    saida_texis = st.session_state.get(f"value_{bloco_id}_texis", saida.get('Texto Inicial de Saída', ''))
    saida_fala = st.session_state.get(f"value_{bloco_id}_fala_s", saida.get('Fala de Saída', ''))
    saida_texfs = st.session_state.get(f"value_{bloco_id}_texfs", saida.get('Texto Final de Saída', ''))
    reacao_s_val = st.session_state.get(f"value_{bloco_id}_reacao_s", reacao_s)
    contexto_s_val = st.session_state.get(f"value_{bloco_id}_contexto_s", contexto_s)

    return {
        'entrada': f"{entrada_texe}—{entrada_fala}.{entrada_tefie}",
        're_e': reacao_val,
        'ce_e': contexto_val,
        'pide': pide_val,
        'saida': f"{saida_texis}—{saida_fala}.{saida_texfs}",
        're_s': reacao_s_val,
        'cs': contexto_s_val,
    }


def _rebuild_all_blocks_with_new_ids(im_key, edited_blocks):
    """Recalcula IDs para todos os blocos editados, mantendo continuidade e evitando duplicatas.

    Args:
        im_key (str): Chave do IM.
        edited_blocks (list): Lista de blocos editados com secs.

    Returns:
        list: Lista de prévias com IDs realinhados.
    """
    # Calcular o max_idx atual no IM
    max_idx = 0
    for bloco in _ida_data.get("IDA", {}).get("IM", {}).get(im_key, {}).get("blocos", []):
        for section in ["Entrada", "Saída"]:
            if section in bloco:
                for sub in bloco[section].values():
                    if isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, dict) and item:
                                first_key = list(item.keys())[0]
                                id_value = item[first_key]
                                if "." in str(id_value):
                                    parts = str(id_value).split(".")
                                    if len(parts) == 2 and parts[0] == str(im_key):
                                        try:
                                            idx = int(parts[1])
                                            max_idx = max(max_idx, idx)
                                        except ValueError:
                                            pass
    running_idx = max_idx
    new_previas = []
    for idx, edited in enumerate(edited_blocks):
        secs = edited['secs']
        try:
            result = _build_ida_um_from_tpl(secs, im_key, running_idx)
            if result is None or len(result) != 3:
                print(f"Erro no bloco {idx + 1}: função retornou {result}")
                continue
            ida_block, um_block, running_idx = result
        except Exception as e:
            print(f"Erro no bloco {idx + 1}: {e}")
            continue
        new_previas.append({
            'ida_block': ida_block,
            'um_block': um_block,
            'im': str(im_key),
            'nome': edited.get('nome', ''),
            'bloco_id': edited.get('bloco_id', idx + 1),
            'secs': secs
        })
    return new_previas


with st.expander("Passo 3 — Prévia IDA/UM (modelo README)", expanded=True):
    st.caption(
        "Cole os blocos no formato TPL, separados por '---' para múltiplos. Mostra extração dos componentes e gera os JSONs IDA/UM sem salvar.")

    tpl_raw = st.text_area("Blocos (TPL) — Separe múltiplos com '---'", height=360, key="tpl_bloco_raw")
    im_for_ids = st.text_input("IM para IDs (ex.: 0)", value=(
        str(st.session_state.get('sel_im', '0')) if st.session_state.get('sel_im') else '0'))
    if st.button("Extrair e Gerar Prévia", type="primary", key="btn_extract_preview"):
        try:
            # Splitar múltiplos blocos por '---'
            raw_blocks = [b.strip() for b in tpl_raw.split('---') if b.strip()]
            if not raw_blocks:
                st.error("Nenhum bloco encontrado. Cole pelo menos um bloco TPL.")
                st.session_state['previas'] = []
            else:
                previas = []
                im_meta = (_ida_data.get("IDA", {}).get("IM", {}).get(im_for_ids) or {})
                im_nome = im_meta.get("nome", "")
                blocos_existentes = im_meta.get("blocos", [])
                try:
                    max_ida = max([b.get("bloco_id", 0) for b in blocos_existentes]) if isinstance(blocos_existentes,
                                                                                                   list) else 0
                except Exception:
                    max_ida = 0

                um_meta = (_um_data.get("UM", {}).get(im_for_ids) or {})
                um_blocos_existentes = um_meta.get("blocos", [])
                try:
                    max_um = max([b.get("bloco_id", 0) for b in um_blocos_existentes]) if isinstance(
                        um_blocos_existentes, list) else 0
                except Exception:
                    max_um = 0
                base_bloco_id = max(max_ida, max_um) + 1 if (max_ida or max_um) else 1
                running_idx = _get_max_idx_in_im(im_meta, im_for_ids)

                for idx, raw_block in enumerate(raw_blocks):
                    secs = _parse_tpl_sections(raw_block)
                    # Gerar IDA/UM para cada bloco
                    ida_block, um_block, running_idx = _build_ida_um_from_tpl(secs, im_for_ids, running_idx)
                    bloco_id = base_bloco_id + idx

                    previas.append({
                        'ida_block': ida_block,
                        'um_block': um_block,
                        'im': str(im_for_ids),
                        'nome': im_nome,
                        'bloco_id': bloco_id,
                        'secs': secs
                    })

                # Criar JSON consolidado com todos os blocos no mesmo IM
                full_ida = {
                    "IDA": {
                        "IM": {
                            str(im_for_ids): {
                                "nome": im_nome,
                                "blocos": [
                                    {
                                        "bloco_id": p['bloco_id'],
                                        **p['ida_block']
                                    } for p in previas
                                ]
                            }
                        }
                    }
                }

                full_um = {
                    "UM": {
                        str(im_for_ids): {
                            "Universo Mãe": im_nome,
                            "blocos": [
                                {
                                    "bloco_id": p['bloco_id'],
                                    **p['um_block']
                                } for p in previas
                            ]
                        }
                    }
                }

                st.markdown("### JSON Consolidado Final")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Inconsciente.json (IDA):**")
                    st.code(json.dumps(full_ida, ensure_ascii=False, indent=2), language="json")
                with col2:
                    st.markdown("**Memoria.json (UM):**")
                    st.code(json.dumps(full_um, ensure_ascii=False, indent=2), language="json")

                st.session_state['previas'] = previas
                st.success(f"Geradas prévias para {len(previas)} bloco(s) no IM {im_for_ids} ({im_nome}).")
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            st.session_state['previas'] = []

    # Seção de Edição de Blocos Existentes
    st.markdown("### Editar Blocos Existentes no IM")
    edit_im = st.selectbox("Selecione o IM para editar blocos", options=_im_keys, key="edit_im")
    if edit_im:
        im_meta = (_ida_data.get("IDA", {}).get("IM", {}).get(edit_im) or {})
        blocos_existentes = im_meta.get("blocos", [])
        if blocos_existentes:
            st.caption(
                "Edite os campos individualmente abaixo. Use os botões para editar, deletar ou reduzir a neutro. Após edições, clique em 'Aplicar Edições e Realinhar IDs' para recalcular tudo.")

            # Botão para deletar todos os blocos
            st.markdown("#### Deletar Todos os Blocos do IM Selecionado")
            del_all_im = st.selectbox("Selecione o IM para deletar TODOS os blocos", options=_im_keys, key="del_all_im")
            col_del_all, col_spacer = st.columns([1, 3])
            with col_del_all:
                confirm_del_all = st.checkbox("Confirmar exclusão de TODOS os blocos do IM selecionado?",
                                              key="confirm_del_all_blocos")
                if st.button("Deletar Todos os Blocos", type="secondary", key="btn_del_all_blocos"):
                    if confirm_del_all:
                        try:
                            if del_all_im in _ida_data["IDA"]["IM"]:
                                _ida_data["IDA"]["IM"][del_all_im]["blocos"] = []
                            if del_all_im in _um_data["UM"]:
                                _um_data["UM"][del_all_im]["blocos"] = []
                            os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                            with open(_OUT_IDA, "w", encoding="utf-8") as f:
                                json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                            with open(_OUT_UM, "w", encoding="utf-8") as f:
                                json.dump(_um_data, f, ensure_ascii=False, indent=2)
                            st.success(f"Todos os blocos do IM {del_all_im} deletados com sucesso.")
                            _reload_data()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Falha ao deletar todos os blocos: {e}")
                    else:
                        st.warning("Marque a confirmação para deletar todos os blocos.")

            edited_blocks = []
            for idx, bloco in enumerate(blocos_existentes):
                bloco_id = bloco.get('bloco_id', idx + 1)
                with st.expander(f"Bloco {bloco_id}:", expanded=False):
                    # Extrair textos dos tokens no IDA
                    entrada_tokens = bloco.get("Entrada", {})
                    texe_text = _extract_text_from_tokens(entrada_tokens.get("Texto Inicial de Entrada", []))
                    faden_text = _extract_text_from_tokens(entrada_tokens.get("Fala de Entrada", []))
                    tefie_text = _extract_text_from_tokens(entrada_tokens.get("Texto Final de Entrada", []))
                    reacao_text = _extract_text_from_tokens(entrada_tokens.get("Reação", []))
                    contexto_text = _extract_text_from_tokens(entrada_tokens.get("Contexto", []))
                    pide_text = _extract_pide_text(bloco.get("Pensamento Interno", []))

                    saida_tokens = bloco.get("Saída", {})
                    texis_text = _extract_text_from_tokens(saida_tokens.get("Texto Inicial de Saída", []))
                    fs_text = _extract_text_from_tokens(saida_tokens.get("Fala de Saída", []))
                    texfs_text = _extract_text_from_tokens(saida_tokens.get("Texto Final de Saída", []))
                    reacao_s_text = _extract_text_from_tokens(saida_tokens.get("Reação de Saída", []))
                    contexto_s_text = _extract_text_from_tokens(saida_tokens.get("Contexto de Saída", []))

                    # Inicializar session_state com valores atuais se não existirem
                    if f"value_{bloco_id}_texe" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_texe"] = texe_text
                    if f"value_{bloco_id}_fala" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_fala"] = faden_text
                    if f"value_{bloco_id}_tefie" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_tefie"] = tefie_text
                    if f"value_{bloco_id}_reacao" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_reacao"] = reacao_text
                    if f"value_{bloco_id}_contexto" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_contexto"] = contexto_text
                    if f"value_{bloco_id}_pide" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_pide"] = pide_text
                    if f"value_{bloco_id}_texis" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_texis"] = texis_text
                    if f"value_{bloco_id}_fala_s" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_fala_s"] = fs_text
                    if f"value_{bloco_id}_texfs" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_texfs"] = texfs_text
                    if f"value_{bloco_id}_reacao_s" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_reacao_s"] = reacao_s_text
                    if f"value_{bloco_id}_contexto_s" not in st.session_state:
                        st.session_state[f"value_{bloco_id}_contexto_s"] = contexto_s_text

                    _display_narrative_edit(bloco_id, texe_text, faden_text, tefie_text, reacao_text, contexto_text,
                                            pide_text, texis_text, fs_text, texfs_text, reacao_s_text, contexto_s_text,
                                            edit_im)

                    # Botão para deletar bloco individual
                    col_del, col_spacer = st.columns([1, 3])
                    with col_del:
                        confirm_del_bloco = st.checkbox(f"Confirmar exclusão do Bloco {bloco_id}?",
                                                        key=f"confirm_del_bloco_{idx}")
                        if st.button(f"Deletar Bloco {bloco_id}", type="secondary", key=f"btn_del_bloco_{idx}"):
                            if confirm_del_bloco:
                                try:
                                    if bloco_id:
                                        _ida_data["IDA"]["IM"][edit_im]["blocos"] = [b for b in
                                                                                     _ida_data["IDA"]["IM"][edit_im][
                                                                                         "blocos"] if
                                                                                     b.get('bloco_id') != bloco_id]
                                        _um_data["UM"][edit_im]["blocos"] = [b for b in
                                                                             _um_data["UM"][edit_im]["blocos"] if
                                                                             b.get('bloco_id') != bloco_id]
                                        os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                                        with open(_OUT_IDA, "w", encoding="utf-8") as f:
                                            json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                                        with open(_OUT_UM, "w", encoding="utf-8") as f:
                                            json.dump(_um_data, f, ensure_ascii=False, indent=2)
                                        st.success(f"Bloco {bloco_id} deletado com sucesso.")
                                        _reload_data()
                                        st.rerun()
                                    else:
                                        st.error("ID do bloco não encontrado.")
                                except Exception as e:
                                    st.error(f"Falha ao deletar bloco: {e}")
                            else:
                                st.warning("Marque a confirmação para deletar.")

                    # Reconstruir secs editados usando session_state
                    secs = _rebuild_secs_from_fields({}, '', '', '', {}, '', '',
                                                     bloco_id)  # Passar vazios, função usa session_state
                    edited_blocks.append({
                        'secs': secs,
                        'nome': im_meta.get('nome', ''),
                        'bloco_id': bloco_id
                    })

            if st.button("Aplicar Edições e Realinhar IDs", type="primary", key="btn_apply_edits"):
                try:
                    new_previas = _rebuild_all_blocks_with_new_ids(edit_im, edited_blocks)
                    # Mesclar com blocos existentes não editados
                    existing_blocos = im_meta.get("blocos", [])
                    edited_bloco_ids = {p['bloco_id'] for p in new_previas}
                    merged_blocos = [b for b in existing_blocos if b.get('bloco_id') not in edited_bloco_ids] + [
                        {
                            "bloco_id": p['bloco_id'],
                            **p['ida_block']
                        } for p in new_previas
                    ]
                    # Atualizar JSON consolidado
                    full_ida = {
                        "IDA": {
                            "IM": {
                                str(edit_im): {
                                    "nome": im_meta.get('nome', ''),
                                    "blocos": merged_blocos
                                }
                            }
                        }
                    }
                    # Para UM também
                    um_meta = (_um_data.get("UM", {}).get(edit_im) or {})
                    existing_um_blocos = um_meta.get("blocos", [])
                    merged_um_blocos = [b for b in existing_um_blocos if b.get('bloco_id') not in edited_bloco_ids] + [
                        {
                            "bloco_id": p['bloco_id'],
                            **p['um_block']
                        } for p in new_previas
                    ]
                    full_um = {
                        "UM": {
                            str(edit_im): {
                                "Universo Mãe": im_meta.get('nome', ''),
                                "blocos": merged_um_blocos
                            }
                        }
                    }
                    st.markdown("### JSON Editado e Realinhado")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Inconsciente.json (IDA):**")
                        st.code(json.dumps(full_ida, ensure_ascii=False, indent=2), language="json")
                    with col2:
                        st.markdown("**Memoria.json (UM):**")
                        st.code(json.dumps(full_um, ensure_ascii=False, indent=2), language="json")
                    st.session_state['edited_previas'] = new_previas
                    st.success("Edições aplicadas e IDs realinhados!")
                except Exception as e:
                    st.error(f"Erro ao aplicar edições: {e}")
        else:
            st.info("Nenhum bloco encontrado no IM selecionado.")

    # Botão para limpar cache
    st.markdown("### Limpar Cache Temporário")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Limpar Cache (previas e edições)", type="secondary", key="btn_clear_cache"):
            st.session_state['previas'] = []
            st.session_state['edited_previas'] = []
            st.success("Cache limpo! Todas as prévias e edições temporárias foram removidas.")
    with col2:
        if st.button("Reiniciar Sessão Completa", type="secondary", key="btn_restart_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Sessão reiniciada! Todos os dados temporários foram limpos.")
            st.rerun()

# =============================================
# Passo 4: Salvar bloco nos arquivos
# =============================================
with st.expander("Passo 4 — Salvar bloco (IDA e UM)", expanded=False):
    st.caption("Salva os blocos gerados acima em inconsciente.json e memoria.json, com bloco_id sequencial.")
    previas = st.session_state.get('previas', [])
    edited_previas = st.session_state.get('edited_previas', [])
    if previas or edited_previas:
        target_previas = edited_previas if edited_previas else previas
        st.markdown("#### Resumo dos blocos a salvar:")
        for idx, p in enumerate(target_previas):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Bloco {idx + 1}:** IM={p['im']}, bloco_id={p['bloco_id']}, nome={p['nome']}")
            with col_b:
                st.write("Será adicionado/atualizado 1 bloco em IDA e 1 bloco em UM")
        st.markdown(
            f"**Total:** {len(target_previas)} bloco(s) para IM={target_previas[0]['im'] if target_previas else ''}")

        # Botões de download
        if target_previas:
            im_id = target_previas[0]['im']
            full_ida = {
                "IDA": {
                    "IM": {
                        str(im_id): {
                            "nome": target_previas[0]['nome'],
                            "blocos": [
                                {
                                    "bloco_id": p['bloco_id'],
                                    **p['ida_block']
                                } for p in target_previas
                            ]
                        }
                    }
                }
            }
            full_um = {
                "UM": {
                    str(im_id): {
                        "Universo Mãe": target_previas[0]['nome'],
                        "blocos": [
                            {
                                "bloco_id": p['bloco_id'],
                                **p['um_block']
                            } for p in target_previas
                        ]
                    }
                }
            }
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Baixar IDA JSON",
                    data=json.dumps(full_ida, ensure_ascii=False, indent=2),
                    file_name="inconsciente.json",
                    mime="application/json",
                    key="download_ida_final"
                )
            with col2:
                st.download_button(
                    label="Baixar UM JSON",
                    data=json.dumps(full_um, ensure_ascii=False, indent=2),
                    file_name="memoria.json",
                    mime="application/json",
                    key="download_um_final"
                )

        if st.button("Salvar todos os blocos no IM", type="primary", key="btn_salvar_blocos_final"):
            try:
                for p in target_previas:
                    im_id = p['im']
                    bloco_id = p['bloco_id']
                    ida_block = p['ida_block']
                    um_block = p['um_block']
                    _ida_data.setdefault("IDA", {}).setdefault("IM", {}).setdefault(im_id, {})
                    _ida_data["IDA"]["IM"][im_id].setdefault("nome", p['nome'])
                    _ida_data["IDA"]["IM"][im_id].setdefault("blocos", [])
                    # Remover bloco antigo se existir
                    _ida_data["IDA"]["IM"][im_id]["blocos"] = [b for b in _ida_data["IDA"]["IM"][im_id]["blocos"] if
                                                               b.get('bloco_id') != bloco_id]
                    _ida_data["IDA"]["IM"][im_id]["blocos"].append({"bloco_id": bloco_id, **ida_block})
                    _um_data.setdefault("UM", {}).setdefault(im_id, {})
                    _um_data["UM"][im_id].setdefault("Universo Mãe", p['nome'])
                    _um_data["UM"][im_id].setdefault("blocos", [])
                    _um_data["UM"][im_id]["blocos"] = [b for b in _um_data["UM"][im_id]["blocos"] if
                                                       b.get('bloco_id') != bloco_id]
                    _um_data["UM"][im_id]["blocos"].append({"bloco_id": bloco_id, **um_block})
                os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
                with open(_OUT_IDA, "w", encoding="utf-8") as f:
                    json.dump(_ida_data, f, ensure_ascii=False, indent=2)
                with open(_OUT_UM, "w", encoding="utf-8") as f:
                    json.dump(_um_data, f, ensure_ascii=False, indent=2)
                st.success(
                    f"Blocos salvos com sucesso. {len(target_previas)} bloco(s) adicionado(s)/atualizado(s) em IM={target_previas[0]['im']}")
                _reload_data()
                st.session_state['previas'] = []
                st.session_state['edited_previas'] = []
            except Exception as e:
                st.error(f"Falha ao salvar blocos: {e}")
    else:
        st.info("Gere prévias ou edite blocos no Passo 3 para habilitar o salvamento.")


def _extract_pide_text(pide_list):
    """Extrai texto de PIDE (lista de strings ou dicts)."""
    if isinstance(pide_list, list):
        texts = []
        for item in pide_list:
            if isinstance(item, dict):
                texts.append(item.get('t', ''))
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(texts)
    return ''
