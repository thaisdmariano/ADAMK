import json
import os
import re
from typing import List, Dict

import streamlit as st

OUT_IDA = os.path.join(os.path.dirname(__file__), '..', 'inconsciente.json')
OUT_UM = os.path.join(os.path.dirname(__file__), '..', 'adam_memoria.json')

st.set_page_config(page_title="INSEPA - Navegador IDA/UM", layout="wide")
st.title("INSEPA - Navegador IDA/UM")

@st.cache_data
def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            return json.loads(txt) if txt else {}
    except Exception:
        return {}

def _save_json(path: str, data: Dict, pretty: bool = True):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        return True
    except Exception:
        return False

def _merge_blocks(ida_cur: Dict, um_cur: Dict, im_key: str, novos_blocos: List[Dict], novo_nome: str | None = None) -> tuple:
    ida_cur = ida_cur or {}
    um_cur = um_cur or {}
    ida_cur.setdefault('IDA', {}).setdefault('IM', {})
    um_cur.setdefault('UM', {})
    # IDA
    IM = ida_cur['IDA']['IM']
    node = IM.setdefault(str(im_key), {"nome": novo_nome or IM.get(str(im_key), {}).get('nome', ''), "blocos": []})
    if novo_nome:
        node['nome'] = novo_nome
    node.setdefault('blocos', [])
    existing_ids = []
    for b in node['blocos']:
        try:
            existing_ids.append(int(b.get('bloco_id')))
        except Exception:
            pass
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    used = set(existing_ids)
    for b in novos_blocos:
        bid = None
        try:
            bid = int(b.get('bloco_id'))
        except Exception:
            bid = None
        if bid is None or bid <= 0 or bid in used:
            bid = next_id
            next_id += 1
        used.add(bid)
        b['bloco_id'] = bid
        node['blocos'].append(b)
    # UM (cabeçalho)
    um_node = um_cur['UM'].setdefault(str(im_key), {"Universo Mãe": node.get('nome') or novo_nome or "Interações", "blocos": []})
    if node.get('nome'):
        um_node['Universo Mãe'] = node['nome']
    um_node.setdefault('blocos', [])
    return ida_cur, um_cur

def _update_block_raw(ida_cur: Dict, im_key: str, bloco_id: int, novo_raw: str) -> Dict:
    """Atualiza o campo 'Bloco (raw)' de um bloco existente pelo bloco_id dentro de um IM específico."""
    ida_cur = ida_cur or {}
    try:
        bid = int(bloco_id)
    except Exception:
        return ida_cur
    im_node = ((ida_cur.get('IDA') or {}).get('IM') or {}).get(str(im_key))
    if not im_node:
        return ida_cur
    blocos = im_node.get('blocos') or []
    for b in blocos:
        try:
            if int(b.get('bloco_id')) == bid:
                b['Bloco (raw)'] = novo_raw
                break
        except Exception:
            continue
    return ida_cur

def _clean_block_labels(text: str) -> str:
    """Remove marcadores de rótulos (Muden/Mudsa/Ade/Adsa, CAE:/CAS:) preservando o conteúdo.
    Mantém cabeçalhos como Entrada:, Saída:, Reação:, Contexto:, Pensamento interno:.
    """
    if not text:
        return text
    # Remover marcadores simples
    text = re.sub(r"\[(?:Muden|Mudsa|Ade|Adsa)\]", "", text, flags=re.IGNORECASE)
    # Remover marcadores CAE:/CAS: com qualquer conteúdo interno
    text = re.sub(r"\[(?:CAE|CAS)\s*:[^\]]+\]", "", text, flags=re.IGNORECASE)
    # Normalizar espaços múltiplos
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Normalizar espaços antes de pontuação comum
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

# ---------------- Termômetro de Sentimento (BY INSEPA) ----------------
THERM_WEIGHTS = {
    "entrada": {"TEXE": 15, "FADEN": 15, "TEFIE": 15, "RE": 25, "CE": 30},
    "saida":   {"TEXIS": 15, "FS": 15, "TEXFS": 15, "RS": 25, "CS": 30},
}
NEUTRAL_PENALTY = 9

def _therm_bucket(score_0_100: int):
    if score_0_100 >= 85:
        return 100, "Positivo"
    if score_0_100 >= 60:
        return 70, "Levemente positivo"
    if score_0_100 >= 45:
        return 50, "Neutro"
    if score_0_100 >= 40:
        return 40, "Negativo"
    return 0, "Negativo grave"

def _therm_score_from_signs(signs: dict, weights: dict):
    sitems = (signs or {}).items()
    pos = sum(weights.get(k, 0) for k, v in sitems if v > 0)
    neg = sum(weights.get(k, 0) for k, v in sitems if v < 0)
    neu_count = sum(1 for k, v in sitems if v == 0 and k in weights)
    score = pos - neg + (neu_count * NEUTRAL_PENALTY)
    score = max(0, min(100, score))
    return score, _therm_bucket(score)

# Carregar dados
ida_data = load_json(OUT_IDA)
um_data = load_json(OUT_UM)

# UI: Bloco de notas (único modo)
st.subheader("Bloco de notas")
ida_im = ((ida_data.get('IDA') or {}).get('IM') or {})
im_keys = sorted(ida_im.keys(), key=lambda k: int(k) if str(k).isdigit() else 10**9)
im_sel = st.selectbox("IDA/IM:", options=im_keys or ['0'], index=(im_keys.index('0') if '0' in im_keys else 0))
nome_atual_val = ((ida_im.get(im_sel) or {}).get('nome') or "Não definido")
st.markdown(f"**IDA/IM: {im_sel}**")
st.markdown(f"**Nome: [{nome_atual_val}]**")

# Editar nome do IDA/IM diretamente
if st.button("Editar nome do IDA/IM"):
    st.session_state['edit_im_name'] = True
if st.session_state.get('edit_im_name'):
    novo_nome_im = st.text_input(
        "Novo nome do IDA/IM",
        value=(nome_atual_val if nome_atual_val and nome_atual_val != "Não definido" else "Interações"),
        key="edit_nome_im_val"
    )
    if st.button("Salvar novo nome"):
        ida_m, um_m = _merge_blocks(dict(ida_data), dict(um_data), im_sel, [], novo_nome_im.strip() if novo_nome_im else "")
        if _save_json(OUT_IDA, ida_m, pretty=True) and _save_json(OUT_UM, um_m, pretty=True):
            st.success(f"IDA/IM: {im_sel}- Nome: {ida_m.get('IDA',{}).get('IM',{}).get(str(im_sel),{}).get('nome','Interações')} salvo com sucesso")
            st.session_state['edit_im_name'] = False
            st.cache_data.clear()
        else:
            st.error("Falha ao salvar novo nome.")

blocos = ((ida_im.get(im_sel) or {}).get('blocos') or [])
if not blocos:
    st.text("<Sem blocos disponíveis por favor colar um novo bloco>")
else:
    last = blocos[-1]
    raw_last = (last or {}).get('Bloco (raw)') or ''
    if raw_last:
        st.markdown("Bloco:")
        st.code(_clean_block_labels(raw_last), language="markdown")

# Opções de edição de bloco (acima de Novo Bloco)
st.markdown("---")
st.markdown("**Editar bloco (versão com rótulos)**")
if not blocos:
    st.caption("Nenhum bloco disponível para edição.")
else:
    try:
        bloco_ids = sorted([int(b.get('bloco_id')) for b in blocos if b.get('bloco_id') is not None])
    except Exception:
        bloco_ids = []
    default_idx = len(bloco_ids) - 1 if bloco_ids else 0
    sel_id = st.selectbox("Escolha o bloco_id para editar", options=bloco_ids or [1], index=default_idx, key=f"edit_sel_{im_sel}")
    # Recuperar conteúdo atual do bloco selecionado
    atual_raw = ""
    for b in blocos:
        try:
            if int(b.get('bloco_id')) == sel_id:
                atual_raw = (b.get('Bloco (raw)') or '')
                break
        except Exception:
            continue
    novo_raw_edit = st.text_area("Editar Bloco (com rótulos)", value=atual_raw, height=220, key=f"edit_raw_{im_sel}_{sel_id}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Salvar edição do bloco", key=f"btn_save_edit_{im_sel}_{sel_id}"):
            if novo_raw_edit and novo_raw_edit.strip():
                ida_upd = _update_block_raw(dict(ida_data), im_sel, sel_id, novo_raw_edit.strip())
                if _save_json(OUT_IDA, ida_upd, pretty=True):
                    im_node = ((ida_upd.get('IDA') or {}).get('IM') or {}).get(str(im_sel), {})
                    nome_msg = im_node.get('nome') or "Interações"
                    st.success(f"IDA/IM: {im_sel} -Nome: {nome_msg}- bloco_id {sel_id} atualizado com sucesso")
                    st.cache_data.clear()
                else:
                    st.error("Falha ao salvar edição do bloco.")
            else:
                st.warning("O conteúdo do bloco não pode ser vazio.")
    with c2:
        st.button("Descartar alterações", key=f"btn_cancel_edit_{im_sel}_{sel_id}")

    # Análise de Sentimento (BY INSEPA)
    st.markdown("---")
    st.markdown("**Análise de Sentimento (BY INSEPA)**")
    st.caption("Escolha os sinais (+/0/-) por componente; usamos pesos e neutro soma 9 pontos por componente.")
    cE1, cE2, cE3, cE4, cE5 = st.columns(5)
    with cE1:
        e_texe = st.selectbox("Entrada TEXE", ["+","0","-"], index=1, key=f"sE_TEXE_{im_sel}_{sel_id}")
    with cE2:
        e_faden = st.selectbox("Entrada FADEN", ["+","0","-"], index=1, key=f"sE_FADEN_{im_sel}_{sel_id}")
    with cE3:
        e_tefie = st.selectbox("Entrada TEFIE", ["+","0","-"], index=1, key=f"sE_TEFIE_{im_sel}_{sel_id}")
    with cE4:
        e_re = st.selectbox("Entrada RE", ["+","0","-"], index=1, key=f"sE_RE_{im_sel}_{sel_id}")
    with cE5:
        e_ce = st.selectbox("Entrada CE", ["+","0","-"], index=1, key=f"sE_CE_{im_sel}_{sel_id}")

    cS1, cS2, cS3, cS4, cS5 = st.columns(5)
    with cS1:
        s_texis = st.selectbox("Saída TEXIS", ["+","0","-"], index=1, key=f"sS_TEXIS_{im_sel}_{sel_id}")
    with cS2:
        s_fs = st.selectbox("Saída FS", ["+","0","-"], index=1, key=f"sS_FS_{im_sel}_{sel_id}")
    with cS3:
        s_texfs = st.selectbox("Saída TEXFS", ["+","0","-"], index=1, key=f"sS_TEXFS_{im_sel}_{sel_id}")
    with cS4:
        s_rs = st.selectbox("Saída RS", ["+","0","-"], index=1, key=f"sS_RS_{im_sel}_{sel_id}")
    with cS5:
        s_cs = st.selectbox("Saída CS", ["+","0","-"], index=1, key=f"sS_CS_{im_sel}_{sel_id}")

    if st.button("Salvar sentimento (BY INSEPA)", key=f"btn_save_sent_{im_sel}_{sel_id}"):
        mapv = {"+": 1, "0": 0, "-": -1}
        sE = {
            "TEXE": mapv.get(e_texe, 0),
            "FADEN": mapv.get(e_faden, 0),
            "TEFIE": mapv.get(e_tefie, 0),
            "RE": mapv.get(e_re, 0),
            "CE": mapv.get(e_ce, 0),
        }
        sS = {
            "TEXIS": mapv.get(s_texis, 0),
            "FS": mapv.get(s_fs, 0),
            "TEXFS": mapv.get(s_texfs, 0),
            "RS": mapv.get(s_rs, 0),
            "CS": mapv.get(s_cs, 0),
        }
        scoreE, (bucketE, labelE) = _therm_score_from_signs(sE, THERM_WEIGHTS["entrada"])
        scoreS, (bucketS, labelS) = _therm_score_from_signs(sS, THERM_WEIGHTS["saida"])
        # Atualizar no IDA
        ida_upd = dict(ida_data)
        im_node = ((ida_upd.get('IDA') or {}).get('IM') or {}).get(str(im_sel)) or {}
        for b in (im_node.get('blocos') or []):
            try:
                if int(b.get('bloco_id')) == sel_id:
                    b.setdefault('Entrada', {})['Sentimento de Entrada'] = {"SDE": str(scoreE), "t": labelE, "Tendência de Entrada": bucketE}
                    b['Sentimento da Saída'] = {"SDS": str(scoreS), "t": labelS, "Tendência da Saída": bucketS}
                    b['Ressonância'] = (bucketE == bucketS)
                    break
            except Exception:
                continue
        if _save_json(OUT_IDA, ida_upd, pretty=True):
            nome_msg = im_node.get('nome') or "Interações"
            st.success(f"IDA/IM: {im_sel} -Nome: {nome_msg}- bloco_id {sel_id} sentimento atualizado | Entrada: {scoreE} ({labelE}) | Saída: {scoreS} ({labelS}) | Ressonância: {bucketE == bucketS}")
            st.cache_data.clear()
        else:
            st.error("Falha ao salvar sentimento.")

novo_raw = st.text_area("Novo Bloco (cole aqui)", height=220, key="novo_bloco_raw")
if st.button("Associar bloco ao IM atual"):
    st.session_state['bn_associate'] = True
if st.session_state.get('bn_associate'):
    nome_preview = nome_atual_val or "Não definido"
    st.markdown(f"**Este bloco será associado ao IDA/IM:{im_sel} Nome: {nome_preview}**")
    if st.button("Deseja atualizar o nome? (Sim)"):
        st.session_state['bn_update_name'] = True
    if st.session_state.get('bn_update_name'):
        novo_nome = st.text_input("Digite novo nome", value=(nome_atual_val if nome_atual_val != "Não definido" else "Interações"), key="bn_novo_nome")
        if st.button("Salvar nome e bloco"):
            novos = []
            if novo_raw and novo_raw.strip():
                novos.append({
                    "bloco_id": None,
                    "Bloco (raw)": novo_raw.strip(),
                    "Entrada": {},
                    "Saída": {},
                })
            ida_m, um_m = _merge_blocks(dict(ida_data), dict(um_data), im_sel, novos, novo_nome.strip() if novo_nome else None)
            if _save_json(OUT_IDA, ida_m, pretty=True) and _save_json(OUT_UM, um_m, pretty=True):
                st.success(f"IDA/IM: {im_sel}- Nome: {ida_m.get('IDA',{}).get('IM',{}).get(str(im_sel),{}).get('nome','Interações')} salvo com sucesso")
                st.session_state['bn_associate'] = False
                st.session_state['bn_update_name'] = False
                st.cache_data.clear()
            else:
                st.error("Falha ao salvar.")
    else:
        if st.button("Salvar bloco (sem alterar nome)"):
            novos = []
            if novo_raw and novo_raw.strip():
                novos.append({
                    "bloco_id": None,
                    "Bloco (raw)": novo_raw.strip(),
                    "Entrada": {},
                    "Saída": {},
                })
            ida_m, um_m = _merge_blocks(dict(ida_data), dict(um_data), im_sel, novos, None)
            if _save_json(OUT_IDA, ida_m, pretty=True) and _save_json(OUT_UM, um_m, pretty=True):
                st.success(f"IDA/IM: {im_sel}- Nome: {ida_m.get('IDA',{}).get('IM',{}).get(str(im_sel),{}).get('nome','Interações')} salvo com sucesso")
                st.session_state['bn_associate'] = False
                st.cache_data.clear()
            else:
                st.error("Falha ao salvar.")

st.divider()
st.markdown("**Deseja criar um novo IDA/IM?**")
if st.button("Criar novo IM"):
    st.session_state['bn_create_im'] = True
if st.session_state.get('bn_create_im'):
    try:
        next_im = str((max([int(k) for k in ida_im.keys() if str(k).isdigit()]) + 1) if ida_im else 0)
    except Exception:
        next_im = '0'
    new_key = st.text_input("Digite a chave do novo IDA/IM (número)", value=next_im, key="bn_new_im_key")
    new_name = st.text_input("Digite novo nome", value="Interações", key="bn_new_im_name")
    if st.button("Criar e definir IM"):
        if not re.match(r"^\d+$", str(new_key or "").strip()):
            st.error("A chave do IM deve ser numérica.")
        else:
            ida_m, um_m = _merge_blocks(dict(ida_data), dict(um_data), str(new_key).strip(), [], new_name.strip())
            if _save_json(OUT_IDA, ida_m, pretty=True) and _save_json(OUT_UM, um_m, pretty=True):
                st.success(f"IDA/IM: {new_key}- Nome: {new_name} salvo com sucesso")
                st.session_state['bn_create_im'] = False
                st.cache_data.clear()
            else:
                st.error("Falha ao criar IM.")

st.caption("Bloco de notas INSEPA — salvos em inconsciente.json (IDA) e cabeçalho em adam_memoria.json (UM).")
