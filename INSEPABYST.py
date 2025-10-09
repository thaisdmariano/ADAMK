import streamlit as st
import json
import os
import re

# =============================================
#Passo 1: Criar o cabeçalho#
# [PASSO-1] Criar o cabeçalho
# =============================================
# Cabeçalho mínimo (sem CRUD, sem parser)
st.set_page_config(page_title="INSEPA — IDA/IM (CRUD)", layout="wide")
st.title("INSEPA — IDA/IM (CRUD: renomear)")

st.info("App mínimo carregado. Próximo passo: definir exatamente o que entra no cabeçalho.")

# ===== FIM do PASSO 1 =====


# =============================================
#Passo 2: CRUD de IM (renomear)#
# [PASSO-2] Renomear IM em inconsciente.json
# =============================================

# Caminho do arquivo (pasta pai de app_streamlit)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_OUT_IDA = os.path.join(_BASE_DIR, "inconsciente.json")
_OUT_UM = os.path.join(_BASE_DIR, "memoria.json")

# Carregar JSON simples (sem cache)
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

st.subheader("IDA/IM — Renomear IM")
_ims = _ida_data["IDA"].get("IM", {})
_im_keys = sorted(_ims.keys(), key=lambda k: int(k) if str(k).isdigit() else float('inf'))

if not _im_keys:
	st.info("Nenhum IM encontrado no inconsciente.json.")
else:
	sel_im = st.selectbox("Selecione o IM", options=_im_keys, key="sel_im")
	cur_nome = (_ims.get(sel_im) or {}).get("nome", "")
	st.caption(f"Chave selecionada: {sel_im}")
	novo_nome = st.text_input("Novo nome do IM", value=cur_nome, key=f"novo_nome_{sel_im}")
	if st.button("Salvar novo nome", type="primary", key=f"btn_salvar_{sel_im}"):
		try:
			_ida_data["IDA"]["IM"].setdefault(sel_im, {})
			_ida_data["IDA"]["IM"][sel_im]["nome"] = (novo_nome or "").strip()
			os.makedirs(os.path.dirname(_OUT_IDA), exist_ok=True)
			with open(_OUT_IDA, "w", encoding="utf-8") as f:
				json.dump(_ida_data, f, ensure_ascii=False, indent=2)
			st.success("Nome atualizado com sucesso.")
		except Exception as e:
			st.error(f"Falha ao salvar inconsciente.json: {e}")

# ===== FIM do PASSO 2 =====


# =============================================
#Passo 3: Colar bloco via TPL e extrair campos#
# [PASSO-3] TPL -> TEXE/FADEN/TEFIE + TEXIS/FS/TEXFS + RE/CE/PIDE/RS/CS
# =============================================

def _extract_text_parts(text: str):
	"""Extrai (antes_do_travessao, fala_ate_ponto, resto_pos_fala) de um texto.
	Regras:
	- antes_do_travessao: tudo antes do primeiro travessão (—). Se não houver, é o texto inteiro.
	- fala_ate_ponto: trecho após o travessão até o primeiro ponto final (.) incluso, se existir.
	- resto_pos_fala: o restante do texto após o primeiro ponto final da fala.
	"""
	text = (text or "").strip()
	if not text:
		return "", "", ""
	idx = text.find("—")
	if idx < 0:
		return text, "", ""
	before = text[:idx].strip()
	after = text[idx+1:].lstrip()  # após o travessão
	p = after.find(".")
	if p < 0:
		fala = after.strip()
		rest = ""
	else:
		fala = after[:p+1].strip()
		rest = after[p+1:].lstrip()
	return before, fala, rest


def _parse_tpl_sections(raw: str):
	"""Extrai seções do TPL único:
	Entrada, Reação (E), Contexto (E), Pensamento interno, Saída, Reação (S), Contexto (S).
	Mantém rótulos como [Muden], [CAE: ...], etc. Não normaliza nada.
	"""
	s = (raw or "").strip() + "\n<END>:\n"
	# Helper para capturar desde um cabeçalho até o próximo de uma lista
	def grab(name_pat: str, next_pats: list, start_pos: int = 0):
		nxt = "|".join([rf"^\s*{p}\s*:" for p in next_pats] + [r"^\s*<END>\s*:"])
		m = re.search(rf"(?ims)^\s*{name_pat}\s*:(.*?)(?={nxt})", s[start_pos:])
		if not m:
			return "", start_pos
		content = (m.group(1) or "").strip()
		end_pos = start_pos + m.end()
		return content, end_pos

	pos = 0
	# Ignorar opcional "Bloco:" no topo
	_, pos = grab(r"Bloco", ["Entrada", "Reação", "Contexto", "Pensamento interno", "Saída"], 0)

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


with st.expander("Passo 3 — Colar bloco (TPL)", expanded=True):
	st.caption("Cole o bloco único no formato indicado. A extração mostrará os componentes, sem salvar.")

	tpl_raw = st.text_area("Bloco (TPL)", height=360, key="tpl_bloco_raw")
	if st.button("Extrair do TPL", type="primary", key="btn_tpl_extract"):
		secs = _parse_tpl_sections(tpl_raw)
		# Entrada
		texe, faden, tefie = _extract_text_parts(secs.get("entrada", ""))
		# Saída
		texis, fs, texfs = _extract_text_parts(secs.get("saida", ""))

		st.markdown("---")
		c1, c2 = st.columns(2)
		with c1:
			st.markdown("#### Extratos — Entrada")
			st.write({
				"TEXE": texe,
				"FADEN": faden,
				"TEFIE": tefie,
				"RE": secs.get("re_e", ""),
				"CE": secs.get("ce_e", ""),
				"PIDE": secs.get("pide", ""),
			})
		with c2:
			st.markdown("#### Extratos — Saída")
			st.write({
				"TEXIS": texis,
				"FS": fs,
				"TEXFS": texfs,
				"RS": secs.get("re_s", ""),
				"CS": secs.get("cs", ""),
			})

# ===== FIM do PASSO 3 =====


# =============================================
#Passo 4: Mapear rótulos e gerar IDA/UM (prévia)#
# [PASSO-4] Modelo exato do README — sem salvar
# =============================================

_REACTION_MAP = {"^^": "um sorriso", "<3>": "um sorriso", "<3": "um sorriso", ":)": "um sorriso"}

def _scan_tokens(text: str):
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

def _emit_tokens(scan, im_key: str, start_idx: int, dado: str, attr_key: str = None, span_marker: str = None):
	out = []
	ids = []
	spans = []  # para multivars
	idx = start_idx
	current_attr_out = None
	current_attr_norm = None
	in_span = False
	span_tokens = []
	span_ids = []

	i = 0
	while i < len(scan):
		tk = scan[i]
		if tk['kind'] == 'marker':
			s = tk['val']
			# CAE/CAS
			if re.match(r"\[(?:CAE|CAS)\s*:[^\]]+\]", s, flags=re.IGNORECASE):
				label_raw = s.split(":",1)[1][:-1].strip()
				label_clean = re.sub(r"[\s\.:,;!?]+$", "", label_raw).strip()
				label_norm = label_clean.lower()
				if current_attr_norm == label_norm:
					current_attr_out = None
					current_attr_norm = None
				else:
					current_attr_out = label_clean
					current_attr_norm = label_norm
			# Muden/Mudsa (span)
			elif span_marker and s.lower() == span_marker.lower():
				in_span = not in_span
				if not in_span and span_tokens:
					spans.append({"Tokens": " ".join(span_tokens).strip(), "ids": span_ids[:]})
					span_tokens, span_ids = [], []
			# Ade/Adsa: tratadas como marcadores de ação (não togglam atributo)
			i += 1
			continue

		# materializa token
		idx += 1
		safe_im = str(im_key or "0").strip()
		tid = f"{safe_im}.{idx}"
		ids.append(tid)
		obj = {dado: tid, "t": tk['val'], "vars": ["0.0"]}
		if current_attr_out and attr_key:
			obj[attr_key] = [f": {current_attr_out}"]
		out.append(obj)
		if in_span:
			span_tokens.append(tk['val'])
			span_ids.append(tid)
		i += 1

	return out, idx, ids, spans

def _split_sentences(text: str):
	if not text:
		return []
	parts = re.split(r"(?<=[\.!?])\s+", text.strip())
	return [p for p in parts if p]

def _build_um_from_attrs(tokens, attr_key: str, fonte: str, dado_key: str):
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
					entries.append({attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key, "Tokens": " ".join(cur_tokens).strip(), dado_key: cur_ids[:]})
				cur = label
				cur_tokens = [t['t']]
				cur_ids = [tid] if tid else []
		else:
			if cur and cur_tokens:
				entries.append({attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key, "Tokens": " ".join(cur_tokens).strip(), dado_key: cur_ids[:]})
			cur, cur_tokens, cur_ids = None, [], []
	if cur and cur_tokens:
		entries.append({attr_key.upper(): cur, "Fonte": fonte, "Dado": dado_key, "Tokens": " ".join(cur_tokens).strip(), dado_key: cur_ids[:]})
	return entries

def _sanitize_ida_um(ida_block, um_block, im_key: str):
	safe_im = str(im_key or "0").strip()
	neutral = f"{safe_im}.0"
	def ensure_neutral_list(objs, key_name):
		if objs:
			return objs
		return [{key_name: neutral, 't': '', 'vars': ['0.0']}]
	def ensure_neutral_ids(ids):
		return ids if ids else [neutral]
	# Entrada (reconstroi no formato e ordem desejados)
	ent_src = ida_block.get("Entrada", {})
	ent = {}
	ent["Texto Inicial DE ENTRADA"] = ensure_neutral_list(ent_src.get("Texto Inicial DE ENTRADA"), 'TEXE')
	ent["Fala DE ENTRADA"] = ensure_neutral_list(ent_src.get("Fala DE ENTRADA"), 'FADEN')
	ent["Texto Final de Entrada"] = ensure_neutral_list(ent_src.get("Texto Final de Entrada"), 'TEFIE')
	ent["Reação"] = ent_src.get("Reação") or [{"RE": neutral, "t": "", "vars": ["0.0"]}]
	ent["Contexto"] = ensure_neutral_list(ent_src.get("Contexto"), 'CE')
	ida_block["Entrada"] = ent

	# Saída (reconstroi no formato e ordem desejados)
	sai_src = ida_block.get("Saída", {})
	sai = {}
	sai["Texto Inicial de SAÍDA"] = ensure_neutral_list(sai_src.get("Texto Inicial de SAÍDA"), 'TEXIS')
	sai["Fala de Saída"] = ensure_neutral_list(sai_src.get("Fala de Saída"), 'FS')
	sai["Texto Final de Saída"] = ensure_neutral_list(sai_src.get("Texto Final de Saída"), 'TEXFS')
	sai["Reação de Saída"] = sai_src.get("Reação de Saída") or [{"RS": neutral, "t": "", "vars": ["0.0"]}]
	sai["Contexto de Saída"] = ensure_neutral_list(sai_src.get("Contexto de Saída"), 'CE')
	ida_block["Saída"] = sai
	# PIDE
	if not ida_block.get("Pensamento Interno"):
		ida_block["Pensamento Interno"] = [{"PIDE": neutral, "t": ""}]
	# Totais
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
		*ids_from(sai.get("Contexto de Saída"), 'CE'),
	]
	ida_block["Total de Entrada"] = total_e if total_e else [neutral]
	ida_block["Total de Saída"] = total_s if total_s else [neutral]

	if um_block is not None:
		um_block["Total de Entrada"] = um_block.get("Total de Entrada") or ida_block["Total de Entrada"]
		um_block["Total de Saída"] = um_block.get("Total de Saída") or ida_block["Total de Saída"]
		um_block["ultimo_child_entrada"] = um_block["Total de Entrada"][-1]
		um_block["ultimo_child_saida"] = um_block["Total de Saída"][-1]
	return ida_block, um_block

def _build_ida_um_from_tpl(secs: dict, im_key: str):
	# Tokenizar partes
	texe_scan = _scan_tokens(secs.get('entrada',''))
	faden_scan = _scan_tokens(secs.get('entrada',''))  # falas vêm da mesma seção após travessão
	tefie_scan = _scan_tokens(secs.get('entrada',''))
	# Separar com base no travessão e ponto
	ent_before, ent_fala, ent_rest = _extract_text_parts(secs.get('entrada',''))
	texe_scan = _scan_tokens(ent_before)
	faden_scan = _scan_tokens(ent_fala)
	tefie_scan = _scan_tokens(ent_rest)
	# Saída
	saida_before, saida_fala, saida_rest = _extract_text_parts(secs.get('saida',''))
	texis_scan = _scan_tokens(saida_before)
	fs_scan = _scan_tokens(saida_fala)
	texfs_scan = _scan_tokens(saida_rest)
	# Contextos e reações
	ctx_e_scan = _scan_tokens(secs.get('ce_e',''))
	ctx_s_scan = _scan_tokens(secs.get('cs',''))
	re_e_val = (secs.get('re_e','') or '').strip()
	re_s_val = (secs.get('re_s','') or '').strip()

	# Emitir objetos com ids e atributos
	idx = 0
	TEXE, idx, ids_texe, mv_texe = _emit_tokens(texe_scan, im_key, idx, 'TEXE', 'cae', span_marker='[Muden]')
	FADEN, idx, ids_faden, mv_faden = _emit_tokens(faden_scan, im_key, idx, 'FADEN', 'cae', span_marker='[Muden]')
	RE_list = []
	if re_e_val:
		sym = re_e_val.split()[0]
		idx += 1
		safe_im = str(im_key or "0").strip()
		RE_list.append({"RE": f"{safe_im}.{idx}", "t": sym, "vars": [_REACTION_MAP.get(sym, "0.0")]})
	TEFIE, idx, ids_tefie, mv_tefie = _emit_tokens(tefie_scan, im_key, idx, 'TEFIE', 'cae', span_marker='[Muden]')
	CE_e, idx, ids_ce_e, _ = _emit_tokens(ctx_e_scan, im_key, idx, 'CE', None, span_marker='[Muden]')

	TEXIS, idx, ids_texis, mv_texis = _emit_tokens(texis_scan, im_key, idx, 'TEXIS', None, span_marker='[Mudsa]')
	FS, idx, ids_fs, mv_fs = _emit_tokens(fs_scan, im_key, idx, 'FS', 'cas', span_marker='[Mudsa]')
	RS_list = []
	if re_s_val:
		sym2 = re_s_val.split()[0]
		idx += 1
		safe_im = str(im_key or "0").strip()
		RS_list.append({"RS": f"{safe_im}.{idx}", "t": sym2, "vars": [_REACTION_MAP.get(sym2, "0.0")]})
	TEXFS, idx, ids_texfs, mv_texfs = _emit_tokens(texfs_scan, im_key, idx, 'TEXFS', 'cas', span_marker='[Mudsa]')
	CE_s, idx, ids_ce_s, _ = _emit_tokens(ctx_s_scan, im_key, idx, 'CE', None, span_marker='[Mudsa]')

	# Garantir neutros e totais
	ida_block = {
		"Entrada": {
			"Texto Inicial DE ENTRADA": TEXE,
			"Fala DE ENTRADA": FADEN,
			"Texto Final de Entrada": TEFIE,
			"Reação": RE_list,
			"Contexto": CE_e,
		},
		"Pensamento Interno": [],
		"Saída": {
			"Texto Inicial de SAÍDA": TEXIS,
			"Fala de Saída": FS,
			"Texto Final de Saída": TEXFS,
			"Reação de Saída": RS_list,
			"Contexto de Saída": CE_s,
		},
	}
	# PIDE por frase
	PIDE = []
	for s in _split_sentences(secs.get('pide','')):
		s2 = s.strip()
		if not s2: continue
		idx += 1
		safe_im = str(im_key or "0").strip()
		tid = f"{safe_im}.{idx}"
		PIDE.append({"PIDE": tid, "t": s2})
	ida_block["Pensamento Interno"] = PIDE

	# UM
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
			if s.get('Tokens'):
				um_block[f"Lista de Multivariações de {fonte}"].append({
					"Fonte": ("Texto de entrada" if fonte=="Entrada" and dado in ("TEXE","TEFIE") else
							  "Fala de entrada" if fonte=="Entrada" else
							  "Texto de saída" if dado in ("TEXIS","TEXFS") else "Fala de saída"),
					"Dado": dado,
					"Tokens": s['Tokens'],
					"Multivars:": ["0.0"],
					dado: s['ids']
				})
	push_mv(mv_texe, 'Entrada', 'TEXE')
	push_mv(mv_faden, 'Entrada', 'FADEN')
	push_mv(mv_tefie, 'Entrada', 'TEFIE')
	push_mv(mv_texis, 'Saída', 'TEXIS')
	push_mv(mv_fs, 'Saída', 'FS')
	push_mv(mv_texfs, 'Saída', 'TEXFS')

	ida_block, um_block = _sanitize_ida_um(ida_block, um_block, im_key)
	return ida_block, um_block


with st.expander("Passo 4 — Prévia IDA/UM (modelo README)", expanded=False):
	st.caption("Gera os JSONs IDA/UM com mapeamento de rótulos para tokens. Não salva arquivos.")
	tpl_src = st.text_area("Cole o mesmo bloco (TPL)", height=260, key="tpl_src_p4")
	im_for_ids = st.text_input("IM para IDs (ex.: 0)", value=(str(st.session_state.get('sel_im', '0')) if st.session_state.get('sel_im') else '0'))
	if st.button("Gerar IDA/UM (prévia)", key="btn_p4_preview"):
		try:
			secs = _parse_tpl_sections(tpl_src)
			ida_block, um_block = _build_ida_um_from_tpl(secs, im_for_ids)
			# Determinar nome e próximo bloco_id com base no arquivo existente
			im_meta = (_ida_data.get("IDA", {}).get("IM", {}).get(im_for_ids) or {})
			im_nome = im_meta.get("nome", "")
			blocos_existentes = im_meta.get("blocos", [])
			prox_bloco_id_ida = (len(blocos_existentes) + 1) if isinstance(blocos_existentes, list) else 1

			um_meta = (_um_data.get("UM", {}).get(im_for_ids) or {})
			um_blocos_existentes = um_meta.get("blocos", [])
			prox_bloco_id_um = (len(um_blocos_existentes) + 1) if isinstance(um_blocos_existentes, list) else 1
			prox_bloco_id = max(prox_bloco_id_ida, prox_bloco_id_um)

			full_ida = {
				"IDA": {
					"IM": {
						str(im_for_ids): {
							"nome": im_nome,
							"blocos": [
								{
									"bloco_id": prox_bloco_id,
									**ida_block
								}
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
								"bloco_id": prox_bloco_id,
								**um_block
							}
						]
					}
				}
			}

			st.markdown("Inconsciente.json (IDA) — com cabeçalho e ordem:")
			st.code(json.dumps(full_ida, ensure_ascii=False, indent=2), language="json")
			st.markdown("Memoria.json (UM) — com cabeçalho e ordem:")
			st.code(json.dumps(full_um, ensure_ascii=False, indent=2), language="json")
		except Exception as e:
			st.error(f"Erro ao gerar prévia: {e}")

# ===== FIM do PASSO 4 =====

