#!/usr/bin/env python3
# coding: utf-8
"""
BlocoINSEPAviaPrompt_final.py
- Lê bloco por stdin (termine com linha vazia)
- Gera inconsciente.json (texto no formato exato que você exigiu)
- Gera adam_memoria.json (JSON válido) contendo alnulu_total_bloco, resumo do bloco e Multivars no IM (por bloco)
- Coloca "Características de usuário" imediatamente após "Contexto"
- Sem inferência: Características de usuário ficam em 0.0 se não fornecidas
- ALNULU calculado a partir do conteúdo literal do bloco
- Quando não houver TEXTO FINAL, emite o placeholder exigido:
  "TEXTO FINAL": [ { "TEXF":"0.0", "tokens": [ {"TEX":"0.0","t":"0.0","vars":["0.0"]} ] } ]
- Multivars (paráfrases) ficam no IM, por bloco, como objetos:
  {
    "Fonte": "Texto de Entrada Inicial",
    "range": { "inicio": "0.x", "fim": "0.y" },
    "text": "Frase original",
    "alternativas": ["Alt 1", "Alt 2"],
    "meta": { opcional }
  }

Entrada de Multivars no prompt (opcional):
- Ao final do bloco, insira:

Multivars:
[Texto de Entrada Inicial] Era de manhã. => Havia amanhecido. || O dia havia começado.
[Fala de Saída] — Olá Socorro. => — Olá. || — Oi, Socorro.

Observações:
- As alternativas não geram marcadores até serem selecionadas em outro pipeline.
- Caso não haja Multivars, o IM terá "Multivars": ["0.0"].
"""
import re, sys, json, os
import unicodedata
from typing import List, Dict, Tuple

OUT_IDA = "inconsciente.json"
OUT_UM = "adam_memoria.json"

# ---------------- Config de marcadores (janela reservada) ----------------
# Parte/prefixo dos marcadores (ex.: "0" gera 0.1, 0.2, ...)
WINDOW_PART = "0"
# Tamanho da janela reservada por bloco (ex.: 59 => 0.1 até 0.59)
WINDOW_MAX = 59
# Enforce: True para limitar a janela e aplicar estratégia de overflow
# Desativado: não impomos limite superior (sem 0.59), sequência é ilimitada.
ENFORCE_WINDOW = False
# Estratégia quando estourar a janela:
#   - "error": aborta com mensagem
#   - "overflow_to_900": usa 0.900, 0.901, ... sem afetar 0.1–0.59
OVERFLOW_STRATEGY = "error"

# ---------------- ALNULU ----------------
def calcular_alnulu(texto: str) -> int:
    mapa = {
        'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,
        'J':-10,'K':11,'L':12,'M':-13,'N':14,'O':15,'P':16,'Q':17,
        'R':18,'S':19,'T':20,'U':21,'V':-22,'W':23,'X':24,'Y':-25,'Z':26,
        '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
        '.':2,'!':3,'?':4,',':1,';':1,':':1,'-':1
    }
    equiv = {
        'Á':'A','À':'A','Â':'A','Ã':'A','Ä':'A',
        'É':'E','Ê':'E','È':'E','Ë':'E',
        'Í':'I','Ì':'I','Î':'I','Ï':'I',
        'Ó':'O','Ò':'O','Ô':'O','Õ':'O','Ö':'O',
        'Ú':'U','Ù':'U','Û':'U','Ü':'U',
        'Ç':'C','Ñ':'N',
        '4':'A','3':'E','1':'I','0':'O','5':'S','7':'T','2':'Z'
    }
    total = 0
    for ch in texto:
        base = equiv.get(ch, ch).upper()
        total += mapa.get(base, 0)
    return total

# ---------------- util ----------------
def tokenize(s: str):
    if not s:
        return []
    # Mantém sequências entre colchetes como um único token (ex.: [Nome do user])
    # Emoji/smileys comuns como um único token (^^, <3, :), :-), :D, ;), :P, etc.)
    # Palavras hifenizadas e com apóstrofo (vê-la, recebê-la, D') como um token
    # Depois divide em palavras e pontuação
    pattern = (
        r"\[[^\]]+\]"                        # marcadores entre colchetes
        r"|<3"                                   # coração
        r"|[:;][\-]?[)D(\[PpOo/\\]"       # smileys básicos :), :-), :D, :(, ;), :P, :o, :/
        r"|\^{2,}"                              # sequências de carets: ^^, ^^^
        r"|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)+"  # hifenizados: vê-la, recebe-la
        r"|[A-Za-zÀ-ÖØ-öø-ÿ]+['’]"                     # apóstrofo ao final: D'
        r"|[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+"                     # palavras/dígitos/underscore (com acentos)
        r"|[^\w\s]"                                 # qualquer pontuação isolada
    )
    return re.findall(pattern, s.replace('"',''), re.UNICODE)

def _strip_accents(s: str) -> str:
    if not s:
        return s
    # NFD para decompor acentos e remover marcas combinantes
    nf = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in nf if unicodedata.category(ch) != 'Mn')

def _canon_text(s: str) -> str:
    """Normalização agressiva para matching: remove acentos, casefold, normaliza pontuação comum e espaços."""
    if s is None:
        return ''
    s2 = s
    # Substituições comuns
    s2 = s2.replace('—', '-')
    s2 = s2.replace('“', '"').replace('”', '"').replace('’', "'")
    # Strip acentos e casefold
    s2 = _strip_accents(s2).casefold()
    # Normalizar espaços antes de pontuação
    s2 = re.sub(r"\s+([,.;:!?])", r"\1", s2)
    # Colapsar múltiplos espaços
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

# ---------------- parse Multivars (do prompt) ----------------
def parse_multivars_from_lines(lines: List[str]):
    """Extrai definições de Multivars a partir de um bloco opcional no input.
    Sintaxe:
    Multivars:
    [Fonte] Frase original => Alt 1 || Alt 2 || Alt 3
    """
    in_mv = False
    mv_lines: List[str] = []
    filtered: List[str] = []
    for ln in lines:
        if not in_mv and re.match(r"^\s*Multivars\s*:\s*$", ln, re.I):
            in_mv = True
            continue
        if in_mv:
            mv_lines.append(ln)
            continue
        filtered.append(ln)

    defs = []
    for raw in mv_lines:
        ln = raw.strip()
        if not ln or ln.startswith('#'):
            continue
        m = re.match(r"^\s*\[(?P<fonte>[^\]]+)\]\s*(?P<text>.+?)\s*=>\s*(?P<alts>.+)$", ln)
        if not m:
            # linha fora do padrão; ignorar silenciosamente
            continue
        fonte = m.group('fonte').strip()
        text = m.group('text').strip()
        alts = [a.strip() for a in m.group('alts').split('||') if a.strip()]
        if text and alts:
            defs.append({"fonte": fonte, "text": text, "alternativas": alts})
    return filtered, defs

# ---------------- parse (pensamento non-greedy) ----------------
def parse_block(lines):
    # Captura e remove bloco de Multivars do input (se houver)
    norm_lines, multivars_defs = parse_multivars_from_lines(lines)
    txt = "\n".join(norm_lines)

    def find_first(pat, flags=re.IGNORECASE | re.DOTALL):
        m = re.search(pat, txt, flags)
        return m.group(1).strip() if m else ""

    # -------- Entrada (entre 'Entrada:' e 'Saída:' ou fim) --------
    m_ent = re.search(r'(?ims)^\s*Entrada\s*:\s*([\s\S]*?)(?=^\s*Saída\s*:|\Z)', txt)
    entrada_block = (m_ent.group(1) if m_ent else "").strip()

    # Extrair e remover Reação/Contexto do bloco de Entrada
    reacao_entrada = ""
    m_re_in = re.search(r'(?im)^\s*Reação\s*:\s*([^\n\r]+)', entrada_block)
    if m_re_in:
        reacao_entrada = m_re_in.group(1).strip()
        entrada_block = entrada_block.replace(m_re_in.group(0), "").strip()
    contexto_entrada = ""
    m_ctx_in = re.search(r'(?im)^\s*Contexto\s*:\s*([^\n\r]+)', entrada_block)
    if m_ctx_in:
        contexto_entrada = m_ctx_in.group(1).strip()
        entrada_block = entrada_block.replace(m_ctx_in.group(0), "").strip()

    # Pensamento Interno (conteúdo entre aspas após rótulo) + linhas subsequentes até 'Saída:'
    m_pens = re.search(r'(?is)Pensamento\s*interno\s*:\s*"([\s\S]*?)"', txt)
    pensamento = ""
    if m_pens:
        pensamento = m_pens.group(1).strip()
        # tudo após o fechamento das aspas até 'Saída:' também pertence ao pensamento
        start_after = m_pens.end()
        m_saida_hdr = re.search(r'(?ims)^\s*Saída\s*:', txt)
        end_before = m_saida_hdr.start() if m_saida_hdr else len(txt)
        extra_thought = txt[start_after:end_before].strip()
        if extra_thought:
            pensamento = (pensamento + " " + extra_thought).strip()
        # remover do bloco de entrada qualquer trecho a partir de 'Pensamento interno:'
        entrada_block = re.sub(r'(?is)Pensamento\s*interno\s*:\s*"[\s\S]*?"[\s\S]*\Z', '', entrada_block).strip()

    # Linhas de Entrada e falas (— ...), agora sem o pensamento
    linhas_ent = [ln.rstrip() for ln in entrada_block.splitlines() if ln.strip()]
    idxs_fala_ent = [i for i, ln in enumerate(linhas_ent) if re.match(r'^\s*—', ln)]
    falas_entrada = [linhas_ent[i] for i in idxs_fala_ent]

    # TEXE: antes da 1ª fala; TEFIE: após a última fala
    if idxs_fala_ent:
        first_idx = idxs_fala_ent[0]
        last_idx = idxs_fala_ent[-1]
        texe_parts = [ln for ln in linhas_ent[:first_idx] if not re.match(r'^\s*—', ln)]
        tefie_parts = [ln for ln in linhas_ent[last_idx+1:] if not re.match(r'^\s*—', ln)]
        entrada_texto = " ".join(texe_parts).strip()
        tefie_texto = " ".join(tefie_parts).strip()
    else:
        entrada_texto = " ".join(linhas_ent).strip()
        tefie_texto = ""

    # -------- Saída (após 'Saída:') --------
    m_saida = re.search(r'(?ims)^\s*Saída\s*:\s*([\s\S]*?)\Z', txt)
    saida_block = (m_saida.group(1) if m_saida else "").strip()

    # Extrair e remover Reação/Contexto da Saída
    reac_saida = ""
    m_re_out = re.search(r'(?im)^\s*Reação\s*:\s*([^\n\r]+)', saida_block)
    if m_re_out:
        reac_saida = m_re_out.group(1).strip()
        saida_block = saida_block.replace(m_re_out.group(0), "").strip()
    contexto_saida = ""
    m_ctx_out = re.search(r'(?im)^\s*Contexto\s*:\s*([^\n\r]+)', saida_block)
    if m_ctx_out:
        contexto_saida = m_ctx_out.group(1).strip()
        saida_block = saida_block.replace(m_ctx_out.group(0), "").strip()

    # Linhas de Saída e falas (— ...)
    linhas_out = [ln.rstrip() for ln in saida_block.splitlines() if ln.strip()]
    idxs_fala_out = [i for i, ln in enumerate(linhas_out) if re.match(r'^\s*—', ln)]
    falas_saida = [linhas_out[i] for i in idxs_fala_out]
    if idxs_fala_out:
        first_idx = idxs_fala_out[0]
        last_idx = idxs_fala_out[-1]
        saida_texto_inicial = " ".join([ln for ln in linhas_out[:first_idx] if not re.match(r'^\s*—', ln)]).strip()
        texto_final_saida = " ".join([ln for ln in linhas_out[last_idx+1:] if not re.match(r'^\s*—', ln)]).strip()
    else:
        saida_texto_inicial = " ".join(linhas_out).strip()
        texto_final_saida = ""
    fs_line = " ".join(falas_saida)

    if not (entrada_texto or falas_entrada or tefie_texto):
        raise SystemExit('Erro: "Entrada" é obrigatória. Use o rótulo Entrada: no início do bloco.')

    return {
        "entrada": {
            "texto_inicial": entrada_texto,
            "falas": falas_entrada,
            "reacao": reacao_entrada,
            "contexto": contexto_entrada,
            "texto_final": tefie_texto,
            "pensamento": pensamento,
        },
        "saida": {
            "texto_inicial": saida_texto_inicial,
            "fala": fs_line,
            "reacao": reac_saida,
            "contexto": contexto_saida,
            "texto_final": texto_final_saida,
        },
        "multivars_defs": multivars_defs,
        "raw": txt,
    }

# ---------------- build & render (ordem garantida) ----------------
def build_render(tpl):
    part = WINDOW_PART

    # Detecta CAE/CAS e Multivars a partir de tokens com colchetes, preservando rótulo digitado
    def detect_campo(token: str, saida: bool):
        raw = token.strip()
        low = raw.lower()
        if low.startswith('[') and low.endswith(']'):
            inside = raw[1:-1].strip()
            # Padrão com prefixo (ex.: CAE: Estilo) e Multivars (abre/fecha)
            m = re.match(r'(?is)^(cae|cas|multivars)\s*:??\s*(.*)$', inside)
            if m:
                kind = m.group(1).upper()
                label = m.group(2).strip()
                if kind in ("CAE","CAS"):
                    # remover pontuação final comum
                    label = re.sub(r'[\s\.;:,]+$','', label)
                    return (kind, label)
                else:
                    # Suportar variações como "Multivars de entrada" e "Multivars de saída/saida"
                    inside_norm = _strip_accents(inside).casefold()
                    if 'multivars' in inside_norm:
                        if 'entrada' in inside_norm:
                            return ("MV_IN", None)
                        if 'saida' in inside_norm or 'saa' in inside_norm:
                            return ("MV_OUT", None)
                    return ("MV", None)
            # Padrões legados (fallback)
            legacy = inside.lower()
            if 'nome do user' in legacy:
                return ('CAE', 'Nome')
            if 'estilo do user' in legacy:
                return ('CAE', 'Estilo')
            if 'cabelo do user' in legacy:
                return ('CAE', 'Cabelo')
            if 'nome do narrador' in legacy:
                return ('CAS', 'Nome')
        return None

    # Gerencia sequência linear de marcadores 0.1, 0.2, ... e estados de spans
    cur = 0
    def next_marks(n):
        nonlocal cur
        marks = []
        for _ in range(n):
            cur += 1
            marks.append(f"{part}.{cur}")
        return marks

    # Estados para spans (Entrada)
    cae_active = {}
    cae_pending_open = set()
    cae_display = {}
    cae_spans = []
    last_mark_in = None

    # Estados para spans (Saída)
    cas_active = {}
    cas_pending_open = set()
    cas_display = {}
    cas_spans = []
    last_mark_out = None

    # Estados Multivars inline
    mv_in_pending_open = False
    mv_in_active = False
    mv_in_start = None
    mv_spans_in = []

    mv_out_pending_open = False
    mv_out_active = False
    mv_out_start = None
    mv_spans_out = []

    # Helper: aplica rótulos por FAIXA entre marcadores pareados [CAE:X] ... [CAE:X]
    def emit_tokens(words, keyname: str, saida_flag: bool):
        out = []
        nonlocal cae_active, cae_pending_open, cae_display, cae_spans, last_mark_in
        nonlocal cas_active, cas_pending_open, cas_display, cas_spans, last_mark_out
        nonlocal mv_in_pending_open, mv_in_active, mv_in_start, mv_spans_in
        nonlocal mv_out_pending_open, mv_out_active, mv_out_start, mv_spans_out
        for w in words:
            campo = detect_campo(w, saida=saida_flag)
            if campo:
                kind, label = campo
                if kind in ('CAE','CAS'):
                    canonical = label.casefold()
                    display = cae_display.get(canonical) if not saida_flag else cas_display.get(canonical)
                    if not display:
                        if saida_flag:
                            cas_display[canonical] = label
                        else:
                            cae_display[canonical] = label
                        display = label
                    # alterna (abre/fecha) a faixa daquele label por seção
                    if kind == 'CAE' and not saida_flag:
                        if canonical in cae_active:
                            if last_mark_in is not None:
                                cae_spans.append({"label": cae_active[canonical]["display"], "inicio": cae_active[canonical]["start"], "fim": last_mark_in})
                            cae_active.pop(canonical, None)
                            cae_pending_open.discard(canonical)
                        else:
                            cae_pending_open.add(canonical)
                    elif kind == 'CAS' and saida_flag:
                        if canonical in cas_active:
                            if last_mark_out is not None:
                                cas_spans.append({"label": cas_active[canonical]["display"], "inicio": cas_active[canonical]["start"], "fim": last_mark_out})
                            cas_active.pop(canonical, None)
                            cas_pending_open.discard(canonical)
                        else:
                            cas_pending_open.add(canonical)
                elif kind in ('MV','MV_IN','MV_OUT'):
                    # Toggle Multivars (entrada/saída explícito ou conforme seção atual)
                    force_in = (kind == 'MV_IN')
                    force_out = (kind == 'MV_OUT')
                    target_out = force_out or (kind == 'MV' and saida_flag and not force_in)
                    if not target_out:
                        # Entrada
                        if mv_in_active:
                            if last_mark_in is not None:
                                mv_spans_in.append({"inicio": mv_in_start, "fim": last_mark_in})
                            mv_in_active = False
                            mv_in_pending_open = False
                            mv_in_start = None
                        else:
                            mv_in_pending_open = True
                    else:
                        # Saída
                        if mv_out_active:
                            if last_mark_out is not None:
                                mv_spans_out.append({"inicio": mv_out_start, "fim": last_mark_out})
                            mv_out_active = False
                            mv_out_pending_open = False
                            mv_out_start = None
                        else:
                            mv_out_pending_open = True
                continue
            # token textual real
            m = next_marks(1)[0]
            item = {keyname: m, "t": w, "vars": ["0.0"]}
            # abrir spans pendentes nesta posição
            if not saida_flag and cae_pending_open:
                for canon in list(cae_pending_open):
                    cae_active[canon] = {"display": cae_display[canon], "start": m}
                    cae_pending_open.remove(canon)
            if saida_flag and cas_pending_open:
                for canon in list(cas_pending_open):
                    cas_active[canon] = {"display": cas_display[canon], "start": m}
                    cas_pending_open.remove(canon)
            # abrir Multivars pendente
            if not saida_flag and mv_in_pending_open:
                mv_in_start = m
                mv_in_active = True
                mv_in_pending_open = False
            if saida_flag and mv_out_pending_open:
                mv_out_start = m
                mv_out_active = True
                mv_out_pending_open = False
            # aplica rótulos ativos
            if not saida_flag and cae_active:
                labels = [cae_display.get(c, cae_active[c]["display"]) for c in sorted(cae_active.keys())]
                if labels:
                    item['cae'] = labels
            if saida_flag and cas_active:
                labels = [cas_display.get(c, cas_active[c]["display"]) for c in sorted(cas_active.keys())]
                if labels:
                    item['cas'] = labels
            out.append(item)
            if saida_flag:
                last_mark_out = m
            else:
                last_mark_in = m
        return out

    # Entrada
    TEXE_list = []
    entrada_words = tokenize(tpl["entrada"].get("texto_inicial", "")) if tpl["entrada"].get("texto_inicial") else []
    if entrada_words:
        TEXE_list = emit_tokens(entrada_words, "TEXE", False)

    FADEN_list = []
    for line in tpl["entrada"].get("falas", []):
        words = tokenize(line)
        if words:
            FADEN_list.extend(emit_tokens(words, "FADEN", False))

    RE_list = []
    reacao_words = tokenize(tpl["entrada"].get("reacao", "")) if tpl["entrada"].get("reacao") else []
    if reacao_words:
        RE_list = emit_tokens(reacao_words, "RE", False)

    CE_in_list = []
    ctx_words = tokenize(tpl["entrada"].get("contexto", "")) if tpl["entrada"].get("contexto") else []
    if ctx_words:
        CE_in_list = emit_tokens(ctx_words, "CE", False)

    TEFIE_list = []
    tefie_words = tokenize(tpl["entrada"].get("texto_final", "")) if tpl["entrada"].get("texto_final") else []
    if tefie_words:
        TEFIE_list = emit_tokens(tefie_words, "TEFIE", False)

    # Pensamento Interno
    PIDE_list = []
    pensamentos = tpl["entrada"].get("pensamento", "") or ""
    if pensamentos:
        pensamentos_norm = re.sub(r'[\r\n]+', ' ', pensamentos)
        pensamentos_norm = re.sub(r'["“”]+', '', pensamentos_norm)
        pensamentos_clean = re.sub(r'(?is)\[\s*(?:cae|cas)\s*:\s*[^\]]+\]', '', pensamentos_norm)
        parts = re.findall(r'[^.?!]+[.?!]|[^.?!]+$', pensamentos_clean)
        for raw in [p.strip() for p in parts if p.strip()]:
            clean_text = re.sub(r'\s+([,.;:!?])', r'\1', raw).strip()
            cae_labels = []
            for m in re.finditer(r'(?is)\[\s*cae\s*:\s*([^\]]+)\]', pensamentos_norm):
                label = re.sub(r'[\s\.;:,]+$', '', m.group(1).strip())
                if label and label not in cae_labels:
                    cae_labels.append(label)
            mark = next_marks(1)[0]
            item = {"PIDE": mark, "t": clean_text}
            if cae_labels:
                item["cae"] = cae_labels
            PIDE_list.append(item)

    Total_de_Entrada = [
        *[x["TEXE"] for x in TEXE_list],
        *[x["FADEN"] for x in FADEN_list],
        *[x["RE"] for x in RE_list],
        *[x["CE"] for x in CE_in_list],
        *[x["TEFIE"] for x in TEFIE_list],
        *[p["PIDE"] for p in PIDE_list],
    ]

    # Saída
    TEXIS_list = []
    texis_words = tokenize(tpl["saida"].get("texto_inicial", "")) if tpl["saida"].get("texto_inicial") else []
    if texis_words:
        TEXIS_list = emit_tokens(texis_words, "TEXIS", True)

    FS_list = []
    fs_words = tokenize(tpl["saida"].get("fala", "")) if tpl["saida"].get("fala") else []
    if fs_words:
        FS_list = emit_tokens(fs_words, "FS", True)

    RS_list = []
    rs_words = tokenize(tpl["saida"].get("reacao", "")) if tpl["saida"].get("reacao") else []
    if rs_words:
        RS_list = emit_tokens(rs_words, "RS", True)

    CE_out_list = []
    ceo_words = tokenize(tpl["saida"].get("contexto", "")) if tpl["saida"].get("contexto") else []
    if ceo_words:
        CE_out_list = emit_tokens(ceo_words, "CE", True)

    TEXFS_list = []
    texfs_words = tokenize(tpl["saida"].get("texto_final", "")) if tpl["saida"].get("texto_final") else []
    if texfs_words:
        TEXFS_list = emit_tokens(texfs_words, "TEXFS", True)

    Total_de_Saida = [
        *[x["TEXIS"] for x in TEXIS_list],
        *[x["FS"] for x in FS_list],
        *[x["RS"] for x in RS_list],
        *[x["CE"] for x in CE_out_list],
        *[x["TEXFS"] for x in TEXFS_list],
    ]

    # Sem limite superior: sequência de marcadores é ilimitada (0.1, 0.2, ...)

    # ALNULU do bloco
    alnulu_src = [
        tpl["entrada"].get("texto_inicial", ""),
        " ".join(tpl["entrada"].get("falas", [])),
        tpl["entrada"].get("reacao", ""),
        tpl["entrada"].get("contexto", ""),
        tpl["entrada"].get("texto_final", ""),
        tpl["entrada"].get("pensamento", ""),
        tpl["saida"].get("texto_inicial", ""),
        tpl["saida"].get("fala", ""),
        tpl["saida"].get("reacao", ""),
        tpl["saida"].get("contexto", ""),
        tpl["saida"].get("texto_final", ""),
    ]
    alnulu_total_bloco = calcular_alnulu(" ".join([s for s in alnulu_src if s]))

    # Fechar spans que ficaram abertos até o final
    for canon, info in list(cae_active.items()):
        if last_mark_in is not None:
            cae_spans.append({"label": info["display"], "inicio": info["start"], "fim": last_mark_in})
    for canon, info in list(cas_active.items()):
        if last_mark_out is not None:
            cas_spans.append({"label": info["display"], "inicio": info["start"], "fim": last_mark_out})
    # Fechar Multivars remanescentes
    if mv_in_active and last_mark_in is not None:
        mv_spans_in.append({"inicio": mv_in_start, "fim": last_mark_in})
    if mv_out_active and last_mark_out is not None:
        mv_spans_out.append({"inicio": mv_out_start, "fim": last_mark_out})

    ida = {
        "IDA": {
            "IM": {
                "0": {
                    "nome": "",
                    "blocos": [
                        {
                            "bloco_id": 1,
                            "Entrada": {
                                "Texto Inicial DE ENTRADA": TEXE_list,
                                "Fala DE ENTRADA": FADEN_list,
                                "Reação": RE_list,
                                "Texto Final de Entrada": TEFIE_list,
                                "Contexto": CE_in_list,
                                "Sentimento de Entrada": {"SDE": "0.0", "t": "Neutro", "Tendência de Entrada": 0.0},
                            },
                            "Pensamento Interno": PIDE_list,
                            "Total de Entrada": Total_de_Entrada,
                            "Saída": {
                                "Texto Inicial de SAÍDA": TEXIS_list,
                                "Fala de Saída": FS_list,
                                "Reação de Saída": RS_list,
                                "Texto Final de Saída": TEXFS_list,
                                "Contexto de Saída": CE_out_list,
                            },
                            "Sentimento da Saída": {"SDS": "0.0", "t": "Neutro", "Tendência da Saída": 0.0, "Ressonância": False},
                            "Total de Saída": Total_de_Saida,
                        }
                    ],
                }
            }
        }
    }

    # Construir índices para materializar spans (texto por marcador)
    entrada_index = []  # (mark, t, secao)
    for tkn in TEXE_list:
        entrada_index.append((tkn["TEXE"], tkn["t"], "Texto de Entrada Inicial"))
    for tkn in FADEN_list:
        entrada_index.append((tkn["FADEN"], tkn["t"], "Fala de Entrada"))
    for tkn in RE_list:
        entrada_index.append((tkn["RE"], tkn["t"], "Reação"))
    for tkn in CE_in_list:
        entrada_index.append((tkn["CE"], tkn["t"], "Contexto"))
    for tkn in TEFIE_list:
        entrada_index.append((tkn["TEFIE"], tkn["t"], "Texto Final de Entrada"))

    saida_index = []
    for tkn in TEXIS_list:
        saida_index.append((tkn["TEXIS"], tkn["t"], "Texto Inicial de Saída"))
    for tkn in FS_list:
        saida_index.append((tkn["FS"], tkn["t"], "Fala de Saída"))
    for tkn in RS_list:
        saida_index.append((tkn["RS"], tkn["t"], "Reação de Saída"))
    for tkn in CE_out_list:
        saida_index.append((tkn["CE"], tkn["t"], "Contexto de Saída"))
    for tkn in TEXFS_list:
        saida_index.append((tkn["TEXFS"], tkn["t"], "Texto Final de Saída"))

    def materializar_span(sp, index_list):
        inicio = sp["inicio"]; fim = sp["fim"]
        texts = []
        fonte = None
        on = False
        for mark, tt, sec in index_list:
            if mark == inicio:
                on = True
                fonte = sec
            if on:
                texts.append(tt)
            if mark == fim:
                break
        joined = " ".join(texts).strip()
        joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
        return fonte, joined

    caracteristicas_entrada = []
    for sp in cae_spans:
        fonte, toks = materializar_span(sp, entrada_index)
        # Sanitizar rótulo (remover ':' inicial, espaços)
        lbl = re.sub(r"^\s*:\s*", "", sp.get("label", "").strip())
        caracteristicas_entrada.append({
            "CAE": lbl,
            "Fonte": fonte or "Entrada",
            "range": {"inicio": sp["inicio"], "fim": sp["fim"]},
            "Tokens": toks,
        })

    caracteristicas_saida = []
    for sp in cas_spans:
        fonte, toks = materializar_span(sp, saida_index)
        lbl = re.sub(r"^\s*:\s*", "", sp.get("label", "").strip())
        caracteristicas_saida.append({
            "CAS": lbl,
            "Fonte": fonte or "Saída",
            "range": {"inicio": sp["inicio"], "fim": sp["fim"]},
            "Tokens": toks,
        })

    # ---------- Multivars (inline + bloco Multivars:) ----------
    def normalize_fonte(s: str) -> str:
        s0 = (s or '').strip()
        # Mapeamentos canônicos esperados no IM/UM
        canon = {
            'texto de entrada inicial': 'Texto de Entrada Inicial',
            'fala de entrada': 'Fala de Entrada',
            'reação': 'Reação',
            'contexto': 'Contexto',
            'texto final de entrada': 'Texto Final de Entrada',
            'pensamento interno': 'Pensamento Interno',
            'texto inicial de saída': 'Texto Inicial de Saída',
            'fala de saída': 'Fala de Saída',
            'reação de saída': 'Reação de Saída',
            'contexto de saída': 'Contexto de Saída',
            'texto final de saída': 'Texto Final de Saída',
        }
        key = s0.casefold()
        return canon.get(key, s0)

    # Mapear fonte -> (lista de tokens, key de marcador)
    fonte_map: Dict[str, Tuple[List[dict], str]] = {
        'Texto de Entrada Inicial': (TEXE_list, 'TEXE'),
        'Fala de Entrada': (FADEN_list, 'FADEN'),
        'Reação': (RE_list, 'RE'),
        'Contexto': (CE_in_list, 'CE'),
        'Texto Final de Entrada': (TEFIE_list, 'TEFIE'),
        'Pensamento Interno': (PIDE_list, 'PIDE'),
        'Texto Inicial de Saída': (TEXIS_list, 'TEXIS'),
        'Fala de Saída': (FS_list, 'FS'),
        'Reação de Saída': (RS_list, 'RS'),
        'Contexto de Saída': (CE_out_list, 'CE'),
        'Texto Final de Saída': (TEXFS_list, 'TEXFS'),
    }

    def find_span(tokens: List[dict], key: str, text: str):
        if not tokens or not text:
            return None
        # Normalizar frase alvo
        seq_tokens = tokenize(text)
        if not seq_tokens:
            return None
        seq_norm = _canon_text(" ".join(seq_tokens))

        # Preparar lista de tokens normalizados e também uma versão que ignora '?' isolado
        tvals = [x['t'] for x in tokens]
        tcanon = [_canon_text(t) for t in tvals]

        # Janela deslizante: juntar tokens normalizados e comparar à frase alvo normalizada
        L = len(tcanon)
        # Limite máximo de extensão da janela: até o dobro do número de tokens alvo + 3 (para absorver '?')
        max_win = max(1, len(seq_tokens) * 2 + 3)
        for i in range(L):
            joined = []
            # Construir incrementalmente e fazer early break por comprimento
            for j in range(i, min(L, i + max_win)):
                tc = tcanon[j]
                # Ignorar tokens que viraram apenas '?' após normalização
                if tc == '?':
                    continue
                joined.append(tc)
                cand = " ".join(joined)
                # Pequena otimização: se cand excede muito o tamanho, podemos parar
                if len(cand) > len(seq_norm) + 5:
                    break
                if cand == seq_norm:
                    # Mapear de volta aos marcadores reais usando índices i..j (inclui tokens ignorados no meio)
                    # Retroceder j real até último índice <= j que não foi apenas '?' para obter fim
                    end_k = j
                    # início no primeiro token considerado (i) ou avançar até primeiro não '?' real
                    start_k = i
                    # Ajustar início para pular tokens '?' isolados se existirem
                    while start_k < L and _canon_text(tvals[start_k]) == '?':
                        start_k += 1
                    while end_k >= start_k and _canon_text(tvals[end_k]) == '?':
                        end_k -= 1
                    if start_k <= end_k:
                        inicio = tokens[start_k][key]
                        fim = tokens[end_k][key]
                        return inicio, fim
        return None

    def collect_traits_in_range(tokens: List[dict], key: str, saida_flag: bool, inicio: str, fim: str):
        """Varre tokens entre [inicio..fim] (inclusive) e agrega rótulos CAE/CAS ativos."""
        traits = set()
        on = False
        lab_key = 'cas' if saida_flag else 'cae'
        for tok in tokens:
            if key not in tok:
                continue
            mk = tok[key]
            if mk == inicio:
                on = True
            if on:
                for lb in tok.get(lab_key, []) or []:
                    traits.add(lb)
            if mk == fim:
                break
        return sorted(traits)

    def _mk_range_key(rng: dict) -> str:
        return f"{rng.get('inicio','')}-{rng.get('fim','')}"

    mv_entries_dict: Dict[str, dict] = {}

    # a) Spans inline marcados com [Multivars]
    def _add_mv_from_span(span: dict, idx_list, fonte_hint_saida: bool):
        inicio = span["inicio"]; fim = span["fim"]
        fonte_detect, text_mat = materializar_span({"inicio": inicio, "fim": fim}, idx_list)
        if not fonte_detect:
            fonte_detect = "Texto Inicial de Saída" if fonte_hint_saida else "Texto de Entrada Inicial"
        toks_list, key = fonte_map.get(fonte_detect, (None, None))
        if not toks_list:
            return
        saida_flag = 'Saída' in fonte_detect
        traits_list = collect_traits_in_range(toks_list, key, saida_flag, inicio, fim)
        traits_tipo = 'CAS' if saida_flag else 'CAE'
        entry = {
            "Fonte": fonte_detect,
            "range": {"inicio": inicio, "fim": fim},
            "text": text_mat,
            "alternativas": [],
            "traits": {"tipo": traits_tipo, "labels": traits_list},
        }
        mv_entries_dict[_mk_range_key(entry["range"])] = entry

    for sp in mv_spans_in:
        _add_mv_from_span(sp, entrada_index, False)
    for sp in mv_spans_out:
        _add_mv_from_span(sp, saida_index, True)

    # b) Mesclar com bloco Multivars:
    for d in tpl.get('multivars_defs', []) or []:
        fonte_c = normalize_fonte(d.get('fonte'))
        toks_list, key = fonte_map.get(fonte_c, (None, None))
        if not toks_list:
            continue
        span = find_span(toks_list, key, d.get('text', ''))
        if not span:
            continue
        inicio, fim = span
        idx_list = saida_index if 'Saída' in fonte_c else entrada_index
        _, text_mat = materializar_span({"inicio": inicio, "fim": fim}, idx_list)
        saida_flag = 'Saída' in fonte_c
        traits_list = collect_traits_in_range(toks_list, key, saida_flag, inicio, fim)
        traits_tipo = 'CAS' if saida_flag else 'CAE'
        rkey = _mk_range_key({"inicio": inicio, "fim": fim})
        if rkey in mv_entries_dict:
            ex = mv_entries_dict[rkey]
            alts = ex.get("alternativas", [])
            for a in d.get('alternativas', []) or []:
                if a and a not in alts:
                    alts.append(a)
            ex["alternativas"] = alts
            if not ex.get("text"):
                ex["text"] = text_mat or d.get('text', '')
            ex["traits"] = {"tipo": traits_tipo, "labels": traits_list}
        else:
            mv_entries_dict[rkey] = {
                "Fonte": fonte_c,
                "range": {"inicio": inicio, "fim": fim},
                "text": text_mat or d.get('text', ''),
                "alternativas": d.get('alternativas', []),
                "traits": {"tipo": traits_tipo, "labels": traits_list},
            }

    mv_entries = list(mv_entries_dict.values())

    # Transformar Multivars em duas listas simples de frases (Entrada/Saída), com deduplicação
    mv_frases_entrada: List[str] = []
    mv_frases_saida: List[str] = []
    seen_e = set()
    seen_s = set()
    for mv in mv_entries:
        fonte = mv.get("Fonte", "") or ""
        base = mv.get("text", "") or ""
        alts = mv.get("alternativas", []) or []
        alvo_saida = ('Saída' in fonte)
        alvo_list = mv_frases_saida if alvo_saida else mv_frases_entrada
        alvo_seen = seen_s if alvo_saida else seen_e
        # incluir base e alternativas
        for t in ([base] + [a for a in alts if a]):
            if not t:
                continue
            key = _canon_text(t)
            if key in alvo_seen:
                continue
            alvo_seen.add(key)
            alvo_list.append(t)

    if not mv_frases_entrada:
        mv_frases_entrada = ["0.0"]
    if not mv_frases_saida:
        mv_frases_saida = ["0.0"]

    um = {
        "UM": {
            "0": {
                "Universo Mãe": "Interações",
                "blocos": [
                    {
                        "bloco_id": 1,
                        "Características de Entrada": caracteristicas_entrada,
                        "Multivars de Entrada": mv_frases_entrada,
                        "Total de Entrada": Total_de_Entrada,
                        "ultimo_child_entrada": Total_de_Entrada[-1] if Total_de_Entrada else None,
                        "Características de Saída": caracteristicas_saida,
                        "Multivars de Saída": mv_frases_saida,
                        "Total de Saída": Total_de_Saida,
                        "ultimo_child_saida": Total_de_Saida[-1] if Total_de_Saida else None,
                        "alnulu_total_bloco": alnulu_total_bloco,
                    }
                ],
            }
        }
    }

    return ida, um, Total_de_Entrada, Total_de_Saida, alnulu_total_bloco

# ---------------- writer: modelo (arrays 1 objeto por linha) ----------------
def write_ida_modelo(ida_obj, fp, indent=2):
    sp = " " * indent
    sp2 = sp * 2
    sp3 = sp * 3
    sp4 = sp * 4
    jd = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))

    IM = ida_obj.get("IDA", {}).get("IM", {})
    if not IM:
        fp.write(jd(ida_obj))
        return
    key0 = next(iter(IM.keys()))
    root0 = IM[key0]
    blocos = root0.get("blocos", [])

    fp.write('{' + "\n")
    fp.write(f"{sp}\"IDA\": {{\n")
    fp.write(f"{sp2}\"IM\": {{\n")
    fp.write(f"{sp3}{jd(key0)}: {{\n")
    fp.write(f"{sp4}\"nome\": {jd(root0.get('nome', ''))},\n")
    fp.write(f"{sp4}\"blocos\": [\n")

    for bi, bloco in enumerate(blocos):
        if bi > 0:
            fp.write(f"{sp4},\n")
        fp.write(f"{sp4}{{\n")
        fp.write(f"{sp4}{sp}\"bloco_id\": {jd(bloco.get('bloco_id'))},\n")

        # Entrada
        ent = bloco.get("Entrada", {})
        fp.write(f"{sp4}{sp}\"Entrada\": {{\n")
        def write_array(name, last=False):
            arr = ent.get(name, [])
            fp.write(f"{sp4}{sp}{sp}{jd(name)}: [\n")
            for i, obj in enumerate(arr):
                line = jd(obj)
                sep = "," if i < len(arr) - 1 else ""
                fp.write(f"{sp4}{sp}{sp}{sp}{line}{sep}\n")
            if last:
                fp.write(f"{sp4}{sp}{sp}]\n")
            else:
                fp.write(f"{sp4}{sp}{sp}],\n")
        write_array("Texto Inicial DE ENTRADA")
        write_array("Fala DE ENTRADA")
        write_array("Reação")
        write_array("Contexto")
        write_array("Texto Final de Entrada", last=True)
        fp.write(f"{sp4}{sp}}},\n")

        # Pensamento Interno
        pides = bloco.get("Pensamento Interno", [])
        fp.write(f"{sp4}{sp}\"Pensamento Interno\": [\n")
        for i, obj in enumerate(pides):
            line = jd(obj)
            sep = "," if i < len(pides) - 1 else ""
            fp.write(f"{sp4}{sp}{sp}{line}{sep}\n")
        fp.write(f"{sp4}{sp}],\n")

        # Total de Entrada
        totE = bloco.get("Total de Entrada", [])
        fp.write(f"{sp4}{sp}\"Total de Entrada\": {jd(totE)},\n")

        # Saída
        sai = bloco.get("Saída", {})
        fp.write(f"{sp4}{sp}\"Saída\": {{\n")
        def write_array_s(name, last=False):
            arr = sai.get(name, [])
            fp.write(f"{sp4}{sp}{sp}{jd(name)}: [\n")
            for i, obj in enumerate(arr):
                line = jd(obj)
                sep = "," if i < len(arr) - 1 else ""
                fp.write(f"{sp4}{sp}{sp}{sp}{line}{sep}\n")
            if last:
                fp.write(f"{sp4}{sp}{sp}]\n")
            else:
                fp.write(f"{sp4}{sp}{sp}],\n")
        write_array_s("Texto Inicial de SAÍDA")
        write_array_s("Fala de Saída")
        write_array_s("Reação de Saída")
        write_array_s("Contexto de Saída")
        write_array_s("Texto Final de Saída", last=True)
        fp.write(f"{sp4}{sp}}},\n")

        # Sentimento, Ressonância, Total de Saída
        fp.write(f"{sp4}{sp}\"Sentimento da Saída\": {jd(bloco.get('Sentimento da Saída', {}))},\n")
        fp.write(f"{sp4}{sp}\"Ressonância\": {jd(bloco.get('Ressonância', False))},\n")
        fp.write(f"{sp4}{sp}\"Total de Saída\": {jd(bloco.get('Total de Saída', []))}\n")

        fp.write(f"{sp4}}}\n")

    fp.write(f"{sp4}]\n")
    fp.write(f"{sp3}}}\n")
    fp.write(f"{sp2}}}\n")
    fp.write(f"{sp}}}\n")

# ---------------- main ----------------
def main():
    print("Cole o bloco (termine com linha vazia):")
    print("Comandos: :clear (limpar buffer), :clearjson (apagar JSONs), :exit (sair), :help (ver comandos), :compact (JSON minificado), :pretty (JSON identado), :modelo (arrays 1 obj/linha)")
    print("Multivars: ao final, insira 'Multivars:' e linhas do tipo [Fonte] frase => alt1 || alt2 ...")
    lines = []
    compact_mode = False
    modelo_mode = False
    try:
        while True:
            ln = input()
            cmd = ln.strip().lower()
            if cmd in (":help", "/help", "help"):
                print("Comandos disponíveis:\n  :clear      -> limpa o buffer atual e o console\n  :clearjson  -> apaga/trunca inconsciente.json e adam_memoria.json\n  :exit       -> sai do programa\n  :compact    -> salva JSON minificado (horizontal)\n  :pretty     -> salva JSON identado (padrão)\n  :modelo     -> arrays com 1 objeto por linha (modelo)\n  (Finalize o bloco com uma linha vazia)\n\nMultivars:\n  No final do bloco, use:\n  Multivars:\n  [Texto de Entrada Inicial] Era de manhã. => Havia amanhecido. || O dia havia começado.")
                continue
            if cmd in (":clear", "/clear", "clear"):
                lines = []
                try:
                    os.system('cls')
                except Exception:
                    pass
                print("Buffer limpo. Cole o bloco (termine com linha vazia):")
                continue
            if cmd in (":clearjson", "/clearjson", "clearjson"):
                try:
                    with open(OUT_IDA, 'w', encoding='utf-8') as f:
                        f.write("")
                    with open(OUT_UM, 'w', encoding='utf-8') as f:
                        f.write("")
                    print(f"Arquivos limpos: {OUT_IDA}, {OUT_UM}")
                except Exception as e:
                    print(f"Falha ao limpar JSONs: {e}")
                continue
            if cmd in (":compact", "/compact", "compact"):
                compact_mode = True
                modelo_mode = False
                print("Modo de saída: JSON minificado (horizontal)")
                continue
            if cmd in (":pretty", "/pretty", "pretty"):
                compact_mode = False
                modelo_mode = False
                print("Modo de saída: JSON identado")
                continue
            if cmd in (":modelo", "/modelo", "modelo"):
                modelo_mode = True
                compact_mode = False
                print("Modo de saída: arrays com 1 objeto por linha (modelo)")
                continue
            if cmd in (":exit", "/exit", "exit", "quit"):
                print("Saindo...")
                return
            if not ln and lines:
                break
            if not ln:
                continue
            lines.append(ln)
    except EOFError:
        pass

    tpl = parse_block(lines)
    ida, um, total_ent, total_sai, alnulu_val = build_render(tpl)

    with open(OUT_IDA, "w", encoding="utf-8") as f:
        if modelo_mode:
            write_ida_modelo(ida, f, indent=2)
        elif compact_mode:
            json.dump(ida, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(ida, f, ensure_ascii=False, indent=2)
    with open(OUT_UM, "w", encoding="utf-8") as f:
        if compact_mode:
            json.dump(um, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(um, f, ensure_ascii=False, indent=2)

    print(f"Gerados: {OUT_IDA} e {OUT_UM}")
    print(f"alnulu_total_bloco: {alnulu_val}")

if __name__ == "__main__":
    # Garantir UTF-8 no stdin/stdout (Windows/PowerShell)
    try:
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()



