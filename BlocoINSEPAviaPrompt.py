#!/usr/bin/env python3
# coding: utf-8
"""
BlocoINSEPAviaPrompt_final.py
- Lê bloco por stdin (termine com linha vazia)
- Gera inconsciente.json (texto no formato exato que você exigiu)
- Gera adam_memoria.json (JSON válido) contendo alnulu_total_bloco e resumo do bloco
- Coloca "Características de usuário" imediatamente após "Contexto"
- Sem inferência: Características de usuário ficam em 0.0 se não fornecidas
- ALNULU calculado a partir do conteúdo literal do bloco
- Quando não houver TEXTO FINAL, emite o placeholder exigido:
  "TEXTO FINAL": [ { "TEXF":"0.0", "tokens": [ {"TEX":"0.0","t":"0.0","vars":["0.0"]} ] } ]
"""
import re, sys, json, os

OUT_IDA = "inconsciente.json"
OUT_UM = "adam_memoria.json"

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
    # Depois divide em palavras e pontuação
    pattern = (
        r"\[[^\]]+\]"                  # marcadores entre colchetes
        r"|<3"                             # coração
        r"|[:;][\-]?[)D(\[PpOo/\\]"   # smileys básicos :), :-), :D, :(, ;), :P, :o, :/ 
        r"|\^{2,}"                        # sequências de carets: ^^, ^^^
        r"|\w+"                           # palavras/dígitos/underscore
        r"|[^\w\s]"                       # qualquer pontuação isolada
    )
    return re.findall(pattern, s.replace('"',''), re.UNICODE)

def gen_markers(base: str, n: int):
    part, _, suf = base.partition(".")
    try:
        cur = int(suf or "0")
    except:
        cur = 0
    return [f"{part}.{cur+i+1}" for i in range(n)]

# ---------------- parse (pensamento non-greedy) ----------------
def parse_block(lines):
    txt = "\n".join(lines)

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

    # Linhas de Entrada e falas (— ...)
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

    # Pensamento Interno (conteúdo entre aspas após rótulo)
    m_pens = re.search(r'(?is)Pensamento\s*interno\s*:\s*"([\s\S]*?)"', txt)
    pensamento = m_pens.group(1).strip() if m_pens else ""

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
        "raw": txt,
    }

# ---------------- build & render (ordem garantida, placeholder TEXTO FINAL) ----------------
def build_render(tpl):
    part = "0"

    # Detecta CAE/CAS a partir de tokens com colchetes, preservando rótulo como digitado
    def detect_campo(token: str, saida: bool):
        raw = token.strip()
        low = raw.lower()
        if low.startswith('[') and low.endswith(']'):
            inside = raw[1:-1].strip()
            # Padrão com prefixo (ex.: CAE: Estilo)
            m = re.match(r'(?is)^(cae|cas)\s*:\s*(.+)$', inside)
            if m:
                kind = m.group(1).upper()
                label = m.group(2).strip()
                # remover pontuação final comum
                label = re.sub(r'[\s\.;:,]+$','', label)
                return (kind, label)
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

    # Gerencia sequência linear de marcadores 0.1, 0.2, ...
    cur = 0
    def next_marks(n):
        nonlocal cur
        marks = [f"{part}.{i}" for i in range(cur + 1, cur + n + 1)]
        cur += n
        return marks

    # Helper: aplica rótulos como estado ativo até próximo marcador ou fim de sentença
    def emit_tokens(words, keyname: str, saida_flag: bool):
        out = []
        active_cae = []  # lista de rótulos CAE ativos
        active_cas = []  # lista de rótulos CAS ativos
        for w in words:
            campo = detect_campo(w, saida=saida_flag)
            if campo:
                kind, label = campo
                if kind == 'CAE':
                    active_cae = [label]
                elif kind == 'CAS':
                    active_cas = [label]
                continue  # não emite token para marcador
            # token textual real
            m = next_marks(1)[0]
            item = {keyname: m, "t": w, "vars": ["0.0"]}
            # aplica rótulos ativos do contexto adequado
            if not saida_flag and active_cae:
                item.setdefault('cae', []).extend(active_cae)
            if saida_flag and active_cas:
                item.setdefault('cas', []).extend(active_cas)
            out.append(item)
            # Se for fim de sentença, limpar rótulos ativos
            if w in ['.', '!', '?']:
                active_cae = []
                active_cas = []
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

    # Pensamento Interno (frases -> PIDE) com extração de CAE dentro do pensamento
    PIDE_list = []
    pensamentos = tpl["entrada"].get("pensamento", "") or ""
    if pensamentos:
        parts = re.findall(r'[^.?!]+[.?!]|[^.?!]+$', pensamentos)
        for raw in [p for p in parts if p.strip()]:
            tokens = tokenize(raw)
            cae_labels = []
            clean_tokens = []
            for tok in tokens:
                lab = detect_campo(tok, saida=False)
                if lab and lab[0] == 'CAE':
                    if lab[1] not in cae_labels:
                        cae_labels.append(lab[1])
                    continue
                clean_tokens.append(tok)
            clean_text = " ".join(clean_tokens).strip()
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
                                "Contexto": CE_in_list,
                                "Texto Final de Entrada": TEFIE_list,
                            },
                            "Pensamento Interno": PIDE_list,
                            "Total de Entrada": Total_de_Entrada,
                            "Saída": {
                                "Texto Inicial de SAÍDA": TEXIS_list,
                                "Fala de Saída": FS_list,
                                "Reação de Saída": RS_list,
                                "Contexto de Saída": CE_out_list,
                                "Texto Final de Saída": TEXFS_list,
                            },
                            "Sentimento da Saída": {"SDS": "0.0", "t": "Neutro", "Tendência da Saída": 0.0},
                            "Ressonância": False,
                            "Total de Saída": Total_de_Saida,
                        }
                    ],
                }
            }
        }
    }

    um = {
        "UM": {
            "0": {
                "Universo Mãe": "Interações",
                "blocos": [
                    {
                        "bloco_id": 1,
                        "Total de Entrada": Total_de_Entrada,
                        "ultimo_child_entrada": Total_de_Entrada[-1] if Total_de_Entrada else None,
                        "Multivars": ["0.0"],
                        "Total de Saída": Total_de_Saida,
                        "ultimo_child_saida": Total_de_Saida[-1] if Total_de_Saida else None,
                        "alnulu_total_bloco": alnulu_total_bloco,
                    }
                ],
            }
        }
    }

    return ida, um, Total_de_Entrada, Total_de_Saida, alnulu_total_bloco

# ---------------- main ----------------
def main():
    print("Cole o bloco (termine com linha vazia):")
    print("Comandos: :clear (limpar), :exit (sair), :help (ver comandos)")
    lines = []
    try:
        while True:
            ln = input()
            cmd = ln.strip().lower()
            if cmd in (":help", "/help", "help"):
                print("Comandos disponíveis:\n  :clear  -> limpa o buffer atual e o console\n  :exit   -> sai do programa\n  (Finalize o bloco com uma linha vazia)")
                continue
            if cmd in (":clear", "/clear", "clear"):
                lines = []
                try:
                    os.system('cls')
                except Exception:
                    pass
                print("Buffer limpo. Cole o bloco (termine com linha vazia):")
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
        json.dump(ida, f, ensure_ascii=False, indent=2)
    with open(OUT_UM, "w", encoding="utf-8") as f:
        json.dump(um, f, ensure_ascii=False, indent=2)

    print(f"Gerados: {OUT_IDA} e {OUT_UM}")
    print(f"alnulu_total_bloco: {alnulu_val}")

if __name__ == "__main__":
    main()



