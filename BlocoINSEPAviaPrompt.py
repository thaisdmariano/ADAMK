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
import re, sys, json

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
    if not s: return []
    return re.findall(r"\w+|[^\w\s]", s.replace('"',''), re.UNICODE)

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
    def find_first(pat, flags=re.IGNORECASE|re.DOTALL):
        m = re.search(pat, txt, flags)
        return m.group(1).strip() if m else ""
    entrada = find_first(r'Entrada:\s*"([^"]*?)"')
    reacao = find_first(r'(?m)^Reação:\s*([^\n\r]+)')
    contexto = find_first(r'Contexto:\s*([^\n\r]+)')
    pensamento = find_first(r'Pensamento Interno:\s*"([^"]*?)"(?=\n^Saída:|\n^Saída\s*:|\Z)', flags=re.MULTILINE|re.DOTALL)
    # captura bloco Saída para extrair Nome se estiver lá
    saida_nome = ""
    m_saida = re.search(r'(?m)^Saída:\s*([\s\S]*?)(?=\n^[A-Za-z].*:|\Z)', txt)
    if m_saida:
        bloco_saida_text = m_saida.group(1)
        m_nome = re.search(r'Nome:\s*"([^"]*)"', bloco_saida_text)
        if m_nome:
            saida_nome = m_nome.group(1).strip()
    if not saida_nome:
        saida_nome = find_first(r'Nome:\s*"([^"]*)"')
    texto_inicial_raw = find_first(r'Texto Inicial:\s*"([^"]*?)"') or find_first(r'Texto Inicial:\s*([^\n\r]+)')
    fs_line = find_first(r'—\s*([^\n\r]+)')
    # última Reação global (para RS)
    reac_saida = ""
    for m in re.finditer(r'(?m)^Reação:\s*([^\n\r]+)', txt):
        reac_saida = m.group(1).strip()
    texto_final_raw = find_first(r'Texto final:\s*"([^"]*?)"') or ""
    if not entrada:
        raise SystemExit('Erro: "Entrada" é obrigatória. Use: Entrada: "..."')
    return {
        "entrada": {"texto": entrada, "reacao": reacao, "contexto": contexto, "pensamento": pensamento},
        "saida": {"nome": saida_nome, "texto_inicial_raw": texto_inicial_raw, "fs_line": fs_line, "reac_saida": reac_saida, "texto_final_raw": texto_final_raw},
        "raw": txt
    }

# ---------------- build & render (ordem garantida, placeholder TEXTO FINAL) ----------------
def build_render(tpl):
    part = "0"

    # Entrada tokens / markers
    entrada_words = tokenize(tpl["entrada"]["texto"])
    Em = gen_markers(f"{part}.0", len(entrada_words))
    Entrada = [{"E": m, "t": w, "vars": ["0.0"]} for m, w in zip(Em, entrada_words)]

    # Reação entrada
    reacao_words = tokenize(tpl["entrada"]["reacao"])
    REm = gen_markers(Em[-1] if Em else f"{part}.0", len(reacao_words))
    Reacao = [{"RE": m, "t": w, "vars": ["0.0"]} for m, w in zip(REm, reacao_words)]

    # Contexto
    ctx_words = tpl["entrada"]["contexto"].split() if tpl["entrada"]["contexto"] else []
    CEm = gen_markers(REm[-1] if REm else (Em[-1] if Em else f"{part}.0"), len(ctx_words))
    Contexto = [{"CE": m, "t": w} for m, w in zip(CEm, ctx_words)]

    # Pensamento Interno
    PIm = gen_markers(CEm[-1] if CEm else (REm[-1] if REm else (Em[-1] if Em else f"{part}.0")), 1)
    Pensamento = [{"PIDE": PIm[0], "t": tpl["entrada"]["pensamento"]}]

    # Total de Entrada (ordem correta)
    Total_de_Entrada = []
    Total_de_Entrada += [e["E"] for e in Entrada]
    Total_de_Entrada += [r["RE"] for r in Reacao]
    Total_de_Entrada += [c["CE"] for c in Contexto]
    Total_de_Entrada += [p["PIDE"] for p in Pensamento]

    # Saída - Nome
    ns_marker = f"{part}.9"
    ns_name = tpl["saida"]["nome"] or "0.0"

    # TEXTO INICIAL -> sentenças e tokens
    texto_inicial = []
    raw = (tpl["saida"]["texto_inicial_raw"] or "").strip()
    mother_idx = 10
    if raw:
        parts = re.findall(r'[^.?!]+[.?!]|[^.?!]+$', raw)
        for sent in [p.strip() for p in parts if p.strip()]:
            mae = f"{part}.{mother_idx}"
            words = tokenize(sent)
            tokens = [{"TEX": f"{mae}.{i+1}", "t": w, "vars": ["0.0"]} for i, w in enumerate(words)]
            texto_inicial.append({"TEXI": mae, "tokens": tokens})
            mother_idx += 1

    # FS
    fs_list = []
    if tpl["saida"]["fs_line"]:
        fs_words = tokenize(tpl["saida"]["fs_line"])
        fs_marks = gen_markers(f"{part}.11", len(fs_words))
        fs_list = [{"FS": m, "t": w, "vars": ["0.0"]} for m, w in zip(fs_marks, fs_words)]

    # RS
    RS = []
    if tpl["saida"]["reac_saida"]:
        rs_words = tokenize(tpl["saida"]["reac_saida"])
        start_rs = fs_list[-1]["FS"] if fs_list else f"{part}.11"
        rs_marks = gen_markers(start_rs, len(rs_words))
        RS = [{"RS": m, "t": w, "vars": ["0.0"]} for m, w in zip(rs_marks, rs_words)]

    # TEXTO FINAL (cria estrutura quando há texto; caso contrário, TEXTO_FINAL fica [])
    TEXTO_FINAL = []
    if tpl["saida"]["texto_final_raw"]:
        texf = f"{part}.19"
        words = tokenize(tpl["saida"]["texto_final_raw"])
        tokens = [{"TEX": f"{texf}.{i+1}", "t": w, "vars": ["0.0"]} for i, w in enumerate(words)]
        if tokens:
            TEXTO_FINAL = [{"TEXF": texf, "tokens": tokens}]

    # Total de Saída (ordem: ns, TEXI mothers, FS, RS, TEXF)
    Total_de_Saida = []
    Total_de_Saida.append(ns_marker)
    Total_de_Saida += [ti["TEXI"] for ti in texto_inicial]
    Total_de_Saida += [f["FS"] for f in fs_list]
    Total_de_Saida += [r["RS"] for r in RS]
    Total_de_Saida += [tf["TEXF"] for tf in TEXTO_FINAL]
    Total_de_Saida = [x for x in Total_de_Saida if x]

    # Características de usuário (após Contexto; sem inferência)
    caracteristicas_usuario = {
        "CAE": "0.0",
        "Nome do usuário": {"E": "0.0", "t": "0.0", "Multivars": ["0.0"]},
        "Cabelo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Olhos": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Pele": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "estilo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Forma": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Personalidade": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
    }

    # Características extraídas de TEXTO (apenas anotações literais)
    char_texto = {
        "Cabelo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Olhos": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Pele": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "estilo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Forma": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Personalidade": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
    }
    if texto_inicial:
        t0 = texto_inicial[0]["tokens"]
        seq = ["olhos", "cor", "de", "mel"]
        for i in range(len(t0) - 3):
            words = [t0[i + j]["t"].lower() for j in range(4)]
            if words == seq:
                char_texto["Olhos"] = [{
                    "TEX": [t0[i]["TEX"], t0[i + 1]["TEX"], t0[i + 2]["TEX"], t0[i + 3]["TEX"]],
                    "t": "olhos cor de mel",
                    "Multivars": ["Olhos castanhos"]
                }]
                break
        for tok in t0:
            if tok["t"].lower() == "doçura":
                char_texto["Personalidade"] = [{"TEX": tok["TEX"], "t": "doçura", "Multivars": ["0.0"]}]
                break

    # ALNULU do bloco
    alnulu_src = [
        tpl["entrada"]["texto"], tpl["entrada"]["reacao"], tpl["entrada"]["contexto"],
        tpl["entrada"]["pensamento"], tpl["saida"]["nome"], tpl["saida"]["texto_inicial_raw"],
        tpl["saida"]["fs_line"], tpl["saida"]["reac_saida"], tpl["saida"]["texto_final_raw"]
    ]
    alnulu_total_bloco = calcular_alnulu(" ".join([s for s in alnulu_src if s]))

    # Render textual exatamente no formato exigido
    lines = []
    lines.append('{')
    lines.append('  "IDA": {')
    lines.append('    "IM": {')
    lines.append('      "0": {')
    lines.append('        "bloco_id": 1,')
    # Entrada
    lines.append('        "Entrada": [')
    for i, it in enumerate(Entrada):
        comma = ',' if i < len(Entrada) - 1 else ''
        lines.append(f'          {{"E":"{it["E"]}","t":"{it["t"]}","vars":["{it["vars"][0]}"]}}{comma}')
    lines.append('        ],')
    lines.append('        "Multivars": ["0.0"],')
    # Reação
    lines.append('        "Reação": [')
    for i, it in enumerate(Reacao):
        comma = ',' if i < len(Reacao) - 1 else ''
        lines.append(f'          {{"RE":"{it["RE"]}","t":"{it["t"]}","vars":["{it["vars"][0]}"]}}{comma}')
    lines.append('        ],')
    # Contexto
    lines.append('        "Contexto": [')
    for i, it in enumerate(Contexto):
        comma = ',' if i < len(Contexto) - 1 else ''
        lines.append(f'          {{"CE":"{it["CE"]}","t":"{it["t"]}"}}{comma}')
    lines.append('        ],')
    # Características de usuário (após Contexto)
    lines.append('          "Características de usuário": {')
    lines.append(f'            "CAE": "{caracteristicas_usuario["CAE"]}",')
    nu = caracteristicas_usuario["Nome do usuário"]
    lines.append(f'            "Nome do usuário": ["E": "{nu["E"]}", "t": "{nu["t"]}", "Multivars": ["{nu["Multivars"][0]}"]],')
    for key in ("Cabelo", "Olhos", "Pele", "estilo", "Forma", "Personalidade"):
        e = caracteristicas_usuario[key][0]
        lines.append(f'            "{key}": [')
        lines.append(f'              {{"E":"{e["E"]}","t":"{e["t"]}","Multivars":["{e["Multivars"][0]}"]}}')
        lines.append('            ],')
    lines.append('          },')
    # Pensamento Interno
    lines.append('        "Pensamento Interno": [')
    lines.append(f'          {{"PIDE":"{Pensamento[0]["PIDE"]}","t":"{Pensamento[0]["t"]}"}}')
    lines.append('        ],')
    # Total de Entrada
    lines.append('        "Total de Entrada": [')
    lines.append('          ' + ",".join(f'"{m}"' for m in Total_de_Entrada))
    lines.append('        ],')
    # Saída (após Total de Entrada)
    lines.append('        "Saída": {')
    # Nome
    lines.append('          "Nome": [')
    lines.append(f'            {{"NS":"{ns_marker}","t":"{ns_name}","multivars":["0.0"]}}')
    lines.append('          ],')
    # TEXTO INICIAL
    lines.append('          "TEXTO INICIAL": [')
    for si, ti in enumerate(texto_inicial):
        comma = ',' if si < len(texto_inicial) - 1 else ''
        lines.append('            {')
        lines.append(f'              "TEXI":"{ti["TEXI"]}",')
        lines.append('              "tokens": [')
        for j, tok in enumerate(ti["tokens"]):
            c2 = ',' if j < len(ti["tokens"]) - 1 else ''
            lines.append(f'                {{"TEX":"{tok["TEX"]}","t":"{tok["t"]}","vars":["0.0"]}}{c2}')
        lines.append('              ]')
        lines.append(f'            }}{comma}')
    lines.append('          ],')
    # FS
    lines.append('          "FS": [')
    for i, f in enumerate(fs_list):
        comma = ',' if i < len(fs_list) - 1 else ''
        lines.append(f'            {{"FS":"{f["FS"]}","t":"{f["t"]}","vars":["0.0"]}}{comma}')
    lines.append('          ],')
    # RS
    lines.append('          "RS": [')
    for i, r in enumerate(RS):
        comma = ',' if i < len(RS) - 1 else ''
        lines.append(f'            {{"RS":"{r["RS"]}","t":"{r["t"]}","vars":["0.0"]}}{comma}')
    lines.append('          ],')
    # TEXTO FINAL (placeholder when empty)
    lines.append('          "TEXTO FINAL": [')
    if not TEXTO_FINAL:
        lines.append('            {')
        lines.append('              "TEXF":"0.0",')
        lines.append('              "tokens": [')
        lines.append('                {"TEX":"0.0","t":"0.0","vars":["0.0"]}')
        lines.append('              ]')
        lines.append('            }')
        lines.append('          ],')
    else:
        for i, tf in enumerate(TEXTO_FINAL):
            comma = ',' if i < len(TEXTO_FINAL) - 1 else ''
            lines.append('            {')
            lines.append(f'              "TEXF":"{tf["TEXF"]}",')
            lines.append('              "tokens": [')
            for j, tok in enumerate(tf["tokens"]):
                c2 = ',' if j < len(tf["tokens"]) - 1 else ''
                lines.append(f'                {{"TEX":"{tok["TEX"]}","t":"{tok["t"]}","vars":["0.0"]}}{c2}')
            lines.append('              ]')
            lines.append(f'            }}{comma}')
        lines.append('          ],')
    # Características extraídas de TEXTO:
    lines.append('          "Características extraídas de TEXTO:": {')
    lines.append('            "Cabelo": [')
    lines.append('              {"TEX":"0.0","t":"0.0","Multivars":["0.0"]}')
    lines.append('            ],')
    vo = char_texto["Olhos"][0]
    if isinstance(vo["TEX"], list):
        tex_field = "[" + ",".join(f'"{x}"' for x in vo["TEX"]) + "]"
    else:
        tex_field = f'"{vo["TEX"]}"'
    lines.append('            "Olhos": [')
    lines.append(f'              {{"TEX":{tex_field},"t":"{vo["t"]}","Multivars":["{vo["Multivars"][0]}"]}}')
    lines.append('            ],')
    lines.append('            "Pele": [')
    lines.append('              {"TEX":"0.0","t":"0.0","Multivars":["0.0"]}')
    lines.append('            ],')
    lines.append('            "estilo": [')
    lines.append('              {"TEX":"0.0","t":"0.0","Multivars":["0.0"]}')
    lines.append('            ],')
    lines.append('            "Forma": [')
    lines.append('              {"TEX":"0.0","t":"0.0","Multivars":["0.0"]}')
    lines.append('            ],')
    vp = char_texto["Personalidade"][0]
    lines.append('            "Personalidade": [')
    lines.append(f'              {{"TEX":"{vp["TEX"]}","t":"{vp["t"]}","Multivars":["{vp["Multivars"][0]}"]}}')
    lines.append('            ]')
    lines.append('          },')
    # Sentimento / Tendência / Ressonância
    lines.append('          "Sentimento da Saída": {')
    lines.append('            "SDS":"0.0",')
    lines.append('            "t":"Neutro"')
    lines.append('          },')
    lines.append('          "Tendência da Saída": 0.0,')
    lines.append('          "Ressonância": false')
    lines.append('        },')
    # Total de Saída
    lines.append('        "Total de Saída": [')
    lines.append('          ' + ", ".join(f'"{x}"' for x in Total_de_Saida))
    lines.append('        ]')
    lines.append('      }')
    lines.append('    }')
    lines.append('  }')
    lines.append('}')
    rendered = "\n".join(lines)

    # UM (JSON válido) com alnulu_total_bloco
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
                        "alnulu_total_bloco": alnulu_total_bloco
                    }
                ]
            }
        }
    }

    return rendered, um, Total_de_Entrada, Total_de_Saida, alnulu_total_bloco

# ---------------- main ----------------
def main():
    print("Cole o bloco (termine com linha vazia):")
    lines = []
    try:
        while True:
            ln = input()
            if not ln and lines:
                break
            if not ln:
                continue
            lines.append(ln)
    except EOFError:
        pass

    tpl = parse_block(lines)
    rendered, um, total_ent, total_sai, alnulu_val = build_render(tpl)

    with open(OUT_IDA, "w", encoding="utf-8") as f:
        f.write(rendered)
    with open(OUT_UM, "w", encoding="utf-8") as f:
        json.dump(um, f, ensure_ascii=False, indent=2)

    print(f"Gerados: {OUT_IDA} e {OUT_UM}")
    print(f"alnulu_total_bloco: {alnulu_val}")

if __name__ == "__main__":
    main()



