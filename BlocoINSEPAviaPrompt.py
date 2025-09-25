#!/usr/bin/env python3
# coding: utf-8
"""
Script minimalista:
- Monta o bloco conforme seu modelo (Nome_cae vindo da Entrada; Nome_ns da Saída)
- Serializa em formato compacto (itens pequenos em uma linha) — estilo do seu exemplo
- Garante exatamente 159 linhas no arquivo por bloco_id (preenchendo FS com entradas simples)
- Produz inconsciente.json e adam_memoria.json
"""
import json, re, sys
from typing import List, Dict, Any

OUT_IDA = "inconsciente.json"
OUT_UM = "adam_memoria.json"

def tokenize(s: str) -> List[str]:
    if not s: return []
    return re.findall(r"\w+|[^\w\s]", s.replace('"',''), re.UNICODE)

def gen_markers(base: str, n: int) -> List[str]:
    part, _, suf = base.partition(".")
    try:
        cur = int(suf or "0")
    except:
        cur = 0
    return [f"{part}.{cur+i+1}" for i in range(n)]

def parse_block(lines: List[str]) -> Dict[str,Any]:
    txt = "\n".join(lines)
    def g(pat):
        m = re.search(pat, txt, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""
    entrada = g(r'Entrada:\s*"([^"]*)"')
    reacao = g(r'Reação:\s*([^\n\r]+)')
    contexto = g(r'Contexto:\s*([^\n\r]+)')
    pensamento = g(r'Pensamento Interno:\s*"([^"]*)"')
    nome_entrada = g(r'Nome:\s*"([^"]*)"')
    texto_inicial_raw = g(r'Texto Inicial:\s*"([^"]*)"') or g(r'Texto Inicial:\s*([^\n\r]*)')
    fs_line = g(r'—\s*([^\n\r]+)')
    reac_personagem = g(r'Reação de personagem:\s*([^\n\r]+)')
    texto_final = g(r'Texto final:\s*"([^"]*)"')
    saida_nome_explicit = g(r'Saída\s+Nome:\s*"([^"]*)"') or g(r'Nome\s+Saída:\s*"([^"]*)"')
    if not entrada:
        raise SystemExit("Erro: campo 'Entrada' ausente. Use: Entrada: \"...\"")
    return {
        "entrada": {"texto": entrada, "reacao": reacao, "contexto": contexto,
                    "pensamento": pensamento, "nome_entrada": nome_entrada},
        "saida": {"texto_inicial_raw": texto_inicial_raw, "fs_line": fs_line,
                  "reac_personagem": reac_personagem, "texto_final": texto_final,
                  "nome_saida_explicit": saida_nome_explicit}
    }

# ----- constrói bloco de acordo com seu modelo -----
def build_block(tpl: Dict[str,Any], agent_name_default: str=None) -> Dict[str,Any]:
    part = "0"
    entrada_words = tokenize(tpl["entrada"]["texto"])
    Em = gen_markers(f"{part}.0", len(entrada_words))
    Entrada = [{"E": m, "t": w, "vars": ["0.0"]} for m, w in zip(Em, entrada_words)]

    reacao_words = tokenize(tpl["entrada"]["reacao"])
    REm = gen_markers(Em[-1] if Em else f"{part}.0", len(reacao_words))
    Reacao = [{"RE": m, "t": w, "vars": ["0.0"]} for m, w in zip(REm, reacao_words)]

    ctx_words = tpl["entrada"]["contexto"].split() if tpl["entrada"]["contexto"] else []
    CEm = gen_markers(REm[-1] if REm else (Em[-1] if Em else f"{part}.0"), len(ctx_words))
    Contexto = [{"CE": m, "t": w} for m, w in zip(CEm, ctx_words)]

    PIm = gen_markers(CEm[-1] if CEm else (REm[-1] if REm else (Em[-1] if Em else f"{part}.0")), 1)
    Pensamento = [{"PIDE": PIm[0], "t": tpl["entrada"]["pensamento"]}]

    Total_de_Entrada = Em + REm + CEm + PIm

    # TEXTO INICIAL -> mães .10, .11...
    texto_inicial = []
    raw = (tpl["saida"]["texto_inicial_raw"] or "").strip()
    if raw:
        parts = re.findall(r'[^.?!]+[.?!]|[^.?!]+$', raw)
        mother_idx = 10
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
    if tpl["saida"]["reac_personagem"]:
        rs_words = tokenize(tpl["saida"]["reac_personagem"])
        start_rs = fs_list[-1]["FS"] if fs_list else f"{part}.11"
        rs_marks = gen_markers(start_rs, len(rs_words))
        RS = [{"RS": m, "t": w, "vars": ["0.0"]} for m, w in zip(rs_marks, rs_words)]

    # TEXTO FINAL
    TEXTO_FINAL = []
    if tpl["saida"]["texto_final"]:
        texf = f"{part}.19"
        words = tokenize(tpl["saida"]["texto_final"])
        tokens = [{"TEX": f"{texf}.{i+1}", "t": w, "vars": ["0.0"]} for i, w in enumerate(words)]
        TEXTO_FINAL = [{"TEXF": texf, "tokens": tokens}]

    Total_de_Saida = [f"{part}.9"] + [ti["TEXI"] for ti in texto_inicial]
    if fs_list: Total_de_Saida += [f["FS"] for f in fs_list]
    if RS: Total_de_Saida += [r["RS"] for r in RS]
    if TEXTO_FINAL: Total_de_Saida += [tf["TEXF"] for tf in TEXTO_FINAL]

    if tpl["entrada"]["nome_entrada"]:
        nome_field = {"E": gen_markers(PIm[-1], 1)[0], "t": tpl["entrada"]["nome_entrada"], "Multivars": ["0.0"]}
    else:
        nome_field = {"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}

    CAE = {
        "CAE": "0.0",
        "Nome_cae": nome_field,
        "Cabelo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Olhos": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Pele": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "estilo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Forma": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Personalidade": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
    }

    if tpl["saida"].get("nome_saida_explicit"):
        ns_name = tpl["saida"]["nome_saida_explicit"]
    elif agent_name_default:
        ns_name = agent_name_default
    else:
        ns_name = "0.0"

    ns_marker = f"{part}.9"
    Saida_obj = {
        "Nome_ns": [{"NS": ns_marker, "t": ns_name, "vars": ["0.0"]}],
        "TEXTO INICIAL": texto_inicial,
        "FS": fs_list,
        "RS": RS,
        "TEXTO FINAL": TEXTO_FINAL,
        "Características extraídas de TEXTO:": {
            "Cabelo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
            "Olhos": [{"TEX": ["0.0"], "t": "0.0", "Multivars": ["0.0"]}],
            "Pele": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
            "estilo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
            "Forma": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
            "Personalidade": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
        },
        "Sentimento da Saída": {"SDS": "0.0", "t": "Neutro"},
        "Tendência da Saída": 0.0,
        "Ressonância": False
    }

    bloco = {
        "bloco_id": 1,
        "Entrada": Entrada,
        "Multivars": ["0.0"],
        "Reação": Reacao,
        "Contexto": Contexto,
        "Características extraídas de entrada:": CAE,
        "Pensamento Interno": Pensamento,
        "Total de Entrada": Total_de_Entrada,
        "Saída": Saida_obj,
        "Total de Saída": Total_de_Saida
    }

    return {"IDA": {"IM": {"0": bloco}}}

# ----- SERIALIZAÇÃO COMPACTA E CONTAGEM -----
def render_compact_like_example(root: Dict[str,Any]) -> str:
    """
    Serializa o bloco de forma compacta (itens pequenos em uma linha).
    Assumimos um formato previsível; não usa regexes mutáveis.
    """
    # Serializar manualmente para obter controle exato de quebras de linha.
    ida = root["IDA"]["IM"]["0"]
    lines: List[str] = []
    lines.append('{')
    lines.append('  "IDA": {')
    lines.append('    "IM": {')
    lines.append('      "0": {')
    # bloco_id
    lines.append(f'        "bloco_id": {ida["bloco_id"]},')
    # Entrada
    lines.append('        "Entrada": [')
    for i, it in enumerate(ida["Entrada"]):
        comma = ',' if i < len(ida["Entrada"]) - 1 else ''
        lines.append(f'          {{"E":"{it["E"]}","t":"{it["t"]}","vars":["{it["vars"][0]}"]}}{comma}')
    lines.append('        ],')
    # Multivars
    lines.append(f'        "Multivars": {json.dumps(ida.get("Multivars", ["0.0"]))},')
    # Reação
    lines.append('        "Reação": [')
    for i, it in enumerate(ida.get("Reação", [])):
        comma = ',' if i < len(ida.get("Reação", [])) - 1 else ''
        lines.append(f'          {{"RE":"{it["RE"]}","t":"{it["t"]}","vars":["{it["vars"][0]}"]}}{comma}')
    lines.append('        ],')
    # Contexto
    lines.append('        "Contexto": [')
    for i, it in enumerate(ida.get("Contexto", [])):
        comma = ',' if i < len(ida.get("Contexto", [])) - 1 else ''
        lines.append(f'          {{"CE":"{it["CE"]}","t":"{it["t"]}"}}{comma}')
    lines.append('        ],')
    # Características extraídas de entrada:
    lines.append('        "Características extraídas de entrada:": {')
    cae = ida["Características extraídas de entrada:"]
    # CAE fixed
    lines.append(f'          "CAE": "{cae.get("CAE","0.0")}",')
    # Nome_cae -> keep key "Nome CAE" if you prefer that exact string
    nome_cae = cae.get("Nome_cae", {"E":"0.0","t":"0.0","Multivars":["0.0"]})
    # ensure if nome_cae is dict or list
    if isinstance(nome_cae, dict):
        lines.append(f'          "Nome CAE": {{"E":"{nome_cae.get("E")}","t":"{nome_cae.get("t")}","Multivars":["{nome_cae.get("Multivars",[ "0.0"])[0]}"]}},')
    else:
        # fallback
        lines.append(f'          "Nome CAE": {{"E":"0.0","t":"0.0","Multivars":["0.0"]}},')
    # other fields keep compact single-line arrays (Cabelo, Olhos, etc.)
    for key in ("Cabelo","Olhos","Pele","estilo","Forma","Personalidade"):
        val = cae.get(key, [{"E":"0.0","t":"0.0","Multivars":["0.0"]}])
        # take first element for compactness
        e = val[0] if isinstance(val, list) and val else {"E":"0.0","t":"0.0","Multivars":["0.0"]}
        lines.append(f'          "{key}": [{{"E":"{e.get("E")}","t":"{e.get("t")}","Multivars":["{e.get("Multivars",[ "0.0"])[0]}"]}}],')
    # remove trailing comma style: replace last comma with nothing later
    # Pensamento Interno
    pen = ida.get("Pensamento Interno", [])
    lines.append('        "Pensamento Interno": [')
    for i, it in enumerate(pen):
        comma = ',' if i < len(pen) - 1 else ''
        lines.append(f'          {{"PIDE":"{it["PIDE"]}","t":"{it["t"]}"}}{comma}')
    lines.append('        ],')
    # Total de Entrada
    tde = ida.get("Total de Entrada", [])
    lines.append('        "Total de Entrada": [' + ",".join(f'"{x}"' for x in tde) + '],')
    # Saída block
    saida = ida.get("Saída", {})
    lines.append('        "Saída": {')
    # Nome_ns (compact)
    nome_ns_list = saida.get("Nome_ns", [])
    if nome_ns_list:
        n = nome_ns_list[0]
        lines.append(f'          "Nome": [{{"NS":"{n.get("NS")}","t":"{n.get("t")}","multivars":["{n.get("vars",[ "0.0"])[0]}"]}}],')
    else:
        lines.append(f'          "Nome": [{{"NS":"0.0","t":"0.0","multivars":["0.0"]}}],')
    # TEXTO INICIAL
    lines.append('          "TEXTO INICIAL": [')
    for i, ti in enumerate(saida.get("TEXTO INICIAL", [])):
        comma = ',' if i < len(saida.get("TEXTO INICIAL", [])) - 1 else ''
        tokens = ti.get("tokens", [])
        token_str = ",".join(f'{{"TEX":"{tok["TEX"]}","t":"{tok["t"]}","vars":["{tok.get("vars",["0.0"])[0]}"]}}' for tok in tokens)
        lines.append(f'            {{"TEXI":"{ti.get("TEXI")}","tokens":[{token_str}]}}{comma}')
    lines.append('          ],')
    # FS
    lines.append('          "FS": [')
    for i, f in enumerate(saida.get("FS", [])):
        comma = ',' if i < len(saida.get("FS", [])) - 1 else ''
        lines.append(f'            {{"FS":"{f.get("FS")}","t":"{f.get("t")}","vars":["{f.get("vars",["0.0"])[0]}"]}}{comma}')
    lines.append('          ],')
    # RS
    lines.append('          "RS": [')
    for i, r in enumerate(saida.get("RS", [])):
        comma = ',' if i < len(saida.get("RS", [])) - 1 else ''
        lines.append(f'            {{"RS":"{r.get("RS")}","t":"{r.get("t")}","vars":["{r.get("vars",["0.0"])[0]}"]}}{comma}')
    lines.append('          ],')
    # TEXTO FINAL
    lines.append('          "TEXTO FINAL": [')
    for i, tf in enumerate(saida.get("TEXTO FINAL", [])):
        comma = ',' if i < len(saida.get("TEXTO FINAL", [])) - 1 else ''
        tokens = tf.get("tokens", [])
        token_str = ",".join(f'{{"TEX":"{tok["TEX"]}","t":"{tok["t"]}","vars":["{tok.get("vars",["0.0"])[0]}"]}}' for tok in tokens)
        lines.append(f'            {{"TEXF":"{tf.get("TEXF")}","tokens":[{token_str}]}}{comma}')
    lines.append('          ],')
    # Características extraídas de TEXTO:
    lines.append('          "Características extraídas de TEXTO:": {')
    cat = saida.get("Características extraídas de TEXTO:", {})
    for key in ("Cabelo","Olhos","Pele","estilo","Forma","Personalidade"):
        v = cat.get(key, [{"TEX":"0.0","t":"0.0","Multivars":["0.0"]}])
        first = v[0] if isinstance(v, list) and v else {"TEX":"0.0","t":"0.0","Multivars":["0.0"]}
        tex = first.get("TEX")
        tval = first.get("t","0.0")
        mult = first.get("Multivars",["0.0"])[0] if isinstance(first.get("Multivars",[]), list) else first.get("Multivars","0.0")
        # TEX may be list or string
        if isinstance(tex, list):
            tex_field = "[" + ",".join(f'"{x}"' for x in tex) + "]"
        else:
            tex_field = f'"{tex}"'
        lines.append(f'            "{key}": [{{"TEX":{tex_field},"t":"{tval}","Multivars":["{mult}"]}}],')
    # Sentimento da Saída etc.
    lines.append('          "Sentimento da Saída": {"SDS":"0.0","t":"Neutro"},')
    lines.append('          "Tendência da Saída": 0.0,')
    lines.append('          "Ressonância": false')
    lines.append('        },')
    # Total de Saída
    tds = ida.get("Total de Saída", [])
    lines.append('        "Total de Saída": [' + ",".join(f'"{x}"' for x in tds) + ']')
    # close braces
    lines.append('      }')
    lines.append('    }')
    lines.append('  }')
    lines.append('}')
    return "\n".join(lines)

# ----- função que garante exatamente 159 linhas -----
def ensure_159_lines(root: Dict[str,Any]) -> Dict[str,Any]:
    """
    Preenche Saída.FS com entradas simples até que a serialização compacta
    produza exatamente 159 linhas (ou remove FS se houver excesso).
    """
    # operate on the single bloco at IM.0
    bloco = root["IDA"]["IM"]["0"]
    # ensure lists exist
    bloco.setdefault("Saída", {})
    s = bloco["Saída"]
    s.setdefault("FS", [])
    bloco.setdefault("Total de Saída", [])

    def lines_count():
        return render_compact_like_example(root).count("\n") + 1

    # loop safe-guard
    limit = 1000
    it = 0
    while it < limit:
        it += 1
        n = lines_count()
        if n == 159:
            break
        if n < 159:
            # add one FS filler (adds exactly 1 line in our compact format)
            # choose marker based on existing Total de Saída numeric suffix
            last_idx = 19
            for m in bloco.get("Total de Saída", []):
                if isinstance(m, str) and "." in m:
                    try:
                        suf = int(m.split(".")[-1])
                        if suf > last_idx: last_idx = suf
                    except:
                        pass
            new_marker = f"0.{last_idx+1}"
            s["FS"].append({"FS": new_marker, "t": "0.0", "vars": ["0.0"]})
            bloco["Total de Saída"].append(new_marker)
            continue
        else:
            # n > 159, remove fillers from end (FS) first
            if s["FS"]:
                s["FS"].pop()
                if bloco.get("Total de Saída"):
                    # remove last string entry if exists
                    for i in range(len(bloco["Total de Saída"]) - 1, -1, -1):
                        if isinstance(bloco["Total de Saída"][i], str):
                            bloco["Total de Saída"].pop(i)
                            break
                continue
            # if no FS to remove, try removing last token of last TEXTO FINAL
            if s.get("TEXTO FINAL"):
                last_tf = s["TEXTO FINAL"][-1]
                if isinstance(last_tf, dict) and last_tf.get("tokens"):
                    last_tf["tokens"].pop()
                    if not last_tf["tokens"]:
                        s["TEXTO FINAL"].pop()
                    continue
            # as last resort, stop to avoid destructive edits
            break
    return root

# ----- build UM (summary) simply produces a small structure -----
def build_um(root: Dict[str,Any]) -> Dict[str,Any]:
    bloco = root["IDA"]["IM"]["0"]
    Entrada = bloco.get("Entrada", [])
    Saida = bloco.get("Saída", {})
    ae = {it["E"]: 0 for it in Entrada}
    # minimal UM structure to match previous behavior
    bloco_um = {
        "bloco_id": bloco.get("bloco_id"),
        "Total de Entrada": bloco.get("Total de Entrada"),
        "ultimo_child_entrada": bloco.get("Total de Entrada")[-1] if bloco.get("Total de Entrada") else None,
        "Multivars": bloco.get("Multivars", []),
        "Total de Saída": bloco.get("Total de Saída", []),
        "ultimo_child_saida": bloco.get("Total de Saída")[-1] if bloco.get("Total de Saída") else None,
        "alnulu_total_bloco": 0
    }
    return {"UM": {"0": {"Universo Mãe": "Interações", "blocos": [bloco_um]}}}

# ----- main -----
def main():
    agent_name = None
    if len(sys.argv) >= 3 and sys.argv[1] in ("--agent-name","-n"):
        agent_name = sys.argv[2]

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
    root = build_block(tpl, agent_name_default=agent_name)
    # ensure Nome fields kept consistent (no copying from Saída to CAE)
    # fill final block id if missing
    root["IDA"]["IM"]["0"]["bloco_id"] = root["IDA"]["IM"]["0"].get("bloco_id", 1)
    # ensure visual 159 lines
    root = ensure_159_lines(root)
    # write files (using compact single-line-items renderer)
    compact_text = render_compact_like_example(root)
    with open(OUT_IDA, "w", encoding="utf-8") as f:
        f.write(compact_text)
    um = build_um(root)
    with open(OUT_UM, "w", encoding="utf-8") as f:
        f.write(json.dumps(um, ensure_ascii=False, indent=2))
    print("Gerados:", OUT_IDA, "e", OUT_UM)
    print("Linhas no arquivo:", compact_text.count("\n") + 1)

if __name__ == "__main__":
    main()


