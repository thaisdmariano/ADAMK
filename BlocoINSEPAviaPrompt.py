#!/usr/bin/env python3
# coding: utf-8
"""
Gera inconsciente.json (formato legível) e adam_memoria.json.

Regras principais:
- Não infere nada: todos os campos de características (CAE e "Características extraídas de TEXTO")
  vêm vazios (0.0) e serão preenchidos posteriormente pelo bloco colado.
- Nenhum nome é inferido ou copiado para CAE. Se houver "Saída: Nome: \"X\"" no bloco,
  esse nome aparece APENAS em Saída.Nome (como NS) e NÃO é tratado como nome de usuário.

Uso: rode e cole o bloco (termina com linha vazia).
"""
import json
import re
from typing import List, Dict, Any

OUT_IDA = "inconsciente.json"
OUT_UM = "adam_memoria.json"


def tokenize(s: str) -> List[str]:
    if not s:
        return []
    return re.findall(r"\w+|[^\w\s]", s.replace('"', ''), re.UNICODE)


def gen_markers(base: str, n: int) -> List[str]:
    part, _, suf = base.partition(".")
    cur = int(suf or "0")
    out = []
    for _ in range(n):
        cur += 1
        out.append(f"{part}.{cur}")
    return out


def parse_block(lines: List[str]) -> Dict[str, Any]:
    joined = "\n".join(lines)
    tpl = {
        "entrada": {"texto": "", "reacao": "", "contexto": "", "pensamento": ""},
        "saida": {
            "nome_raw": "",               # Nome da SAÍDA (NS), não do usuário
            "texto_initial_raw": "",
            "fs_line": "",
            "reacao_personagem": "",
            "texto_final_raw": ""
        }
    }

    # Entrada
    m = re.search(r'Entrada:\s*"([^"]*)"', joined, re.IGNORECASE)
    if m: tpl["entrada"]["texto"] = m.group(1).strip()

    m = re.search(r'Reação:\s*([^\n\r]+)', joined, re.IGNORECASE)
    if m: tpl["entrada"]["reacao"] = m.group(1).strip()

    m = re.search(r'Contexto:\s*([^\n\r]+)', joined, re.IGNORECASE)
    if m: tpl["entrada"]["contexto"] = m.group(1).strip()

    m = re.search(r'Pensamento Interno:\s*"([^"]*)"', joined, re.IGNORECASE)
    if m: tpl["entrada"]["pensamento"] = m.group(1).strip()

    # Saída: Nome (NS)
    m = re.search(r'Saída:\s*Nome:\s*"([^"]*)"', joined, re.IGNORECASE)
    if m: tpl["saida"]["nome_raw"] = m.group(1).strip()

    # Saída: Texto Inicial
    m = re.search(r'Texto Inicial:\s*"([^"]*)"', joined, re.IGNORECASE)
    if m:
        tpl["saida"]["texto_initial_raw"] = m.group(1).strip()
        # pega possível oração seguinte até o primeiro ponto
        rest = joined.split(m.group(0), 1)[1] if m.group(0) in joined else ""
        extra = re.search(r'\s*([^.]*\.)', rest)
        if extra:
            ex = extra.group(1).strip()
            if ex:
                tpl["saida"]["texto_initial_raw"] += " " + ex
    else:
        m2 = re.search(r'Texto Inicial:\s*([^\n\r]*)', joined, re.IGNORECASE)
        if m2: tpl["saida"]["texto_initial_raw"] = m2.group(1).strip()

    # Saída: fala (linha começando com travessão —)
    m = re.search(r'—\s*([^\n\r]+)', joined)
    if m: tpl["saida"]["fs_line"] = m.group(1).strip()

    # Saída: reação de personagem
    m = re.search(r'Reação de personagem:\s*([^\n\r]+)', joined, re.IGNORECASE)
    if m: tpl["saida"]["reacao_personagem"] = m.group(1).strip()

    # Saída: texto final
    m = re.search(r'Texto final:\s*"([^"]*)"', joined, re.IGNORECASE)
    if m: tpl["saida"]["texto_final_raw"] = m.group(1).strip()

    if not tpl["entrada"]["texto"]:
        raise SystemExit("Entrada vazia. Forneça 'Entrada: \"...\"'.")

    return tpl


def build_structure(tpl: Dict[str, Any]) -> Dict[str, Any]:
    part = "0"

    # Entrada
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

    # Saída: Texto Inicial
    texto_inicial = []
    raw = tpl["saida"]["texto_initial_raw"].strip()
    if raw:
        parts = re.findall(r'[^.?!]+[.?!]|[^.?!]+$', raw)
        parts = [p.strip() for p in parts if p.strip()]
        mother_idx = 10
        for sent in parts:
            mae = f"{part}.{mother_idx}"
            words = tokenize(sent)
            tokens = [{"TEX": f"{mae}.{i + 1}", "t": w, "vars": ["0.0"]} for i, w in enumerate(words)]
            texto_inicial.append({"TEXI": mae, "tokens": tokens})
            mother_idx += 1

    # Saída: FS/RS/TEXTO_FINAL
    fs_list = []
    if tpl["saida"]["fs_line"]:
        fs_words = tokenize(tpl["saida"]["fs_line"])
        fs_marks = gen_markers(f"{part}.11", len(fs_words))
        fs_list = [{"FS": m, "t": w, "vars": ["0.0"]} for m, w in zip(fs_marks, fs_words)]

    RS = []
    if tpl["saida"]["reacao_personagem"]:
        rs_words = tokenize(tpl["saida"]["reacao_personagem"])
        start_rs_base = fs_list[-1]["FS"] if fs_list else f"{part}.11"
        rs_marks = gen_markers(start_rs_base, len(rs_words))
        RS = [{"RS": m, "t": w, "vars": ["0.0"]} for m, w in zip(rs_marks, rs_words)]

    TEXTO_FINAL = []
    if tpl["saida"]["texto_final_raw"]:
        texf = f"{part}.19"
        words = tokenize(tpl["saida"]["texto_final_raw"])
        tokens = [{"TEX": f"{texf}.{i + 1}", "t": w, "vars": ["0.0"]} for i, w in enumerate(words)]
        TEXTO_FINAL = [{"TEXF": texf, "tokens": tokens}]

    # Saída: Nome (NS apenas, se fornecido)
    ns_marker = f"{part}.9"
    saida_nome = []
    if tpl["saida"]["nome_raw"]:
        saida_nome.append({"NS": ns_marker, "t": tpl["saida"]["nome_raw"], "vars": ["0.0"]})

    # CAE: tudo vazio (sem inferência; nenhum nome de usuário)
    CAE = {
        "CAE": "0.0",
        "Nome": {"E": "0.0", "t": "0.0", "Multivars": ["0.0"]},
        "Cabelo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Olhos": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Pele": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "estilo": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Forma": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Personalidade": [{"E": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
    }

    # Características extraídas de TEXTO: tudo vazio (sem inferência)
    caracteristicas_texto = {
        "Cabelo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Olhos": [{"TEX": ["0.0"], "t": "0.0", "Multivars": ["0.0"]}],
        "Pele": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "estilo": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Forma": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}],
        "Personalidade": [{"TEX": "0.0", "t": "0.0", "Multivars": ["0.0"]}]
    }

    Saida = {
        "Nome": saida_nome,  # NS só
        "TEXTO INICIAL": texto_inicial,
        "FS": fs_list,
        "RS": RS,
        "TEXTO FINAL": TEXTO_FINAL,
        "Características extraídas de TEXTO:": caracteristicas_texto,
        "Sentimento da Saída": {"SDS": "0.0", "t": "Neutro"},
        "Tendência da Saída": 0.0,
        "Ressonância": False
    }

    # Total de Saída
    total_saida = ([ns_marker] if saida_nome else []) + [ti["TEXI"] for ti in texto_inicial]
    if fs_list: total_saida += [f["FS"] for f in fs_list]
    if RS: total_saida += [r["RS"] for r in RS]
    if TEXTO_FINAL: total_saida += [tf["TEXF"] for tf in TEXTO_FINAL]

    bloco = {
        "bloco_id": 1,
        "Entrada": Entrada,
        "Multivars": ["0.0"],
        "Reação": Reacao,
        "Contexto": Contexto,
        "Características extraídas de entrada:": CAE,
        "Pensamento Interno": Pensamento,
        "Total de Entrada": Total_de_Entrada,
        "Saída": Saida,
        "Total de Saída": total_saida
    }

    root = {"IDA": {"IM": {"0": bloco}}}
    return root


def collapse_token_arrays(pretty_json: str) -> str:
    # Colapsa arrays com "TEX" quando há muitos itens
    pattern = re.compile(r'''


\[\s*
        (\{[^{}]*"TEX"[^{}]*\}
        (?:\s*,\s*\{[^{}]*"TEX"[^{}]*\}\s*){5,})
        \s*\]


        ''', re.VERBOSE | re.DOTALL)

    def repl(m):
        arr = m.group(0)
        single = re.sub(r'\s+', ' ', arr)
        single = single.replace(' ,', ',').replace('{ ', '{').replace(' }', '}')
        return single

    pretty_json = pattern.sub(repl, pretty_json)

    # Colapsa arrays "FS" com >= 6 itens
    pattern2 = re.compile(r'''


\[\s*
        (\{[^{}]*"FS"[^{}]*\}
        (?:\s*,\s*\{[^{}]*"FS"[^{}]*\}\s*){5,})
        \s*\]


        ''', re.VERBOSE | re.DOTALL)

    pretty_json = pattern2.sub(repl, pretty_json)
    return pretty_json


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
    root = build_structure(tpl)

    pretty = json.dumps(root, ensure_ascii=False, indent=2)
    compacted = collapse_token_arrays(pretty)

    with open(OUT_IDA, "w", encoding="utf-8") as f:
        f.write(compacted)

    # UM simples, sem nome de usuário
    um = {"UM": {"0": {"Universo Mãe": "Interações", "blocos": []}}}
    with open(OUT_UM, "w", encoding="utf-8") as f:
        json.dump(um, f, ensure_ascii=False, indent=2)

    print(f"Gerado {OUT_IDA} (compactado-legível) e {OUT_UM}.")


if __name__ == "__main__":
    main()



