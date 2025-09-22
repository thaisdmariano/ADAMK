#!/usr/bin/env python3
# coding: utf-8

import sys
import json
import re
from typing import List, Dict, Any, Optional, Tuple

JSON_IDA = "inconsciente.json"
JSON_UM  = "adam_memoria.json"

def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        sys.exit(f"Erro ao ler {path}: {e}")

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def Token(text: str) -> List[str]:
    clean = text.replace('"', "")
    return re.findall(r"\w+|[^\w\s]", clean, re.UNICODE)

def next_marker(prev: str) -> str:
    mom, _, suf = prev.partition(".")
    return f"{mom}.{int(suf or '0') + 1}"

def generate_markers(start: str, count: int) -> List[str]:
    seq, cur = [], start
    for _ in range(count):
        cur = next_marker(cur)
        seq.append(cur)
    return seq

def calcular_alnulu(texto: str) -> int:
    mapa = {
        "A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,
        "H":8,"I":9,"J":-10,"K":11,"L":12,"M":-13,"N":14,
        "O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,
        "V":-22,"W":23,"X":24,"Y":-25,"Z":26,
        "0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
        ".":2,"!":3,"?":4,",":1,";":1,":":1,"-":1
    }
    equiv = {
        "Á":"A","À":"A","Â":"A","Ã":"A","Ä":"A",
        "É":"E","Ê":"E","È":"E","Ë":"E",
        "Í":"I","Ì":"I","Î":"I","Ï":"I",
        "Ó":"O","Ò":"O","Ô":"O","Õ":"O","Ö":"O",
        "Ú":"U","Ù":"U","Û":"U","Ü":"U",
        "Ç":"C","Ñ":"N",
        "4":"A","3":"E","1":"I","0":"O","5":"S","7":"T","2":"Z"
    }
    total = 0
    for ch in texto:
        norm = equiv.get(ch.upper(), ch.upper())
        total += mapa.get(norm, 0)
    return total

def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_block(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "bloco_id": None,
        "entrada": {"texto": "", "reacao": "", "contexto": "", "pensamento_interno": []},
        "saida":   {"TEX_FR": [], "FS": [], "reacao": "", "contexto": ""}
    }
    sec: Optional[str] = None

    def add_texto_integral(content: str) -> None:
        text = content.strip()
        if text:
            tpl["saida"]["TEX_FR"].append(text)

    def add_texto_por_aspas(content: str) -> None:
        # captura blocos "..."; resíduos fora das aspas entram como TEX único
        quoted = re.findall(r'"([^"]+)"', content)
        residual = re.sub(r'"[^"]+"', " ", content).strip()
        for q in quoted:
            q = q.strip()
            if q:
                tpl["saida"]["TEX_FR"].append(q)
        residual = re.sub(r"\s+", " ", residual).strip()
        if residual:
            tpl["saida"]["TEX_FR"].append(residual)

    def add_texto_por_ponto(content: str) -> None:
        parts = re.findall(r'[^.?!]+[.?!]', content)
        parts = [p.strip() for p in parts if p.strip()]
        tpl["saida"]["TEX_FR"].extend(parts)

    for raw in lines:
        line = raw.strip()
        low  = line.lower()

        if low.startswith("entrada:"):
            tpl["entrada"]["texto"] = line.partition(":")[2].strip()
            sec = "ent"; continue

        if sec == "ent" and low.startswith("reação:"):
            tpl["entrada"]["reacao"] = line.partition(":")[2].strip(); continue

        if sec == "ent" and low.startswith("contexto:"):
            tpl["entrada"]["contexto"] = line.partition(":")[2].strip(); continue

        if sec == "ent" and low.startswith("pensamento interno:"):
            tpl["entrada"]["pensamento_interno"].append(
                line.partition(":")[2].strip().strip('"')
            ); continue

        # Saída: escolha do modo de Texto
        if low.startswith("texto.:"):
            content = line.split(":", 1)[1].strip()
            add_texto_por_ponto(content)
            sec = "out_txt"; continue

        if low.startswith("texto:"):
            content = line.split(":", 1)[1].strip()
            if '"' in content:
                add_texto_por_aspas(content)
            else:
                add_texto_integral(content)
            sec = "out_txt"; continue

        if low == "saída:":
            sec = "out_txt"; continue

        if sec == "out_txt" and line.startswith("—"):
            fala = line.lstrip("—").strip()
            if fala:
                tpl["saida"]["FS"].append(fala)
            continue

        if sec in ("out_txt","out_meta") and low.startswith("reação:"):
            tpl["saida"]["reacao"] = line.partition(":")[2].strip()
            sec = "out_meta"; continue

        if sec == "out_meta" and low.startswith("contexto:"):
            tpl["saida"]["contexto"] = line.partition(":")[2].strip(); continue

    if not tpl["entrada"]["texto"]:
        sys.exit("Erro: cada bloco precisa de 'Entrada'.")
    return tpl

def build_ida_block(
    tpl: Dict[str, Any],
    last_marker: str
) -> Tuple[Dict[str, Any], str, str]:
    # Entrada
    E   = Token(tpl["entrada"]["texto"])
    RE  = Token(tpl["entrada"]["reacao"])
    CE  = Token(tpl["entrada"]["contexto"])
    SI  = tpl["entrada"]["pensamento_interno"][:3]

    # Saída
    tex_frases: List[str] = unique_preserve_order(tpl["saida"].get("TEX_FR", []))
    fs_list: List[str]    = tpl["saida"].get("FS", [])

    RS = Token(tpl["saida"]["reacao"])
    CS = Token(tpl["saida"]["contexto"])

    # Contagens para marcadores
    TEX_count = len(tex_frases)
    S_tokens_per_fala = [Token(fs) for fs in fs_list]
    S_count = sum(len(toks) for toks in S_tokens_per_fala)

    total = len(E) + len(RE) + len(CE) + len(SI) + TEX_count + S_count + len(RS) + len(CS)

    # Marcadores
    marks = generate_markers(last_marker, total)
    ptr = 0

    Em  = marks[ptr:ptr+len(E)];    ptr += len(E)
    REm = marks[ptr:ptr+len(RE)];   ptr += len(RE)
    CEm = marks[ptr:ptr+len(CE)];   ptr += len(CE)
    PIm = marks[ptr:ptr+len(SI)];   ptr += len(SI)

    TM  = marks[ptr:ptr+TEX_count]; ptr += TEX_count
    SM  = marks[ptr:ptr+S_count];   ptr += S_count

    RSm = marks[ptr:ptr+len(RS)];   ptr += len(RS)
    CSm = marks[ptr:ptr+len(CS)]

    # Monta TEX
    saida_texto_items: List[Dict[str, Any]] = [
        {"TEX": tm, "t": frase, "vars": tpl.get("vars_TEX",{}).get(tm,["0.0"])}
        for tm, frase in zip(TM, tex_frases)
    ]

    # Monta S (todas as falas, tokenizadas)
    sm_iter = iter(SM)
    for tokens in S_tokens_per_fala:
        for tok in tokens:
            m = next(sm_iter)
            saida_texto_items.append(
                {"S": m, "t": tok, "vars": tpl.get("vars_TEX",{}).get(m,["0.0"])}
            )

    bloco = {
        "bloco_id": tpl.get("bloco_id"),
        "Entrada": [
            {"E": m, "t": tok, "vars": tpl.get("vars_entrada",{}).get(m,["0.0"])}
            for m,tok in zip(Em, E)
        ],
        "Multivars": tpl.get("Multivars", []),
        "Reação": [
            {"RE": m, "t": tok, "vars": tpl.get("vars_reacao",{}).get(m,["0.0"])}
            for m,tok in zip(REm, RE)
        ],
        "Contexto": [
            {"CE": m, "t": tok}
            for m,tok in zip(CEm, CE)
        ],
        "Sentimento da Entrada": tpl.get("Sentimento da Entrada","Neutro"),
        "Tendência da Entrada": tpl.get("Tendência da Entrada","0.0"),
        "Pensamento Interno": [
            {"PIDE": m, "t": seg}
            for m,seg in zip(PIm, SI)
        ],
        "Total de Entrada": Em + REm + CEm + PIm,
        "Saída": {
            "Texto": saida_texto_items,
            "Multivars": tpl.get("Multivars_saida", []),
            "Reação de Saída": [
                {"RS": m, "t": tok, "vars": tpl.get("vars_reacao_saida",{}).get(m,["0.0"])}
                for m,tok in zip(RSm, RS)
            ],
            "Contexto de Saída": [
                {"CS": m, "t": tok}
                for m,tok in zip(CSm, CS)
            ],
            "Sentimento da Saída": "Neutro",
            "Tendência da Saída": "0.0",
            "Ressonância": False
        },
        "Total de Saída": TM + SM + RSm + CSm
    }
    return bloco, (PIm or CEm or REm or Em)[-1], (CSm or RSm or SM or TM)[-1]

def prompt_vars_and_multivars(bloco: Dict[str, Any], entrada_texto: str) -> None:
    # VARS ATÔMICAS NA ENTRADA
    marc_ent = [e["E"] for e in bloco["Entrada"]]
    print("\nTokens de Entrada segmentados:")
    for i, ent in enumerate(bloco["Entrada"], 1):
        print(f"  {i}) {ent['t']}  [{ent['E']}]")

    while True:
        sel = input("Variação de token? Índice, marcador ou ENTER: ").strip()
        if not sel:
            break
        if sel in marc_ent:
            idx = marc_ent.index(sel)
        elif sel.isdigit() and 1 <= int(sel) <= len(bloco["Entrada"]):
            idx = int(sel) - 1
        else:
            print("Entrada inválida."); continue

        ent = bloco["Entrada"][idx]
        new_vars: List[str] = []
        while True:
            v = input(f"  Nova var para '{ent['t']}' (ENTER interrompe): ").strip()
            if not v:
                break
            new_vars.append(v)
        if new_vars:
            ent["vars"] = new_vars

    # MULTIVARS DE ENTRADA
    ent_mvs: List[str] = []
    print("\nMultivars para a frase de Entrada:")
    while True:
        mv = input("  Digite multivars (ENTER interrompe): ").strip()
        if not mv:
            break
        ent_mvs.append(mv)
    if ent_mvs:
        bloco["Multivars"] = ent_mvs

    # VARS ATÔMICAS NA SAÍDA (TEX e S)
    falas = bloco["Saída"]["Texto"]
    marc_sai = [f.get("S", f.get("TEX")) for f in falas]
    print("\nTokens de Saída segmentados (TEX/S):")
    for i, f in enumerate(falas, 1):
        ident = f.get("S", f.get("TEX"))
        print(f"  {i}) {f['t']}  [{ident}]")

    while True:
        sel = input("Variação de token na Saída? Índice, marcador ou ENTER: ").strip()
        if not sel:
            break
        if sel in marc_sai:
            idx = marc_sai.index(sel)
        elif sel.isdigit() and 1 <= int(sel) <= len(falas):
            idx = int(sel) - 1
        else:
            print("Entrada inválida."); continue

        f = falas[idx]
        new_vars: List[str] = []
        while True:
            v = input(f"  Nova var para '{f['t']}' (ENTER interrompe): ").strip()
            if not v:
                break
            new_vars.append(v)
        if new_vars:
            f["vars"] = new_vars

    # MULTIVARS DE SAÍDA
    sa_mvs: List[str] = []
    print("\nMultivars para a frase de Saída:")
    while True:
        mv = input("  Digite multivars (ENTER interrompe): ").strip()
        if not mv:
            break
        sa_mvs.append(mv)
    if sa_mvs:
        bloco["Saída"]["Multivars"] = sa_mvs

def print_block_summary(bloco: Dict[str, Any]) -> None:
    ent = bloco["Entrada"][0]["t"] if bloco["Entrada"] else ""
    re_ent = bloco["Reação"][0]["t"] if bloco["Reação"] else ""
    ctx_ent = " ".join(c["t"] for c in bloco["Contexto"]) if bloco["Contexto"] else ""
    pin = " ".join(p["t"] for p in bloco["Pensamento Interno"])
    tex_items = bloco["Saída"]["Texto"]
    re_sai = bloco["Saída"]["Reação de Saída"][0]["t"] if bloco["Saída"]["Reação de Saída"] else ""
    ctx_sai = " ".join(c["t"] for c in bloco["Saída"]["Contexto de Saída"]) if bloco["Saída"]["Contexto de Saída"] else ""

    print("\n--- Resumo do Bloco ---")
    print(f'Entrada: "{ent}"')
    print(f'Multivars (Entrada): {bloco.get("Multivars", [])}')
    print(f'Reação: {re_ent} | Contexto: {ctx_ent}')
    print(f'Pensamento Interno: "{pin}"')
    print("Saída → Texto (TEX/S):")
    for item in tex_items:
        ident = item.get("TEX", item.get("S"))
        tipo  = "TEX" if "TEX" in item else "S"
        print(f'  {tipo}={ident} → "{item["t"]}"')
    print(f'Multivars (Saída): {bloco["Saída"].get("Multivars", [])}')
    print(f'Reação de Saída: {re_sai} | Contexto de Saída: {ctx_sai}')
    print("-----------------------")

def derive_um(idx_mae: str) -> None:
    ida_all = load_json(JSON_IDA)
    uni_ida = ida_all["IDA"]["IM"][idx_mae]
    nome = uni_ida.get("nome","")

    um_all = load_json(JSON_UM)
    um_root = um_all.setdefault("UM", {})
    blocos = []

    for b in uni_ida["blocos"]:
        aln_ent = {it["E"]: calcular_alnulu(it["t"]) for it in b["Entrada"]}

        tex_items = b["Saída"]["Texto"]
        aln_tex = {}
        for it in tex_items:
            if "TEX" in it:
                aln_tex[it["TEX"]] = calcular_alnulu(it["t"])
            elif "S" in it:
                aln_tex[it["S"]] = calcular_alnulu(it["t"])

        rs_items = b["Saída"]["Reação de Saída"]
        aln_rs  = {it["RS"]: calcular_alnulu(it["t"]) for it in rs_items}
        cs_items = b["Saída"]["Contexto de Saída"]
        aln_cs  = {it["CS"]: calcular_alnulu(it["t"]) for it in cs_items}

        total_ent = sum(aln_ent.values())
        total_sai = sum(aln_tex.values()) + sum(aln_rs.values()) + sum(aln_cs.values())

        blocos.append({
            "bloco_id": b["bloco_id"],
            "Total de Entrada": b["Total de Entrada"],
            "ultimo_child_entrada": b["Total de Entrada"][-1],
            "Multivars": b.get("Multivars", []),
            "Total de Saída": b["Total de Saída"],
            "ultimo_child_saida": b["Total de Saída"][-1],
            "Multivars de Saída": b["Saída"].get("Multivars", []),
            "alnulu_por_palavra_entrada": aln_ent,
            "alnulu_total_entrada": total_ent,
            "alnulu_por_palavra_saida": {**aln_tex, **aln_rs, **aln_cs},
            "alnulu_total_saida": total_sai,
            "alnulu_total_bloco": total_ent + total_sai
        })

    seq = [blk["bloco_id"] for blk in uni_ida["blocos"]]
    um_root[idx_mae] = {
        "Universo Mãe": nome,
        "blocos": blocos,
        "cb": {"status": "Indisponível", "sdb":[{"sequencia":seq,"VA":False}]}
    }
    save_json(um_all, JSON_UM)

def manage_universes() -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA",{}).setdefault("IM",{})
    while True:
        print("\n=== Universos ===")
        print("1) Listar  2) Criar  3) Renomear  4) Excluir  5) Voltar")
        opt = input("Opção [1-5]: ").strip()
        if opt == "1":
            for idx,u in ims.items():
                print(f"  IM={idx} → '{u.get('nome','')}'")
        elif opt == "2":
            idx = input("Novo IM: ").strip()
            if not idx.isdigit() or idx in ims:
                print("Inválido."); continue
            nome = input("Nome: ").strip()
            if input(f"Criar IM={idx}? (S/N): ").lower()!="s": continue
            ims[idx]={"nome":nome,"blocos":[]}
            save_json(ida,JSON_IDA); derive_um(idx)
        elif opt == "3":
            idx=input("IM para renomear: ").strip()
            if idx not in ims: print("Não existe."); continue
            novo=input("Novo nome: ").strip()
            if input("Confirma? (S/N): ").lower()!="s": continue
            ims[idx]['nome']=novo; save_json(ida,JSON_IDA); derive_um(idx)
        elif opt == "4":
            idx=input("IM para excluir: ").strip()
            if idx not in ims: print("Não existe."); continue
            if input("Excluir? (S/N): ").lower()!="s": continue
            del ims[idx]; save_json(ida,JSON_IDA)
            um = load_json(JSON_UM); um.get("UM",{}).pop(idx,None)
            save_json(um,JSON_UM)
        elif opt == "5":
            break
        else:
            print("Inválido.")

def manage_blocks() -> None:
    ida = load_json(JSON_IDA).get("IDA",{}).get("IM",{})
    idx = input("IM para blocos: ").strip()
    if idx not in ida:
        print("Não encontrado."); return

    print("Cole até 20 blocos, termine com linha em branco:")
    lines: List[str] = []
    while True:
        ln = input().rstrip()
        if not ln: break
        lines.append(ln)

    raw_blocks, current = [], []
    for line in lines:
        if line.strip().startswith("---"):
            if current:
                raw_blocks.append(current); current = []
        else:
            current.append(line)
    if current:
        raw_blocks.append(current)

    for raw in raw_blocks[:20]:
        tpl = parse_block(raw)
        process_block(idx, tpl)
    print(f"{len(raw_blocks[:20])} bloco(s) processado(s).")

def process_block(idx_mae: str, tpl: Dict[str, Any]) -> None:
    ida = load_json(JSON_IDA)
    uni = ida.setdefault("IDA",{}).setdefault("IM",{}).setdefault(idx_mae,{"nome":"","blocos":[]})
    um_all = load_json(JSON_UM).get("UM",{})
    last = f"{idx_mae}.0"
    if um_all.get(idx_mae,{}).get("blocos"):
        last = um_all[idx_mae]["blocos"][-1]["ultimo_child_saida"]

    bloco, _, _ = build_ida_block(tpl, last)
    prompt_vars_and_multivars(bloco, tpl["entrada"]["texto"])
    print_block_summary(bloco)

    if bloco["bloco_id"] is None:
        bloco["bloco_id"] = max((b["bloco_id"] for b in uni["blocos"]), default=0) + 1

    for i,b in enumerate(uni["blocos"]):
        if b["bloco_id"] == bloco["bloco_id"]:
            uni["blocos"][i] = bloco
            break
    else:
        uni["blocos"].append(bloco)

    save_json(ida, JSON_IDA)
    derive_um(idx_mae)

def main() -> None:
    while True:
        print("\n=== Insepa CLI ===")
        print("1) Universos  2) Blocos  3) Sair")
        opt = input("Opção [1-3]: ").strip()
        if opt == "1":
            manage_universes()
        elif opt == "2":
            manage_blocks()
        elif opt == "3":
            print("Saindo."); break
        else:
            print("Inválido.")

if __name__ == "__main__":
    main()


