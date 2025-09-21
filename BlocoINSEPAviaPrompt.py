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

def parse_block(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "bloco_id": None,
        "entrada": {
            "texto": "", "reacao": "", "contexto": "", "pensamento_interno": []
        },
        "saida": {
            "TEX": [], "reacao": "", "contexto": ""
        }
    }
    sec: Optional[str] = None

    for raw in lines:
        line = raw.strip()
        low = line.lower()

        if low.startswith("entrada:"):
            tpl["entrada"]["texto"] = line.partition(":")[2].strip()
            sec = "ent"; continue

        if sec == "ent" and low.startswith("reação:"):
            tpl["entrada"]["reacao"] = line.partition(":")[2].strip(); continue

        if sec == "ent" and low.startswith("contexto:"):
            tpl["entrada"]["contexto"] = line.partition(":")[2].strip(); continue

        if sec == "ent" and low.startswith("pensamento interno:"):
            tpl["entrada"]["pensamento_interno"].append(
                line.partition(":")[2].strip()
            ); continue

        if low.startswith("texto:"):
            content = line.partition(":")[2].strip()
            tpl["saida"]["TEX"].insert(0, content)
            sec = "out_txt"; continue

        if low == "saída:":
            sec = "out_txt"; continue

        if sec == "out_txt" and line.startswith("—"):
            tpl["saida"]["TEX"].append(line.lstrip("—").strip()); continue

        if sec in ("out_txt", "out_meta") and low.startswith("reação:"):
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
    # 1) tokenização de entrada
    E   = Token(tpl["entrada"]["texto"])
    RE  = Token(tpl["entrada"]["reacao"])
    CE  = Token(tpl["entrada"]["contexto"])
    SI  = tpl["entrada"]["pensamento_interno"][:3]

    # 2) split narrativo em segmentos por aspas
    raw   = tpl["saida"]["TEX"].pop(0)
    parts = re.split(r'(".*?")', raw)
    parts = [p.strip() for p in parts if p.strip()]

    # 3) falas restantes
    fale_strs = tpl["saida"]["TEX"]

    # 4) reação/contexto de saída
    RS = Token(tpl["saida"]["reacao"])
    CS = Token(tpl["saida"]["contexto"])

    # 5) quantos marcadores TEX precisamos
    seg_count   = len(parts)
    fala_tok_ct = [len(Token(f)) for f in fale_strs]
    TEX_count   = seg_count + sum(fala_tok_ct)

    # 6) total geral
    total = len(E) + len(RE) + len(CE) + len(SI) \
          + TEX_count + len(RS) + len(CS)

    # 7) gera marcadores
    marks = generate_markers(last_marker, total)
    ptr = 0

    Em  = marks[ptr:ptr+len(E)];    ptr += len(E)
    REm = marks[ptr:ptr+len(RE)];   ptr += len(RE)
    CEm = marks[ptr:ptr+len(CE)];   ptr += len(CE)
    PIm = marks[ptr:ptr+len(SI)];   ptr += len(SI)

    SM = marks[ptr:ptr+seg_count];  ptr += seg_count

    FM_slices: List[List[str]] = []
    for ct in fala_tok_ct:
        FM_slices.append(marks[ptr:ptr+ct])
        ptr += ct

    RSm = marks[ptr:ptr+len(RS)];    ptr += len(RS)
    CSm = marks[ptr:ptr+len(CS)]

    # 8) monta JSON do bloco
    bloco = {
        "bloco_id": tpl.get("bloco_id"),
        "Entrada": [
            {"E": m, "t": tok, "vars": tpl.get("vars_entrada",{}).get(m,["0.0"])}
            for m,tok in zip(Em, E)
        ],
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
            "Tex": {
                "S": [
                    *[
                        {"S": sm, "t": part, "vars": tpl.get("vars_TEX",{}).get(sm,["0.0"])}
                        for sm,part in zip(SM, parts)
                    ],
                    *[
                        {"S": m, "t": tok, "vars": tpl.get("vars_TEX",{}).get(m,["0.0"])}
                        for slice_, fala in zip(FM_slices, fale_strs)
                        for m,tok in zip(slice_, Token(fala))
                    ]
                ]
            },
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
        "Total de Saída": SM + [m for sl in FM_slices for m in sl] + RSm + CSm
    }
    return bloco, (PIm or CEm or REm or Em)[-1], (CSm or RSm or SM)[-1]

def print_block_summary(bloco: Dict[str, Any]) -> None:
    ent_lst     = bloco["Entrada"]
    re_ent_lst  = bloco["Reação"]
    ctx_ent_lst = bloco["Contexto"]
    sent_ent    = bloco["Sentimento da Entrada"]
    tend_ent    = bloco["Tendência da Entrada"]
    pin_lst     = bloco["Pensamento Interno"]
    tex_items   = bloco["Saída"]["Tex"]["S"]
    re_sai_lst  = bloco["Saída"]["Reação de Saída"]
    ctx_sai_lst = bloco["Saída"]["Contexto de Saída"]

    e_txt   = ent_lst[0]["t"] if ent_lst else ""
    r_txt   = re_ent_lst[0]["t"] if re_ent_lst else ""
    c_txt   = ctx_ent_lst[0]["t"] if ctx_ent_lst else ""
    pin_txt = " ".join(p["t"] for p in pin_lst)
    r_sai   = re_sai_lst[0]["t"] if re_sai_lst else ""
    c_sai   = ctx_sai_lst[0]["t"] if ctx_sai_lst else ""

    print("Bloco:")
    print(f'Entrada: "{e_txt}"')
    print(f'Reação: {r_txt} Contexto: {c_txt}')
    print(f'Sentimento da Entrada: {sent_ent} Tendência da Entrada: {tend_ent}')
    print(f'Pensamento Interno: "{pin_txt}"')
    print("Saída → Tex → S:")
    for item in tex_items:
        print(f'  {{S: "{item["S"]}", t: "{item["t"]}"}}')
    print(f'Reação de Saída: {r_sai}')
    print(f'Contexto de Saída: {c_sai}')
    print('---')

def derive_um(idx_mae: str) -> None:
    ida_all = load_json(JSON_IDA)
    uni_ida = ida_all["IDA"]["IM"][idx_mae]
    nome_mae = uni_ida.get("nome","")

    um_all = load_json(JSON_UM)
    um_root = um_all.setdefault("UM", {})

    blocos = []
    for b in uni_ida["blocos"]:
        aln_ent = {it["E"]: calcular_alnulu(it["t"]) for it in b["Entrada"]}
        tex_items = b["Saída"]["Tex"]["S"]
        aln_tex = {it["S"]: calcular_alnulu(it["t"]) for it in tex_items}
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
            "alnulu_por_palavra_entrada": aln_ent,
            "alnulu_total_entrada": total_ent,
            "Total de Saída": b["Total de Saída"],
            "ultimo_child_saida": b["Total de Saída"][-1],
            "alnulu_por_palavra_saida": {**aln_tex, **aln_rs, **aln_cs},
            "alnulu_total_saida": total_sai,
            "alnulu_total_bloco": total_ent + total_sai
        })

    seq = [blk["bloco_id"] for blk in uni_ida["blocos"]]
    um_root[idx_mae] = {
        "Universo Mãe": nome_mae,
        "blocos": blocos,
        "cb": {"status": "Indisponível", "sdb":[{"sequencia":seq,"VA":False}]}
    }
    save_json(um_all, JSON_UM)

def manage_universes() -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA",{}).setdefault("IM",{})

    while True:
        print("\n=== Gerenciar Universos (IDA) ===")
        print("1) Listar  2) Criar  3) Renomear  4) Excluir  5) Voltar")
        opt = input("Escolha [1-5]: ").strip()
        if opt == "1":
            if not ims:
                print("  (nenhum universo)")
            else:
                for idx,u in ims.items():
                    print(f"  IM={idx} → '{u.get('nome','')}'")
        elif opt == "2":
            idx = input("Novo índice IM: ").strip()
            if not idx.isdigit() or idx in ims:
                print("Inválido ou já existe."); continue
            nome= input("Nome do Universo: ").strip()
            if input(f"Criar IM={idx} '{nome}'? (S/N): ").lower()!="s":
                print("Cancelado."); continue
            ims[idx]={"nome":nome,"blocos":[]}
            save_json(ida,JSON_IDA); derive_um(idx)
        elif opt=="3":
            idx=input("IM a renomear: ").strip()
            if idx not in ims: print("Não existe."); continue
            novo=input(f"Novo nome ('{ims[idx]['nome']}'): ").strip()
            if input("Confirma? (S/N): ").lower()!="s": print("Cancelado."); continue
            ims[idx]['nome']=novo; save_json(ida,JSON_IDA); derive_um(idx)
        elif opt=="4":
            idx=input("IM para excluir: ").strip()
            if idx not in ims: print("Não existe."); continue
            if input(f"Excluir IM={idx}? (S/N): ").lower()!="s":
                print("Cancelado."); continue
            del ims[idx]; save_json(ida,JSON_IDA)
            um = load_json(JSON_UM); um.get("UM",{}).pop(idx,None)
            save_json(um,JSON_UM)
        elif opt=="5":
            break
        else:
            print("Inválido.")

def manage_blocks() -> None:
    ida = load_json(JSON_IDA).get("IDA",{}).get("IM",{})
    idx = input("IM para gerir blocos: ").strip()
    if idx not in ida:
        print("Universo não encontrado."); return

    print("Cole até 20 blocos, separados por '---'. Termine com linha em branco:")
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

    count = min(len(raw_blocks),20)
    for raw in raw_blocks[:count]:
        tpl = parse_block(raw)
        process_block(idx, tpl)
    print(f"{count} bloco(s) processado(s).")

def process_block(idx_mae: str, tpl: Dict[str, Any]) -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA",{}).setdefault("IM",{})
    uni = ims.setdefault(idx_mae,{"nome":"","blocos":[]})

    um_all = load_json(JSON_UM).get("UM",{})
    last = f"{idx_mae}.0"
    if um_all.get(idx_mae,{}).get("blocos"):
        last = um_all[idx_mae]["blocos"][-1]["ultimo_child_saida"]

    bloco, _, _ = build_ida_block(tpl,last)
    print_block_summary(bloco)

    if bloco["bloco_id"] is None:
        bloco["bloco_id"] = max((b["bloco_id"] for b in uni["blocos"]),default=0)+1

    for i,b in enumerate(uni["blocos"]):
        if b["bloco_id"]==bloco["bloco_id"]:
            uni["blocos"][i]=bloco; break
    else:
        uni["blocos"].append(bloco)

    save_json(ida,JSON_IDA)
    derive_um(idx_mae)

def main() -> None:
    while True:
        print("\n=== Insepa CLI ===")
        print("1) Gerenciar Universos  2) Gerenciar Blocos  3) Sair")
        opt = input("Escolha [1-3]: ").strip()
        if opt == "1":
            manage_universes()
        elif opt == "2":
            manage_blocks()
        elif opt == "3":
            print("Encerrando."); break
        else:
            print("Inválido.")

if __name__ == "__main__":
    main()


