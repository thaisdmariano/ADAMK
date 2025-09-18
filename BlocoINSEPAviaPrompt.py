#!/usr/bin/env python3
# coding: utf-8
"""
insepa_update.py — CLI para INSEPA com JSON no molde pedido

• inconsciente.json (IDA) → cada IM tem "nome" e lista de "blocos"
  – fases com marcadores; vars só em E, RE, EXDS, IME, S (default ["0.0"])
  – PIDE, EXDS, IME são segmentos completos (até 3 itens)
  – Sentimento/Tendência da Entrada e Saída = "Neutro"/"0.0"
  – não calcula alnulu aqui
• adam_memoria.json (UM) → cada UM tem "Universo Mãe", "blocos" com alnulu e "cb"
"""

import sys, json, re
from typing import List, Dict, Any, Optional, Tuple

JSON_IDA = "inconsciente.json"
JSON_UM  = "adam_memoria.json"

def load_json(path: str) -> Dict[str,Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        sys.exit(f"Erro ao ler {path}: {e}")

def save_json(obj: Dict[str,Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def Token(text: str) -> List[str]:
    clean = text.replace('"', '')
    return re.findall(r'\w+|[^\w\s]', clean, re.UNICODE)

def next_marker(prev: str) -> str:
    mom, _, suf = prev.partition('.')
    return f"{mom}.{int(suf or '0') + 1}"

def generate_markers(start: str, count: int) -> List[str]:
    seq, cur = [], start
    for _ in range(count):
        cur = next_marker(cur)
        seq.append(cur)
    return seq

# —————————————————————————————————————————————————————————————
# cálculo de alnulu exato conforme mapa/equiv fornecido
# —————————————————————————————————————————————————————————————
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
        up   = ch.upper()
        norm = equiv.get(up, up)
        total += mapa.get(norm, 0)
    return total

# —————————————————————————————————————————————————————————————
# parse_block corrigido para não falhar em None e tratar explicação/imersão
# —————————————————————————————————————————————————————————————
def parse_block(lines: List[str]) -> Dict[str,Any]:
    tpl = {
        "bloco_id": None,
        "entrada": {"texto":"","reacao":"","contexto":"","pensamento_interno":[]},
        "saida":   {"textos":[],"explicacao":[],"imersao":[],"reacao":"","contexto":""}
    }

    section: Optional[str] = None
    for raw in lines:
        line = raw.strip()

        # Entrada
        if line.lower().startswith("entrada:"):
            tpl["entrada"]["texto"] = line.partition(":")[2].strip()
            section = "ent"; continue

        if section=="ent" and line.lower().startswith("reação:"):
            tpl["entrada"]["reacao"] = line.partition(":")[2].strip(); continue

        if section=="ent" and line.lower().startswith("contexto:"):
            tpl["entrada"]["contexto"] = line.partition(":")[2].strip(); continue

        if section=="ent" and line.lower().startswith("pensamento interno:"):
            tpl["entrada"]["pensamento_interno"].append(line.partition(":")[2].strip()); continue

        # Saída
        if line.lower() == "saída:":
            section = "out_txt"; continue

        if section=="out_txt" and re.match(r'^\d+\.\s+', line):
            tpl["saida"]["textos"].append(line.partition(" ")[2].strip()); continue

        if line.lower().startswith("explicação:"):
            tpl["saida"]["explicacao"].append(line.partition(":")[2].strip()); continue

        if line.lower().startswith("imersão:"):
            tpl["saida"]["imersao"].append(line.partition(":")[2].strip()); continue

        if section in ("out_txt","out_meta") and line.lower().startswith("reação:"):
            tpl["saida"]["reacao"] = line.partition(":")[2].strip()
            section = "out_meta"; continue

        if section=="out_meta" and line.lower().startswith("contexto:"):
            tpl["saida"]["contexto"] = line.partition(":")[2].strip(); continue

    if not tpl["entrada"]["texto"]:
        sys.exit("Erro: cada bloco precisa de 'Entrada'.")
    return tpl

# —————————————————————————————————————————————————————————————
# build_ida_block: tokeniza e monta bloco para IDA (sem alnulu)
# —————————————————————————————————————————————————————————————
def build_ida_block(
    tpl: Dict[str,Any],
    last_marker: str
) -> Tuple[Dict[str,Any], str, str]:
    # tokenização básica
    E  = Token(tpl["entrada"]["texto"])
    RE = Token(tpl["entrada"]["reacao"])
    CE = Token(tpl["entrada"]["contexto"])
    SI = tpl["entrada"]["pensamento_interno"][:3]   # segmentos
    EX = tpl["saida"]["explicacao"][:3]
    IM = tpl["saida"]["imersao"][:3]

    texts = []
    for t in tpl["saida"]["textos"]:
        texts += Token(t)
    RS = Token(tpl["saida"]["reacao"])
    CS = Token(tpl["saida"]["contexto"])

    total = len(E)+len(RE)+len(CE)+len(SI)+len(EX)+len(IM)+len(texts)+len(RS)+len(CS)
    marks = generate_markers(last_marker, total)

    ptr = 0
    Em  = marks[ptr: ptr+len(E)];    ptr+=len(E)
    REm = marks[ptr: ptr+len(RE)];   ptr+=len(RE)
    CEm = marks[ptr: ptr+len(CE)];   ptr+=len(CE)
    PIm = marks[ptr: ptr+len(SI)];   ptr+=len(SI)
    EXm = marks[ptr: ptr+len(EX)];   ptr+=len(EX)
    IMm = marks[ptr: ptr+len(IM)];   ptr+=len(IM)
    Sm  = marks[ptr: ptr+len(texts)]; ptr+=len(texts)
    RSm = marks[ptr: ptr+len(RS)];    ptr+=len(RS)
    CSm = marks[ptr: ptr+len(CS)]

    fim_ent = CEm[-1] if CEm else last_marker
    fim_out = CSm[-1] if CSm else fim_ent

    def gv(phase: str, mark: str) -> List[str]:
        return tpl.get(f"vars_{phase}", {}).get(mark, ["0.0"])

    bloco = {
        "bloco_id": tpl.get("bloco_id"),
        "Entrada": [
            {"E": m, "t": tok, "vars": gv("entrada", m)}
            for m,tok in zip(Em, E)
        ],
        "Reação": [
            {"RE": m, "t": tok, "vars": gv("reacao", m)}
            for m,tok in zip(REm, RE)
        ],
        "Contexto": [
            {"CE": m, "t": tok}
            for m,tok in zip(CEm, CE)
        ],
        "Sentimento da Entrada": "Neutro",
        "Tendência da Entrada": "0.0",
        "Pensamento Interno": [
            {"PIDE": m, "t": seg}
            for m,seg in zip(PIm, SI)
        ],
        "Total de Entrada": Em + REm + CEm + PIm,
        "Saída": {
            "Explicação": [
                {"EXDS": m, "t": seg}
                for m,seg in zip(EXm, EX)
            ],
            "Imersão": [
                {"IME": m, "t": seg, "vars": gv("imersao", m)}
                for m,seg in zip(IMm, IM)
            ],
            "Textos": [
                {"S": m, "t": tok, "vars": gv("textos", m)}
                for m,tok in zip(Sm, texts)
            ],
            "Reação de Saída": [
                {"RS": m, "t": tok}
                for m,tok in zip(RSm, RS)
            ],
            "Contexto de Saída": [
                {"CS": m, "t": tok}
                for m,tok in zip(CSm, CS)
            ],
            "Sentimento da Saída": "Neutro",
            "Tendência da Saída": "0.0",
            "Ressonância": True
        },
        "Total de Saída": EXm + IMm + Sm + RSm + CSm
    }

    return bloco, fim_ent, fim_out

def process_block(idx_mae: str, tpl: Dict[str,Any]) -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA", {}).setdefault("IM", {})
    uni = ims.setdefault(idx_mae, {"nome":"", "blocos":[]})

    um_all = load_json(JSON_UM).get("UM", {})
    last = f"{idx_mae}.0"
    if um_all.get(idx_mae,{}).get("blocos"):
        last = um_all[idx_mae]["blocos"][-1]["ultimo_child_saida"]

    bloco, _, _ = build_ida_block(tpl, last)
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

def derive_um(idx_mae: str) -> None:
    ida_all  = load_json(JSON_IDA)
    uni_ida  = ida_all["IDA"]["IM"][idx_mae]
    nome_mae = uni_ida.get("nome","")

    um_all  = load_json(JSON_UM)
    um_root = um_all.setdefault("UM", {})

    blocos = []
    for b in uni_ida["blocos"]:
        text_ent = "".join(item["t"] for item in b["Entrada"])
        blocos.append({
            "bloco_id": b["bloco_id"],
            "Total de Entrada":     b["Total de Entrada"],
            "ultimo_child_entrada": b["Total de Entrada"][-1],
            "alnulu":               calcular_alnulu(text_ent),
            "Total de Saída":       b["Total de Saída"],
            "ultimo_child_saida":   b["Total de Saída"][-1]
        })

    seq = [b["bloco_id"] for b in uni_ida["blocos"]]
    sdb = um_root.get(idx_mae,{}).get("cb",{}).get("sdb",[])
    sdb.append({"sequencia": seq, "VA": False})

    um_root[idx_mae] = {
        "Universo Mãe": nome_mae,
        "blocos":       blocos,
        "cb": {
            "status": uni_ida.get("cb",{}).get("status","Indisponível"),
            "sdb":    sdb
        }
    }
    save_json(um_all, JSON_UM)

def manage_universes() -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA", {}).setdefault("IM", {})

    while True:
        print("\n=== Gerenciar Universos (IDA) ===")
        print("1) Listar\n2) Criar\n3) Renomear\n4) Excluir\n5) Voltar")
        opt = input("Escolha [1-5]: ").strip()

        if opt == "1":
            if not ims:
                print("  (nenhum universo cadastrado)")
            else:
                for idx,u in ims.items():
                    print(f"  IM={idx} → nome='{u.get('nome','')}'")

        elif opt == "2":
            idx  = input("Novo índice IM: ").strip()
            if not idx.isdigit() or idx in ims:
                print("Índice inválido ou já existe."); continue
            nome = input("Nome do Universo: ").strip()
            if input(f"Criar IM={idx} '{nome}'? (S/N): ").lower() != "s":
                print("Cancelado."); continue
            ims[idx] = {"nome": nome, "blocos": []}
            save_json(ida, JSON_IDA)
            derive_um(idx)

        elif opt == "3":
            idx = input("IM a renomear: ").strip()
            uni = ims.get(idx)
            if not uni:
                print("Não encontrado."); continue
            novo = input(f"Novo nome para IM={idx} ('{uni['nome']}'): ").strip()
            if input("Confirmar? (S/N): ").lower() != "s":
                print("Cancelado."); continue
            uni["nome"] = novo
            save_json(ida, JSON_IDA)
            derive_um(idx)

        elif opt == "4":
            idx = input("IM para excluir: ").strip()
            if idx not in ims:
                print("Não encontrado."); continue
            if input(f"Excluir IM={idx} '{ims[idx]['nome']}'? (S/N): ").lower() != "s":
                print("Cancelado."); continue
            del ims[idx]
            save_json(ida, JSON_IDA)
            um = load_json(JSON_UM)
            um.get("UM",{}).pop(idx,None)
            save_json(um, JSON_UM)

        elif opt == "5":
            break
        else:
            print("Opção inválida.")

def manage_blocks() -> None:
    ida = load_json(JSON_IDA).get("IDA",{}).get("IM",{})
    idx = input("Índice IM para gerir blocos: ").strip()
    if idx not in ida:
        print("Universo não encontrado."); return
    print("Cole um bloco e termine em branco:")
    lines: List[str] = []
    while True:
        ln = input().rstrip()
        if not ln: break
        lines.append(ln)
    tpl = parse_block(lines)
    process_block(idx, tpl)

def main() -> None:
    while True:
        print("\n=== Insepa CLI ===")
        print("1) Gerenciar Universos\n2) Gerenciar Blocos\n3) Sair")
        opt = input("Escolha [1-3]: ").strip()
        if opt == "1":
            manage_universes()
        elif opt == "2":
            manage_blocks()
        elif opt == "3":
            print("Encerrando."); break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()

