#!/usr/bin/env python3
# coding: utf-8
"""
insepa_update.py — CLI para INSEPA com JSON no molde pedido

• inconsciente.json (IDA):
    – Blocos com Entrada, Reação, Contexto, Pensamento Interno, Saída
    – Todas as fases recebem marcador neutro x.0 se vierem vazias
    – Vars apenas em E, RE, EXDS, IME, S e RS
    – Sentimento/Tendência (Entrada e Saída): "Neutro"/"0.0"
    – Ressonância = False se Saída estiver Neutro/0.0

• adam_memoria.json (UM):
    – "Universo Mãe", lista de "blocos" e "cb"
    – Para cada bloco:
        • Total de Entrada / Saída (listas de marcadores)
        • ultimo_child_entrada / ultimo_child_saida
        • alnulu_por_palavra_entrada / alnulu_por_palavra_saida
        • alnulu_total_entrada / alnulu_total_saida / alnulu_total_bloco
    – Em “cb.sdb” guarda apenas um snapshot com a sequência atual de blocos
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
        norm = equiv.get(ch.upper(), ch.upper())
        total += mapa.get(norm, 0)
    return total

def parse_block(lines: List[str]) -> Dict[str,Any]:
    tpl = {
        "bloco_id": None,
        "entrada": {"texto":"", "reacao":"", "contexto":"", "pensamento_interno":[]},
        "saida":   {"textos":[], "explicacao":[], "imersao":[], "reacao":"", "contexto":""}
    }
    sec: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        low  = line.lower()
        if low.startswith("entrada:"):
            tpl["entrada"]["texto"] = line.partition(":")[2].strip()
            sec = "ent"; continue
        if sec=="ent" and low.startswith("reação:"):
            tpl["entrada"]["reacao"] = line.partition(":")[2].strip(); continue
        if sec=="ent" and low.startswith("contexto:"):
            tpl["entrada"]["contexto"] = line.partition(":")[2].strip(); continue
        if sec=="ent" and low.startswith("pensamento interno:"):
            tpl["entrada"]["pensamento_interno"].append(line.partition(":")[2].strip()); continue
        if low == "saída:":
            sec = "out_txt"; continue
        if sec=="out_txt" and re.match(r'^\d+\.\s+', line):
            tpl["saida"]["textos"].append(line.partition(" ")[2].strip()); continue
        if low.startswith("explicação:"):
            tpl["saida"]["explicacao"].append(line.partition(":")[2].strip()); continue
        if low.startswith("imersão:"):
            tpl["saida"]["imersao"].append(line.partition(":")[2].strip()); continue
        if sec in ("out_txt","out_meta") and low.startswith("reação:"):
            tpl["saida"]["reacao"] = line.partition(":")[2].strip(); sec="out_meta"; continue
        if sec=="out_meta" and low.startswith("contexto:"):
            tpl["saida"]["contexto"] = line.partition(":")[2].strip(); continue
    if not tpl["entrada"]["texto"]:
        sys.exit("Erro: cada bloco precisa de 'Entrada'.")
    return tpl

def build_ida_block(
    tpl: Dict[str,Any],
    last_marker: str
) -> Tuple[Dict[str,Any], str, str]:
    E  = Token(tpl["entrada"]["texto"])
    RE = Token(tpl["entrada"]["reacao"])
    CE = Token(tpl["entrada"]["contexto"])
    SI = tpl["entrada"]["pensamento_interno"][:3]
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
    Em  = marks[ptr:ptr+len(E)];    ptr+=len(E)
    REm = marks[ptr:ptr+len(RE)];   ptr+=len(RE)
    CEm = marks[ptr:ptr+len(CE)];   ptr+=len(CE)
    PIm = marks[ptr:ptr+len(SI)];   ptr+=len(SI)
    EXm = marks[ptr:ptr+len(EX)];   ptr+=len(EX)
    IMm = marks[ptr:ptr+len(IM)];   ptr+=len(IM)
    Sm  = marks[ptr:ptr+len(texts)]; ptr+=len(texts)
    RSm = marks[ptr:ptr+len(RS)];    ptr+=len(RS)
    CSm = marks[ptr:ptr+len(CS)]

    universo = last_marker.split('.')[0]
    neutro   = f"{universo}.0"

    if not Em:  Em  = [neutro]
    if not REm: REm = [neutro]
    if not CEm: CEm = [neutro]
    if not PIm: PIm = [neutro]
    if not EXm: EXm = [neutro]
    if not IMm: IMm = [neutro]
    if not Sm:  Sm  = [neutro]
    if not RSm: RSm = [neutro]
    if not CSm: CSm = [neutro]

    fim_ent = (CEm or PIm or REm or Em)[-1]
    fim_out = (CSm or RSm or Sm or IMm or EXm)[-1]

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
                {"EXDS": m, "t": seg, "vars": gv("explicacao", m)}
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
                {"RS": m, "t": tok, "vars": gv("reacao_saida", m)}
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
        "Total de Saída": EXm + IMm + Sm + RSm + CSm
    }

    return bloco, fim_ent, fim_out

def process_block(idx_mae: str, tpl: Dict[str,Any]) -> None:
    ida     = load_json(JSON_IDA)
    ims     = ida.setdefault("IDA", {}).setdefault("IM", {})
    uni     = ims.setdefault(idx_mae, {"nome":"", "blocos":[]})

    um_all  = load_json(JSON_UM).get("UM", {})
    last    = f"{idx_mae}.0"
    if um_all.get(idx_mae, {}).get("blocos"):
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

    um_all   = load_json(JSON_UM)
    um_root  = um_all.setdefault("UM", {})

    blocos = []
    for b in uni_ida["blocos"]:
        aln_ent   = {it["E"]: calcular_alnulu(it["t"]) for it in b["Entrada"]}
        aln_txt   = {it["S"]: calcular_alnulu(it["t"]) for it in b["Saída"]["Textos"]}
        aln_rs    = {it["RS"]: calcular_alnulu(it["t"]) for it in b["Saída"]["Reação de Saída"]}
        aln_cs    = {it["CS"]: calcular_alnulu(it["t"]) for it in b["Saída"]["Contexto de Saída"]}

        aln_saida = {}
        aln_saida.update(aln_txt)
        aln_saida.update(aln_rs)
        aln_saida.update(aln_cs)

        total_ent = sum(aln_ent.values())
        total_sai = sum(aln_saida.values())

        blocos.append({
            "bloco_id":                   b["bloco_id"],
            "Total de Entrada":           b["Total de Entrada"],
            "ultimo_child_entrada":       b["Total de Entrada"][-1],
            "alnulu_por_palavra_entrada": aln_ent,
            "alnulu_total_entrada":       total_ent,
            "Total de Saída":             b["Total de Saída"],
            "ultimo_child_saida":         b["Total de Saída"][-1],
            "alnulu_por_palavra_saida":   aln_saida,
            "alnulu_total_saida":         total_sai,
            "alnulu_total_bloco":         total_ent + total_sai
        })

    seq = [blk["bloco_id"] for blk in uni_ida["blocos"]]
    # gera somente um snapshot por vez, sem repetir sequência vazia
    if seq:
        sdb = [{"sequencia": seq, "VA": False}]
    else:
        sdb = []

    um_root[idx_mae] = {
        "Universo Mãe": nome_mae,
        "blocos":       blocos,
        "cb": {
            "status": uni_ida.get("cb", {}).get("status","Indisponível"),
            "sdb":    sdb
        }
    }

    save_json(um_all, JSON_UM)

def manage_universes() -> None:
    ida = load_json(JSON_IDA)
    ims = ida.setdefault("IDA", {}).setdefault("IM", {})

    while True:
        print("\n=== Gerenciar Universos (IDA) ===")
        print("1) Listar  2) Criar  3) Renomear  4) Excluir  5) Voltar")
        opt = input("Escolha [1-5]: ").strip()
        if opt == "1":
            if not ims:
                print("  (nenhum universo cadastrado)")
            else:
                for idx,u in ims.items():
                    print(f"  IM={idx} → nome='{u.get('nome','')}'")
        elif opt == "2":
            idx = input("Novo índice IM: ").strip()
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
            um.get("UM", {}).pop(idx, None)
            save_json(um, JSON_UM)
        elif opt == "5":
            break
        else:
            print("Opção inválida.")

def manage_blocks() -> None:
    ida = load_json(JSON_IDA).get("IDA", {}).get("IM", {})
    idx = input("Índice IM para gerir blocos: ").strip()
    if idx not in ida:
        print("Universo não encontrado."); return
    print("Cole um bloco e termine em branco:")
    lines: List[str] = []
    while True:
        ln = input().rstrip()
        if not ln:
            break
        lines.append(ln)
    tpl = parse_block(lines)
    process_block(idx, tpl)

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
            print("Opção inválida.")

if __name__ == "__main__":
    main()

