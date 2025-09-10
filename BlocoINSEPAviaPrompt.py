#!/usr/bin/env python3
# coding: utf-8
r"""
insepa_update.py — menu interativo + INSEPA CRUD básico
com validador de Índice Mãe duplicado

Fluxo:
 1. python insepa_update.py
 2. Lista IMs existentes e opções:
      (Enter p/ prosseguir / E p/ editar / I p/ info / 'Sair' p/ encerrar)
 3. 'I' → ver detalhes, com Enter p/ prosseguir, V p/ voltar, 'Sair' p/ encerrar
 4. Enter → cola o template INSEPA e pressiona Enter em branco p/ finalizar
 5. Valida: se IM existe, checa nome; se não existir, cria IM novo
 6. Gera marcadores, calcula ALNULU, anexa bloco, atualiza cb.sdb e salva JSON
"""

import sys
import json
import re
from typing import List, Dict, Any, Optional

def Token(text: str) -> List[str]:
    r"""INSEPA tokenização: palavras, pontuação, emojis, stopwords."""
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def next_marker(prev: str) -> str:
    mom, _, suf = prev.partition('.')
    if not mom.isdigit():
        raise ValueError(f"Marcador inválido: {prev!r}")
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
        up = ch.upper()
        norm = equiv.get(up, up)
        total += mapa.get(norm, 0)
    return total

def parse_template(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "indice_mae": None, "nome": "",
        "entrada": {"texto": "", "reacao": "", "contexto": "", "pensamento_interno": ""},
        "saida":   {"textos": [], "reacao": "", "contexto": "", "explicacao": "", "imersao": ""}
    }
    section: Optional[str] = None
    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue
        if m := re.match(r'^Índice mãe:\s*(\d+)', line):
            tpl["indice_mae"] = int(m.group(1)); continue
        if m := re.match(r'^Nome:\s*(.+)', line):
            tpl["nome"] = m.group(1).strip(); continue
        if m := re.match(r'^Entrada:\s*(.+)', line):
            tpl["entrada"]["texto"] = m.group(1).strip()
            section = "entrada"; continue
        if line.startswith("Saída:"):
            section = "saida_textos"; continue
        if m := re.match(r'^[Rr]eação:\s*(.+)', line):
            target = tpl["entrada"] if section=="entrada" else tpl["saida"]
            target["reacao"] = m.group(1).strip(); continue
        if section == "entrada":
            if m := re.match(r'^[Tt]exto:\s*(.+)', line):
                tpl["entrada"]["texto"] = m.group(1).strip()
            elif m := re.match(r'^[Cc]ontexto:\s*(.+)', line):
                tpl["entrada"]["contexto"] = m.group(1).strip()
            elif m := re.match(r'^[Pp]ensamento\s+[Ii]nterno:\s*(.+)', line):
                tpl["entrada"]["pensamento_interno"] = m.group(1).strip()
            continue
        if section == "saida_textos":
            if m := re.match(r'^\d+\.\s*(.+)', line):
                tpl["saida"]["textos"].append(m.group(1).strip()); continue
            section = "saida_meta"
        if section == "saida_meta":
            if m := re.match(r'^[Cc]ontexto:\s*(.+)', line):
                tpl["saida"]["contexto"] = m.group(1).strip()
            elif m := re.match(r'^[Ee]xplica[cç][ãa]o:\s*(.+)', line):
                tpl["saida"]["explicacao"] = m.group(1).strip()
            elif m := re.match(r'^[Ii]mers[ãa]o:\s*(.+)', line):
                tpl["saida"]["imersao"] = m.group(1).strip()
            continue
    if tpl["indice_mae"] is None:
        sys.exit("Erro: 'Índice mãe' não encontrado no template.")
    return tpl  # type: ignore

def show_summary(base: Dict[str, Any]) -> None:
    print("\n=== Índices Mãe Existentes em adam_memoria.json ===")
    ims = base.get("IM", {})
    if not ims:
        print("  (nenhum universo IM cadastrado)")
    for mom, uni in ims.items():
        print(f"  IM '{mom}': nome='{uni.get('nome','')}'")

def show_info(base: Dict[str, Any], mom: str) -> None:
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print(f"IM '{mom}' não encontrado.")
        return
    print(f"\nÍndice mãe: {mom}")
    print(f"Nome: {uni.get('nome','')}")
    for bloco in uni.get("blocos", []):
        print(f"\n🧱 Bloco {bloco['bloco_id']}")
        e = bloco["entrada"]
        print(f"Entrada: {e['texto']}")
        print(f"Reação: {e['reacao']}")
        print(f"Contexto: {e['contexto']}")
        print(f"Pensamento interno: \"{e['pensamento_interno']}\"")
        print("Saída:")
        s0 = bloco["saidas"][0]
        for i, txt in enumerate(s0["textos"], 1):
            print(f"{i}. \t{txt}")
        print(f"Reação: {s0['reacao']}")
        print(f"Contexto: {s0['contexto']}")
        print(f"Explicação: {s0['explicacao']}")
        print(f"Imersão: {s0['imersao']}")
    print()

def main():
    base_file = "adam_memoria.json"
    try:
        with open(base_file, encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = {"IM": {}}
    except Exception as e:
        print(f"Erro ao abrir {base_file}: {e}")
        sys.exit(1)

    # Menu interativo
    while True:
        show_summary(base)
        opt = input(
            "\n(Enter p/ prosseguir / E p/ editar / I p/ info / 'Sair' p/ encerrar): "
        ).strip().lower()
        if opt in ("sair", "exit"):
            print("Encerrando.")
            sys.exit(0)
        if opt == "i":
            mom = input("\nDigite o número do índice mãe: ").strip()
            show_info(base, mom)
            cont = input(
                "(Enter p/ prosseguir / V p/ voltar ao menu / 'Sair' p/ encerrar): "
            ).strip().lower()
            if cont in ("sair","exit"):
                print("Encerrando."); sys.exit(0)
            if cont == "v":
                continue
            break
        if opt == "e":
            print("Funcionalidade CRUD ainda não implementada.\n")
            continue
        break

    # Cole e parseie template INSEPA
    print("\nCole seu bloco template INSEPA e pressione Enter em branco para finalizar:")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)

    tpl = parse_template(lines)
    mom = str(tpl["indice_mae"])
    universo = base["IM"].get(mom)

    # Validador: não permite criar IM duplicado
    if universo:
        # se nome difere do existente, aborta
        nome_exist = universo.get("nome","")
        if tpl["nome"] and tpl["nome"] != nome_exist:
            print(f"Erro: Índice mãe '{mom}' já existe com nome '{nome_exist}'.")
            sys.exit(1)
    else:
        universo = {
            "nome": tpl["nome"] or f"IM_{mom}",
            "ultimo_child": f"{mom}.0",
            "blocos": [], "cb": {"status":"Indisponível","sdb":[]}
        }
        base["IM"][mom] = universo

    last = universo["ultimo_child"]
    alnulu_val = calcular_alnulu(tpl["entrada"]["texto"])

    # tokenização de entrada
    E    = Token(tpl["entrada"]["texto"])
    RE   = Token(tpl["entrada"]["reacao"])
    CE   = Token(tpl["entrada"]["contexto"])
    PIDE = Token(tpl["entrada"]["pensamento_interno"])[:3]

    # tokenização de saída
    S: List[str] = []
    for t in tpl["saida"]["textos"]:
        S += Token(t)
    RS   = Token(tpl["saida"]["reacao"])
    CS   = Token(tpl["saida"]["contexto"])
    EXDS = Token(tpl["saida"]["explicacao"])[:3]
    IME  = Token(tpl["saida"]["imersao"])[:3]

    total_ent = len(E)+len(RE)+len(CE)+len(PIDE)
    total_out = len(S)+len(RS)+len(CS)+len(EXDS)+len(IME)
    markers   = generate_markers(last, total_ent+total_out)
    ent_marks = markers[:total_ent]
    out_marks = markers[total_ent:]
    fim_ent   = ent_marks[-1] if ent_marks else last
    fim_out   = out_marks[-1] if out_marks else fim_ent

    # subdividir marcadores
    idx    = 0
    E_m    = ent_marks[idx: idx+len(E)];    idx+=len(E)
    RE_m   = ent_marks[idx: idx+len(RE)];   idx+=len(RE)
    CE_m   = ent_marks[idx: idx+len(CE)];   idx+=len(CE)
    PIDE_m = ent_marks[idx: idx+len(PIDE)]
    jdx    = 0
    S_m    = out_marks[jdx: jdx+len(S)];    jdx+=len(S)
    RS_m   = out_marks[jdx: jdx+len(RS)];   jdx+=len(RS)
    CS_m   = out_marks[jdx: jdx+len(CS)];   jdx+=len(CS)
    EXDS_m = out_marks[jdx: jdx+len(EXDS)]; jdx+=len(EXDS)
    IME_m  = out_marks[jdx: jdx+len(IME)]

    next_id = max((b["bloco_id"] for b in universo["blocos"]), default=0) + 1
    new_block = {
        "bloco_id": next_id,
        "entrada": {
            **tpl["entrada"],
            "tokens": {
                "E":    E_m, "RE":   RE_m,
                "CE":   CE_m, "PIDE": PIDE_m,
                "TOTAL": ent_marks
            },
            "fim": fim_ent, "alnulu": alnulu_val
        },
        "saidas": [{
            **tpl["saida"],
            "tokens": {
                "S":     S_m,  "RS":    RS_m,
                "CS":    CS_m, "EXDS":  EXDS_m,
                "IME":   IME_m,"TOTAL": out_marks
            },
            "fim": fim_out
        }],
        "open": True
    }

    universo["blocos"].append(new_block)
    universo["ultimo_child"] = fim_out
    sequencia = [b["bloco_id"] for b in universo["blocos"]]
    universo.setdefault("cb", {}).setdefault("sdb", []).append({
        "sequencia": sequencia, "VA": False
    })

    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {base_file} atualizado. Último marker: {fim_out}  ALNULU={alnulu_val}")

if __name__ == "__main__":
    main()
