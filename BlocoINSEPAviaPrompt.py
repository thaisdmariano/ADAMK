#!/usr/bin/env python3
# coding: utf-8
r"""
insepa_update.py — Cole o bloco template INSEPA e atualize
adam_memoria.json em um único passo, respeitando INSEPA sem warnings.

Fluxo:
 1. python insepa_update.py
 2. Cole (Ctrl+V) seu bloco template INSEPA completo
 3. Pressione Enter em branco para finalizar
 4. O script cria o IM se não existir, gera marcadores mom.child,
    anexa o bloco, atualiza cb.sdb e salva em adam_memoria.json
"""

import sys
import json
import re
from typing import List, Dict, Any, Optional

def Token(text: str) -> List[str]:
    r"""
    INSEPA tokenização:
      aceita palavras, pontuação, emojis, stopwords, tudo.
    Regex usada: r'\w+|[^\w\s]'
    """
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def next_marker(prev: str) -> str:
    """Incrementa sem arredondar: 0.99 → 0.100"""
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

def parse_template(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "indice_mae": None,
        "nome": "",
        "entrada": {
            "texto": "",
            "reacao": "",
            "contexto": "",
            "pensamento_interno": ""
        },
        "saida": {
            "textos": [],
            "reacao": "",
            "contexto": "",
            "explicacao": "",
            "imersao": ""
        }
    }
    section: Optional[str] = None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue

        # Índice mãe
        if m := re.match(r'^Índice mãe:\s*(\d+)', line):
            tpl["indice_mae"] = int(m.group(1))
            continue

        # Nome
        if m := re.match(r'^Nome:\s*(.+)', line):
            tpl["nome"] = m.group(1).strip()
            continue

        # Entrada inline
        if m := re.match(r'^Entrada:\s*(.+)', line):
            tpl["entrada"]["texto"] = m.group(1).strip()
            section = "entrada"
            continue

        # Seções
        if line.startswith("Entrada:"):
            section = "entrada"
            continue
        if line.startswith("Saída:"):
            section = "saida_textos"
            continue

        # Reação (entrada ou saída)
        if m := re.match(r'^[Rr]eação:\s*(.+)', line):
            if section == "entrada":
                tpl["entrada"]["reacao"] = m.group(1).strip()
            else:
                tpl["saida"]["reacao"] = m.group(1).strip()
            continue

        # Detalhes de entrada
        if section == "entrada":
            if m := re.match(r'^[Tt]exto:\s*(.+)', line):
                tpl["entrada"]["texto"] = m.group(1).strip()
            elif m := re.match(r'^[Cc]ontexto:\s*(.+)', line):
                tpl["entrada"]["contexto"] = m.group(1).strip()
            elif m := re.match(r'^[Pp]ensamento\s+[Ii]nterno:\s*(.+)', line):
                tpl["entrada"]["pensamento_interno"] = m.group(1).strip()
            continue

        # Textos de saída numerados
        if section == "saida_textos":
            if m := re.match(r'^\d+\.\s*(.+)', line):
                tpl["saida"]["textos"].append(m.group(1).strip())
                continue
            section = "saida_meta"

        # Meta-saída
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

    print("Cole seu bloco template INSEPA e pressione Enter em branco para finalizar:")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)

    tpl = parse_template(lines)
    mom = str(tpl["indice_mae"])
    universo = base["IM"].get(mom)

    # Criar IM se não existir (sem sobrescrever!)
    if universo is None:
        universo = {
            "nome": tpl["nome"] or f"IM_{mom}",
            "ultimo_child": f"{mom}.0",
            "blocos": [],
            "cb": {"status": "Indisponível", "sdb": []}
        }
        base["IM"][mom] = universo

    last = universo["ultimo_child"]

    # Tokenização de entrada
    E    = Token(tpl["entrada"]["texto"])
    RE   = Token(tpl["entrada"]["reacao"])
    CE   = Token(tpl["entrada"]["contexto"])
    PIDE = Token(tpl["entrada"]["pensamento_interno"])[:3]

    # Tokenização de saída
    S: List[str] = []
    for t in tpl["saida"]["textos"]:
        S += Token(t)
    RS   = Token(tpl["saida"]["reacao"])
    CS   = Token(tpl["saida"]["contexto"])
    EXDS = Token(tpl["saida"]["explicacao"])[:3]
    IME  = Token(tpl["saida"]["imersao"])[:3]

    total_ent = len(E) + len(RE) + len(CE) + len(PIDE)
    total_out = len(S) + len(RS) + len(CS) + len(EXDS) + len(IME)

    markers    = generate_markers(last, total_ent + total_out)
    ent_marks  = markers[:total_ent]
    out_marks  = markers[total_ent:]

    fim_ent = ent_marks[-1] if ent_marks else last
    fim_out = out_marks[-1] if out_marks else fim_ent

    # Subdivide marcadores de entrada
    idx    = 0
    E_m    = ent_marks[idx: idx+len(E)];    idx += len(E)
    RE_m   = ent_marks[idx: idx+len(RE)];   idx += len(RE)
    CE_m   = ent_marks[idx: idx+len(CE)];   idx += len(CE)
    PIDE_m = ent_marks[idx: idx+len(PIDE)]

    # Subdivide marcadores de saída
    jdx    = 0
    S_m    = out_marks[jdx: jdx+len(S)];    jdx += len(S)
    RS_m   = out_marks[jdx: jdx+len(RS)];   jdx += len(RS)
    CS_m   = out_marks[jdx: jdx+len(CS)];   jdx += len(CS)
    EXDS_m = out_marks[jdx: jdx+len(EXDS)]; jdx += len(EXDS)
    IME_m  = out_marks[jdx: jdx+len(IME)]

    next_id = max((b["bloco_id"] for b in universo["blocos"]), default=0) + 1

    new_block: Dict[str, Any] = {
        "bloco_id": next_id,
        "entrada": {
            **tpl["entrada"],
            "tokens": {
                "E":     E_m,
                "RE":    RE_m,
                "CE":    CE_m,
                "PIDE":  PIDE_m,
                "TOTAL": ent_marks
            },
            "fim":    fim_ent,
            "alnulu": tpl["entrada"].get("alnulu", 100)
        },
        "saidas": [{
            **tpl["saida"],
            "tokens": {
                "S":     S_m,
                "RS":    RS_m,
                "CS":    CS_m,
                "EXDS":  EXDS_m,
                "IME":   IME_m,
                "TOTAL": out_marks
            },
            "fim":    fim_out
        }],
        "open": True
    }

    universo["blocos"].append(new_block)
    universo["ultimo_child"] = fim_out

    # Atualiza cb.sdb
    sequencia = [b["bloco_id"] for b in universo["blocos"]]
    universo.setdefault("cb", {}).setdefault("sdb", []).append({
        "sequencia": sequencia,
        "VA": False
    })

    # Salva de volta
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {base_file} atualizado. Último marker: {fim_out}")

if __name__ == "__main__":
    main()
