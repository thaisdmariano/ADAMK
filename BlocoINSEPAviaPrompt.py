#!/usr/bin/env python3
# coding: utf-8
"""
insepa_update.py — Cole o bloco template INSEPA e atualize
adam_memoria.json em um único passo, sem sobrescrever IMs existentes.

Fluxo:
 1. Execute: python insepa_update.py
 2. Cole (Ctrl+V) seu bloco template INSEPA completo
 3. Pressione Enter em branco para finalizar a leitura
 4. O script cria o IM se não existir, gera marcadores
    mom.child, anexa o bloco, atualiza cb.sdb e salva em adam_memoria.json
"""

import sys
import json
import re
from typing import List, Dict, Any, Optional

def Token(text: str) -> List[str]:
    """INSEPA tokenização: mantém palavras, pontuação, emojis, stopwords."""
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
        "entrada": {"texto":"", "reacao":"", "contexto":"", "pensamento_interno":""},
        "saida":   {"textos":[], "reacao":"", "contexto":"", "explicacao":"", "imersao":""}
    }
    section: Optional[str] = None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue

        # Índice mãe e Nome
        if line.startswith("Índice mãe:"):
            tpl["indice_mae"] = int(line.split(":",1)[1].strip())
            continue
        if line.startswith("Nome:"):
            tpl["nome"] = line.split(":",1)[1].strip()
            continue

        # Entrada inline
        m_ent = re.match(r'^Entrada:\s*(.+)', line)
        if m_ent:
            tpl["entrada"]["texto"] = m_ent.group(1).strip()
            section = "entrada"
            continue

        # Seções
        if line.startswith("Entrada:"):
            section = "entrada"
            continue
        if line.startswith("Saída:"):
            section = "saida_textos"
            continue

        # Campos de entrada
        if section == "entrada":
            if line.startswith("Texto:"):
                tpl["entrada"]["texto"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("reação:"):
                tpl["entrada"]["reacao"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("contexto:"):
                tpl["entrada"]["contexto"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("pensamento interno:"):
                tpl["entrada"]["pensamento_interno"] = line.split(":",1)[1].strip()

        # Linhas de saída
        elif section == "saida_textos":
            m = re.match(r'^\d+\.\s*(.+)', line)
            if m:
                tpl["saida"]["textos"].append(m.group(1).strip())
            else:
                section = "saida_meta"
                # continua para processar meta nesta mesma linha

        # Campos de meta-saída
        elif section == "saida_meta":
            if line.lower().startswith("reação:"):
                tpl["saida"]["reacao"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("contexto:"):
                tpl["saida"]["contexto"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("explicação:"):
                tpl["saida"]["explicacao"] = line.split(":",1)[1].strip()
            elif line.lower().startswith("imersão:"):
                tpl["saida"]["imersao"] = line.split(":",1)[1].strip()

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

    # Cria IM se não existir (não sobrescrever!)
    if universo is None:
        universo = {
            "nome": tpl["nome"] or f"IM_{mom}",
            "ultimo_child": f"{mom}.0",
            "blocos": [],
            "cb": {"status": "Indisponível", "sdb": []}
        }
        base["IM"][mom] = universo

    last = universo["ultimo_child"]

    # Tokenização de subcampos de entrada
    E    = Token(tpl["entrada"]["texto"])
    RE   = Token(tpl["entrada"]["reacao"])
    CE   = Token(tpl["entrada"]["contexto"])
    PIDE = Token(tpl["entrada"]["pensamento_interno"])[:3]

    # Tokenização de subcampos de saída
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
    idx  = 0
    E_m   = ent_marks[idx: idx+len(E)];    idx += len(E)
    RE_m  = ent_marks[idx: idx+len(RE)];   idx += len(RE)
    CE_m  = ent_marks[idx: idx+len(CE)];   idx += len(CE)
    PIDE_m= ent_marks[idx: idx+len(PIDE)]

    # Subdivide marcadores de saída
    jdx  = 0
    S_m   = out_marks[jdx: jdx+len(S)];    jdx += len(S)
    RS_m  = out_marks[jdx: jdx+len(RS)];   jdx += len(RS)
    CS_m  = out_marks[jdx: jdx+len(CS)];   jdx += len(CS)
    EXDS_m= out_marks[jdx: jdx+len(EXDS)]; jdx += len(EXDS)
    IME_m = out_marks[jdx: jdx+len(IME)]

    next_id = max((b["bloco_id"] for b in universo["blocos"]), default=0) + 1

    new_block: Dict[str, Any] = {
        "bloco_id": next_id,
        "entrada": {
            **tpl["entrada"],
            "tokens": {
                "E":    E_m,
                "RE":   RE_m,
                "CE":   CE_m,
                "PIDE": PIDE_m,
                "TOTAL": ent_marks
            },
            "fim": fim_ent,
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
    sequencia_atual = [b["bloco_id"] for b in universo["blocos"]]
    universo.setdefault("cb", {}).setdefault("sdb", []).append({
        "sequencia": sequencia_atual,
        "VA": False
    })

    # Salva de volta
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {base_file} atualizado. Último marker: {fim_out}")

if __name__ == "__main__":
    main()