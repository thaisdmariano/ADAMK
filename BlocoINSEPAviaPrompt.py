#!/usr/bin/env python3
# coding: utf-8
r"""
insepa_update.py — CLI em português para INSEPA com:
  • CRUD de Universos (criar, renomear, excluir)
  • CRUD de Blocos (listar, criar/atualizar, excluir)
  • Criação em lote de blocos pelo cabeçalho “Bloco:” ou “🧱”
  • Geração de tokens na ordem EXDS → IME → S → RS → CS
"""

import sys
import json
import re
from typing import List, Dict, Any, Optional

def Token(text: str) -> List[str]:
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

def parse_block(lines: List[str]) -> Dict[str, Any]:
    tpl = {
        "bloco_id": None,
        "entrada": {"texto":"", "reacao":"", "contexto":"", "pensamento_interno":""},
        "saida":   {"textos":[], "reacao":"", "contexto":"", "explicacao":"", "imersao":""}
    }
    # quebra linhas que tiveram vários campos na mesma linha
    expanded: List[str] = []
    for raw in lines:
        tmp = re.sub(
            r'(Bloco:|🧱|Entrada:|Reação:|Contexto:|Pensamento Interno:|Explicação:|Imersão:|Saída:|\d+\.)',
            r'\n\1',
            raw
        )
        expanded += [l.strip() for l in tmp.split('\n') if l.strip()]

    section: Optional[str] = None
    for line in expanded:
        if re.match(r'^(?:Bloco:|🧱)\s*$', line):
            continue
        if m := re.match(r'^[Ee]xplica[cç][ãa]o:\s*(.+)', line):
            tpl["saida"]["explicacao"] = m.group(1).strip(); continue
        if m := re.match(r'^[Ii]mers[ãa]o:\s*(.+)', line):
            tpl["saida"]["imersao"] = m.group(1).strip(); continue
        if m := re.match(r'^Entrada:\s*(.+)', line):
            tpl["entrada"]["texto"] = m.group(1).strip(); section = "entrada"; continue
        if section == "entrada" and (m := re.match(r'^[Rr]eação:\s*(.+)', line)):
            tpl["entrada"]["reacao"] = m.group(1).strip(); continue
        if section == "entrada" and (m := re.match(r'^[Cc]ontexto:\s*(.+)', line)):
            tpl["entrada"]["contexto"] = m.group(1).strip(); continue
        if m := re.match(r'^[Pp]ensamento\s+[Ii]nterno:\s*(.+)', line):
            tpl["entrada"]["pensamento_interno"] = m.group(1).strip(); continue
        if re.match(r'^Saída:\s*$', line):
            section = "saida_textos"; continue
        if section == "saida_textos" and (m := re.match(r'^\d+\.\s*(.+)', line)):
            tpl["saida"]["textos"].append(m.group(1).strip()); continue
        if section == "saida_textos" and re.match(r'^[Rr]eação:', line):
            section = "saida_meta"
        if section == "saida_meta" and (m := re.match(r'^[Rr]eação:\s*(.+)', line)):
            tpl["saida"]["reacao"] = m.group(1).strip(); continue
        if section == "saida_meta" and (m := re.match(r'^[Cc]ontexto:\s*(.+)', line)):
            tpl["saida"]["contexto"] = m.group(1).strip(); continue

    if not tpl["entrada"]["texto"]:
        sys.exit("Erro: cada bloco deve ter ao menos 'Entrada'.")
    return tpl

def save_json(base: Dict[str, Any], path: str="adam_memoria.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

# ——— Gerenciar Universos —————————————————————————————————

def show_summary(base: Dict[str, Any]) -> None:
    ims = base.get("IM", {})
    if not ims:
        print("  (nenhum universo cadastrado)")
        return
    for mom, uni in ims.items():
        print(f"  IM '{mom}': nome='{uni.get('nome','')}'")

def create_universe(base: Dict[str, Any]) -> None:
    mom = input("Novo Índice Mãe: ").strip()
    if not mom.isdigit() or mom in base.get("IM", {}):
        print("Índice inválido ou já existe."); return
    nome = input("Nome do Universo: ").strip()
    if input(f"Criar IM {mom} '{nome}'? (S/N): ").lower() != "s":
        print("Cancelado."); return
    base["IM"][mom] = {
        "nome": nome,
        "ultimo_child": f"{mom}.0",
        "blocos": [], "cb": {"status":"Indisponível","sdb":[]}
    }
    save_json(base)
    print(f"IM '{mom}' criado com sucesso.")

def rename_universe(base: Dict[str, Any]) -> None:
    mom = input("Índice Mãe a renomear: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print("Universo não encontrado."); return
    novo = input(f"Novo nome para '{uni['nome']}': ").strip()
    if input("Confirmar? (S/N): ").lower() != "s":
        print("Cancelado."); return
    uni["nome"] = novo
    save_json(base)
    print("Renomeado com sucesso.")

def delete_universe(base: Dict[str, Any]) -> None:
    mom = input("Índice Mãe para excluir: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print("Universo não encontrado."); return
    if input(f"Excluir IM {mom} '{uni['nome']}'? (S/N): ").lower() != "s":
        print("Cancelado."); return
    del base["IM"][mom]
    save_json(base)
    print("Universo excluído com sucesso.")

def manage_universes(base: Dict[str, Any]) -> None:
    while True:
        print("\n=== Gerenciar Universos ===")
        print("1) Listar Universos")
        print("2) Criar Universo")
        print("3) Renomear Universo")
        print("4) Excluir Universo")
        print("5) Voltar")
        opt = input("Escolha [1-5]: ").strip()
        if opt == "1":
            show_summary(base)
        elif opt == "2":
            create_universe(base)
        elif opt == "3":
            rename_universe(base)
        elif opt == "4":
            delete_universe(base)
        elif opt == "5":
            break
        else:
            print("Opção inválida.")

# ——— Gerenciar Blocos —————————————————————————————————

def list_blocks(uni: Dict[str, Any]) -> None:
    if not uni["blocos"]:
        print("  (nenhum bloco cadastrado)")
        return
    for b in uni["blocos"]:
        snippet = b["entrada"]["texto"][:30].replace("\n", " ")
        print(f"  Bloco {b['bloco_id']}: \"{snippet}...\"")

def delete_block(base: Dict[str, Any], mom: str) -> None:
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print("Universo não encontrado."); return
    bid = input("Número do bloco p/ deletar: ").strip()
    if not bid.isdigit():
        print("ID inválido."); return
    if input(f"Excluir bloco {bid}? (S/N): ").lower() != "s":
        print("Cancelado."); return
    before = len(uni["blocos"])
    uni["blocos"] = [b for b in uni["blocos"] if b["bloco_id"] != int(bid)]
    save_json(base)
    print("Bloco excluído." if len(uni["blocos"]) < before else "Bloco não encontrado.")

def process_block(base: Dict[str, Any], mom: str, tpl: Dict[str, Any]) -> None:
    uni = base["IM"].setdefault(mom, {
        "nome": mom,
        "ultimo_child": f"{mom}.0",
        "blocos": [], "cb": {"status":"Indisponível","sdb":[]}
    })
    last = uni["ultimo_child"]
    aln = calcular_alnulu(tpl["entrada"]["texto"])

    E    = Token(tpl["entrada"]["texto"])
    RE   = Token(tpl["entrada"]["reacao"])
    CE   = Token(tpl["entrada"]["contexto"])
    PIDE = Token(tpl["entrada"]["pensamento_interno"])[:3]

    EXDS = Token(tpl["saida"]["explicacao"])[:3]
    IME  = Token(tpl["saida"]["imersao"])[:3]

    S_list = []
    for t in tpl["saida"]["textos"]:
        S_list += Token(t)

    RS   = Token(tpl["saida"]["reacao"])
    CS   = Token(tpl["saida"]["contexto"])

    te = len(E) + len(RE) + len(CE) + len(PIDE)
    to = len(EXDS) + len(IME) + len(S_list) + len(RS) + len(CS)

    marks = generate_markers(last, te + to)
    ent, out = marks[:te], marks[te:]
    fim_ent = ent[-1] if ent else last
    fim_out = out[-1] if out else fim_ent

    idx = 0
    Em  = ent[idx: idx+len(E)];     idx += len(E)
    REm = ent[idx: idx+len(RE)];    idx += len(RE)
    CEm = ent[idx: idx+len(CE)];    idx += len(CE)
    PIm = ent[idx: idx+len(PIDE)]

    j    = 0
    EXm  = out[j: j+len(EXDS)];    j += len(EXDS)
    IMm  = out[j: j+len(IME)];     j += len(IME)
    Sm   = out[j: j+len(S_list)];  j += len(S_list)
    RSm  = out[j: j+len(RS)];      j += len(RS)
    CSm  = out[j: j+len(CS)]

    uni["ultimo_child"] = fim_out

    bloco = {
        **tpl,
        "entrada": {
            **tpl["entrada"],
            "tokens": {"E":Em,"RE":REm,"CE":CEm,"PIDE":PIm,"TOTAL":ent},
            "fim": fim_ent, "alnulu": aln
        },
        "saidas": [{
            **tpl["saida"],
            "tokens": {"EXDS":EXm,"IME":IMm,"S":Sm,"RS":RSm,"CS":CSm,"TOTAL":out},
            "fim": fim_out
        }],
        "open": True
    }

    if tpl.get("bloco_id") is not None:
        bid = tpl["bloco_id"]
        for i,b in enumerate(uni["blocos"]):
            if b["bloco_id"] == bid:
                bloco["bloco_id"] = bid
                uni["blocos"][i] = bloco
                print(f"Bloco {bid} atualizado.")
                break
        else:
            bloco["bloco_id"] = bid
            uni["blocos"].append(bloco)
            print(f"Bloco {bid} criado.")
    else:
        bid = max((b["bloco_id"] for b in uni["blocos"]), default=0) + 1
        bloco["bloco_id"] = bid
        uni["blocos"].append(bloco)
        print(f"Bloco {bid} criado.")

    seq = [b["bloco_id"] for b in uni["blocos"]]
    uni.setdefault("cb", {}).setdefault("sdb", []).append({"sequencia": seq, "VA": False})

def manage_blocks(base: Dict[str, Any]) -> None:
    mom = input("Índice Mãe para gerenciar blocos: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print("Universo não encontrado."); return

    while True:
        print(f"\n=== Gerenciar Blocos de IM '{mom}' ===")
        print("1) Listar Blocos")
        print("2) Criar/Atualizar Bloco(s)")
        print("3) Excluir Bloco")
        print("4) Voltar")
        opt = input("Escolha [1-4]: ").strip()

        if opt == "1":
            list_blocks(uni)

        elif opt == "2":
            multi = input("Criar vários blocos? (S/N): ").strip().lower()
            if multi == "s":
                print("Cole os blocos (cada um começando em 'Bloco:' ou '🧱') e termine com linha em branco:")
                buf: List[str] = []
                blank = False
                while True:
                    ln = input()
                    if ln == "":
                        if blank:
                            break
                        blank = True
                        buf.append("")
                    else:
                        blank = False
                        buf.append(ln)

                groups: List[List[str]] = []
                cur: List[str] = []
                for ln in buf:
                    if re.match(r'^(?:Bloco:|🧱)', ln):
                        if cur:
                            groups.append(cur)
                            cur = []
                        cur.append(ln)
                    elif ln.strip() == "":
                        if cur:
                            groups.append(cur)
                            cur = []
                    else:
                        cur.append(ln)
                if cur:
                    groups.append(cur)

                created = 0
                for grp in groups:
                    tpl = parse_block(grp)
                    process_block(base, mom, tpl)
                    created += 1
                save_json(base)
                print(f"\n✅ Lote concluído: {created}/{len(groups)} blocos processados.")
                input("Enter para voltar ao menu de blocos...")
                return

            else:
                print("Cole um bloco e termine com linha em branco:")
                lines: List[str] = []
                while True:
                    ln = input().rstrip()
                    if not ln:
                        break
                    lines.append(ln)
                tpl = parse_block(lines)
                process_block(base, mom, tpl)
                save_json(base)
                input("Enter para voltar ao menu de blocos...")
                return

        elif opt == "3":
            delete_block(base, mom)

        elif opt == "4":
            break

        else:
            print("Opção inválida.")

def main():
    try:
        with open("adam_memoria.json", encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = {"IM": {}}
    except Exception as e:
        print(f"Erro ao abrir JSON: {e}")
        sys.exit(1)

    while True:
        print("\n=== Menu Principal ===")
        print("1) Gerenciar Universos")
        print("2) Gerenciar Blocos")
        print("3) Sair")
        opt = input("Escolha [1-3]: ").strip()
        if opt == "1":
            manage_universes(base)
        elif opt == "2":
            manage_blocks(base)
        elif opt == "3":
            print("Encerrando.")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()


