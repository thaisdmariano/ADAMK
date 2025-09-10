#!/usr/bin/env python3
# coding: utf-8
r"""
insepa_update.py — CLI completo para INSEPA com CRUD de Universos e Blocos,
agora com confirmações antes de renomear universos ou criar blocos não existentes.
"""

import sys, json, re
from typing import List, Dict, Any, Optional

# —————————————————————————————————————
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
    """Checksum ALNULU: cada caractere contribui por mapa e equiv."""
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
        "indice_mae": None, "nome": "", "bloco_id": None,
        "entrada": {"texto":"", "reacao":"", "contexto":"", "pensamento_interno":""},
        "saida":   {"textos":[], "reacao":"", "contexto":"", "explicacao":"", "imersao":""}
    }
    section: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if m := re.match(r'^Índice mãe:\s*(\d+)', line):
            tpl["indice_mae"] = int(m.group(1))
            continue
        if m := re.match(r'^Nome:\s*(.+)', line):
            tpl["nome"] = m.group(1).strip()
            continue
        if m := re.match(r'^🧱\s*Bloco\s*(\d+)', line):
            tpl["bloco_id"] = int(m.group(1))
            continue
        if m := re.match(r'^Entrada:\s*(.+)', line):
            tpl["entrada"]["texto"] = m.group(1).strip()
            section = "entrada"
            continue
        if m := re.match(r'^[Rr]eação:\s*(.+)', line):
            target = tpl["entrada"] if section=="entrada" else tpl["saida"]
            target["reacao"] = m.group(1).strip()
            continue
        if m := re.match(r'^[Cc]ontexto:\s*(.+)', line):
            if section=="entrada":
                tpl["entrada"]["contexto"] = m.group(1).strip()
            else:
                tpl["saida"]["contexto"] = m.group(1).strip()
            continue
        if m := re.match(r'^[Pp]ensamento\s+[Ii]nterno:\s*(.+)', line):
            tpl["entrada"]["pensamento_interno"] = m.group(1).strip()
            continue
        if line.startswith("Saída:"):
            section = "saida_textos"
            continue
        if section=="saida_textos" and (m := re.match(r'^\d+\.\s*(.+)', line)):
            tpl["saida"]["textos"].append(m.group(1).strip())
            continue
        if section=="saida_textos":
            section = "saida_meta"
        if section=="saida_meta":
            if m := re.match(r'^[Ee]xplica[cç][ãa]o:\s*(.+)', line):
                tpl["saida"]["explicacao"] = m.group(1).strip()
                continue
            if m := re.match(r'^[Ii]mers[ãa]o:\s*(.+)', line):
                tpl["saida"]["imersao"] = m.group(1).strip()
                continue
    if tpl["indice_mae"] is None:
        sys.exit("Erro: 'Índice mãe' não encontrado no template.")
    return tpl  # type: ignore

def save_json(base: Dict[str,Any], path: str="adam_memoria.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

# —————————————————————————————————————
def show_summary(base: Dict[str,Any]) -> None:
    print("\n=== Universos (IM) Existentes ===")
    ims = base.get("IM", {})
    if not ims:
        print("  (nenhum universo cadastrado)")
    for mom, uni in ims.items():
        print(f"  IM '{mom}': nome='{uni.get('nome','')}'")

def create_universe(base: Dict[str,Any]) -> None:
    mom = input("Novo Índice Mãe: ").strip()
    if not mom.isdigit():
        print("Índice inválido."); return
    if mom in base.get("IM", {}):
        print(f"IM '{mom}' já existe. Abortando.")
        return
    nome = input("Nome do Universo: ").strip()
    confirm = input(f"Criar IM {mom} com nome '{nome}'? (S/N): ").strip().lower()
    if confirm != "s":
        print("Cancelado."); return
    base["IM"][mom] = {
        "nome": nome,
        "ultimo_child": f"{mom}.0",
        "blocos": [],
        "cb": {"status":"Indisponível","sdb":[]}
    }
    save_json(base)
    print(f"IM '{mom}' criado com sucesso.")

def rename_universe(base: Dict[str,Any]) -> None:
    mom = input("Índice Mãe a renomear: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print(f"IM '{mom}' não encontrado."); return
    atual = uni.get("nome","")
    novo = input(f"Nome atual '{atual}'. Novo nome: ").strip()
    confirm = input(f"Alterar para '{novo}'? (S/N): ").strip().lower()
    if confirm != "s":
        print("Cancelado."); return
    uni["nome"] = novo
    save_json(base)
    print("Renomeado com sucesso.")

def show_info(base: Dict[str,Any]) -> None:
    mom = input("Digite Índice Mãe: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print(f"IM '{mom}' não existe."); return
    print(f"\nÍndice mãe: {mom}\nNome: {uni['nome']}")
    for blk in uni.get("blocos", []):
        e, s0 = blk["entrada"], blk["saidas"][0]
        print(f"\n🧱 Bloco {blk['bloco_id']}")
        print(f"Entrada: {e['texto']}\nReação: {e['reacao']}\nContexto: {e['contexto']}")
        print(f"Pensamento interno: \"{e['pensamento_interno']}\"")
        print("Saída:")
        for i, t in enumerate(s0["textos"], 1):
            print(f"  {i}. {t}")
        print(f"Reação: {s0['reacao']}\nContexto: {s0['contexto']}")
        print(f"Explicação: {s0['explicacao']}\nImersão: {s0['imersao']}")
    input("\nEnter para continuar...")

def list_blocks(uni: Dict[str,Any]) -> None:
    if not uni["blocos"]:
        print("  (nenhum bloco cadastrado)")
        return
    for blk in uni["blocos"]:
        snippet = blk["entrada"]["texto"][:30].replace("\n"," ")
        print(f"  Bloco {blk['bloco_id']}: \"{snippet}...\"")

def create_or_update_block(base: Dict[str,Any], tpl: Dict[str,Any]) -> bool:
    mom = str(tpl["indice_mae"])
    uni = base["IM"].setdefault(
        mom,
        {"nome": tpl["nome"] or mom, "ultimo_child":f"{mom}.0", "blocos":[], "cb":{"status":"Indisponível","sdb":[]}}
    )

    # confirmar renomeação se o nome no template difere
    if tpl["nome"] and tpl["nome"] != uni["nome"]:
        ans = input(
            f"Nome no template '{tpl['nome']}' difere de '{uni['nome']}'. "
            "Deseja renomear? (S/N): "
        ).strip().lower()
        if ans == "s":
            uni["nome"] = tpl["nome"]
            print("Universo renomeado.")
        else:
            print("Nome do universo mantido.")

    last = uni["ultimo_child"]
    aln = calcular_alnulu(tpl["entrada"]["texto"])

    # tokenização
    E    = Token(tpl["entrada"]["texto"])
    RE   = Token(tpl["entrada"]["reacao"])
    CE   = Token(tpl["entrada"]["contexto"])
    PIDE = Token(tpl["entrada"]["pensamento_interno"])[:3]
    S = []
    for t in tpl["saida"]["textos"]:
        S += Token(t)
    RS   = Token(tpl["saida"]["reacao"])
    CS   = Token(tpl["saida"]["contexto"])
    EXDS = Token(tpl["saida"]["explicacao"])[:3]
    IME  = Token(tpl["saida"]["imersao"])[:3]

    te = len(E) + len(RE) + len(CE) + len(PIDE)
    to = len(S) + len(RS) + len(CS) + len(EXDS) + len(IME)
    marks = generate_markers(last, te + to)
    ent, out = marks[:te], marks[te:]
    fim_ent = ent[-1] if ent else last
    fim_out = out[-1] if out else fim_ent

    # subdividir marcadores
    idx = 0
    Em = ent[idx:idx+len(E)]; idx += len(E)
    REm = ent[idx:idx+len(RE)]; idx += len(RE)
    CEm = ent[idx:idx+len(CE)]; idx += len(CE)
    PIm = ent[idx:idx+len(PIDE)]
    jdx = 0
    Sm = out[jdx:jdx+len(S)]; jdx += len(S)
    RSm = out[jdx:jdx+len(RS)]; jdx += len(RS)
    CSm = out[jdx:jdx+len(CS)]; jdx += len(CS)
    EXm = out[jdx:jdx+len(EXDS)]; jdx += len(EXDS)
    IMm = out[jdx:jdx+len(IME)]

    bloco = {
        "entrada": {
            **tpl["entrada"],
            "tokens": {"E":Em, "RE":REm, "CE":CEm, "PIDE":PIm, "TOTAL":ent},
            "fim": fim_ent, "alnulu": aln
        },
        "saidas": [{
            **tpl["saida"],
            "tokens": {"S":Sm, "RS":RSm, "CS":CSm, "EXDS":EXm, "IME":IMm, "TOTAL":out},
            "fim": fim_out
        }],
        "open": True
    }

    # UPDATE se bloco_id fornecido
    if tpl.get("bloco_id") is not None:
        bid = tpl["bloco_id"]
        for i, b in enumerate(uni["blocos"]):
            if b["bloco_id"] == bid:
                bloco["bloco_id"] = bid
                uni["blocos"][i] = bloco
                print(f"Bloco {bid} atualizado.")
                uni["ultimo_child"] = fim_out
                seq = [blk["bloco_id"] for blk in uni["blocos"]]
                uni.setdefault("cb", {}).setdefault("sdb", []).append({"sequencia": seq, "VA": False})
                return True
        # não encontrou → confirmar criação
        ans = input(f"Bloco {bid} não encontrado. Criar novo bloco com ID {bid}? (S/N): ").strip().lower()
        if ans != "s":
            print("Operação de bloco cancelada.")
            return False
        bloco["bloco_id"] = bid
        uni["blocos"].append(bloco)
        print(f"Bloco {bid} criado.")
        uni["ultimo_child"] = fim_out
        seq = [blk["bloco_id"] for blk in uni["blocos"]]
        uni.setdefault("cb", {}).setdefault("sdb", []).append({"sequencia": seq, "VA": False})
        return True

    # CREATE sem bloco_id
    ans = input("Criar novo bloco? (S/N): ").strip().lower()
    if ans != "s":
        print("Operação de bloco cancelada.")
        return False
    bid = max((b["bloco_id"] for b in uni["blocos"]), default=0) + 1
    bloco["bloco_id"] = bid
    uni["blocos"].append(bloco)
    print(f"Bloco {bid} criado.")
    uni["ultimo_child"] = fim_out
    seq = [blk["bloco_id"] for blk in uni["blocos"]]
    uni.setdefault("cb", {}).setdefault("sdb", []).append({"sequencia": seq, "VA": False})
    return True

def delete_block(base: Dict[str,Any], mom: str) -> None:
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print(f"IM '{mom}' não existe."); return
    bid = input("Número do bloco p/ deletar: ").strip()
    if not bid.isdigit():
        print("ID inválido."); return
    bid = int(bid)
    if input(f"Confirmar exclusão do bloco {bid}? (S/N): ").strip().lower() != "s":
        print("Cancelado."); return
    before = len(uni["blocos"])
    uni["blocos"] = [b for b in uni["blocos"] if b["bloco_id"] != bid]
    if len(uni["blocos"]) < before:
        print(f"Bloco {bid} excluído.")
    else:
        print(f"Bloco {bid} não encontrado.")

def manage_blocks(base: Dict[str,Any]) -> None:
    mom = input("Índice Mãe para gerenciar: ").strip()
    uni = base.get("IM", {}).get(mom)
    if not uni:
        print(f"IM '{mom}' não encontrado."); return

    while True:
        print(f"\n=== Gerenciar Blocos de IM '{mom}' ===")
        print(" 1) Listar Blocos\n 2) Criar/Atualizar Bloco\n 3) Excluir Bloco\n 4) Voltar")
        opt = input("Opção [1-4]: ").strip()
        if opt == "1":
            list_blocks(uni)
            input("Enter para continuar...")
        elif opt == "2":
            print("\nCole template INSEPA (use '🧱 Bloco N' para editar):")
            lines: List[str] = []
            while True:
                ln = input()
                if not ln:
                    break
                lines.append(ln)
            tpl = parse_template(lines)
            if str(tpl["indice_mae"]) != mom:
                print("Índice mãe no template não coincide. Abortando.")
                continue
            if create_or_update_block(base, tpl):
                save_json(base)
        elif opt == "3":
            delete_block(base, mom)
            save_json(base)
        elif opt == "4":
            break
        else:
            print("Opção inválida.")

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

    while True:
        show_summary(base)
        print("\n1) Criar Universo\n2) Renomear Universo\n3) Info Universo\n4) Gerenciar Blocos\n5) Sair")
        opt = input("Escolha [1-5]: ").strip()
        if opt == "1":
            create_universe(base)
        elif opt == "2":
            rename_universe(base)
        elif opt == "3":
            show_info(base)
        elif opt == "4":
            manage_blocks(base)
        elif opt == "5":
            print("Encerrando.")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()

