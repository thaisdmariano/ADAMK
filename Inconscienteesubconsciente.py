#!/usr/bin/env python3
# subconscious_manager.py

import json
import os
import re
import sys
import readline

SUB_FILE = "subconsciente.json"
INC_FILE = "inconsciente.json"


# ————— Utilitário de Inline-Edit —————

def input_prefill(prompt, text):
    """Exibe o texto atual já preenchido no prompt para edição inline."""
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook(None)


# ————— Carregamento / Salvamento —————

def load_subconsciente():
    if os.path.exists(SUB_FILE):
        with open(SUB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "maes": {
            "0": {
                "nome": "Interações",
                "ultimo_child": 0,
                "blocos": []
            }
        }
    }

def save_subconsciente(data):
    with open(SUB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_inconsciente():
    if os.path.exists(INC_FILE):
        with open(INC_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_inconsciente(textos):
    with open(INC_FILE, 'w', encoding='utf-8') as f:
        json.dump(textos, f, ensure_ascii=False, indent=2)


# ————— CRUD Inconsciente —————

def list_inconsciente():
    textos = load_inconsciente()
    if not textos:
        print("\n🧠 Nenhum texto salvo em 'inconsciente'.")
        return
    print("\n🧠 Textos em 'inconsciente':")
    for i, t in enumerate(textos, 1):
        print(f" {i}. {t}")

def edit_inconsciente():
    textos = load_inconsciente()
    if not textos:
        print("\n🧠 Nada para editar.")
        return
    list_inconsciente()
    idx = input("\nID do texto a editar: ").strip()
    if not idx.isdigit() or not (1 <= int(idx) <= len(textos)):
        print("ID inválido.")
        return
    i = int(idx) - 1
    old = textos[i]
    novo = input_prefill("Novo texto: ", old).strip()
    if not novo:
        print("Cancelado.")
        return
    textos[i] = novo
    save_inconsciente(textos)
    print(f"🧠 Texto #{idx} atualizado.")

def remove_inconsciente():
    textos = load_inconsciente()
    if not textos:
        print("\n🧠 Nada para remover.")
        return
    list_inconsciente()
    idx = input("\nID do texto a remover: ").strip()
    if not idx.isdigit() or not (1 <= int(idx) <= len(textos)):
        print("ID inválido.")
        return
    i = int(idx) - 1
    texto = textos.pop(i)
    save_inconsciente(textos)
    print(f"🧠 Texto removido: {texto}")


# ————— Gestão de Mães & Blocos —————

def list_maes(data):
    print("\nMães cadastradas:")
    for mid, m in data["maes"].items():
        print(f"  ID={mid}: {m['nome']} (ultimo_child={m['ultimo_child']})")

def add_mae(data):
    nome = input("\nNome da nova mãe: ").strip()
    if not nome:
        print("Cancelado.")
        return
    ids = list(map(int, data["maes"].keys()))
    novo = str(max(ids) + 1)
    data["maes"][novo] = {"nome": nome, "ultimo_child": 0, "blocos": []}
    print(f"Mãe '{nome}' adicionada com ID={novo}.")

def select_mae(data):
    chaves = list(data["maes"].keys())
    if len(chaves) == 1:
        return chaves[0]
    list_maes(data)
    escolha = input("\nEscolha ID da mãe (enter=0): ").strip() or "0"
    return escolha if escolha in data["maes"] else "0"

def segment_text(texto):
    partes = re.split(r'(?<=[.?!])\s+', texto.strip())
    return [p.strip() for p in partes if p.strip()]

def calcular_alnulu(texto):
    mapa = {
        'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,
        'J':-10,'K':11,'L':12,'M':-13,'N':14,'O':15,'P':16,
        'Q':17,'R':18,'S':19,'T':20,'U':21,'V':-22,'W':23,
        'X':24,'Y':-25,'Z':26,'.':2,'!':3,'?':4,',':1,';':1,':':1,'-':1,
        '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
    }
    equiv = {
        'Á':'A','À':'A','Â':'A','Ã':'A','Ä':'A',
        'É':'E','Ê':'E','È':'E',
        'Í':'I','Ì':'I','Î':'I',
        'Ó':'O','Ò':'O','Ô':'O','Õ':'O','Ö':'O',
        'Ú':'U','Ù':'U','Û':'U','Ü':'U','Ç':'C','Ñ':'N'
    }
    total = 0
    for c in texto.upper():
        c = equiv.get(c, c)
        total += mapa.get(c, 0)
    return total

def get_last_index(mae):
    last = 0
    for b in mae["blocos"]:
        for part in ("entrada", "saida"):
            for tok in b[part]["tokens"]["TOTAL"]:
                idx = int(tok.split('.')[1])
                last = max(last, idx)
    return last

def generate_tokens(mae_id, start, e_cnt, re_cnt, ce_cnt):
    fmt = lambda i: f"{mae_id}.{i}"
    E   = [fmt(start + i) for i in range(e_cnt)]
    RE  = [fmt(start + e_cnt + i) for i in range(re_cnt)]
    CE  = [fmt(start + e_cnt + re_cnt + i) for i in range(ce_cnt)]
    TOTAL = E + RE + CE
    last_idx = start + e_cnt + re_cnt + ce_cnt - 1
    return {"E":E, "RE":RE, "CE":CE, "TOTAL":TOTAL}, last_idx

def create_entrada_block(data, mae_id, texto):
    mae = data["maes"][mae_id]
    re_ent  = input("Reação (entrada): ").strip()
    ctx_ent = input("Contexto (entrada): ").strip()
    aln_ent = calcular_alnulu(texto)
    last0   = get_last_index(mae)

    e_cnt  = len(re.findall(r'\S+', texto))
    re_cnt = len(re.findall(r'\S+', re_ent))
    ce_cnt = len(re.findall(r'\S+', ctx_ent))

    toks, last_e = generate_tokens(mae_id, last0+1, e_cnt, re_cnt, ce_cnt)
    fim_ent = toks["TOTAL"][-1]

    bloco = {
        "bloco_id": len(mae["blocos"])+1,
        "entrada": {
            "texto": texto,
            "reacao": re_ent,
            "contexto": ctx_ent,
            "tokens": toks,
            "fim": fim_ent,
            "alnulu": aln_ent
        },
        "saida": {}
    }
    return bloco, last_e

def add_saida_to_block(data, mae_id, bloco, last_idx, sugestoes):
    mae = data["maes"][mae_id]
    saidas = [s for s in sugestoes if s != bloco["entrada"]["texto"]]
    if not saidas:
        return last_idx

    print("\nSugestões para SAÍDA:")
    for i, s in enumerate(saidas,1):
        print(f" {i}. {s}")

    i = 0
    while i < len(saidas):
        seg = saidas[i]
        print(f"\n--- Saída Sugestão {i+1}/{len(saidas)} ---")
        print(seg)
        op = input("(i)nput / (e)ditar / (r)ejetar / (q)quit > ").lower().strip()
        if op=="q":
            break
        if op=="e":
            old = saidas[i]
            nova = input_prefill(" Novo texto: ", old).strip()
            if nova:
                saidas[i] = nova
                print("Texto de saída atualizado.")
            else:
                print("Mantido o original.")
            continue
        if op=="r":
            i += 1
            continue
        if op=="i":
            re_sai  = input("Reação (saída): ").strip()
            ctx_sai = input("Contexto (saída): ").strip()
            aln_sai = calcular_alnulu(seg)

            s_cnt  = len(re.findall(r'\S+', seg))
            rs_cnt = len(re.findall(r'\S+', re_sai))
            cs_cnt = len(re.findall(r'\S+', ctx_sai))

            toks_s, last_s = generate_tokens(mae_id, last_idx+1,
                                             s_cnt, rs_cnt, cs_cnt)
            fim_sai = toks_s["TOTAL"][-1]

            bloco["saida"] = {
                "texto": seg,
                "reacao": re_sai,
                "contexto": ctx_sai,
                "tokens": toks_s,
                "fim": fim_sai,
                "alnulu": aln_sai
            }
            return last_s
        print("Inválido.")
    return last_idx

def process_flow(data):
    mae_id = select_mae(data)

    inconsc = load_inconsciente()
    list_inconsciente()
    esc = input("\nSelecione ID do texto (enter=último): ").strip()

    if esc=="" or esc=="0":
        if inconsc:
            texto = inconsc[-1]
            print(f"\nUsando texto mais recente:\n{texto}")
        else:
            texto = input("\nNenhum texto. Digite novo:\n> ").strip()
            if not texto:
                print("Nada a processar.")
                return data
            inconsc.append(texto)
            save_inconsciente(inconsc)
            print("🧠 Texto salvo.")
    elif esc.isdigit() and 1 <= int(esc) <= len(inconsc):
        texto = inconsc[int(esc)-1]
        print(f"\nUsando texto #{esc}:\n{texto}")
    else:
        print("ID inválido.")
        return data

    sugestoes = segment_text(texto)
    print("\nSugestões segmentadas:")
    for i, s in enumerate(sugestoes, 1):
        print(f" {i}. {s}")

    for i, s in enumerate(sugestoes, 1):
        print(f"\nTrecho: {s}")
        op = input("(i)nput / (e)dit / (r)ej / (q)quit > ").lower().strip()
        if op == "q":
            break
        if op == "r":
            continue
        if op == "e":
            old = sugestoes[i-1]
            nova = input_prefill(" Novo texto: ", old).strip()
            if nova:
                sugestoes[i-1] = nova
                print("Texto atualizado.")
            else:
                print("Mantido o original.")
            continue
        if op == "i":
            bloco, last_e = create_entrada_block(data, mae_id, s)
            last_full = add_saida_to_block(data, mae_id, bloco, last_e, sugestoes)
            data["maes"][mae_id]["blocos"].append(bloco)
            data["maes"][mae_id]["ultimo_child"] = last_full
            print(f"✅ Bloco #{bloco['bloco_id']} salvo.")
            continue
        print("Inválido.")
    return data


# ————— CRUD de Blocos —————

def list_blocos(data):
    list_maes(data)
    mid = input("\nID da mãe para listar blocos: ").strip()
    if mid not in data["maes"]:
        print("Mãe não encontrada.")
        return
    mae = data["maes"][mid]
    blocos = mae["blocos"]
    if not blocos:
        print("Nenhum bloco cadastrado.")
        return
    print(f"\nBlocos de '{mae['nome']}':")
    for b in blocos:
        ent = b["entrada"]["texto"]
        sai = b["saida"].get("texto","")
        print(f" #{b['bloco_id']} → ENTRADA: {ent} | SAÍDA: {sai}")

def edit_bloco(data):
    list_blocos(data)
    mid = input("\nID da mãe para editar bloco: ").strip()
    if mid not in data["maes"]:
        print("Mãe não encontrada.")
        return
    blocos = data["maes"][mid]["blocos"]
    bid = input("ID do bloco: ").strip()
    bloco = next((b for b in blocos if str(b["bloco_id"]) == bid), None)
    if not bloco:
        print("Bloco não existe.")
        return
    part = input("Editar (e)ntrada ou (s)aída? ").lower().strip()
    if part not in ("e","s"):
        print("Inválido.")
        return
    key = "entrada" if part == "e" else "saida"
    campo = input("Campo (t)exto/(r)eação/(c)onteúdo? ").lower().strip()
    m = {"t":"texto","r":"reacao","c":"contexto"}.get(campo)
    if not m:
        print("Inválido.")
        return
    old = bloco[key][m]
    novo = input_prefill(f"Novo valor para {key}.{m}: ", old).strip()
    if novo:
        bloco[key][m] = novo
        print("✅ Atualizado.")
    else:
        print("Cancelado.")

def remove_bloco(data):
    list_blocos(data)
    mid = input("\nID da mãe para remover bloco: ").strip()
    if mid not in data["maes"]:
        print("Mãe não encontrada.")
        return
    blocos = data["maes"][mid]["blocos"]
    bid = input("ID do bloco: ").strip()
    idx = next((i for i,b in enumerate(blocos) if str(b["bloco_id"]) == bid), None)
    if idx is None:
        print("Bloco não existe.")
        return
    blocos.pop(idx)
    for i,b in enumerate(blocos, 1):
        b["bloco_id"] = i
    data["maes"][mid]["ultimo_child"] = get_last_index(data["maes"][mid])
    print("✅ Bloco removido.")

def menu_subconsciente(data):
    while True:
        print("""
7. Gerenciar subconsciente (blocos)
   1) Listar blocos
   2) Editar bloco
   3) Remover bloco
   0) Voltar
""")
        op = input("Opção: ").strip()
        if op == "1":
            list_blocos(data)
        elif op == "2":
            edit_bloco(data)
        elif op == "3":
            remove_bloco(data)
        elif op == "0":
            break
        else:
            print("Inválido.")


# ————— Menu Principal —————

def menu():
    subcon = load_subconsciente()
    while True:
        print("""
=== SUBCONSCIOUS MANAGER ===
1) Listar mães
2) Adicionar mãe
3) Processar texto
4) Listar inconsciente
5) Editar inconsciente
6) Remover inconsciente
7) Gerenciar subconsciente (blocos)
0) Sair
""")
        cmd = input("Opção: ").strip()
        if cmd == "1":
            list_maes(subcon)
        elif cmd == "2":
            add_mae(subcon)
        elif cmd == "3":
            subcon = process_flow(subcon)
        elif cmd == "4":
            list_inconsciente()
        elif cmd == "5":
            edit_inconsciente()
        elif cmd == "6":
            remove_inconsciente()
        elif cmd == "7":
            menu_subconsciente(subcon)
        elif cmd == "0":
            break
        else:
            print("Inválido.")
        save_subconsciente(subcon)

    save_subconsciente(subcon)
    print("Dados salvos. Até logo!")

if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando...")
        save_subconsciente(load_subconsciente())
        sys.exit(0)
