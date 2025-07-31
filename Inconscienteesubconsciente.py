#!/usr/bin/env python3
# subconscious_manager.py

import json
import os
import re
import sys

DATA_FILE = "subconsciente.json"
INCONSCIENTE_FILE = "inconsciente.json"


def load_data():
    """Carrega o JSON de mães e blocos, ou inicializa estrutura padrão."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
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


def save_data(data):
    """Persiste o JSON de mães e blocos no disco."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_inconsciente():
    """Carrega ou inicializa o dataset 'inconsciente'."""
    if os.path.exists(INCONSCIENTE_FILE):
        with open(INCONSCIENTE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_inconsciente(dataset):
    """Persiste o dataset 'inconsciente' no disco."""
    with open(INCONSCIENTE_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def list_inconsciente():
    """Exibe os textos já salvos em 'inconsciente'."""
    inconsc = load_inconsciente()
    if not inconsc:
        print("\n🧠 Nenhum texto salvo em 'inconsciente'.")
        return
    print("\n🧠 Textos em 'inconsciente':")
    for idx, txt in enumerate(inconsc, start=1):
        print(f" {idx}. {txt}")


def edit_inconsciente():
    """Edita um texto existente em 'inconsciente'."""
    inconsc = load_inconsciente()
    if not inconsc:
        print("\n🧠 Nada para editar.")
        return
    list_inconsciente()
    escolha = input("\nID do texto a editar: ").strip()
    if not escolha.isdigit() or not (1 <= int(escolha) <= len(inconsc)):
        print("ID inválido.")
        return
    idx = int(escolha) - 1
    novo = input("Novo texto: ").strip()
    if not novo:
        print("Cancelado.")
        return
    inconsc[idx] = novo
    save_inconsciente(inconsc)
    print(f"🧠 Texto #{escolha} atualizado.")


def remove_inconsciente():
    """Remove um texto de 'inconsciente'."""
    inconsc = load_inconsciente()
    if not inconsc:
        print("\n🧠 Nada para remover.")
        return
    list_inconsciente()
    escolha = input("\nID do texto a remover: ").strip()
    if not escolha.isdigit() or not (1 <= int(escolha) <= len(inconsc)):
        print("ID inválido.")
        return
    idx = int(escolha) - 1
    texto = inconsc.pop(idx)
    save_inconsciente(inconsc)
    print(f"🧠 Texto removido: {texto}")


def list_maes(data):
    """Exibe mães cadastradas."""
    print("\nMães cadastradas:")
    for mid, m in data["maes"].items():
        print(f"  ID={mid}: {m['nome']} (ultimo_child={m['ultimo_child']})")


def add_mae(data):
    """Adiciona nova mãe."""
    nome = input("\nNome da nova mãe: ").strip()
    if not nome:
        print("Cancelado.")
        return
    ids = [int(k) for k in data["maes"]]
    novo = str(max(ids) + 1)
    data["maes"][novo] = {"nome": nome, "ultimo_child": 0, "blocos": []}
    print(f"Mãe '{nome}' adicionada com ID={novo}.")


def select_mae(data):
    """Seleciona mãe ativa; se só houver uma, retorna '0'."""
    chaves = list(data["maes"])
    if len(chaves) == 1:
        return chaves[0]
    list_maes(data)
    escolha = input("\nDigite ID da mãe (enter=0): ").strip() or "0"
    return escolha if escolha in data["maes"] else "0"


def segment_text(texto):
    """Divide texto em sentenças."""
    partes = re.split(r'(?<=[.?!])\s+', texto.strip())
    return [p.strip() for p in partes if p.strip()]


def calcular_alnulu(texto):
    """Soma valores de caracteres conforme mapas."""
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
    """Retorna maior índice X de todos os tokens 'Y.X'."""
    last = 0
    for b in mae["blocos"]:
        for part in ("entrada", "saida"):
            for t in b[part]["tokens"]["TOTAL"]:
                idx = int(t.split('.')[1])
                last = max(last, idx)
    return last


def generate_tokens(mae_id, start, e_count, re_count, ce_count):
    """
    Gera tokens E, RE, CE e TOTAL sequenciais no formato 'mae_id.idx'.
    Retorna (dict, último_idx).
    """
    fmt = lambda i: f"{mae_id}.{i}"
    E   = [fmt(start + i) for i in range(e_count)]
    RE  = [fmt(start + e_count + i) for i in range(re_count)]
    CE  = [fmt(start + e_count + re_count + i) for i in range(ce_count)]
    TOTAL = E + RE + CE
    last_idx = start + e_count + re_count + ce_count - 1
    return {"E": E, "RE": RE, "CE": CE, "TOTAL": TOTAL}, last_idx


def create_entrada_block(data, mae_id, texto):
    """
    Cria a parte 'entrada' de um bloco:
    - solicita reação/contexto
    - calcula alnulu
    - gera tokens
    - determina 'fim' como último token
    Retorna (bloco_parcial, último_idx).
    """
    mae = data["maes"][mae_id]
    re_ent  = input("Reação (entrada): ").strip()
    ctx_ent = input("Contexto (entrada): ").strip()
    aln_ent = calcular_alnulu(texto)
    last0   = get_last_index(mae)

    e_cnt  = len(re.findall(r'\S+', texto))
    re_cnt = len(re.findall(r'\S+', re_ent))
    ce_cnt = len(re.findall(r'\S+', ctx_ent))

    toks_ent, last_e = generate_tokens(mae_id, last0 + 1, e_cnt, re_cnt, ce_cnt)
    fim_ent = toks_ent["TOTAL"][-1]

    bloco = {
        "bloco_id": len(mae["blocos"]) + 1,
        "entrada": {
            "texto": texto,
            "reacao": re_ent,
            "contexto": ctx_ent,
            "tokens": toks_ent,
            "fim": fim_ent,
            "alnulu": aln_ent
        },
        "saida": {}
    }
    return bloco, last_e


def add_saida_to_block(data, mae_id, bloco, last_idx, sugestoes):
    """
    Para um bloco já criado com 'entrada', apresenta sugestões de saída
    e aplica CRUD idêntico ao de entrada.
    Retorna último_idx atualizado.
    """
    mae = data["maes"][mae_id]
    saidas = [s for s in sugestoes if s != bloco["entrada"]["texto"]]
    if not saidas:
        return last_idx

    print("\nSugestões para SAÍDA:")
    for i, s in enumerate(saidas, 1):
        print(f" {i}. {s}")

    i = 0
    while i < len(saidas):
        seg = saidas[i]
        print(f"\n--- Saída Sugestão {i+1}/{len(saidas)} ---")
        print(seg)
        action = input("(i)nput / (e)ditar / (r)ejetar / (q)quit > ").lower().strip()

        if action == "q":
            break
        if action == "e":
            saidas[i] = input(" Novo texto: ").strip()
            continue
        if action == "r":
            i += 1
            continue
        if action == "i":
            re_sai  = input("Reação (saída): ").strip()
            ctx_sai = input("Contexto (saída): ").strip()
            aln_sai = calcular_alnulu(seg)

            s_cnt  = len(re.findall(r'\S+', seg))
            rs_cnt = len(re.findall(r'\S+', re_sai))
            cs_cnt = len(re.findall(r'\S+', ctx_sai))

            toks_sai, last_s = generate_tokens(mae_id, last_idx + 1,
                                               s_cnt, rs_cnt, cs_cnt)
            fim_sai = toks_sai["TOTAL"][-1]

            bloco["saida"] = {
                "texto": seg,
                "reacao": re_sai,
                "contexto": ctx_sai,
                "tokens": toks_sai,
                "fim": fim_sai,
                "alnulu": aln_sai
            }
            return last_s

        print("Inválido.")
    return last_idx


def process_flow(data):
    """
    Fluxo principal: segmenta texto existente ou novo e cria blocos.
    """
    mae_id = select_mae(data)

    inconsc = load_inconsciente()
    list_inconsciente()
    escolha = input("\nSelecione ID do texto (enter=último): ").strip()

    if escolha == "" or escolha == "0":
        if inconsc:
            texto = inconsc[-1]
            print(f"\nUsando texto mais recente: {texto}")
        else:
            texto = input("\nNenhum texto salvo. Digite novo texto:\n> ").strip()
            if not texto:
                print("Nada a processar.")
                return data
            inconsc.append(texto)
            save_inconsciente(inconsc)
            print("🧠 Texto salvo em 'inconsciente'.")
    elif escolha.isdigit() and 1 <= int(escolha) <= len(inconsc):
        texto = inconsc[int(escolha) - 1]
        print(f"\nUsando texto #{escolha}: {texto}")
    else:
        print("ID inválido.")
        return data

    sugestoes = segment_text(texto)
    print("\nSugestões segmentadas:")
    for i, s in enumerate(sugestoes, 1):
        print(f" {i}. {s}")

    for s in sugestoes:
        print(f"\nTrecho: {s}")
        op = input("(i)nput / (e)edit / (r)eject / (q)quit > ").lower().strip()
        if op == "q":
            break
        if op == "r":
            continue
        if op == "e":
            nova = input(" Novo texto: ").strip()
            if nova:
                sugestoes[sugestoes.index(s)] = nova
            continue
        if op == "i":
            bloco, last_e = create_entrada_block(data, mae_id, s)
            last_full = add_saida_to_block(data, mae_id, bloco, last_e, sugestoes)
            data["maes"][mae_id]["blocos"].append(bloco)
            data["maes"][mae_id]["ultimo_child"] = last_full
            print(f"✅ Bloco #{bloco['bloco_id']} salvo. ultimo_child={last_full}")
            continue
        print("Inválido.")

    return data


def menu():
    data = load_data()
    while True:
        print("\n=== SUBCONSCIOUS MANAGER ===")
        print("1. Listar mães")
        print("2. Adicionar mãe")
        print("3. Processar texto")
        print("4. Listar inconsciente")
        print("5. Editar inconsciente")
        print("6. Remover inconsciente")
        print("0. Sair")
        cmd = input("Opção: ").strip()

        if cmd == "1":
            list_maes(data)
        elif cmd == "2":
            add_mae(data)
        elif cmd == "3":
            data = process_flow(data)
        elif cmd == "4":
            list_inconsciente()
        elif cmd == "5":
            edit_inconsciente()
        elif cmd == "6":
            remove_inconsciente()
        elif cmd == "0":
            break
        else:
            print("Inválido.")

        save_data(data)

    save_data(data)
    print(f"\nDados salvos em '{DATA_FILE}'")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando...")
        save_data(load_data())
        sys.exit(0)