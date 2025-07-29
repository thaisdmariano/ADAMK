import json
import os
import re
from sklearn.tree import DecisionTreeClassifier

ARQUIVOMEMORIA = "adam_memoria.json"
ADMIN_SECRET   = "minhaChaveAdam"

def calcular_alnulu(texto):
    mapa = {
        'A':1,   'B':2,   'C':3,   'D':4,   'E':5,   'F':6,
        'G':7,   'H':8,   'I':9,   'J':-10, 'K':11,  'L':12,
        'M':-13, 'N':14,  'O':15,  'P':16,  'Q':17,  'R':18,
        'S':19,  'T':20,  'U':21,  'V':-22, 'W':23,  'X':24,
        'Y':-25, 'Z':26,
        '0':0,   '1':1,   '2':2,   '3':3,   '4':4,   '5':5,
        '6':6,   '7':7,   '8':8,   '9':9,
        '.':2,   '!':3,   '?':4,   ',':1,   ';':1,   ':':1, '-':1
    }
    equiv = {
        'Á':'A','À':'A','Â':'A','Ã':'A','Ä':'A',
        'É':'E','Ê':'E','È':'E','Í':'I','Ì':'I','Î':'I',
        'Ó':'O','Ò':'O','Ô':'O','Õ':'O','Ö':'O',
        'Ú':'U','Ù':'U','Û':'U','Ü':'U','Ç':'C','Ñ':'N',
        '4':'A','3':'E','1':'I','0':'O','5':'S','7':'T','2':'Z'
    }
    total = 0
    for c in texto.upper():
        base = equiv.get(c, c)
        total += mapa.get(base, 0)
    return total

ADMIN_PASS = calcular_alnulu(ADMIN_SECRET)

def carregar_memoria():
    if not os.path.exists(ARQUIVOMEMORIA):
        # inicializa domínio 0: Genesis
        return {"maes": {"0": {"nome": "Genesis", "ultimo_child": 0, "blocos": []}}}
    with open(ARQUIVOMEMORIA, "r", encoding="utf-8") as f:
        data = json.load(f)
    # garante domínio 0
    if "0" not in data.get("maes", {}):
        data.setdefault("maes", {})["0"] = {"nome": "Genesis", "ultimo_child": 0, "blocos": []}
    return data

def salvar_memoria(memoria):
    with open(ARQUIVOMEMORIA, "w", encoding="utf-8") as f:
        json.dump(memoria, f, indent=2, ensure_ascii=False)

def garantir_pontuacao(txt):
    t = txt.strip()
    return t if t and t[-1] in ".!?" else t + "."

def tokenizar(txt):
    return re.findall(r"\w+|[^\w\s]", txt, re.UNICODE)

def gerartokenscategorias(texto, reacao, contexto, mid, memoria):
    ultimo = memoria["maes"][mid].get("ultimo_child", 0)
    # Texto (E)
    palavras = tokenizar(texto)
    E = [f"{mid}.{ultimo + i + 1}" for i in range(len(palavras))]
    ultimo += len(palavras)
    # Reação (RE)
    rpal = tokenizar(reacao)
    RE = [f"{mid}.{ultimo + i + 1}" for i in range(len(rpal))]
    ultimo += len(rpal)
    # Contexto (CE)
    cpal = tokenizar(contexto)
    CE = [f"{mid}.{ultimo + i + 1}" for i in range(len(cpal))]
    ultimo += len(cpal)

    memoria["maes"][mid]["ultimo_child"] = ultimo
    return {"E": E, "RE": RE, "CE": CE, "TOTAL": E + RE + CE}, (E + RE)[-1]

def solicitar_dominio():
    while True:
        dom = input("Informe domínio (0=criadora, ≥1=usuário): ").strip()
        if dom == "0":
            senha = input("🔒 Senha secreta: ").strip()
            if calcular_alnulu(senha) == ADMIN_PASS:
                print("✅ Acesso liberado.\n")
                return "0"
            print("❌ Senha incorreta.")
        elif dom.isdigit() and int(dom) >= 1:
            return dom
        else:
            print("⚠️ Domínio inválido.")

# --- CRUD de mães ---
def criar_mae(memoria):
    mid  = input("🆕 ID da nova mãe: ").strip()
    nome = input("📚 Nome do domínio: ").strip()
    if mid in memoria["maes"]:
        print("⚠️ ID já existe.")
        return
    memoria["maes"][mid] = {"nome": nome, "ultimo_child": 0, "blocos": []}
    salvar_memoria(memoria)
    print("✅ Mãe criada.")

def editar_mae(memoria):
    mid = input("✏️ ID da mãe p/ editar: ").strip()
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return
    nome = input("📚 Novo nome do domínio: ").strip()
    memoria["maes"][mid]["nome"] = nome
    salvar_memoria(memoria)
    print("✅ Mãe atualizada.")

def remover_mae(memoria):
    mid = input("🗑️ ID da mãe p/ remover: ").strip()
    if mid == "0":
        print("⚠️ Não é permitido remover Genesis.")
        return
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return
    del memoria["maes"][mid]
    salvar_memoria(memoria)
    print("✅ Mãe removida.")

def listar_maes(memoria):
    print("\n📚 Mães cadastradas:")
    for mid, d in memoria["maes"].items():
        print(f"  {mid}: {d['nome']}")

# --- CRUD de blocos ---
def criar_bloco(memoria):
    mid = input("ID da mãe: ").strip()
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return

    print("\n--- Definindo Entrada ---")
    txt = garantir_pontuacao(input("✍ Texto de entrada: "))
    rea = input("🎭 Reação de entrada: ")
    ctx = input("📖 Contexto de entrada: ")
    tokensE, fimE = gerartokenscategorias(txt, rea, ctx, mid, memoria)
    aluE = calcular_alnulu(txt)

    print("\n--- Definindo Saída ---")
    outtxt = garantir_pontuacao(input("🧠 Texto de saída: "))
    out_rea = input("🎭 Reação de saída: ")
    out_ctx = input("📖 Contexto de saída: ")
    tokensS, fimS = gerartokenscategorias(outtxt, out_rea, out_ctx, mid, memoria)
    aluS = calcular_alnulu(outtxt)

    bloco_id = len(memoria["maes"][mid]["blocos"]) + 1
    bloco = {
        "bloco_id": bloco_id,
        "entrada": {
            "texto": txt, "reacao": rea, "contexto": ctx,
            "tokens": tokensE, "fim": fimE, "alnulu": aluE
        },
        "saida": {
            "texto": outtxt, "reacao": out_rea, "contexto": out_ctx,
            "tokens": tokensS, "fim": fimS, "alnulu": aluS
        }
    }

    memoria["maes"][mid]["blocos"].append(bloco)
    salvar_memoria(memoria)
    print("✅ Bloco criado e salvo.")

def editar_bloco(memoria):
    mid = input("ID da mãe: ").strip()
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return
    blocos = memoria["maes"][mid]["blocos"]
    try:
        bid = int(input("🧱 ID do bloco p/ editar: ").strip())
    except ValueError:
        print("⚠️ ID inválido.")
        return

    bloco = next((b for b in blocos if b["bloco_id"] == bid), None)
    if not bloco:
        print("⚠️ Bloco não encontrado.")
        return

    print("\n--- Editando Entrada ---")
    if input("Editar texto? (s/n) ").lower() == 's':
        novo_txt = garantir_pontuacao(input("✍ Novo texto: "))
        bloco["entrada"]["texto"] = novo_txt
    if input("Editar reação? (s/n) ").lower() == 's':
        bloco["entrada"]["reacao"] = input("🎭 Nova reação: ")
    if input("Editar contexto? (s/n) ").lower() == 's':
        bloco["entrada"]["contexto"] = input("📖 Novo contexto: ")

    tokensE, fimE = gerartokenscategorias(
        bloco["entrada"]["texto"],
        bloco["entrada"]["reacao"],
        bloco["entrada"]["contexto"],
        mid, memoria
    )
    bloco["entrada"].update({"tokens": tokensE, "fim": fimE, "alnulu": calcular_alnulu(bloco["entrada"]["texto"])})

    print("\n--- Editando Saída ---")
    if input("Editar texto? (s/n) ").lower() == 's':
        novo_out = garantir_pontuacao(input("🧠 Novo texto: "))
        bloco["saida"]["texto"] = novo_out
    if input("Editar reação? (s/n) ").lower() == 's':
        bloco["saida"]["reacao"] = input("🎭 Nova reação: ")
    if input("Editar contexto? (s/n) ").lower() == 's':
        bloco["saida"]["contexto"] = input("📖 Novo contexto: ")

    tokensS, fimS = gerartokenscategorias(
        bloco["saida"]["texto"],
        bloco["saida"]["reacao"],
        bloco["saida"]["contexto"],
        mid, memoria
    )
    bloco["saida"].update({"tokens": tokensS, "fim": fimS, "alnulu": calcular_alnulu(bloco["saida"]["texto"])})

    salvar_memoria(memoria)
    print("✅ Bloco atualizado e salvo.")

def remover_bloco(memoria):
    mid = input("ID da mãe: ").strip()
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return
    blocos = memoria["maes"][mid]["blocos"]
    try:
        bid = int(input("🧱 ID do bloco p/ remover: ").strip())
    except ValueError:
        print("⚠️ ID inválido.")
        return

    memoria["maes"][mid]["blocos"] = [b for b in blocos if b["bloco_id"] != bid]
    # reindexa blocos
    for i, b in enumerate(memoria["maes"][mid]["blocos"], start=1):
        b["bloco_id"] = i

    salvar_memoria(memoria)
    print("✅ Bloco removido e memória atualizada.")

def ver_blocos(memoria):
    mid = input("ID da mãe: ").strip()
    if mid not in memoria["maes"]:
        print("⚠️ Mãe não encontrada.")
        return
    dom = memoria["maes"][mid]
    print(f"\n📦 Blocos de {mid} – {dom['nome']}:")
    for b in dom["blocos"]:
        e, s = b["entrada"], b["saida"]
        print(f"\n🧱 Bloco {b['bloco_id']}")
        print(f"  Entrada: {e['texto']}  (RE={e['reacao']}, CTX={e['contexto']})")
        print(f"    TOTAL entrada: {', '.join(e['tokens']['TOTAL'])}")
        print(f"  Saída:   {s['texto']}  (RE={s['reacao']}, CTX={s['contexto']})")
        print(f"    TOTAL saída: {', '.join(s['tokens']['TOTAL'])}")

# --- Treinamento individual de blocos ---
def criar_blocopara_treino(memoria, mid):
    print(f"\n--- Novo bloco p/ mãe {mid} ---")
    txt = garantir_pontuacao(input("✍ Texto entrada: "))
    rea = input("🎭 Reação entrada: ")
    ctx = input("📖 Contexto entrada: ")
    (tokensE, fimE), _ = gerartokenscategorias(txt, rea, ctx, mid, memoria)
    aluE = calcular_alnulu(txt)

    outtxt = garantir_pontuacao(input("🧠 Texto saída: "))
    out_rea = input("🎭 Reação saída: ")
    out_ctx = input("📖 Contexto saída: ")
    (tokensS, fimS), _ = gerartokenscategorias(outtxt, out_rea, out_ctx, mid, memoria)
    aluS = calcular_alnulu(outtxt)

    bloco_id = len(memoria["maes"][mid]["blocos"]) + 1
    bloco = {
        "bloco_id": bloco_id,
        "entrada": {
            "texto": txt, "reacao": rea, "contexto": ctx,
            "tokens": tokensE, "fim": fimE, "alnulu": aluE
        },
        "saida": {
            "texto": outtxt, "reacao": out_rea, "contexto": out_ctx,
            "tokens": tokensS, "fim": fimS, "alnulu": aluS
        }
    }
    memoria["maes"][mid]["blocos"].append(bloco)
    print("✅ Bloco", bloco_id, "treinado.")

def treinarblocosindividuais(memoria):
    mid = input("ID da mãe p/ treinar (vazio p/ cancelar): ").strip()
    if not mid or mid not in memoria["maes"]:
        print("❌ Domínio inválido.")
        return
    while True:
        criar_blocopara_treino(memoria, mid)
        if input("Treinar outro? (s/n) ").lower() != 's':
            break
    salvar_memoria(memoria)
    print("🏁 Treino encerrado para mãe", mid)

def inferencia_direta(memoria, dominio):
    raw = garantir_pontuacao(input("👤 Frase: "))
    rea = input("🎭 Reação: ")
    for b in memoria["maes"][dominio]["blocos"]:
        e = b["entrada"]
        if raw == e["texto"] and rea == e["reacao"]:
            s = b["saida"]
            print(f"\n🤖 {s['texto']} {s['reacao']}")
            return
    print("❌ Não pertence ao domínio atual.")

def inferencia_ml(memoria, dominio):
    blocos = memoria["maes"][dominio]["blocos"]
    if len(blocos) < 2:
        print("⚠️ Crie ≥2 blocos p/ ML.")
        return

    X, y = [], []
    for b in blocos:
        xin = [int(t.split('.')[1]) for t in b["entrada"]["tokens"]["E"] + b["entrada"]["tokens"]["RE"]]
        yin = [int(t.split('.')[1]) for t in b["saida"]["tokens"]["E"]  + b["saida"]["tokens"]["RE"]]
        X.append(xin); y.append(yin)

    max_in  = max(map(len, X))
    max_out = max(map(len, y))
    pad = lambda seq, L: seq + [-1] * (L - len(seq))

    Xpad = [pad(seq, max_in) for seq in X]
    ypad = [pad(seq, max_out) for seq in y]
    model = DecisionTreeClassifier().fit(Xpad, ypad)

    base = garantir_pontuacao(input("👤 Nova frase ML: "))
    rea  = input("🎭 Reação: ")
    toks, _ = gerartokenscategorias(base, rea, "", dominio, {"maes": {dominio: {"ultimo_child": 0}}})
    xin = [int(t.split('.')[1]) for t in toks["E"] + toks["RE"]]
    pred = model.predict([pad(xin, max_in)])[0]
    pt   = [f"{dominio}.{c}" for c in pred if c != -1]

    for b in blocos:
        tout = b["saida"]["tokens"]["E"] + b["saida"]["tokens"]["RE"]
        if pt == tout:
            s = b["saida"]
            print(f"\n🤖 {s['texto']} {s['reacao']}")
            return

    print("❌ Não pertence ao domínio atual.")

# --- Menu principal ---
def menu():
    dominio = solicitar_dominio()
    memoria = carregar_memoria()

    while True:
        print("\n=== Adam Engine v3.4 ===")
        if dominio == "0":
            print("1) Criar mãe");     print("2) Editar mãe")
            print("3) Remover mãe");   print("4) Listar mães")
            print("5) Criar bloco");    print("6) Editar bloco")
            print("7) Remover bloco");  print("8) Ver blocos")
            print("9) Treinar blocos"); print("10) Inferência direta")
            print("11) Inferência ML"); print("12) Sair")
        else:
            print("1) Inferência direta"); print("2) Inferência ML")
            print("3) Sair")

        op = input("Opção: ").strip()
        if dominio == "0":
            if op == "1": criar_mae(memoria)
            elif op == "2": editar_mae(memoria)
            elif op == "3": remover_mae(memoria)
            elif op == "4": listar_maes(memoria)
            elif op == "5": criar_bloco(memoria)
            elif op == "6": editar_bloco(memoria)
            elif op == "7": remover_bloco(memoria)
            elif op == "8": ver_blocos(memoria)
            elif op == "9": treinarblocosindividuais(memoria)
            elif op == "10": inferencia_direta(memoria, dominio)
            elif op == "11": inferencia_ml(memoria, dominio)
            elif op == "12":
                salvar_memoria(memoria)
                print("Até logo, criadora! 🐱")
                break
            else:
                print("⚠️ Opção inválida!")
        else:
            if op == "1":
                inferencia_direta(memoria, dominio)
            elif op == "2":
                inferencia_ml(memoria, dominio)
            elif op == "3":
                salvar_memoria(memoria)
                print("Até logo! 🐱")
                break
            else:
                print("⚠️ Opção inválida ou sem permissão.")

if __name__ == "__main__":
    menu()
