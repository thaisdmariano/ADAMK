import streamlit as st
import json
import re
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Caminhos fixos
# ────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SUB_FILE   = SCRIPT_DIR / "adam_memoria.json"
INC_FILE   = SCRIPT_DIR / "inconsciente.json"

# ────────────────────────────────────────────────────────────────────────────────
# Helpers para JSON
# ────────────────────────────────────────────────────────────────────────────────
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

# ────────────────────────────────────────────────────────────────────────────────
# Tokenização, índices e INSEPA
# ────────────────────────────────────────────────────────────────────────────────
def reindex_maes(maes_dict):
    items    = sorted(maes_dict.items(), key=lambda x: int(x[0]))
    new_maes = {str(i): m for i, (_, m) in enumerate(items)}
    if not new_maes:
        new_maes["0"] = {
            "nome": "Interações",
            "ultimo_child": "0.0",
            "blocos": []
        }
    return new_maes

def segment_text(text):
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

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
    # tokens de entrada
    for bloco in mae.get("blocos", []):
        for tok in bloco["entrada"]["tokens"]["TOTAL"]:
            idx = int(tok.split(".")[1])
            last = max(last, idx)
        # tokens de todas variações de saída
        for saida in bloco.get("saidas", []):
            for tok in saida["tokens"]["TOTAL"]:
                idx = int(tok.split(".")[1])
                last = max(last, idx)
    return last

def generate_tokens(mae_id, start, cnt_e, cnt_re, cnt_ce):
    fmt = lambda i: f"{mae_id}.{i}"
    E     = [fmt(start + i) for i in range(cnt_e)]
    RE    = [fmt(start + cnt_e + i) for i in range(cnt_re)]
    CE    = [fmt(start + cnt_e + cnt_re + i) for i in range(cnt_ce)]
    TOTAL = E + RE + CE
    return {"E": E, "RE": RE, "CE": CE, "TOTAL": TOTAL}, start + cnt_e + cnt_re + cnt_ce - 1

def create_entrada_block(data, mae_id, texto, re_ent, ctx_ent):
    mae   = data["maes"][mae_id]
    aln   = calcular_alnulu(texto)
    last0 = get_last_index(mae)

    e_units  = re.findall(r'\w+|[^\w\s]+', texto, re.UNICODE)
    re_units = [re_ent] if re_ent else []
    ce_units = re.findall(r'\w+|[^\w\s]+', ctx_ent, re.UNICODE)

    toks, last_idx = generate_tokens(
        mae_id, last0 + 1,
        len(e_units),
        len(re_units),
        len(ce_units)
    )

    bloco = {
        "bloco_id": len(mae["blocos"]) + 1,
        "entrada": {
            "texto":    texto,
            "reacao":   re_ent,
            "contexto": ctx_ent,
            "tokens":   toks,
            "fim":      toks["TOTAL"][-1] if toks["TOTAL"] else "",
            "alnulu":   aln
        },
        "saidas": [],
        "open": True
    }
    return bloco, last_idx

def add_saida_to_block(data, mae_id, bloco, last_idx, seg, re_sai, ctx_sai):
    aln      = calcular_alnulu(seg)
    s_units  = re.findall(r'\w+|[^\w\s]+', seg, re.UNICODE)
    rs_units = [re_sai] if re_sai else []
    cs_units = re.findall(r'\w+|[^\w\s]+', ctx_sai, re.UNICODE)

    toks_raw, last2 = generate_tokens(
        mae_id, last_idx + 1,
        len(s_units),
        len(rs_units),
        len(cs_units)
    )

    if not bloco["saidas"]:
        # primeira variação
        saida = {
            "textos":   [seg],
            "reacao":   re_sai,
            "contexto": ctx_sai,
            "tokens": {
                "S":     toks_raw["E"],
                "RS":    toks_raw["RE"],
                "CS":    toks_raw["CE"],
                "TOTAL": toks_raw["TOTAL"]
            },
            "fim": toks_raw["TOTAL"][-1] if toks_raw["TOTAL"] else ""
        }
        bloco["saidas"].append(saida)
    else:
        # acumula no último objeto
        saida = bloco["saidas"][-1]
        saida["textos"].append(seg)
        saida["tokens"]["S"].extend(toks_raw["E"])
        saida["tokens"]["RS"].extend(toks_raw["RE"])
        saida["tokens"]["CS"].extend(toks_raw["CE"])
        saida["tokens"]["TOTAL"].extend(toks_raw["TOTAL"])
        saida["fim"] = toks_raw["TOTAL"][-1]

    return last2

def insepa_tokenizar_texto(text_id, texto):
    units  = re.findall(r'\w+|[^\w\s]+', texto, re.UNICODE)
    tokens = [f"{text_id}.{i+1}" for i in range(len(units))]
    alnulu = calcular_alnulu(texto)
    ultimo = tokens[-1] if tokens else ""
    return {
        "nome":         f"Texto {text_id}",
        "texto":        texto,
        "tokens":       {"TOTAL": tokens},
        "ultimo_child": ultimo,
        "fim":          ultimo,
        "alnulu":       alnulu
    }

# ────────────────────────────────────────────────────────────────────────────────
# Início do App
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Subconscious Manager")
st.title("🧠 Subconscious Manager")
st.write("📂 Salvando JSON em:", SUB_FILE, INC_FILE)

subcon = load_json(
    SUB_FILE,
    {"maes": {"0": {"nome": "Interações",
                    "ultimo_child": "0.0",
                    "blocos": []}}}
)
subcon["maes"] = reindex_maes(subcon["maes"])
inconsc = load_json(INC_FILE, [])

menu = st.sidebar.radio(
    "Navegação",
    ["Mães", "Inconsciente", "Processar Texto", "Blocos"]
)

# ────────────────────────────────────────────────────────────────────────────────
# Aba “Mães”
# ────────────────────────────────────────────────────────────────────────────────
if menu == "Mães":
    st.header("Mães Cadastradas")
    for mid in sorted(subcon["maes"].keys(), key=int):
        m = subcon["maes"][mid]
        st.write(f"ID {mid}: {m['nome']} (último={m['ultimo_child']})")

    with st.form("add_mae"):
        nome = st.text_input("Nome da nova mãe")
        ok   = st.form_submit_button("Adicionar Mãe")
    if ok and nome.strip():
        new_id = str(max(map(int, subcon["maes"].keys())) + 1)
        subcon["maes"][new_id] = {
            "nome": nome.strip(),
            "ultimo_child": "0.0",
            "blocos": []
        }
        subcon["maes"] = reindex_maes(subcon["maes"])
        save_json(SUB_FILE, subcon)
        st.success(f"Mãe '{nome}' (ID={new_id}) adicionada")
        st.experimental_rerun()

    with st.form("remove_mae"):
        escolha = st.selectbox(
            "Selecionar mãe para remover",
            sorted(subcon["maes"].keys(), key=int),
            format_func=lambda x: f"{x} - {subcon['maes'][x]['nome']}"
        )
        ok2 = st.form_submit_button("Remover Mãe")
    if ok2:
        nome = subcon["maes"].pop(escolha)["nome"]
        subcon["maes"] = reindex_maes(subcon["maes"])
        save_json(SUB_FILE, subcon)
        st.success(f"Mãe '{nome}' removida")
        st.experimental_rerun()

    with st.form("edit_mae"):
        escolha   = st.selectbox(
            "Selecionar mãe para editar",
            sorted(subcon["maes"].keys(), key=int),
            format_func=lambda x: f"{x} - {subcon['maes'][x]['nome']}"
        )
        novo_nome = st.text_input(
            "Novo nome",
            subcon["maes"][escolha]["nome"]
        )
        ok3 = st.form_submit_button("Atualizar Nome")
    if ok3 and novo_nome.strip():
        subcon["maes"][escolha]["nome"] = novo_nome.strip()
        save_json(SUB_FILE, subcon)
        st.success("Nome atualizado")
        st.experimental_rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Aba “Inconsciente”
# ────────────────────────────────────────────────────────────────────────────────
elif menu == "Inconsciente":
    st.header("Inconsciente")
    converted = False
    for i, e in enumerate(inconsc):
        if isinstance(e, str):
            inconsc[i] = insepa_tokenizar_texto(str(i+1), e)
            converted = True
    if converted:
        save_json(INC_FILE, inconsc)

    st.subheader("Textos disponíveis")
    if inconsc:
        for i, e in enumerate(inconsc, 1):
            txt = e["texto"]
            exc = txt[:100] + ("..." if len(txt) > 100 else "")
            st.write(f"{i}. {exc}")
    else:
        st.info("Nenhum texto ainda.")

    with st.form("add_texto"):
        novo_txt = st.text_area("Inserir texto", height=200)
        uploads  = st.file_uploader(
            "Ou upload .txt",
            type=["txt"],
            accept_multiple_files=True
        )
        ok_add = st.form_submit_button("Adicionar Texto")
    if ok_add:
        cnt = 0
        if uploads:
            for f in uploads:
                content = f.read().decode("utf-8")
                inconsc.append(
                    insepa_tokenizar_texto(str(len(inconsc)+1), content)
                )
                cnt += 1
        elif novo_txt.strip():
            inconsc.append(
                insepa_tokenizar_texto(str(len(inconsc)+1), novo_txt)
            )
            cnt = 1
        if cnt:
            save_json(INC_FILE, inconsc)
            st.success(f"{cnt} texto(s) adicionado(s).")
        else:
            st.warning("Nada para adicionar.")

    with st.form("edit_texto"):
        idx     = st.number_input(
            "ID do texto", min_value=1,
            max_value=len(inconsc), value=1
        )
        updated = st.text_area(
            "Novo conteúdo",
            inconsc[idx-1]["texto"],
            height=200
        )
        ok_edit = st.form_submit_button("Editar Texto")
    if ok_edit:
        inconsc[idx-1] = insepa_tokenizar_texto(str(idx), updated)
        save_json(INC_FILE, inconsc)
        st.success(f"Texto #{idx} atualizado.")

    with st.form("remove_texto"):
        rid     = st.number_input(
            "Remover ID", min_value=1,
            max_value=len(inconsc), value=1
        )
        ok_rem  = st.form_submit_button("Remover Texto")
    if ok_rem:
        removed = inconsc.pop(rid-1)
        for i, e in enumerate(inconsc, 1):
            inconsc[i-1] = insepa_tokenizar_texto(str(i), e["texto"])
        save_json(INC_FILE, inconsc)
        st.success(f"Removido: {removed['nome']}")

# ────────────────────────────────────────────────────────────────────────────────
# Aba “Processar Texto”
# ────────────────────────────────────────────────────────────────────────────────
elif menu == "Processar Texto":
    st.header("Processar Texto")

    mae_ids = sorted(subcon["maes"].keys(), key=int)
    mae_id  = st.selectbox(
        "Selecionar mãe",
        mae_ids,
        format_func=lambda x: f"{x} - {subcon['maes'][x]['nome']}"
    )

    opts = ["Último salvo"] + [
        f"{i+1}. {t['texto'][:30]}{'...' if len(t['texto'])>30 else ''}"
        for i, t in enumerate(inconsc)
    ]
    escolha = st.selectbox("Escolha texto", opts)

    if escolha == "Último salvo" and inconsc:
        texto = inconsc[-1]["texto"]
    elif escolha != "Último salvo":
        idx   = int(escolha.split(".")[0]) - 1
        texto = inconsc[idx]["texto"]
    else:
        texto = st.text_area("Digite texto", "")

    if st.button("Segmentar"):
        st.session_state.sugestoes  = segment_text(texto)
        st.session_state.texto_base = texto
        st.session_state.mae_id     = mae_id

    if "sugestoes" in st.session_state:
        for idx, seg in enumerate(st.session_state.sugestoes, 1):
            st.subheader(f"Trecho {idx}")
            st.write(seg)
            action = st.radio(
                "Ação", ["Ignorar", "Entrada", "Saída"], key=f"act{idx}"
            )

            if action == "Entrada":
                ent_txt = st.text_input("Entrada", seg, key=f"ent{idx}")
                re_ent  = st.text_input("Reação", key=f"reac_ent{idx}")
                ctx_ent = st.text_input("Contexto", key=f"ctx_ent{idx}")
                if st.button("Salvar Entrada", key=f"save_ent{idx}"):
                    bloco, last = create_entrada_block(
                        subcon, st.session_state.mae_id,
                        ent_txt, re_ent, ctx_ent
                    )
                    subcon["maes"][mae_id]["blocos"].append(bloco)
                    subcon["maes"][mae_id]["ultimo_child"] = last
                    save_json(SUB_FILE, subcon)
                    st.success(f"Bloco #{bloco['bloco_id']} criado")

            elif action == "Saída":
                sai_txt = st.text_input("Saída", seg, key=f"sai{idx}")
                re_sai  = st.text_input("Reação", key=f"reac_sai{idx}")
                ctx_sai = st.text_input("Contexto", key=f"ctx_sai{idx}")

                blocos = subcon["maes"][mae_id]["blocos"]
                pend   = [b for b in blocos if b.get("open", False)]

                if not pend:
                    st.warning("Sem bloco pendente de saída.")
                else:
                    if len(pend) == 1:
                        bobj = pend[0]
                    else:
                        alvo_id = st.selectbox(
                            "Selecionar bloco pendente",
                            [b["bloco_id"] for b in pend],
                            key=f"target{idx}"
                        )
                        bobj = next(b for b in pend if b["bloco_id"] == alvo_id)

                    if bobj.get("saidas"):
                        st.markdown("**Variações atuais deste bloco:**")
                        for i, s in enumerate(bobj["saidas"], 1):
                            st.write(f"{i}. {s['textos'][-1]}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("➕ Adicionar variação", key=f"add_sai_{idx}"):
                            last0 = get_last_index(subcon["maes"][mae_id])
                            newidx = add_saida_to_block(
                                subcon, mae_id, bobj,
                                last0, sai_txt, re_sai, ctx_sai
                            )
                            subcon["maes"][mae_id]["ultimo_child"] = newidx
                            save_json(SUB_FILE, subcon)
                            st.success("📥 Variação adicionada!")
                            st.experimental_rerun()

                    with col2:
                        if st.button("✔️ Concluir bloco", key=f"close_block_{idx}"):
                            bobj["open"] = False
                            save_json(SUB_FILE, subcon)
                            st.success("🛑 Bloco finalizado")
                            st.experimental_rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Aba “Blocos”
# ────────────────────────────────────────────────────────────────────────────────
elif menu == "Blocos":
    st.header("Gerenciar Blocos")
    mae_ids = sorted(subcon["maes"].keys(), key=int)
    mae_id  = st.selectbox(
        "Escolha mãe", mae_ids,
        format_func=lambda x: f"{x} - {subcon['maes'][x]['nome']}"
    )
    blocos = subcon["maes"][mae_id]["blocos"]

    if not blocos:
        st.info("Nenhum bloco cadastrado.")
    else:
        for b in blocos:
            st.write(f"#{b['bloco_id']} → ENTRADA: {b['entrada']['texto']}")
            if b.get("saidas"):
                for i, s in enumerate(b["saidas"], 1):
                    # mostra todas as variações acumuladas
                    st.write(f"   SAÍDA {i}: {s['textos']}")
            else:
                st.write("   (Sem saídas)")

        bid   = st.number_input("ID do bloco", 1, len(blocos), 1)
        campo = st.radio(
            "Campo",
            ["entrada.texto", "entrada.reacao", "entrada.contexto"]
        )
        novo  = st.text_input("Novo valor")
        if st.button("Atualizar"):
            part, key = campo.split(".")
            subcon["maes"][mae_id]["blocos"][bid-1][part][key] = novo
            save_json(SUB_FILE, subcon)
            st.success("Bloco atualizado")
            st.experimental_rerun()

        rem = st.number_input("Remover ID", 1, len(blocos), 1, key="rem2")
        if st.button("Remover"):
            subcon["maes"][mae_id]["blocos"].pop(rem-1)
            for i, bb in enumerate(subcon["maes"][mae_id]["blocos"], 1):
                bb["bloco_id"] = i
            save_json(SUB_FILE, subcon)
            st.success(f"Bloco #{rem} removido")
            st.experimental_rerun()

        seq = st.text_input("Intervalo (ex:2-5)")
        if st.button("Remover Seq"):
            m = re.match(r"(\d+)-(\d+)", seq)
            if m:
                start, end = map(int, m.groups())
                subcon["maes"][mae_id]["blocos"] = [
                    bb for bb in subcon["maes"][mae_id]["blocos"]
                    if not (start <= bb["bloco_id"] <= end)
                ]
                for i, bb in enumerate(subcon["maes"][mae_id]["blocos"], 1):
                    bb["bloco_id"] = i
                save_json(SUB_FILE, subcon)
                st.success(f"Blocos {start}-{end} removidos")
                st.experimental_rerun()
            else:
                st.error("Intervalo inválido")

# ────────────────────────────────────────────────────────────────────────────────
# Rodapé
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.write("❤️ Desenvolvido com Streamlit")