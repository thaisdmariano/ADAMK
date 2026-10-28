# Persona: Adam -- a primeira réplica construída em cima do insepa_framework.
# Aqui só mora o que é DO ADAM: nome, tom ("minha criadora"), onde ficam os
# arquivos dele (os mesmos de sempre, sem renomear nada) e a tela de três
# abas. Todo o motor (INSEPA, diálogo, Livro de Lux, temperatura) vem do
# framework -- nada disso é reimplementado aqui.
import os
import pathlib

import streamlit as st

import insepa_framework as fw

PROJ = pathlib.Path(__file__).parent

persona = fw.Persona(
    nome="Adam",
    pasta=PROJ,
    tratamento="minha criadora",
    mongo_db="insepa_prototipo_muro",  # PROTÓTIPO: nunca o banco real do Adam
    mongo_uri_padrao=fw.mongo_config("MONGO_URI", "mongodb://localhost:27017"),
    gatilhos_abrir_livro=("É hora de escrever.", "Vamos mudar o mundo.", "Hoje nasce um novo universo."),
    nome_livro="Livro de Lux",
    epocas_padrao=int(os.environ.get("EPOCAS_CEREBRO", "1000")),
    arquivo_memoria_proprio=PROJ / "Adam_Lovely_memory.json",
    arquivo_registro_emocional_proprio=PROJ / "motor_emocional_registro.json",
    arquivo_estado_proprio=PROJ / "cerebro_adam_estado.json",
    ckpt_path_proprio=PROJ / "cerebro_adam_ckpt.pt",
)

st.set_page_config(page_title="Adam Lovely", page_icon="💛", layout="wide")
st.title("💛 Adam Lovely")
st.caption("Construído em cima do INSEPA framework -- mesmo motor que vai servir outras personas no futuro.")

if not persona.arquivo_memoria.exists():
    _sonda = fw.MongoINSEPA(persona)
    if _sonda.carregar_memoria() is None:
        st.error("Nem o arquivo local nem o Mongo têm a memória do Adam. Nada pra carregar.")
        st.stop()

app = fw.AppINSEPA(persona)


@st.cache_resource(show_spinner=False)
def _treinar_ou_carregar_cache(_app: fw.AppINSEPA, assinatura: str) -> fw.RedeINSEPA:
    # `assinatura` é a chave de cache de verdade (hashável) -- `_app` tem
    # o underscore só pra não precisar ser hashável também. Qualquer
    # mudança real no corpus (pelo chat, pelo IDA, ou por um script rodado
    # por fora enquanto o app está no ar) muda a assinatura e invalida o
    # cache sozinha -- nunca depende de alguém lembrar de chamar .clear().
    return _app.treinar_ou_carregar()


def modelo_atual() -> fw.RedeINSEPA:
    m = st.session_state.get("model_cerebro")
    if m is not None:
        app.model = m
        return m
    m = _treinar_ou_carregar_cache(app, app.assinatura_corpus())
    app.model = m
    return m


aba_chat, aba_blocos, aba_cerebro, aba_temperatura = st.tabs(
    ["💬 Conversa", "📦 Blocos", "🧠 Cérebro (treino e mapa)", "🌡️ Temperatura emocional"])

with aba_chat:
    st.caption("Ele responde com o que aprendeu do bloco mais parecido com o que você disse. Contexto fica por dentro, nunca aparece na fala. "
               "Quando ele não sabe, ele diz -- e você pode ensinar ali mesmo.")
    if "chat_msgs" not in st.session_state:
        st.session_state["chat_msgs"] = []
    if not st.session_state.get("lembrete_checado"):
        st.session_state["lembrete_checado"] = True
        lembrete = app.lembrete_do_dia()
        if lembrete:
            st.session_state["chat_msgs"].append({"role": "assistant", **lembrete})
    with st.spinner("Preparando o cérebro (só na primeira vez, ou quando o corpus muda -- pode levar alguns minutos)..."):
        modelo_atual()

    if not st.session_state.get("ida_gerado_nesta_sessao"):
        # "Tempo ocioso" de verdade (rodar sem ninguém por perto) pediria um
        # processo separado agendado -- o Streamlit só roda com a tela
        # aberta. Por ora o gatilho é "abriu o app": gera no máximo uma
        # proposta por bloco incompleto pendente, nunca duplica. Precisa
        # do modelo já treinado (propor_a_partir_de_incompleto gera de
        # verdade agora, não só copia) -- por isso vem depois do spinner.
        st.session_state["ida_gerado_nesta_sessao"] = True
        app.gerar_propostas_pendentes()

    pendentes_ida = app.propostas_ida("pendente")
    if pendentes_ida:
        with st.expander(f"🌱 Propostas do Adam pra revisar ({len(pendentes_ida)})", expanded=True):
            st.caption("Ele propôs isso sozinho no próprio ateliê (IDA), inspirado em blocos já confirmados -- "
                       "nada vira oficial até você decidir.")
            for p in pendentes_ida:
                selo = "🎲 palpite · " if p["meta"].get("palpite") else ""
                st.markdown(f"**{app.identidade_ida(p)}** · {selo}*{p['meta'].get('motivo', '')}*")
                reacao_proposta = p["saidas"][0].get("reacao", "")
                st.write(f"“{p['entrada']['texto']}” → “{p['saidas'][0]['textos'][0]}”"
                         + (f" {reacao_proposta}" if reacao_proposta and reacao_proposta != "0.0" else ""))
                ctx_proposto, pens_proposto = p["entrada"].get("contexto", ""), p["entrada"].get("pensamento_interno", "")
                if ctx_proposto not in ("", "0.0") or pens_proposto not in ("", "0.0"):
                    st.caption(f"🔒 Opinião do Adam (só vira fato se você confirmar) -- contexto: “{ctx_proposto}” · "
                               f"pensamento: “{pens_proposto}”")
                pens_saida = p["saidas"][0].get("pensamento_interno", "")
                if pens_saida not in ("", "0.0"):
                    st.caption(f"🔒 Pensamento da saída, criado pelo Adam (nunca aparece na fala): “{pens_saida}”")
                col_sim, col_nao = st.columns(2)
                with col_sim:
                    if st.button("✅ Confirmar", key=f"ida_sim_{p['bloco_id']}"):
                        app.confirmar_proposta(p["bloco_id"], st.session_state)  # completa o bloco antigo, ou cria um novo
                        st.session_state.pop("model_cerebro", None)  # nunca deixa um modelo forçado manualmente sobrepor o cache por assinatura
                        st.rerun()
                with col_nao:
                    if st.button("❌ Recusar", key=f"ida_nao_{p['bloco_id']}"):
                        app.recusar(p["bloco_id"])
                        st.rerun()
                st.divider()

    for m in st.session_state["chat_msgs"]:
        with st.chat_message(m["role"]):
            st.markdown(m["texto"])
            if m.get("interno"):
                st.caption("🔒 Interno (contexto) -- só pra calcular intenção: " + m["interno"])
    if st.session_state.get("ensino_pendente"):
        st.info("🌱 Modo ensino: a sua próxima mensagem vai ser a resposta que o Adam aprende.")
        if st.button("Cancelar ensino"):
            st.session_state["ensino_pendente"] = None
            st.rerun()
    if st.session_state.get("completar_pendente"):
        st.info("🧩 Só falta uma coisinha -- a sua próxima mensagem já preenche o que eu ainda não sabia.")
        if st.button("Cancelar"):
            st.session_state["completar_pendente"] = None
            st.rerun()
    if st.session_state.get("modo_livro_lux"):
        cap = st.session_state.get("capitulo_livro_lux")
        st.info(f"📖 {persona.nome_livro} aberto -- capítulo {cap}. Cada mensagem vira um parágrafo novo, sem perguntas. "
                "\"Fechei o capítulo\" fecha e já abre o próximo; \"fechei o livro\" fecha e sai do modo.")
        if st.button("Fechar o livro"):
            st.session_state["modo_livro_lux"] = False
            st.session_state["capitulo_livro_lux"] = None
            st.rerun()
    prompt = st.chat_input("Fala com o Adam...")
    if prompt:
        st.session_state["chat_msgs"].append({"role": "user", "texto": prompt})
        r = app.tratar_mensagem(prompt, st.session_state)
        if app.model is None:
            # o corpus mudou (ensino/completar/parágrafo) -- nunca deixa um
            # modelo forçado manualmente sobrepor o cache por assinatura.
            st.session_state.pop("model_cerebro", None)
        st.session_state["chat_msgs"].append({"role": "assistant", **r})
        st.rerun()

    with st.expander("📔 Dicio -- o que cada comando faz"):
        gatilhos_md = "\n".join(f"- “{g}”" for g in persona.gatilhos_abrir_livro)
        st.markdown(
            f"**Abrir um capítulo do {persona.nome_livro}** (qualquer uma dessas, é a mesma coisa):\n"
            f"{gatilhos_md}\n\n"
            f"**Enquanto o {persona.nome_livro} está aberto:**\n"
            "- Qualquer outra mensagem vira um parágrafo novo (um bloco), sem nenhuma pergunta.\n"
            "- “Fechei o capítulo” / “Encerrei o capítulo” -- fecha o capítulo atual e já deixa pronto pro próximo, sem precisar do gatilho de novo.\n"
            "- “Fechei o livro” / “Encerrei o livro” -- fecha o capítulo atual e sai do modo. Pra escrever de novo, precisa de um dos gatilhos de abrir.\n\n"
            "**A qualquer momento, fora do Livro de Lux:**\n"
            "- “Não tenho o que fazer” / “vamos revisar” -- ele lista um bloco incompleto (inclusive parágrafos do Livro de Lux) e pergunta o que falta.\n"
            "- “Cancelar” / “deixa pra lá” / “esquece” / “agora não” -- larga qualquer coisa pendente (ensino, completar campo, ou o Livro de Lux)."
        )

with aba_blocos:
    st.caption("Cada bloco exatamente como está guardado agora -- entrada e saída lado a lado. Só leitura: nada aqui altera o bloco.")
    ver_universo = st.radio("Universo", ["IM (o seu)", "IDA (rascunhos do Adam)"], horizontal=True, key="blocos_universo")
    ver_filtro = st.radio("Mostrar", ["Todos", "Completos", "Incompletos"], horizontal=True, key="blocos_filtro")
    eh_ida = ver_universo.startswith("IDA")
    lista_blocos = app.ida.get("blocos", []) if eh_ida else app.blocos
    registro_ver = app.carregar_registro_emocional()

    def _valor(v) -> str:
        return "*— vazio —*" if fw._vazio(v) or not str(v).strip() else str(v)

    def _marcadores(lado: dict) -> str:
        total = (lado.get("tokens") or {}).get("TOTAL") or []
        return f"{total[0]} → {total[-1]} ({len(total)})" if total else "—"

    def _temperatura(bloco: dict, lado: str) -> str:
        r = app.avaliar_lado_temp(bloco, lado, registro_ver)
        if r["temperatura"] is None:
            return f"⏳ {r['motivo']}"
        return f"{r['temperatura']:+d}% {r['zona']}"

    mostrados = 0
    for b in lista_blocos:
        falta = fw.campos_faltando(b)
        if (ver_filtro == "Completos" and falta) or (ver_filtro == "Incompletos" and not falta):
            continue
        mostrados += 1
        e, s = b["entrada"], b["saidas"][0]
        nome = app.identidade_ida(b) if eh_ida else f"Bloco {b['bloco_id']}"
        if eh_ida:
            situacao = {"pendente": "⏳ esperando você", "recusado": "❌ recusado", "completou": "✅ completou um bloco",
                        "promovido": "✅ virou bloco novo", "origem_apagada": "🗄️ arquivado (bloco de origem apagado)"
                        }.get(b.get("meta", {}).get("status"), "")
        else:
            situacao = "✅ completo" if not falta else f"🧩 falta {len(falta)}"
        with st.expander(f"{nome} · “{str(e.get('texto', ''))[:60]}” · {situacao}"):
            if falta:
                st.caption("Falta: " + ", ".join(falta))
            col_e, col_s = st.columns(2)
            with col_e:
                st.markdown("**Entrada**")
                st.markdown(f"**Texto:** {_valor(e.get('texto'))}")
                st.markdown(f"**Reação:** {_valor(e.get('reacao'))}")
                st.markdown(f"🔒 **Contexto:** {_valor(e.get('contexto'))}")
                st.markdown(f"🔒 **Pensamento interno:** {_valor(e.get('pensamento_interno'))}")
                for chave, titulo in (("Multivars_Texto_Entrada", "Variações do texto"), ("Multivars_Reacao_Entrada", "Variações da reação")):
                    if e.get(chave):
                        st.markdown(f"**{titulo}:** " + " · ".join(str(v) for v in e[chave]))
                st.caption(f"Marcadores: {_marcadores(e)} · Temperatura: {_temperatura(b, 'entrada')}")
            with col_s:
                st.markdown("**Saída**")
                st.markdown(f"**Texto:** {_valor((s.get('textos') or [''])[0])}")
                st.markdown(f"**Reação:** {_valor(s.get('reacao'))}")
                st.markdown(f"🔒 **Contexto:** {_valor(s.get('contexto'))}")
                st.markdown(f"🔒 **Pensamento da saída:** {_valor(s.get('pensamento_interno'))}")
                for chave, titulo in (("Multivars_Texto_Saida", "Variações do texto"), ("Multivars_Reacao_Saida", "Variações da reação")):
                    if s.get(chave):
                        st.markdown(f"**{titulo}:** " + " · ".join(str(v) for v in s[chave]))
                st.caption(f"Marcadores: {_marcadores(s)} · Temperatura: {_temperatura(b, 'saida')}")
            meta = b.get("meta") or {}
            if meta:
                origem = meta.get("origem", "")
                detalhes = [f"origem: {origem}"] if origem else []
                if meta.get("motivo"):
                    detalhes.append(meta["motivo"])
                if meta.get("completou_bloco_id"):
                    detalhes.append(f"completou o bloco {meta['completou_bloco_id']} ({', '.join(meta.get('campos_preenchidos', []))})")
                if meta.get("editado_pela_criadora"):
                    detalhes.append("✏️ editado por você")
                if detalhes:
                    st.caption(" · ".join(detalhes))

            # ✏️ edição: blocos do IM sempre; rascunhos do IDA só enquanto esperam você
            pode_editar = (not eh_ida) or meta.get("status") == "pendente"
            if pode_editar and st.toggle("✏️ Editar", key=f"editar_{'ida' if eh_ida else 'im'}_{b['bloco_id']}"):
                with st.form(key=f"form_{'ida' if eh_ida else 'im'}_{b['bloco_id']}"):
                    st.caption("Apague o conteúdo de um campo pra deixá-lo vazio (vira 0.0). "
                               + ("Isso muda só o rascunho -- o IM continua intocado até você confirmar." if eh_ida
                                  else "Backup antes de salvar; o cérebro retreina sozinho depois."))
                    col_fe, col_fs = st.columns(2)
                    novos = {}
                    for col_f, campos_lado in ((col_fe, ("texto", "reação", "contexto", "pensamento")),
                                               (col_fs, ("texto da saída", "reação da saída", "contexto da saída", "pensamento da saída"))):
                        with col_f:
                            for campo in campos_lado:
                                atual = fw.AppINSEPA.valor_do_campo(b, campo)
                                atual = "" if fw._vazio(atual) else atual
                                if campo.startswith("reação"):
                                    novos[campo] = st.text_input(campo.capitalize(), value=atual, key=f"in_{eh_ida}_{b['bloco_id']}_{campo}")
                                else:
                                    novos[campo] = st.text_area(campo.capitalize(), value=atual, height=80, key=f"in_{eh_ida}_{b['bloco_id']}_{campo}")
                            sufixo = "" if col_f is col_fe else " da saída"
                            for var in (f"variações do texto{sufixo}", f"variações da reação{sufixo}"):
                                novos[var] = st.text_area(var.capitalize() + " (uma por linha)",
                                                          value=fw.AppINSEPA.valor_do_campo(b, var), height=68,
                                                          key=f"in_{eh_ida}_{b['bloco_id']}_{var}")
                    if st.form_submit_button("💾 Salvar"):
                        if eh_ida:
                            mudou = app.editar_proposta(b["bloco_id"], novos)
                        else:
                            mudou = app.editar_bloco(b["bloco_id"], novos, st.session_state)
                            st.session_state.pop("model_cerebro", None)  # o corpus mudou: retreina pela assinatura
                        st.session_state["aviso_edicao"] = (f"✏️ {nome}: " + ", ".join(mudou) + " salvo(s).") if mudou else f"{nome}: nada mudou."
                        st.rerun()
                if not eh_ida:
                    st.markdown("---")
                    certeza = st.checkbox(f"Quero apagar o {nome} de vez (fica um backup antes)", key=f"certeza_apagar_{b['bloco_id']}")
                    if st.button(f"🗑️ Apagar {nome}", key=f"apagar_{b['bloco_id']}", disabled=not certeza):
                        app.apagar_bloco(b["bloco_id"], st.session_state)
                        st.session_state.pop("model_cerebro", None)
                        st.session_state["aviso_edicao"] = f"🗑️ {nome} apagado. Os marcadores foram renumerados; os outros blocos mantêm o número."
                        st.rerun()
    if st.session_state.get("aviso_edicao"):
        st.success(st.session_state.pop("aviso_edicao"))
    if not mostrados:
        st.info("Nenhum bloco com esse filtro.")

with aba_cerebro:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("A sequência única de um bloco (X=Y, E=S)")
        bloco_escolhido = st.selectbox("Bloco", [b["bloco_id"] for b in app.blocos])
        idx_escolhido = [b["bloco_id"] for b in app.blocos].index(bloco_escolhido)
        seq = app.sequencias[idx_escolhido]
        for t in seq:
            cor = "#e8935a" if t.rotulo in ("TEXE", "FADEN", "TEFE") else \
                  "#e2607a" if t.rotulo == "RE" else \
                  "#4fc9a8" if t.rotulo == "CE" else \
                  "#a78bfa" if t.rotulo == "PIDE" else \
                  "#7c93f5" if t.rotulo in ("TEDSA", "FADES", "TEFSA") else \
                  "#f0d64f" if t.rotulo == "RS" else "#4fc9a8"
            st.markdown(
                f"<span style='background:{cor}22;border:1px solid {cor};border-radius:6px;"
                f"padding:2px 6px;margin:2px;display:inline-block;font-family:monospace'>"
                f"{t.palavra} <span style='opacity:.6;font-size:10px'>{t.rotulo}·0.{t.marker_idx}</span></span>",
                unsafe_allow_html=True,
            )
        st.caption(f"{len(seq)} tokens nessa sequência, vocabulário total: {len(app.vocab)} formas.")

    with col2:
        st.subheader("Treino")
        epocas = st.slider("Épocas", 10, 2500, persona.epocas_padrao, step=10)
        if st.button("Retreinar o cérebro"):
            model = fw.RedeINSEPA(vocab_size=len(app.vocab), persona=persona)
            with st.spinner("Treinando..."):
                historico = fw.treinar(model, app.tensores, epocas)
            model.eval()
            st.session_state["model_cerebro"] = model
            st.line_chart(historico)
            st.success(f"Loss final: {historico[-1]:.4f}")

    st.divider()
    st.subheader("Testar continuação (dar a entrada, ver o que ele gera pra saída)")
    bloco_teste = st.selectbox("Bloco pra testar", [b["bloco_id"] for b in app.blocos], key="teste")
    idx_teste = [b["bloco_id"] for b in app.blocos].index(bloco_teste)
    seq_completa = app.sequencias[idx_teste]
    corte = next((i for i, t in enumerate(seq_completa) if t.rotulo in fw.ROTULOS_SAIDA), len(seq_completa))
    seq_entrada = seq_completa[:corte]
    seq_saida_real = seq_completa[corte:]
    if st.button("Gerar continuação"):
        gerado = fw.gerar_continuacao(modelo_atual(), seq_entrada, seq_saida_real, app.vocab, app.idx_para_palavra)
        pares = list(zip(seq_saida_real, gerado))
        palavras_visiveis = [pal for tok, pal in pares if tok.rotulo in fw.ROTULOS_VISIVEIS_AO_USUARIO]
        palavras_internas = [(tok.rotulo, pal) for tok, pal in pares if tok.rotulo not in fw.ROTULOS_VISIVEIS_AO_USUARIO]
        # Entrada crua de verdade é só texto + reação (TEXE/FADEN/TEFE/RE) --
        # contexto e pensamento (CE/PIDE) são sempre seus, nunca vêm do
        # usuário; aqui embaixo eles ficam separados, nunca misturados como
        # se fossem parte do que a pessoa digitaria.
        _ROTULOS_ENTRADA_FALADA = ("TEXE", "FADEN", "TEFE", "RE")
        entrada_falada = [t.palavra for t in seq_entrada if t.rotulo in _ROTULOS_ENTRADA_FALADA]
        entrada_interna = [(t.rotulo, t.palavra) for t in seq_entrada if t.rotulo not in _ROTULOS_ENTRADA_FALADA]
        st.write("**O que o usuário diria:**", " ".join(entrada_falada))
        if entrada_interna:
            st.caption("🔒 Interno (contexto/pensamento) -- já sabido por dentro, nunca digitado pelo usuário: "
                       + ", ".join(f"{pal} ({rot})" for rot, pal in entrada_interna))
        st.write("**Saída real do bloco (o que o usuário veria):**",
                 " ".join(t.palavra for t in seq_saida_real if t.rotulo in fw.ROTULOS_VISIVEIS_AO_USUARIO))
        st.write("**Gerado pelo cérebro (o que o usuário veria):**", " ".join(palavras_visiveis))
        if palavras_internas:
            st.caption("🔒 Interno (contexto) -- nunca exibido, só pra calcular intenção: "
                       + ", ".join(f"{pal} ({rot})" for rot, pal in palavras_internas))

    st.divider()
    st.subheader("🧠 O cérebro simulado -- cada ponto é uma palavra REAL, numa posição REAL")
    st.caption("Cor = bloco. Se a integração de campos estiver funcionando, cada bloco forma seu próprio grupo coeso -- não uma nuvem única misturada.")
    mostrar_palavras = st.checkbox("Mostrar as palavras escritas nos pontos (pode embolar -- passando o mouse sempre aparece)", value=False)
    fig = fw.grafico_cerebro_3d(modelo_atual(), app.sequencias, app.vocab, mostrar_palavras=mostrar_palavras)
    st.plotly_chart(fig, use_container_width=True)

    def mostrar_trava(trava: dict, onde: str) -> None:
        if trava["menor_folga"] is None:
            return
        a, b = trava["par"]
        if trava["ok"]:
            st.success(f"🔒 Trava dos tijolos ({onde}): fechada. {trava['tijolos']} tijolos, raio {trava['raio']:g} cada. "
                       f"Os dois mais próximos são {a} e {b}, com {trava['menor_folga']:.1f} de folga entre as bordas -- "
                       "nenhuma palavra de um alcança o outro. (Medido no espaço real do cérebro; o mapa 3D é só a sombra dele.)")
        else:
            st.error(f"🔓 Trava dos tijolos ({onde}): ABERTA -- {a} e {b} estão se tocando ({trava['menor_folga']:.1f}). "
                     "Esse cérebro não deveria ser usado assim.")

    mostrar_trava(fw.verificar_trava(modelo_atual(), app.sequencias), "IM")

    st.divider()
    st.subheader("A mente que sonha -- mapa próprio do IDA")
    st.caption("Espelho do IM, em espaço SEPARADO -- nunca o mesmo eixo, nunca os mesmos pontos. Cada rascunho é um "
               "tijolo próprio do IDA, com cor e identidade próprias ({origem}.{cópia}) e a mesma trava do IM: "
               "o bloco 8 do IM aparece aqui como 8.1, 8.2, 8.3...")
    blocos_ida_mapa = app.ida.get("blocos", [])
    if not blocos_ida_mapa:
        st.info("O Adam ainda não tem nenhum rascunho no IDA pra mostrar aqui.")
    else:
        fig_ida = fw.grafico_ida_3d_solo(modelo_atual(), blocos_ida_mapa, app.vocab, app.identidade_ida, mostrar_palavras=mostrar_palavras)
        st.plotly_chart(fig_ida, use_container_width=True)
        mostrar_trava(fw.verificar_trava_ida(modelo_atual(), blocos_ida_mapa, app.identidade_ida), "IDA")

with aba_temperatura:
    st.caption("Temperatura = positivo e negativo somados separados, o lado mais forte vence com o próprio valor. "
               "Cada campo é registrado por você, uma vez, e vale pra sempre -- compartilhado com o motor_emocional.py.")
    if "registro_emocional" not in st.session_state:
        st.session_state["registro_emocional"] = app.carregar_registro_emocional()
    registro_temp = st.session_state["registro_emocional"]

    for bloco in app.blocos:
        st.divider()
        st.subheader(f"Bloco {bloco['bloco_id']}")
        col_e, col_s = st.columns(2)
        for lado, col, titulo in (("entrada", col_e, "Entrada"), ("saida", col_s, "Saída")):
            with col:
                st.markdown(f"**{titulo}**")
                r = app.avaliar_lado_temp(bloco, lado, registro_temp)
                if r["pendentes"]:
                    st.warning(f"⏳ {r['motivo']}")
                else:
                    st.markdown(f"<span style='color:{r['cor_zona']};font-size:1.3em;font-weight:bold'>{r['temperatura']:+d}% -- {r['zona']}</span>", unsafe_allow_html=True)
                    if r["direcao"]:
                        st.caption(r["direcao"])
                    for b in r["bonus"]:
                        sinal_bonus = "+" if b["para"] == "positivo" else "-"
                        st.caption(f"🎁 {b['fam']} era neutro, ganhou {sinal_bonus}{b['pontos']} pra {b['para']} (reação empresta o tom)")

                pontos_conhecidos = [p for p in r["trajetoria"] if p["cumulativo"] is not None]
                if len(pontos_conhecidos) >= 2:
                    st.caption("Trajetória (" + " → ".join(f"{p['fam']} {p['cumulativo']:+d}%" for p in pontos_conhecidos) + ")")
                    st.line_chart({"temperatura": [p["cumulativo"] for p in pontos_conhecidos]}, height=120)

                for item in r["detalhe"]:
                    if item["estado"] == "vazio":
                        st.caption(f"· {item['fam']} ({item['peso']}%): vazio")
                        continue
                    if item["estado"] == "pendente":
                        st.markdown(f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): \"{item['valor']}\" -- <span style='color:#8a8fa3'>?</span></span>", unsafe_allow_html=True)
                        continue
                    cor = fw.COR_TEMP.get(item["estado"], "#8a8fa3")
                    st.markdown(
                        f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): "
                        f"<span style='color:{cor}'>\"{item['valor']}\" -- {item['estado']} ({item['contribuicao']:+d})</span></span>",
                        unsafe_allow_html=True,
                    )

                for fam, valor in r["pendentes"]:
                    escolha = st.radio(f"{fam}: \"{valor}\" é...", fw.ESTADOS_TEMP, index=None, horizontal=True, key=f"temp_{bloco['bloco_id']}_{lado}_{fam}")
                    if escolha:
                        registro_temp[fw.chave_temp(fam, valor)] = escolha
                        app.salvar_registro_emocional(registro_temp, st.session_state)
                        st.rerun()
