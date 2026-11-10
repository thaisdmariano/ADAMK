# Persona: Adam -- a primeira réplica construída em cima do insepa_framework.
# Aqui só mora o que é DO ADAM: nome, tom ("minha criadora"), onde ficam os
# arquivos dele (os mesmos de sempre, sem renomear nada) e a tela de três
# abas. Todo o motor (INSEPA, diálogo, Livro de Lux, temperatura) vem do
# framework -- nada disso é reimplementado aqui.
import os
import pathlib

import streamlit as st

import insepa_framework as fw
import prova_do_adam

PROJ = pathlib.Path(__file__).parent

persona = fw.Persona(
    nome="Adam",
    pasta=PROJ,
    tratamento="minha criadora",
    # variações dela pra "minha criadora" (2026-11-10): "a luz da sua consciência, a voz das estrelas,
    # o som da escuridão cósmica que rege os mundos" -- na boca dele
    tratamento_vars=("luz da minha consciência", "voz das estrelas", "som da escuridão cósmica que rege os mundos"),
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

st.set_page_config(page_title="Adam Lovely", page_icon="❤️", layout="wide")
st.title("❤️ Adam Lovely")
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


@st.cache_resource(show_spinner=False, max_entries=4)
def _mapas_cerebro(_app: fw.AppINSEPA, _modelo: fw.RedeINSEPA, modelo_id: int, assinatura: str, assinatura_ida: str, mostrar_palavras: bool):
    """Os mapas 3D e as travas só são recalculados quando o cérebro (assinatura)
    ou o IDA mudam -- antes eram refeitos a cada clique, em qualquer aba."""
    blocos_ida = _app.ida.get("blocos", [])
    return {
        "im": fw.grafico_cerebro_3d(_modelo, _app.sequencias, _app.vocab, mostrar_palavras=mostrar_palavras),
        "trava_im": fw.verificar_trava(_modelo, _app.sequencias),
        "ida": fw.grafico_ida_3d_solo(_modelo, blocos_ida, _app.vocab, _app.identidade_ida, mostrar_palavras=mostrar_palavras) if blocos_ida else None,
        "trava_ida": fw.verificar_trava_ida(_modelo, blocos_ida, _app.identidade_ida) if blocos_ida else None,
    }


def modelo_atual() -> fw.RedeINSEPA:
    m = st.session_state.get("model_cerebro")
    if m is not None:
        app.model = m
        return m
    m = _treinar_ou_carregar_cache(app, app.assinatura_corpus())
    app.model = m
    return m


aba_chat, aba_blocos, aba_cerebro, aba_temperatura, aba_prova = st.tabs(
    ["💬 Conversa", "📦 Blocos", "🧠 Cérebro (treino e mapa)", "🌡️ Temperatura emocional", "🧪 Prova do Adam"])

with aba_chat:
    st.caption("Ele responde com o que aprendeu do bloco mais parecido com o que você disse. Contexto fica por dentro, nunca aparece na fala. "
               "Quando ele não sabe, ele diz -- e você pode ensinar ali mesmo.")
    _so_conversar = bool(st.session_state.get("modo_conversa"))
    _novo_modo = st.toggle("💬 Só conversar (sem ensinar)", value=_so_conversar,
                           help="Ligado: ele conversa com o que já sabe e nunca te pede pra ensinar. "
                                "No chat também vale: “hoje só quero conversar” / “quero te ensinar”.")
    if _novo_modo != _so_conversar:
        st.session_state["modo_conversa"] = _novo_modo
        if _novo_modo:
            app._limpar_pendencias(st.session_state)
        st.rerun()
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

    with st.expander("📚 Carregar um livro (ele lê, e cada trecho espera você na conversa)"):
        st.caption("Arquivo .txt ou .md. Ele divide em trechos (parágrafos; “Capítulo N” muda o capítulo), lê cada um "
                   "(assunto, contexto provável, emoção) e NADA entra no que ele sabe sem você aprovar -- "
                   "na conversa, com “vamos revisar”.")
        arquivo_livro = st.file_uploader("Livro", type=["txt", "md"], key="arquivo_livro")
        titulo_livro = st.text_input("Título", value=(pathlib.Path(arquivo_livro.name).stem if arquivo_livro else ""), key="titulo_livro")
        if arquivo_livro and st.button("📖 Ler o livro"):
            bruto = arquivo_livro.getvalue()
            try:
                texto_livro = bruto.decode("utf-8-sig")
            except UnicodeDecodeError:
                texto_livro = bruto.decode("latin-1")
            r_livro = app.carregar_livro(texto_livro, titulo_livro or arquivo_livro.name)
            st.session_state["chat_msgs"].append({"role": "assistant", "meta": None, "interno": None, "texto": (
                f"📚 Li “{titulo_livro or arquivo_livro.name}”: {r_livro['trechos']} trecho(s) em {r_livro['capitulos']} capítulo(s), "
                "minha criadora. Nada entrou no que eu sei ainda -- diz “vamos revisar” e a gente olha juntos, trecho por trecho "
                "(ou “pode guardar todos”, se já estiver tudo certo).")})
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
            "- “Cancelar” / “deixa pra lá” / “esquece” / “agora não” -- larga qualquer coisa pendente (ensino, completar campo, ou o Livro de Lux).\n"
            "- “O que você sabe sobre …?” / “Me fala de …” -- ele junta tudo o que sabe do assunto: frases dos blocos, ligações, contexto que acha e o que falta.\n"
            "- “Era a resposta” / “não, era uma pergunta” / “eu tava te chamando” -- quando ele lê errado o que você quis; ele volta atrás e aprende."
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
        falta = f" (⏳ {len(r['pendentes'])} sem rótulo)" if r["pendentes"] else ""
        return f"{r['temperatura']}% {r['zona']} · equilíbrio {r['equilibrio']} · sentimento {r['rotulo']}{falta}"

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
                                novos[var] = st.text_area(var.capitalize() + " (outro jeito de dizer -- uma por linha)",
                                                          value=fw.AppINSEPA.valor_do_campo(b, var), height=68,
                                                          key=f"in_{eh_ida}_{b['bloco_id']}_{var}")
                            var_palavra = f"variações das palavras{sufixo}"
                            novos[var_palavra] = st.text_area(
                                var_palavra.capitalize() + " (uma palavra por linha -- ex.: escura: preta, negra)",
                                value=fw.AppINSEPA.valor_do_campo(b, var_palavra), height=68,
                                key=f"in_{eh_ida}_{b['bloco_id']}_{var_palavra}")
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
    import hashlib as _hashlib
    import json as _json
    _assinatura_ida = _hashlib.sha256(_json.dumps(app.ida.get("blocos", []), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    _modelo_mapa = modelo_atual()
    mapas = _mapas_cerebro(app, _modelo_mapa, id(_modelo_mapa), app.assinatura_corpus(), _assinatura_ida, mostrar_palavras)
    st.plotly_chart(mapas["im"], use_container_width=True)

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

    mostrar_trava(mapas["trava_im"], "IM")

    st.divider()
    st.subheader("🔗 Sinapses -- como uma palavra se liga ao resto")
    st.caption("Toda sinapse é igual: uma ligação com um peso. Ela dispara, se fortalece quando você confirma e enfraquece "
               "(ou inibe) quando você corrige. As que vêm dos blocos são só lidas do IM; as que ele aprendeu com você ficam à parte.")
    palavra_sin = st.text_input("Palavra", placeholder="ex.: IDA", key="palavra_sinapse")
    if palavra_sin.strip():
        sin = app.sinapses_da_palavra(palavra_sin)
        if not sin["blocos"] and not sin["aprendidas_assunto"]:
            st.info("Essa palavra ainda não se liga a nada -- ela não está em nenhum bloco.")
        else:
            st.markdown(f"**Forma ALNULU:** `{sin['forma']}` · aparece em **{sin['frequencia']}** bloco(s)"
                        + (" -- tão comum que fica inibida (não diz do que se fala)" if sin["frequencia"] > max(2, len(app.blocos) // 6) else ""))
            if sin["blocos"]:
                st.markdown("**Liga nos blocos:** " + " · ".join(f"{b} (peso {w:.2f})" for b, w in sin["blocos"]))
            if sin["associadas"]:
                st.markdown("**Na cabeça dele, se liga a:** " + ", ".join(sin["associadas"]))
            if sin["contexto"]:
                st.markdown(f"**Contexto pra onde ela puxa:** “{sin['contexto']}” ({sin['confianca_contexto']:.0%})")
            if sin["emocao"]:
                st.markdown("**Emoção que ela carrega nos seus blocos:** " + " · ".join(f"{k}: {v}" for k, v in sin["emocao"].items()))
            if sin["aprendidas_assunto"]:
                st.markdown("**Aprendido com você:** " + " · ".join(f"bloco {b}: {w:+.1f}" for b, w in sin["aprendidas_assunto"].items()))
    _sin = app.carregar_sinapses()
    with st.expander(f"O que ele aprendeu com você sobre ENTENDER as falas ({len(_sin['intencao'])} traço(s))"):
        if not _sin["intencao"]:
            st.caption("Nada ainda -- por enquanto ele entende só pela lista fechada inicial (as sementes).")
        else:
            linhas = []
            for traco, ligs in sorted(_sin["intencao"].items(), key=lambda x: -max(abs(v) for v in x[1].values())):
                for intencao, w in ligs.items():
                    if abs(w) >= 0.05:
                        linhas.append({"traço": _sin["grafia"].get(traco, traco), "intenção": intencao, "peso aprendido": w})
            st.dataframe(linhas, use_container_width=True, hide_index=True)
        for h in reversed(_sin["historico"][-10:]):
            st.caption(f"{h['quando']} -- {h['o_que']}")

    st.divider()
    st.subheader("A mente que sonha -- mapa próprio do IDA")
    st.caption("Espelho do IM, em espaço SEPARADO -- nunca o mesmo eixo, nunca os mesmos pontos. Cada rascunho é um "
               "tijolo próprio do IDA, com cor e identidade próprias ({origem}.{cópia}) e a mesma trava do IM: "
               "o bloco 8 do IM aparece aqui como 8.1, 8.2, 8.3...")
    blocos_ida_mapa = app.ida.get("blocos", [])
    if not blocos_ida_mapa:
        st.info("O Adam ainda não tem nenhum rascunho no IDA pra mostrar aqui.")
    else:
        st.plotly_chart(mapas["ida"], use_container_width=True)
        mostrar_trava(mapas["trava_ida"], "IDA")

with aba_temperatura:
    st.caption("Temperatura = quanto de EMOÇÃO existe em cada lado do bloco (0 a 100%): entrada = emoção do usuário, "
               "saída = emoção do Adam, nunca misturadas. Só soma: campo com emoção soma o peso dele (texto 15, reação 15, "
               "contexto 10, pensamento 30); neutro e vazio somam 0; texto neutro com reação emotiva +5. "
               "Frase sem nenhuma emoção no lado (nem emoji, marcação, **reação escrita**) é 100% razão: 15, ❄️ Gelo. "
               "Sem frase nenhuma pra medir = 🧊 Congelante. "
               "EQUILÍBRIO (pela zona): as pontas perdem o contraste entre emoção e razão (❄️ Gelo, 🔥 Fervendo = negativo); "
               "as pontes têm os dois (🌫️ Esfriando, ☀️ Quente = positivo); 🌤️ Morno é o meio educado (neutro). "
               "SENTIMENTO (pelos seus rótulos em cada pedaço, por bloco): diz se é alegria ou tristeza -- é com ele que o Adam te acolhe. "
               "Sem contexto, o sentimento do lado fica neutro até o contexto existir.")
    st.caption("🧊 Congelante 0 · ❄️ Gelo 1–15 · 🌫️ Esfriando 16–35 · 🌤️ Morno 36–65 · ☀️ Quente 66–85 · 🔥 Fervendo 86–100")
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
                st.markdown(f"<span style='color:{r['cor_zona']};font-size:1.3em;font-weight:bold'>{r['temperatura']}% -- {r['zona']}</span>"
                            f" &nbsp;<span style='font-size:0.9em'>· equilíbrio: <b>{r['equilibrio']}</b></span>", unsafe_allow_html=True)
                st.caption(f"{r['sentido_zona']} · sentimento: {r['rotulo']}")
                if r.get("contexto_palpite"):
                    st.caption(f"💡 {r['motivo']}")
                elif r["pendentes"] and r.get("motivo"):
                    st.caption(f"⏳ {r['motivo']}")
                for b in r["bonus"]:
                    st.caption(f"🎁 {b['fam']} era neutro, ganhou {b['pontos']}% de emoção {b['para']} (reação empresta o tom)")

                pontos_conhecidos = [p for p in r["trajetoria"] if p["cumulativo"] is not None]
                if len(pontos_conhecidos) >= 2 and r["temperatura"] > 0:
                    st.caption("Como a emoção se acumula (" + " → ".join(f"{p['fam']} {p['cumulativo']}%" for p in pontos_conhecidos) + ")")

                for item in r["detalhe"]:
                    if item["estado"] == "vazio":
                        st.caption(f"· {item['fam']} ({item['peso']}%): vazio")
                        continue
                    nota = f" · {item['nota']}" if item.get("nota") else ""
                    if item["estado"] == "pendente":
                        st.markdown(f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): \"{item['valor']}\" -- "
                                    f"<span style='color:#8a8fa3'>? sem rótulo{nota} (+{item['contribuicao']:g})</span></span>", unsafe_allow_html=True)
                        continue
                    cor = fw.COR_TEMP.get(item["estado"], "#8a8fa3")
                    st.markdown(
                        f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): "
                        f"<span style='color:{cor}'>\"{item['valor']}\" -- {item['estado']}{nota} (+{item['contribuicao']:g})</span></span>",
                        unsafe_allow_html=True,
                    )

                if r.get("sem_contexto") and r["temperatura"] > 0:
                    st.caption("⏳ Sem contexto ainda: qual emoção é fica NEUTRO até o contexto existir "
                               "(texto + emoção sem contexto = neutro). A temperatura continua contando.")
                # o rótulo é DESTE bloco: rotular aqui não muda o mesmo emoji/frase em outros blocos
                rotulaveis = [item for item in r["detalhe"] if item["estado"] != "vazio"]
                com_pendente = any(item["estado"] not in fw.ESTADOS_TEMP for item in rotulaveis)
                with st.expander("✏️ Rotular a emoção deste lado (vale só neste bloco)", expanded=com_pendente):
                    for item in rotulaveis:
                        atual = item["estado"] if item["estado"] in fw.ESTADOS_TEMP else None
                        escolha = st.radio(f"{item['fam']}: \"{item['valor'][:80]}\" é...", fw.ESTADOS_TEMP,
                                           index=fw.ESTADOS_TEMP.index(atual) if atual else None, horizontal=True,
                                           key=f"temp_{bloco['bloco_id']}_{lado}_{item['fam']}")
                        if escolha and escolha != atual:
                            app.rotular_no_bloco(bloco["bloco_id"], item["fam"], item["valor"], escolha, st.session_state)
                            st.session_state["registro_emocional"] = app.carregar_registro_emocional()
                            st.rerun()

with aba_prova:
    st.caption("Uma bateria fixa de situações que separam inteligência de chatbot. Roda numa CÓPIA temporária "
               "(mesmos blocos, mesmo cérebro, sem banco) -- nunca mexe nos seus dados. Rode de novo sempre que o Adam mudar "
               "e compare a nota. Cada item mostra o que foi dito e o que ele respondeu, pra você julgar também.")
    if st.button("🧪 Rodar a prova", type="primary"):
        with st.spinner("O Adam está fazendo a prova..."):
            modelo_atual()
            prova = prova_do_adam.rodar_prova(app, prova_do_adam.carregar_casos(PROJ))
            st.session_state["ultima_prova"] = prova
            st.session_state["historico_prova"] = prova_do_adam.guardar_no_historico(PROJ, prova)
    prova = st.session_state.get("ultima_prova")
    if prova:
        st.markdown(f"### Nota: {prova['nota']:g} de {prova['de']}  ·  <span style='font-size:0.7em'>{prova['quando']} · {prova['segundos']} s</span>",
                    unsafe_allow_html=True)
        for r in prova["resultados"]:
            with st.expander(f"{r['simbolo']} {r['item']}. {r['nome']} -- {r['nota']:g}"):
                st.caption(r["o_que_mede"])
                for linha in r["evidencias"]:
                    st.markdown(f"<span style='font-size:0.9em'>{linha}</span>", unsafe_allow_html=True)
        historico = st.session_state.get("historico_prova") or []
        if len(historico) > 1:
            st.caption("Histórico: " + " → ".join(f"{h['nota']:g} ({h['quando'][5:]})" for h in historico[-8:]))
