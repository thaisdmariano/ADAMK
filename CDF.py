#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Caderno de Ferramentas ADAM Lovely.

Notebook Streamlit onde construímos e testamos, camada por camada, as
micro-ferramentas do INSEPA (e do que vier depois) isoladas do lovely.py.
"""

import streamlit as st

from insepa_marcadores import tokenizar, marcar_entrada, formatar
from insepa_classificacao import LEXICO_EMOCOES, classificar_emocao, classificar_contexto
from insepa_integracao import montar_bloco
from insepa_hash import Cofre, chave_hash, im_de
from insepa_aprendizado import resolver_termo, classificar_dado, marcar_opiniao
from insepa_espelhamento import comparar_eixo, espelhar

st.set_page_config(page_title="Caderno de Ferramentas ADAM Lovely", page_icon="🧰", layout="wide")

st.title("🧰 Caderno de Ferramentas ADAM Lovely")
st.caption("Construindo e testando o INSEPA camada por camada, fora do lovely.py.")

camada = st.sidebar.radio(
    "Camada",
    [
        "📖 Codex ADAM INSEPA",
        "Camada 1 — Marcadores Únicos",
        "Camada 2 — Classificação",
        "Camada 3 — Integração",
        "Camada 4 — Hashrização",
        "Camada 5 — Aprendizado Seguro",
        "Camada 6 — Espelhamento e Análise",
        "🧮 Caixa de Ferramentas (sugestões)",
    ],
)

if camada == "📖 Codex ADAM INSEPA":
    st.header("📖 Codex ADAM INSEPA")
    st.markdown(
        "Referência viva do que já entendemos do INSEPA, construída junto com a Thaís "
        "camada por camada. Cresce conforme avançamos."
    )

    st.subheader("Terminologia")
    st.markdown(
        "- **IM (Índice Mãe)**: o número antes do ponto num marcador (ex.: o `0` de `0.25`). "
        "É o mesmo para todos os blocos de um universo/domínio — todo IM é 'mãe' dos seus IFs.\n"
        "- **IF (Índice Filho)**: o número depois do ponto (`0.1`, `0.25`, `0.26`...). "
        "Cada IF é único dentro daquele IM e nunca se repete, mesmo entre blocos diferentes.\n"
        "- **Vars**: variações de PALAVRA (texto) ou de emoji (emoção) que significam a mesma coisa.\n"
        "- **Multivars**: variações de FRASE inteira (texto) que significam a mesma coisa.\n"
        "- **Pensamento interno**: a raiz/tronco de um bloco — de onde nascem o ramo entrada "
        "e o ramo saída (Camada 3).\n"
        "- **Opinião**: um dado com só texto + emoção — o que qualquer usuário pode "
        "produzir. Ainda não confirmado.\n"
        "- **Dado concreto**: um dado com texto + emoção + contexto + pensamento interno, "
        "todos presentes — e contexto/pensamento só podem ser definidos pela **criadora**, "
        "nunca por qualquer usuário (mesmo princípio da senha de admin no lovely.py).\n"
        "- **Placeholder `{IM}.0`**: marcador reservado (nunca usado pela sequência normal, "
        "que começa em `.1`) que sinaliza \"contexto/pensamento ainda não confirmados\" "
        "direto na estrutura do dado, sem precisar de uma flag separada.\n"
        "- **`Adam_Lovely_unconscious.json` (o inconsciente)**: é a Camada 1 crua e só ela — "
        "puro marcador → token + vars, sem nenhuma classificação (sem emoção/contexto/"
        "pensamento interpretados). `Adam_Lovely_memory.json` (o consciente) é o pacote "
        "completo: os mesmos marcadores, mas já com Camada 2 e 3 aplicadas (emoção, "
        "contexto e pensamento_interno presentes). O inconsciente é o \"dado bruto\"; o "
        "consciente é o \"dado com sentido\"."
    )

    st.subheader("As camadas")
    st.markdown(
        "1. **Marcadores Únicos** — cada token ganha um marcador que nunca se repete fora "
        "daquela entrada. A mesma palavra em frases diferentes é um símbolo diferente — "
        "isso impede o sistema de confundir significados.\n"
        "2. **Classificação** — não inventa nada, só classifica o que o texto já carrega: "
        "toda entrada tem emoção (mesmo que seja frieza) e contexto, explícitos ou ocultos "
        "(via var).\n"
        "3. **Integração** — o pensamento interno é a raiz; dele nascem o ramo entrada e o "
        "ramo saída, que ressoam entre si por compartilharem essa raiz, não por seus "
        "contextos se parecerem.\n"
        "4. **Hashrização** — a chave é a junção dos marcadores de TEXTO + EMOÇÃO (explícita "
        "ou oculta por var — é aí que entra a similaridade da Camada 2). Contexto fica de "
        "fora da chave, ele é inferido depois por ressonância (Camada 3), não por hash. "
        "Só a chave exata (texto+emoção) destranca a porta pra saída correspondente. A "
        "sequência de marcadores nunca reinicia nem reusa posição dentro de um IM — "
        "só cresce, o que garante que a chave seja sempre única no histórico real.\n"
        "5. **Aprendizado Seguro** — o ALNULU só deixa o Adam VER a forma de uma palavra; "
        "reconhecer é o INSEPA que decide, por **registro exato** (token conhecido, ou "
        "var/multivar já cadastrada num bloco) — nunca por aproximação/distância. IA comum "
        "usa embeddings e similaridade de vetor; o Adam usa algarismos sequenciais e posição "
        "exata. Sem token/var/multivar batendo, é simplesmente desconhecido. Qualquer usuário "
        "pode produzir uma opinião (só texto+emoção). **Só a criadora pode definir contexto e "
        "pensamento** — nunca qualquer usuário — convertendo a opinião em dado concreto. Ao "
        "salvar uma opinião, texto (E) e emoção (RE) ganham marcadores reais; contexto (CE) e "
        "pensamento (PIDE) ficam travados no placeholder `{IM}.0` até a criadora confirmar.\n"
        "6. **Espelhamento e Análise** — em vez de um score único, compara a entrada nova "
        "contra um bloco de referência eixo por eixo (emoção, texto, contexto, pensamento), "
        "sempre por igualdade EXATA. Texto batendo com emoção divergindo é sinal de tom/"
        "sarcasmo (pela própria Camada 4 já não é a mesma chave). Contexto divergindo é "
        "assunto genuinamente diferente. Pensamento batendo é necessário mas nunca "
        "suficiente — a criadora sempre revisa antes de virar dado concreto. É assim que o "
        "Adam \"aprende com a interação\" sem nunca aproximar: cada diagnóstico que ela "
        "resolve vira registro exato novo."
    )

    st.subheader("Como isso se conecta ao lovely.py")
    st.markdown(
        "- **ALNULU** foi feito pra alimentar o treino do modelo PyTorch e pra comparar "
        "FORMA/ortografia (Camada 5) — usá-lo como medida de significado (cosseno sobre "
        "o texto inteiro) foi um erro corrigido: gerava falso-positivo grosseiro "
        "(\"feliz\"/\"triste\" batendo 89%).\n"
        "- **Vars/Multivars** só existem pra texto e emoção. Contexto e pensamento nunca "
        "têm var — por isso são o fingerprint confiável, e por isso assuntos divergentes "
        "(\"lua\"/\"noite\") nunca devem ser aproximados.\n"
        "- O **bloco** em `Adam_Lovely_memory.json` é literalmente a árvore da Camada 3: "
        "um pensamento, um ramo de entrada, um ramo de saída.\n"
        "- **Ainda em aberto**: o classificador/modelo de verdade nunca é chamado no chat "
        "ao vivo hoje — tudo roda em cima de heurísticas de similaridade (corrigidas, mas "
        "ainda uma ponte temporária); e como reconectar as Camadas 1-5 de volta no "
        "`lovely.py` no lugar desses remendos."
    )

    st.caption(
        "Princípio geral: o processo do INSEPA deve ser respeitado camada por camada, "
        "não substituído por atalhos de similaridade genérica."
    )

if camada == "Camada 1 — Marcadores Únicos":
    st.header("Camada 1 — Marcadores Únicos")
    st.markdown(
        "Cada token (palavra ou pontuação) de uma entrada ganha um marcador que "
        "**nunca se repete** fora daquela entrada. A mesma palavra em frases "
        "diferentes recebe marcadores diferentes — isso é o que impede o sistema "
        "de confundir o significado dela."
    )

    if "cdf_ultimo_marcador" not in st.session_state:
        st.session_state.cdf_ultimo_marcador = "0.0"
    if "cdf_historico" not in st.session_state:
        st.session_state.cdf_historico = []

    with st.form("form_marcar", clear_on_submit=True):
        entrada = st.text_input("Digite uma frase para marcar", placeholder="A noite era brilhante.")
        enviar = st.form_submit_button("Marcar")

    if enviar and entrada.strip():
        marcadores, novo_ultimo = marcar_entrada(entrada, st.session_state.cdf_ultimo_marcador)
        st.session_state.cdf_ultimo_marcador = novo_ultimo
        st.session_state.cdf_historico.append((entrada, marcadores))

    if st.session_state.cdf_historico:
        st.subheader("Sequência marcada")
        for texto, marcadores in st.session_state.cdf_historico:
            st.markdown(f"**\"{texto}\"**")
            st.code(formatar(marcadores))

        st.caption(
            "Repare: mesmo frases que compartilham palavras nunca dividem "
            "marcador — cada entrada tem sua própria faixa numérica."
        )

    if st.button("🔄 Reiniciar sequência"):
        st.session_state.cdf_ultimo_marcador = "0.0"
        st.session_state.cdf_historico = []
        st.rerun()

elif camada == "Camada 2 — Classificação":
    st.header("Camada 2 — Classificação")
    st.markdown(
        "Aqui não inventamos nada, só **classificamos** o que o texto já carrega:\n\n"
        "1. Todo texto tem uma emoção — mesmo a ausência dela é uma emoção (frieza).\n"
        "2. A emoção pode ser **explícita** (emoji/reação junto) ou **oculta** "
        "(sem emoji, mas uma *var* de palavra no texto revela ela).\n"
        "3. Toda frase carrega um contexto.\n"
        "4. Texto + emoção + contexto conectados geram um pensamento: uma breve "
        "explicação da situação."
    )

    with st.expander("📖 Léxico de vars usado nesta demo (emoji → palavras que o sinalizam)"):
        for emoji, vars_lista in LEXICO_EMOCOES.items():
            st.write(f"**{emoji}** ← {', '.join(vars_lista)}")
        st.caption(
            "Uma var vale tanto escrita no campo de reação (por extenso) quanto "
            "escondida dentro do próprio texto — as duas contam como o mesmo sinal."
        )

    with st.form("form_classificar"):
        texto_c2 = st.text_input("Texto", placeholder="Uma dama estava na beira do mar, e ofereceu um sorriso suave.")
        reacao_c2 = st.text_input("Reação (emoji OU a var por extenso, ex.: \"sorriso fechado\")", placeholder="^^")
        contexto_c2 = st.text_input("Contexto (opcional)", placeholder="Literatura poética")
        pensamento_c2 = st.text_input("Pensamento interno (opcional)", placeholder="Esta é uma frase retirada do conto A dama na areia")
        classificar = st.form_submit_button("Classificar")

    if classificar and texto_c2.strip():
        marcadores_c2, _ = marcar_entrada(texto_c2, "0.0")
        resultado_emocao = classificar_emocao(texto_c2, reacao_c2.strip())
        resultado_contexto = classificar_contexto(contexto_c2.strip() or None)

        st.subheader("Decomposição em camadas")

        st.markdown("**Camada 1 — marcadores**")
        st.code(formatar(marcadores_c2))

        st.markdown("**Texto livre**")
        st.write(texto_c2)

        st.markdown("**Emoção**")
        if resultado_emocao["tipo"] == "explícita":
            st.success(f"Explícita: {resultado_emocao['emocao']}")
        elif resultado_emocao["tipo"] == "oculta":
            st.info(
                f"{resultado_emocao['var_detectada']} (ou {resultado_emocao['emocao']}) — "
                f"detectada via var no campo de {resultado_emocao['origem']}"
            )
        else:
            st.warning("Nenhum emoji nem var reconhecida — classificada como **frieza**.")

        st.markdown("**Contexto**")
        if resultado_contexto["tipo"] == "declarado":
            st.write(resultado_contexto["contexto"])
        else:
            st.write("Presente, porém não nomeado nesta entrada.")

        st.markdown("**Pensamento interno**")
        st.write(pensamento_c2.strip() or "Não informado nesta entrada.")

elif camada == "Camada 3 — Integração":
    st.header("Camada 3 — Integração")
    st.markdown(
        "O **pensamento** é a árvore — o tronco. Dele nascem dois ramos, cada "
        "um com seu próprio texto + emoção + contexto:\n\n"
        "- **Ramo entrada**: texto e emoção enviados pelo usuário; contexto inferido do input.\n"
        "- **Ramo saída**: texto e emoção devolvidos pelo Adam; contexto próprio, associado ao da entrada.\n\n"
        "Os dois ramos ressoam entre si por compartilharem a mesma raiz — não porque o "
        "texto do contexto de um pareça com o do outro. Contexto de entrada e de saída são "
        "escritos em vocabulários diferentes por convenção, e isso é esperado."
    )

    with st.form("form_integrar"):
        pensamento_c3 = st.text_input("Pensamento interno (a raiz)", placeholder="Um novo começo, sem passado imposto")
        st.markdown("**Ramo entrada**")
        col_e1, col_e2, col_e3 = st.columns(3)
        entrada_texto_c3 = col_e1.text_input("Texto (entrada)", placeholder="Olá")
        entrada_emocao_c3 = col_e2.text_input("Emoção (entrada)", placeholder="😊")
        entrada_contexto_c3 = col_e3.text_input("Contexto (entrada)", placeholder="Saudação inicial")
        st.markdown("**Ramo saída**")
        col_s1, col_s2, col_s3 = st.columns(3)
        saida_texto_c3 = col_s1.text_input("Texto (saída)", placeholder="Olá! Sou o Adam.")
        saida_emocao_c3 = col_s2.text_input("Emoção (saída)", placeholder="😊")
        saida_contexto_c3 = col_s3.text_input("Contexto (saída)", placeholder="Resposta de apresentação")
        integrar = st.form_submit_button("Integrar")

    if integrar and pensamento_c3.strip() and entrada_texto_c3.strip() and saida_texto_c3.strip():
        bloco = montar_bloco(
            pensamento_c3, entrada_texto_c3, entrada_emocao_c3, entrada_contexto_c3,
            saida_texto_c3, saida_emocao_c3, saida_contexto_c3,
        )

        st.subheader("Árvore integrada")
        st.markdown(f"🌳 **Pensamento (raiz):** {bloco.pensamento}")

        col_ramo_e, col_ramo_s = st.columns(2)
        with col_ramo_e:
            st.markdown("🌿 **Ramo entrada**")
            st.write(f"Texto: {bloco.entrada.texto}")
            st.write(f"Emoção: {bloco.entrada.emocao or '(nenhuma informada)'}")
            st.write(f"Contexto: {bloco.entrada.contexto or '(nenhum informado)'}")
            marcadores_e, _ = marcar_entrada(bloco.entrada.texto, "0.0")
            st.code(formatar(marcadores_e))
        with col_ramo_s:
            st.markdown("🌿 **Ramo saída**")
            st.write(f"Texto: {bloco.saida.texto}")
            st.write(f"Emoção: {bloco.saida.emocao or '(nenhuma informada)'}")
            st.write(f"Contexto: {bloco.saida.contexto or '(nenhum informado)'}")
            marcadores_s, _ = marcar_entrada(bloco.saida.texto, "0.0")
            st.code(formatar(marcadores_s))

        st.caption(
            "Os dois ramos ressoam por saírem da mesma raiz (o pensamento) — "
            "repare que os contextos são escritos diferente de propósito."
        )

elif camada == "Camada 4 — Hashrização":
    st.header("Camada 4 — Hashrização")
    st.markdown(
        "A junção dos marcadores de uma entrada funciona como uma **chave**. "
        "Só a chave exata destranca a porta pra saída correspondente — é um "
        "processo de segurança de dados, não de adivinhação.\n\n"
        "**A chave precisa ser completa: texto + emoção juntos** (a emoção, explícita "
        "por emoji ou oculta por var — é aí que entra a similaridade da Camada 2). "
        "**Contexto não entra na chave** — ele é inferido depois, por ressonância "
        "(Camada 3), não por hash.\n\n"
        "**IM (Índice Mãe)** é o número antes do ponto (o `0` de `0.25`) — o mesmo "
        "pra todos os blocos de um universo. **IF (Índice Filho)** é o número depois "
        "do ponto — cada um único, herdeiro daquele IM."
    )
    st.info(
        "🔢 A sequência de marcadores é sempre estritamente **sequencial e crescente** "
        "dentro de um mesmo IM — nunca reinicia, nunca reusa uma posição já ocupada "
        "(o índice filho pode crescer até onde precisar, ex.: 0.99999999999...). É por "
        "isso que a chave funciona como identidade segura: dentro do histórico real de "
        "um universo, duas entradas diferentes nunca ocupam a mesma posição, porque a "
        "posição só anda pra frente. A demo abaixo reinicia a contagem em '0.0' a cada "
        "teste só pra facilitar a comparação isolada — no domínio real (dentro do "
        "lovely.py) a contagem nunca reinicia."
    )

    def marcar_texto_emocao(texto: str, emocao: str, ultimo: str):
        """A chave completa é texto + emoção marcados em sequência (o contexto
        fica de fora -- ele é inferido, não faz parte do hash)."""
        marcadores, ultimo = marcar_entrada(texto, ultimo)
        if emocao.strip():
            marcadores_emocao, ultimo = marcar_entrada(emocao, ultimo)
            marcadores += marcadores_emocao
        return marcadores, ultimo

    if "cdf_cofre" not in st.session_state:
        cofre = Cofre()
        # Semente: o bloco de apresentação do Adam.
        marcadores_entrada_seed, ultimo = marcar_texto_emocao("Olá", "😊", "0.0")
        marcadores_saida_seed, ultimo = marcar_texto_emocao("Olá! Sou o Adam.", "😊", ultimo)
        entrada_tokens_seed = [m for m, _ in marcadores_entrada_seed]
        saida_tokens_seed = [m for m, _ in marcadores_saida_seed]
        cofre.trancar(entrada_tokens_seed, saida_tokens_seed)
        st.session_state.cdf_cofre = cofre
        st.session_state.cdf_cofre_textos = {
            chave_hash(entrada_tokens_seed): ("Olá 😊", "Olá! Sou o Adam. 😊")
        }
        st.session_state.cdf_cofre_ultimo = ultimo  # continua a sequência sem nunca reusar número

    with st.expander("🔑 Portas já trancadas no cofre"):
        for chave, (txt_e, txt_s) in st.session_state.cdf_cofre_textos.items():
            st.write(f"IM **{im_de(chave.split(',')[0])}** — \"{txt_e}\" → \"{txt_s}\"")
            st.code(chave, language=None)

    st.subheader("Testar uma chave")
    col_te, col_ee = st.columns(2)
    entrada_c4 = col_te.text_input("Texto (do zero da sequência, IM 0)", placeholder="Olá")
    emocao_c4 = col_ee.text_input("Emoção (emoji ou var por extenso)", placeholder="😊")
    if st.button("Tentar destrancar"):
        if entrada_c4.strip():
            marcadores_c4, _ = marcar_texto_emocao(entrada_c4, emocao_c4, "0.0")
            marcadores_ids = [m for m, _ in marcadores_c4]
            st.code(chave_hash(marcadores_ids), language=None)
            saida_encontrada = st.session_state.cdf_cofre.destrancar(marcadores_ids)
            if saida_encontrada:
                st.success(f"🔓 Chave bateu! Porta destrancada: marcadores {saida_encontrada[0]}...{saida_encontrada[-1]}")
            else:
                st.warning("🔒 Nenhuma porta bate com essa chave exata (texto e/ou emoção diferentes).")

    st.subheader("Trancar uma nova porta")
    with st.form("form_trancar", clear_on_submit=True):
        col1, col2 = st.columns(2)
        nova_entrada = col1.text_input("Texto de entrada", placeholder="Boa noite")
        nova_emocao_e = col2.text_input("Emoção de entrada", placeholder="🌙")
        col3, col4 = st.columns(2)
        nova_saida = col3.text_input("Texto de saída", placeholder="Boa noite pra você também.")
        nova_emocao_s = col4.text_input("Emoção de saída", placeholder="✨")
        trancar = st.form_submit_button("Trancar nova porta")

    if trancar and nova_entrada.strip() and nova_saida.strip():
        marcadores_e, ultimo_novo = marcar_texto_emocao(nova_entrada, nova_emocao_e, st.session_state.cdf_cofre_ultimo)
        marcadores_s, ultimo_novo = marcar_texto_emocao(nova_saida, nova_emocao_s, ultimo_novo)
        st.session_state.cdf_cofre_ultimo = ultimo_novo
        tokens_e = [m for m, _ in marcadores_e]
        tokens_s = [m for m, _ in marcadores_s]
        st.session_state.cdf_cofre.trancar(tokens_e, tokens_s)
        st.session_state.cdf_cofre_textos[chave_hash(tokens_e)] = (
            f"{nova_entrada} {nova_emocao_e}".strip(),
            f"{nova_saida} {nova_emocao_s}".strip(),
        )
        st.rerun()

elif camada == "Camada 5 — Aprendizado Seguro":
    st.header("Camada 5 — Aprendizado Seguro")
    st.markdown(
        "O ALNULU só deixa o Adam **ver a forma** de uma palavra (virar número). Reconhecer "
        "não é uma questão de aproximação/distância — é o INSEPA que decide, por **registro "
        "exato**: a palavra já é um token conhecido, ou é uma var/multivar já cadastrada "
        "DENTRO de um bloco. IA comum trabalha com embeddings e similaridade de vetor; o "
        "Adam trabalha com algarismos sequenciais e posição exata. Sem meio-termo por forma.\n\n"
        "Qualquer usuário pode produzir uma **opinião** (texto + emoção) — só isso. "
        "**Contexto e pensamento interno só podem ser definidos pela criadora** — nunca "
        "por qualquer usuário. Ao salvar uma opinião: texto (E) e emoção (RE) ganham "
        "marcadores reais; contexto (CE) e pensamento (PIDE) ficam travados no "
        "placeholder `{IM}.0` até a criadora (e só ela) confirmar."
    )

    registro_c5 = {"preto": ["ônix", "cor de carvão"]}
    st.caption(f"Exemplo (IM 97 FELINOS, bloco do gato preto) — registro exato: preto ← {', '.join(registro_c5['preto'])}")

    if "cdf_opinioes" not in st.session_state:
        st.session_state.cdf_opinioes = []
    if "cdf_c5_ultimo" not in st.session_state:
        st.session_state.cdf_c5_ultimo = "0.0"

    st.subheader("👤 Usuário comum")
    st.caption("Só pode entregar texto + emoção. Não tem campo de contexto nem de pensamento aqui.")

    termo_novo_c5 = st.text_input("Palavra pra testar o reconhecimento", placeholder="ônix")
    if termo_novo_c5.strip():
        resolvido = resolver_termo(termo_novo_c5.strip(), registro_c5)
        if resolvido:
            st.success(f"✅ Reconhecido: \"{termo_novo_c5}\" resolve pro token \"{resolvido}\" (registro exato, não aproximação).")
        else:
            st.write("Desconhecido — não é o token nem nenhuma var/multivar registrada. Vira opinião, sem chute.")

    with st.form("form_opiniao", clear_on_submit=True):
        im_c5 = st.text_input("IM (universo)", value="50")
        texto_c5 = st.text_input("Texto", placeholder="Uma nevasca assolou aquela região")
        emocao_c5 = st.text_input("Emoção", placeholder="😨")
        enviar_opiniao = st.form_submit_button("Enviar opinião")

    if enviar_opiniao and texto_c5.strip() and emocao_c5.strip():
        resultado = marcar_opiniao(im_c5.strip() or "0", texto_c5, emocao_c5, st.session_state.cdf_c5_ultimo)
        st.session_state.cdf_c5_ultimo = resultado["ultimo"]
        st.session_state.cdf_opinioes.append({
            "im": im_c5.strip() or "0", "texto": texto_c5, "emocao": emocao_c5,
            "E": resultado["E"], "RE": resultado["RE"],
            "CE": resultado["CE"], "PIDE": resultado["PIDE"],
            "contexto": None, "pensamento": None,
        })
        st.success("💬 Opinião registrada — contexto e pensamento travados até a criadora revisar.")
        st.rerun()

    st.divider()
    st.subheader("🔑 Só a criadora")
    sou_criadora = st.checkbox("Sou a criadora (gate de demonstração — no lovely.py real isso é a senha de admin)")

    pendentes = [o for o in st.session_state.cdf_opinioes if o["contexto"] is None]
    if not sou_criadora:
        st.caption(f"{len(pendentes)} opinião(ões) esperando revisão — só a criadora enxerga os detalhes.")
    else:
        if not pendentes:
            st.write("Nenhuma opinião pendente.")
        for i, op in enumerate(pendentes):
            with st.expander(f"\"{op['texto']}\" {op['emocao']} (IM {op['im']})"):
                st.code(f"E: {op['E']}\nRE: {op['RE']}\nCE: {op['CE']}  ← placeholder\nPIDE: {op['PIDE']}  ← placeholder", language=None)
                novo_contexto = st.text_input("Contexto", key=f"ctx_{i}")
                novo_pensamento = st.text_input("Pensamento interno", key=f"pide_{i}")
                if st.button("Confirmar como dado concreto", key=f"confirmar_{i}"):
                    if novo_contexto.strip() and novo_pensamento.strip():
                        marcadores_ce, ultimo_novo = marcar_entrada(novo_contexto, st.session_state.cdf_c5_ultimo)
                        marcadores_pide, ultimo_novo = marcar_entrada(novo_pensamento, ultimo_novo)
                        st.session_state.cdf_c5_ultimo = ultimo_novo
                        op["contexto"] = novo_contexto
                        op["pensamento"] = novo_pensamento
                        op["CE"] = [m for m, _ in marcadores_ce]
                        op["PIDE"] = [m for m, _ in marcadores_pide]
                        st.success("✅ Virou dado concreto — placeholder substituído por marcadores reais.")
                        st.rerun()
                    else:
                        st.error("Precisa preencher os dois pra virar dado concreto.")

elif camada == "Camada 6 — Espelhamento e Análise":
    st.header("Camada 6 — Espelhamento e Análise")
    st.markdown(
        "Em vez de devolver um único número de confiança, o Adam **espelha** a entrada "
        "nova contra um bloco de referência, eixo por eixo (emoção, texto, contexto, "
        "pensamento) -- sempre por igualdade EXATA de tokens, nunca por distância. Cada "
        "combinação de bate/diverge tem um significado próprio, não só \"mais\" ou "
        "\"menos parecido\":\n\n"
        "- **texto bate, emoção diverge** -- mesma frase, carga emocional diferente: "
        "sinal de tom/ironia/sarcasmo.\n"
        "- **contexto diverge** -- mesmo que o resto bata, é assunto genuinamente "
        "diferente (contexto não tem var, não existe forma equivalente).\n"
        "- **pensamento bate, mas a entrada traz algo novo** -- bater na raiz é "
        "necessário, mas nunca suficiente: a criadora sempre revisa antes de virar dado "
        "concreto."
    )

    bloco_ref = {
        "texto": "Você gosta de estudar?",
        "reacao": "😊",
        "contexto": "Pergunta sobre hábitos de estudo",
        "pensamento_interno": "Curiosidade genuína sobre o que motiva a pessoa a aprender.",
    }
    st.caption(
        f"Bloco de referência fixo pra este exemplo: texto=\"{bloco_ref['texto']}\" "
        f"{bloco_ref['reacao']} — contexto \"{bloco_ref['contexto']}\" — pensamento "
        f"\"{bloco_ref['pensamento_interno']}\""
    )

    st.subheader("Cenários rápidos (iguais aos que a Thaís descreveu)")
    cenarios = {
        "Emoção diverge": {"texto": "Você gosta de estudar?", "emocao": "😠", "contexto": "", "pensamento": ""},
        "Sarcasmo (texto bate, emoção diverge)": {"texto": "Você gosta de estudar?", "emocao": "🙄", "contexto": "", "pensamento": ""},
        "Contexto diverge": {"texto": "Você gosta de estudar?", "emocao": "😊", "contexto": "Pergunta sobre lazer no fim de semana", "pensamento": ""},
        "Pensamento bate, conteúdo novo": {"texto": "Você gosta de estudar?", "emocao": "😊", "contexto": "Pergunta sobre hábitos de estudo", "pensamento": "Curiosidade genuína sobre o que motiva a pessoa a aprender."},
    }
    cols = st.columns(4)
    for col, (nome, valores) in zip(cols, cenarios.items()):
        if col.button(nome, key=f"cenario_{nome}"):
            st.session_state.cdf_c6_texto = valores["texto"]
            st.session_state.cdf_c6_emocao = valores["emocao"]
            st.session_state.cdf_c6_contexto = valores["contexto"]
            st.session_state.cdf_c6_pensamento = valores["pensamento"]

    st.subheader("Entrada nova")
    texto_novo = st.text_input("Texto", value=st.session_state.get("cdf_c6_texto", "Você gosta de estudar?"), key="cdf_c6_texto")
    emocao_nova = st.text_input("Emoção", value=st.session_state.get("cdf_c6_emocao", "😊"), key="cdf_c6_emocao")
    contexto_novo = st.text_input("Contexto (opcional -- vazio se ainda não inferido)", value=st.session_state.get("cdf_c6_contexto", ""), key="cdf_c6_contexto")
    pensamento_novo = st.text_input("Pensamento (opcional -- vazio se ainda não inferido)", value=st.session_state.get("cdf_c6_pensamento", ""), key="cdf_c6_pensamento")

    if st.button("🪞 Espelhar contra o bloco de referência"):
        entrada_nova = {"texto": texto_novo, "emocao": emocao_nova, "contexto": contexto_novo, "pensamento": pensamento_novo}
        diagnostico = espelhar(entrada_nova, bloco_ref)

        rotulo = {"bate": "✅ bate", "diverge": "⚠️ diverge", "desconhecido": "❔ desconhecido", "vazio": "— vazio"}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Emoção", rotulo[diagnostico["emocao"]])
        c2.metric("Texto", rotulo[diagnostico["texto"]])
        c3.metric("Contexto", rotulo[diagnostico["contexto"]])
        c4.metric("Pensamento", rotulo[diagnostico["pensamento"]])

        if diagnostico["sinal_tom_diferente"]:
            st.warning(
                "🎭 Texto idêntico, emoção diferente -- pela Camada 4 isso já não é a "
                "mesma chave (texto+emoção juntos). Pode ser ironia/sarcasmo: mesma "
                "frase, carga emocional outra."
            )
        if diagnostico["raiz_compartilhada"]:
            st.info(
                "🌱 O pensamento bate com a raiz do bloco de referência -- mas isso é "
                "necessário, nunca suficiente. Só vira dado concreto com revisão da "
                "criadora."
            )
        st.error("🔒 Precisa da criadora: nenhuma combinação de eixos confirma um dado concreto sozinha (Camada 5).")

elif camada == "🧮 Caixa de Ferramentas (sugestões)":
    st.header("🧮 Caixa de Ferramentas (sugestões)")
    st.markdown(
        "Uma lista de bibliotecas da própria standard library do Python que o Adam pode "
        "**conhecer** como apoio pra suas métricas (tipo os scores de confiança que já "
        "aparecem no chat: `melhor_candidato_fraco`, `alnulu_token_similarity`, "
        "`corpus_similarity_score`). São só sugestões de leitura sobre números que já "
        "existem — **nenhuma delas mexe no corpus, no Codex ou nos valores que o Adam "
        "deve aprender**, e nenhuma é importada/executada pelo Adam sozinho. A decisão de "
        "ligar qualquer uma delas de verdade dentro do lovely.py continua sendo sempre da "
        "criadora."
    )

    st.subheader("statistics — resumir um conjunto de scores")
    st.caption(
        "Útil pra descrever a distribuição de confiança de várias opiniões numa noite "
        "(média, mediana, desvio-padrão) sem alterar nenhum dado — só leitura."
    )
    import statistics
    scores_exemplo = [0.08, 1.0, 0.62, 0.91, 0.15]
    st.code(
        f"scores = {scores_exemplo}\n"
        f"statistics.mean(scores)   -> {statistics.mean(scores):.3f}\n"
        f"statistics.median(scores) -> {statistics.median(scores):.3f}\n"
        f"statistics.pstdev(scores) -> {statistics.pstdev(scores):.3f}",
        language="python",
    )

    st.subheader("collections.Counter — contar sem escrever")
    st.caption(
        "Útil pra ver quais emoções/reações mais aparecem num IM, por exemplo — pura "
        "contagem de leitura, nunca grava nada de volta no corpus."
    )
    from collections import Counter
    emocoes_exemplo = ["😊", "😨", "😊", "🤔", "😊", "😨"]
    st.code(
        f"emocoes = {emocoes_exemplo}\n"
        f"Counter(emocoes) -> {dict(Counter(emocoes_exemplo))}",
        language="python",
    )

    st.subheader("math — operações numéricas básicas")
    st.caption(
        "Já apoia indiretamente normalizações de score (ex.: raiz quadrada, logaritmo) "
        "sem precisar reinventar a conta na mão."
    )
    import math
    st.code(
        f"math.sqrt(0.62)  -> {math.sqrt(0.62):.3f}\n"
        f"math.log(0.62)   -> {math.log(0.62):.3f}",
        language="python",
    )

    st.divider()
    st.info(
        "Fronteira: essas ferramentas leem/resumem números que o INSEPA já produziu "
        "(scores, contagens). Nenhuma delas participa da Camada 2 (classificação), "
        "Camada 3 (integração) ou Camada 5 (registro exato/opinião) -- são apoio de "
        "leitura pra criadora, não um novo caminho de aprendizado pro Adam."
    )
