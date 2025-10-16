import streamlit as st
import json
import os
import subprocess

# =============================================================================
# GUIA PARA MODIFICAÇÕES NO SISTEMA INSEPA (PARA O "EU DO FUTURO")
# =============================================================================
# Este app processa blocos de conversa AI, extrai texto limpo, tokeniza com TEXE sequenciais,
# aplica features CAE/Muden delimitadas por tags, e gera JSON estruturado.
#
# ESTRUTURA GERAL:
# - Parsing: Divide blocos em seções (Entrada, Saída, etc.) usando .find() para marcadores.
# - Separação: Usa "—" para fala e split('\n', 1) para reações/contexto.
# - Tokenização: Cria números sequenciais (ex.: 1.1) baseados no IM; aplica features apenas entre tags iguais.
# - JSON: Monta estrutura IDA/IM/blocos com ponto_neutro para vazios.
#
# ONDE MEXER PARA ADICIONAR/ALTERAR:
# 1. NOVOS CAMPOS NO BLOCO: Adicione .find() no parsing (~linha 150) e ajuste parts/offsets.
#    Ex.: novo_campo_start = bloco_txt.find("Novo Campo:")
#    Depois, adicione em parts e no bloco_data (~linha 300).
#
# 2. NOVOS MAPEAMENTOS CAE: Edite o dict 'mapping' na função extrair_texto_e_features (~linha 220).
#    Ex.: "altura": "Altura"
#
# 3. NOVAS CATEGORIAS DE FEATURES: Inclua na lista if categoria not in ['CAE', 'Muden', 'NovaCat'] (~linha 210).
#
# 4. TOKENIZAR OUTROS TEXTOS: Copie a lógica de tokenização (~linha 260) e ajuste parts.get().
#    Ex.: Para FADEN (Fala de Entrada), copie após tokenização do texto_inicial.
#
# 5. NOVOS CAMPOS NO JSON: Adicione no bloco_data (~linha 290) e no display_html (~linha 350).
#
# DICAS:
# - Sempre teste com blocos de exemplo após mudanças.
# - Use ponto_neutro (f"{im_numero}.0") para campos vazios.
# - Normalização de tags ignora case/pontuação para matching robusto.
# - Se quebrar, verifique offsets no parsing e posições em secoes.
# - IMPORTANTE: Na contagem inicial de tokens (total_tokens_no_im), inclua TODOS os tipos de tokens salvos (TEXE, FADEN, etc.) para evitar gaps na numeração sequencial entre blocos.
#
# =============================================================================

st.title("Sistema INSEPA")

# Botões de manutenção para reiniciar/limpar
with st.expander("Manutenção do Servidor (Clique para expandir)"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Limpar Cache Interno e Recarregar"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache interno limpo! Recarregando...")
            st.rerun()
    with col2:
        if st.button("Limpar Cache do Streamlit"):
            try:
                result = subprocess.run(["streamlit", "cache", "clear"], capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    st.success("Cache do Streamlit limpo com sucesso!")
                else:
                    st.error(f"Erro ao limpar cache: {result.stderr}")
            except Exception as e:
                st.error(f"Erro ao executar comando: {str(e)}")
    with col3:
        if st.button("Reiniciar Servidor (Experimental)"):
            st.warning(
                "Este botão tenta reiniciar o servidor, mas pode não funcionar perfeitamente. Use o terminal se necessário.")
            try:
                # Tenta iniciar um novo processo do Streamlit
                subprocess.Popen(["streamlit", "run", "app.py"], shell=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                st.success("Novo servidor iniciado! Feche esta aba e recarregue em uma nova.")
            except Exception as e:
                st.error(
                    f"Erro ao reiniciar: {str(e)}. Use Ctrl+C no terminal para parar e execute 'streamlit run app.py'.")
    with col4:
        if st.button("Instruções para Reiniciar Servidor"):
            st.info(
                "Para reiniciar o servidor completamente:\n1. Pare o servidor (Ctrl+C no terminal).\n2. Execute: streamlit run app.py\n3. Recarregue a página no navegador (Ctrl+F5).")

# Carregar IMs salvos do arquivo
ims_file = "ims.json"
if os.path.exists(ims_file):
    with open(ims_file, 'r', encoding='utf-8') as f:
        ims_salvos = json.load(f)
else:
    ims_salvos = {}

# Migrar blocos antigos: converter 'TEXE:' string para 'TEXE' lista
for im_key in ims_salvos:
    for bloco in ims_salvos[im_key]['blocos']:
        entrada = bloco.get('Entrada', {})
        texto = entrada.get('TEXTO Inicial DE ENTRADA', {})
        if 'TEXE:' in texto and isinstance(texto['TEXE:'], str):
            lines = texto['TEXE:'].split('\n')
            texe_list = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    item = {}
                    for part in parts:
                        if ': ' in part:
                            key, value = part.split(': ', 1)
                            if key == 'TEXE':
                                item['TEXE'] = value
                            elif key == 'token':
                                item['t'] = value
                            elif key == 'vars':
                                item['vars'] = eval(value)
                            else:
                                item[key] = eval(value)
                    texe_list.append(item)
            texto['TEXE'] = texe_list
            del texto['TEXE:']
    # Salvar após migração
    with open(ims_file, 'w', encoding='utf-8') as f:
        json.dump(ims_salvos, f, ensure_ascii=False, indent=4)

# IDA é fixo, IM editável
st.write("**IDA (Inconsciente do Adam):** IDA")
ida_value = "IDA"
im_numero = st.number_input("Número do IM", value=0, min_value=0)
im_nome = st.text_input("Nome do IM", value="-")

# Mostrar IMs salvos
if ims_salvos:
    st.write("**IMs Salvos:**")
    for num, data in ims_salvos.items():
        st.write(f"- IM {num}: {data['nome']} ({len(data.get('blocos', []))} blocos)")

        # Expander para mostrar os blocos do IM
        with st.expander(f"Ver blocos do IM {num}"):
            blocos = data.get('blocos', [])
            if blocos:
                saved_display_html = ""
                for bloco in blocos:
                    bloco_id = bloco['bloco_id']
                    entrada = bloco.get('Entrada', {})
                    saida = bloco.get('Saída', {})
                    pensamento = bloco.get('Pensamento Interno', '')

                    # Extrair dados para display
                    texe_list = entrada.get('TEXTO Inicial DE ENTRADA', {}).get('TEXE', [])
                    hash_texe = entrada.get('TEXTO Inicial DE ENTRADA', {}).get('hash_TEXE', '')
                    texto_inicial = entrada.get('TEXTO Inicial DE ENTRADA', {}).get('Texto', '')
                    faden_list = entrada.get('Fala DE ENTRADA', {}).get('FADEN', [])
                    hash_faden = entrada.get('Fala DE ENTRADA', {}).get('hash_FADEN', '')
                    fala_entrada = entrada.get('Fala DE ENTRADA', {}).get('Texto', '')
                    re_list = entrada.get('Reação', {}).get('RE', [])
                    hash_re = entrada.get('Reação', {}).get('hash_RE', '')
                    reacao_texto = entrada.get('Reação', {}).get('Texto', '')
                    texto_final_entrada = entrada.get('Texto Final de Entrada', '')
                    contexto_entrada = entrada.get('Contexto', '')
                    texto_inicial_saida = saida.get('Texto Inicial de SAÍDA', '')
                    fala_saida = saida.get('Fala de Saída', '')
                    reacao_saida = saida.get('Reação de Saída', '')
                    texto_final_saida = saida.get('Texto Final de Saída', '')
                    contexto_saida = saida.get('Contexto de Saída', '')

                    saved_display_html += f"""
                    <h4>Bloco: {bloco_id}</h4>
                    <h4>Entrada:</h4>
                    <p><strong>TEXTO Inicial de Entrada:</strong> {(texto_inicial or '').replace('\n', '<br>')}</p>
                    <p><strong>TEXE:</strong><br>{'<br>'.join([f'TEXE: {item["TEXE"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["TEXE", "t", "vars"]]) if any(k not in ["TEXE", "t", "vars"] for k in item) else '') for item in texe_list])}<br><strong>Hash TEXE:</strong> {hash_texe}</p>
                    <p><strong>Fala de Entrada:</strong> {(fala_entrada or '').replace('\n', '<br>')}</p>
                    <p><strong>FADEN:</strong><br>{'<br>'.join([f'FADEN: {item["FADEN"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["FADEN", "t", "vars"]]) if any(k not in ["FADEN", "t", "vars"] for k in item) else '') for item in faden_list])}<br><strong>Hash FADEN:</strong> {hash_faden}</p>
                    <p><strong>Reação:</strong> {(reacao_texto or '').replace('\n', '<br>')}</p>
                    <p><strong>RE:</strong><br>{'<br>'.join([f'RE: {item["RE"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["RE", "t", "vars"]]) if any(k not in ["RE", "t", "vars"] for k in item) else '') for item in re_list])}<br><strong>Hash RE:</strong> {hash_re}</p>
                    <p><strong>Texto Final de Entrada:</strong> {(texto_final_entrada or '').replace('\n', '<br>')}</p>
                    <p><strong>Contexto:</strong> {contexto_entrada}</p>
                    <p><strong>Pensamento interno:</strong> {(pensamento or '').replace('\n', '<br>')}</p>
                    <h4>Saída:</h4>
                    <p><strong>Texto Inicial de Saída:</strong> {(texto_inicial_saida or '').replace('\n', '<br>')}</p>
                    <p><strong>Fala de Saída:</strong> {(fala_saida or '').replace('\n', '<br>')}</p>
                    <p><strong>Reação:</strong> {reacao_saida}</p>
                    <p><strong>Texto Final de Saída:</strong> {(texto_final_saida or '').replace('\n', '<br>')}</p>
                    <p><strong>Contexto:</strong> {contexto_saida}</p>
                    <hr>
                    """
                st.markdown(saved_display_html, unsafe_allow_html=True)
            else:
                st.write("Nenhum bloco salvo para este IM.")

    # Opção para remover IM
    im_to_remove = st.selectbox("Selecione IM para remover", list(ims_salvos.keys()), key="remove_select")
    if st.button("Remover IM Selecionado"):
        if im_to_remove in ims_salvos:
            del ims_salvos[im_to_remove]
            with open(ims_file, 'w', encoding='utf-8') as f:
                json.dump(ims_salvos, f, ensure_ascii=False, indent=4)
            st.success(f"IM {im_to_remove} removido!")
            st.rerun()
else:
    st.write("**Nenhum IM salvo ainda.**")

if st.button("Salvar IM Atual"):
    if str(im_numero) not in ims_salvos:
        ims_salvos[str(im_numero)] = {"nome": im_nome, "blocos": []}
        with open(ims_file, 'w', encoding='utf-8') as f:
            json.dump(ims_salvos, f, ensure_ascii=False, indent=4)
        st.success(f"IM {im_numero} criado vazio!")
    else:
        st.warning(f"IM {im_numero} já existe. Use para adicionar blocos.")
    st.rerun()

# Botão para baixar todos os IMs salvos
if ims_salvos:
    all_ims_data = json.dumps({ida_value: {"IM": ims_salvos}}, ensure_ascii=False, indent=4)
    st.download_button(
        label="Baixar JSON Completo com Todos os IMs",
        data=all_ims_data,
        file_name="json_completo.json",
        mime="application/json",
        help="Baixe o JSON completo com IDA contendo todos os IMs e seus blocos."
    )

# Aqui a gente cria colunas para centralizar o conteúdo na tela, tipo um quadro no meio
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # ID automático do bloco
    if 'bloco_id' not in st.session_state:
        st.session_state['bloco_id'] = 1
    st.write(f"**ID do Bloco Atual:** {st.session_state['bloco_id']} (será incrementado para múltiplos)")

    # Uma caixa de texto grande onde o usuário cola o bloco inteiro de conversa
    bloco_texto = st.text_area(
        "Cole os blocos:",
        height=400,
        placeholder="Cole os blocos aqui..."
    )

    # Começa um quadro verde para mostrar os resultados
    st.markdown("""
    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; text-align: left; font-family: Arial, sans-serif; margin-top: 20px;">
        <h3>Visualização Separada por Campos:</h3>
    """, unsafe_allow_html=True)

    if bloco_texto:
        import re

        # Verifica se há múltiplos blocos separados por ---
        if "---" in bloco_texto:
            bloco_texts = [b.strip() for b in bloco_texto.split("---") if b.strip()]
        else:
            bloco_texts = [bloco_texto]

        # Calcular último token usado no IM atual e max bloco_id
        ultimo_token = 0
        max_bloco_id = 0
        if str(im_numero) in ims_salvos:
            for bloco in ims_salvos[str(im_numero)].get('blocos', []):
                for item in bloco.get('Entrada', {}).get('TEXTO Inicial DE ENTRADA', {}).get('TEXE', []):
                    num = int(item['TEXE'].split('.')[1])
                    if num > ultimo_token:
                        ultimo_token = num
                for item in bloco.get('Entrada', {}).get('Fala DE ENTRADA', {}).get('FADEN', []):
                    num = int(item['FADEN'].split('.')[1])
                    if num > ultimo_token:
                        ultimo_token = num
                for item in bloco.get('Entrada', {}).get('Reação', {}).get('RE', []):
                    num = int(item['RE'].split('.')[1])
                    if num > ultimo_token:
                        ultimo_token = num
                if bloco['bloco_id'] > max_bloco_id:
                    max_bloco_id = bloco['bloco_id']

        token_counter = ultimo_token

        # Ponto neutro baseado no IM
        ponto_neutro = f"{im_numero}.0"

        all_blocos = []
        display_html = ""

        for i, bloco_txt in enumerate(bloco_texts):
            # Parsing: encontra marcadores de seção (ver guia no topo para adicionar novos).
            entrada_start = bloco_txt.find("Entrada:")
            reacao1_start = bloco_txt.find("Reação:")
            contexto1_start = bloco_txt.find("Contexto:")
            pensamento_start = bloco_txt.find("Pensamento interno:")
            saida_start = bloco_txt.find("Saída:")
            reacao2_start = bloco_txt.find("Reação:", reacao1_start + 1) if reacao1_start != -1 else -1
            contexto2_start = bloco_txt.find("Contexto:", contexto1_start + 1) if contexto1_start != -1 else -1

            parts = {}

            # Separação: divide subpartes com "—" e split('\n', 1) (ver guia para adaptar).

            # Separa a entrada: divide em texto inicial e fala usando o travessão
            if entrada_start != -1 and reacao1_start != -1:
                entrada_full = bloco_txt[entrada_start + 8:reacao1_start].strip()
                if "—" in entrada_full:
                    inicial, fala = entrada_full.split("—", 1)
                    parts["texto_inicial_entrada"] = inicial.strip()
                    parts["fala_entrada"] = "—" + fala.strip()
                else:
                    parts["texto_inicial_entrada"] = entrada_full
                    parts["fala_entrada"] = ""

            # Para a reação da entrada: primeira linha é a reação, o resto é texto final
            if reacao1_start != -1 and contexto1_start != -1:
                reacao_content = bloco_txt[reacao1_start + 7:contexto1_start].strip()
                lines = reacao_content.split('\n', 1)
                parts["reacao_entrada"] = lines[0].strip() if lines else ""
                parts["texto_final_entrada"] = lines[1].strip() if len(lines) > 1 else ""

            # Contexto da entrada
            if contexto1_start != -1 and pensamento_start != -1:
                parts["contexto_entrada"] = bloco_txt[contexto1_start + 9:pensamento_start].strip()

            # Pensamento interno
            if pensamento_start != -1 and saida_start != -1:
                parts["pensamento_interno"] = bloco_txt[pensamento_start + 19:saida_start].strip()

            # Separa a saída: texto inicial e fala
            if saida_start != -1 and reacao2_start != -1:
                saida_full = bloco_txt[saida_start + 6:reacao2_start].strip()
                if "—" in saida_full:
                    inicial, fala = saida_full.split("—", 1)
                    parts["texto_inicial_saida"] = inicial.strip()
                    parts["fala_saida"] = "—" + fala.strip()
                else:
                    parts["texto_inicial_saida"] = saida_full
                    parts["fala_saida"] = ""

            # Reação da saída: primeira linha reação, resto texto final
            if reacao2_start != -1 and contexto2_start != -1:
                reacao_content = bloco_txt[reacao2_start + 7:contexto2_start].strip()
                lines = reacao_content.split('\n', 1)
                parts["reacao_saida"] = lines[0].strip() if lines else ""
                parts["texto_final_saida"] = lines[1].strip() if len(lines) > 1 else ""

            # Contexto da saída
            if contexto2_start != -1:
                parts["contexto_saida"] = bloco_txt[contexto2_start + 9:].strip()

            # Tokenização: processa texto_inicial_entrada (copie para outros campos se precisar).
            texto_inicial = parts.get('texto_inicial_entrada', '')

            # Ponto neutro baseado no IM
            ponto_neutro = f"{im_numero}.0"


            # Função extrair_texto_e_features: lógica central para CAE/Muden (ver guia para modificações).
            def extrair_texto_e_features(texto):
                # Regex para tags
                faixa_pattern = r'\[([^\]]+)\]'
                # Remove tags para texto limpo
                texto_limpo = re.sub(faixa_pattern, '', texto).strip()

                # Normaliza tags para matching
                def normalize_tag(tag):
                    return re.sub(r'[^\w:]', '', tag).lower()

                # Encontra tags
                tags = list(re.finditer(faixa_pattern, texto))
                secoes = []
                for i, match in enumerate(tags):
                    tag_content = match.group(1)
                    categoria = tag_content.split(":")[0] if ":" in tag_content else tag_content
                    if categoria not in ['CAE', 'Muden']:
                        continue
                    val = tag_content.split(":", 1)[1].strip() if ":" in tag_content else (
                        "Muden" if categoria == 'Muden' else ponto_neutro)
                    # Mapeia CAE (adicione novos no dict abaixo)
                    if categoria == 'CAE':
                        mapping = {
                            "nome": "Nome",
                            "cabelo": "Cabelo",
                            "cabelos": "Cabelo",
                            "olhos": "Olhos",
                            "olho": "Olhos",
                            "pele": "Pele",
                            "forma": "Forma",
                            "estilo": "Estilo",
                            "personalidade": "Personalidade"
                        }
                        val_lower = val.lower()
                        val = mapping.get(val_lower, val)
                    start = match.end()
                    # Encontra próxima tag igual (normalizada)
                    next_start = len(texto)
                    for j in range(i + 1, len(tags)):
                        if normalize_tag(tags[j].group(1)) == normalize_tag(tag_content):
                            next_start = tags[j].start()
                            break
                    if next_start < len(texto):
                        secao = texto[start:next_start]
                        secao_limpa = re.sub(faixa_pattern, '', secao).strip()
                        if secao_limpa:
                            # Posição no texto_limpo
                            pos = texto_limpo.find(secao_limpa)
                            if pos != -1:
                                start_pos = pos
                                end_pos = pos + len(secao_limpa)
                                secoes.append((start_pos, end_pos, categoria.lower(), val))
                return texto_limpo, secoes


            texto_limpo, secoes = extrair_texto_e_features(texto_inicial)

            # Tokenização: cria números sequenciais; aplica features em seções delimitadas.
            tokens = []
            for match in re.finditer(r"\w+'|\w+|[^\w\s]", texto_limpo):
                token = match.group()
                start = match.start()
                token_counter += 1
                token_dict = {"palavra": token, "numero": f"{im_numero}.{token_counter}"}
                # Aplica feature se dentro de seção
                for s_start, s_end, key, val in secoes:
                    if s_start <= start < s_end:
                        token_dict[key] = [val]
                        break
                tokens.append(token_dict)

            # Ponto neutro baseado no IM
            ponto_neutro = f"{im_numero}.0"

            # TEXE: converte tokens em dicts para JSON e display.
            texe_list = [{"TEXE": t["numero"], "t": t["palavra"], "vars": [ponto_neutro],
                          **{k: v for k, v in t.items() if k not in ["palavra", "numero"]}} for t in tokens]

            # Tokenização para FADEN (Fala de Entrada): mesma lógica, ajustada para fala_entrada.
            fala_entrada_texto = parts.get('fala_entrada', '').strip()
            texto_limpo_fala, secoes_fala = extrair_texto_e_features(fala_entrada_texto)
            tokens_fala = []
            for match in re.finditer(r"\w+'|\w+|[^\w\s]", texto_limpo_fala):
                token = match.group()
                start = match.start()
                token_counter += 1
                token_dict = {"palavra": token, "numero": f"{im_numero}.{token_counter}"}
                # Aplica feature se dentro de seção
                for s_start, s_end, key, val in secoes_fala:
                    if s_start <= start < s_end:
                        token_dict[key] = [val]
                        break
                tokens_fala.append(token_dict)
            faden_list = [{"FADEN": t["numero"], "t": t["palavra"], "vars": [ponto_neutro],
                           **{k: v for k, v in t.items() if k not in ["palavra", "numero"]}} for t in tokens_fala]

            # Tokenização para RE (Reação de Entrada): ajustada para agrupar emojis/caretas como tokens únicos.
            reacao_entrada_texto = parts.get('reacao_entrada', '').strip()
            texto_limpo_re, secoes_re = extrair_texto_e_features(reacao_entrada_texto)
            tokens_re = []
            for match in re.finditer(r"\w+|[^\w\s]+", texto_limpo_re):
                token = match.group()
                start = match.start()
                token_counter += 1
                token_dict = {"palavra": token, "numero": f"{im_numero}.{token_counter}"}
                # Aplica feature se dentro de seção
                for s_start, s_end, key, val in secoes_re:
                    if s_start <= start < s_end:
                        token_dict[key] = [val]
                        break
                tokens_re.append(token_dict)
            re_list = [{"RE": t["numero"], "t": t["palavra"], "vars": [ponto_neutro],
                        **{k: v for k, v in t.items() if k not in ["palavra", "numero"]}} for t in tokens_re]

            # JSON do bloco: monta estrutura (adicione novos campos aqui e no display).
            bloco_data = {
                "bloco_id": max_bloco_id + 1 + i,
                "Entrada": {
                    "TEXTO Inicial DE ENTRADA": {
                        "Texto": texto_limpo or ponto_neutro,
                        "TEXE": texe_list,
                        "hash_TEXE": " ".join([t['numero'] for t in tokens]) or ponto_neutro
                    },
                    "Fala DE ENTRADA": {
                        "Texto": texto_limpo_fala or ponto_neutro,
                        "FADEN": faden_list,
                        "hash_FADEN": " ".join([t['numero'] for t in tokens_fala]) or ponto_neutro
                    },
                    "Reação": {
                        "Texto": texto_limpo_re or ponto_neutro,
                        "RE": re_list,
                        "hash_RE": " ".join([t['numero'] for t in tokens_re]) or ponto_neutro
                    },
                    "Texto Final de Entrada": parts.get('texto_final_entrada', '') or ponto_neutro,
                    "Contexto": parts.get('contexto_entrada', '') or ponto_neutro,
                    "Sentimento de Entrada": {"SDE": ponto_neutro, "t": "Neutro",
                                              "Tendência de Entrada": float(ponto_neutro)}
                },
                "Pensamento Interno": parts.get('pensamento_interno', '') or ponto_neutro,
                "Saída": {
                    "Texto Inicial de SAÍDA": parts.get('texto_inicial_saida', '') or ponto_neutro,
                    "Fala de Saída": parts.get('fala_saida', '') or ponto_neutro,
                    "Reação de Saída": parts.get('reacao_saida', '') or ponto_neutro,
                    "Texto Final de Saída": parts.get('texto_final_saida', '') or ponto_neutro,
                    "Contexto de Saída": parts.get('contexto_saida', '') or ponto_neutro,
                    "Sentimento da Saída": {"SDS": ponto_neutro, "t": "Neutro",
                                            "Tendência da Saída": float(ponto_neutro), "Ressonância": False}
                }
            }
            all_blocos.append(bloco_data)  # Adiciona ao display
            display_html += f"""
            <h4>Bloco: {max_bloco_id + 1 + i}</h4>
            <h4>Entrada:</h4>
            <p><strong>TEXTO Inicial de Entrada:</strong> {(texto_limpo or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>TEXE:</strong><br>{'<br>'.join([f'TEXE: {item["TEXE"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["TEXE", "t", "vars"]]) if any(k not in ["TEXE", "t", "vars"] for k in item) else '') for item in bloco_data['Entrada']['TEXTO Inicial DE ENTRADA']['TEXE']])}<br><strong>Hash TEXE:</strong> {bloco_data['Entrada']['TEXTO Inicial DE ENTRADA']['hash_TEXE']}</p>
            <p><strong>Fala de Entrada:</strong> {(texto_limpo_fala or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>FADEN:</strong><br>{'<br>'.join([f'FADEN: {item["FADEN"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["FADEN", "t", "vars"]]) if any(k not in ["FADEN", "t", "vars"] for k in item) else '') for item in bloco_data['Entrada']['Fala DE ENTRADA']['FADEN']])}<br><strong>Hash FADEN:</strong> {bloco_data['Entrada']['Fala DE ENTRADA']['hash_FADEN']}</p>
            <p><strong>Reação:</strong> {(texto_limpo_re or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>RE:</strong><br>{'<br>'.join([f'RE: {item["RE"]}, token: {item["t"]}, vars: {item["vars"]}' + (', ' + ', '.join([f'{k}: {v}' for k, v in item.items() if k not in ["RE", "t", "vars"]]) if any(k not in ["RE", "t", "vars"] for k in item) else '') for item in bloco_data['Entrada']['Reação']['RE']])}<br><strong>Hash RE:</strong> {bloco_data['Entrada']['Reação']['hash_RE']}</p>
            <p><strong>Texto Final de Entrada:</strong> {(parts.get('texto_final_entrada', '') or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>Contexto:</strong> {parts.get('contexto_entrada', '') or ponto_neutro}</p>
            <p><strong>Pensamento interno:</strong> {(parts.get('pensamento_interno', '') or ponto_neutro).replace('\n', '<br>')}</p>
            <h4>Saída:</h4>
            <p><strong>Texto Inicial de Saída:</strong> {(parts.get('texto_inicial_saida', '') or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>Fala de Saída:</strong> {(parts.get('fala_saida', '') or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>Reação:</strong> {parts.get('reacao_saida', '') or ponto_neutro}</p>
            <p><strong>Texto Final de Saída:</strong> {(parts.get('texto_final_saida', '') or ponto_neutro).replace('\n', '<br>')}</p>
            <p><strong>Contexto:</strong> {parts.get('contexto_saida', '') or ponto_neutro}</p>
            <hr>
            """

        st.markdown(display_html, unsafe_allow_html=True)

        # Salvar IM: atualiza JSON (ver guia).
        import json

        json_structure = {
            ida_value: {
                "IM": {
                    str(im_numero): {
                        "nome": im_nome,
                        "blocos": all_blocos
                    }
                }
            }
        }

        json_data = json.dumps(json_structure, ensure_ascii=False, indent=4)
        num_blocos = len(all_blocos)

        # Exibir JSON
        with st.expander("Ver JSON Estruturado"):
            if ims_salvos:
                full_json = {ida_value: {"IM": ims_salvos}}
                st.json(full_json)
            else:
                st.json(json_structure)

        if st.button("Salvar IM com Blocos Atuais"):
            if str(im_numero) in ims_salvos:
                ims_salvos[str(im_numero)]["blocos"].extend(all_blocos)
            else:
                ims_salvos[str(im_numero)] = {"nome": im_nome, "blocos": all_blocos}
            with open(ims_file, 'w', encoding='utf-8') as f:
                json.dump(ims_salvos, f, ensure_ascii=False, indent=4)
            st.success(f"IM {im_numero} salvo com {len(all_blocos)} bloco(s) adicionado(s)!")
            st.rerun()

        # Removido o download individual, pois o foco é no JSON completo
    else:
        st.markdown("<p>Cole um bloco para visualizar os campos separados.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
