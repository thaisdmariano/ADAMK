import streamlit as st
from robo_cosmico import RoboCosmico

# Criação do robô cósmico
robo = RoboCosmico()

# Título da aplicação
st.title('Robô Cósmico')

# Caixa de texto para o usuário inserir perguntas
pergunta = st.text_input('Faça uma pergunta ao robô:')

# Botão para enviar a pergunta
if st.button('Enviar'):
    if pergunta:
        resposta = robo.responder_a_pergunta(pergunta)
        st.write(resposta)
    else:
        st.write('Por favor, insira uma pergunta.')
