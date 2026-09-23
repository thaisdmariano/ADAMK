# MOTOR DE TEMPERATURA EMOCIONAL -- protótipo isolado (2026-09-22).
# Esquema de triangulação escrito por Thaís com apoio do Claude Opus
# ("Claudin") -- implementado à risca, sem trocar por índices, sem média
# quando os campos divergem.
#
# Temperatura NÃO é campo novo -- é o RESULTADO da leitura integrada de
# Texto + Reação + Contexto: Temperatura = Σ(peso do marcador × polaridade)
# onde polaridade é +1 (positivo), 0 (neutro/vazio), -1 (negativo). Cada
# marcador é registrado INDIVIDUALMENTE por ela ("isso é positivo/
# negativo/neutro"), uma vez -- reaproveitado em qualquer bloco futuro com
# o mesmo valor exato naquele rótulo. Nunca um dicionário de palavra boa/
# palavra ruim.
#
#   TEXE/TEDSA = 15%   FADEN/FADES = 15%   TEFE/TEFSA = 15%
#   RE/RS = 15%        CE/CS = 40%                        = 100%
#
# Entrada e saída são cálculos INDEPENDENTES -- o Adam não é espelho, ele
# processa e pode discordar do tom de quem falou com ele.
#
# Campo vazio contribui 0 -- nunca precisa de registro, nunca inventa
# contexto que não existe (bloco 4: sem contexto, teto natural = 60%).
# Campo com conteúdo mas SEM registro ainda fica pendente -- nunca vira 0
# por adivinhação.
#
# Quando os marcadores DIVERGEM, isso é INFORMAÇÃO (conflito emocional),
# nunca resolvido por média -- a soma líquida já carrega esse conflito.
#
# Roda como app Streamlit próprio: `streamlit run motor_emocional.py`.

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

PROJ = Path(__file__).parent
ARQUIVO_MEMORIA = PROJ / "Adam_Lovely_memory.json"
ARQUIVO_REGISTRO = PROJ / "motor_emocional_registro.json"

PESOS = {"TEXE": 15, "FADEN": 15, "TEFE": 15, "RE": 15, "CE": 40}
PESOS_SAIDA = {"TEDSA": 15, "FADES": 15, "TEFSA": 15, "RS": 15, "CS": 40}
FAMS_TEXTO = ("TEXE", "FADEN", "TEFE")
FAMS_TEXTO_SAIDA = ("TEDSA", "FADES", "TEFSA")
FAM_REACAO = {"entrada": "RE", "saida": "RS"}

# 2026-09-22, Thaís: "a frase + a reação = emoção" -- um campo de TEXTO
# neutro ao lado de uma reação já registrada como positiva/negativa ganha
# +5 pontos NA DIREÇÃO da reação (empresta o tom do emoji). Só ativa
# quando a reação em si é positiva ou negativa -- uma reação neutra não
# empresta tom nenhum.
# 2026-09-23: religado -- o Exemplo 2 do documento (Ela chegou no local /
# e observou, TEXE e FADEN neutros, RE positivo, CE positivo) passa de
# 55% pra 65% com o bônus ligado. Pra esse resultado continuar dentro da
# zona de Equilíbrio (como o documento descreve esse exemplo), a Thaís
# subiu o teto do Equilíbrio de 60% pra 65% -- ver ZONAS abaixo.
APLICAR_BONUS_FRASE_REACAO = True
BONUS_NEUTRO_COM_REACAO = 5

ESTADOS = ["positivo", "negativo", "neutro"]
SINAL = {"positivo": 1, "negativo": -1, "neutro": 0}
COR = {"positivo": "#4fc9a8", "negativo": "#e2607a", "neutro": "#8a8fa3"}

# Faixas de temperatura -- modelo simplificado de 4 faixas (2026-09-23),
# substituindo o esquema de 6 do documento original. Fronteiras EXATAS
# dadas pela Thaís, uma por uma, nunca inferidas por espelhamento (ela
# rejeitou explicitamente inferir o lado negativo copiando o positivo):
#   16% a 85%   → Morno/Equilibrado/Estável ("85% é o final de morno")
#   86% a 100%  → Fervendo ("86% é fervendo")
#   -85% a 15%  → Esfriando ("15% pra baixo é esfriando", "-85% é Esfriando")
#   -86% a -100% → Gelo ("-86% é GELO")
# "Frio" e "Quente" não existem mais como faixas separadas -- foram
# absorvidos em Esfriando e em Morno/Equilibrado/Estável.
ZONAS = [
    (86, "🔥 Fervendo", "#e2607a"),
    (16, "🌤️ Morno / Equilibrado / Estável", "#4fc9a8"),
    (-85, "🌫️ Esfriando", "#7c93f5"),
    (-101, "🧊 Gelo", "#2f3fa0"),
]


def classificar_zona(temperatura: float) -> Tuple[str, str]:
    for minimo, nome, cor in ZONAS:
        if temperatura >= minimo:
            return nome, cor
    return ZONAS[-1][1], ZONAS[-1][2]


def tokenizar(texto: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", texto or "", re.UNICODE)


def dividir_travessao(texto: str) -> Tuple[str, str, str]:
    partes = (texto or "").split("—")
    if len(partes) < 3:
        return texto or "", "", ""
    return partes[0].strip(), partes[1].strip(), "—".join(partes[2:]).strip()


def _vazio(valor) -> bool:
    v = str(valor or "").strip()
    return not v or re.fullmatch(r"\d+\.0", v) is not None


@st.cache_data
def carregar_memoria() -> dict:
    return json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))


def carregar_registro() -> Dict[str, str]:
    if ARQUIVO_REGISTRO.exists():
        return json.loads(ARQUIVO_REGISTRO.read_text(encoding="utf-8"))
    return {}


def salvar_registro(registro: Dict[str, str]) -> None:
    if not st.session_state.get("backup_registro_feito") and ARQUIVO_REGISTRO.exists():
        carimbo = __import__("time").strftime("%Y%m%d_%H%M%S")
        ARQUIVO_REGISTRO.with_name(f"motor_emocional_registro.BACKUP_{carimbo}.json").write_text(
            ARQUIVO_REGISTRO.read_text(encoding="utf-8"), encoding="utf-8")
        st.session_state["backup_registro_feito"] = True
    tmp = ARQUIVO_REGISTRO.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARQUIVO_REGISTRO)


def chave(rotulo: str, valor: str) -> str:
    """Registro é por CAMPO exato -- mesmo valor no mesmo rótulo, em
    qualquer bloco, reaproveita o registro. Nunca por bloco inteiro."""
    return f"{rotulo}:{valor.strip()}"


def valores_do_lado(bloco: dict, lado: str) -> Dict[str, str]:
    """Devolve {rotulo: valor_exato} pros 5 campos de um lado do bloco
    (entrada ou saída) -- só os que existem de verdade (vazio vira '')."""
    if lado == "entrada":
        e = bloco["entrada"]
        texe, faden, tefe = dividir_travessao(e.get("texto", ""))
        return {"TEXE": texe, "FADEN": faden, "TEFE": tefe, "RE": e.get("reacao", ""), "CE": e.get("contexto", "")}
    s = bloco["saidas"][0]
    texto_saida = (s.get("textos") or [""])[0]
    tedsa, fades, tefsa = dividir_travessao(texto_saida)
    return {"TEDSA": tedsa, "FADES": fades, "TEFSA": tefsa, "RS": s.get("reacao", ""), "CS": s.get("contexto", "")}


def avaliar_lado(bloco: dict, lado: str, registro: Dict[str, str]) -> Dict[str, object]:
    """2026-09-23, Thaís: "o maior peso existe, ele só não anula o outro"
    -- positivo e negativo são somados SEPARADOS, um nunca cancela o
    outro. O lado mais forte vence, e a temperatura mostrada é o peso
    PRÓPRIO daquele lado (não a diferença entre os dois). Um bloco com
    55% positivo e 45% negativo dá +55%, nunca "+10%" escondendo os 45%
    que perderam.

    Campo vazio contribui 0 (nunca precisa de registro, nunca inventa
    contexto). Campo com conteúdo mas SEM registro ainda fica pendente --
    nunca vira 0 por adivinhação."""
    pesos = PESOS if lado == "entrada" else PESOS_SAIDA
    valores = valores_do_lado(bloco, lado)
    detalhe = []
    pendentes = []
    for fam, peso in pesos.items():
        valor = valores.get(fam, "")
        if _vazio(valor):
            detalhe.append({"fam": fam, "valor": "", "peso": peso, "estado": "vazio", "contribuicao": 0})
            continue
        k = chave(fam, valor)
        estado = registro.get(k)
        if estado is None:
            pendentes.append((fam, valor))
            detalhe.append({"fam": fam, "valor": valor, "peso": peso, "estado": "pendente", "contribuicao": None})
            continue
        contribuicao = peso * SINAL[estado]
        detalhe.append({"fam": fam, "valor": valor, "peso": peso, "estado": estado, "contribuicao": contribuicao})

    # "A frase + a reação = emoção" (Thaís, 2026-09-22): um campo de TEXTO
    # neutro ao lado de uma reação já positiva/negativa empresta +5 pontos
    # na direção dela -- reação neutra não empresta tom nenhum. O bônus
    # entra DIRETO na contribuição do campo, pra curva da trajetória e o
    # número final nunca discordarem entre si.
    estado_por_fam = {item["fam"]: item["estado"] for item in detalhe}
    fams_texto = FAMS_TEXTO if lado == "entrada" else FAMS_TEXTO_SAIDA
    estado_reacao = estado_por_fam.get(FAM_REACAO[lado])
    bonus = []
    if APLICAR_BONUS_FRASE_REACAO and estado_reacao in ("positivo", "negativo"):
        for item in detalhe:
            if item["fam"] in fams_texto and item["estado"] == "neutro":
                pontos = BONUS_NEUTRO_COM_REACAO if estado_reacao == "positivo" else -BONUS_NEUTRO_COM_REACAO
                item["contribuicao"] += pontos
                bonus.append({"fam": item["fam"], "para": estado_reacao, "pontos": BONUS_NEUTRO_COM_REACAO})

    def sinal_vencedor(total_positivo: int, total_negativo: int) -> int:
        if total_positivo > total_negativo:
            return total_positivo
        if total_negativo > total_positivo:
            return -total_negativo
        return 0

    # 5.1 do documento -- Temperatura como Trajetória: a curva campo por
    # campo, na mesma ordem real dos marcadores (TEXE→FADEN→TEFE→RE→CE).
    # Em cada ponto, positivo e negativo acumulados até ali (sem cancelar)
    # decidem quem vence NAQUELE momento -- a curva pode trocar de lado
    # no meio do bloco. Campo pendente trunca a curva dali pra frente.
    trajetoria = []
    pos_acum = neg_acum = 0
    truncado = False
    for item in detalhe:
        if item["estado"] == "pendente" or truncado:
            trajetoria.append({"fam": item["fam"], "contribuicao": item["contribuicao"], "cumulativo": None})
            truncado = True
            continue
        c = item["contribuicao"]
        pos_acum += max(c, 0)
        neg_acum += max(-c, 0)
        trajetoria.append({"fam": item["fam"], "contribuicao": c, "cumulativo": sinal_vencedor(pos_acum, neg_acum)})

    direcao = None
    if not pendentes and trajetoria:
        primeiro, ultimo = trajetoria[0]["cumulativo"], trajetoria[-1]["cumulativo"]
        if ultimo > primeiro:
            direcao = "📈 ascendente -- esquentando ao longo do bloco"
        elif ultimo < primeiro:
            direcao = "📉 descendente -- esfriando ao longo do bloco"
        else:
            direcao = "➡️ estável"

    if pendentes:
        return {"temperatura": None, "zona": None, "cor_zona": None, "motivo": f"{len(pendentes)} campo(s) sem registro",
                "detalhe": detalhe, "pendentes": pendentes, "bonus": bonus, "trajetoria": trajetoria, "direcao": direcao}
    temperatura = trajetoria[-1]["cumulativo"] if trajetoria else 0
    zona, cor_zona = classificar_zona(temperatura)
    return {"temperatura": temperatura, "zona": zona, "cor_zona": cor_zona, "motivo": None,
            "detalhe": detalhe, "pendentes": pendentes, "bonus": bonus, "trajetoria": trajetoria, "direcao": direcao}


# ────────────────────────────────────────────────────────────────────────
# Streamlit
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Motor de temperatura emocional", layout="wide")
st.title("🌡️ Motor de temperatura emocional -- protótipo isolado")
st.caption("Temperatura = soma líquida dos campos já registrados. Cada campo é registrado por você, uma vez, e vale pra sempre.")

if not ARQUIVO_MEMORIA.exists():
    st.error(f"Não achei {ARQUIVO_MEMORIA.name} ao lado deste arquivo.")
    st.stop()

memoria = carregar_memoria()
blocos = memoria["IM"]["0"]["blocos"]
if "registro" not in st.session_state:
    st.session_state["registro"] = carregar_registro()
registro = st.session_state["registro"]

for bloco in blocos:
    st.divider()
    st.subheader(f"Bloco {bloco['bloco_id']}")
    col_e, col_s = st.columns(2)

    for lado, col, titulo in (("entrada", col_e, "Entrada"), ("saida", col_s, "Saída")):
        with col:
            st.markdown(f"**{titulo}**")
            r = avaliar_lado(bloco, lado, registro)
            if r["pendentes"]:
                st.warning(f"⏳ {r['motivo']}")
            else:
                st.markdown(f"<span style='color:{r['cor_zona']};font-size:1.3em;font-weight:bold'>{r['temperatura']:+d}% -- {r['zona']}</span>", unsafe_allow_html=True)
                if r["direcao"]:
                    st.caption(r["direcao"])
                for b in r["bonus"]:
                    sinal_bonus = "+" if b["para"] == "positivo" else "-"
                    st.caption(f"🎁 {b['fam']} era neutro, ganhou {sinal_bonus}{b['pontos']} pra {b['para']} (reação empresta o tom)")

            # 5.1 -- trajetória: a curva acumulada campo por campo, não só
            # o número final. Mostra mesmo com pendentes (trunca onde faltar).
            pontos_conhecidos = [p for p in r["trajetoria"] if p["cumulativo"] is not None]
            if len(pontos_conhecidos) >= 2:
                st.caption("Trajetória (" + " → ".join(f"{p['fam']} {p['cumulativo']:+d}%" for p in pontos_conhecidos) + ")")
                st.line_chart({"temperatura acumulada": [p["cumulativo"] for p in pontos_conhecidos]}, height=120)

            for item in r["detalhe"]:
                if item["estado"] == "vazio":
                    st.caption(f"· {item['fam']} ({item['peso']}%): vazio")
                    continue
                if item["estado"] == "pendente":
                    st.markdown(f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): \"{item['valor']}\" -- <span style='color:#8a8fa3'>?</span></span>", unsafe_allow_html=True)
                    continue
                cor = COR.get(item["estado"], "#8a8fa3")
                st.markdown(
                    f"<span style='font-size:0.85em'>· {item['fam']} ({item['peso']}%): "
                    f"<span style='color:{cor}'>\"{item['valor']}\" -- {item['estado']} ({item['contribuicao']:+d})</span></span>",
                    unsafe_allow_html=True,
                )

            for fam, valor in r["pendentes"]:
                escolha = st.radio(f"{fam}: \"{valor}\" é...", ESTADOS, index=None, horizontal=True, key=f"{bloco['bloco_id']}_{lado}_{fam}")
                if escolha:
                    registro[chave(fam, valor)] = escolha
                    salvar_registro(registro)
                    st.rerun()
