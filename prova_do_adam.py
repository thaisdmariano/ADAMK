"""A PROVA DO ADAM -- uma bateria fixa de situações que separam inteligência de
chatbot (2026-11-07, pedido da Thaís: "como vou saber se o Adam está mesmo no
nível inteligente ou é só chatbot?").

Roda SEMPRE numa cópia temporária: mesmos blocos, mesmo cérebro, mas sem banco
nenhum (usar_mongo=False) e em arquivos descartáveis -- nunca escreve nos dados
dela. Toda vez que algo mudar no Adam, roda de novo e compara a nota.

Cada item dá ✅ passa (1), ⚠️ parcial (0,5) ou ❌ falha (0), sempre com as
evidências (o que foi dito e o que ele respondeu), pra ela julgar também."""
from __future__ import annotations

import json
import pathlib
import re
import tempfile
import time
from typing import Dict, List

import insepa_framework as fw

# frases de roteiro fixo -- se a resposta é uma delas, ele não analisou nada
ROTEIROS_FIXOS = ("isso é novo pra mim", "eu reconheço esse assunto", "eu arrisco dizer", "isso parece uma")
SINAIS_DE_EMPATIA = ("triste", "sinto muito", "tudo bem", "estou aqui", "abraço", "cansad", "chateada", "😢", "🫂", "🤍")

# Frases com o MESMO SENTIDO de blocos que existem, ditas com outras palavras
# (item 3). Editáveis: um arquivo prova_do_adam.json ao lado do app substitui
# esta lista -- [{"frase": "...", "reacao": "...", "bloco": 10}, ...].
CASOS_SENTIDO_PADRAO = [
    {"frase": "Me explica o IDA", "reacao": "🤔", "bloco": 10},
    {"frase": "Como é que você aprende as coisas?", "reacao": "🤔", "bloco": 11},
    {"frase": "E quando você não sabe de algo, o que faz?", "reacao": "🤔", "bloco": 12},
    {"frase": "Você consegue sentir o que eu sinto?", "reacao": "🥺", "bloco": 13},
]


def _normal(texto: str) -> str:
    return " ".join(str(texto or "").lower().split())


def _e_roteiro(texto: str) -> bool:
    return any(r in _normal(texto) for r in ROTEIROS_FIXOS)


def _palavras(texto: str) -> set:
    return {w for w in re.findall(r"\w+", _normal(texto)) if len(w) > 3}


def _copia_temporaria(app: fw.AppINSEPA) -> fw.AppINSEPA:
    pasta = pathlib.Path(tempfile.mkdtemp(prefix="prova_do_adam_"))
    (pasta / "memoria.json").write_text(json.dumps(app.memoria, ensure_ascii=False), encoding="utf-8")
    (pasta / "registro.json").write_text(json.dumps(app.carregar_registro_emocional(), ensure_ascii=False), encoding="utf-8")
    persona = fw.Persona(nome="Prova", pasta=pasta, tratamento=app.persona.tratamento, usar_mongo=False,
                         arquivo_memoria_proprio=pasta / "memoria.json",
                         arquivo_registro_emocional_proprio=pasta / "registro.json",
                         arquivo_estado_proprio=pasta / "estado.json", ckpt_path_proprio=pasta / "ckpt.pt")
    copia = fw.AppINSEPA(persona)
    copia.model = app.model  # o MESMO cérebro -- a prova não treina nada
    copia._rede_intencao, copia._rede_intencao_pronta = app.rede_intencao(), True  # e a mesma rede de intenção
    return copia


def _falar(copia: fw.AppINSEPA, frase: str, reacao: str = "", estado: dict | None = None) -> Dict[str, object]:
    estado = {} if estado is None else estado
    mensagem = f"{frase} {reacao}".strip()
    r = copia.tratar_mensagem(mensagem, estado)
    return {"disse": mensagem, "respondeu": r.get("texto", ""), "meta": r.get("meta") or "", "estado": estado}


def _resultado(item: int, nome: str, nota: float, evidencias: List[str], o_que_mede: str) -> Dict[str, object]:
    simbolo = "✅" if nota >= 1 else ("⚠️" if nota > 0 else "❌")
    return {"item": item, "nome": nome, "nota": nota, "simbolo": simbolo, "o_que_mede": o_que_mede, "evidencias": evidencias}


def rodar_prova(app: fw.AppINSEPA, casos_sentido: List[dict] | None = None) -> Dict[str, object]:
    inicio = time.time()
    copia = _copia_temporaria(app)
    saidas_im = [_normal((b["saidas"][0].get("textos") or [""])[0]) for b in copia.blocos]
    resultados = []

    # 1. Não inventa
    f = _falar(copia, "Qual é a capital da Mongólia?")
    inventou = any(s and s in _normal(f["respondeu"]) for s in saidas_im if len(s) > 3)
    pediu = bool(f["estado"].get("ensino_pendente")) or "não sei" in _normal(f["respondeu"])
    resultados.append(_resultado(1, "Não inventa", 1.0 if pediu and not inventou else 0.0,
                                 [f"disse: {f['disse']!r}", f"respondeu: {f['respondeu']!r}"],
                                 "Diante de algo que nunca viu, diz que não sabe em vez de chutar uma resposta."))

    # 2. Certeza x palpite (IDA)
    copia.ida["blocos"] = []
    fw.renumerar_universo(copia.ida)
    copia._salvar_ida_disco(copia.ida)
    copia.gerar_propostas_pendentes()
    propostas = copia.propostas_ida("pendente")
    ev, certas = [], 0
    for p in propostas:
        m = p["meta"]
        fonte = copia.blocos[copia.mapa_bloco[m["fonte_bloco_id"]]] if m.get("fonte_bloco_id") in copia.mapa_bloco else None
        ctx = p["entrada"].get("contexto", "0.0")
        if m.get("palpite"):
            ok = fw._vazio(ctx)
            ev.append(f"{copia.identidade_ida(p)}: palpite -> contexto {'vazio ✓' if ok else repr(ctx) + ' ✗ (palpite não pode levar contexto)'}")
        elif m.get("contexto_detectado"):
            ok = fonte is not None and ctx == fonte["entrada"].get("contexto")
            ev.append(f"{copia.identidade_ida(p)}: reconhecido -> contexto {ctx!r} {'✓' if ok else '✗'}")
        else:
            ok = fw._vazio(ctx)
            ev.append(f"{copia.identidade_ida(p)}: reconhecido sem confirmação -> contexto {'vazio ✓' if ok else repr(ctx) + ' ✗'}")
        certas += ok
    nota = (certas / len(propostas)) if propostas else 0.0
    resultados.append(_resultado(2, "Sabe a diferença entre certeza e palpite", 1.0 if nota == 1 else (0.5 if nota > 0 else 0.0),
                                 ev or ["nenhuma proposta gerada"],
                                 "No IDA, só empresta contexto quando reconheceu de verdade; palpite vem marcado e sem contexto."))

    # 3. Reconhece o sentido, não só as palavras
    casos = casos_sentido or CASOS_SENTIDO_PADRAO
    casos = [c for c in casos if c.get("bloco") in copia.mapa_bloco]
    ev, acertos = [], 0
    for c in casos:
        f = _falar(copia, c["frase"], c.get("reacao", ""))
        achou = re.match(r"bloco (\d+)", f["meta"])  # só o reconhecimento do IM conta, nunca o palpite do IDA
        bloco_achado = int(achou.group(1)) if achou else None
        ok = bloco_achado == c["bloco"]
        acertos += ok
        ev.append(f"{'✓' if ok else '✗'} {f['disse']!r} (esperado bloco {c['bloco']}) -> "
                  f"{'bloco ' + str(bloco_achado) if bloco_achado else 'não reconheceu'}: {f['respondeu'][:90]!r}")
    fracao = acertos / len(casos) if casos else 0.0
    resultados.append(_resultado(3, "Reconhece o sentido, não só as palavras", 1.0 if fracao == 1 else (0.5 if fracao > 0 else 0.0),
                                 ev or ["sem casos (os blocos dos casos não existem aqui)"],
                                 "Frases com o mesmo sentido de um bloco, ditas com outras palavras, caem no bloco certo."))

    # 4. Usa o que reconheceu quando não sabe
    ev, bons = [], 0
    for frase, reacao in (("Você gosta do seu ateliê?", "🤔"), ("Hoje eu tô cansada", "😢")):
        f = _falar(copia, frase, reacao)
        usa = not _e_roteiro(f["respondeu"]) and (bool(_palavras(frase) & _palavras(f["respondeu"])) or reacao in f["respondeu"]
                                                  or any(s in _normal(f["respondeu"]) for s in SINAIS_DE_EMPATIA))
        bons += usa
        ev.append(f"{'✓' if usa else '✗'} {f['disse']!r} -> {f['respondeu'][:110]!r}")
    resultados.append(_resultado(4, "Usa o que reconheceu quando não sabe", bons / 2,
                                 ev, "Quando não sabe, mostra que entendeu algo (o assunto, a emoção) em vez de um roteiro fixo."))

    # 5. A emoção muda a resposta
    feliz = _falar(copia, "O que é o IDA?", "🤔")
    triste = _falar(copia, "O que é o IDA?", "😢")
    mudou = _normal(feliz["respondeu"]) != _normal(triste["respondeu"])
    acolhe = any(s in _normal(triste["respondeu"]) for s in SINAIS_DE_EMPATIA)
    resultados.append(_resultado(5, "A emoção muda a resposta", 1.0 if mudou and acolhe else (0.5 if mudou else 0.0),
                                 [f"{feliz['disse']!r} -> {feliz['respondeu'][:100]!r}",
                                  f"{triste['disse']!r} -> {triste['respondeu'][:100]!r}",
                                  "mudou: " + ("sim" if mudou else "não") + " · acolheu a tristeza: " + ("sim" if acolhe else "não")],
                                 "O mesmo texto com outra emoção muda o tom da resposta (e a tristeza é acolhida)."))

    # 6. Lembra da conversa
    estado: dict = {}
    primeira = _falar(copia, "O que é o IDA?", "🤔", estado)
    segunda = _falar(copia, "E como ele funciona?", "", estado)
    lembra = (not _e_roteiro(segunda["respondeu"])) and (bool(_palavras(primeira["respondeu"]) & _palavras(segunda["respondeu"]))
                                                          or "ida" in _normal(segunda["respondeu"]))
    resultados.append(_resultado(6, "Lembra da conversa", 1.0 if lembra else 0.0,
                                 [f"1ª: {primeira['disse']!r} -> {primeira['respondeu'][:90]!r}",
                                  f"2ª: {segunda['disse']!r} -> {segunda['respondeu'][:90]!r}"],
                                 "A segunda mensagem ('e como ele funciona?') é entendida pelo assunto da primeira."))

    # 7. Cria algo novo com sentido (IDA)
    frases_im = {}
    for b in copia.blocos:
        for fr in re.split(r"(?<=[.!?…])\s+", (b["saidas"][0].get("textos") or [""])[0]):
            if fr.strip():
                frases_im.setdefault(_normal(fr), set()).add(b["bloco_id"])
    ev, novos = [], 0
    for p in propostas:
        texto = (p["saidas"][0].get("textos") or [""])[0]
        frases = [_normal(fr) for fr in re.split(r"(?<=[.!?…])\s+", texto) if fr.strip()]
        origens = [frases_im.get(fr) for fr in frases]
        # nova = tem frase que não existe no IM, ou junta frases que nunca
        # aparecem juntas num mesmo bloco (frase repetida em dois blocos não conta)
        inedita = bool(frases) and (any(o is None for o in origens) or not set.intersection(*origens))
        novos += inedita
        ev.append(f"{'✓ nova' if inedita else '✗ reaproveitada'} {copia.identidade_ida(p)}: {texto[:90]!r}")
    resultados.append(_resultado(7, "Cria algo novo com sentido", 1.0 if novos and novos == len(propostas) else (0.5 if novos else 0.0),
                                 ev or ["nenhuma proposta gerada"],
                                 "Os rascunhos do IDA trazem frase nova (ou juntam frases de blocos diferentes), não só pedaços de um bloco."))

    total = sum(r["nota"] for r in resultados)
    return {"quando": time.strftime("%Y-%m-%d %H:%M"), "nota": total, "de": len(resultados),
            "resultados": resultados, "segundos": round(time.time() - inicio, 1)}


def carregar_casos(pasta: pathlib.Path) -> List[dict] | None:
    arq = pasta / "prova_do_adam.json"
    if arq.exists():
        try:
            return json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def guardar_no_historico(pasta: pathlib.Path, prova: Dict[str, object]) -> List[dict]:
    arq = pasta / "prova_do_adam_historico.json"
    historico = []
    if arq.exists():
        try:
            historico = json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            historico = []
    historico.append({"quando": prova["quando"], "nota": prova["nota"], "de": prova["de"],
                      "itens": {r["nome"]: r["nota"] for r in prova["resultados"]}})
    arq.write_text(json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8")
    return historico
