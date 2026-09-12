#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INSEPA - Camada 7: Tela.

Um blueprint explícito da forma que um universo (IM) deve ter: nome, léxico
inconsciente (marcador -> palavra) e memória (blocos com entrada/saída
completas, cada um virando dado concreto de verdade -- Camada 5). Sem essa
tela, o Adam (e a gente) só enxerga JSON cru -- não tem como saber de relance
se o que está ali é a forma certa ou se algo divergiu pelo caminho.

A tela não julga se um dado é bom ou ruim (isso é papel de outras camadas) --
ela só confere se a FORMA bate com o esperado, sempre por presença/igualdade
exata, nunca por aproximação. Mesmo vocabulário da Camada 6: "bate" / "diverge"
/ "ausente".
"""

from __future__ import annotations

import html as _html
from typing import Any, Dict, List, Set, Tuple


def _ordenar_por_marcador(pares: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    """Ordena (marcador, dado) por IM.IF numérico de verdade -- "0.10" vem
    depois de "0.9", não entre "0.1" e "0.2" como ordenaria texto puro."""
    def chave(item):
        mae, _, filho = item[0].partition(".")
        return (int(mae), int(filho or 0))
    return sorted(pares, key=chave)


def visao_bloco_inconsciente(bloco_inco: dict) -> str:
    """Renderiza UM bloco do INCONSCIENTE: cada palavra junto do marcador que
    define sua posição -- "token(marcador)", sem classificação nenhuma, de
    propósito. O inconsciente é caos organizado só por posição -- não tem E/RE/
    CE/PIDE separado aqui, é tudo junto, igual ao formatar() da Camada 1."""
    entrada_pares = _ordenar_por_marcador(list(bloco_inco.get("Entrada", {}).items()))
    saida_pares = _ordenar_por_marcador(list(bloco_inco.get("SAÍDA", {}).items()))
    entrada_txt = " ".join(f"{d['token']}({m})" for m, d in entrada_pares)
    saida_txt = " ".join(f"{d['token']}({m})" for m, d in saida_pares)
    return (
        f"Bloco {bloco_inco['Bloco_id']}:\n"
        f"  Entrada: {entrada_txt}\n"
        f"  Saída:   {saida_txt}"
    )


def visao_universo_inconsciente(nome: str, dominio: str, blocos_inco: List[dict]) -> str:
    """A visão completa do INCONSCIENTE de um universo: título (IM = Universo X)
    seguido de cada bloco, palavra por palavra, desordenado e sem classificação
    -- exatamente como o Inconsciente é: dado bruto, só posição, nada mais."""
    titulo = f"IM = Universo {dominio} ({nome})"
    corpo = "\n\n".join(visao_bloco_inconsciente(b) for b in blocos_inco)
    return f"{titulo}\n\n{corpo}" if blocos_inco else titulo


_ROTULOS_CAMPO = {
    "E": "Texto (E)", "RE": "Reação (RE)", "CE": "Contexto (CE)", "PIDE": "Pensamento (PIDE)",
    "S": "Texto (S)", "RS": "Reação (RS)", "CS": "Contexto (CS)",
}


def _parear_campo(campo: str, palavras: List[str], marcadores: List[str]) -> str:
    """Junta palavra com marcador campo a campo -- só até onde os dois listam
    junto (o PIDE na memória guarda só os 3 marcadores-chave, por exemplo, não
    um por palavra do pensamento inteiro; isso é assim de propósito, não bug)."""
    pares = " ".join(f"{p}({m})" for p, m in zip(palavras, marcadores))
    return pares if pares else "(vazio)"


def visao_bloco_memoria(bloco: dict, tokenizar) -> str:
    """Renderiza UM bloco da MEMÓRIA (consciente) respeitando a classificação
    que a criadora desenhou: ENTRADA e SAÍDA separadas, e dentro de cada uma,
    cada campo (E/RE/CE/PIDE ou S/RS/CS) mostrado à parte -- nunca tudo
    misturado numa linha só como o inconsciente."""
    entrada = bloco.get("entrada", {})
    saida = bloco.get("saidas", [{}])[0]
    tokens_e = entrada.get("tokens", {})
    tokens_s = saida.get("tokens", {})

    linhas = [f"Bloco {bloco['bloco_id']}:", "  ENTRADA"]
    linhas.append(f"    {_ROTULOS_CAMPO['E']}: {_parear_campo('E', tokenizar(entrada.get('texto', '')), tokens_e.get('E', []))}")
    reacao = entrada.get("reacao", "")
    linhas.append(f"    {_ROTULOS_CAMPO['RE']}: {_parear_campo('RE', tokenizar(reacao) if reacao else [], tokens_e.get('RE', []))}")
    linhas.append(f"    {_ROTULOS_CAMPO['CE']}: {_parear_campo('CE', tokenizar(entrada.get('contexto', '')), tokens_e.get('CE', []))}")
    linhas.append(f"    {_ROTULOS_CAMPO['PIDE']}: {_parear_campo('PIDE', tokenizar(entrada.get('pensamento_interno', '')), tokens_e.get('PIDE', []))}  [texto completo: \"{entrada.get('pensamento_interno', '')}\"]")

    linhas.append("  SAÍDA")
    saida_texto = (saida.get("textos") or [""])[0]
    linhas.append(f"    {_ROTULOS_CAMPO['S']}: {_parear_campo('S', tokenizar(saida_texto), tokens_s.get('S', []))}")
    saida_reacao = saida.get("reacao", "")
    linhas.append(f"    {_ROTULOS_CAMPO['RS']}: {_parear_campo('RS', tokenizar(saida_reacao) if saida_reacao else [], tokens_s.get('RS', []))}")
    linhas.append(f"    {_ROTULOS_CAMPO['CS']}: {_parear_campo('CS', tokenizar(saida.get('contexto', '')), tokens_s.get('CS', []))}")

    return "\n".join(linhas)


def visao_universo_memoria(nome: str, dominio: str, blocos: List[dict], tokenizar) -> str:
    """A visão completa da MEMÓRIA (consciente) de um universo: título seguido
    de cada bloco, com a classificação em campos respeitada -- ENTRADA/SAÍDA e
    E/RE/CE/PIDE/S/RS/CS sempre separados, nunca uma sopa só de palavra+marcador.
    `tokenizar` é injetado (ex.: Token de lovely_test.py) pra não duplicar regex aqui."""
    titulo = f"IM = Universo {dominio} ({nome})"
    corpo = "\n\n".join(visao_bloco_memoria(b, tokenizar) for b in blocos)
    return f"{titulo}\n\n{corpo}" if blocos else titulo


## VISÃO EM HTML -- mesmas funções acima, só que desenhadas: cor por campo,
## espaçamento uniforme. As funções de texto puro continuam existindo (são a
## lógica testável); isso aqui é só a camada de apresentação em cima delas.

_COR_INCONSCIENTE = "#6b7280"  # cinza neutro -- de propósito sem cor por campo, é caos sem classificação
_CORES_CAMPO = {
    "E": "#2563eb", "S": "#2563eb",      # texto -- azul
    "RE": "#db2777", "RS": "#db2777",    # reação -- rosa
    "CE": "#059669", "CS": "#059669",    # contexto -- verde
    "PIDE": "#7c3aed",                    # pensamento -- roxo
}
_LARGURA_ROTULO = "150px"  # mesma largura pra todo rótulo -- é isso que deixa o espaçamento uniforme


def _chip_html(palavra: str, marcador: str, cor: str, vars_lista: List[str] = None) -> str:
    p, m = _html.escape(str(palavra)), _html.escape(str(marcador))
    vars_html = ""
    if vars_lista:
        vars_txt = _html.escape(", ".join(str(v) for v in vars_lista))
        vars_html = f'<span style="color:{cor};font-size:9px;opacity:0.75;font-style:italic;max-width:150px;">≈ {vars_txt}</span>'
    return (
        f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
        f'background:{cor}22;border:1px solid {cor};border-radius:6px;'
        f'padding:2px 8px;margin:3px;font-family:monospace;line-height:1.3;">'
        f'<span style="color:{cor};font-weight:600;">{p}</span>'
        f'<span style="color:{cor};font-size:10px;opacity:0.85;">{m}</span>'
        f'{vars_html}'
        f'</span>'
    )


def _campo_row_html(rotulo: str, cor: str, palavras: List[str], marcadores: List[str], vars_por_marcador: Dict[str, List[str]] = None) -> str:
    vars_por_marcador = vars_por_marcador or {}
    chips = "".join(
        _chip_html(p, m, cor, [v for v in vars_por_marcador.get(m, []) if v != "0.0"])
        for p, m in zip(palavras, marcadores)
    )
    if not chips:
        chips = f'<span style="color:{cor};opacity:0.55;font-style:italic;">(vazio)</span>'
    return (
        f'<div style="display:flex;align-items:flex-start;margin:2px 0;">'
        f'<div style="min-width:{_LARGURA_ROTULO};flex-shrink:0;font-weight:700;color:{cor};padding-top:6px;">{_html.escape(rotulo)}</div>'
        f'<div style="display:flex;flex-wrap:wrap;">{chips}</div>'
        f'</div>'
    )


def _multivars_row_html(rotulo: str, cor: str, lista: List[str]) -> str:
    """Multivars são variações de FRASE inteira (registradas direto no bloco,
    em Multivars_Texto_*/Multivars_Reacao_*) -- diferente das vars por palavra,
    que vêm do inconsciente. Só aparece se a lista não for vazia."""
    if not lista:
        return ""
    itens = "".join(
        f'<span style="display:inline-block;background:{cor}15;border:1px dashed {cor};'
        f'border-radius:6px;padding:2px 8px;margin:2px;font-size:12px;color:{cor};">'
        f'{_html.escape(str(item))}</span>'
        for item in lista
    )
    return (
        f'<div style="display:flex;align-items:flex-start;margin:2px 0;">'
        f'<div style="min-width:{_LARGURA_ROTULO};flex-shrink:0;font-size:11px;font-style:italic;opacity:0.7;padding-top:6px;">↳ multivars</div>'
        f'<div style="display:flex;flex-wrap:wrap;">{itens}</div>'
        f'</div>'
    )


def _cartao_html(bloco_id, secoes_html: str) -> str:
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.4);border-radius:10px;'
        f'padding:14px 18px;margin:12px 0;">'
        f'<div style="font-weight:800;font-size:15px;margin-bottom:8px;">Bloco {_html.escape(str(bloco_id))}</div>'
        f'{secoes_html}'
        f'</div>'
    )


def _titulo_html(nome: str, dominio: str) -> str:
    return f'<div style="font-size:20px;font-weight:800;margin-bottom:14px;">IM = Universo {_html.escape(str(dominio))} ({_html.escape(str(nome))})</div>'


def visao_bloco_memoria_html(bloco: dict, tokenizar, bloco_inco: dict = None) -> str:
    """Mesma classificação de visao_bloco_memoria, desenhada: cada campo com
    sua cor, rótulos todos com a mesma largura -- fácil de escanear com os
    olhos. Se `bloco_inco` (o par deste bloco no inconsciente) for passado,
    cada palavra de Texto/Reação também mostra suas vars registradas (word-
    level, vindas do inconsciente -- Contexto e Pensamento NUNCA têm var, por
    definição, então não são cruzados aqui). Multivars (frase inteira) vêm
    direto do próprio bloco da memória, não do inconsciente."""
    entrada = bloco.get("entrada", {})
    saida = bloco.get("saidas", [{}])[0]
    tokens_e = entrada.get("tokens", {})
    tokens_s = saida.get("tokens", {})
    pensamento = entrada.get("pensamento_interno", "")
    reacao = entrada.get("reacao", "")
    saida_texto = (saida.get("textos") or [""])[0]
    saida_reacao = saida.get("reacao", "")

    entrada_inco = (bloco_inco or {}).get("Entrada", {})
    saida_inco = (bloco_inco or {}).get("SAÍDA", {})
    vars_e = {m: d.get("vars", []) for m, d in entrada_inco.items()}
    vars_s = {m: d.get("vars", []) for m, d in saida_inco.items()}

    secoes = (
        '<div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.6;margin:6px 0 2px;">ENTRADA</div>'
        + _campo_row_html("Texto (E)", _CORES_CAMPO["E"], tokenizar(entrada.get("texto", "")), tokens_e.get("E", []), vars_e)
        + _multivars_row_html("Multivars (texto)", _CORES_CAMPO["E"], entrada.get("Multivars_Texto_Entrada", []))
        + _campo_row_html("Reação (RE)", _CORES_CAMPO["RE"], tokenizar(reacao) if reacao else [], tokens_e.get("RE", []), vars_e)
        + _multivars_row_html("Multivars (reação)", _CORES_CAMPO["RE"], entrada.get("Multivars_Reacao_Entrada", []))
        + _campo_row_html("Contexto (CE)", _CORES_CAMPO["CE"], tokenizar(entrada.get("contexto", "")), tokens_e.get("CE", []))
        + _campo_row_html("Pensamento (PIDE)", _CORES_CAMPO["PIDE"], tokenizar(pensamento), tokens_e.get("PIDE", []))
        + f'<div style="margin:0 0 6px {_LARGURA_ROTULO};font-size:12px;opacity:0.65;font-style:italic;">texto completo: "{_html.escape(pensamento)}"</div>'
        + '<div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.6;margin:10px 0 2px;">SAÍDA</div>'
        + _campo_row_html("Texto (S)", _CORES_CAMPO["S"], tokenizar(saida_texto), tokens_s.get("S", []), vars_s)
        + _multivars_row_html("Multivars (texto)", _CORES_CAMPO["S"], saida.get("Multivars_Texto_Saida", []))
        + _campo_row_html("Reação (RS)", _CORES_CAMPO["RS"], tokenizar(saida_reacao) if saida_reacao else [], tokens_s.get("RS", []), vars_s)
        + _multivars_row_html("Multivars (reação)", _CORES_CAMPO["RS"], saida.get("Multivars_Reacao_Saida", []))
        + _campo_row_html("Contexto (CS)", _CORES_CAMPO["CS"], tokenizar(saida.get("contexto", "")), tokens_s.get("CS", []))
    )
    return _cartao_html(bloco["bloco_id"], secoes)


def visao_universo_memoria_html(nome: str, dominio: str, blocos: List[dict], tokenizar, blocos_inco: List[dict] = None) -> str:
    inco_por_id = {b["Bloco_id"]: b for b in (blocos_inco or [])}
    corpo = "".join(
        visao_bloco_memoria_html(b, tokenizar, inco_por_id.get(str(b["bloco_id"])))
        for b in blocos
    )
    return _titulo_html(nome, dominio) + corpo


def visao_bloco_inconsciente_html(bloco_inco: dict) -> str:
    """Mesmo caos de visao_bloco_inconsciente, desenhado: uma cor só (cinza,
    de propósito -- não tem classificação aqui), espaçamento uniforme."""
    entrada_pares = _ordenar_por_marcador(list(bloco_inco.get("Entrada", {}).items()))
    saida_pares = _ordenar_por_marcador(list(bloco_inco.get("SAÍDA", {}).items()))
    secoes = (
        _campo_row_html("Entrada", _COR_INCONSCIENTE, [d["token"] for _, d in entrada_pares], [m for m, _ in entrada_pares])
        + _campo_row_html("Saída", _COR_INCONSCIENTE, [d["token"] for _, d in saida_pares], [m for m, _ in saida_pares])
    )
    return _cartao_html(bloco_inco["Bloco_id"], secoes)


def visao_universo_inconsciente_html(nome: str, dominio: str, blocos_inco: List[dict]) -> str:
    return _titulo_html(nome, dominio) + "".join(visao_bloco_inconsciente_html(b) for b in blocos_inco)


def conferir_bloco(bloco: dict) -> Dict[str, str]:
    """Confere se UM bloco tem a forma de um dado concreto de verdade: texto,
    contexto e pensamento presentes (não vazios, não placeholder "{IM}.0"), pelo
    menos uma saída com texto, e marcadores de entrada/saída nunca se
    sobrepondo dentro do próprio bloco."""
    entrada = bloco.get("entrada", {})
    saidas = bloco.get("saidas", [])

    def presente(campo: str) -> bool:
        valor = (entrada.get(campo) or "").strip()
        return bool(valor) and not valor.endswith(".0")

    resultado = {
        "texto_entrada": "bate" if (entrada.get("texto") or "").strip() else "ausente",
        "contexto": "bate" if presente("contexto") else "ausente",
        "pensamento": "bate" if presente("pensamento_interno") else "ausente",
        "saida": "bate" if (saidas and (saidas[0].get("textos") or [""])[0].strip()) else "ausente",
    }

    marcadores_entrada = set(entrada.get("tokens", {}).get("TOTAL", []))
    marcadores_saida = set(saidas[0].get("tokens", {}).get("TOTAL", [])) if saidas else set()
    resultado["marcadores_sem_sobreposicao"] = "bate" if not (marcadores_entrada & marcadores_saida) else "diverge"

    return resultado


def conferir_universo(memoria: dict, inconsciente: dict, dominio: str) -> Dict[str, Any]:
    """Confere a FORMA inteira de um universo:
    - nome presente;
    - cada bloco com a forma de dado concreto (via conferir_bloco);
    - nenhum marcador repetido entre blocos diferentes do MESMO universo
      (Camada 1 -- nunca aproximação, igualdade exata de string);
    - memória (consciente) e inconsciente espelhando os mesmos bloco_ids.
    """
    universo = memoria.get("IM", {}).get(dominio, {})
    inco = inconsciente.get("INCO", {}).get(dominio, {})

    blocos = universo.get("blocos", [])
    blocos_inco = inco.get("Blocos", [])

    diagnostico_blocos = {b["bloco_id"]: conferir_bloco(b) for b in blocos}

    # Unicidade de marcador em TODO o universo (Camada 1).
    todos_marcadores: List[str] = []
    for b in blocos:
        todos_marcadores += b.get("entrada", {}).get("tokens", {}).get("TOTAL", [])
        for s in b.get("saidas", []):
            todos_marcadores += s.get("tokens", {}).get("TOTAL", [])
    vistos: Set[str] = set()
    duplicados: Set[str] = set()
    for m in todos_marcadores:
        if m in vistos:
            duplicados.add(m)
        vistos.add(m)

    # Espelhamento memória <-> inconsciente: todo bloco de um lado precisa ter par no outro.
    ids_memoria = {str(b["bloco_id"]) for b in blocos}
    ids_inco = {b["Bloco_id"] for b in blocos_inco}

    return {
        "nome": "bate" if (universo.get("nome") or "").strip() else "ausente",
        "total_blocos": len(blocos),
        "diagnostico_blocos": diagnostico_blocos,
        "marcadores_duplicados": duplicados,
        "so_na_memoria": ids_memoria - ids_inco,
        "so_no_inconsciente": ids_inco - ids_memoria,
    }
