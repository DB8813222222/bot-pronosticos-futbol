from __future__ import annotations

from typing import Any, Dict, Optional


def gemini_analysis(
    api_key: str,
    model: str,
    match_context: Dict[str, Any],
    use_google_search: bool = True,
) -> str:
    if not api_key:
        return "Gemini no está activado. Pega tu API Key para generar análisis con IA."
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        return f"No se pudo importar google-genai. Instala requirements.txt. Detalle: {exc}"

    prompt = f"""
Actúa como analista profesional de apuestas pre-partido.
No prometas ganancias seguras. Da un análisis claro, directo y prudente.

Datos del partido:
{match_context}

Entrega en español:
1) Resumen rápido
2) Lectura de cuotas y riesgo
3) Mejores 3 mercados del partido
4) Pick más seguro
5) Pick que evitarías
6) Mensaje final corto para enviar por WhatsApp
""".strip()

    try:
        client = genai.Client(api_key=api_key)
        if use_google_search:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool])
        else:
            config = types.GenerateContentConfig()

        response = client.models.generate_content(
            model=model or "gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        return response.text or "Gemini no devolvió texto."
    except Exception as exc:
        # Reintento sin Google Search por compatibilidad de modelos/cuentas.
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model or "gemini-2.5-flash",
                contents=prompt,
            )
            return (response.text or "Gemini no devolvió texto.") + "\n\nNota: el análisis salió sin Google Search porque el modo con búsqueda falló."
        except Exception as exc2:
            return f"No se pudo generar análisis con Gemini. Error: {exc2}"
