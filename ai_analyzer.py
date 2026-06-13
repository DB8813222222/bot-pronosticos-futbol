from __future__ import annotations

from typing import Any, Dict


def gemini_analysis(
    api_key: str,
    model: str,
    match_context: Dict[str, Any],
    use_google_search: bool = True,
) -> str:
    if not api_key:
        return "Gemini no está activado. Pega tu API Key para generar análisis con IA."

    prompt = f"""
Actúa como analista profesional de apuestas pre-partido.

Reglas:
- No prometas ganancias seguras.
- Habla claro, directo y prudente.
- Explica riesgos.
- Da recomendaciones útiles para una banca pequeña.
- Si no hay datos suficientes, dilo claramente.

Datos del partido:
{match_context}

Entrega en español:

1) Resumen rápido del partido
2) Lectura de cuotas y riesgo
3) Probabilidad más lógica según los datos
4) Mejores 3 mercados del partido
5) Pick más seguro
6) Pick con mejor valor
7) Pick que evitarías
8) Mensaje final corto para enviar por WhatsApp
""".strip()

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        return f"No se pudo importar google-genai. Revisa requirements.txt. Detalle: {exc}"

    try:
        client = genai.Client(api_key=api_key)

        selected_model = model or "gemini-2.5-flash"

        if use_google_search:
            try:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0.4,
                )

                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                )

                if response and getattr(response, "text", None):
                    return response.text

            except Exception:
                pass

        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )

        if response and getattr(response, "text", None):
            return response.text

        return "Gemini respondió, pero no devolvió texto."

    except Exception as exc:
        return (
            "No se pudo generar análisis con Gemini.\n\n"
            f"Error: {exc}\n\n"
            "Posibles causas:\n"
            "- La clave API no es válida para Gemini.\n"
            "- La clave empieza con AQ. y tu cuenta requiere OAuth en vez de API Key clásica.\n"
            "- El modelo elegido no está disponible para tu cuenta.\n"
            "- Prueba con el modelo gemini-2.0-flash o gemini-1.5-flash."
        )
