# Bot Predictor Automático de Fútbol

Bot en Python + Streamlit para buscar partidos, analizar cuotas, simular miles de escenarios y generar pronósticos pre-partido.

## Qué hace

- Busca partidos automáticamente con The Odds API.
- Permite elegir cualquier partido encontrado.
- Analiza cuotas 1X2 y mercados de goles.
- Simula miles de escenarios con Monte Carlo/Poisson.
- Calcula probabilidades de local, empate, visitante, doble oportunidad, over/under y ambos marcan.
- Genera ranking de picks y ticket sugerido.
- Integra Gemini opcional para análisis con IA y búsqueda de Google.
- Incluye scripts para ejecutar en Windows y subir a GitHub.

## Uso rápido en Windows

1. Extrae el ZIP.
2. Entra a la carpeta del bot.
3. Dale doble clic a `run_windows.bat`.
4. Se instalará todo y se abrirá en Chrome.

URL local:

```text
http://127.0.0.1:8501
```

## APIs opcionales

Puedes pegar las claves directamente en la interfaz.

### The Odds API
Sirve para buscar deportes, partidos y cuotas.

### Football-Data
Sirve para buscar calendarios y partidos por competición.

### Gemini API
Sirve para generar análisis textual con IA. Si tu cuenta/modelo lo permite, puede usar Google Search Grounding.

## Subir a GitHub

Opción automática:

```bat
github_crear_y_subir.bat
```

Necesita Git y GitHub CLI instalados.

Opción manual:

```bat
github_subir_manual.bat
```

## Advertencia

Este bot no garantiza ganancias. Las apuestas tienen riesgo. El bot estima probabilidades y ayuda a evitar jugadas impulsivas.
