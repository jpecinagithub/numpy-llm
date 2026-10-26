# Visualizador educativo de numpy-llm (MVP, Fases 0–1)

Web app Vite + React + TypeScript que visualiza el modelo Python real de
`numpy-llm`. No reescribe el modelo en JavaScript: el frontend pide los
resultados a un pequeño backend local en Python que importa `numpy_llm`
tal cual, el mismo código que usa la CLI.

## Qué incluye este MVP

- **Recorrido:** texto → tokens → formas de los tensores en cada etapa
  (`[B, T]` → `[B, T, D]` → `[B, T, V]`).
- **Tokenizador y vocabulario:** tira de tokens con identificadores,
  desconocidos marcados como `<UNK>` y lista buscable de los 50 tokens.
- **Inspector del modelo:** configuración, número total de parámetros
  (110.642), desglose por componente y tabla completa de parámetros.
- **Generación:** prompt, tokens nuevos, temperatura, top-k y semilla,
  con el texto completo y los identificadores generados.
- **Límites de la Fase 1:** qué sí se puede ver, qué no se puede esperar
  y qué cambiaría la Fase 2.

Fuera de este MVP, a propósito: mapa de calor de atención, gráfico de
logits y probabilidades, generación paso a paso y explorador de pérdida
por posición. Esas son las siguientes fases del visualizador, no de
esta entrega.

## Cómo ponerlo en marcha (local)

Necesitas dos terminales, desde la raíz del repositorio `numpy-llm`.

### Terminal 1: backend Python

```bash
.venv/bin/python visualizer/server.py
```

En Windows (PowerShell):

```powershell
.venv\Scripts\python visualizer/server.py
```

El backend escucha en `http://127.0.0.1:8000` y expone:

- `GET /api/health`
- `GET /api/info`
- `POST /api/tokenize`
- `POST /api/forward`
- `POST /api/generate`

### Terminal 2: frontend Vite

La primera vez, instala las dependencias del frontend:

```bash
cd visualizer/frontend
npm install
npm run dev
```

Después, abre la URL que muestra Vite (normalmente
`http://localhost:5173`). Vite redirige `/api` al backend local.

## Comprobaciones realizadas

- Backend verificado contra la CLI: vocabulario 50, 110.642 parámetros,
  formas `[1, 8]`, `[1, 8, 64]` y `[1, 8, 50]` para “La intel”, y
  generación con semilla 42 y top-k 20 idéntica a la de la CLI.
- Frontend compilado con `npm run build` (TypeScript + Vite) sin errores.
- Proxy de Vite verificado (`/api/health`, `/api/info` y
  `/api/generate` responden a través del servidor Vite, con la misma
  generación que la CLI para semilla 42 y top-k 20).

## Notas importantes

- El modelo **no está entrenado**: sus pesos son aleatorios y el texto
  generado es esencialmente aleatorio. La web lo avisa en pantalla.
- Esta versión es **local**: no está pensada para desplegarse como web
  estática en Vercel, porque necesita el backend Python. Una versión
  compartible sin backend (exportación de pesos o Pyodide) es una fase
  posterior, fuera de este MVP.
- El frontend no modifica pesos ni entrena. Solo observa y genera.
