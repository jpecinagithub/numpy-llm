# Visualizador educativo de numpy-llm (Fases 1 y 2)

Web app Vite + React + TypeScript que visualiza el modelo Python real de
`numpy-llm`. No reescribe el modelo en JavaScript: el frontend pide los
resultados a un pequeño backend local en Python que importa `numpy_llm`
tal cual, el mismo código que usa la CLI.

La web está separada en dos partes muy claras, porque es un proyecto
educativo y mezclarlas confunde:

- **FASE 1 · Cómo predice (sin entrenar).** El modelo principal tiene
  pesos aleatorios: sirve para ver la máquina, no para obtener buen
  texto.
- **FASE 2 · Cómo aprende (entrenamiento).** Conceptos, laboratorios
  con código real y los resultados del entrenamiento real del repo.

## Qué incluye la Fase 1

- **Recorrido:** texto → tokens → formas de los tensores en cada etapa
  (`[B, T]` → `[B, T, D]` → `[B, T, V]`).
- **Tokenizador y vocabulario:** tira de tokens con identificadores,
  desconocidos marcados como `<UNK>` y lista buscable de los 50 tokens.
- **Inspector del modelo:** configuración, número total de parámetros
  (110.642), desglose por componente y tabla completa de parámetros.
- **Generación:** prompt, tokens nuevos, temperatura, top-k y semilla,
  con el texto completo y los identificadores generados.
- **Límites de la Fase 1:** qué sí se puede ver y qué no se puede
  esperar sin entrenamiento. Calcular la pérdida, por sí solo, no
  entrena.

## Qué incluye la Fase 2

- **El ciclo de entrenamiento explicado paso a paso:** datos con
  teacher forcing → forward → pérdida (entropía cruzada) → gradiente
  (backward, regla de la cadena) → actualización (solo
  `optimizer.step`, AdamW) → volver a predecir. Incluye las
  confusiones habituales: pérdida ≠ entrenar, backward ≠ actualizar,
  train ≠ validation, y qué controlan la tasa de aprendizaje, el
  clipping y el warmup/coseno.
- **Laboratorio “Mide la pérdida y el gradiente”** (`POST
  /api/loss-demo`): una ventana de 9 tokens de un texto del usuario,
  con un modelo nuevo sin entrenar. Muestra pérdida, perplexity,
  exactitud, probabilidad del carácter correcto en cada posición y
  norma del gradiente por grupo. Mide, no entrena, y no toca el modelo
  principal.
- **Laboratorio “Entrena en miniatura”** (`POST /api/train-demo`):
  un mini-entrenamiento real con el código de la Fase 2 (modelo
  diminuto aparte, D=16, H=2, L=1, F=32, vocabulario de 4 tokens, un
  solo lote fijo `abcabc…`, AdamW lr 3e-3). Con 60 actualizaciones la
  pérdida pasa de 1,4342 a 0,0506, la exactitud de 12,5 % a 100 % y el
  greedy pasa de ruido a repetir `abc`. Es una prueba de
  memorización, no de generalización, y la web lo dice así.
- **El entrenamiento real del repo:** configuración demo (D=64, H=4,
  L=2, F=256, AdamW con warmup 50 y coseno 1e-3 → 1e-4, 500
  actualizaciones), validation inicial 3,9564 / perplexity 52,27,
  final 2,3728 / 10,73 y mejor checkpoint (paso 300) 2,0707 / 7,93,
  frente a baselines uniforme (3,8712 / 48,0) y unigrama (2,9928 /
  19,94). Con el límite honesto: la validation cubre una sola ventana
  de 64 objetivos, así que no demuestra generalización.

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
- `POST /api/loss-demo` (Fase 2: pérdida y gradiente de una ventana)
- `POST /api/train-demo` (Fase 2: mini-entrenamiento `abcabc…`)

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

- El modelo **principal no está entrenado**: sus pesos son aleatorios
  y el texto generado en la Fase 1 es esencialmente aleatorio. La web
  lo avisa en pantalla. Los laboratorios de la Fase 2 usan modelos
  aparte: uno nuevo sin entrenar para medir la pérdida, y uno diminuto
  para el mini-entrenamiento. Ninguno entrena el modelo principal.
- Esta versión es **local**: no está pensada para desplegarse como web
  estática en Vercel, porque necesita el backend Python. Una versión
  compartible sin backend (exportación de pesos o Pyodide) es una fase
  posterior, fuera de este MVP.
- El frontend de la Fase 1 solo observa y genera. El de la Fase 2
  lanza cálculos reales en el backend (medir y mini-entrenar), pero
  siempre sobre modelos separados del principal.

## Cómo ver el entrenamiento en funcionamiento (guía para alumnos)

1. Arranca backend y front como arriba y abre la web.
2. Fase 1: analiza “La inteligencia” y genera texto: verás que es
   aleatorio. Eso es lo que hay que cambiar.
3. Fase 2, laboratorio de pérdida: mide una ventana y fíjate en la
   probabilidad del carácter correcto (cerca del 2 % con pesos
   aleatorios) y en la norma del gradiente por grupo. Eso es el error
   y su reparto, antes de mover ningún peso.
4. Fase 2, laboratorio de mini-entrenamiento: pulsa “Entrenar en
   miniatura” (60 pasos) y observa la curva bajar y el greedy pasar a
   `abcabc…`. Eso es aprender, en el caso más simple posible.
5. Fuera de la web, en la CLI: `overfit` (memorización completa),
   `train` con `configs/demo.json` (el entrenamiento real) y
   `evaluate` sobre `runs/demo/best.npz`. Los comandos exactos están
   en la propia sección 8 de la web.
