# numpy-llm: un pequeño modelo de lenguaje Transformer desde cero

> Aunque el proyecto se denomine “LLM desde cero”, aquí construimos un
> **modelo de lenguaje diminuto con fines educativos**, basado en
> principios utilizados en modelos de mayor escala. No es un LLM
> entrenado ni comparable a un modelo de producción.

## Qué construye este proyecto

Un modelo autorregresivo Transformer decoder-only, inspirado en GPT,
implementado solo con Python y NumPy. El objetivo es entender las
operaciones matemáticas, el recorrido de los datos y la función de cada
componente, no obtener texto de calidad.

## Qué puede hacer esta primera fase

- Convertir texto en tokens (tokenizador a nivel de carácter).
- Transformar tokens en embeddings de token + posición.
- Procesar representaciones con bloques Transformer pre-norm.
- Calcular logits para predecir el siguiente token.
- Calcular probabilidades y pérdida de entropía cruzada.
- Generar texto de forma autorregresiva con pesos aleatorios.
- Mostrar dimensiones, parámetros y pesos de atención.
- Guardar y recuperar configuración, vocabulario y pesos.

## Qué añade la fase 2 (entrenamiento, implementada 2026-10-26)

- Backward manual de todos los componentes (Linear, ReLU/MLP,
  LayerNorm, embeddings con `np.add.at`, softmax, atención causal y
  bloques pre-norm con residuales), con registro de gradientes
  (`named_gradients()`), intermediarios con `retain_cache` y
  comprobación numérica por diferencias finitas en float64
  (`gradcheck.py`).
- Optimizadores SGD y AdamW (decay desacoplado solo en matrices de
  capas lineales), clipping por norma global, acumulación de
  gradientes ponderada por objetivos y scheduler constante o coseno
  con warmup.
- Partición train/validation 90/10 sobre el texto (vocabulario solo
  de train, `<UNK>` en validación), lotes con teacher forcing,
  evaluación con stride T, perplexity y baselines uniforme y unigrama.
- Trainer, checkpoints `.npz` atómicos (`last.npz`/`best.npz`) con
  reanudación exacta probada (optimizador, scheduler y generadores).
- Configuraciones reproducibles `configs/smoke.json`,
  `configs/overfit.json` y `configs/demo.json`, y comandos
  `gradcheck`, `trace-step`, `train`, `overfit`, `evaluate`,
  `resume` y `generate --checkpoint`. Detalle en
  `docs/entrenamiento.md`, `docs/derivadas.md`,
  `docs/experimentos.md` y `docs/problemas_frecuentes.md`.
  (`docs/entrenamiento_fase_2.md` era la preparación conceptual
  previa; la implementación es esta fase 2.)

Sin PyTorch/TensorFlow/JAX ni autograd general: solo Python, NumPy y
la biblioteca estándar. Sin cambios de arquitectura ni de tokenizador
respecto a la fase 1; la proyección al vocabulario sigue sin compartir
pesos con los embeddings.

## Instalación

NumPy es la única dependencia externa. Se recomienda un entorno virtual
(en sistemas con Python gestionado externamente, `pip install` directo
puede ser rechazado; dentro del venv funciona):

```bash
cd numpy-llm
python -m venv .venv
.venv/bin/python -m pip install -e .
```

En un entorno que lo permita, el comando equivalente es
`python -m pip install -e .`.

## Pruebas

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Resultado comprobado (fase 2, 2026-10-26): **104 pruebas, OK**
(43 de la fase 1, conservadas, + 61 nuevas de backward, gradcheck,
optimizadores, clipping, acumulación, datos, evaluación y reanudación).

## Entrenar (fase 2)

```bash
PYTHONPATH=src python3 -m numpy_llm gradcheck --scope all --seed 42
PYTHONPATH=src python3 -m numpy_llm trace-step --config configs/smoke.json --corpus data/ejemplo.txt
PYTHONPATH=src python3 -m numpy_llm train --config configs/smoke.json --corpus data/ejemplo.txt --outdir runs/smoke
PYTHONPATH=src python3 -m numpy_llm overfit --config configs/overfit.json --outdir runs/overfit
PYTHONPATH=src python3 -m numpy_llm train --config configs/demo.json --corpus data/ejemplo.txt --outdir runs/demo
PYTHONPATH=src python3 -m numpy_llm evaluate --checkpoint runs/demo/best.npz --corpus data/ejemplo.txt --split val --full
PYTHONPATH=src python3 -m numpy_llm resume --checkpoint runs/demo/last.npz --corpus data/ejemplo.txt
PYTHONPATH=src python3 -m numpy_llm generate --checkpoint runs/demo/best.npz --prompt "La inteligencia" --max-new-tokens 200 --temperature 0.8 --seed 42
```

Resultados reales de estas ejecuciones (2026-10-26): ver
`docs/experimentos.md`. En resumen: gradcheck completo del modelo
diminuto con error absoluto máximo 4,84e-09 (257 componentes);
overfit 1,3996 → 0,0025 con exactitud 100 %; demo 500 actualizaciones
con pérdida de validación inicial 3,9564, final 2,3728 y mejor
2,0707 (paso 300), frente a baselines uniforme 3,8712 y unigrama
2,9928. La validación de la demo cubre solo 1 ventana de 64 objetivos
(corpus de demostración muy pequeño): es una medida, no una garantía
de generalización.

## Inspeccionar las dimensiones

```bash
.venv/bin/python -m numpy_llm inspect --corpus data/ejemplo.txt --seed 42
```

Resultado comprobado con `data/ejemplo.txt`: vocabulario V = 50,
configuración por defecto (C = 64, D = 64, H = 4, L = 2, F = 256),
**110.642 parámetros**, entrada `[1, 8]`, embeddings `[1, 8, 64]`,
Q/K/V_values `[1, 4, 8, 16]`, puntajes y pesos `[1, 4, 8, 8]`, logits
`[1, 8, 50]`, pérdida inicial sobre un lote `[4, 16]` de **3,9422**,
frente a `log(50) = 3,9120`.

## Generar texto

```bash
.venv/bin/python -m numpy_llm generate --corpus data/ejemplo.txt --prompt "La inteligencia" --max-new-tokens 120 --temperature 1.0 --top-k 20 --seed 42
```

## Guardar y recuperar el modelo

```bash
.venv/bin/python -m numpy_llm save --corpus data/ejemplo.txt --output artifacts/modelo_inicial --seed 42
.venv/bin/python -m numpy_llm generate --model-dir artifacts/modelo_inicial --prompt "La inteligencia" --max-new-tokens 120 --seed 42
```

El guardado usa JSON para configuración y vocabulario y NPZ para los
pesos (`allow_pickle=False`, sin pickle). Al cargar se verifican todos
los parámetros, se rechazan los inesperados y se comprueban las formas
y la compatibilidad vocabulario/configuración. Guardar pesos aleatorios
no significa que el modelo esté entrenado.

## Qué resultado es razonable esperar con pesos aleatorios

Texto esencialmente aleatorio y una pérdida cercana a `log(V)`, pero no
exactamente igual. **Esto no indica un fallo de implementación**: es el
resultado esperado antes de entrenar. Ejemplo real con el comando
anterior (semilla 42, top-k 20):

```
La inteligencia¿jí¡?óá¿Asnñ¿áf;r!éz«dóó¿Cj?Do«óbipCApUvfz¡LzéiczCC
```

La calidad literaria del texto aleatorio no es un criterio de éxito;
lo son la corrección matemática, la causalidad y la reproducibilidad.

## Estructura

```
numpy-llm/
├── pyproject.toml
├── README.md
├── data/ejemplo.txt
├── docs/fundamentos.md
├── docs/entrenamiento.md        # fase 2: ciclo completo
├── docs/derivadas.md            # fase 2: derivación por componente
├── docs/experimentos.md         # fase 2: resultados reales
├── docs/problemas_frecuentes.md # fase 2: diagnósticos
├── docs/entrenamiento_fase_2.md # preparación conceptual previa
├── configs/smoke.json | overfit.json | demo.json
├── src/numpy_llm/
│   ├── config.py        # ModelConfig, notación B/T/C/V/D/H/Dh/F/L, dtype
│   ├── tokenizer.py     # Tokenizador a nivel de carácter, <UNK> = 0
│   ├── data.py          # Ventanas x/y, partición train/val, muestreo
│   ├── functional.py    # ReLU, softmax, log-softmax, entropía cruzada + grad
│   ├── losses.py        # Entrada explícita a la pérdida con gradiente
│   ├── layers.py        # Embeddings, Linear, LayerNorm (+ backward)
│   ├── attention.py     # Atención causal multicabeza (+ backward)
│   ├── transformer.py   # Feed-forward y bloque pre-norm (+ backward)
│   ├── model.py         # Modelo completo, named_parameters/gradients
│   ├── gradcheck.py     # Diferencias finitas en float64
│   ├── optimizers.py    # SGD, AdamW, clipping por norma global
│   ├── schedules.py     # Constante y coseno con warmup
│   ├── training.py      # Trainer, acumulación, trace-step
│   ├── evaluation.py    # Pérdida, perplexity, exactitud, baselines
│   ├── checkpoint.py    # Checkpoints .npz atómicos y reanudación
│   ├── generation.py    # Generación autorregresiva, temperatura/top-k
│   ├── serialization.py # JSON + NPZ, sin pickle (fase 1)
│   └── cli.py           # inspect / generate / save / gradcheck / train / ...
└── tests/               # unittest, sin dependencias extra
```

La documentación detallada de cada componente está en
`docs/fundamentos.md`, con intuición, ecuación, formas, código, ejemplo
y errores habituales.

## Visualizador educativo (MVP, Fases 0–1)

Una web Vite + React que visualiza el modelo Python real (recorrido con
formas, tokenizador y vocabulario, inspector de parámetros y
generación). Es local: necesita el backend Python, no es una web
estática. Instrucciones completas en `visualizer/README.md`:

```bash
.venv/bin/python visualizer/server.py
cd visualizer/frontend && npm install && npm run dev
```
