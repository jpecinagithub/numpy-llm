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

## Qué falta para entrenarlo

Todo lo de la fase 2: backpropagation, gradientes, diferenciación
automática, optimizadores, bucles de entrenamiento y actualización de
pesos. Calcular una pérdida no entrena el modelo. Véase
`docs/entrenamiento_fase_2.md`. No hay comandos de entrenamiento, y no
debe inventarse ninguno.

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
.venv/bin/python -m unittest discover -s tests -v
```

Resultado comprobado: **43 pruebas, OK**.

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
├── docs/entrenamiento_fase_2.md
├── src/numpy_llm/
│   ├── config.py        # ModelConfig, notación B/T/C/V/D/H/Dh/F/L
│   ├── tokenizer.py     # Tokenizador a nivel de carácter, <UNK> = 0
│   ├── data.py          # Ventanas x/y desplazadas, sin padding
│   ├── functional.py    # ReLU, softmax, log-softmax, entropía cruzada
│   ├── layers.py        # Embeddings, Linear, LayerNorm
│   ├── attention.py     # Atención causal multicabeza
│   ├── transformer.py   # Feed-forward y bloque pre-norm
│   ├── model.py         # Modelo completo, named_parameters()
│   ├── generation.py    # Generación autorregresiva, temperatura/top-k
│   ├── serialization.py # JSON + NPZ, sin pickle
│   └── cli.py           # inspect / generate / save
└── tests/               # unittest, sin dependencias extra
```

La documentación detallada de cada componente está en
`docs/fundamentos.md`, con intuición, ecuación, formas, código, ejemplo
y errores habituales.
