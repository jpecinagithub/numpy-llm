# Experimentos (fase 2): resultados reales

Fecha de ejecución: 2026-10-26. Entorno: Python 3.12.3, NumPy 1.26.4.
Semilla base: 42 en todas las configuraciones. Todos los números de
esta página provienen de ejecuciones reales citadas con su comando;
ninguno es una estimación ni un objetivo declarado como resultado.

Corpus: `data/ejemplo.txt` local (sin descargas), 1127 caracteres.
Vocabulario completo: 50 tokens (49 caracteres + `<UNK>`).
Vocabulario de train (90 %): 48 tokens. Partición: train 1014 tokens,
validación 113 tokens (SHA-256
`fc209e7397849ee5584094b9335fa53db2c85fdb4cac5a4b9b9a2918dc2395a7`,
guardado en `runs/demo/config.json`). Tasa de `<UNK>` en validación:
0,0177 (2 caracteres de validación ausentes en train); en train: 0,0.

## 1. Gradcheck

```bash
PYTHONPATH=src python3 -m numpy_llm gradcheck --scope all --seed 42
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

- Modelo diminuto (B=2, T=3, V=5, D=4, H=2, F=8, L=1), modo completo:
  **257 componentes, pasa, error absoluto máximo 4,84e-09**
  (`tests/test_gradcheck.py`, determinista).
- CLI `--scope all` (tokens aleatorios con semilla 42): pasa, error
  absoluto máximo **1,15e-08**, 257 componentes.
- CLI `--scope fast` (3 componentes por tensor): pasa, error absoluto
  máximo 1,15e-08, 66 componentes.
- Tolerancias: `h = 1e-5`, `atol = 1e-7`, `rtol = 1e-4`, float64.
  El error relativo máximo (1,0) corresponde a gradientes casi nulos
  y no indica fallo: ver `docs/derivadas.md`.
- Suite completa: **104 pruebas, OK** (43 de fase 1 + 61 nuevas).

## 2. Trace-step (diagnóstico)

```bash
PYTHONPATH=src python3 -m numpy_llm trace-step --config configs/smoke.json --corpus data/ejemplo.txt
```

Salida real: entrada `[2, 8]`, logits `[2, 8, 48]`, pérdida **3,8777**,
todos finitos, normas de gradiente por grupo: embeddings 1,1722,
bloques 1,5171, final_ln 0,0328, lm_head 1,0585. No actualiza pesos.

## 3. Smoke (20 actualizaciones)

```bash
PYTHONPATH=src python3 -m numpy_llm train --config configs/smoke.json --corpus data/ejemplo.txt --outdir runs/smoke
```

Configuración: D=16, H=2, L=1, F=32, C=T=8, B=2, AdamW lr=1e-3
constante, weight_decay=0, acumulación=1.

- Evaluación inicial (val, completa, 112 objetivos): loss **3,8712**,
  perplexity 47,998, exactitud 0,0625.
- Evaluación final (val, paso 20): loss **3,6450**, perplexity
  38,282, exactitud 0,2054.
- Mejor val: 3,6450 (paso 20). Tokens procesados: 320.
- Baselines: uniforme loss 3,8712 / perplexity 48,0; unigrama (train)
  loss 2,9928 / perplexity 19,942 sobre val.
- Lectura honesta: con 20 actualizaciones el modelo mejora sobre su
  inicio y sobre el uniforme, pero sigue por debajo del unigrama.
  Es un smoke test (que el sistema entrena sin NaN y los parámetros
  cambian), no una demostración de calidad.

## 4. Overfit deliberado (hasta 300 actualizaciones)

```bash
PYTHONPATH=src python3 -m numpy_llm overfit --config configs/overfit.json --outdir runs/overfit
```

Configuración: la misma arquitectura que smoke, AdamW lr=3e-3
constante, weight_decay=0, lote fijo `abcabc...` (B=2, T=8), sin
validación (es una prueba de memorización, no de generalización).

- Pérdida inicial: **1,3996**. Pérdida final: **0,0025**
  (0,2 % de la inicial; criterio: < 20 % → SUPERADO).
- Exactitud final sobre el lote memorizado: **1,0000**
  (criterio: ≥ 95 % → SUPERADO). Actualizaciones: 300, semilla 42.
- Muestra greedy después (prompt `abc`):
  `abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc`
- Esta prueba, por sí sola, no demuestra que todos los componentes
  sean correctos: debe acompañarse del gradcheck (sección 1).

## 5. Demo (500 actualizaciones)

```bash
PYTHONPATH=src python3 -m numpy_llm train --config configs/demo.json --corpus data/ejemplo.txt --outdir runs/demo
```

Configuración: D=64, H=4, L=2, F=256, C=T=64, B=4, AdamW,
lr_peak=1e-3, lr_min=1e-4, warmup=50, weight_decay=0,01,
acumulación=1, evaluación cada 50. Duración real: ~40 s.

- Evaluación inicial (val): loss **3,9564**, perplexity 52,269,
  exactitud 0,0156 (64 objetivos).
- Evaluación final (val, paso 500): loss **2,3728**, perplexity
  10,727, exactitud 0,3281.
- Mejor validación: loss **2,0707**, perplexity **7,930**, exactitud
  0,3750, en el **paso 300** (`runs/demo/best.npz`).
- Tokens objetivo procesados: 128.000.
- Baselines: uniforme loss 3,8712 / perplexity 48,0; unigrama loss
  2,9928 / perplexity 19,942. El modelo final y el mejor superan a
  ambos baselines en este corpus y protocolo.
- Verificación independiente del mejor checkpoint:

```bash
PYTHONPATH=src python3 -m numpy_llm evaluate --checkpoint runs/demo/best.npz --corpus data/ejemplo.txt --split val --full
# loss=2,0707 perplexity=7,9300 accuracy=0,3750 objetivos=64 cobertura=full
# Ventanas: 1 / 1, tokens fuera: 48
PYTHONPATH=src python3 -m numpy_llm evaluate --checkpoint runs/demo/last.npz --corpus data/ejemplo.txt --split val --full
# loss=2,3728 perplexity=10,7269 accuracy=0,3281 objetivos=64 cobertura=full
```

- Límites, en serio: la validación cubre **solo 1 ventana de 64
  objetivos** (la partición de validación tiene 113 tokens y T=64,
  así que el resto, 48 tokens, queda fuera del protocolo de stride
  T). Con tan pocos objetivos, la diferencia entre el paso 300 y el
  500 es ruidosa y no permite afirmar una tendencia de
  generalización. Además, el modelo final es peor que el mejor: con
  un corpus tan pequeño, 500 actualizaciones sobreajustan. El
  checkpoint `best.npz` existe precisamente para ese caso.

## 6. Generación antes y después (demo)

Muestras del propio entrenador (`runs/demo/samples_before.txt` y
`samples_after.txt`, mismo prompt `La inteligencia`, misma semilla
de muestreo; 100 tokens nuevos, temperatura 0,8):

Antes (pesos aleatorios):

```
La inteligenciaéen«?u:. !lédlhceúch¡cD¡EDolvmn
Ljx a.rxñ<UNK>f?«cí!s»<UNK>iifcprcdth«»úhu,jLdíí¿b»mém¡nxñU¿<UNK>l«xr¡zíejémynív
```

Después (500 actualizaciones):

```
La inteligenciarten iaba erten lúmos endero en ero endizafir?: me endertl mestroda e sto prones.
Es esiroso es este
```

Generación con el mejor checkpoint (paso 300, temperatura 0,8,
200 tokens, semilla 42):

```
La inteligenciar ué «l¿denúmtengarondode e cistende el ero rodera aprode caco pomodal encacingulara al ciadendigrarte y mo do ele sprente e ha codonen y pamosilabre qualaciror ta ento vil ua ezsostenten pes colarode
```

Greedy con el mejor checkpoint (100 tokens):

```
La inteligenciar ende ender ero ender endo mo endo endo ero endo ero en en en en en en en un amen un amoro un untor
```

Lectura: después de entrenar aparecen patrones del corpus
(fragmentos de palabras, espacios, estructura de frase) y desaparece
el ruido de caracteres aleatorios, pero el texto sigue sin ser
coherente y el greedy cae en repeticiones (`ender`, `endo`, `en`).
Que el texto sea más plausible no significa que el modelo
"entienda" el texto: describe patrones observados y sus límites.

## 7. Reanudación

Prueba de integración (`tests/test_resume.py`): ejecución A de 6
actualizaciones consecutivas frente a ejecución B de 2 + guardar +
cargar + continuar hasta 6 con el mismo horizonte total. Coinciden
parámetros finales, estados m/v, contador t, historial de tasas,
próximos lotes, pérdidas y contadores (tolerancia 1e-6/1e-7 en
float32; las operaciones son las mismas, así que las diferencias son
de redondeo).

Prueba por CLI: checkpoint parcial de smoke (10/20) reanudado con
`resume` completa hasta 20 con evaluación final **3,6450** y mejor
**3,6450 (paso 20)**, idénticos a la ejecución ininterrumpida de
smoke. Reanudar `runs/smoke/last.npz` (20/20) informa de que el
entrenamiento ya completó su objetivo, sin reiniciar ni extender el
scheduler.

## 8. Qué NO se demuestra aquí

- Generalización a texto nuevo (corpus diminuto, validación de 64
  objetivos en la demo).
- Hiperparámetros óptimos: las configuraciones son puntos de partida
  educativos, no recetas universales.
- Calidad de generación: el modelo es diminuto y el corpus, un texto
  de ejemplo de 1127 caracteres.

Pasos posteriores posibles (fuera de esta fase): corpus mayor con
validación amplia, más actualizaciones, y solo entonces comparar
configuraciones; nada de eso cambia la corrección del backward, que
ya está verificada por gradcheck.
