# Problemas frecuentes (fase 2): síntomas y comprobaciones

Regla general: ante un síntoma, primero se comprueba la
implementación (gradcheck, desplazamiento de objetivos, registro de
parámetros, escala de gradientes, optimizador) y solo después se
tocan los hiperparámetros. Cambiar la tasa de aprendizaje para tapar
un backward incorrecto solo esconde el error.

## La pérdida no baja

Comprobaciones, en orden:
1. `gradcheck --scope all`: si falla, el backward tiene un error.
   No sigas entrenando.
2. Desplazamiento de objetivos: `y` debe ser `x` desplazado una
   posición (`data.sample_batch`, `build_examples`). Si `y == x`, el
   modelo aprende a copiar la entrada y la pérdida se estanca en un
   valor engañoso.
3. ¿Se llama a `optimizer.step`? `backward` solo acumula gradientes;
   sin step, los pesos no cambian (compruébalo con `trace-step` y
   comparando pesos antes/después de un step).
4. ¿Se llama a `zero_grad` antes de cada grupo? Sin él, gradientes
   viejos contaminan el paso.
5. Tasa de aprendizaje: demasiado pequeña se ve como "no baja" en
   pocos pasos; comprueba `metrics.jsonl` (columna `lr`).
6. Prueba de sobreajuste (`overfit`): si ni siquiera memoriza
   `abcabc...`, el problema es de implementación, no de datos.

## La pérdida se convierte en NaN

1. Revisa `grad_norm_before` en `metrics.jsonl`: un salto brusco
   antes del NaN apunta a un paso demasiado grande o a un lote
   patológico.
2. Comprueba que los gradientes son finitos antes del clipping: el
   código detiene la ejecución en ese punto con el nombre del
   parámetro. Un NaN en gradientes con pérdida finita apunta al
   backward (típicamente, operar con los `-inf` de la máscara causal
   en lugar de usar los pesos de atención, que ya son 0 ahí).
3. Los `-inf` de los puntajes enmascarados son intencionales: no son
   un error. El error sería un NaN en logits, pérdida o gradientes.
4. Baja la tasa de aprendizaje o el `max_grad_norm` solo después de
   descartar (2).

## Gradientes demasiado grandes

1. Mide: `trace-step` da la norma por grupo. ¿Es un grupo concreto
   (p. ej. `lm_head`) o todos?
2. ¿Doble división por el tamaño del lote? La pérdida ya divide entre
   M una sola vez (`cross_entropy_with_grad`); si alguna capa volviera
   a dividir por B o T, los gradientes serían demasiado pequeños, no
   grandes — pero una normalización olvidada en la acumulación
   (pesos `M_i/M_total` ausentes) infla el gradiente en proporción al
   número de microlotes. Compruébalo con `test_gradient_accumulation`.
3. Clipping: verifica en `metrics.jsonl` que `clipped` es true y que
   `grad_norm_after` es `max_grad_norm` cuando se aplica.

## Gradientes inesperadamente nulos

1. Primero, descarta que sean legítimos: ciertas simetrías del
   softmax pueden anular gradientes de algunos biases, y las
   posiciones de embedding no usadas tienen gradiente 0 por
   construcción. Un cero aislado no es un error.
2. Si TODOS los gradientes son cero: ¿el `dlogits` llega al modelo?
   ¿`backward` se llamó con `retain_cache=True` previo? ¿La pérdida
   depende realmente de los pesos (o se está derivando una constante)?
3. Si solo una rama es cero: revisa las sumas residuales
   (`dX = dH + LN.backward(dA)`) y la suma de las tres ramas de
   atención (`dX_Q + dX_K + dX_V`): una sobrescritura en lugar de
   una suma anula una contribución entera.
4. Embeddings: si un token frecuente tiene gradiente cero, revisa que
   el backward usa `np.add.at` y que el forward de entrenamiento
   guardó los identificadores correctos.

## División doble por el tamaño del lote

Síntoma: la pérdida baja muchísimo más despacio de lo esperado y los
gradientes son ~B×T veces menores de lo que da el gradcheck (que usa
la pérdida correcta). Causa: dividir `dlogits` entre M y además
dividir en alguna capa, o dividir la pérdida acumulada entre el
número de microlotes además de ponderar por `M_i/M_total`.
Comprobación: `compute_batch_gradients` con un lote grande frente al
mismo lote dividido debe dar los mismos gradientes
(`test_gradient_accumulation.py`).

## Embeddings que pierden contribuciones repetidas

Síntoma: el gradcheck de embeddings falla solo en filas de tokens
repetidos, y el resto pasa. Causa: `grad_E[ids] += dX` en lugar de
`np.add.at`. NumPy bufferiza la indexación avanzada y la última
contribución sobrescribe a las anteriores. Comprobación:
`test_backward_layers.py::test_repeated_tokens_accumulate` (token 1
aparece dos veces; su gradiente debe ser la suma).

## Errores en transposiciones

Síntoma: gradcheck falla en atención o en capas lineales con formas
que "casi" coinciden, o errores de forma en el backward. Causas
habituales: confundir `K.transpose(0, 1, 3, 2)` con otra permutación;
invertir mal el merge de cabezas (merge es
`transpose(0, 2, 1, 3) + reshape`; su inversa es
`reshape + transpose(0, 2, 1, 3)`, no la misma operación);
olvidar el `/ sqrt(Dh)` en `dQ`/`dK` (el gradcheck lo detecta como
un factor constante de error). Comprobación: `test_split_merge_inverse`
y el gradcheck de atención.

## Máscara causal incorrecta

Síntomas: la prueba causal de la fase 1 falla (cambiar el sufijo
altera los logits del prefijo); o la pérdida es sospechosamente baja
(el modelo "ve" el futuro). Causas: enmascarar después de softmax
(las probabilidades ya no suman 1); enmascarar con 0 en lugar de
`-inf` (el futuro recibe probabilidad no nula); o no anular el
gradiente en posiciones prohibidas en el backward. Comprobación:
pesos de atención exactamente 0 en `j > i`, y prefijos invariantes
(`test_backward_attention.py`, `test_backward_model.py`).

## Reanudación que produce resultados distintos

1. ¿Se reanudó o solo se cargaron pesos? Cargar pesos con un
   optimizador nuevo (`t = 0`, `m = v = 0`) NO es reanudar.
2. ¿Mismo horizonte del scheduler? La primera parte no debe
   configurarse como un entrenamiento de K pasos: el scheduler se
   construye siempre con el total N original.
3. ¿Mismo vocabulario y mismo corpus (SHA-256)? Un orden de tokens
   distinto invalida todos los pesos.
4. ¿Generadores restaurados antes del siguiente lote? El checkpoint
   guarda `train_rng_state`; `Trainer.from_checkpoint` lo restaura.
5. Comprobación definitiva: `tests/test_resume.py` (A: N pasos
   seguidos; B: K + guardar + cargar + continuar hasta N; comparar
   parámetros, m/v, t, tasas, próximos lotes y pérdidas).

## Validación artificialmente buena por filtración

Síntomas: validación mucho mejor que train, o mejor que el unigrama
por un margen increíble en un corpus pequeño. Causas: vocabulario
construido con todo el corpus (incluida la validación); ventanas de
validación que solapan texto de train; o evaluar sobre train
llamándolo validación. Comprobaciones: el vocabulario debe venir
solo de train (`split_corpus`); la partición se hace sobre el texto
antes de construir ventanas; los caracteres de validación ausentes
en train deben aparecer como `<UNK>` (tasa registrada en
`config.json` del run). Ojo: una partición sin índices compartidos no
garantiza independencia semántica si el corpus tiene fragmentos
duplicados.

## Texto generado que repite secuencias

Es el comportamiento esperado de un modelo diminuto con muestreo
greedy (o temperatura baja) sobre un corpus pequeño: el greedy
elige siempre el token más probable y puede entrar en un ciclo
(`ender endo en en en...` en la demo). No es, por sí solo, un error
de implementación. Comprobaciones: que con temperatura > 0 el texto
varíe entre semillas; que el greedy sea determinista; y que la
métrica principal siga siendo la pérdida/perplexity, no la
impresión subjetiva del texto.
