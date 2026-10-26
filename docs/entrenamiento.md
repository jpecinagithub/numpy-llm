# Entrenamiento (fase 2): el ciclo completo

```
datos → predicciones → pérdida → gradientes → actualización → nuevas predicciones
```

## 1. Qué significa una derivada respecto a un peso

La pérdida `loss(θ)` es un número que mide lo mal que el modelo
predice el siguiente token. La derivada `∂loss/∂θ_i` (abreviada `dθ_i`)
dice cuánto cambiaría la pérdida si `θ_i` aumentase un poco, manteniendo
el resto fijo. Si `dθ_i` es positivo, bajar `θ_i` baja la pérdida; si
es negativo, subir `θ_i` la baja. El gradiente es el vector de todas
esas derivadas. Entrenar es, en esencia, seguir esa información
cuesta abajo, paso a paso.

Ejemplo concreto: si el peso que conecta la característica "la
posición anterior era 'a'" con el logit de 'b' tiene gradiente
negativo grande, el optimizador aumentará ese peso, y el modelo
asignará más probabilidad a 'b' después de 'a'.

## 2. La regla de la cadena

Cada capa es una función de la anterior: `loss = f_L(...f_2(f_1(x)))`.
La regla de la cadena dice que la derivada respecto a una capa
intermedia es el producto de las derivadas de todas las capas
posteriores, aplicadas en orden inverso. Por eso el backward recorre
el modelo al revés: empieza con `dlogits` (el gradiente respecto a
los logits, que se calcula directamente desde la pérdida) y va
propagando hacia atrás por la proyección, la LayerNorm final, los
bloques y los embeddings. Cada componente solo necesita saber derivar
su propia operación; la composición hace el resto.

## 3. Backward frente a optimizer.step

Son dos responsabilidades distintas, y confundirlas es un error común:

- `backward(dlogits)` **calcula y acumula gradientes**. No toca los
  pesos. Se puede llamar varias veces (acumulación) y los gradientes
  se suman.
- `optimizer.step(lr)` **modifica los pesos** (in-place, para no
  romper las referencias del optimizador), usando los gradientes
  acumulados. Es el único lugar donde los pesos cambian.

Entre ambos van, en orden obligatorio: comprobar finitos → medir la
norma → clipping → step. Y antes del siguiente grupo de microlotes,
`zero_grad()`.

## 4. Por qué hay que limpiar los gradientes, y cuándo se acumulan

`backward` suma (`grad += contribución`) por diseño: así funcionan las
ramas residuales, los tokens repetidos y la acumulación de microlotes.
Si no se limpiaran, el gradiente del lote nuevo se sumaría al del
lote anterior y el optimizador usaría una mezcla sin sentido de
lotes distintos. Por eso el bucle hace `zero_grad()` una vez antes de
cada grupo de microlotes.

La acumulación deliberada es la excepción controlada: varios
microlotes comparten un mismo `zero_grad()`, cada uno contribuye con
peso `M_i / M_total`, y solo al final se aplica clipping y step. Así,
un lote grande dividido en microlotes da (salvo redondeo) los mismos
gradientes que el lote entero: lo comprueba
`tests/test_gradient_accumulation.py`.

## 5. Qué controla la tasa de aprendizaje

`η` (eta) es el tamaño del paso: `θ = θ - η * g` (SGD) o la versión
con momentos de AdamW. Demasiado grande y la pérdida oscila o diverge;
demasiado pequeña y el entrenamiento no avanza en un tiempo razonable.
El scheduler la programa por actualización: constante (`lr_peak`
siempre) o warmup lineal + descenso coseno hasta `lr_min`. El warmup
evita pasos grandes con gradientes iniciales ruidosos; el descenso
coseno afina al final. El scheduler depende de actualizaciones del
optimizador, nunca de microlotes.

## 6. Qué conserva el estado de AdamW

Por cada parámetro, dos medias móviles: `m` (media del gradiente) y
`v` (media del cuadrado del gradiente), más el contador `t`. `m`
suaviza la dirección; `v` adapta el paso por parámetro (parámetros
con gradientes grandes e inestables reciben pasos efectivos menores).
La corrección de sesgo (`m / (1 - β1**t)`, `v / (1 - β2**t)`) compensa
que `m` y `v` empiecen en cero. Sin restaurar `m`, `v` y `t`, una
"reanudación" no es exacta: por eso el checkpoint los guarda todos,
junto con los generadores y el horizonte del scheduler.

El weight decay de AdamW está desacoplado: `θ -= η * wd * θ` se resta
directamente, sin pasar por `m`/`v` ni sumarse al gradiente. En este
proyecto se aplica solo a matrices de capas lineales (`*.weight`) y se
excluye de biases, LayerNorm y embeddings (`optimizers.decay_for_name`).

## 7. Qué hace el clipping

Si la norma global de todos los gradientes supera `max_grad_norm`
(1.0 por defecto), todos se multiplican por el mismo factor
`max_grad_norm / norma`. Esto limita el tamaño del paso sin cambiar
su dirección. Es global, no por matriz: recortar cada matriz por
separado cambiaría la dirección del paso. Si aparece un NaN o un
infinito, el entrenamiento se detiene con un diagnóstico: nunca se
sustituyen silenciosamente por ceros.

## 8. Por qué una pérdida de train baja no garantiza generalización

Una pérdida de train baja puede significar dos cosas: que el modelo
aprendió patrones que se repiten en texto nuevo, o que memorizó el
corpus de entrenamiento. Solo la pérdida de validación (texto que no
se usó para entrenar, con vocabulario construido solo con train y
ventanas que no cruzan la frontera) distingue ambas. Por eso evaluar
sobre train no sustituye a validation, y por eso la prueba de
sobreajuste deliberado (`overfit`) se presenta como lo que es: una
prueba de que el sistema *puede* memorizar, no una prueba de que
generaliza. Ver `docs/experimentos.md` para los números reales y sus
límites (la validación de la demo cubre solo 64 objetivos).

## 9. Reproducibilidad

Generadores independientes, derivados con `SeedSequence`:
inicialización del modelo (`ModelConfig.seed`), muestreo de train
(`SeedSequence([seed, 1])`) y muestreo de generación
(`SeedSequence([seed, 2])`). La evaluación es determinista y no
consume el generador de entrenamiento. Nunca se usa `np.random.seed`
global. La reproducibilidad se garantiza en el mismo entorno; no se
promete igualdad bit a bit entre sistemas, BLAS o versiones.

## 10. Comandos

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

Cada ejecución crea un directorio con `config.json`, `metrics.jsonl`,
`last.npz`, `best.npz`, muestras y `report.md`. Reanudar un
entrenamiento ya completado informa de ello sin reiniciar ni extender
el scheduler. Los modelos de inferencia de la fase 1 (directorios de
`save_model`) se siguen pudiendo cargar con `generate --model-dir`.
