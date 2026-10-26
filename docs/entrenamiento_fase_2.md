# Preparación conceptual de la segunda fase: entrenamiento

> NOTA (fase 2 implementada el 2026-10-26): este documento era la
> preparación conceptual previa. La implementación existe ahora en
> `src/numpy_llm/` (backward, optimizadores, Trainer, checkpoints) y
> está documentada en `docs/entrenamiento.md`, `docs/derivadas.md`,
> `docs/experimentos.md` y `docs/problemas_frecuentes.md`. Lo que sigue
> se conserva como contexto, no como estado actual del proyecto.

> Este documento describe cómo se entrenaría el modelo más adelante.
> **No implementa esa fase.** No hay aquí métodos `backward` vacíos,
> optimizadores de ejemplo ni código de entrenamiento incompleto.
> Calcular una pérdida no entrena el modelo: para aprender hay que
> calcular gradientes y actualizar parámetros.

## El ciclo de entrenamiento

```
texto
→ tokens
→ lotes de entradas y objetivos desplazados
→ forward
→ pérdida
→ backward
→ gradientes
→ optimizador
→ pesos actualizados
```

Y vuelta a empezar, lote tras lote, época tras época. La pérdida de la
fase 1 es exactamente el objetivo que se optimizará en la fase 2: la
entropía cruzada media sobre la predicción del siguiente token.

## Qué necesitaremos implementar

### 1. Derivar e implementar los gradientes de cada operación
Cada operación del forward (lineal, LayerNorm, softmax, atención,
ReLU, embeddings, conexiones residuales) necesita su regla de
derivación. No basta con saber que "hay que derivar": hay que escribir
la derivada correcta para cada forma de tensor y comprobarla.

### 2. Acumular correctamente gradientes en conexiones residuales
En `h = x + atención(...)`, el gradiente que llega a `h` se reparte:
una parte va por la rama de la atención y otra parte va directamente a
`x`. Si se olvida una rama, el gradiente queda incompleto.

### 3. Acumular gradientes en las filas de embeddings utilizadas varias veces
Un mismo token puede aparecer muchas veces en un lote. Su fila de
embedding recibe gradiente de cada aparición: los gradientes se suman,
no se sobrescriben. Lo mismo ocurre con las posiciones, que se comparten
entre todos los ejemplos del lote.

### 4. Mantener los intermediarios necesarios del forward
El backward necesita valores calculados en el forward (entradas de cada
capa, pesos de atención, medias y varianzas de LayerNorm, máscaras de
ReLU, etc.). Habrá que guardarlos de forma explícita durante el
forward, porque en la fase 1 no se guardan.

### 5. Verificar los gradientes mediante diferencias finitas
Antes de confiar en un backward, se compara con una aproximación
numérica: perturbar un parámetro en `+ε` y `-ε`, medir el cambio en la
pérdida y estimar la derivada. Si el gradiente analítico y el numérico
no coinciden dentro de una tolerancia, hay un error en la derivada.
Esta comprobación es lenta, así que se hace sobre modelos diminutos y
pocos parámetros, no sobre el modelo completo en cada paso.

### 6. Implementar un optimizador
El optimizador decide cómo convertir gradientes en cambios de pesos.
El descenso de gradiente más sencillo es
`peso = peso - tasa_de_aprendizaje * gradiente`. Adam añade momentos
(medias móviles del gradiente y de su cuadrado) y corrección de sesgo.
La tasa de aprendizaje, el tamaño de lote y el número de épocas serán
hiperparámetros nuevos que habrá que ajustar con cuidado.

### 7. Separar datos de entrenamiento y validación
La pérdida de entrenamiento dice si el modelo está aprendiendo el
corpus de entrenamiento; la pérdida de validación (sobre texto que el
modelo no ha visto en entrenamiento) dice si está generalizando o solo
memorizando. Sin esta separación, una pérdida baja puede ser puro
sobreajuste.

### 8. Evitar filtraciones entre particiones
Dos filtraciones sutiles que habrá que vigilar:

- **Vocabulario**: si el vocabulario se construye con todo el corpus
  (incluida la validación), la validación ya no es del todo "no vista".
  En rigor, el vocabulario debe construirse solo con el entrenamiento,
  o documentarse explícitamente qué se hizo.
- **Muestreo de ventanas**: las ventanas de entrenamiento y validación
  no deben solaparse ni compartir el mismo texto; una ventana de
  validación que continúa un fragmento de entrenamiento ya no mide
  generalización.

### 9. Comprobar que el modelo puede sobreajustar un corpus diminuto
Antes de entrenar "en serio", la prueba inicial es deliberadamente
pequeña: un corpus de pocas frases. Un modelo correcto debe ser capaz
de memorizarlo y llevar su pérdida casi a cero. Si no puede
sobreajustar ni un corpus diminuto, algo está mal en el modelo, en los
gradientes o en el optimizador, y no tiene sentido escalar.

### 10. Medir pérdida y calidad de generación durante el aprendizaje
Durante el entrenamiento se registrará la pérdida (entrenamiento y
validación) a lo largo de los pasos y se generarán muestras de texto
periódicamente con la misma semilla y el mismo prompt, para ver si el
texto pasa de aleatorio a plausible. La pérdida es la métrica
principal; la generación es la comprobación cualitativa.

## Qué no cambia en la fase 2

La arquitectura (embeddings, atención causal, bloques pre-norm,
proyección final) y la definición de la pérdida no cambian. Lo que se
añade es el camino de vuelta: gradientes y actualización de pesos. Por
eso la fase 1 insiste tanto en que el forward sea correcto y esté bien
probado: todo el entrenamiento se apoya sobre él.
