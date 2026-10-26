# Fundamentos del modelo

Este documento explica cada componente con la misma estructura:
intuición, ecuación, forma de los tensores, correspondencia con el
código, un ejemplo pequeño cuando ayuda y errores habituales.

> Aunque el proyecto se llame “LLM desde cero”, lo que construimos aquí
> es un **modelo de lenguaje diminuto con fines educativos**, basado en
> principios utilizados en modelos de mayor escala. No es un LLM
> entrenado ni comparable en capacidad a un modelo de producción.

## Recorrido de las dimensiones

| Paso | Forma | Significado |
|---|---|---|
| Entrada `token_ids` | `[B, T]` | Identificadores de token |
| Embeddings de tokens | `[B, T, D]` | Una fila de `E` por token |
| Embeddings de posición | `[T, D]` | Se suma por broadcasting |
| Entrada a los bloques | `[B, T, D]` | Token + posición |
| Dentro de cada bloque | `[B, T, D]` | La dimensión D no cambia |
| Q, K, V_values por cabeza | `[B, H, T, Dh]` | Tras separar cabezas |
| Puntajes de atención | `[B, H, T, T]` | Cada fila, una distribución sobre el pasado |
| Pesos de atención | `[B, H, T, T]` | Tras softmax y máscara causal |
| Contexto por cabeza | `[B, H, T, Dh]` | Media ponderada de valores |
| Salida de atención | `[B, T, D]` | Cabezas unidas + proyección |
| Salida de cada bloque | `[B, T, D]` | Con conexiones residuales |
| Tras LayerNorm final | `[B, T, D]` | Normalizado por token |
| Logits | `[B, T, V]` | Una puntuación por token del vocabulario |

## 1. Tokens y vocabulario

### Intuición
Un modelo no lee letras directamente: trabaja con números. El
tokenizador asigna un identificador entero a cada carácter distinto del
corpus.

### Ecuación
No hay ecuación: es una tabla `carácter -> id`.

### Formas
Texto -> lista de ids de longitud T -> array `[B, T]`.

### Código
`tokenizer.py`: `CharTokenizer.from_text`, `encode`, `decode`.
El id 0 es `<UNK>`; el resto se ordena por código Unicode.

### Ejemplo
Corpus `"ab a"` -> vocabulario `["<UNK>", " ", "a", "b"]`;
`encode("ab") = [2, 3]`; `decode([2, 3]) = "ab"`.

### Errores habituales
- Pensar que construir el vocabulario es entrenar: no lo es; solo
  enumera los caracteres posibles.
- Esperar que `decode(encode(texto)) == texto` con caracteres
  desconocidos: varios caracteres distintos colapsan en `<UNK>`.
- Confundir token con palabra: aquí un token es un carácter.

## 2. Ventanas de entrada y objetivo

### Intuición
Para aprender "qué viene después", se toma una ventana de texto y se
desplaza una posición: la entrada es el texto hasta T, el objetivo es
el mismo texto desplazado.

### Ecuación
`x = tokens[i:i+T]`, `y = tokens[i+1:i+T+1]`.

### Formas
`x: [B, T]`, `y: [B, T]`.

### Código
`data.py`: `build_examples`.

### Ejemplo
`A B C D E`, T = 4 -> `x = A B C D`, `y = B C D E`.

### Errores habituales
- Desplazar en la dirección contraria.
- Repetir el corpus para fabricar ejemplos cuando es demasiado corto:
  aquí eso es un error, no una solución silenciosa.
- Usar padding en esta fase: no se usa; las ventanas son completas.

## 3. Embeddings de tokens y posiciones

### Intuición
Cada token se representa con un vector denso de dimensión D que, al
principio, es aleatorio y, tras entrenar, debería capturar
regularidades del texto. La posición se suma como otro vector, porque
la atención por sí sola no distingue el orden.

### Ecuación
`x = E[token_ids] + P[0:T]`

### Formas
`E: [V, D]`, `P: [C, D]`, `x: [B, T, D]`.

### Código
`layers.py`: `Embeddings`.

### Ejemplo
Con V = 4, D = 3, el token 2 selecciona la fila 2 de `E` (un vector de
3 números); se le suma la fila de `P` correspondiente a su posición.

### Errores habituales
- Construir one-hots de tamaño V para multiplicar por `E`: la
  indexación ya hace esa selección, sin el coste del one-hot.
- Sumar `P` sin recortar a T: las formas no cuadran.
- Creer que embeddings aleatorios ya "significan" algo: todavía no.

## 4. Capa lineal

### Intuición
Una transformación afín: rota/escala el espacio (matriz) y lo desplaza
(bias).

### Ecuación
`y = x @ W + b`

### Formas
`W: [input_dim, output_dim]`, `b: [output_dim]`, `y` conserva las
dimensiones iniciales de `x` y cambia solo la última.

### Código
`layers.py`: `Linear`.

### Ejemplo
`x = [1, 1]`, `W = [[1, 2], [3, 4]]`, `b = [10, 20]` ->
`y = [14, 26]`.

### Errores habituales
- Transponer `W` por costumbre de otros frameworks: aquí la convención
  es explícita, `W: [entrada, salida]`.
- Olvidar el bias o sumarlo en el eje equivocado.

## 5. Layer Normalization

### Intuición
Cada token se "centra y escala" a sí mismo: media 0 y varianza 1 en sus
D características, y luego se reaplica una escala (`gamma`) y un
desplazamiento (`beta`) aprendibles.

### Ecuación
```
mean = mean(x, axis=-1, keepdims=True)
variance = mean((x - mean) ** 2, axis=-1, keepdims=True)
normalized = (x - mean) / sqrt(variance + eps)
output = gamma * normalized + beta
```

### Formas
Entrada y salida: `[..., D]`; `gamma, beta: [D]`.

### Código
`layers.py`: `LayerNorm`.

### Ejemplo
`[1, 2, 3, 4]` -> media 2.5, varianza 1.25 -> normalizado
aproximadamente `[-1.34, -0.45, 0.45, 1.34]` (con `gamma = 1`,
`beta = 0`).

### Errores habituales
- Confundirla con BatchNorm: LayerNorm normaliza por token, no a
  través del lote.
- Normalizar sobre el eje de la secuencia: mezclaría posiciones.
- Olvidar `eps`: con una entrada constante, la varianza es 0 y la
  división fallaría.

## 6. Atención causal multicabeza

### Intuición
Cada posición consulta todas las posiciones anteriores y construye una
media ponderada de sus valores. Los pesos dicen "cuánto miro a cada
posición pasada".

### Ecuación
```
Q = x @ Wq + bq
K = x @ Wk + bk
V_values = x @ Wv + bv
scores = Q @ K^T / sqrt(Dh)          # [B, H, T, T]
scores[j > i] = -inf                 # máscara causal
attention_weights = softmax(scores, axis=-1)
context = attention_weights @ V_values
output = merged_heads @ Wo + bo
```

### Formas
`Q, K, V_values: [B, H, T, Dh]`; `scores, weights: [B, H, T, T]`;
`context: [B, H, T, Dh]`; `output: [B, T, D]`.

### Código
`attention.py`: `CausalMultiHeadAttention`. Las cabezas se separan con
reshape + transpose; no hay bucles sobre tokens ni cabezas.

### Ejemplo pequeño
Con T = 3, la fila de la posición 1 solo puede tener pesos no nulos en
las columnas 0 y 1; la columna 2 es 0 porque es futuro.

### Errores habituales
- Aplicar la máscara después de softmax: las filas ya no sumarían 1.
- Aplicar softmax sobre el eje equivocado: debe ser sobre las claves
  (último eje), no sobre las consultas.
- No dividir entre `sqrt(Dh)`: los puntajes crecen con Dh y softmax se
  satura.
- Confundir `V_values` (valores de atención) con V (tamaño del
  vocabulario).

## 7. Red feed-forward

### Intuición
Una MLP por posición: expande a F, aplica ReLU y vuelve a D. Añade
capacidad no lineal; no mezcla tokens entre sí.

### Ecuación
```
hidden = relu(x @ W1 + b1)
output = hidden @ W2 + b2
```

### Formas
`W1: [D, F]`, `b1: [F]`, `W2: [F, D]`, `b2: [D]`.

### Código
`transformer.py`: `FeedForward`.

### Errores habituales
- Omitir la activación: dos lineales seguidas equivalen a una sola.
- Usar parámetros distintos por posición: aquí se comparten en todas
  las posiciones.
- Esperar que la MLP mezcle información entre tokens: eso lo hace la
  atención, no la MLP.

## 8. Bloque Transformer (pre-norm)

### Intuición
Dos subcapas (atención y MLP), cada una precedida por LayerNorm y
rodeada por una conexión residual que deja pasar la información
directamente.

### Ecuación
```
h = x + attention(layer_norm_1(x))
output = h + feed_forward(layer_norm_2(h))
```

### Formas
Entrada y salida: `[B, T, D]`.

### Código
`transformer.py`: `TransformerBlock`.

### Errores habituales
- Aplicar la norma después de la subcapa (post-norm) cuando la
  especificación pide pre-norm: el orden cambia el comportamiento.
- Sumar el residuo equivocado (por ejemplo, `ln(x)` en lugar de `x`).

## 9. Modelo completo y logits

### Intuición
Embeddings -> L bloques -> LayerNorm final -> proyección a vocabulario.
Cada posición produce un vector de V puntuaciones (logits).

### Ecuación
`logits = normalized_hidden @ W_vocab + b_vocab`

### Formas
`normalized_hidden: [B, T, D]`; `W_vocab: [D, V]`; `b_vocab: [V]`;
`logits: [B, T, V]`.

### Código
`model.py`: `TransformerLanguageModel.forward`.

### Errores habituales
- Compartir `W_vocab` con los embeddings en esta fase: aquí no se hace.
- Aplicar softmax en el forward "porque sí": el forward devuelve
  logits; softmax pertenece a la pérdida o a la generación.
- Mezclar logits, pérdida y generación en una sola función.

## 10. Softmax, log-softmax y entropía cruzada

### Intuición
Un logit es una puntuación sin normalizar. Softmax la convierte en
probabilidad. La pérdida es la log-probabilidad negativa del token
correcto: si el modelo le da poca probabilidad, la pérdida es grande.

### Ecuación
```
m = max(z)
softmax(z) = exp(z - m) / sum(exp(z - m))
loss = -mean(log_softmax(logits)[b, t, targets[b, t]])
```

### Formas
`logits: [B, T, V]`; `targets: [B, T]`; `loss`: escalar.

### Código
`functional.py`: `softmax`, `log_softmax`, `cross_entropy_loss`.

### Ejemplo
Logits `[0, log(3)]` -> probabilidades `[0.25, 0.75]`; si el objetivo
es el segundo token, la pérdida es `-log(0.75) ≈ 0.288`.

### Errores habituales
- Calcular `log(softmax(z))` directamente: pierde estabilidad.
- Aplicar softmax antes de la pérdida: la pérdida ya lo calcula.
- Esperar que la pérdida inicial sea exactamente `log(V)`: con pesos
  aleatorios es cercana, no exacta.
- Evaluar la calidad literaria del texto aleatorio como criterio de
  éxito: no lo es; el criterio es la corrección matemática.

## 11. Generación autorregresiva

### Intuición
El modelo solo predice el siguiente token. Para generar N tokens, se
predice uno, se añade a la secuencia y se vuelve a predecir.

### Ecuación
En cada paso: `p = softmax(logits_última_posición / temperature)`,
con filtrado top-k previo si se indica.

### Código
`generation.py`: `generate`.

### Errores habituales
- Creer que el modelo genera la frase entera de golpe: no; es
  token a token.
- Confundir paralelismo de evaluación con generación: al evaluar una
  secuencia conocida, todas las posiciones se predicen en paralelo
  (gracias a la máscara causal); al generar, cada token nuevo depende
  del anterior, así que hay que añadirlo sucesivamente.
- Dividir por temperatura 0: el caso greedy usa argmax, sin división.

## Distinciones clave

- **Parámetros y activaciones**: los parámetros son los pesos que se
  aprenden (o aprenderían); las activaciones son los valores
  intermedios que produce el forward para una entrada concreta.
- **Hiperparámetros y parámetros**: los hiperparámetros (D, H, L, F,
  C...) se eligen antes; los parámetros se ajustan entrenando.
- **Tokens y palabras**: aquí un token es un carácter; una palabra
  puede ser varios tokens en otros tokenizadores.
- **Logits y probabilidades**: los logits no suman 1 y pueden ser
  negativos; las probabilidades sí suman 1 y están en [0, 1].
- **Forward, generación y entrenamiento**: forward calcula logits;
  generación usa esos logits para elegir tokens uno a uno;
  entrenamiento (fase 2) calcularía gradientes y actualizaría pesos.
- **Construcción del vocabulario y aprendizaje de pesos**: construir
  el vocabulario enumera caracteres; aprender pesos ajusta matrices.
