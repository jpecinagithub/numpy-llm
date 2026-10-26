# Derivadas (fase 2): forward, intermediarios y backward por componente

Notación: B lote, T secuencia, C contexto, V vocabulario, D embedding,
H cabezas, Dh = D/H, F MLP interna, L bloques, M = B×T objetivos,
`dX` = `∂loss/∂X`. El factor `1/M` de la pérdida se aplica UNA sola
vez, en la entropía cruzada; ninguna capa vuelve a dividir por B o T.
Todas las comprobaciones numéricas usan float64, `h = 1e-5`,
`atol = 1e-7`, `rtol = 1e-4`, diferencias centrales.

Resultado global de la comprobación (modelo diminuto B=2, T=3, V=5,
D=4, H=2, F=8, L=1, 257 componentes): **pasa, error absoluto máximo
4,84e-09** (ejecución de `tests/test_gradcheck.py`; la CLI
`gradcheck --scope all` con tokens aleatorios da 1,15e-08, también
dentro de tolerancia). Un error relativo de 1,0 en algún componente
corresponde a gradientes analíticos exactamente cero con aproximación
numérica ~1e-21: el criterio que manda es el absoluto combinado con
la tolerancia, y esos casos pasan.

## 1. Softmax + entropía cruzada

- Forward: `log_probs = logits - logsumexp(logits)`,
  `loss = -sum(log_probs[target]) / M`.
- Derivación: derivando el log-softmax respecto a un logit,
  `∂loss/∂logit_j = p_j - 1{j == target}`, dividido entre M.
  Intuición: si el modelo ya asigna probabilidad 1 al token correcto,
  el gradiente es 0; si asigna 0, es -1/M (empuja ese logit hacia
  arriba) y `+p_j/M` para el resto (los empuja hacia abajo).
- Vectorizado: `dlogits = softmax(logits)` (copia); restar 1 por
  indexación en el objetivo; dividir entre M. Sin one-hot completo,
  sin temperatura ni top-k en entrenamiento.
- Código: `functional.cross_entropy_with_grad`, `losses.py`.
- Comprobación: parte del gradcheck del modelo completo.

## 2. Capa lineal `Y = X @ W + b`

- Formas: `X [..., Din]`, `W [Din, Dout]`, `b [Dout]`, `Y [..., Dout]`.
- Intermediario: la entrada `X`.
- Backward: `dX = dY @ W.T`;
  `X_flat = X.reshape(-1, Din)`, `dY_flat = dY.reshape(-1, Dout)`;
  `dW = X_flat.T @ dY_flat`; `db = sum(dY_flat, axis=0)`.
- Por qué: `dX` conserva la forma de `X` (la transpuesta devuelve la
  forma de entrada); `dW` tiene la forma de `W` (cada peso multiplica
  una entrada para una salida: su gradiente es la suma de esos
  productos); `db` suma sobre todas las posiciones porque el bias se
  reutiliza por broadcasting, y el backward invierte ese broadcasting
  sumando.
- Ejemplo verificable a mano: `X = [[1, 2]]`, `W = [[1, 0], [0, 1]]`,
  `b = [0, 0]`, `dY = [[3, 4]]` → `dX = [[3, 4]]`,
  `dW = [[3, 4], [6, 8]]`, `db = [3, 4]`.
- Código: `layers.Linear`. Comprobación: `test_backward_layers.py`.

## 3. ReLU y MLP

- Forward: `Z1 = X @ W1 + b1`, `A1 = relu(Z1)`, `Y = A1 @ W2 + b2`.
- Intermediario: máscara `Z1 > 0`.
- Backward: `dA1 = fc2.backward(dY)`; `dZ1 = dA1 * (Z1 > 0)`;
  `dX = fc1.backward(dZ1)`. Se actualizan TODOS los parámetros de la
  MLP, no solo los de la segunda capa.
- ReLU no es diferenciable exactamente en 0; aquí la derivada en 0 se
  define como 0, y las pruebas usan entradas lejos de 0 (o una región
  lineal controlada) para que las diferencias finitas sean comparables.
- Código: `transformer.FeedForward`. Comprobación: gradcheck de la MLP.

## 4. LayerNorm

- Forward: `mu = mean(X)`, `variance = mean((X-mu)^2)`,
  `inv_std = 1/sqrt(variance + eps)`, `X_hat = (X-mu)*inv_std`,
  `Y = gamma*X_hat + beta` (última dimensión, mismo `eps`).
- Intermediarios: `X_hat`, `inv_std`.
- Parámetros: `dgamma = sum(dY * X_hat)`, `dbeta = sum(dY)` (todos los
  ejes salvo el último).
- Con `G = dY * gamma`:
  `dX = inv_std * (G - mean(G) - X_hat * mean(G * X_hat))`.
  Relación con la regla de la cadena: `mu` y `variance` dependen de
  `X`, no son constantes. El término `-mean(G)` es el camino a través
  de `mu`; `-X_hat * mean(G*X_hat)` es el camino a través de la
  varianza; `inv_std` es el factor común. No mezcla ejemplos ni
  posiciones: cada vector se normaliza con su propia media/varianza.
- Código: `layers.LayerNorm`. Comprobación: entradas ordinarias,
  constantes y casi constantes; `X`, `gamma`, `beta`; sin mezcla
  entre posiciones (`test_backward_layers.py`).

## 5. Embeddings

- Forward: `X = E[token_ids] + P[:T]`, `E [V, D]`, `P [C, D]`.
- Intermediarios: los identificadores usados y `T`.
- Backward: `np.add.at(grad_E, token_ids.reshape(-1), dX.reshape(-1, D))`;
  `grad_P[:T] += sum(dX, axis=0)`; las posiciones no usadas reciben 0.
- Los índices enteros no se derivan: lo que se aprende son los valores
  de las filas seleccionadas. La suma de embeddings reparte el mismo
  `dX` a ambas tablas. El fallo clásico es `grad_E[ids] += dX`, que
  pierde contribuciones con índices repetidos por el buffering de
  NumPy; `np.add.at` las suma correctamente.
- Código: `layers.Embeddings`. Comprobación: tokens repetidos dentro
  de una secuencia y entre ejemplos del lote.

## 6. Softmax general (para la atención)

- Si `A = softmax(S)` y llega `dA`:
  `dS = A * (dA - sum(dA * A, axis=-1, keepdims=True))`.
- El Jacobiano por fila es `diag(A) - A A^T`; aplicarlo a `dA` da
  exactamente esa expresión, sin materializar ninguna matriz `T×T`.
  El eje de reducción coincide con el eje normalizado en el forward.
- Código: `functional.softmax_backward`.

## 7. Atención multicabeza

- Formas: `Q, K, V_values [B, H, T, Dh]`; `S, A [B, H, T, T]`;
  `R [B, H, T, Dh]`.
- Forward: `S = Q @ K^T / sqrt(Dh)`, máscara causal (`-inf` antes de
  softmax), `A = softmax(S)`, `R = A @ V_values`, concatenar cabezas
  y proyección de salida.
- Intermediarios: `Q, K, V_values, A` y la máscara causal.
- Backward, en orden: (a) proyección de salida (Linear);
  (b) invertir merge de cabezas → `dR [B, H, T, Dh]`;
  (c) `dA = dR @ V^T`, `dV = A^T @ dR`;
  (d) softmax (sección 6); (e) máscara causal: cero en posiciones
  prohibidas — como `A` ya es 0 ahí, `dS` ya es 0 ahí, y nunca se
  opera con los `-inf`, así que no hay NaN en gradientes;
  (f) `dQ = (dS @ K)/sqrt(Dh)`, `dK = (dS^T @ Q)/sqrt(Dh)` (sin olvidar
  el escalado); (g) backward de las tres proyecciones, con
  `dX = dX_Q + dX_K + dX_V` (suma, nunca sobrescritura).
- Código: `attention.CausalMultiHeadAttention`. Comprobación:
  gradcheck de proyecciones y entrada, causalidad y finitud.

## 8. Conexiones residuales y bloque pre-norm

- Forward: `A = LN1(X)`, `B = Attention(A)`, `H = X + B`,
  `C = LN2(H)`, `D_ff = FFN(C)`, `Y = H + D_ff`.
- Backward: `dD_ff = dY`; `dC = FFN.backward(dD_ff)`;
  `dH = dY + LN2.backward(dC)` (la suma residual reparte `dY` por la
  rama directa y por la rama de la MLP); `dB = dH`;
  `dA = Attention.backward(dB)`; `dX = dH + LN1.backward(dA)`
  (de nuevo, la suma residual reparte por la rama directa y por la
  atención). Cada suma tiene su origen en una `Y = X + F(X)`,
  cuyo gradiente es `dX = dY + backward_F(dY)`.
- Sin operaciones in-place sobre activaciones compartidas: las sumas
  crean arrays nuevos.
- Código: `transformer.TransformerBlock`. Comprobación: gradcheck
  específico del bloque completo (`test_backward_model.py`), que
  detecta errores invisibles al probar las capas por separado.

## 9. Modelo completo

- Orden inverso: `dlogits` → proyección al vocabulario → LayerNorm
  final → bloques del último al primero → embeddings.
- No se devuelve gradiente de los identificadores enteros.
- Se comprueba: todos los parámetros tienen buffer de gradiente con
  nombre, forma y dtype coincidentes; sin NaN ni infinitos;
  `backward` no modifica pesos; dos contribuciones acumuladas
  equivalen a su suma; `zero_grad` limpia; y un gradiente
  legítimamente nulo (p. ej. ciertas simetrías del softmax en algunos
  biases) no se confunde con una implementación rota.
- Código: `model.TransformerLanguageModel.backward`.
