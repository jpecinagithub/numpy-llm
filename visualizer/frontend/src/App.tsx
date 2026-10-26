import { useEffect, useState } from "react";
import {
  displayChar,
  fetchInfo,
  forward,
  formatShape,
  generate,
  lossDemo,
  tokenize,
  trainDemo,
  type ForwardResult,
  type GenerateResult,
  type LossDemoResult,
  type ModelInfo,
  type TokenizeResult,
  type TrainDemoResult,
} from "./api.ts";

function TokenStrip({ tokens }: { tokens: TokenizeResult["tokens"] }) {
  return (
    <div className="token-strip" aria-label="Tokens">
      {tokens.map((token, index) => (
        <span
          key={index}
          className={token.is_unknown ? "token token-unknown" : "token"}
          title={`Posición ${index}: carácter ${JSON.stringify(token.char)} → id ${token.id}`}
        >
          <span className="token-char">{displayChar(token.display === "<UNK>" ? "?" : token.char)}</span>
          <span className="token-id">{token.id}</span>
        </span>
      ))}
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return <p className="error">Error: {message}</p>;
}

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [text, setText] = useState("La inteligencia");
  const [tokenResult, setTokenResult] = useState<TokenizeResult | null>(null);
  const [forwardResult, setForwardResult] = useState<ForwardResult | null>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [vocabQuery, setVocabQuery] = useState("");

  const [prompt, setPrompt] = useState("La inteligencia");
  const [maxNewTokens, setMaxNewTokens] = useState(30);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState("20");
  const [seed, setSeed] = useState(42);
  const [generation, setGeneration] = useState<GenerateResult | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const [lossText, setLossText] = useState("La inteligencia");
  const [lossResult, setLossResult] = useState<LossDemoResult | null>(null);
  const [lossError, setLossError] = useState<string | null>(null);
  const [lossLoading, setLossLoading] = useState(false);
  const [trainUpdates, setTrainUpdates] = useState(60);
  const [trainResult, setTrainResult] = useState<TrainDemoResult | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainLoading, setTrainLoading] = useState(false);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch((error: unknown) => {
        setInfoError(error instanceof Error ? error.message : "No se pudo contactar con el backend.");
      });
  }, []);

  async function handleAnalyze() {
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const [tokens, fwd] = await Promise.all([tokenize(text), forward(text)]);
      setTokenResult(tokens);
      setForwardResult(fwd);
    } catch (error: unknown) {
      setAnalyzeError(error instanceof Error ? error.message : "Error al analizar el texto.");
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleLossDemo() {
    setLossLoading(true);
    setLossError(null);
    try {
      setLossResult(await lossDemo(lossText));
    } catch (error: unknown) {
      setLossError(error instanceof Error ? error.message : "Error al calcular la pérdida.");
    } finally {
      setLossLoading(false);
    }
  }

  async function handleTrainDemo() {
    setTrainLoading(true);
    setTrainError(null);
    try {
      setTrainResult(await trainDemo(trainUpdates));
    } catch (error: unknown) {
      setTrainError(error instanceof Error ? error.message : "Error en el mini-entrenamiento.");
    } finally {
      setTrainLoading(false);
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    setGenerationError(null);
    try {
      const parsedTopK = topK.trim() === "" ? null : Number.parseInt(topK, 10);
      const result = await generate({
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: parsedTopK,
        seed,
      });
      setGeneration(result);
    } catch (error: unknown) {
      setGenerationError(error instanceof Error ? error.message : "Error al generar texto.");
    } finally {
      setGenerating(false);
    }
  }

  const filteredVocab = (info?.vocab ?? []).filter((char, id) => {
    if (vocabQuery === "") return true;
    return char.includes(vocabQuery) || String(id) === vocabQuery;
  });

  return (
    <div className="page">
      <header className="header">
        <p className="eyebrow">numpy-llm · Visualizador educativo · Fases 1 y 2</p>
        <h1>¿Qué hace el modelo por dentro… y cómo aprende?</h1>
        <p className="subtitle">
          Un pequeño Transformer decoder-only en Python y NumPy. Esta web no reescribe el modelo:
          pide los resultados al backend Python real, el mismo código que usa la CLI.
        </p>
        <p className="notice">
          ⚠️ El modelo principal que ves en la Fase 1 está <strong>sin entrenar</strong>: sus pesos
          son aleatorios y el texto que genera es esencialmente aleatorio. Eso no es un fallo, es el
          punto de partida. Los laboratorios de la Fase 2 usan modelos aparte y no entrenan ese
          modelo principal.
        </p>
        <p className="credit">Proyecto educativo de Jon Peciña (jpecina).</p>
      </header>

      <nav className="nav" aria-label="Secciones">
        <a href="#fase1">Fase 1: predecir</a>
        <a href="#recorrido">Recorrido</a>
        <a href="#tokenizador">Tokenizador</a>
        <a href="#inspector">Inspector</a>
        <a href="#generacion">Generación</a>
        <a href="#fase2">Fase 2: aprender</a>
        <a href="#ciclo">El ciclo de entrenamiento</a>
        <a href="#lab-perdida">Laboratorio: pérdida y gradiente</a>
        <a href="#lab-entrena">Laboratorio: entrena en miniatura</a>
        <a href="#demo-real">El entrenamiento real del repo</a>
      </nav>

      <section className="phase-banner phase-1" id="fase1">
        <h2>FASE 1 · Cómo predice un modelo (sin entrenar)</h2>
        <p>
          Idea en una frase: <strong>el modelo no “sabe” nada todavía; solo convierte texto en
          números, los mezcla y puntúa qué carácter cree que viene después.</strong> Todo lo de
          esta parte funciona con pesos aleatorios: sirve para entender la máquina, no para obtener
          buen texto.
        </p>
      </section>

      {infoError && (
        <section className="card">
          <h2>Backend no disponible</h2>
          <ErrorBox message={infoError} />
          <p>
            Arranca primero el backend desde la raíz del repositorio, en otra terminal, y después
            recarga esta página:
          </p>
          <pre>
            <code>.venv/bin/python visualizer/server.py</code>
          </pre>
          <p>En Windows (PowerShell): <code>.venv\Scripts\python visualizer/server.py</code></p>
        </section>
      )}

      <section className="card" id="recorrido">
        <h2>1. El recorrido: del texto a la predicción</h2>
        <p>
          Escribe un texto y pulsa “Analizar”. Verás sus tokens y, debajo, la forma que tiene el
          tensor en cada etapa: la dimensión no cambia dentro de los bloques, y solo cambia al final,
          cuando se proyecta al vocabulario.
        </p>
        <label className="field">
          <span>Texto para analizar</span>
          <input value={text} onChange={(event) => setText(event.target.value)} />
        </label>
        <button onClick={handleAnalyze} disabled={analyzing || text === ""}>
          {analyzing ? "Analizando…" : "Analizar"}
        </button>
        {analyzeError && <ErrorBox message={analyzeError} />}
        {tokenResult && (
          <>
            <h3>Tokens (aquí, token = carácter)</h3>
            <TokenStrip tokens={tokenResult.tokens} />
            <p className="small">
              Identificadores: [{tokenResult.token_ids.join(", ")}]. Los tokens en rojo son
              desconocidos y se codifican como <code>&lt;UNK&gt;</code> (id 0).
            </p>
          </>
        )}
        {forwardResult && (
          <>
            <h3>Formas de los tensores</h3>
            <ol className="steps">
              {forwardResult.steps.map((step) => (
                <li key={step.step}>
                  <strong>{step.description}</strong>
                  <span className="shape">{formatShape(step.shape)}</span>
                </li>
              ))}
            </ol>
            <p className="small">
              Lectura: <code>[1, 8]</code> es 1 ejemplo con 8 tokens; <code>[1, 8, 64]</code> es ese
              ejemplo con un vector de 64 números por token; <code>[1, 8, 50]</code> son 50
              puntuaciones (logits) por cada posición, una por cada carácter posible.
            </p>
          </>
        )}
      </section>

      <section className="card" id="tokenizador">
        <h2>2. Tokenizador y vocabulario</h2>
        <p>
          El vocabulario no está predefinido: se construye con los caracteres de{" "}
          <code>data/ejemplo.txt</code>, ordenados, más <code>&lt;UNK&gt;</code>. Por eso faltan
          mayúsculas como la <code>H</code> y minúsculas como la <code>k</code> o la <code>w</code>.
        </p>
        <label className="field">
          <span>Buscar en el vocabulario (carácter o id)</span>
          <input
            value={vocabQuery}
            onChange={(event) => setVocabQuery(event.target.value)}
            placeholder="Por ejemplo: a, ñ o 13"
          />
        </label>
        <p className="small">
          Mostrando {filteredVocab.length} de {info?.vocab_size ?? "…"} tokens.
        </p>
        <div className="vocab-grid">
          {(info?.vocab ?? []).map((char, id) => {
            if (vocabQuery !== "" && !char.includes(vocabQuery) && String(id) !== vocabQuery) {
              return null;
            }
            return (
              <span key={id} className="vocab-item" title={`id ${id}`}>
                <span>{displayChar(char === "<UNK>" ? "?" : char)}</span>
                <small>{id}</small>
              </span>
            );
          })}
        </div>
      </section>

      <section className="card" id="inspector">
        <h2>3. Inspector del modelo</h2>
        {info ? (
          <>
            <dl className="config-grid">
              {Object.entries(info.config).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{String(value)}</dd>
                </div>
              ))}
            </dl>
            <p>
              <strong>Número total de parámetros: {info.total_parameters.toLocaleString("es-ES")}</strong>
              . Cada parámetro es un número ajustable del modelo; ahora mismo son aleatorios.
            </p>
            <h3>Desglose por componente</h3>
            <ul className="breakdown">
              {info.parameter_breakdown.map((group) => (
                <li key={group.name}>
                  <span>{group.name}</span>
                  <span>{group.size.toLocaleString("es-ES")}</span>
                  <span
                    className="bar"
                    style={{ width: `${Math.max(2, (group.size / info.total_parameters) * 100)}%` }}
                  />
                </li>
              ))}
            </ul>
            <details>
              <summary>Ver todos los parámetros ({info.parameters.length})</summary>
              <table>
                <thead>
                  <tr>
                    <th>Nombre</th>
                    <th>Forma</th>
                    <th>Tamaño</th>
                  </tr>
                </thead>
                <tbody>
                  {info.parameters.map((parameter) => (
                    <tr key={parameter.name}>
                      <td>
                        <code>{parameter.name}</code>
                      </td>
                      <td>{formatShape(parameter.shape)}</td>
                      <td>{parameter.size.toLocaleString("es-ES")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </>
        ) : (
          <p>Cargando información del modelo…</p>
        )}
      </section>

      <section className="card" id="generacion">
        <h2>4. Generación: el modelo en acción</h2>
        <p>
          El modelo solo predice el siguiente carácter. Generar es repetir ese paso: predecir, añadir
          el carácter y volver a predecir. Prueba primero con pocos tokens nuevos para verlo rápido.
        </p>
        <label className="field">
          <span>Prompt</span>
          <input value={prompt} onChange={(event) => setPrompt(event.target.value)} />
        </label>
        <div className="controls">
          <label className="field">
            <span>Tokens nuevos</span>
            <input
              type="number"
              min={0}
              max={200}
              value={maxNewTokens}
              onChange={(event) => setMaxNewTokens(Number.parseInt(event.target.value, 10) || 0)}
            />
          </label>
          <label className="field">
            <span>Temperatura</span>
            <input
              type="number"
              min={0}
              step={0.1}
              value={temperature}
              onChange={(event) => setTemperature(Number.parseFloat(event.target.value) || 0)}
            />
          </label>
          <label className="field">
            <span>Top-k (vacío = todo el vocabulario)</span>
            <input value={topK} onChange={(event) => setTopK(event.target.value)} placeholder="20" />
          </label>
          <label className="field">
            <span>Semilla</span>
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number.parseInt(event.target.value, 10) || 0)}
            />
          </label>
        </div>
        <p className="small">
          Temperatura 0 = greedy (siempre el carácter con mayor puntuación). Top-k 1 equivale al
          greedy. Con la misma semilla y los mismos parámetros, el resultado se repite.
        </p>
        <div className="actions">
          <button onClick={handleGenerate} disabled={generating || prompt === ""}>
            {generating ? "Generando…" : "Generar"}
          </button>
          <button
            type="button"
            onClick={() => {
              setPrompt("Hola");
              setMaxNewTokens(20);
            }}
          >
            Probar “Hola” (con &lt;UNK&gt;)
          </button>
          <button
            type="button"
            onClick={() => {
              setTemperature(0);
              setTopK("");
            }}
          >
            Poner modo greedy
          </button>
        </div>
        {generationError && <ErrorBox message={generationError} />}
        {generation && (
          <div className="generation-result">
            <h3>Texto completo</h3>
            <p className="generated-text">
              <strong>{generation.prompt}</strong>
              {generation.generated_text}
            </p>
            <p className="small">
              Tokens generados: {generation.generated_token_ids.length} · Identificadores generados: [
              {generation.generated_token_ids.join(", ")}]
            </p>
          </div>
        )}
      </section>

      <section className="card" id="limites">
        <h2>Hasta dónde llega esta Fase 1</h2>
        <ul>
          <li>
            <strong>Sí puedes ver:</strong> tokens, embeddings, formas, parámetros, generación
            autorregresiva, reproducibilidad con semilla y el efecto de temperatura y top-k.
          </li>
          <li>
            <strong>No puedes esperar:</strong> texto con sentido, porque los pesos son aleatorios y
            no ha habido entrenamiento.
          </li>
          <li>
            <strong>Límites claros:</strong> vocabulario de solo 50 caracteres, contexto máximo de 64
            caracteres y un modelo diminuto de 110.642 parámetros.
          </li>
          <li>
            <strong>Qué cambiaría la Fase 2:</strong> calcular gradientes y actualizar los pesos para
            que la pérdida baje y el texto generado deje de ser aleatorio. Calcular la pérdida, por
            sí solo, no entrena.
          </li>
        </ul>
        <p className="small">
          En la Fase 1 esto era todo: calcular una pérdida no entrena. Lo que faltaba —gradientes y
          actualización de pesos— es exactamente la Fase 2 que sigue abajo.
        </p>
      </section>

      <section className="phase-banner phase-2" id="fase2">
        <h2>FASE 2 · Cómo aprende un modelo (entrenamiento)</h2>
        <p>
          Idea en una frase: <strong>el modelo compara su predicción con el carácter real, mide el
          error (pérdida), calcula en qué dirección debe cambiar cada peso (gradiente) y mueve los
          pesos un poco en esa dirección.</strong> Repetido miles de veces, el error baja y el texto
          deja de ser aleatorio. La arquitectura no cambia: cambia lo que valen los pesos.
        </p>
      </section>

      <section className="card" id="ciclo">
        <h2>5. El ciclo de entrenamiento, paso a paso</h2>
        <p>
          Cada actualización del optimizador es siempre la misma vuelta. Si entiendes esta vuelta,
          entiendes el entrenamiento; el resto son detalles (qué optimizador, qué tasa, cuántos
          lotes) sobre esta misma vuelta.
        </p>
        <ol className="steps">
          <li>
            <strong>Datos (teacher forcing).</strong> Se toma una ventana real del texto: la entrada
            son T caracteres y el objetivo son esos mismos caracteres desplazados uno: cada posición
            debe predecir el carácter siguiente. El modelo nunca ve sus propias predicciones como
            entrada durante el entrenamiento, y la máscara causal le impide mirar al futuro.
          </li>
          <li>
            <strong>Predicción (forward).</strong> La ventana recorre embeddings, bloques y
            proyección final y produce logits: una puntuación por cada carácter posible, en cada
            posición.
          </li>
          <li>
            <strong>Pérdida.</strong> Entropía cruzada media:{" "}
            <code>loss = -media(log(probabilidad del carácter correcto))</code>. Si el modelo da
            poca probabilidad al carácter correcto, la pérdida es alta. Perplexity es{" "}
            <code>exp(loss)</code>: “entre cuántos caracteres duda, de media”. Un modelo uniforme
            sobre 48 caracteres tiene pérdida log(48) ≈ 3,87 y perplexity 48.
          </li>
          <li>
            <strong>Gradiente (backward).</strong> La regla de la cadena reparte el error hacia
            atrás, capa por capa, hasta cada peso:{" "}
            <code>d loss / d logit = probabilidad predicha − 1 si es el carácter correcto</code>.
            El gradiente no es el cambio: es la dirección y la magnitud del cambio que reduciría la
            pérdida. <code>backward</code> calcula y acumula gradientes; no toca los pesos.
          </li>
          <li>
            <strong>Actualización (optimizer.step).</strong> Solo este paso mueve los pesos, y lo
            hace in-place: SGD hace <code>peso − tasa × gradiente</code>; AdamW usa medias móviles
            del gradiente y de su cuadrado (momentos m y v, con corrección de sesgo) y un weight
            decay desacoplado, solo en matrices lineales, nunca sumado al gradiente. Antes del paso
            se limpian los gradientes (<code>zero_grad</code>) y se recorta su norma global si
            supera 1,0 (clipping): sin limpiar, los gradientes se sumarían de un lote al siguiente
            sin querer.
          </li>
          <li>
            <strong>Volver a predecir.</strong> Con los pesos nuevos, la misma ventana debería dar
            (en tendencia, no en cada paso aislado) menos pérdida. La pérdida del lote se mide antes
            de actualizar; la evaluación se mide después, sobre texto separado.
          </li>
        </ol>
        <h3>Conceptos que suelen confundirse</h3>
        <ul>
          <li>
            <strong>Calcular la pérdida ≠ entrenar.</strong> La Fase 1 ya calculaba la pérdida y el
            modelo seguía siendo aleatorio. Entrenar exige backward + optimizer.step.
          </li>
          <li>
            <strong>Backward ≠ actualizar.</strong> Backward escribe gradientes; el optimizador
            escribe pesos. Por eso hay que limpiar gradientes entre lotes, salvo que se acumulen a
            propósito (varios microlotes antes de un solo paso).
          </li>
          <li>
            <strong>Train ≠ validation.</strong> El vocabulario se construye solo con train (90 % del
            texto) y validation (10 %) se codifica con ese vocabulario fijo; lo desconocido es{" "}
            <code>&lt;UNK&gt;</code>. Ninguna ventana cruza la frontera. Una pérdida baja en train
            puede ser memorización; solo validation (texto no entrenado) habla de generalización, y
            con un corpus diminuto hay que leerla con cautela.
          </li>
          <li>
            <strong>Gradiente cero ≠ implementación rota.</strong> Algunos gradientes pueden ser
            legítimamente cero (por ejemplo, por simetrías del softmax). Lo que se comprueba con
            diferencias finitas (gradcheck) es que el gradiente analítico coincida con el numérico.
          </li>
          <li>
            <strong>Tasa de aprendizaje.</strong> Demasiado alta: el modelo salta y la pérdida no
            baja o se vuelve NaN. Demasiado baja: aprende lentísimo. En la demo se usa warmup
            (sube de 0 a 1e-3 en 50 pasos) y luego descenso coseno hasta 1e-4.
          </li>
        </ul>
      </section>

      <section className="card" id="lab-perdida">
        <h2>6. Laboratorio: mide la pérdida y el gradiente</h2>
        <p>
          Esto usa un modelo nuevo sin entrenar (misma semilla que el principal) y <strong>no
          entrena nada</strong>: solo mide. Escribe al menos 9 caracteres; se toma la primera
          ventana de 9 tokens: los 8 primeros son la entrada y los 8 siguientes el objetivo. Mira
          la probabilidad que da al carácter correcto: cuando es baja, la pérdida es alta, y ese es
          exactamente el error que el backward repartiría.
        </p>
        <label className="field">
          <span>Texto (mínimo 9 caracteres)</span>
          <input value={lossText} onChange={(event) => setLossText(event.target.value)} />
        </label>
        <button onClick={handleLossDemo} disabled={lossLoading || lossText.length < 9}>
          {lossLoading ? "Calculando…" : "Medir pérdida y gradiente"}
        </button>
        {lossError && <ErrorBox message={lossError} />}
        {lossResult && (
          <>
            <p>
              Entrada: <code>{lossResult.input_text}</code> · Objetivo:{" "}
              <code>{lossResult.target_text}</code>
            </p>
            <dl className="config-grid">
              <div>
                <dt>Pérdida (entropía cruzada media)</dt>
                <dd>{lossResult.loss.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Perplexity (exp de la pérdida)</dt>
                <dd>{lossResult.perplexity.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Exactitud por token</dt>
                <dd>{(lossResult.token_accuracy * 100).toFixed(1)} %</dd>
              </div>
              <div>
                <dt>Norma total del gradiente</dt>
                <dd>{lossResult.grad_norm_total.toFixed(4)}</dd>
              </div>
            </dl>
            <p className="small">
              Referencia: un modelo uniforme sobre 50 caracteres tendría pérdida log(50) ≈ 3,91.
              Que la pérdida esté cerca de ahí con pesos aleatorios es lo esperado, no un fallo.
              Todos los gradientes son finitos: {lossResult.all_finite ? "sí" : "no"}.
            </p>
            <h3>Posición a posición</h3>
            <table>
              <thead>
                <tr>
                  <th>Posición</th>
                  <th>Entrada → objetivo</th>
                  <th>Predicción del modelo</th>
                  <th>Prob. del carácter correcto</th>
                </tr>
              </thead>
              <tbody>
                {lossResult.input_token_ids.map((_, t) => (
                  <tr key={t}>
                    <td>{t}</td>
                    <td>
                      <code>{lossResult.input_text[t]}</code> →{" "}
                      <code>{lossResult.target_text[t]}</code>
                    </td>
                    <td>
                      <code>{lossResult.predicted_token_ids[t]}</code>{" "}
                      {lossResult.predicted_token_ids[t] === lossResult.target_token_ids[t]
                        ? "✓ acierto"
                        : "✗ fallo"}
                    </td>
                    <td>{(lossResult.prob_correct[t] * 100).toFixed(2)} %</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3>Norma del gradiente por grupo</h3>
            <p className="small">
              Cada grupo es un conjunto de parámetros. La norma dice cuánto “tiraría” el error de
              ese grupo en esta ventana; no es un peso, es la magnitud de su gradiente.
            </p>
            <ul className="breakdown">
              {Object.entries(lossResult.grad_norm_by_group).map(([name, norm]) => (
                <li key={name}>
                  <span>{name}</span>
                  <span>{norm.toFixed(4)}</span>
                  <span
                    className="bar"
                    style={{
                      width: `${Math.max(2, (norm / lossResult.grad_norm_total) * 100)}%`,
                    }}
                  />
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="card" id="lab-entrena">
        <h2>7. Laboratorio: entrena en miniatura (y míralo funcionar)</h2>
        <p>
          Aquí sí se entrena de verdad, pero en miniatura: un modelo diminuto aparte (D=16, H=2,
          L=1, F=32, vocabulario de solo 4 tokens), un único lote fijo con la secuencia{" "}
          <code>abcabc…</code> y AdamW con tasa constante 3e-3. Pulsa “Entrenar” y verás la curva de
          pérdida bajar y la generación greedy pasar de ruido a repetir <code>abc</code>.
        </p>
        <label className="field">
          <span>Actualizaciones (entre 10 y 300; 60 es rápido y suficiente)</span>
          <input
            type="number"
            min={10}
            max={300}
            value={trainUpdates}
            onChange={(event) => setTrainUpdates(Number.parseInt(event.target.value, 10) || 60)}
          />
        </label>
        <button onClick={handleTrainDemo} disabled={trainLoading}>
          {trainLoading ? "Entrenando…" : "Entrenar en miniatura"}
        </button>
        {trainError && <ErrorBox message={trainError} />}
        {trainResult && (
          <>
            <p className="small">
              {trainResult.task} · {trainResult.model} · {trainResult.optimizer}
            </p>
            <dl className="config-grid">
              <div>
                <dt>Pérdida inicial</dt>
                <dd>{trainResult.loss_before.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Pérdida final</dt>
                <dd>{trainResult.loss_after.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Pérdida final / inicial</dt>
                <dd>
                  {trainResult.loss_ratio !== null
                    ? `${(trainResult.loss_ratio * 100).toFixed(1)} %`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Exactitud final</dt>
                <dd>{(trainResult.accuracy_after * 100).toFixed(1)} %</dd>
              </div>
            </dl>
            <h3>Curva de pérdida (antes de cada actualización mostrada)</h3>
            <ul className="curve">
              {trainResult.curve.map((point) => (
                <li key={point.update}>
                  <span>Paso {point.update}</span>
                  <span>{point.loss_before_update.toFixed(4)}</span>
                  <span
                    className="bar"
                    style={{
                      width: `${Math.max(2, (point.loss_before_update / trainResult.loss_before) * 100)}%`,
                    }}
                  />
                </li>
              ))}
            </ul>
            <h3>Generación greedy con el prompt “abc”</h3>
            <p>
              Antes: <code>{trainResult.greedy_before}</code>
            </p>
            <p>
              Después: <code>{trainResult.greedy_after}</code>
            </p>
            <p className="notice">{trainResult.warning}</p>
          </>
        )}
      </section>

      <section className="card" id="demo-real">
        <h2>8. El entrenamiento real que aplicamos en este repo</h2>
        <p>
          Los laboratorios de arriba son juguetes controlados para entender el mecanismo. El
          entrenamiento real de la Fase 2 se hizo con la CLI, sobre <code>data/ejemplo.txt</code>
          (1.127 caracteres), con partición 90/10 (train 1.014 tokens, validation 113 tokens,
          vocabulario construido solo con train, 48 tokens, <code>&lt;UNK&gt;</code> en validation
          1,77 %).
        </p>
        <dl className="config-grid">
          <div>
            <dt>Configuración demo</dt>
            <dd>D=64, H=4, L=2, F=256, C=T=64, B=4</dd>
          </div>
          <div>
            <dt>Optimizador y tasa</dt>
            <dd>AdamW, warmup 50 → coseno 1e-3 → 1e-4, decay 0,01</dd>
          </div>
          <div>
            <dt>Actualizaciones</dt>
            <dd>500 (128.000 tokens procesados, ~40 s)</dd>
          </div>
          <div>
            <dt>Clipping</dt>
            <dd>Norma global máxima 1,0</dd>
          </div>
        </dl>
        <table>
          <thead>
            <tr>
              <th>Medida (validation, completa)</th>
              <th>Pérdida</th>
              <th>Perplexity</th>
              <th>Exactitud</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Antes de entrenar</td>
              <td>3,9564</td>
              <td>52,27</td>
              <td>1,56 %</td>
            </tr>
            <tr>
              <td>Después (paso 500)</td>
              <td>2,3728</td>
              <td>10,73</td>
              <td>32,81 %</td>
            </tr>
            <tr>
              <td>Mejor checkpoint (paso 300)</td>
              <td>2,0707</td>
              <td>7,93</td>
              <td>37,50 %</td>
            </tr>
            <tr>
              <td>Baseline uniforme</td>
              <td>3,8712</td>
              <td>48,00</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Baseline unigrama (solo train)</td>
              <td>2,9928</td>
              <td>19,94</td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
        <p className="small">
          Generación con el mismo prompt y semilla: antes, ruido casi aleatorio; después,{" "}
          <em>
            “La inteligenciarten iaba erten lúmos endero en ero endizafir?: me endertl mestroda e
            sto prones. Es esiroso es este”
          </em>
          . Más plausible no significa que “entienda”: son patrones de caracteres aprendidos.
        </p>
        <p className="notice">
          ⚠️ Límite honesto: la validation de la demo cubre una sola ventana de 64 objetivos (48
          tokens quedan fuera del protocolo stride-T), así que la diferencia entre el paso 300 y el
          500 es ruidosa y el modelo final es peor que el mejor. Por eso existe{" "}
          <code>best.npz</code>. Nada de esto demuestra generalización: el corpus es diminuto.
        </p>
        <h3>Cómo verlo en funcionamiento tú mismo</h3>
        <ol className="steps">
          <li>
            Arranca el backend y el front (dos terminales, desde la raíz del repo):{" "}
            <code>.venv/bin/python visualizer/server.py</code> y{" "}
            <code>cd visualizer/frontend && npm run dev</code>.
          </li>
          <li>
            En esta web: primero el laboratorio de pérdida (mide, sin entrenar) y después el de
            mini-entrenamiento (entrena de verdad, en segundos).
          </li>
          <li>
            En la CLI, la prueba de memorización completa:{" "}
            <code>
              PYTHONPATH=src python3 -m numpy_llm overfit --config configs/overfit.json --outdir
              runs/overfit
            </code>{" "}
            (pérdida 1,3996 → 0,0025, exactitud 100 % en 300 pasos).
          </li>
          <li>
            En la CLI, la demo completa y su evaluación:{" "}
            <code>
              PYTHONPATH=src python3 -m numpy_llm train --config configs/demo.json --corpus
              data/ejemplo.txt --outdir runs/demo
            </code>{" "}
            y{" "}
            <code>
              PYTHONPATH=src python3 -m numpy_llm evaluate --checkpoint runs/demo/best.npz --corpus
              data/ejemplo.txt --split val --full
            </code>
            .
          </li>
          <li>
            Comprueba los gradientes antes de creer ningún entrenamiento:{" "}
            <code>PYTHONPATH=src python3 -m numpy_llm gradcheck --scope all --seed 42</code>{" "}
            (error absoluto máximo 4,84e-09 en el modelo diminuto).
          </li>
        </ol>
        <p className="small">
          Detalle completo en <code>docs/entrenamiento.md</code>, <code>docs/derivadas.md</code> y{" "}
          <code>docs/experimentos.md</code>. Próximos pasos del visualizador (fuera de esta
          entrega): mapa de calor de atención por bloque y cabeza, gráfico de logits y
          probabilidades, generación paso a paso y explorador de pérdida por posición.
        </p>
      </section>
    </div>
  );
}
