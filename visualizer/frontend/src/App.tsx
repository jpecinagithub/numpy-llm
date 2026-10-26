import { useEffect, useState } from "react";
import {
  displayChar,
  fetchInfo,
  forward,
  formatShape,
  generate,
  tokenize,
  type ForwardResult,
  type GenerateResult,
  type ModelInfo,
  type TokenizeResult,
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
        <p className="eyebrow">numpy-llm · Fase 1 · Visualizador educativo</p>
        <h1>¿Qué hace el modelo por dentro?</h1>
        <p className="subtitle">
          Un pequeño Transformer decoder-only en Python y NumPy. Esta web no reescribe el modelo:
          pide los resultados al backend Python real, el mismo código que usa la CLI.
        </p>
        <p className="notice">
          ⚠️ Modelo <strong>sin entrenar</strong>: sus pesos son aleatorios y el texto generado es
          esencialmente aleatorio. Eso no es un fallo, es el punto de partida antes de entrenar.
        </p>
        <p className="credit">Proyecto educativo de Jon Peciña (jpecina).</p>
      </header>

      <nav className="nav" aria-label="Secciones">
        <a href="#recorrido">Recorrido</a>
        <a href="#tokenizador">Tokenizador</a>
        <a href="#inspector">Inspector</a>
        <a href="#generacion">Generación</a>
        <a href="#limites">Límites de la Fase 1</a>
      </nav>

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
        <h2>5. Hasta dónde llega esta Fase 1</h2>
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
          Próximos pasos del visualizador (fuera de este MVP): mapa de calor de atención por bloque y
          cabeza, gráfico de logits y probabilidades, generación paso a paso y explorador de pérdida
          por posición.
        </p>
      </section>
    </div>
  );
}
