export interface ParameterInfo {
  name: string;
  shape: number[];
  size: number;
}

export interface ModelInfo {
  config: Record<string, number | string>;
  vocab_size: number;
  vocab: string[];
  total_parameters: number;
  parameter_breakdown: { name: string; size: number }[];
  parameters: ParameterInfo[];
  untrained_notice: string;
}

export interface TokenView {
  char: string;
  id: number;
  is_unknown: boolean;
  display: string;
}

export interface TokenizeResult {
  text: string;
  token_ids: number[];
  tokens: TokenView[];
}

export interface ForwardStep {
  step: string;
  description: string;
  shape: number[];
}

export interface ForwardResult {
  text: string;
  token_ids: number[];
  tokens: TokenView[];
  steps: ForwardStep[];
  logits_shape: number[];
}

export interface GenerateResult {
  prompt: string;
  prompt_token_ids: number[];
  generated_token_ids: number[];
  full_token_ids: number[];
  generated_text: string;
  full_text: string;
  temperature: number;
  top_k: number | null;
  seed: number;
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error((data as { error?: string }).error ?? `Error en ${path}`);
  }
  return data as T;
}

export function fetchInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/info");
}

export function tokenize(text: string): Promise<TokenizeResult> {
  return request<TokenizeResult>("/api/tokenize", { text });
}

export function forward(text: string): Promise<ForwardResult> {
  return request<ForwardResult>("/api/forward", { text });
}

export interface LossDemoResult {
  text: string;
  input_token_ids: number[];
  target_token_ids: number[];
  input_text: string;
  target_text: string;
  loss: number;
  perplexity: number;
  token_accuracy: number;
  predicted_token_ids: number[];
  prob_correct: number[];
  grad_norm_by_group: Record<string, number>;
  grad_norm_total: number;
  all_finite: boolean;
}

export interface TrainDemoResult {
  task: string;
  model: string;
  optimizer: string;
  updates: number;
  loss_before: number;
  loss_after: number;
  accuracy_before: number;
  accuracy_after: number;
  loss_ratio: number | null;
  curve: { update: number; loss_before_update: number }[];
  greedy_before: string;
  greedy_after: string;
  prompt: string;
  warning: string;
}

export function lossDemo(text: string): Promise<LossDemoResult> {
  return request<LossDemoResult>("/api/loss-demo", { text });
}

export function trainDemo(updates: number): Promise<TrainDemoResult> {
  return request<TrainDemoResult>("/api/train-demo", { updates });
}

export function generate(input: {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number;
}): Promise<GenerateResult> {
  return request<GenerateResult>("/api/generate", input);
}

export function formatShape(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

export function displayChar(char: string): string {
  if (char === "\n") return "↵";
  if (char === " ") return "␣";
  return char;
}
