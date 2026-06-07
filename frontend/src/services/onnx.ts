// Runs the trained Random Forest entirely in the browser via ONNX Runtime Web.
import * as ort from 'onnxruntime-web';
import meta from '../model_meta.json';
import { PredictionResponse } from '../types';

// Serve the matching WASM from CDN (works on static hosting)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

const NAMES: string[] = (meta as any).features;
const MEDIANS: number[] = (meta as any).medians;
const MEAN: number[] = (meta as any).mean;
const SCALE: number[] = (meta as any).scale;
const CLASSES: string[] = (meta as any).classes;

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

let session: ort.InferenceSession | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!session) {
    const url = `${process.env.PUBLIC_URL || ''}/model.onnx`;
    session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  }
  return session;
}

// Build the standardized feature vector: provided finite value, else imputed median.
// Non-finite inputs (NaN, ±Inf) are treated as missing, matching the training imputer.
function vectorize(features: Record<string, number>): Float32Array {
  const vec = new Float32Array(NAMES.length);
  for (let i = 0; i < NAMES.length; i++) {
    const raw = features[NAMES[i]];
    const filled = Number.isFinite(raw) ? raw : MEDIANS[i];
    vec[i] = (filled - MEAN[i]) / SCALE[i];
  }
  return vec;
}

export async function predictLocal(features: Record<string, number>): Promise<PredictionResponse> {
  const matched = NAMES.filter((n) => Number.isFinite(features[n])).length;
  if (matched === 0) {
    throw new Error(
      `No recognised features. Expected names like ${NAMES.slice(0, 4).join(', ')}, ...`
    );
  }
  const t0 = performance.now();
  const sess = await getSession();
  const input = new ort.Tensor('float32', vectorize(features), [1, NAMES.length]);
  const output = await sess.run({ input });

  // The probabilities output is the 2-D tensor
  const probaKey = Object.keys(output).find((k) => output[k].dims.length === 2)!;
  const proba = Array.from(output[probaKey].data as Float32Array);

  let best = 0;
  for (let i = 1; i < proba.length; i++) if (proba[i] > proba[best]) best = i;

  const probabilities: Record<string, number> = {};
  CLASSES.forEach((c, i) => (probabilities[cap(c)] = proba[i]));

  return {
    prediction: best,
    class_name: cap(CLASSES[best]),
    probabilities,
    confidence: proba[best],
    model_used: 'Random Forest (ONNX, in-browser)',
    inference_time_ms: performance.now() - t0,
    timestamp: new Date().toISOString(),
  };
}

export const FEATURE_NAMES = NAMES;
