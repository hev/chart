// Runs the real Worker handler with native fetch to the owned gateway, no browser.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const [gateway, namespace] = process.argv.slice(2);
assert(gateway && namespace);
const source = await readFile(new URL('../src/worker.js', import.meta.url), 'utf8');
const { default: worker } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const env = { CHART_LUXIR_DEMO: '1', LAYER_GATEWAY_URL: gateway, CHART_NAMESPACE: namespace, LAYER_GATEWAY_API_KEY: 'local-demo' };
async function call(path, status = 200, expected) {
  const response = await worker.fetch(new Request('http://127.0.0.1:8787' + path), env);
  const value = await response.json();
  console.log(JSON.stringify({ path, status: response.status, response: value }));
  assert.equal(response.status, status);
  if (expected) assert.deepEqual(value.rows.map(r => r.id).sort(), expected.sort());
  if (value.rows) assert.equal(value.routing, null);
  return value;
}
assert.equal((await call('/api/config')).profile, 'luxir_lexical_demo');
await call('/api/search', 200, ['case_001','case_002','case_003','case_004','case_005','case_006']);
await call('/api/search?q=dyspnea', 200, ['case_001','case_002','case_004']);
await call('/api/search?q=dyspnea&f=specialty:cardiology&f=age_band:older_adult', 200, ['case_001']);
await call('/api/search?q=metformin', 200, ['case_003']);
for (const path of ['/api/facets','/api/facet-counts','/api/storage','/api/similar/case_001','/api/search?agentic=1','/api/search?li=1','/api/search?f=events:x']) await call(path,422);
const unsafe = await worker.fetch(new Request('http://127.0.0.1:8787/api/config'), {...env, LAYER_GATEWAY_URL: 'https://example.com'});
assert.equal(unsafe.status, 400);
console.log('PASS real Worker handler -> Layer -> native Luxir; no embedding service or browser');
