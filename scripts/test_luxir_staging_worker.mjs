import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../src/worker.js', import.meta.url), 'utf8');
const {default: worker} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const env = {CHART_LUXIR_DEMO:'1', CHART_LUXIR_STAGING:'1', CHART_NAMESPACE:'chart_demo_staging',
 LAYER_GATEWAY_URL:'http://chart-luxir-layer-gateway.chart-luxir.svc.cluster.local:8080',
 LAYER_GATEWAY_API_KEY:'test-only', CHART_DEMO_BASIC_USER:'viewer', CHART_DEMO_BASIC_PASSWORD:'test-only'};
const request = (path,auth) => new Request('https://chart-luxir.example'+path, {headers: auth ? {authorization:auth} : {}});
for (const path of ['/', '/api/config', '/api/search']) assert.equal((await worker.fetch(request(path),env)).status,401);
assert.equal((await worker.fetch(request('/api/config','Basic '+btoa('viewer:wrong')),env)).status,401);
assert.equal((await worker.fetch(request('/api/config','Basic '+btoa('viewer:test-only')),env)).status,200);
for (const overrides of [{LAYER_GATEWAY_URL:'http://127.0.0.1:8080'}, {CHART_NAMESPACE:'chart-notes'}, {LAYER_GATEWAY_API_KEY:''}, {CHART_DEMO_BASIC_USER:''}, {CHART_DEMO_BASIC_PASSWORD:''}]) {
 assert.equal((await worker.fetch(request('/api/config'), {...env,...overrides})).status,400);
}
// A staging flag alone must never enable demo behavior in an ordinary Worker.
assert.equal((await worker.fetch(request('/api/config'), {...env,CHART_LUXIR_STAGING:'',LAYER_GATEWAY_URL:env.LAYER_GATEWAY_URL})).status,400);
assert.equal((await worker.fetch(request('/api/config'), {...env,CHART_LUXIR_DEMO:''})).status,400);
assert.equal((await worker.fetch(request('/healthz'),env)).status,200);
console.log('PASS staging Worker auth and target confinement; no backend mocked or called');
