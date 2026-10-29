// Real browser/HTTP acceptance. Production writes are restricted to registered test identities.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
const { chromium } = await import(pathToFileURL(process.env.SIGNALFEED_PLAYWRIGHT || path.resolve('frontend/node_modules/playwright/index.mjs')));
const base = process.env.SIGNALFEED_BASE_URL;
const output = process.env.SIGNALFEED_ACCEPT_REPORT;
if (!base || !output) throw Error('Explicit base URL and output required');
const health = await (await fetch(base + '/api/health')).json();
if (!health.test_events && process.env.SIGNALFEED_ACCEPT_PRODUCTION !== 'yes') throw Error('Production acceptance not enabled');
const browser = await chromium.launch({headless:true});
const report = {base, release:health.release, started:new Date().toISOString(), browsers:[], first:[], concurrent:[], checks:[]};
function assert(ok, message) { if (!ok) throw Error(message); }
async function context(label) {
  const ctx=await browser.newContext();
  const initial=await ctx.request.get(base+'/api/interests');
  assert(initial.ok(), 'Create anonymous identity');
  const cookie=(await ctx.cookies()).find(c=>c.name==='signalfeed_session');
  assert(cookie, 'Session cookie missing');
  if (!health.test_events) {
    const file=process.env.SIGNALFEED_TEST_USERS_FILE;
    if (!file) throw Error('Explicit exclusion policy required');
    const policy=JSON.parse(await fs.readFile(file,'utf8'));
    policy.users.push({id:cookie.value,reason:`Release acceptance ${label}; all feedback excluded`});
    const temp=file+'.accept-tmp';
    await fs.writeFile(temp,JSON.stringify(policy,null,2),{mode:0o600});
    await fs.rename(temp,file);
  }
  return {ctx,label,identity_sha256:createHash('sha256').update(cookie.value).digest('hex')};
}
async function interests(user, topics) {
  const res=await user.ctx.request.put(base+'/api/interests',{data:{topics}});
  assert(res.ok(),'Set interests');
}
async function coldHTTP(user) {
  const start=performance.now();
  const response=await user.ctx.request.get(base+'/api/feed?cold=true');
  const ms=performance.now()-start;
  const data=await response.json();
  assert(response.ok() && data.items?.length,'Cold HTTP nonempty response');
  return {label:user.label,ms,status:data.experiment.status,request_id:data.request_id,timing:response.headers()['server-timing'],data};
}
async function clickCold(page, first) {
  const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/feed?cold=true'));
  const start=performance.now();
  await page.getByRole('button',{name:first?'兴趣加速（实验）':'根据最新阅读更新',exact:true}).click();
  const response=await responsePromise;
  const received=performance.now();
  const data=await response.json();
  assert(response.ok() && data.items?.length,'Cold browser response');
  await page.locator('.loading').waitFor({state:'hidden'});
  await page.locator('article').first().waitFor();
  await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  return {ms:performance.now()-start,response_ms:received-start,status:data.experiment.status,request_id:data.request_id,timing:response.headers()['server-timing'],data};
}
function publicSample(sample) {const {data,...rest}=sample;return rest;}
try {
  const a=await context('A'), b=await context('B');
  await interests(b,['Security','Science']);
  for (const user of [a,b]) {
    const page=await user.ctx.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);await page.locator('article').first().waitFor();
    const samples=[];
    samples.push(publicSample(await clickCold(page,true)));
    if(user===a) await interests(a,['Artificial Intelligence','Programming']);
    for(let i=0;i<12;i++) {
      const sample=publicSample(await clickCold(page,false));samples.push(sample);
      if(i>=2 && sample.status==='cache') break;
      // Let a cold worker finish; busy responses are still included in the report.
      await new Promise(r=>setTimeout(r,350));
    }
    // Distinct bookmarks persist; click history is written through the public event API.
    const feed=await coldHTTP(user);
    const item=feed.data.items[user===a?0:1];
    for(const kind of ['impression','click','bookmark']) {
      const res=await user.ctx.request.post(base+'/api/events',{data:{event_id:randomUUID(),request_id:feed.data.request_id,item_id:item.id,kind,position:item.position}});
      assert(res.ok(),kind+' persisted');
    }
    user.saved=item.id;
    user.feed=feed;
    report.browsers.push({label:user.label,identity_sha256:user.identity_sha256,samples,errors});
    assert(!errors.length,'Browser JS errors');
    await page.close();
  }
  for (const [user,topics] of [[a,['Artificial Intelligence','Programming']],[b,['Security','Science']]]) {
    const own=await (await user.ctx.request.get(base+'/api/interests')).json();
    assert(JSON.stringify(own.topics)===JSON.stringify(topics),'Interest isolation');
    const saved=await (await user.ctx.request.get(base+'/api/bookmarks')).json();
    assert(saved.items.length===1 && saved.items[0].id===user.saved,'Bookmark isolation');
  }
  const foreign=await b.ctx.request.post(base+'/api/events',{data:{event_id:randomUUID(),request_id:a.feed.data.request_id,item_id:a.feed.data.items[0].id,kind:'click',position:0}});
  assert(foreign.status()===404,'Cross-user feedback must be rejected');
  report.checks.push('interests independent','bookmarks independent','foreign request rejected');
  report.concurrent=(await Promise.all(Array.from({length:8},(_,i)=>coldHTTP(i%2?a:b)))).map(publicSample);
  for(let i=0;i<Number(process.env.SIGNALFEED_FIRST_SAMPLES||20);i++) {
    const user=await context('new-'+i);
    // Every measured request is the identity's first ColdFuse request.
    const topics=i%2?['Security','Science']:['Artificial Intelligence','Programming'];
    await interests(user,topics);
    const page=await user.ctx.newPage();await page.goto(base);await page.locator('article').first().waitFor();
    const sample=await clickCold(page,true);
    report.first.push({...publicSample(sample),label:user.label,identity_sha256:user.identity_sha256});
    await user.ctx.close();
  }
  const all=report.browsers.flatMap(b=>b.samples).concat(report.concurrent,report.first);
  const personalized=all.filter(s=>s.status!=='no-personal-signal');
  const statuses={};for(const s of personalized)statuses[s.status]=(statuses[s.status]||0)+1;
  const times=report.first.map(s=>s.ms).sort((a,b)=>a-b);
  report.summary={first_n:times.length,first_p95_ms:times[Math.ceil(times.length*.95)-1],first_max_ms:times.at(-1),personalized_n:personalized.length,statuses};
  assert(personalized.some(s=>s.status==='research'),'Real personalized inference unavailable');
  assert(personalized.some(s=>s.status==='cache'),'Personalized cache never hit');
  assert(report.summary.first_p95_ms<=3000,'First browser p95 exceeds target');
  if(health.release) assert(all.every(s=>s.timing?.includes('server_total')),'Server-Timing missing');
  report.passed=true;
} catch(error) {report.passed=false;report.error=String(error);process.exitCode=1;}
finally {await browser.close();await fs.writeFile(output,JSON.stringify(report,null,2),{mode:0o600});}
console.log(JSON.stringify({passed:report.passed,error:report.error,summary:report.summary,report:output}));
