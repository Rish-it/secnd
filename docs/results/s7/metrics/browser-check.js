async () => {
  const settle = async test => {
    const deadline=Date.now()+5000;
    while(!test()) { if(Date.now()>deadline)throw Error('UI did not settle'); await new Promise(r=>setTimeout(r,30)); }
  };
  const button = text => [...document.querySelectorAll('button')].find(b=>b.textContent.trim()===text);
  const text = () => document.querySelector('.metrics-workspace')?.innerText ?? '';
  const select = async name => { button(name).click(); await settle(()=>text().includes(`Recorded ${name} evaluation`)); };
  const report={datasets:{},source_revision:'bae2b3f',api_revision:'d3deb29',timestamp:new Date().toISOString()};
  for(const name of ['MTOP','BANKING77','IPSA']) {
    await select(name);
    const response=await fetch('/api/v1/workspaces/default/metrics?dataset='+name.toLowerCase());
    const data=await response.json();
    const p=n=>(n*100).toFixed(2)+'%';
    if(!text().includes(p(data.card.coverage)))throw Error('Coverage mismatch '+name);
    if(data.card.precision!=null&&!text().includes(p(data.card.precision)))throw Error('Precision mismatch '+name);
    if((data.card.precision!=null&&data.card.precision<.99)!==text().includes('precision target missed'))throw Error('Incorrect target state');
    if(!text().includes(p(data.card.robustness_pass_rate)))throw Error('Re-split pass rate mismatch '+name);
    if(data.card.threshold==null&&!text().includes('Not certified yet'))throw Error('Missing abstention state');
    const bins=document.querySelectorAll('svg[aria-label^="Reliability"] circle').length;
    if(bins!==data.curves.reliability.length)throw Error('Reliability bin mismatch');
    if(!text().includes('99% target'))throw Error('Missing precision target line');
    report.datasets[name]={coverage:data.card.coverage,precision:data.card.precision,threshold:data.card.threshold,pass_rate:data.card.robustness_pass_rate,reliability_bins:bins,tradeoff_points:data.curves.coverage_precision.length,mode:response.headers.get('X-Secnd-Mode')};
  }
  await select('MTOP');
  button('MTOP').click();
  await new Promise(r=>setTimeout(r,50));
  if(!button('Refresh measurements')||button('Refresh measurements').disabled)throw Error('Active selection disabled refresh');
  report.active_selection_refresh=true;
  document.querySelector('summary').click();
  if(!document.querySelector('details').open)throw Error('Measurement table unavailable');
  document.querySelector('summary').click();
  report.data_table=true;
  const original=window.fetch;
  try {
    window.fetch=(url,init)=>String(url).includes('/metrics')?Promise.resolve(new Response(JSON.stringify({message:'Simulated unavailable evaluation'}),{status:503,headers:{'Content-Type':'application/json'}})):original(url,init);
    button('Refresh measurements').click();
    await settle(()=>document.querySelector('[role="alert"]')?.textContent.includes('Simulated unavailable'));
  } finally { window.fetch=original; }
  button('Refresh measurements').click();
  await settle(()=>!document.querySelector('[role="alert"]'));
  report.retry_recovered=true;
  try {
    window.fetch=(url,init)=>String(url).endsWith('/models')?Promise.resolve(new Response(JSON.stringify({message:'Unavailable registry'}),{status:503,headers:{'Content-Type':'application/json'}})):original(url,init);
    button('Refresh measurements').click();
    await settle(()=>text().includes('Workspace versions are unavailable'));
    if(!text().includes('42.42%'))throw Error('Registry failure hid evaluation');
  } finally { window.fetch=original; }
  button('Refresh measurements').click();
  await settle(()=>!text().includes('Workspace versions are unavailable'));
  report.registry_failure_isolated=true;
  if(!text().includes('Your first measured learning curve')||!text().includes('No measured handoff run yet'))throw Error('Unmeasured curve fabricated');
  report.learning_and_handoff_empty=true;
  report.width={document:document.documentElement.scrollWidth,viewport:innerWidth};
  return report;
}
