(() => {
  const mn = (value, digits) => CourseNumbers.mathml(value, digits);
  const math = body => `<math>${body}</math>`;
  const value = number => math(mn(number));
  const equal = (name, number) => math(`<mrow><mi${name.length > 1 ? ' mathvariant="normal"' : ''}>${name}</mi><mo>=</mo>${mn(number)}</mrow>`);
  const root = document.querySelector('#neuron-fit'); if (!root) return;
  const q = s => root.querySelector(s), xs = [-2,-1,0,1,2], ys = [-3,-1,1,3,5];
  const sx = x => 60+(x+3)*85, sy = y => 285-(y+5)*20;
  const node = (tag, attrs, text) => { const n=document.createElementNS('http://www.w3.org/2000/svg',tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if(text) n.textContent=text; return n; };
  function draw() {
    const w=Number(q('[data-w]').value), b=Number(q('[data-b]').value), svg=q('svg');
    svg.replaceChildren(node('title',{id:'fit-title'},'Sensor calibration'),node('desc',{id:'fit-desc'},'Solid model line, target dots and dashed residuals. Exact values follow the plot.'));
    for(let x=-3;x<=3;x++) svg.append(node('line',{x1:sx(x),x2:sx(x),y1:25,y2:285,stroke:'var(--line)'}),node('text',{x:sx(x),y:312,fill:'currentColor','text-anchor':'middle'},String(x)));
    for(let y=-4;y<=8;y+=2) svg.append(node('text',{x:45,y:sy(y)+5,fill:'currentColor','text-anchor':'end'},String(y)));
    const defs=node('defs',{}),clip=node('clipPath',{id:'fit-clip'});clip.append(node('rect',{x:60,y:25,width:510,height:260}));defs.append(clip);svg.append(defs);
    svg.append(node('line',{x1:sx(-3),y1:sy(-3*w+b),x2:sx(3),y2:sy(3*w+b),stroke:'var(--accent)','stroke-width':3,'clip-path':'url(#fit-clip)'}));
    xs.forEach((x,i)=>svg.append(node('line',{x1:sx(x),x2:sx(x),y1:sy(ys[i]),y2:sy(w*x+b),stroke:'currentColor','stroke-dasharray':'4 4','clip-path':'url(#fit-clip)'}),node('circle',{cx:sx(x),cy:sy(ys[i]),r:5,fill:'currentColor'})));
    const loss=xs.reduce((s,x,i)=>s+(w*x+b-ys[i])**2,0)/5;
    q('[data-readout]').innerHTML=`${equal('w',w)}; ${equal('b',b)}; ${equal('MSE',loss)}; prediction at ${equal('x','0.5')}: ${value((w*0.5+b))}. Lines outside the plot are clipped; values remain in the table.`;
    q('[data-table]').innerHTML='<table><caption>Predictions and residuals (prediction minus target)</caption><thead><tr><th><math><mi>x</mi></math></th><th>Target</th><th>Prediction</th><th>Residual</th></tr></thead><tbody>'+xs.map((x,i)=>`<tr><td>${value(x)}</td><td>${value(ys[i])}</td><td>${value((w*x+b))}</td><td>${value((w*x+b-ys[i]))}</td></tr>`).join('')+'</tbody></table>';
  }
  root.addEventListener('input',draw);q('[data-reset]').addEventListener('click',()=>{q('[data-w]').value=0;q('[data-b]').value=0;draw();});draw();
})();
