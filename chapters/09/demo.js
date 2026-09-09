(() => {
  'use strict';
  // Fixed MathML templates; dynamic values are finite numbers, never input markup.
  const math = body => '<math xmlns="http://www.w3.org/1998/Math/MathML">'+body+'</math>';
  const scalar = value => { const text=String(value); return text.startsWith('-')?'<mrow><mo>−</mo><mn>'+text.slice(1)+'</mn></mrow>':'<mn>'+text+'</mn>'; };
  const number = value => math(scalar(value));
  const vector = values => math('<mrow><mo>[</mo>'+values.map(scalar).join('<mo>,</mo>')+'<mo>]</mo></mrow>');
  const formula = {"X": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>X</mi></mrow></math>", "xs": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>X</mi><mspace width=\"0.25em\"></mspace><mo stretchy=\"false\">[</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo stretchy=\"false\">]</mo></mrow></math>", "ws": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>W</mi><mspace width=\"0.25em\"></mspace><mo stretchy=\"false\">[</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo stretchy=\"false\">]</mo></mrow></math>", "ys": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>Y</mi><mspace width=\"0.25em\"></mspace><mo stretchy=\"false\">[</mo><mn>2</mn><mo>,</mo><mn>2</mn><mo stretchy=\"false\">]</mo></mrow></math>", "xoffset": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>X</mi><mo stretchy=\"false\">[</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo stretchy=\"false\">]</mo></mrow></math>", "offset": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mn>1</mn><mo>×</mo><mn>3</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>5</mn></mrow></math>", "woffset": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>W</mi><mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mn>2</mn><mo stretchy=\"false\">]</mo></mrow></math>", "output": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mn>0</mn><mo stretchy=\"false\">]</mo></mrow></math>", "i": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>i</mi></mrow></math>", "xi": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>X</mi><mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mi>i</mi><mo stretchy=\"false\">]</mo></mrow></math>", "wi": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>W</mi><mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mi>i</mi><mo stretchy=\"false\">]</mo></mrow></math>", "inputshape": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mo stretchy=\"false\">[</mo><mn>1</mn><mo>,</mo><mn>15</mn><mo stretchy=\"false\">]</mo></mrow></math>", "weightshape": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mo stretchy=\"false\">[</mo><mn>10</mn><mo>,</mo><mn>15</mn><mo stretchy=\"false\">]</mo></mrow></math>", "scoreshape": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mo stretchy=\"false\">[</mo><mn>1</mn><mo>,</mo><mn>10</mn><mo stretchy=\"false\">]</mo></mrow></math>", "betas": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><msub><mi>β</mi><mn>1</mn></msub><mo>=</mo><mn>0.9</mn><mo>,</mo><mspace width=\"1em\"></mspace><msub><mi>β</mi><mn>2</mn></msub><mo>=</mo><mn>0.999</mn><mo>,</mo><mspace width=\"1em\"></mspace><mi>ε</mi><mo>=</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>8</mn></mrow></msup></mrow></math>", "theta": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>θ</mi></mrow></math>", "m": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>m</mi></mrow></math>", "v": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>v</mi></mrow></math>"};
  const glyphs=['111101101101111','010110010010111','111001111100111','111001111001111','101101111001001','111100111001111','111100111101111','111001010010010','111101111101111','111101111001111'];
  function dense(scale){return [-2*scale+0.5,4*scale-0.5,-2*scale+0.5,13*scale-0.5];}
  function digit(key,scale){
    const pixels=(key==='blank'?'000000000000000':glyphs[Number(key)]).split('').map(v=>Number(v)*scale);
    // Prototype-distance scores: -||x-prototype||² with the common -||x||² omitted.
    const logits=glyphs.map(g=>g.split('').reduce((s,v,i)=>s+2*Number(v)*pixels[i]-Number(v),0));
    const maximum=Math.max(...logits),exps=logits.map(z=>Math.exp(z-maximum)),sum=exps.reduce((a,b)=>a+b,0);
    return {pixels,logits,probabilities:exps.map(v=>v/sum),prediction:logits.indexOf(maximum)};
  }
  function optimizer(scale){
    let momentum=1,velocity=0,adam=1,m=0,v=0;
    return [2*scale,0].map((g,index)=>{
      velocity=.9*velocity+g;momentum-=.1*velocity;
      m=.9*m+.1*g;v=.999*v+.001*g*g;
      const t=index+1,mh=m/(1-Math.pow(.9,t)),vh=v/(1-Math.pow(.999,t));adam-=.1*mh/(Math.sqrt(vh)+1e-8);
      return {t,g,velocity,momentum,m,v,adam};
    });
  }
  if(typeof module==='object')module.exports={dense,digit,optimizer};
  if(typeof document==='undefined')return;
  const root=document.querySelector('#tensor-digits');if(!root)return;
  const stage=root.querySelector('[data-stage]'),scale=root.querySelector('[data-scale]'),fixture=root.querySelector('[data-digit]');
  const table=(caption,heads,rows)=>'<table><caption>'+caption+'</caption><thead><tr>'+heads.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(s=>'<td>'+number(s)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  function render(){
    const s=Number(scale.value),out=root.querySelector('[data-readout]'),body=root.querySelector('[data-table]'),note=root.querySelector('[data-note]');
    if(!scale.value||!Number.isFinite(s)||s<0||s>2){out.innerHTML='Enter a multiplier from '+number(0)+' to '+number(2)+'.';return;}
    fixture.disabled=stage.value!=='digits';
    const grid=root.querySelector('[data-digit-grid]');grid.hidden=stage.value!=='digits';
    if(stage.value==='matrix'){
      out.innerHTML=`${formula.xs} multiplied by ${number(s)}; ${formula.ws} and bias unchanged. ${formula.ys} has values ${vector(dense(s).map(v=>v.toFixed(2)))}. Flat ${formula.xoffset} is offset ${formula.offset}; ${formula.woffset} is offset ${number(2)}.`;
      body.innerHTML=table('Output '+formula.output+': three products plus bias '+number(.5),[formula.i,formula.xi,formula.wi,'Product'],[1,2,3].map((x,i)=>[i,(s*x).toFixed(2),[1,0,-1][i],(s*x*[1,0,-1][i]).toFixed(2)]));
      note.innerHTML='Each weight row is one output unit. Bias is added once, so scaling '+formula.X+' does not scale the entire output. Deterministic scalar illustration, not a Rust run.';
    }else if(stage.value==='digits'){
      const r=digit(fixture.value,s);
      out.innerHTML=`Synthetic block input ${formula.inputshape}, weights ${formula.weightshape}, scores ${formula.scoreshape}. Predicted class ${number(r.prediction)}. Probabilities sum to ${number(r.probabilities.reduce((a,b)=>a+b,0).toFixed(6))}. Pixel rows: ${math('<mrow><mo>[</mo><mtable>'+Array.from({length:5},(_,i)=>'<mtr>'+r.pixels.slice(i*3,i*3+3).map(v=>'<mtd>'+scalar(v.toFixed(1))+'</mtd>').join('')+'</mtr>').join('')+'</mtable><mo>]</mo></mrow>')}.`;
      grid.innerHTML='<svg width="180" height="235" viewBox="0 0 180 235" role="img" aria-labelledby="digit-grid-title digit-grid-desc"><title id="digit-grid-title">Synthetic 5 by 3 digit pixel grid</title><desc id="digit-grid-desc">Each square is one row-major pixel; its printed value is the model input. Rows and columns map to fifteen features, not handwriting recognition.</desc>'+r.pixels.map((v,i)=>{const row=Math.floor(i/3),col=i%3;const shade=Math.round(255*(1-Math.min(1,v/2)));return `<rect x="${col*55+5}" y="${row*44+5}" width="50" height="39" fill="rgb(${shade},${shade},${shade})" stroke="currentColor"/><text x="${col*55+14}" y="${row*44+30}" fill="${shade>116?'#000':'#fff'}" font-size="15">${v.toFixed(1)}</text>`;}).join('')+'</svg>';
      body.innerHTML=table('Hand-built prototype scores and maximum-shifted softmax',['Class','Score','Probability'],r.logits.map((z,c)=>[c,z.toFixed(3),r.probabilities[c].toFixed(4)]));
      note.textContent='Prototype weights are twice each binary glyph; bias is minus its bright-pixel count. These hand-built scores are not trained weights or MNIST results. Identical blanks always give identical predictions; confidence does not make them readable.';
    }else{
      out.innerHTML=`One parameter starts at ${number(1)}. Gradients ${vector([2*s,0])}, rate ${number(.1)}, ${formula.betas}. Both states begin at ${number(0)}.`;
      body.innerHTML=table('Two complete state updates',['Step','Gradient','Velocity','Momentum '+formula.theta,'Adam '+formula.m,'Adam '+formula.v,'Adam '+formula.theta],optimizer(s).map(r=>[r.t,r.g,r.velocity.toFixed(4),r.momentum.toFixed(6),r.m.toFixed(4),r.v.toFixed(6),r.adam.toFixed(6)]));
      note.textContent=s===0?'Both gradients are zero; both moment buffers stay zero and neither parameter moves. A scalar illustration, not a training benchmark.':'The second gradient is zero, but both optimizers keep moving from remembered state. Adam corrects its moments using the displayed step count. A scalar illustration, not a training benchmark.';
    }
  }
  [stage,scale,fixture].forEach(control=>control.addEventListener('change',render));
  root.querySelector('[data-reset]').addEventListener('click',()=>{stage.value='matrix';scale.value='1';fixture.value='0';render();});render();
})();
