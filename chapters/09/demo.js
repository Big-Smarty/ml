(() => {
  'use strict';
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
  const table=(caption,heads,rows)=>'<table><caption>'+caption+'</caption><thead><tr>'+heads.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(s=>'<td>'+s+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  function render(){
    const s=Number(scale.value),out=root.querySelector('[data-readout]'),body=root.querySelector('[data-table]'),note=root.querySelector('[data-note]');
    if(!scale.value||!Number.isFinite(s)||s<0||s>2){out.textContent='Enter a multiplier from 0 to 2.';return;}
    fixture.disabled=stage.value!=='digits';
    const grid=root.querySelector('[data-digit-grid]');grid.hidden=stage.value!=='digits';
    if(stage.value==='matrix'){
      out.textContent=`X [2,3] multiplied by ${s}; W [2,3] and bias unchanged. Y [2,2]=[${dense(s).map(v=>v.toFixed(2)).join(', ')}]. Flat X[1,2] is offset 1×3+2=5; W[0,2] is offset 2.`;
      body.innerHTML=table('Output [0,0]: three products plus bias 0.5',['i','X[0,i]','W[0,i]','Product'],[1,2,3].map((x,i)=>[i,(s*x).toFixed(2),[1,0,-1][i],(s*x*[1,0,-1][i]).toFixed(2)]));
      note.textContent='Each weight row is one output unit. Bias is added once, so scaling X does not scale the entire output. Deterministic scalar illustration, not a Rust run.';
    }else if(stage.value==='digits'){
      const r=digit(fixture.value,s);
      out.textContent=`Synthetic block input [1,15], weights [10,15], scores [1,10]. Predicted class ${r.prediction}. Probabilities sum to ${r.probabilities.reduce((a,b)=>a+b,0).toFixed(6)}. Pixel rows: ${Array.from({length:5},(_,i)=>r.pixels.slice(i*3,i*3+3).map(v=>v.toFixed(1)).join(' ')).join(' / ')}.`;
      grid.innerHTML='<svg width="180" height="235" viewBox="0 0 180 235" role="img" aria-labelledby="digit-grid-title digit-grid-desc"><title id="digit-grid-title">Synthetic 5 by 3 digit pixel grid</title><desc id="digit-grid-desc">Each square is one row-major pixel; its printed value is the model input. Rows and columns map to fifteen features, not handwriting recognition.</desc>'+r.pixels.map((v,i)=>{const row=Math.floor(i/3),col=i%3;const shade=Math.round(255*(1-Math.min(1,v/2)));return `<rect x="${col*55+5}" y="${row*44+5}" width="50" height="39" fill="rgb(${shade},${shade},${shade})" stroke="currentColor"/><text x="${col*55+14}" y="${row*44+30}" fill="${shade>116?'#000':'#fff'}" font-size="15">${v.toFixed(1)}</text>`;}).join('')+'</svg>';
      body.innerHTML=table('Hand-built prototype scores and maximum-shifted softmax',['Class','Score','Probability'],r.logits.map((z,c)=>[c,z.toFixed(3),r.probabilities[c].toFixed(4)]));
      note.textContent='Prototype weights are twice each binary glyph; bias is minus its bright-pixel count. These hand-built scores are not trained weights or MNIST results. Identical blanks always give identical predictions; confidence does not make them readable.';
    }else{
      out.textContent=`One parameter starts at 1. Gradients [${2*s},0], rate 0.1, β₁=0.9, β₂=0.999, ε=1e−8. Both states begin at zero.`;
      body.innerHTML=table('Two complete state updates',['Step','Gradient','Velocity','Momentum θ','Adam m','Adam v','Adam θ'],optimizer(s).map(r=>[r.t,r.g,r.velocity.toFixed(4),r.momentum.toFixed(6),r.m.toFixed(4),r.v.toFixed(6),r.adam.toFixed(6)]));
      note.textContent=s===0?'Both gradients are zero; both moment buffers stay zero and neither parameter moves. A scalar illustration, not a training benchmark.':'The second gradient is zero, but both optimizers keep moving from remembered state. Adam corrects its moments using the displayed step count. A scalar illustration, not a training benchmark.';
    }
  }
  [stage,scale,fixture].forEach(control=>control.addEventListener('change',render));
  root.querySelector('[data-reset]').addEventListener('click',()=>{stage.value='matrix';scale.value='1';fixture.value='0';render();});render();
})();
