"use strict";Object.defineProperty(exports,Symbol.toStringTag,{value:"Module"});const E=require("three"),s=require("./controller-268bf0da.js"),tr=require("three/addons/renderers/CSS3DRenderer.js"),rr=require("./ui-3a557476.js");/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sr(r){s.assert(Array.isArray(r),()=>"The argument passed to tf.addN() must be a list of tensors"),s.assert(r.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${r.length}`);const e=r.map((n,i)=>s.convertToTensor(n,`tensors${i}`,"addN")),t=e[0];e.forEach(n=>{if(n.dtype!==t.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(n=>{if(!s.arraysEqual(n.shape,t.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const a=e;return s.ENGINE.runKernel(s.AddN,a)}const Ve=s.op({addN_:sr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ar(r,e,t,a,n,i){const l=s.convertToTensor(r,"forgetBias","basicLSTMCell"),u=s.convertToTensor(e,"lstmKernel","basicLSTMCell"),p=s.convertToTensor(t,"lstmBias","basicLSTMCell"),c=s.convertToTensor(a,"data","basicLSTMCell"),m=s.convertToTensor(n,"c","basicLSTMCell"),d=s.convertToTensor(i,"h","basicLSTMCell"),h=s.concat([c,d],1),f=s.matMul(h,u),b=s.add(f,p),g=b.shape[0],y=b.shape[1]/4,N=[g,y],w=s.slice(b,[0,0],N),S=s.slice(b,[0,y],N),T=s.slice(b,[0,y*2],N),A=s.slice(b,[0,y*3],N),k=s.add(s.mul(s.sigmoid(w),s.tanh(S)),s.mul(m,s.sigmoid(s.add(l,T)))),I=s.mul(s.tanh(k),s.sigmoid(A));return[k,I]}const Be=s.op({basicLSTMCell_:ar});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nr(r,e){const t=s.convertToTensor(r,"x","bitwiseAnd"),a=s.convertToTensor(e,"y","bitwiseAnd");if(!s.arraysEqual(t.shape,a.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${t.shape}, y: ${a.shape}`);if(t.dtype!=="int32"||a.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${t.dtype} and type of y: ${a.dtype}`);const n={a:t,b:a};return s.ENGINE.runKernel(s.BitwiseAnd,n)}const je=s.op({bitwiseAnd_:nr});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ir(r,e){const t=s.convertToTensor(r,"s0","broadcastArgs","int32"),a=s.convertToTensor(e,"s1","broadcastArgs","int32");if(t.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${t.rank}`);if(a.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${a.rank}`);const n={s0:t,s1:a};return s.ENGINE.runKernel(s.BroadcastArgs,n)}const qe=s.op({broadcastArgs_:ir});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function or(r){const t={x:s.convertToTensor(r,"x","diag")};return s.ENGINE.runKernel(s.Diag,t)}const He=s.op({diag_:or});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ur(r,e){const t=s.convertToTensor(r,"x","ensureShape","string_or_numeric");if(!s.arraysEqualWithNull(t.shape,e))throw new Error(`EnsureShape: Shape of tensor ${t.shape} is not compatible with expected shape ${e}`);return r}const Ge=s.op({ensureShape_:ur});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function We(r,e,t){if(t<=0)throw new Error("The number of values should be positive.");const a={start:r,stop:e,num:t};return s.ENGINE.runKernel(s.LinSpace,{},a)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const j=2147483648;function lr(r,e,t="left"){const a=s.convertToTensor(r,"sortedSequence","searchSorted"),n=s.convertToTensor(e,"values","searchSorted"),i=a.shape[a.shape.length-1],l=n.shape[n.shape.length-1],u=s.reshape(a,[-1,i]),p=s.reshape(n,[-1,l]);if(u.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(u.shape[0]!==p.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(s.sizeFromShape(p.shape)>=j)throw new Error(`values tensor size must less than ${j}`);if(u.shape[1]>=j)throw new Error(`trailing dim_size must less than ${j} for int32 output type, was ${u.shape[1]}`);const c={sortedSequence:u,values:p},m={side:t};return s.ENGINE.runKernel(s.SearchSorted,c,m)}const X=s.op({searchSorted_:lr});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ue(r,e){return X(r,e,"left")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pr(r,e,t,a,n=!1){const l={x:s.convertToTensor(r,"x","maxPoolWithArgmax")},u={filterSize:e,strides:t,pad:a,includeBatchInIndex:n},p=s.ENGINE.runKernel(s.MaxPoolWithArgmax,l,u);return{result:p[0],indexes:p[1]}}const Ke=s.op({maxPoolWithArgmax_:pr});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Je(r,e,{indexing:t="xy"}={}){if(t!=="xy"&&t!=="ij")throw new TypeError(`${t} is not a valid third argument to meshgrid`);if(r===void 0)return[];let a=s.convertToTensor(r,"x","meshgrid",r instanceof s.Tensor?r.dtype:"float32");if(e===void 0)return[a];let n=s.convertToTensor(e,"y","meshgrid",e instanceof s.Tensor?e.dtype:"float32");const i=s.sizeFromShape(a.shape),l=s.sizeFromShape(n.shape);return t==="xy"?(a=s.reshape(a,[1,-1]),n=s.reshape(n,[-1,1]),[s.matMul(s.ones([l,1],a.dtype),a),s.matMul(n,s.ones([1,i],n.dtype))]):(a=s.reshape(a,[-1,1]),n=s.reshape(n,[1,-1]),[s.matMul(a,s.ones([1,l],a.dtype)),s.matMul(s.ones([i,1],n.dtype),n)])}function cr(r,e,t,a){const n=s.convertToTensor(e,"data","multiRNNCell"),i=s.convertToTensorArray(t,"c","multiRNNCell"),l=s.convertToTensorArray(a,"h","multiRNNCell");let u=n;const p=[];for(let d=0;d<r.length;d++){const h=r[d](u,i[d],l[d]);p.push(h[0]),p.push(h[1]),u=h[1]}const c=[],m=[];for(let d=0;d<p.length;d+=2)c.push(p[d]),m.push(p[d+1]);return[c,m]}const Qe=s.op({multiRNNCell_:cr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mr(r,e,t,a=!1){const n=s.convertToTensor(r,"logits","multinomial"),i=n.size,l=n.rank;if(i<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${i}.`);if(l>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${l}`);t=t||Math.random();const p={logits:l===1?s.reshape(n,[1,-1]):n},c={numSamples:e,seed:t,normalized:a},m=s.ENGINE.runKernel(s.Multinomial,p,c);return l===1?s.reshape(m,[m.size]):m}const Xe=s.op({multinomial_:mr});function dr(r,e){const t=s.convertToTensor(r,"v1","outerProduct"),a=s.convertToTensor(e,"v2","outerProduct");s.assert(t.rank===1&&a.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${t.rank} and ${a.rank}.`);const n=s.reshape(t,[-1,1]),i=s.reshape(a,[1,-1]);return s.matMul(n,i)}const Ze=s.op({outerProduct_:dr});function hr(r,e,t=0){return s.assert(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),s.pad(r,[e],t)}const Ye=s.op({pad1d_:hr});function fr(r,e,t=0){return s.assert(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),s.pad(r,e,t)}const Me=s.op({pad2d_:fr});function yr(r,e,t=0){return s.assert(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),s.pad(r,e,t)}const et=s.op({pad3d_:yr});function gr(r,e,t=0){return s.assert(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),s.pad(r,e,t)}const tt=s.op({pad4d_:gr});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nr(r,e,t,a){const n=r.map((m,d)=>s.convertToTensor(m,`tensors${d}`,"raggedGather","int32")),i=s.convertToTensor(e,"paramsDenseValues","raggedGather"),l=s.convertToTensor(t,"indices","raggedGather","int32"),u={paramsNestedSplits:n,paramsDenseValues:i,indices:l},p={outputRaggedRank:a},c=s.ENGINE.runKernel(s.RaggedGather,u,p);return{outputNestedSplits:c.slice(0,c.length-1),outputDenseValues:c[c.length-1]}}const rt=s.op({raggedGather_:Nr});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function br(r,e,t){const a=s.convertToTensor(r,"starts","raggedRange"),n=s.convertToTensor(e,"limits","raggedRange",a.dtype),i=s.convertToTensor(t,"deltas","raggedRange",a.dtype),l={starts:a,limits:n,deltas:i},u=s.ENGINE.runKernel(s.RaggedRange,l);return{rtNestedSplits:u[0],rtDenseValues:u[1]}}const st=s.op({raggedRange_:br});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tr(r,e,t,a,n){const i=s.convertToTensor(r,"shape","raggedTensorToTensor","int32"),l=s.convertToTensor(e,"values","raggedTensorToTensor"),u=s.convertToTensor(t,"defaultValue","raggedTensorToTensor",l.dtype),p=a.map((d,h)=>s.convertToTensor(d,`tensors${h}`,"raggedTensorToTensor","int32")),c={shape:i,values:l,defaultValue:u,rowPartitionTensors:p},m={rowPartitionTypes:n};return s.ENGINE.runKernel(s.RaggedTensorToTensor,c,m)}const at=s.op({raggedTensorToTensor_:Tr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wr(r,e,t){s.assertNonNegativeIntegerDimensions(r);const a=s.sizeFromShape(r);let n=null;if(t==null||t==="float32")n=new Float32Array(a);else if(t==="int32")n=new Int32Array(a);else if(t==="bool")n=new Uint8Array(a);else throw new Error(`Unknown data type ${t}`);for(let i=0;i<a;i++)n[i]=e();return s.ENGINE.makeTensor(n,r,t)}const nt=s.op({rand_:wr});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sr=.001,it=.1;function vr(r,e,t){return t==null&&(t=de()),te(r,e,(a,n)=>he(a,n,t))}function de(){return s.ENGINE.backend.floatPrecision()===32?Sr:it}function te(r,e,t){let a=!0;if((s.isTypedArray(r)||s.isTypedArray(e))&&(a=!1),s.isTypedArray(r)&&s.isTypedArray(e)&&(a=!0),a){const l=r.constructor.name,u=e.constructor.name;if(l!==u)throw new Error(`Arrays are of different type. Actual: ${l}. Expected: ${u}`)}if(Array.isArray(r)&&Array.isArray(e)){const l=s.inferShape(r),u=s.inferShape(e);if(!s.arraysEqual(l,u))throw new Error(`Arrays have different shapes. Actual: [${l}]. Expected: [${u}]`)}const n=s.isTypedArray(r)?r:s.flatten(r),i=s.isTypedArray(e)?e:s.flatten(e);if(n.length!==i.length)throw new Error(`Arrays have different lengths actual: ${n.length} vs expected: ${i.length}.
Actual:   ${n}.
Expected: ${i}.`);for(let l=0;l<i.length;++l){const u=n[l],p=i[l];if(!t(u,p))throw new Error(`Arrays differ: actual[${l}] = ${u}, expected[${l}] = ${p}.
Actual:   ${n}.
Expected: ${i}.`)}typeof expect<"u"&&expect().nothing()}function Or(r,e){r().then(()=>e.fail(),()=>e()),typeof expect<"u"&&expect().nothing()}function Er(r,e){const t=typeof e=="string"||typeof e=="number"||typeof e=="boolean"?[e]:e;return s.isString(r)||s.isString(r[0])||s.isString(e)||s.isString(e[0])?te(r,t,(a,n)=>a==n):te(r,e,(a,n)=>he(a,n,0))}function Ar(r,e,t){if(t==null&&(t=de()),!he(r,e,t))throw new Error(`Numbers differ: actual === ${r}, expected === ${e}`);typeof expect<"u"&&expect().nothing()}function he(r,e,t){return!isFinite(r)&&!isFinite(e)?!0:!(isNaN(r)||isNaN(e)||Math.abs(r-e)>t)}function _r(r,e,t){for(let a=0;a<r.length;a++)if(r[a]<e||r[a]>t)throw new Error(`Value out of range:${r[a]} low: ${e}, high: ${t}`)}function kr(r,e){const t=new Float32Array(r),a=new Float32Array(e);if(t.length!==a.length)throw new Error(`Expected ArrayBuffer to be of length ${a.length}, but it was ${t.length}`);for(let n=0;n<a.length;n++)if(t[n]!==a[n])throw new Error(`Expected ArrayBuffer value at ${n} to be ${a[n]} but got ${t[n]} instead`)}function ot(r){for(let e=0;e<r.length;e++){const t=r[e];Array.isArray(t)?ot(t):r[e]=s.encodeString(t)}return r}function Ir(r){const e=document.createElement("video");return"playsInline"in e&&(e.playsInline=!0),e.muted=!0,e.loop=!0,e.style.position="fixed",e.style.left="0px",e.style.top="0px",e.preload="auto",e.appendChild(r),new Promise(t=>{e.addEventListener("loadeddata",a=>t(e)),e.load()})}async function Dr(r){await r.play(),"requestVideoFrameCallback"in r&&await new Promise(e=>{r.requestVideoFrameCallback(e)})}const Cr=Object.freeze(Object.defineProperty({__proto__:null,TEST_EPSILON_FLOAT16:it,createVideoElement:Ir,encodeStrings:ot,expectArrayBuffersEqual:kr,expectArraysClose:vr,expectArraysEqual:Er,expectNumbersClose:Ar,expectPromiseToFail:Or,expectValuesInRange:_r,play:Dr,testEpsilon:de},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $r(r,e,t=1,a="float32",n){if(s.assertNonNegativeIntegerDimensions(r),t==null&&(t=1),a==null&&(a="float32"),a!=="float32"&&a!=="int32")throw new Error(`Unsupported data type ${a}`);const i=new s.RandGamma(e,t,a,n),l=s.buffer(r,a);for(let u=0;u<l.values.length;u++)l.values[u]=i.nextValue();return l.toTensor()}const ut=s.op({randomGamma_:$r});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xr(r,e,t){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return s.randomNormal(r,0,1,e,t)}const lt=s.op({randomStandardNormal_:xr});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zr(r,e,t,a){return s.randomUniform(r,e,t,"int32",a)}const pt=s.op({randomUniformInt_:zr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lr(r){const e=s.convertToTensor(r,"x","reverse");return s.assert(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),s.reverse(e,0)}const ct=s.op({reverse1d_:Lr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pr(r,e){const t=s.convertToTensor(r,"x","reverse");return s.assert(t.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${t.rank}.`),s.reverse(t,e)}const mt=s.op({reverse2d_:Pr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fr(r,e){const t=s.convertToTensor(r,"x","reverse");return s.assert(t.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${t.rank}.`),s.reverse(t,e)}const dt=s.op({reverse3d_:Fr});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rr(r,e){const t=s.convertToTensor(r,"x","reverse");return s.assert(t.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${t.rank}.`),s.reverse(t,e)}const ht=s.op({reverse4d_:Rr});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Vr(r,e){const t=s.convertToTensor(r,"x","setdiff1d"),a=s.convertToTensor(e,"y","setdiff1d");s.assert(t.dtype===a.dtype,()=>`x and y should have the same dtype, but got x (${t.dtype}) and y (${a.dtype}).`),s.assert(t.rank===1,()=>`x should be 1D tensor, but got x (${t.shape}).`),s.assert(a.rank===1,()=>`y should be 1D tensor, but got y (${a.shape}).`);const n=await t.data(),i=await a.data(),l=new Set(i);let u=0;for(let m=0;m<n.length;m++)l.has(n[m])||u++;const p=new s.TensorBuffer([u],t.dtype),c=new s.TensorBuffer([u],"int32");for(let m=0,d=0;m<n.length;m++)l.has(n[m])||(p.values[d]=n[m],c.values[d]=m,d++);return[p.toTensor(),c.toTensor()]}const ft=Vr;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yt(r,e,t){if(s.assertNonNull(r),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const a=s.inferShape(r,t);if(a.length!==4&&a.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return s.makeTensor(r,e,a,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gt(r,e,t){if(s.assertNonNull(r),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const a=s.inferShape(r,t);if(a.length!==5&&a.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return s.makeTensor(r,e,a,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nt(r,e,t){if(s.assertNonNull(r),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const a=s.inferShape(r,t);if(a.length!==6&&a.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||a,s.makeTensor(r,e,a,t)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Br(r,e,t){const a=s.convertToTensor(r,"tensor","tensorScatterupdate"),n=s.convertToTensor(e,"indices","tensorScatterupdate","int32"),i=s.convertToTensor(t,"updates","tensorScatterupdate");if(s.validateInput(i,n,a.shape),a.dtype!==i.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${a.dtype} and ${i.dtype}.`);const l={tensor:a,indices:n,updates:i},u={};return s.ENGINE.runKernel(s.TensorScatterUpdate,l,u)}const bt=s.op({tensorScatterUpdate_:Br});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tt(r,e){return X(r,e,"right")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function jr(r){const e=s.convertToTensor(r,"condition","whereAsync","bool"),t=await e.data(),a=s.whereImpl(e.shape,t);return r!==e&&e.dispose(),a}const fe=jr;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function qr(r,e,t){const a=s.convertToTensor(r,"tensor","boolMask"),n=s.convertToTensor(e,"mask","boolMask","bool"),i=t??0,l=n.rank,u=a.shape;s.assert(l>0,()=>"mask cannot be scalar"),s.assertShapesMatch(u.slice(i,i+l),n.shape,"mask's shape must match the first K dimensions of tensor's shape,");let p=1;for(let g=i;g<i+l;g++)p*=u[g];const c=u.slice(0,i).concat([p],u.slice(i+l)),m=s.reshape(a,c),d=s.reshape(n,[-1]),h=await fe(d),f=s.squeeze(h,[1]),b=s.gather(m,f,i);return r!==a&&a.dispose(),e!==n&&n.dispose(),f.dispose(),m.dispose(),d.dispose(),h.dispose(),b}const wt=qr;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hr(r,e,t,a,n=!0){const i=s.convertToTensor(r,"v","movingAverage"),l=s.convertToTensor(e,"x","movingAverage"),u=s.convertToTensor(t,"decay","movingAverage");s.assertTypesMatch(i,l),s.assert(s.arraysEqual(i.shape,l.shape),()=>"Shape mismatch in v and x");const p=s.scalar(1),c=s.sub(p,u);let m=s.mul(s.sub(l,i),c);if(n){s.assert(a!=null,()=>"When using zeroDebias: true, step is required.");const d=s.convertToTensor(a,"step","movingAverage");m=s.div(m,s.sub(p,s.pow(u,d)))}return s.add(i,m)}const St=s.op({movingAverage_:Hr});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gr(r,e,t){s.assertNonNegativeIntegerDimensions(t);const a=s.convertToTensor(r,"indices","scatterND","int32"),n=s.convertToTensor(e,"updates","scatterND");s.validateInput(n,a,t);const i={indices:a,updates:n},l={shape:t};return s.ENGINE.runKernel(s.ScatterNd,i,l)}const vt=s.op({scatterND_:Gr});function Wr(r,e,t,a){if(r.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${r.dtype}.`);if(r.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${r.shape}.`);const n=r.rank>0?r.shape[0]:1,i=r.rank>1?r.shape[1]:1;if(t.length!==i)throw new Error(`outputShape has incorrect number of elements:, ${t.length}, should be: ${i}.`);const l=e.size;if(!(e.rank===0||e.rank===1&&l===n))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${n}]`);if(e.dtype!==a.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ur(r,e,t,a=0){s.assertNonNegativeIntegerDimensions(t);const n=s.convertToTensor(r,"sparseIndices","sparseToDense","int32"),i=s.convertToTensor(e,"sparseValues","sparseToDense","string_or_numeric"),l=s.convertToTensor(a,"defaultValue","sparseToDense",i.dtype);Wr(n,i,t,l);const u={sparseIndices:n,sparseValues:i,defaultValue:l},p={outputShape:t};return s.ENGINE.runKernel(s.SparseToDense,u,p)}const Ot=s.op({sparseToDense_:Ur});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kr(r,e){const t=s.convertToTensor(e,"indices","gatherND","int32"),n={params:s.convertToTensor(r,"x","gatherND","string_or_numeric"),indices:t};return s.ENGINE.runKernel(s.GatherNd,n)}const Et=s.op({gatherND_:Kr});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Jr(r,e,t=1){const a=s.convertToTensor(r,"predictions","inTopK"),n=s.convertToTensor(e,"targets","inTopK");s.assert(a.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${a.rank}`),s.assert(a.rank-1===n.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${a.rank} and targets rank ${n.rank}`),s.assertShapesMatch(a.shape.slice(0,a.shape.length-1),n.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const i=a.shape[a.shape.length-1];s.assert(t>0&&t<=i,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${i}), but got ${t}`);const l=await a.data(),u=await n.data(),[p,c]=[l.length/i,i],m=s.getTypedArrayFromDType("bool",p);for(let d=0;d<p;d++){const h=d*c,f=l.subarray(h,h+c),b=[];for(let g=0;g<f.length;g++)b.push({value:f[g],index:g});b.sort((g,y)=>y.value-g.value),m[d]=0;for(let g=0;g<t;g++)if(b[g].index===u[d]){m[d]=1;break}}return r!==a&&a.dispose(),e!==n&&n.dispose(),s.tensor(m,n.shape,"bool")}const At=Jr;/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qr({x:r,filter:e,strides:t,pad:a,dataFormat:n="NHWC",dilations:i=[1,1],dimRoundingMode:l,bias:u,activation:p="linear",preluActivationWeights:c,leakyreluAlpha:m}){if(s.shouldFuse(s.ENGINE.state.gradientDepth,p)===!1){let A=s.depthwiseConv2d(r,e,t,a,n,i,l);return u!=null&&(A=s.add(A,u)),s.applyActivation(A,p,c,m)}const d=s.convertToTensor(r,"x","depthwiseConv2d","float32"),h=s.convertToTensor(e,"filter","depthwiseConv2d","float32");let f=d,b=!1;d.rank===3&&(b=!0,f=s.reshape(d,[1,d.shape[0],d.shape[1],d.shape[2]])),s.assert(f.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${f.rank}.`),s.assert(h.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${h.rank}.`),s.assert(f.shape[3]===h.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${f.shape[3]}) must match the inChannels dimension in filter ${h.shape[2]}.`),i==null&&(i=[1,1]),s.assert(s.eitherStridesOrDilationsAreOne(t,i),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),s.checkPadOnDimRoundingMode("fused depthwiseConv2d",a,l);const g=s.computeConv2DInfo(f.shape,h.shape,t,i,a,l,!0);let y;u!=null&&(y=s.convertToTensor(u,"bias","fused conv2d"),[y]=s.makeTypesMatch(y,d),s.assertAndGetBroadcastShape(g.outShape,y.shape));let N;c!=null&&(N=s.convertToTensor(c,"prelu weights","fused depthwiseConv2d"));const w=(A,k)=>{s.assert(s.tupleValuesAreOne(i),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${i}'`);const[I,R,C,$]=k,Z=s.getFusedDyActivation(A,C,p),Ee=s.depthwiseConv2dNativeBackpropInput(R.shape,Z,I,t,a,i,l),Ae=s.depthwiseConv2dNativeBackpropFilter(R,Z,I.shape,t,a,i,l);if($!=null){const er=s.getFusedBiasGradient(y,Z);return[Ee,Ae,er]}return[Ee,Ae]},S={x:f,filter:h,bias:y,preluActivationWeights:N},T={strides:t,pad:a,dataFormat:n,dilations:i,dimRoundingMode:l,activation:p,leakyreluAlpha:m};return u==null?s.customGrad((k,I,R)=>{let C=s.ENGINE.runKernel(s.FusedDepthwiseConv2D,S,T);return R([I,k,C]),b&&(C=s.reshape(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:w}})(f,h):s.customGrad((k,I,R,C)=>{let $=s.ENGINE.runKernel(s.FusedDepthwiseConv2D,S,T);return C([I,k,$,R]),b&&($=s.reshape($,[$.shape[1],$.shape[2],$.shape[3]])),{value:$,gradFunc:w}})(f,h,y)}const Xr=s.op({fusedDepthwiseConv2d_:Qr});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _t=Object.freeze(Object.defineProperty({__proto__:null,conv2d:s.conv2d,depthwiseConv2d:Xr,matMul:s.matMul$1},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zr="model",Yr=".json",Mr=".weights.bin";function _e(r){return new Promise(e=>setTimeout(e)).then(r)}class P{constructor(e){if(!s.env().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(P.URL_SCHEME)&&(e=e.slice(P.URL_SCHEME.length)),(e==null||e.length===0)&&(e=Zr),this.modelJsonFileName=e+Yr,this.weightDataFileName=e+Mr}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=s.CompositeArrayBuffer.join(e.weightData),a=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const n=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],i=s.getModelJSONForModelArtifacts(e,n),l=window.URL.createObjectURL(new Blob([JSON.stringify(i)],{type:"application/json"})),u=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(u.download=this.modelJsonFileName,u.href=l,await _e(()=>u.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const p=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;p.download=this.weightDataFileName,p.href=a,await _e(()=>p.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:s.getModelArtifactsInfoForJSON(e)}}}}P.URL_SCHEME="downloads://";class es{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const a=new FileReader;a.onload=n=>{const i=JSON.parse(n.target.result),l=i.modelTopology;if(l==null){t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(i.weightsManifest==null){t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:l});return}const p=s.getModelArtifactsForJSON(i,c=>this.loadWeights(c));e(p)},a.onerror=n=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),a.readAsText(this.jsonFile)})}loadWeights(e){const t=[],a=[];for(const l of e)t.push(...l.weights),a.push(...l.paths);const n=this.checkManifestAndWeightFiles(e),i=a.map(l=>this.loadWeightsFile(l,n[l]));return Promise.all(i).then(l=>[t,l])}loadWeightsFile(e,t){return new Promise((a,n)=>{const i=new FileReader;i.onload=l=>{const u=l.target.result;a(u)},i.onerror=l=>n(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],a=this.weightsFiles.map(i=>s.basename(i.name)),n={};for(const i of e)i.paths.forEach(l=>{const u=s.basename(l);if(t.indexOf(u)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${u}'`);if(t.push(u),a.indexOf(u)===-1)throw new Error(`Weight file with basename '${u}' is not provided.`);n[l]=this.weightsFiles[a.indexOf(u)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return n}}const ts=r=>s.env().getBool("IS_BROWSER")&&!Array.isArray(r)&&r.startsWith(P.URL_SCHEME)?rs(r.slice(P.URL_SCHEME.length)):null;s.IORouterRegistry.registerSaveRouter(ts);function rs(r="model"){return new P(r)}function ss(r){return new es(r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Y{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class kt{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class as{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function ns(r,e,t,a){const n=arguments;return new as(K(...n))}function K(r,e,t,a){return arguments.length===1?r.modelTopology!=null||r.weightSpecs!=null?new Y(r):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Y({modelTopology:r})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Y({modelTopology:r,weightSpecs:e,weightData:t,trainingConfig:a}))}function is(r){return new kt(r)}function os(r){return new kt(r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ye=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:s.CompositeArrayBuffer,browserFiles:ss,browserHTTPRequest:s.browserHTTPRequest,concatenateArrayBuffers:s.concatenateArrayBuffers,copyModel:s.copyModel,decodeWeights:s.decodeWeights,encodeWeights:s.encodeWeights,fromMemory:ns,fromMemorySync:K,getLoadHandlers:s.getLoadHandlers,getModelArtifactsForJSON:s.getModelArtifactsForJSON,getModelArtifactsForJSONSync:s.getModelArtifactsForJSONSync,getModelArtifactsInfoForJSON:s.getModelArtifactsInfoForJSON,getSaveHandlers:s.getSaveHandlers,getWeightSpecs:s.getWeightSpecs,http:s.http,isHTTPScheme:s.isHTTPScheme,listModels:s.listModels,loadWeights:s.loadWeights,moveModel:s.moveModel,registerLoadRouter:s.registerLoadRouter,registerSaveRouter:s.registerSaveRouter,removeModel:s.removeModel,weightsLoaderFactory:s.weightsLoaderFactory,withSaveHandler:is,withSaveHandlerSync:os},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function us(r,e,t){const a=s.convertToTensor(r,"labels","confusionMatrix"),n=s.convertToTensor(e,"predictions","confusionMatrix");s.assert(t==null||t>0&&Number.isInteger(t),()=>`If provided, numClasses must be a positive integer, but got ${t}`),s.assert(a.rank===1,()=>`Expected the rank of labels to be 1, but got ${a.rank}`),s.assert(n.rank===1,()=>`Expected the rank of predictions to be 1, but got ${n.rank}`),s.assert(a.shape[0]===n.shape[0],()=>`Mismatch in the number of examples: ${a.shape[0]} vs. ${n.shape[0]}. Labels and predictions should have the same number of elements.`),s.assert(t>0&&Number.isInteger(t),()=>`numClasses is required to be a positive integer, but got ${t}`);const i=s.oneHot(s.cast(a,"int32"),t),l=s.oneHot(s.cast(n,"int32"),t),u=s.transpose(i),p=s.matMul(u,l);return s.cast(p,"int32")}const ls=s.op({confusionMatrix_:us});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ps=Object.freeze(Object.defineProperty({__proto__:null,confusionMatrix:ls},Symbol.toStringTag,{value:"Module"}));/** @license See the LICENSE file. */const It="4.10.0";/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cs=Object.freeze(Object.defineProperty({__proto__:null,nonMaxSuppressionV3Impl:s.nonMaxSuppressionV3Impl,nonMaxSuppressionV4Impl:s.nonMaxSuppressionV4Impl,nonMaxSuppressionV5Impl:s.nonMaxSuppressionV5Impl,whereImpl:s.whereImpl},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function ms(r){return new s.MaxNorm(r)}function ds(r){return new s.UnitNorm(r)}function hs(){return new s.NonNeg}function fs(r){return new s.MinMaxNorm(r)}const ys=Object.freeze(Object.defineProperty({__proto__:null,maxNorm:ms,minMaxNorm:fs,nonNeg:hs,unitNorm:ds},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function gs(){return new s.Zeros}function Ns(){return new s.Ones}function bs(r){return new s.Constant(r)}function Ts(r){return new s.RandomUniform(r)}function ws(r){return new s.RandomNormal(r)}function Ss(r){return new s.TruncatedNormal(r)}function vs(r){return new s.Identity(r)}function Os(r){return new s.VarianceScaling(r)}function Es(r){return new s.GlorotUniform(r)}function As(r){return new s.GlorotNormal(r)}function _s(r){return new s.HeNormal(r)}function ks(r){return new s.HeUniform(r)}function Is(r){return new s.LeCunNormal(r)}function Ds(r){return new s.LeCunUniform(r)}function Cs(r){return new s.Orthogonal(r)}const $s=Object.freeze(Object.defineProperty({__proto__:null,constant:bs,glorotNormal:As,glorotUniform:Es,heNormal:_s,heUniform:ks,identity:vs,leCunNormal:Is,leCunUniform:Ds,ones:Ns,orthogonal:Cs,randomNormal:ws,randomUniform:Ts,truncatedNormal:Ss,varianceScaling:Os,zeros:gs},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function xs(r){return new s.LayersModel(r)}function zs(r){return new s.Sequential(r)}function Dt(r){return s.Input(r)}function Ls(r,e){s.CallbackConstructorRegistry.registerCallbackConstructor(r,e)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Ps(r){return new s.InputLayer(r)}function Fs(r){return new s.ELU(r)}function Rs(r){return new s.ReLU(r)}function Vs(r){return new s.LeakyReLU(r)}function Bs(r){return new s.PReLU(r)}function js(r){return new s.Softmax(r)}function qs(r){return new s.ThresholdedReLU(r)}function Hs(r){return new s.Conv1D(r)}function Gs(r){return new s.Conv2D(r)}function Ws(r){return new s.Conv2DTranspose(r)}function Us(r){return new s.Conv3D(r)}function Ks(r){return new s.Conv3DTranspose(r)}function Js(r){return new s.SeparableConv2D(r)}function Qs(r){return new s.Cropping2D(r)}function Xs(r){return new s.UpSampling2D(r)}function Zs(r){return new s.DepthwiseConv2D(r)}function Ys(r){return new s.Activation(r)}function Ms(r){return new s.Dense(r)}function ea(r){return new s.Dropout(r)}function ta(r){return new s.SpatialDropout1D(r)}function ra(r){return new s.Flatten(r)}function sa(r){return new s.RepeatVector(r)}function aa(r){return new s.Reshape(r)}function na(r){return new s.Permute(r)}function ia(r){return new s.Embedding(r)}function oa(r){return new s.Add(r)}function ua(r){return new s.Average(r)}function la(r){return new s.Concatenate(r)}function pa(r){return new s.Maximum(r)}function ca(r){return new s.Minimum(r)}function ma(r){return new s.Multiply(r)}function da(r){return new s.Dot(r)}function ha(r){return new s.BatchNormalization(r)}function fa(r){return new s.LayerNormalization(r)}function ya(r){return new s.ZeroPadding2D(r)}function ge(r){return new s.AveragePooling1D(r)}function ga(r){return ge(r)}function Na(r){return ge(r)}function Ne(r){return new s.AveragePooling2D(r)}function ba(r){return Ne(r)}function Ta(r){return Ne(r)}function be(r){return new s.AveragePooling3D(r)}function wa(r){return be(r)}function Sa(r){return be(r)}function va(r){return new s.GlobalAveragePooling1D(r)}function Oa(r){return new s.GlobalAveragePooling2D(r)}function Ct(r){return new s.GlobalMaxPooling1D(r)}function $t(r){return new s.GlobalMaxPooling2D(r)}function xt(r){return new s.MaxPooling1D(r)}function zt(r){return new s.MaxPooling2D(r)}function Ea(r){return new s.MaxPooling3D(r)}function Aa(r){return new s.GRU(r)}function _a(r){return new s.GRUCell(r)}function ka(r){return new s.LSTM(r)}function Ia(r){return new s.LSTMCell(r)}function Da(r){return new s.SimpleRNN(r)}function Ca(r){return new s.SimpleRNNCell(r)}function $a(r){return new s.ConvLSTM2D(r)}function xa(r){return new s.ConvLSTM2DCell(r)}function za(r){return new s.RNN(r)}function La(r){return new s.StackedRNNCells(r)}function Pa(r){return new s.Bidirectional(r)}function Fa(r){return new s.TimeDistributed(r)}const Ra=Ct,Va=$t,Ba=xt,ja=zt;function qa(r){return new s.GaussianNoise(r)}function Ha(r){return new s.GaussianDropout(r)}function Ga(r){return new s.AlphaDropout(r)}function Wa(r){return new s.Masking(r)}function Ua(r){return new s.Rescaling(r)}function Ka(r){return new s.CenterCrop(r)}function Ja(r){return new s.Resizing(r)}function Qa(r){return new s.CategoryEncoding(r)}function Xa(r){return new s.RandomWidth(r)}const Za=Object.freeze(Object.defineProperty({__proto__:null,Layer:s.Layer,RNN:s.RNN,RNNCell:s.RNNCell,activation:Ys,add:oa,alphaDropout:Ga,average:ua,averagePooling1d:ge,averagePooling2d:Ne,averagePooling3d:be,avgPool1d:ga,avgPool2d:ba,avgPool3d:wa,avgPooling1d:Na,avgPooling2d:Ta,avgPooling3d:Sa,batchNormalization:ha,bidirectional:Pa,categoryEncoding:Qa,centerCrop:Ka,concatenate:la,conv1d:Hs,conv2d:Gs,conv2dTranspose:Ws,conv3d:Us,conv3dTranspose:Ks,convLstm2d:$a,convLstm2dCell:xa,cropping2D:Qs,dense:Ms,depthwiseConv2d:Zs,dot:da,dropout:ea,elu:Fs,embedding:ia,flatten:ra,gaussianDropout:Ha,gaussianNoise:qa,globalAveragePooling1d:va,globalAveragePooling2d:Oa,globalMaxPool1d:Ra,globalMaxPool2d:Va,globalMaxPooling1d:Ct,globalMaxPooling2d:$t,gru:Aa,gruCell:_a,input:Dt,inputLayer:Ps,layerNormalization:fa,leakyReLU:Vs,lstm:ka,lstmCell:Ia,masking:Wa,maxPool1d:Ba,maxPool2d:ja,maxPooling1d:xt,maxPooling2d:zt,maxPooling3d:Ea,maximum:pa,minimum:ca,multiply:ma,permute:na,prelu:Bs,randomWidth:Xa,reLU:Rs,repeatVector:sa,rescaling:Ua,reshape:aa,resizing:Ja,rnn:za,separableConv2d:Js,simpleRNN:Da,simpleRNNCell:Ca,softmax:js,spatialDropout1d:ta,stackedRNNCells:La,thresholdedReLU:qs,timeDistributed:Fa,upSampling2d:Xs,zeroPadding2d:ya},Symbol.toStringTag,{value:"Module"}));function Ya(r,e){return s.binaryAccuracy(r,e)}function Ma(r,e){return s.binaryCrossentropy(r,e)}function en(r,e){return s.sparseCategoricalAccuracy(r,e)}function tn(r,e){return s.categoricalAccuracy(r,e)}function rn(r,e){return s.categoricalCrossentropy(r,e)}function sn(r,e){return s.precision(r,e)}function an(r,e){return s.recall(r,e)}function nn(r,e){return s.cosineProximity(r,e)}function on(r,e){return s.meanAbsoluteError(r,e)}function un(r,e){return s.meanAbsolutePercentageError(r,e)}function ln(r,e){return s.meanAbsolutePercentageError(r,e)}function pn(r,e){return s.meanAbsolutePercentageError(r,e)}function cn(r,e){return s.meanSquaredError(r,e)}function mn(r,e){return s.meanSquaredError(r,e)}function dn(r,e){return s.meanSquaredError(r,e)}const hn=Object.freeze(Object.defineProperty({__proto__:null,MAPE:ln,MSE:mn,binaryAccuracy:Ya,binaryCrossentropy:Ma,categoricalAccuracy:tn,categoricalCrossentropy:rn,cosineProximity:nn,mape:pn,meanAbsoluteError:on,meanAbsolutePercentageError:un,meanSquaredError:cn,mse:dn,precision:sn,recall:an,sparseCategoricalAccuracy:en},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const fn=Object.freeze(Object.defineProperty({__proto__:null,modelFromJSON:s.modelFromJSON},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function yn(r){return new s.L1L2(r)}function gn(r){return s.l1(r)}function Nn(r){return s.l2(r)}const bn=Object.freeze(Object.defineProperty({__proto__:null,l1:gn,l1l2:yn,l2:Nn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Lt extends s.BaseCallback{constructor(){super(...arguments),this.model=null}setModel(e){if(!(e instanceof s.LayersModel))throw new Error("model must be a LayersModel, not some other Container");this.model=e}}function q(r,e){return r<e}function ke(r,e){return r>e}class Pt extends Lt{constructor(e){if(super(),e==null&&(e={}),e.restoreBestWeights)throw new s.NotImplementedError("restoreBestWeights = True is not implemented in EarlyStopping yet.");this.monitor=e.monitor||"val_loss",this.minDelta=Math.abs(e.minDelta||0),this.patience=e.patience||0,this.verbose=e.verbose||0,this.mode=e.mode||"auto",this.baseline=e.baseline,["auto","min","max"].indexOf(this.mode)===-1&&(console.warn(`EarlyStopping mode '${this.mode}' is invalid. Falling back to mode 'auto'.`),this.mode="auto"),this.mode==="min"?this.monitorFunc=q:this.mode==="max"?this.monitorFunc=ke:this.monitor.indexOf("acc")!==-1?this.monitorFunc=ke:this.monitorFunc=q,this.monitorFunc===q&&(this.minDelta*=-1)}async onTrainBegin(e){this.wait=0,this.stoppedEpoch=0,this.baseline!=null?this.best=this.baseline:this.best=this.monitorFunc===q?1/0:-1/0}async onEpochEnd(e,t){await s.resolveScalarsInLogs(t);const a=this.getMonitorValue(t);a!=null&&(this.monitorFunc(a-this.minDelta,this.best)?(this.best=a,this.wait=0):(this.wait++,this.wait>=this.patience&&(this.stoppedEpoch=e,this.model.stopTraining=!0)))}async onTrainEnd(e){this.stoppedEpoch>0&&this.verbose&&console.log(`Epoch ${this.stoppedEpoch}: early stopping.`)}getMonitorValue(e){e==null&&(e={});const t=e[this.monitor];return t==null&&console.warn(`Metric for EarlyStopping ${this.monitor} is not available. Available metrics are: ${Object.keys(e)}`),t}}function Tn(r){return new Pt(r)}const wn={earlyStopping:Tn};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Te={};function Sn(r,e){const t={tfOpName:r,category:"custom",inputs:[],attrs:[],customExecutor:e};Te[r]=t}function Ft(r){return Te[r]}function vn(r){delete Te[r]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function o(r,e,t,a,n){const i=e.inputParams[r];if(i&&i.inputIndexStart!==void 0){const u=i.inputIndexStart,p=i.inputIndexEnd===0?void 0:i.inputIndexEnd===void 0?u+1:i.inputIndexEnd,c=u<0?e.inputNames.length+u:u;if(i.type==="tensor")return v(e.inputNames[c],t,a,n);if(i.type==="tensors"){const h=e.inputs.slice(u,p);return e.inputNames.slice(u,p).filter((b,g)=>{var y;return((y=h[g])===null||y===void 0?void 0:y.op)!=="NoOp"}).map(b=>v(b,t,a,n))}const m=v(e.inputNames[c],t,a,n),d=m.dataSync();return i.type==="number"?d[0]:s.toNestedArray(m.shape,d)}const l=e.attrParams[r];return l&&l.value}function v(r,e,t,a){const[n,i]=_(r,t);if(a!=null){const u=a.getHashTableHandleByName(n);if(u!=null)return u}const l=t.currentContextIds.find(u=>!!e[J(n,u)]);return l!==void 0?e[J(n,l)][i]:void 0}function Ie(r,e,t){return e[J(r,t.currentContextId)]}function x(r,e){const[t,a,n]=_(r,e);return[J(t,e&&e.currentContextId),a,n]}function J(r,e){return e?`${r}-${e}`:r}function _(r,e){if(r==="")return["",0,void 0];const t=e!=null&&e.parseNodeNameCache!=null;if(t){const i=e.parseNodeNameCache.get(r);if(i!=null)return i}const a=r.split(":");let n;if(a.length===1)n=[r,0,void 0];else{const i=a[0],l=a.length===3?a[1]:void 0,u=Number(a[a.length-1]);n=[i,u,l]}return t&&e.parseNodeNameCache.set(r,n),n}function U(r,e,t){let a=o("pad",r,e,t);if(a==="explicit"){a=o("explicitPaddings",r,e,t);const n=[[0,0],[0,0],[0,0],[0,0]];for(let i=0;i<4;i++)n[i][0]=a[i*2],n[i][1]=a[i*2+1];return n}return a}function z(r){return r.kept?r:s.clone(r)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const On=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],En=Object.freeze(Object.defineProperty({__proto__:null,json:On},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const An=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],_n=Object.freeze(Object.defineProperty({__proto__:null,json:An},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kn=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],In=Object.freeze(Object.defineProperty({__proto__:null,json:kn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dn=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],Cn=Object.freeze(Object.defineProperty({__proto__:null,json:Dn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $n=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],xn=Object.freeze(Object.defineProperty({__proto__:null,json:$n},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zn=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Ln=Object.freeze(Object.defineProperty({__proto__:null,json:zn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pn=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],Fn=Object.freeze(Object.defineProperty({__proto__:null,json:Pn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rn=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],Vn=Object.freeze(Object.defineProperty({__proto__:null,json:Rn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bn=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],jn=Object.freeze(Object.defineProperty({__proto__:null,json:Bn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qn=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Hn=Object.freeze(Object.defineProperty({__proto__:null,json:qn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gn=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],Wn=Object.freeze(Object.defineProperty({__proto__:null,json:Gn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Un=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],Kn=Object.freeze(Object.defineProperty({__proto__:null,json:Un},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jn=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],Qn=Object.freeze(Object.defineProperty({__proto__:null,json:Jn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xn=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],Zn=Object.freeze(Object.defineProperty({__proto__:null,json:Xn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yn=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Mn=Object.freeze(Object.defineProperty({__proto__:null,json:Yn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ei=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],ti=Object.freeze(Object.defineProperty({__proto__:null,json:ei},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ri=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],si=Object.freeze(Object.defineProperty({__proto__:null,json:ri},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ai=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],ni=Object.freeze(Object.defineProperty({__proto__:null,json:ai},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ii=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],oi=Object.freeze(Object.defineProperty({__proto__:null,json:ii},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class De{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[En,_n,In,Cn,xn,Ln,Fn,Vn,jn,Hn,Wn,Kn,Qn,Zn,Mn,ti,si,ni,oi],t=[].concat(...e.map(a=>a.json));this.opMappers=t.reduce((a,n)=>(a[n.tfOpName]=n,a),{})}transformGraph(e,t={}){const a=e.node,n=[],i=[],l=[],u=a.reduce((g,y)=>(g[y.name]=this.mapNode(y),y.op.startsWith("Placeholder")?n.push(g[y.name]):y.op==="Const"?i.push(g[y.name]):(y.input==null||y.input.length===0)&&l.push(g[y.name]),g),{});let p=[];const c=[];let m={},d={};t!=null&&(m=this.mapSignatureEntries(t.inputs),d=this.mapSignatureEntries(t.outputs));const h=Object.keys(u);h.forEach(g=>{const y=u[g];y.inputNames.forEach((N,w)=>{const[S,,T]=x(N),A=u[S];if(A.outputs!=null){const k=A.outputs.indexOf(T);if(k!==-1){const I=`${S}:${k}`;y.inputNames[w]=I}}y.inputs.push(A),A.children.push(y)})}),Object.keys(d).length===0?h.forEach(g=>{const y=u[g];y.children.length===0&&c.push(y)}):Object.keys(d).forEach(g=>{const[y]=x(g),N=u[y];N!=null&&(N.signatureKey=d[g],c.push(N))}),Object.keys(m).length>0?Object.keys(m).forEach(g=>{const[y]=x(g),N=u[y];N&&(N.signatureKey=m[g],p.push(N))}):p=n;let f={};e.library!=null&&e.library.function!=null&&(f=e.library.function.reduce((g,y)=>(g[y.signature.name]=this.mapFunction(y),g),{}));const b={nodes:u,inputs:p,outputs:c,weights:i,placeholders:n,signature:t,functions:f};return l.length>0&&(b.initNodes=l),b}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,a)=>(t[e[a].name]=a,t),{})}mapNode(e){const t=Ft(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const a={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(n=>n.startsWith("^")?n.slice(1):n),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(a.inputParams=t.inputs.reduce((n,i)=>(n[i.name]={type:i.type,inputIndexStart:i.start,inputIndexEnd:i.end},n),{})),t.attrs!=null&&(a.attrParams=t.attrs.reduce((n,i)=>{const l=i.type;let u;switch(i.type){case"string":u=re(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=re(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"string[]":u=le(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=le(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number":u=ae(e.attr,i.tfName,i.defaultValue||0),u===void 0&&i.tfDeprecatedName&&(u=ae(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number[]":u=ue(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=ue(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool":u=se(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=se(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool[]":u=ce(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=ce(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape":u=oe(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=oe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape[]":u=pe(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=pe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype":u=ne(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=ne(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype[]":u=ie(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=ie(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"func":u=Ce(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Ce(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${i.type} for op: ${e.op}`)}return n[i.name]={value:u,type:l},n},{})),a}mapFunction(e){const t=e.nodeDef,a=[],n=[];let i={};t!=null&&(i=t.reduce((d,h)=>(d[h.name]=this.mapNode(h),h.op==="Const"&&n.push(d[h.name]),d),{}));const l=[],u=[];e.signature.inputArg.forEach(d=>{const[h]=x(d.name),f={name:h,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:we(d.type),type:"dtype"}},children:[]};f.signatureKey=d.name,l.push(f),i[h]=f}),Object.keys(i).forEach(d=>{const h=i[d];h.inputNames.forEach((f,b)=>{const[g,,y]=x(f),N=i[g];if(N.outputs!=null){const w=N.outputs.indexOf(y);if(w!==-1){const S=`${g}:${w}`;h.inputNames[b]=S}}h.inputs.push(N),N.children.push(h)})});const c=e.ret;e.signature.outputArg.forEach(d=>{const[h,f]=x(c[d.name]),b=i[h];b!=null&&(b.defaultOutput=f,u.push(b))});const m=this.mapArgsToSignature(e);return{nodes:i,inputs:l,outputs:u,weights:n,placeholders:a,signature:m}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((t,a)=>(t[a.name]=this.mapArgToTensorInfo(a),t),{}),outputs:e.signature.outputArg.reduce((t,a)=>(t[a.name]=this.mapArgToTensorInfo(a,e.ret),t),{})}}mapArgToTensorInfo(e,t){let a=e.name;return t!=null&&(a=t[a]),{name:a,dtype:e.type}}}function ui(r){const e=s.env().global;if(typeof e.atob<"u")return e.atob(r);if(typeof Buffer<"u")return new Buffer(r,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Rt(r,e){const t=Array.isArray(r)?String.fromCharCode.apply(null,r):ui(r);return e?t:t.toLowerCase()}function re(r,e,t,a=!1){const n=r[e];return n!=null?Rt(n.s,a):t}function se(r,e,t){const a=r[e];return a?a.b:t}function ae(r,e,t){const a=r[e]||{},n=a.i!=null?a.i:a.f!=null?a.f:t;return typeof n=="number"?n:parseInt(n,10)}function we(r){switch(typeof r=="string"&&(r=s.DataType[r]),r){case s.DataType.DT_FLOAT:case s.DataType.DT_HALF:return"float32";case s.DataType.DT_INT32:case s.DataType.DT_INT64:case s.DataType.DT_INT8:case s.DataType.DT_UINT8:return"int32";case s.DataType.DT_BOOL:return"bool";case s.DataType.DT_DOUBLE:return"float32";case s.DataType.DT_STRING:return"string";default:return null}}function Ce(r,e,t){const a=r[e];return a&&a.func?a.func.name:t}function ne(r,e,t){const a=r[e];return a&&a.type?we(a.type):t}function ie(r,e,t){const a=r[e];return a&&a.list&&a.list.type?a.list.type.map(n=>we(n)):t}function Vt(r){if(!r.unknownRank)return r.dim!=null?r.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function oe(r,e,t){const a=r[e];return a&&a.shape?Vt(a.shape):t}function ue(r,e,t){const a=r[e];return a?((a.list.f&&a.list.f.length?a.list.f:a.list.i)||[]).map(n=>typeof n=="number"?n:parseInt(n,10)):t}function le(r,e,t,a=!1){const n=r[e];return n&&n.list&&n.list.s?n.list.s.map(i=>Rt(i,a)):t}function pe(r,e,t){const a=r[e];return a&&a.list&&a.list.shape?a.list.shape.map(n=>Vt(n)):t}function ce(r,e,t){const a=r[e];return a&&a.list&&a.list.b?a.list.b:t}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class li{constructor(e,t,a){this.node=e,this.tensorMap=t,this.context=a,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(n=>this.getInput(n)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((n,i)=>(n[i]=this.getAttr(i),n),{}))}getInput(e){return v(e,this.tensorMap,this.context)}getAttr(e,t){const a=this.node.rawAttrs[e];if(a.tensor!=null)return v(e,this.tensorMap,this.context);if(a.i!=null||a.f!=null)return ae(this.node.rawAttrs,e,t);if(a.s!=null)return re(this.node.rawAttrs,e,t);if(a.b!=null)return se(this.node.rawAttrs,e,t);if(a.shape!=null)return oe(this.node.rawAttrs,e,t);if(a.type!=null)return ne(this.node.rawAttrs,e,t);if(a.list!=null){if(a.list.i!=null||a.list.f!=null)return ue(this.node.rawAttrs,e,t);if(a.list.s!=null)return le(this.node.rawAttrs,e,t);if(a.list.shape!=null)return pe(this.node.rawAttrs,e,t);if(a.list.b!=null)return ce(this.node.rawAttrs,e,t);if(a.list.type!=null)return ie(this.node.rawAttrs,e,t)}return t}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const O=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:s.OP_SCOPE_SUFFIX,abs:s.abs,acos:s.acos,acosh:s.acosh,add:s.add,addN:Ve,all:s.all,any:s.any,argMax:s.argMax,argMin:s.argMin,asin:s.asin,asinh:s.asinh,atan:s.atan,atan2:s.atan2,atanh:s.atanh,avgPool:s.avgPool,avgPool3d:s.avgPool3d,basicLSTMCell:Be,batchNorm:s.batchNorm,batchNorm2d:s.batchNorm2d,batchNorm3d:s.batchNorm3d,batchNorm4d:s.batchNorm4d,batchToSpaceND:s.batchToSpaceND,bincount:s.bincount,bitwiseAnd:je,booleanMaskAsync:wt,broadcastArgs:qe,broadcastTo:s.broadcastTo,buffer:s.buffer,cast:s.cast,ceil:s.ceil,clipByValue:s.clipByValue,clone:s.clone,complex:s.complex,concat:s.concat,concat1d:s.concat1d,concat2d:s.concat2d,concat3d:s.concat3d,concat4d:s.concat4d,conv1d:s.conv1d,conv2d:s.conv2d$1,conv2dTranspose:s.conv2dTranspose,conv3d:s.conv3d,conv3dTranspose:s.conv3dTranspose,cos:s.cos,cosh:s.cosh,cosineWindow:s.cosineWindow,cumprod:s.cumprod,cumsum:s.cumsum,denseBincount:s.denseBincount,depthToSpace:s.depthToSpace,depthwiseConv2d:s.depthwiseConv2d,diag:He,dilation2d:s.dilation2d,div:s.div,divNoNan:s.divNoNan,dot:s.dot,dropout:s.dropout,einsum:s.einsum,elu:s.elu,enclosingPowerOfTwo:s.enclosingPowerOfTwo,ensureShape:Ge,equal:s.equal,erf:s.erf,euclideanNorm:s.euclideanNorm,exp:s.exp,expandDims:s.expandDims,expm1:s.expm1,eye:s.eye,fft:s.fft,fill:s.fill,floor:s.floor,floorDiv:s.floorDiv,fused:_t,gather:s.gather,gatherND:Et,greater:s.greater,greaterEqual:s.greaterEqual,ifft:s.ifft,imag:s.imag,image:s.image,inTopKAsync:At,irfft:s.irfft,isFinite:s.isFinite,isInf:s.isInf,isNaN:s.isNaN,leakyRelu:s.leakyRelu,less:s.less,lessEqual:s.lessEqual,linalg:s.linalg,linspace:We,localResponseNormalization:s.localResponseNormalization,log:s.log,log1p:s.log1p,logSigmoid:s.logSigmoid,logSoftmax:s.logSoftmax,logSumExp:s.logSumExp,logicalAnd:s.logicalAnd,logicalNot:s.logicalNot,logicalOr:s.logicalOr,logicalXor:s.logicalXor,losses:s.losses,lowerBound:Ue,matMul:s.matMul,max:s.max,maxPool:s.maxPool,maxPool3d:s.maxPool3d,maxPoolWithArgmax:Ke,maximum:s.maximum,mean:s.mean,meshgrid:Je,min:s.min,minimum:s.minimum,mirrorPad:s.mirrorPad,mod:s.mod,moments:s.moments,movingAverage:St,mul:s.mul,multiRNNCell:Qe,multinomial:Xe,neg:s.neg,norm:s.norm,notEqual:s.notEqual,oneHot:s.oneHot,ones:s.ones,onesLike:s.onesLike,op:s.op,outerProduct:Ze,pad:s.pad,pad1d:Ye,pad2d:Me,pad3d:et,pad4d:tt,pool:s.pool,pow:s.pow,prelu:s.prelu,print:s.print,prod:s.prod,raggedGather:rt,raggedRange:st,raggedTensorToTensor:at,rand:nt,randomGamma:ut,randomNormal:s.randomNormal,randomStandardNormal:lt,randomUniform:s.randomUniform,randomUniformInt:pt,range:s.range,real:s.real,reciprocal:s.reciprocal,relu:s.relu,relu6:s.relu6,reshape:s.reshape,reverse:s.reverse,reverse1d:ct,reverse2d:mt,reverse3d:dt,reverse4d:ht,rfft:s.rfft,round:s.round,rsqrt:s.rsqrt,scalar:s.scalar,scatterND:vt,searchSorted:X,selu:s.selu,separableConv2d:s.separableConv2d,setdiff1dAsync:ft,sigmoid:s.sigmoid,sign:s.sign,signal:s.signal,sin:s.sin,sinh:s.sinh,slice:s.slice,slice1d:s.slice1d,slice2d:s.slice2d,slice3d:s.slice3d,slice4d:s.slice4d,softmax:s.softmax,softplus:s.softplus,spaceToBatchND:s.spaceToBatchND,sparse:s.sparse,sparseToDense:Ot,spectral:s.spectral,split:s.split,sqrt:s.sqrt,square:s.square,squaredDifference:s.squaredDifference,squeeze:s.squeeze,stack:s.stack,step:s.step,stridedSlice:s.stridedSlice,string:s.string,sub:s.sub,sum:s.sum,tan:s.tan,tanh:s.tanh,tensor:s.tensor,tensor1d:s.tensor1d,tensor2d:s.tensor2d,tensor3d:s.tensor3d,tensor4d:yt,tensor5d:gt,tensor6d:Nt,tensorScatterUpdate:bt,tile:s.tile,topk:s.topk,transpose:s.transpose,truncatedNormal:s.truncatedNormal,unique:s.unique,unsortedSegmentSum:s.unsortedSegmentSum,unstack:s.unstack,upperBound:Tt,variable:s.variable,where:s.where,whereAsync:fe,zeros:s.zeros,zerosLike:s.zerosLike},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pi=(r,e,t,a=O)=>{switch(r.op){case"BiasAdd":case"AddV2":case"Add":return[a.add(o("a",r,e,t),o("b",r,e,t))];case"AddN":return[a.addN(o("tensors",r,e,t))];case"FloorMod":case"Mod":return[a.mod(o("a",r,e,t),o("b",r,e,t))];case"Mul":return[a.mul(o("a",r,e,t),o("b",r,e,t))];case"RealDiv":case"Div":return[a.div(o("a",r,e,t),o("b",r,e,t))];case"DivNoNan":return[a.divNoNan(o("a",r,e,t),o("b",r,e,t))];case"FloorDiv":return[a.floorDiv(o("a",r,e,t),o("b",r,e,t))];case"Sub":return[a.sub(o("a",r,e,t),o("b",r,e,t))];case"Minimum":return[a.minimum(o("a",r,e,t),o("b",r,e,t))];case"Maximum":return[a.maximum(o("a",r,e,t),o("b",r,e,t))];case"Pow":return[a.pow(o("a",r,e,t),o("b",r,e,t))];case"SquaredDifference":return[a.squaredDifference(o("a",r,e,t),o("b",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ci=(r,e,t,a=O)=>{switch(r.op){case"Abs":case"ComplexAbs":return[a.abs(o("x",r,e,t))];case"Acos":return[a.acos(o("x",r,e,t))];case"Acosh":return[a.acosh(o("x",r,e,t))];case"Asin":return[a.asin(o("x",r,e,t))];case"Asinh":return[a.asinh(o("x",r,e,t))];case"Atan":return[a.atan(o("x",r,e,t))];case"Atan2":return[a.atan2(o("x",r,e,t),o("y",r,e,t))];case"Atanh":return[a.atanh(o("x",r,e,t))];case"Ceil":return[a.ceil(o("x",r,e,t))];case"Complex":return[a.complex(o("real",r,e,t),o("imag",r,e,t))];case"Cos":return[a.cos(o("x",r,e,t))];case"Cosh":return[a.cosh(o("x",r,e,t))];case"Elu":return[a.elu(o("x",r,e,t))];case"Erf":return[a.erf(o("x",r,e,t))];case"Exp":return[a.exp(o("x",r,e,t))];case"Expm1":return[a.expm1(o("x",r,e,t))];case"Floor":return[a.floor(o("x",r,e,t))];case"Log":return[a.log(o("x",r,e,t))];case"Log1p":return[a.log1p(o("x",r,e,t))];case"Imag":return[a.imag(o("x",r,e,t))];case"Neg":return[a.neg(o("x",r,e,t))];case"Reciprocal":return[a.reciprocal(o("x",r,e,t))];case"Real":return[a.real(o("x",r,e,t))];case"Relu":return[a.relu(o("x",r,e,t))];case"Round":return[a.round(o("x",r,e,t))];case"Selu":return[a.selu(o("x",r,e,t))];case"Sigmoid":return[a.sigmoid(o("x",r,e,t))];case"Sin":return[a.sin(o("x",r,e,t))];case"Sign":return[a.sign(o("x",r,e,t))];case"Sinh":return[a.sinh(o("x",r,e,t))];case"Softplus":return[a.softplus(o("x",r,e,t))];case"Sqrt":return[a.sqrt(o("x",r,e,t))];case"Square":return[a.square(o("x",r,e,t))];case"Tanh":return[a.tanh(o("x",r,e,t))];case"Tan":return[a.tan(o("x",r,e,t))];case"ClipByValue":return[a.clipByValue(o("x",r,e,t),o("clipValueMin",r,e,t),o("clipValueMax",r,e,t))];case"Relu6":return[a.relu6(o("x",r,e,t))];case"Rsqrt":return[a.rsqrt(v(r.inputNames[0],e,t))];case"LeakyRelu":return[a.leakyRelu(o("x",r,e,t),o("alpha",r,e,t))];case"Prelu":return[a.prelu(o("x",r,e,t),o("alpha",r,e,t))];case"IsNan":return[a.isNaN(v(r.inputNames[0],e,t))];case"IsInf":return[a.isInf(v(r.inputNames[0],e,t))];case"IsFinite":return[a.isFinite(v(r.inputNames[0],e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D(r,e,t=""){if(!(typeof r=="number"||typeof e=="number")){s.assert(r.length===e.length,()=>t+` Shapes ${r} and ${e} must match`);for(let a=0;a<r.length;a++){const n=r[a],i=e[a];s.assert(n<0||i<0||n===i,()=>t+` Shapes ${r} and ${e} must match`)}}}function $e(r){return!(typeof r=="number"||r.some(e=>e<0))}function V(r,e,t){let a=me(r,t);const n=!$e(a);if(n&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${a}`);if(n&&e.forEach(i=>{a=me(i.shape,a)}),!$e(a))throw new Error(`Non-fully-defined elementShape: ${a}`);return a}function me(r,e){if(typeof r=="number")return e;if(typeof e=="number")return r;if(r.length!==e.length)throw new Error(`Incompatible ranks during merge: ${r} vs. ${e}`);const t=[];for(let a=0;a<r.length;++a){const n=r[a],i=e[a];if(n>=0&&i>=0&&n!==i)throw new Error(`Incompatible shape during merge: ${r} vs. ${e}`);t[a]=n>=0?n:i}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mi{constructor(e,t,a,n,i,l,u){this.name=e,this.dtype=t,this.maxSize=a,this.elementShape=n,this.identicalElementShapes=i,this.dynamicSize=l,this.clearAfterRead=u,this.tensors=[],this.closed_=!1,this.idTensor=s.scalar(0),s.keep(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(t=>this.read(t))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const a=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),D(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),a.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(a.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);a.tensor=t,s.keep(t),a.written=!0,this.tensors[e]=a}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((a,n)=>this.write(a,t[n]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let n=0;n<this.size();n++)e.push(n)}if(e.length===0)return s.tensor([],[0].concat(this.elementShape));const a=this.readMany(e);return D(this.elementShape,a[0].shape,"TensorArray shape mismatch: "),s.stack(a,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return s.tensor([],[0].concat(this.elementShape));const t=[];for(let n=0;n<this.size();n++)t.push(n);const a=this.readMany(t);return D(this.elementShape,a[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${a[0].shape})`),s.concat(a,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const a=Math.max(...e);if(!this.dynamicSize&&a>=this.maxSize)throw new Error(`Max index must be < array size (${a}  vs. ${this.maxSize})`);this.writeMany(e,s.unstack(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let a=0;const n=e.map(p=>(a+=p,a));if(a!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${a}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const i=a===0?0:t.size/a,l=[];s.tidy(()=>{t=s.reshape(t,[1,a,i]);for(let p=0;p<e.length;++p){const m=[0,p===0?0:n[p-1],0],d=[1,e[p],i];l[p]=s.reshape(s.slice(t,m,d),this.elementShape)}return l});const u=[];for(let p=0;p<e.length;p++)u[p]=p;this.writeMany(u,l)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class F{get id(){return this.idTensor.id}constructor(e,t,a,n=-1){this.tensors=e,this.elementShape=t,this.elementDtype=a,e!=null&&e.forEach(i=>{if(a!==i.dtype)throw new Error(`Invalid data types; op elements ${a}, but list elements ${i.dtype}`);D(t,i.shape,"TensorList shape mismatch: "),s.keep(i)}),this.idTensor=s.scalar(0),this.maxNumElements=n,s.keep(this.idTensor)}copy(){return new F([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,a=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(a!==-1&&this.tensors.length!==a)throw new Error(`Operation expected a list with ${a} elements but got a list with ${this.tensors.length} elements.`);D(e,this.elementShape,"TensorList shape mismatch: ");const n=V(this.elementShape,this.tensors,e);return s.tidy(()=>{const i=this.tensors.map(l=>s.reshape(l,n));return s.stack(i,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const a=V(this.elementShape,this.tensors,e),n=this.tensors.pop();return n.kept=!1,D(n.shape,e,"TensorList shape mismatch: "),s.reshape(n,a)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(D(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");s.keep(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new F([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let a=0;a<Math.min(this.tensors.length,e);++a)t.tensors[a]=this.tensors[a];return t}getItem(e,t,a){if(a!==this.elementDtype)throw new Error(`Invalid data types; op elements ${a}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);D(this.tensors[e].shape,t,"TensorList shape mismatch: ");const n=V(this.elementShape,this.tensors,t);return s.reshape(this.tensors[e],n)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);D(this.elementShape,t.shape,"TensorList shape mismatch: "),s.keep(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,a){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);D(this.elementShape,a,"TensorList shape mismatch: "),e=e.slice(0,this.size());const n=V(this.elementShape,this.tensors,a);return e.length===0?s.tensor([],[0].concat(n)):s.tidy(()=>{const i=e.map(l=>s.reshape(this.tensors[l],n));return s.stack(i,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);D(this.elementShape,t,"TensorList shape mismatch: ");const a=V(this.elementShape,this.tensors,t);return this.size()===0?s.tensor([],[0].concat(a)):s.tidy(()=>{const n=this.tensors.map(i=>s.reshape(i,a));return s.concat(n,0)})}}function di(r,e,t){const a=r.dtype;if(r.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${r.shape}`);if(r.dtype!==t)throw new Error(`Invalid data types; op elements ${r.dtype}, but list elements ${t}`);const n=r.shape.slice(1);D(n,e,"TensorList shape mismatch: ");const i=s.unstack(r);return new F(i,e,a)}function hi(r,e,t,a){return new F([],r,e,a)}function fi(r,e,t,a){if(e.length!==r.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${r.shape[0]}`);const n=Math.max(...e);if(a!=null&&a!==-1&&n>=a)throw new Error(`Max index must be < array size (${n}  vs. ${a})`);const i=new F([],t,r.dtype,a),l=s.unstack(r,0);return e.forEach((u,p)=>{i.setItem(u,l[p])}),i}function yi(r,e,t){let a=0;const n=e.map(m=>(a+=m,a));if(a!==r.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${a}, and tensor's shape is: ${r.shape}`);const i=r.shape.slice(1),l=me(i,t),u=a===0?0:r.size/a,p=s.tidy(()=>{const m=[];r=s.reshape(r,[1,a,u]);for(let d=0;d<e.length;++d){const f=[0,d===0?0:n[d-1],0],b=[1,e[d],u];m[d]=s.reshape(s.slice(r,f,b),l)}return r.dispose(),m}),c=new F([],t,r.dtype,e.length);for(let m=0;m<p.length;m++)c.setItem(m,p[m]);return c}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gi=async(r,e,t)=>{switch(r.op){case"If":case"StatelessIf":{const a=o("thenBranch",r,e,t),n=o("elseBranch",r,e,t),i=o("cond",r,e,t),l=o("args",r,e,t);return(await i.data())[0]?t.functionMap[a].executeFunctionAsync(l,t.tensorArrayMap,t.tensorListMap):t.functionMap[n].executeFunctionAsync(l,t.tensorArrayMap,t.tensorListMap)}case"While":case"StatelessWhile":{const a=o("body",r,e,t),n=o("cond",r,e,t),i=o("args",r,e,t),l=await t.functionMap[n].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap),u=i.map(m=>m.id);let p=await l[0].data();l.forEach(m=>{!m.kept&&u.indexOf(m.id)===-1&&m.dispose()});let c=i;for(;p[0];){const m=c;c=await t.functionMap[a].executeFunctionAsync(c,t.tensorArrayMap,t.tensorListMap);const d=c.map(f=>f.id);m.forEach(f=>{!f.kept&&u.indexOf(f.id)===-1&&d.indexOf(f.id)===-1&&f.dispose()});const h=await t.functionMap[n].executeFunctionAsync(c,t.tensorArrayMap,t.tensorListMap);p=await h[0].data(),h.forEach(f=>{!f.kept&&u.indexOf(f.id)===-1&&d.indexOf(f.id)===-1&&f.dispose()})}return c}case"LoopCond":{const a=o("pred",r,e,t);return[z(a)]}case"Switch":{const a=o("pred",r,e,t);let n=o("data",r,e,t);return n.kept||(n=z(n)),(await a.data())[0]?[void 0,n]:[n,void 0]}case"Merge":{const a=r.inputNames.find(n=>v(n,e,t)!==void 0);if(a){const n=v(a,e,t);return[z(n)]}return}case"Enter":{const a=o("frameName",r,e,t),n=o("tensor",r,e,t);return t.enterFrame(a),[z(n)]}case"Exit":{const a=o("tensor",r,e,t);return t.exitFrame(),[z(a)]}case"NextIteration":{const a=o("tensor",r,e,t);return t.nextIteration(),[z(a)]}case"TensorArrayV3":{const a=o("size",r,e,t),n=o("dtype",r,e,t),i=o("elementShape",r,e,t),l=o("dynamicSize",r,e,t),u=o("clearAfterRead",r,e,t),p=o("identicalElementShapes",r,e,t),c=o("name",r,e,t),m=new mi(c,n,a,i,p,l,u);return t.addTensorArray(m),[m.idTensor,s.scalar(1)]}case"TensorArrayWriteV3":{const a=o("tensorArrayId",r,e,t),n=o("index",r,e,t),i=o("tensor",r,e,t),l=t.getTensorArray(a.id);return l.write(n,i),[l.idTensor]}case"TensorArrayReadV3":{const a=o("tensorArrayId",r,e,t),n=o("index",r,e,t);return[t.getTensorArray(a.id).read(n)]}case"TensorArrayGatherV3":{const a=o("tensorArrayId",r,e,t),n=o("indices",r,e,t),i=o("dtype",r,e,t);return[t.getTensorArray(a.id).gather(n,i)]}case"TensorArrayScatterV3":{const a=o("tensorArrayId",r,e,t),n=o("indices",r,e,t),i=o("tensor",r,e,t),l=t.getTensorArray(a.id);return l.scatter(n,i),[l.idTensor]}case"TensorArrayConcatV3":{const a=o("tensorArrayId",r,e,t),n=t.getTensorArray(a.id),i=o("dtype",r,e,t);return[n.concat(i)]}case"TensorArraySplitV3":{const a=o("tensorArrayId",r,e,t),n=o("tensor",r,e,t),i=o("lengths",r,e,t),l=t.getTensorArray(a.id);return l.split(i,n),[l.idTensor]}case"TensorArraySizeV3":{const a=o("tensorArrayId",r,e,t),n=t.getTensorArray(a.id);return[s.scalar(n.size(),"int32")]}case"TensorArrayCloseV3":{const a=o("tensorArrayId",r,e,t),n=t.getTensorArray(a.id);return n.clearAndClose(),[n.idTensor]}case"TensorListSetItem":{const a=o("tensorListId",r,e,t),n=o("index",r,e,t),i=o("tensor",r,e,t),l=t.getTensorList(a.id);return l.setItem(n,i),[l.idTensor]}case"TensorListGetItem":{const a=o("tensorListId",r,e,t),n=o("index",r,e,t),i=o("elementShape",r,e,t),l=o("elementDType",r,e,t);return[t.getTensorList(a.id).getItem(n,i,l)]}case"TensorListScatterV2":case"TensorListScatter":{const a=o("indices",r,e,t),n=o("tensor",r,e,t),i=o("elementShape",r,e,t),l=o("numElements",r,e,t),u=fi(n,a,i,l);return t.addTensorList(u),[u.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const a=o("elementShape",r,e,t),n=o("elementDType",r,e,t);let i;r.op==="TensorListReserve"?i="numElements":i="maxNumElements";const l=o(i,r,e,t),u=r.op==="TensorListReserve"?-1:l,p=hi(a,n,l,u);return t.addTensorList(p),[p.idTensor]}case"TensorListGather":{const a=o("tensorListId",r,e,t),n=o("indices",r,e,t),i=o("elementShape",r,e,t),l=o("elementDType",r,e,t);return[t.getTensorList(a.id).gather(n,l,i)]}case"TensorListStack":{const a=o("tensorListId",r,e,t),n=o("elementShape",r,e,t),i=o("elementDType",r,e,t),l=o("numElements",r,e,t);return[t.getTensorList(a.id).stack(n,i,l)]}case"TensorListFromTensor":{const a=o("tensor",r,e,t),n=o("elementShape",r,e,t),i=o("elementDType",r,e,t),l=di(a,n,i);return t.addTensorList(l),[l.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const a=o("tensorListId",r,e,t),n=t.getTensorList(a.id),i=o("dtype",r,e,t),l=o("elementShape",r,e,t);return[n.concat(i,l)]}case"TensorListPushBack":{const a=o("tensorListId",r,e,t),n=o("tensor",r,e,t),i=t.getTensorList(a.id);return i.pushBack(n),[i.idTensor]}case"TensorListPopBack":{const a=o("tensorListId",r,e,t),n=o("elementShape",r,e,t),i=o("elementDType",r,e,t);return[t.getTensorList(a.id).popBack(n,i)]}case"TensorListSplit":{const a=o("tensor",r,e,t),n=o("elementShape",r,e,t),i=o("lengths",r,e,t),l=yi(a,i,n);return t.addTensorList(l),[l.idTensor]}case"TensorListLength":{const a=o("tensorListId",r,e,t),n=t.getTensorList(a.id);return[s.scalar(n.size(),"int32")]}case"TensorListResize":{const a=o("tensorListId",r,e,t),n=o("size",r,e,t),l=t.getTensorList(a.id).resize(n);return t.addTensorList(l),[l.idTensor]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xe(r,e,t){const[a,n]=o("fusedOps",r,e,t),i=a==="biasadd",l=!i,u=n==="prelu",p=a==="fusedbatchnorm",c=o("numArgs",r,e,t);if(i){if(u&&c!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!u&&i&&c!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(p)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const m=o("strides",r,e,t),d=U(r,e,t),h=o("dataFormat",r,e,t).toUpperCase(),f=o("dilations",r,e,t);let[b,g]=o("args",r,e,t);l&&(g=b,b=void 0);const y=o("leakyreluAlpha",r,e,t);return{stride:m,pad:d,dataFormat:h,dilations:f,biasArg:b,preluArg:g,activationFunc:n,leakyreluAlpha:y}}const Ni=(r,e,t,a=O)=>{switch(r.op){case"Conv1D":{const n=o("stride",r,e,t),i=o("pad",r,e,t),l=o("dataFormat",r,e,t).toUpperCase(),u=o("dilation",r,e,t);return[a.conv1d(o("x",r,e,t),o("filter",r,e,t),n,i,l,u)]}case"Conv2D":{const n=o("strides",r,e,t),i=U(r,e,t),l=o("dataFormat",r,e,t).toUpperCase(),u=o("dilations",r,e,t);return[a.conv2d(o("x",r,e,t),o("filter",r,e,t),[n[1],n[2]],i,l,[u[1],u[2]])]}case"_FusedConv2D":{const{stride:n,pad:i,dataFormat:l,dilations:u,biasArg:p,preluArg:c,activationFunc:m,leakyreluAlpha:d}=xe(r,e,t);return[a.fused.conv2d({x:o("x",r,e,t),filter:o("filter",r,e,t),strides:[n[1],n[2]],pad:i,dataFormat:l,dilations:[u[1],u[2]],bias:p,activation:m,preluActivationWeights:c,leakyreluAlpha:d})]}case"FusedDepthwiseConv2dNative":{const{stride:n,pad:i,dataFormat:l,dilations:u,biasArg:p,preluArg:c,activationFunc:m,leakyreluAlpha:d}=xe(r,e,t);return[a.fused.depthwiseConv2d({x:o("x",r,e,t),filter:o("filter",r,e,t),strides:[n[1],n[2]],pad:i,dataFormat:l,dilations:[u[1],u[2]],bias:p,activation:m,preluActivationWeights:c,leakyreluAlpha:d})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const n=o("outputShape",r,e,t),i=o("strides",r,e,t),l=U(r,e,t);return[a.conv2dTranspose(o("x",r,e,t),o("filter",r,e,t),n,[i[1],i[2]],l)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const n=o("strides",r,e,t),i=U(r,e,t),l=o("dilations",r,e,t),u=o("dataFormat",r,e,t).toUpperCase();return[a.depthwiseConv2d(o("input",r,e,t),o("filter",r,e,t),[n[1],n[2]],i,u,[l[1],l[2]])]}case"Conv3D":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("dataFormat",r,e,t).toUpperCase(),u=o("dilations",r,e,t);return[a.conv3d(o("x",r,e,t),o("filter",r,e,t),[n[1],n[2],n[3]],i,l,[u[1],u[2],u[3]])]}case"AvgPool":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("kernelSize",r,e,t);return[a.avgPool(o("x",r,e,t),[l[1],l[2]],[n[1],n[2]],i)]}case"MaxPool":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("kernelSize",r,e,t);return[a.maxPool(o("x",r,e,t),[l[1],l[2]],[n[1],n[2]],i)]}case"MaxPoolWithArgmax":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("kernelSize",r,e,t),u=o("includeBatchInIndex",r,e,t),{result:p,indexes:c}=a.maxPoolWithArgmax(o("x",r,e,t),[l[1],l[2]],[n[1],n[2]],i,u);return[p,c]}case"AvgPool3D":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("kernelSize",r,e,t);return[a.avgPool3d(o("x",r,e,t),[l[1],l[2],l[3]],[n[1],n[2],n[3]],i)]}case"MaxPool3D":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("kernelSize",r,e,t);return[a.maxPool3d(o("x",r,e,t),[l[1],l[2],l[3]],[n[1],n[2],n[3]],i)]}case"Dilation2D":{const n=o("strides",r,e,t),i=o("pad",r,e,t),l=o("dilations",r,e,t),u=n[1],p=n[2],c=l[1],m=l[2];return[a.dilation2d(o("x",r,e,t),o("filter",r,e,t),[u,p],i,[c,m],"NHWC")]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bi=(r,e,t,a=O)=>{switch(r.op){case"Fill":{const n=o("shape",r,e,t),i=o("dtype",r,e,t),l=o("value",r,e,t);return[a.fill(n,l,i)]}case"LinSpace":{const n=o("start",r,e,t),i=o("stop",r,e,t),l=o("num",r,e,t);return[a.linspace(n,i,l)]}case"Multinomial":{const n=o("logits",r,e,t),i=o("numSamples",r,e,t),l=o("seed",r,e,t);return[a.multinomial(n,i,l)]}case"OneHot":{const n=o("indices",r,e,t),i=o("depth",r,e,t),l=o("onValue",r,e,t),u=o("offValue",r,e,t),p=o("dtype",r,e,t);return[a.oneHot(n,i,l,u,p)]}case"Ones":return[a.ones(o("shape",r,e,t),o("dtype",r,e,t))];case"OnesLike":return[a.onesLike(o("x",r,e,t))];case"RandomStandardNormal":return[a.randomStandardNormal(o("shape",r,e,t),o("dtype",r,e,t),o("seed",r,e,t))];case"RandomUniform":return[a.randomUniform(o("shape",r,e,t),o("minval",r,e,t),o("maxval",r,e,t),o("dtype",r,e,t))];case"RandomUniformInt":return[a.randomUniformInt(o("shape",r,e,t),o("minval",r,e,t),o("maxval",r,e,t),o("seed",r,e,t))];case"Range":{const n=o("start",r,e,t),i=o("stop",r,e,t),l=o("step",r,e,t);return[a.range(n,i,l,o("dtype",r,e,t))]}case"TruncatedNormal":{const n=o("shape",r,e,t),i=o("mean",r,e,t),l=o("stdDev",r,e,t),u=o("seed",r,e,t);return[a.truncatedNormal(n,i,l,o("dtype",r,e,t),u)]}case"Zeros":return[a.zeros(o("shape",r,e,t),o("dtype",r,e,t))];case"ZerosLike":return[a.zerosLike(o("x",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function M(r,e,t){const a=o("boxes",r,e,t),n=o("scores",r,e,t),i=o("maxOutputSize",r,e,t),l=o("iouThreshold",r,e,t),u=o("scoreThreshold",r,e,t),p=o("softNmsSigma",r,e,t);return{boxes:a,scores:n,maxOutputSize:i,iouThreshold:l,scoreThreshold:u,softNmsSigma:p}}const Ti=async(r,e,t,a,n=O)=>{switch(r.op){case"NonMaxSuppressionV5":{const{boxes:i,scores:l,maxOutputSize:u,iouThreshold:p,scoreThreshold:c,softNmsSigma:m}=M(r,e,t),d=await n.image.nonMaxSuppressionWithScoreAsync(i,l,u,p,c,m);return[d.selectedIndices,d.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:i,scores:l,maxOutputSize:u,iouThreshold:p,scoreThreshold:c}=M(r,e,t),m=o("padToMaxOutputSize",r,e,t),d=await n.image.nonMaxSuppressionPaddedAsync(i,l,u,p,c,m);return[d.selectedIndices,d.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:i,scores:l,maxOutputSize:u,iouThreshold:p,scoreThreshold:c}=M(r,e,t);return[await n.image.nonMaxSuppressionAsync(i,l,u,p,c)]}case"Where":{const i=n.cast(o("condition",r,e,t),"bool"),l=[await n.whereAsync(i)];return i.dispose(),l}case"ListDiff":return n.setdiff1dAsync(o("x",r,e,t),o("y",r,e,t));default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wi=(r,e,t,a=O)=>{switch(r.op){case"LowerBound":{const n=o("sortedSequence",r,e,t),i=o("values",r,e,t);return[a.lowerBound(n,i)]}case"TopKV2":{const n=o("x",r,e,t),i=o("k",r,e,t),l=o("sorted",r,e,t),u=a.topk(n,i,l);return[u.values,u.indices]}case"UpperBound":{const n=o("sortedSequence",r,e,t),i=o("values",r,e,t);return[a.upperBound(n,i)]}case"Unique":{const n=o("x",r,e,t),i=a.unique(n);return[i.values,i.indices]}case"UniqueV2":{const n=o("x",r,e,t),i=o("axis",r,e,t),l=a.unique(n,i);return[l.values,l.indices]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Si=(r,e,t,a=O)=>{switch(r.op){case"Const":return e[r.name];case"PlaceholderWithDefault":const n=o("default",r,e,t);return[v(r.name,e,t)||n];case"Placeholder":return[v(r.name,e,t)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const m=o("x",r,e,t);return[z(m)]}case"IdentityN":return o("x",r,e,t).map(m=>z(m));case"Snapshot":const i=o("x",r,e,t);return[z(i)];case"Shape":return[a.tensor1d(o("x",r,e,t).shape,"int32")];case"ShapeN":return o("x",r,e,t).map(m=>a.tensor1d(m.shape));case"Size":return[a.scalar(o("x",r,e,t).size,"int32")];case"Rank":return[a.scalar(o("x",r,e,t).rank,"int32")];case"NoOp":return[a.scalar(1)];case"Print":const l=o("x",r,e,t),u=o("data",r,e,t),p=o("message",r,e,t),c=o("summarize",r,e,t);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(p);for(let m=0;m<u.length;m++)console.log(Array.prototype.slice.call(u[m].dataSync()).slice(0,c));return[l];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vi{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=s.scalar(0),this.tensorMap=new Map,s.keep(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return s.scalar(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const a=await e.data();return this.tensorMap.forEach(n=>n.dispose()),this.tensorMap.clear(),s.tidy(()=>{const n=s.unstack(t),i=a.length,l=n.length;s.assert(i===l,()=>`The number of elements doesn't match, keys has ${i} elements, the values has ${l} elements.`);for(let u=0;u<i;u++){const p=a[u],c=n[u];s.keep(c),this.tensorMap.set(p,c)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const a=await e.data();return s.tidy(()=>{const n=[];for(let i=0;i<a.length;i++){const l=a[i],u=this.findWithDefault(l,t);n.push(u)}return s.stack(n)})}findWithDefault(e,t){const a=this.tensorMap.get(e);return a??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oi=async(r,e,t,a)=>{switch(r.op){case"HashTable":case"HashTableV2":{const n=a.getHashTableHandleByName(r.name);if(n!=null)return[n];{const i=o("keyDType",r,e,t),l=o("valueDType",r,e,t),u=new vi(i,l);return a.addHashTable(r.name,u),[u.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const n=o("tableHandle",r,e,t,a),i=o("keys",r,e,t),l=o("values",r,e,t);return[await a.getHashTableById(n.id).import(i,l)]}case"LookupTableFind":case"LookupTableFindV2":{const n=o("tableHandle",r,e,t,a),i=o("keys",r,e,t),l=o("defaultValue",r,e,t);return[await a.getHashTableById(n.id).find(i,l)]}case"LookupTableSize":case"LookupTableSizeV2":{const n=o("tableHandle",r,e,t,a);return[a.getHashTableById(n.id).tensorSize()]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ei=(r,e,t,a=O)=>{switch(r.op){case"ResizeBilinear":{const n=o("images",r,e,t),i=o("size",r,e,t),l=o("alignCorners",r,e,t),u=o("halfPixelCenters",r,e,t);return[a.image.resizeBilinear(n,[i[0],i[1]],l,u)]}case"ResizeNearestNeighbor":{const n=o("images",r,e,t),i=o("size",r,e,t),l=o("alignCorners",r,e,t),u=o("halfPixelCenters",r,e,t);return[a.image.resizeNearestNeighbor(n,[i[0],i[1]],l,u)]}case"CropAndResize":{const n=o("image",r,e,t),i=o("boxes",r,e,t),l=o("boxInd",r,e,t),u=o("cropSize",r,e,t),p=o("method",r,e,t),c=o("extrapolationValue",r,e,t);return[a.image.cropAndResize(n,i,l,u,p,c)]}case"ImageProjectiveTransformV3":{const n=o("images",r,e,t),i=o("transforms",r,e,t),l=o("outputShape",r,e,t),u=o("fillValue",r,e,t),p=o("interpolation",r,e,t),c=o("fillMode",r,e,t);return[a.image.transform(n,i,p.toLowerCase(),c.toLowerCase(),u,l)]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ai=(r,e,t,a=O)=>{switch(r.op){case"Equal":return[a.equal(o("a",r,e,t),o("b",r,e,t))];case"NotEqual":return[a.notEqual(o("a",r,e,t),o("b",r,e,t))];case"Greater":return[a.greater(o("a",r,e,t),o("b",r,e,t))];case"GreaterEqual":return[a.greaterEqual(o("a",r,e,t),o("b",r,e,t))];case"Less":return[a.less(o("a",r,e,t),o("b",r,e,t))];case"LessEqual":return[a.lessEqual(o("a",r,e,t),o("b",r,e,t))];case"LogicalAnd":return[a.logicalAnd(o("a",r,e,t),o("b",r,e,t))];case"LogicalNot":return[a.logicalNot(o("a",r,e,t))];case"LogicalOr":return[a.logicalOr(o("a",r,e,t),o("b",r,e,t))];case"Select":case"SelectV2":return[a.where(o("condition",r,e,t),o("a",r,e,t),o("b",r,e,t))];case"BitwiseAnd":return[a.bitwiseAnd(o("a",r,e,t),o("b",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _i=(r,e,t,a=O)=>{switch(r.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[a.matMul(o("a",r,e,t),o("b",r,e,t),o("transposeA",r,e,t),o("transposeB",r,e,t))];case"Einsum":return[a.einsum(o("equation",r,e,t),...o("tensors",r,e,t))];case"Transpose":return[a.transpose(o("x",r,e,t),o("perm",r,e,t))];case"_FusedMatMul":const[n,i]=o("fusedOps",r,e,t),l=n==="biasadd",u=i==="prelu",p=o("numArgs",r,e,t),c=o("leakyreluAlpha",r,e,t);if(l){if(u&&p!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!u&&p!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[m,d]=o("args",r,e,t);return[a.fused.matMul({a:o("a",r,e,t),b:o("b",r,e,t),transposeA:o("transposeA",r,e,t),transposeB:o("transposeB",r,e,t),bias:m,activation:i,preluActivationWeights:d,leakyreluAlpha:c})];case"MatrixBandPart":return[a.linalg.bandPart(o("a",r,e,t),o("numLower",r,e,t),o("numUpper",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ki=(r,e,t,a=O)=>{switch(r.op){case"EuclideanNorm":return[a.euclideanNorm(o("x",r,e,t),o("axis",r,e,t),o("keepDims",r,e,t))];case"FusedBatchNorm":case"FusedBatchNormV2":return[a.batchNorm(o("x",r,e,t),o("mean",r,e,t),o("variance",r,e,t),o("offset",r,e,t),o("scale",r,e,t),o("epsilon",r,e,t))];case"FusedBatchNormV3":return[a.batchNorm(o("x",r,e,t),o("mean",r,e,t),o("variance",r,e,t),o("offset",r,e,t),o("scale",r,e,t),o("epsilon",r,e,t))];case"LRN":return[a.localResponseNormalization(o("x",r,e,t),o("radius",r,e,t),o("bias",r,e,t),o("alpha",r,e,t),o("beta",r,e,t))];case"Softmax":return[a.softmax(o("x",r,e,t))];case"LogSoftmax":return[a.logSoftmax(o("x",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ii=(r,e,t,a=O)=>{switch(r.op){case"RaggedGather":{const{outputNestedSplits:n,outputDenseValues:i}=a.raggedGather(o("paramsNestedSplits",r,e,t),o("paramsDenseValues",r,e,t),o("indices",r,e,t),o("outputRaggedRank",r,e,t));return n.concat(i)}case"RaggedRange":{const{rtNestedSplits:n,rtDenseValues:i}=a.raggedRange(o("starts",r,e,t),o("limits",r,e,t),o("splits",r,e,t));return[n,i]}case"RaggedTensorToTensor":return[a.raggedTensorToTensor(o("shape",r,e,t),o("values",r,e,t),o("defaultValue",r,e,t),o("rowPartitionTensors",r,e,t),o("rowPartitionTypes",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Di=(r,e,t,a=O)=>{switch(r.op){case"Max":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.max(o("x",r,e,t),u,p)]}case"Mean":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.mean(o("x",r,e,t),u,p)]}case"Min":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.min(o("x",r,e,t),u,p)]}case"Sum":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.sum(o("x",r,e,t),u,p)]}case"All":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.all(o("x",r,e,t),u,p)]}case"Any":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.any(o("x",r,e,t),u,p)]}case"ArgMax":{const u=o("axis",r,e,t);return[a.argMax(o("x",r,e,t),u)]}case"ArgMin":{const u=o("axis",r,e,t);return[a.argMin(o("x",r,e,t),u)]}case"Prod":{const u=o("axis",r,e,t),p=o("keepDims",r,e,t);return[a.prod(o("x",r,e,t),u,p)]}case"Cumprod":{const u=o("axis",r,e,t),p=o("exclusive",r,e,t),c=o("reverse",r,e,t);return[a.cumprod(o("x",r,e,t),u,p,c)]}case"Cumsum":{const u=o("axis",r,e,t),p=o("exclusive",r,e,t),c=o("reverse",r,e,t);return[a.cumsum(o("x",r,e,t),u,p,c)]}case"Bincount":const n=o("x",r,e,t),i=o("weights",r,e,t),l=o("size",r,e,t);return[a.bincount(n,i,l)];case"DenseBincount":{const u=o("x",r,e,t),p=o("weights",r,e,t),c=o("size",r,e,t),m=o("binaryOutput",r,e,t);return[a.denseBincount(u,p,c,m)]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ci=(r,e,t,a=O)=>{switch(r.op){case"ConcatV2":case"Concat":{const n=o("n",r,e,t),i=o("axis",r,e,t);let l=o("tensors",r,e,t);return l=l.slice(0,n),[a.concat(l,i)]}case"Gather":{const n=o("x",r,e,t),i=o("indices",r,e,t);return[a.gather(n,a.cast(i,"int32"),0)]}case"GatherV2":{const n=o("axis",r,e,t),i=o("batchDims",r,e,t),l=o("x",r,e,t),u=o("indices",r,e,t);return[a.gather(l,a.cast(u,"int32"),n,i)]}case"Reverse":{const n=o("dims",r,e,t),i=[];for(let u=0;u<n.length;u++)n[u]&&i.push(u);const l=o("x",r,e,t);return[a.reverse(l,i)]}case"ReverseV2":{const n=o("axis",r,e,t),i=o("x",r,e,t);return[a.reverse(i,n)]}case"Slice":{const n=o("begin",r,e,t),i=o("size",r,e,t);return[a.slice(o("x",r,e,t),n,i)]}case"StridedSlice":{const n=o("begin",r,e,t),i=o("end",r,e,t),l=o("strides",r,e,t),u=o("beginMask",r,e,t),p=o("endMask",r,e,t),c=o("ellipsisMask",r,e,t),m=o("newAxisMask",r,e,t),d=o("shrinkAxisMask",r,e,t),h=o("x",r,e,t);return[a.stridedSlice(h,n,i,l,u,p,c,m,d)]}case"Pack":return s.tidy(()=>{const n=o("axis",r,e,t),i=o("tensors",r,e,t),l=i[0].shape,u=a.squeeze(i[0]).shape,p=i.map(c=>{const m=s.arraysEqual(c.shape,l);if(!m&&!s.arraysEqual(a.squeeze(c).shape,u))throw new Error("the input tensors shape does not match");return m?c:a.reshape(c,l)});return[a.stack(p,n)]});case"Unpack":{const n=o("axis",r,e,t),i=o("tensor",r,e,t);return a.unstack(i,n)}case"Tile":{const n=o("reps",r,e,t);return[a.tile(o("x",r,e,t),n)]}case"Split":case"SplitV":{const n=o("axis",r,e,t),i=o("numOrSizeSplits",r,e,t),l=o("x",r,e,t);return a.split(l,i,n)}case"ScatterNd":{const n=o("indices",r,e,t),i=o("values",r,e,t),l=o("shape",r,e,t);return[a.scatterND(n,i,l)]}case"GatherNd":{const n=o("x",r,e,t),i=o("indices",r,e,t);return[a.gatherND(n,i)]}case"SparseToDense":{const n=o("sparseIndices",r,e,t),i=o("outputShape",r,e,t),l=o("sparseValues",r,e,t),u=o("defaultValue",r,e,t);return[a.sparseToDense(n,l,i,l.dtype===u.dtype?u:a.cast(u,l.dtype))]}case"TensorScatterUpdate":{const n=o("indices",r,e,t),i=o("values",r,e,t),l=o("tensor",r,e,t);return[a.tensorScatterUpdate(l,n,i)]}default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $i=(r,e,t,a=O)=>{switch(r.op){case"SparseFillEmptyRows":{const{outputIndices:n,outputValues:i,emptyRowIndicator:l,reverseIndexMap:u}=a.sparse.sparseFillEmptyRows(o("indices",r,e,t),o("values",r,e,t),o("denseShape",r,e,t),o("defaultValue",r,e,t));return[n,i,l,u]}case"SparseReshape":{const{outputIndices:n,outputShape:i}=a.sparse.sparseReshape(o("inputIndices",r,e,t),o("inputShape",r,e,t),o("newShape",r,e,t));return[n,i]}case"SparseSegmentMean":return[a.sparse.sparseSegmentMean(o("data",r,e,t),o("indices",r,e,t),o("segmentIds",r,e,t))];case"SparseSegmentSum":return[a.sparse.sparseSegmentSum(o("data",r,e,t),o("indices",r,e,t),o("segmentIds",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xi=(r,e,t,a=O)=>{switch(r.op){case"FFT":return[a.fft(o("x",r,e,t))];case"IFFT":return[a.ifft(o("x",r,e,t))];case"RFFT":return[a.rfft(o("x",r,e,t))];case"IRFFT":return[a.irfft(o("x",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zi=(r,e,t,a=O)=>{switch(r.op){case"StaticRegexReplace":return[a.string.staticRegexReplace(o("input",r,e,t),o("pattern",r,e,t),o("rewrite",r,e,t),o("replaceGlobal",r,e,t))];case"StringNGrams":{const{nGrams:n,nGramsSplits:i}=a.string.stringNGrams(o("data",r,e,t),o("dataSplits",r,e,t),o("separator",r,e,t),o("nGramWidths",r,e,t),o("leftPad",r,e,t),o("rightPad",r,e,t),o("padWidth",r,e,t),o("preserveShortSequences",r,e,t));return[n,i]}case"StringSplit":{const{indices:n,values:i,shape:l}=a.string.stringSplit(o("input",r,e,t),o("delimiter",r,e,t),o("skipEmpty",r,e,t));return[n,i,l]}case"StringToHashBucketFast":return[a.string.stringToHashBucketFast(o("input",r,e,t),o("numBuckets",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Li=(r,e,t,a=O)=>{switch(r.op){case"Cast":return[a.cast(o("x",r,e,t),o("dtype",r,e,t))];case"ExpandDims":{const n=o("axis",r,e,t);return[a.expandDims(o("x",r,e,t),n)]}case"Squeeze":{const n=o("axis",r,e,t);return[a.squeeze(o("x",r,e,t),n)]}case"Reshape":return[a.reshape(o("x",r,e,t),o("shape",r,e,t))];case"EnsureShape":return[a.ensureShape(o("x",r,e,t),o("shape",r,e,t))];case"MirrorPad":return[a.mirrorPad(o("x",r,e,t),o("padding",r,e,t),o("mode",r,e,t))];case"PadV2":case"Pad":return[a.pad(o("x",r,e,t),o("padding",r,e,t),o("constantValue",r,e,t))];case"SpaceToBatchND":{const n=o("blockShape",r,e,t),i=o("paddings",r,e,t);return[a.spaceToBatchND(o("x",r,e,t),n,i)]}case"BatchToSpaceND":{const n=o("blockShape",r,e,t),i=o("crops",r,e,t);return[a.batchToSpaceND(o("x",r,e,t),n,i)]}case"DepthToSpace":{const n=o("blockSize",r,e,t),i=o("dataFormat",r,e,t).toUpperCase();return[a.depthToSpace(o("x",r,e,t),n,i)]}case"BroadcastTo":return[a.broadcastTo(o("x",r,e,t),o("shape",r,e,t))];case"BroadcastArgs":return[a.broadcastArgs(o("s0",r,e,t),o("s1",r,e,t))];default:throw TypeError(`Node type ${r.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ze(r,e,t,a,n=s.tidy){const i=((l,u,p)=>{switch(l.category){case"arithmetic":return n(()=>pi(l,u,p));case"basic_math":return n(()=>ci(l,u,p));case"control":return gi(l,u,p);case"convolution":return n(()=>Ni(l,u,p));case"creation":return n(()=>bi(l,u,p));case"dynamic":return Ti(l,u,p);case"evaluation":return n(()=>wi(l,u,p));case"image":return n(()=>Ei(l,u,p));case"graph":return n(()=>Si(l,u,p));case"logical":return n(()=>Ai(l,u,p));case"matrices":return n(()=>_i(l,u,p));case"normalization":return n(()=>ki(l,u,p));case"ragged":return n(()=>Ii(l,u,p));case"reduction":return n(()=>Di(l,u,p));case"slice_join":return n(()=>Ci(l,u,p));case"sparse":return n(()=>$i(l,u,p));case"spectral":return n(()=>xi(l,u,p));case"string":return n(()=>zi(l,u,p));case"transformation":return n(()=>Li(l,u,p));case"hash_table":return Oi(l,u,p,a);case"custom":const c=Ft(l.op);if(c&&c.customExecutor)return c.customExecutor(new li(l,u,p));throw TypeError(`Custom op ${l.op} is not registered.`);default:throw TypeError(`Unknown op '${l.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(r,e,t);return s.isPromise(i)?i.then(l=>[].concat(l)):[].concat(i)}class Le{constructor(e={},t={},a={},n={},i){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=a,this.functionMap=n,this.parseNodeNameCache=i,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const a=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(a))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(t=>t.id===0&&t.iterationId===0?"":`${t.frameName}-${t.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pe(r,e,t,a){const n=new Set,i=[];let l=null,u=null;const p=new Set,c=new Set(Object.keys(r).map(h=>_(h)[0]));a=a||[];const m=new Set(a.map(h=>_(h.name)[0])),d=[...e];for(;d.length>0;){const h=d.pop();if((L(h)||Hi(h)||Gi(h))&&l==null&&(l=h,u=l.children.map(f=>f.name).filter(f=>n.has(f))),n.add(h.name),t[h.name]==null&&!c.has(h.name)&&!m.has(h.name)){if(h.inputs.length===0){i.push(h.name);continue}h.inputs.forEach(f=>{p.has(f.name)||(p.add(f.name),d.push(f))})}}return{inputs:r,outputs:e,usedNodes:n,missingInputs:i,dynamicNode:l,syncInputs:u}}function Pi(r,e){const{usedNodes:t,inputs:a}=e,n=Object.keys(a).map(y=>_(y)[0]).map(y=>r.nodes[y]),i=r.initNodes||[],l=y=>t.has(typeof y=="string"?y:y.name);function u(y){return[...new Map(y.map(N=>[N.name,N])).values()]}const p=u([...n,...r.weights,...i]).filter(l),c=u([...p,...Object.values(r.nodes)]).filter(l),m=new Map(c.map(y=>[y.name,y])),d={};for(const y of c){d[y.name]=d[y.name]||0;for(const N of y.children)l(N)||(d[N.name]=Number.POSITIVE_INFINITY),d[N.name]=(d[N.name]||0)+1}const h=Object.entries(d).filter(([,y])=>y===0).map(([y])=>y),f=[...h];for(;h.length>0;){const y=h.pop(),N=m.get(y);for(const w of N.children.filter(l))--d[w.name]===0&&(f.push(w.name),h.push(w.name))}const b=f.map(y=>m.get(y)),g=Fi(b,p);return Ri(g,p),g}function Fi(r,e){const t=new Map(r.map(l=>[l.name,l])),a=e.map(l=>l.name),n=new Set(a);for(;a.length>0;){const l=a.pop(),u=t.get(l);for(const p of u.children)!t.has(p.name)||n.has(p.name)||(n.add(p.name),a.push(p.name))}return r.filter(l=>n.has(l.name))}class H extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function Ri(r,e){const t=new Map(r.map((u,p)=>[u.name,p])),a=new Set(e.map(u=>u.name)),n=u=>a.has(typeof u=="string"?u:u.name),i=new Set(r.map(u=>u.name)),l=u=>i.has(typeof u=="string"?u:u.name);for(const u of r){for(const p of u.children.filter(l)){if(!t.has(p.name))throw new H(`Child ${p.name} of node ${u.name} is unreachable.`);if(t.get(u.name)>t.get(p.name))throw new H(`Node ${u.name} is scheduled to run after its child ${p.name}.`)}if(!n(u))for(const p of u.inputs){if(!t.has(p.name))throw new H(`Input ${p.name} of node ${u.name} is unreachable.`);if(t.get(p.name)>t.get(u.name))throw new H(`Node ${u.name} is scheduled to run before its input ${p.name}.`)}}}function Vi(r){const e=new Map(r.map((u,p)=>[u.name,p])),t=Number.MAX_SAFE_INTEGER,a=r.map((u,p)=>L(u)?t:p),n=u=>{const p=a[e.get(u.name)];return p??-1},i=r.map((u,p)=>u.children.map(n).reduce((c,m)=>Math.max(c,m),a[p])),l=new Map;for(let u=0;u<r.length;++u){const p=i[u];if(p===t)continue;const c=r[u],m=r[p];l.has(m.name)||l.set(m.name,[]),l.get(m.name).push(c)}return l}const Bi=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),ji=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),qi=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function L(r){return Bi.has(r.op)}function Hi(r){return ji.has(r.op)}function Gi(r){return qi.has(r.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Q{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(a=>e[a].map(n=>n.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(a=>{this._functionExecutorMap[a]=new Q(e.functions[a],this)})}getCompilationKey(e,t){const a=e.map(i=>i.name).sort(),n=t.map(i=>i.name).sort();return a.join(this.SEPARATOR)+"--"+n.join(this.SEPARATOR)}compile(e,t){const a=Pe(e,t,this.weightMap,this._initNodes),{missingInputs:n,dynamicNode:i,syncInputs:l}=a;if(i!=null)throw new Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${l}]`);if(n.length>0){const c=t.map(d=>d.name),m=Object.keys(e);throw new Error(`Cannot compute the outputs [${c}] from the provided inputs [${m}]. Missing the following inputs: [${n}]`)}const u=Pi(this.graph,a),p=Vi(u);return{orderedNodes:u,nodeLiveUntilMap:p}}cloneAndKeepTensor(e){if(e==null)return null;const t=e.clone();return s.keep(t),t}cloneTensorList(e){return e?e.map(a=>this.cloneAndKeepTensor(a)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([t,a])=>[t,this.cloneTensorList(a)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const a=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const n=a.map(h=>this.graph.nodes[_(h)[0]]),i=t.map(h=>_(h)[0]),l=new Set(i);let u=i.map(h=>this.graph.nodes[h]);u.length===0&&(u=this._outputs);const p=this.getCompilationKey(n,u);let c=this.compiledMap.get(p);c==null&&(c=this.compile(e,u),this.compiledMap.set(p,c));try{this.keepIntermediateTensors=s.env().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(h){this.keepIntermediateTensors=!1,console.warn(h.message)}const m={},d={};return s.tidy(()=>{const h=new Le(this.weightMap,m,d,this.functionExecutorMap,this.parseNodeNameCache),f=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(N=>{const[w,S]=_(N,h),T=[];T[S]=e[N],f[w]=T,this.keepIntermediateTensors&&(this.clonedTensorsMap[w]=this.cloneTensorList(T))});const b=this.getFrozenTensorIds(f),{orderedNodes:g,nodeLiveUntilMap:y}=c;for(const N of g){if(f[N.name])continue;const w=ze(N,f,h,this._resourceManager);if(s.isPromise(w))throw new Error(`The execution of the op '${N.op}' returned a promise. Please use model.executeAsync() instead.`);f[N.name]=w,this.keepIntermediateTensors&&(this.clonedTensorsMap[N.name]=this.cloneTensorList(w)),this.checkTensorForDisposalWithNodeLiveUntilInfo(N,f,h,b,l,y.get(N.name))}return this.parent==null&&h.dispose(b),t.map(N=>v(N,f,h))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(a=>e[a]).map(a=>a.map(n=>n.id)));return new Set(t)}checkTensorForDisposal(e,t,a,n,i,l,u){if(!(L(t)||l.has(e))){for(const p of a[e])p!=null&&(u[p.id]=(u[p.id]||0)+t.children.length);for(const p of t.inputs){if(L(p))continue;const c=Ie(p.name,a,n);if(c!=null)for(const m of c){if(!m||m.kept||i.has(m.id))continue;const d=u[m.id];d===1?(m.dispose(),delete u[m.id]):d!=null&&u[m.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,a,n,i,l){function u(p){return L(p)||i.has(p.name)}if(!(L(e)||l==null))for(const p of l){if(u(p))continue;const c=Ie(p.name,t,a);for(const m of c)!m||m.kept||n.has(m.id)||m.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,a=!1,n={},i={}){this.disposeIntermediateTensors(),a||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=s.env().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(h){this.keepIntermediateTensors=!1,console.warn(h.message)}const l=new Le(this.weightMap,n,i,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const u=await this.executeWithControlFlow(e,l,t,a),p=t.map(h=>v(h,u,l)),c=p.map(h=>h.id),m=Object.keys(e).map(h=>e[h].id),d=new Set([...c,...m,...this.weightIds]);return Object.values(u).forEach(h=>{h.forEach(f=>{f&&!f.isDisposed&&!d.has(f.id)&&f.dispose()})}),this.parent==null&&l.dispose(d),p}async executeFunctionAsync(e,t,a){const n=e.reduce((i,l,u)=>(i[this.inputs[u].name]=l,i),{});return this._executeAsync(n,this.outputNodes,!0,t,a)}async executeWithControlFlow(e,t,a,n){const i=Object.keys(e),l=i.map(T=>this.graph.nodes[_(T)[0]]),u=a.map(T=>_(T)[0]),p=new Set(u);let c=u.map(T=>this.graph.nodes[T]);c.length===0&&(c=this._outputs);const{usedNodes:m,missingInputs:d,dynamicNode:h,syncInputs:f}=Pe(e,c,this.weightMap,this._initNodes),b=[...l,...this.graph.weights,...this._initNodes||[]].map(T=>({node:T,contexts:t.currentContext})),g=Object.assign({},this.weightMap);Object.keys(e).forEach(T=>{const[A,k]=_(T),I=[];I[k]=e[T],g[A]=I});const y={},N=this.getFrozenTensorIds(g),w={};for(;b.length>0;){const T=this.processStack(l,b,t,g,w,N,p,y,m);await Promise.all(T)}h==null&&!n&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const S=c.filter(T=>!L(T)&&!v(T.name,g,t)).map(T=>T.name);if(S.length>0){let T="";throw h!=null&&(T=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${f}]`),new Error(`Cannot compute the outputs [${S}] from the provided inputs [${i}]. Consider providing the following inputs: [${d}]. ${T}`)}return g}processStack(e,t,a,n,i,l,u,p,c){const m=[];for(;t.length>0;){const d=t.pop();a.currentContext=d.contexts;let h="";if(d.node.op==="Enter"&&o("isConstant",d.node,n,a)&&([h]=x(d.node.name,a)),n[d.node.name]==null){const f=ze(d.node,n,a,this._resourceManager);h||([h]=x(d.node.name,a));const b=a.currentContext;s.isPromise(f)?m.push(f.then(g=>(n[h]=g,this.keepIntermediateTensors&&(this.clonedTensorsMap[h]=this.cloneTensorList(g)),a.currentContext=b,this.checkTensorForDisposal(h,d.node,n,a,l,u,p),this.processChildNodes(d.node,t,a,n,i,c),g))):(n[h]=f,this.keepIntermediateTensors&&(this.clonedTensorsMap[h]=this.cloneTensorList(f)),this.checkTensorForDisposal(h,d.node,n,a,l,u,p),this.processChildNodes(d.node,t,a,n,i,c))}else this.processChildNodes(d.node,t,a,n,i,c)}return m}processChildNodes(e,t,a,n,i,l){e.children.forEach(u=>{const[p]=x(u.name,a);i[p]||!l.has(u.name)||(u.op==="Merge"?u.inputNames.some(c=>!!v(c,n,a))&&(i[p]=!0,t.push({contexts:a.currentContext,node:u})):u.inputNames.every(c=>!!v(c,n,a))&&(i[p]=!0,t.push({contexts:a.currentContext,node:u})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(t=>t.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const a=e[t],[n]=_(t),i=this.graph.nodes[n];if(i.attrParams.shape&&i.attrParams.shape.value){const l=i.attrParams.shape.value,u=l.length===a.shape.length&&a.shape.every((p,c)=>l[c]===-1||l[c]===p);s.assert(u,()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${l}], but was [${a.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&s.assert(a.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${a.dtype}`)})}mapInputs(e){var t,a;const n={};for(const i in e){const l=(a=(t=this._signature)===null||t===void 0?void 0:t.inputs)===null||a===void 0?void 0:a[i];l!=null?n[l.name]=e[i]:n[i]=e[i]}return n}checkInputs(e){const t=Object.keys(e).filter(a=>{const[n]=_(a);return this.graph.nodes[n]==null});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(t=>{var a,n;const i=(n=(a=this._signature)===null||a===void 0?void 0:a.outputs)===null||n===void 0?void 0:n[t];return i!=null?i.name:t},{})}checkOutputs(e){e.forEach(t=>{const[a]=_(t);if(!this.graph.nodes[a])throw new Error(`The output '${t}' is not found in the graph`)})}}class Wi{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ui="?tfjs-format=file",Ki="model.json";class Se{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},a=ye){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=a,t==null&&(this.loadOptions={}),this.resourceManager=new Wi}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return s.isPromise(e)?e.then(t=>this.loadSync(t)):this.loadSync(e)}loadSync(e){this.artifacts=e;const t=this.artifacts.modelTopology;let a=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const i=this.artifacts.userDefinedMetadata;i.signature!=null&&(a=i.signature),i.structuredOutputKeys!=null&&(this.structuredOutputKeys=i.structuredOutputKeys)}this.signature=a,this.version=`${t.versions.producer}.${t.versions.minConsumer}`;const n=this.io.decodeWeights(this.artifacts.weightData,this.artifacts.weightSpecs);if(this.executor=new Q(De.Instance.transformGraph(t,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(n),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const i=De.Instance.transformGraph(e.modelInitializer);this.initializer=new Q(i),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e=="string"){const a=this.io.getSaveHandlers(e);if(a.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(a.length>1)throw new Error(`Found more than one (${a.length}) save handlers for URL '${e}'`);e=a[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t=e instanceof s.Tensor?[e]:e,a={};return t.forEach((n,i)=>a[this.structuredOutputKeys[i]]=n),a}return e}predict(e,t){const a=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(a)}async predictAsync(e,t){const a=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(a)}normalizeInputs(e){var t;if(!(e instanceof s.Tensor)&&!Array.isArray(e)){const i=(t=this.signature)===null||t===void 0?void 0:t.inputs;if(i!=null)for(const l in i){const u=i[l];u.resourceId!=null&&(e[l]=this.resourceIdToCapturedInput[u.resourceId])}return e}e=Array.isArray(e)?e:[e];const a=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+a!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-a} non-resource placeholders, while there are ${e.length} input tensors provided.`);let n=0;return this.inputNodes.reduce((i,l)=>{var u,p,c;const m=(c=(p=(u=this.signature)===null||u===void 0?void 0:u.inputs)===null||p===void 0?void 0:p[l])===null||c===void 0?void 0:c.resourceId;return m!=null?i[l]=this.resourceIdToCapturedInput[m]:i[l]=e[n++],i},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,a=Object.keys(t);for(let n=0;n<a.length;n++){const i=a[n],l=t[i];this.resourceIdToCapturedInput[l.resourceId]=e[n]}}}execute(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const a=this.executor.execute(e,t);return a.length>1?a:a[0]}async executeAsync(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const a=await this.executor.executeAsync(e,t);return a.length>1?a:a[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,a)=>(t[a]=[e[a]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&s.dispose(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function Ji(r,e={},t=ye){if(r==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof r=="string"&&(r=Xi(r));const a=new Se(r,e,t);return await a.load(),a}function Qi(r){if(r==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let e;if(r instanceof Array){const[a,n]=r;if(!a)throw new Error("modelJSON must be the first element of the array");if(!n||!(n instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in a))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in a))throw new Error("Model JSON is missing 'weightsManifest'");const i=s.getWeightSpecs(a.weightsManifest),l=s.getModelArtifactsForJSONSync(a,i,n);e=K(l)}else if("load"in r)e=r;else if("modelTopology"in r&&"weightSpecs"in r&&"weightData"in r)e=K(r);else throw new Error("Unknown model format");const t=new Se(e);return t.load(),t}function Xi(r){return r.endsWith("/")||(r=r+"/"),`${r}${Ki}${Ui}`}/** @license See the LICENSE file. */const Bt="4.10.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class jt extends s.Dataset{constructor(e){super(),this.input=e}async iterator(){return(await this.input.iterator()).decodeUTF8().split(`
`).map(n=>(n.endsWith("\r")&&(n=n.slice(0,-1)),n))}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */const G='"',B=Symbol("out"),Fe=Symbol("field"),W=Symbol("quote"),ee=Symbol("quoteafterquote"),Re=Symbol("quoteinquote");class qt extends s.Dataset{async columnNames(){return this.columnNamesValidated||await this.setColumnNames(),this.configuredColumnsOnly?Object.keys(this.columnConfigs):this.fullColumnNames}async setColumnNames(){const e=await this.maybeReadHeaderLine();if(!this.fullColumnNames&&!e)throw new Error("Column names must be provided if there is no header line.");this.fullColumnNames&&e&&s.assert(e.length===this.fullColumnNames.length,()=>"The length of provided columnNames ("+this.fullColumnNames.length.toString()+") does not match the length of the header line read from file ("+e.length.toString()+")."),this.fullColumnNames||(this.fullColumnNames=e);const t=this.fullColumnNames.reduce((n,i)=>(n[i]=n[i]+1||1,n),{}),a=Object.keys(t).filter(n=>t[n]>1);if(s.assert(a.length===0,()=>"Duplicate column names found: "+a.toString()),this.columnConfigs){for(const n of Object.keys(this.columnConfigs))if(this.fullColumnNames.indexOf(n)===-1)throw new Error('The key "'+n+'" provided in columnConfigs does not match any of the column names ('+this.fullColumnNames.toString()+").")}this.columnNamesValidated=!0}async maybeReadHeaderLine(){if(this.hasHeader){const t=await(await this.base.iterator()).next();if(t.done)throw new Error("No data was found for CSV parsing.");const a=t.value;return this.parseRow(a,!1)}else return null}constructor(e,t){super(),this.input=e,this.hasHeader=!0,this.fullColumnNames=null,this.columnNamesValidated=!1,this.columnConfigs=null,this.configuredColumnsOnly=!1,this.delimiter=",",this.delimWhitespace=!1,this.base=new jt(e),t||(t={}),this.hasHeader=t.hasHeader!==!1,this.fullColumnNames=t.columnNames,this.columnConfigs=t.columnConfigs,this.configuredColumnsOnly=t.configuredColumnsOnly,t.delimWhitespace?(s.assert(t.delimiter==null,()=>"Delimiter should not be provided when delimWhitespace is true."),this.delimWhitespace=!0,this.delimiter=" "):this.delimiter=t.delimiter?t.delimiter:","}async iterator(){this.columnNamesValidated||await this.setColumnNames();let e=await this.base.iterator();return this.hasHeader&&(e=e.skip(1)),e.map(t=>this.makeDataElement(t))}makeDataElement(e){const t=this.parseRow(e),a={},n={};for(let i=0;i<this.fullColumnNames.length;i++){const l=this.fullColumnNames[i],u=this.columnConfigs?this.columnConfigs[l]:null;if(!(this.configuredColumnsOnly&&!u)){const p=t[i];let c=null;if(p==="")if(u&&u.default!==void 0)c=u.default;else{if(u&&(u.required||u.isLabel))throw new Error(`Required column ${l} is empty in this line: ${e}`);c=void 0}else{const m=Number(p);if(isNaN(m))u&&u.dtype==="bool"?c=this.getBoolean(p):c=p;else if(!u||!u.dtype)c=m;else switch(u.dtype){case"float32":c=m;break;case"int32":c=Math.floor(m);break;case"bool":c=this.getBoolean(p);break;default:c=m}}u&&u.isLabel?n[l]=c:a[l]=c}}return Object.keys(n).length===0?a:{xs:a,ys:n}}getBoolean(e){return e==="1"||e.toLowerCase()==="true"?1:0}parseRow(e,t=!0){const a=[];let n=0;const i=e.length;let l=B;for(let u=0;u<i;u++)switch(l){case B:switch(e.charAt(u)){case G:n=u+1,l=W;break;case this.delimiter:if(n=u+1,this.delimiter===" "&&this.delimWhitespace)break;a.push(""),l=B;break;default:l=Fe,n=u;break}break;case Fe:switch(e.charAt(u)){case this.delimiter:a.push(e.substring(n,u)),l=B,n=u+1;break}break;case W:switch(e.charAt(u)){case G:l=ee;break}break;case ee:switch(e.charAt(u)){case this.delimiter:a.push(e.substring(n,u-1)),l=B,n=u+1;break;case G:l=W;break;default:l=Re;break}break;case Re:switch(e.charAt(u)){case G:l=W;break}break}if(l===ee?a.push(e.substring(n,i-1)):a.push(e.substring(n)),t&&a.length!==this.fullColumnNames.length)throw new Error(`Invalid row in csv file. Should have ${this.fullColumnNames.length} elements in a row, but got ${a}`);return a}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class ve extends s.LazyIterator{constructor(e){super(),this.microphoneConfig=e,this.isClosed=!1,this.fftSize=e.fftSize||1024;const t=Math.log2(this.fftSize);if(this.fftSize<0||t<4||t>14||!Number.isInteger(t))throw new Error(`Invalid fftSize: it must be a power of 2 between 2 to 4 and 2 to 14, but got ${this.fftSize}`);if(this.numFrames=e.numFramesPerSpectrogram||43,this.sampleRateHz=e.sampleRateHz,this.columnTruncateLength=e.columnTruncateLength||this.fftSize,this.audioTrackConstraints=e.audioTrackConstraints,this.smoothingTimeConstant=e.smoothingTimeConstant||0,this.includeSpectrogram=e.includeSpectrogram!==!1,this.includeWaveform=e.includeWaveform===!0,!this.includeSpectrogram&&!this.includeWaveform)throw new Error("Both includeSpectrogram and includeWaveform are false. At least one type of data should be returned.")}summary(){return"microphone"}static async create(e={}){if(!s.env().get("IS_BROWSER"))throw new Error("microphone API is only supported in browser environment.");const t=new ve(e);return await t.start(),t}async start(){try{this.stream=await navigator.mediaDevices.getUserMedia({audio:this.audioTrackConstraints==null?!0:this.audioTrackConstraints,video:!1})}catch(a){throw new Error(`Error thrown while initializing video stream: ${a.message}`)}if(!this.stream)throw new Error("Could not obtain audio from microphone.");const e=window.AudioContext||window.webkitAudioContext;if(this.audioContext=new e,!this.sampleRateHz)this.sampleRateHz=this.audioContext.sampleRate;else if(this.audioContext.sampleRate!==this.sampleRateHz)throw new Error(`Mismatch in sampling rate: Expected: ${this.sampleRateHz}; Actual: ${this.audioContext.sampleRate}`);const t=this.audioContext.createMediaStreamSource(this.stream);this.analyser=this.audioContext.createAnalyser(),this.analyser.fftSize=this.fftSize*2,this.analyser.smoothingTimeConstant=this.smoothingTimeConstant,t.connect(this.analyser),this.freqData=new Float32Array(this.fftSize),this.timeData=new Float32Array(this.fftSize)}async next(){if(this.isClosed)return{value:null,done:!0};let e,t;const a=await this.getAudioData();if(this.includeSpectrogram){const n=this.flattenQueue(a.freqDataQueue);e=this.getTensorFromAudioDataArray(n,[this.numFrames,this.columnTruncateLength,1])}if(this.includeWaveform){const n=this.flattenQueue(a.timeDataQueue);t=this.getTensorFromAudioDataArray(n,[this.numFrames*this.fftSize,1])}return{value:{spectrogram:e,waveform:t},done:!1}}async capture(){return(await this.next()).value}async getAudioData(){const e=[],t=[];let a=0;return new Promise(n=>{const i=setInterval(()=>{this.includeSpectrogram&&(this.analyser.getFloatFrequencyData(this.freqData),this.freqData[0]===-1/0&&n({freqDataQueue:e,timeDataQueue:t}),e.push(this.freqData.slice(0,this.columnTruncateLength))),this.includeWaveform&&(this.analyser.getFloatTimeDomainData(this.timeData),t.push(this.timeData.slice())),++a===this.numFrames&&(clearInterval(i),n({freqDataQueue:e,timeDataQueue:t}))},this.fftSize/this.sampleRateHz*1e3)})}stop(){this.isClosed||(this.isClosed=!0,this.analyser.disconnect(),this.audioContext.close(),this.stream!=null&&this.stream.getTracks().length>0&&this.stream.getTracks()[0].stop())}toArray(){throw new Error("Can not convert infinite audio stream to array.")}getSampleRate(){return this.sampleRateHz}flattenQueue(e){const t=e[0].length,a=new Float32Array(e.length*t);return e.forEach((n,i)=>a.set(n,i*t)),a}getTensorFromAudioDataArray(e,t){const a=new Float32Array(s.sizeFromShape(t));return a.set(e,a.length-e.length),s.tensor(a,t)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Oe extends s.LazyIterator{constructor(e,t){if(super(),this.webcamVideoElement=e,this.webcamConfig=t,this.isClosed=!0,this.resize=!1,this.needToResize())if(this.resize=!0,this.cropSize=[this.webcamConfig.resizeHeight,this.webcamConfig.resizeWidth],this.cropBoxInd=s.tensor1d([0],"int32"),this.webcamConfig.centerCrop){const a=this.webcamConfig.resizeWidth*1/this.webcamVideoElement.width,n=this.webcamConfig.resizeHeight*1/this.webcamVideoElement.height,i=(1-a)/2,l=(1-n)/2,u=i+a,p=n+l;this.cropBox=s.tensor2d([l,i,p,u],[1,4])}else this.cropBox=s.tensor2d([0,0,1,1],[1,4])}summary(){return"webcam"}static async create(e,t={}){if(!s.env().get("IS_BROWSER"))throw new Error("tf.data.webcam is only supported in browser environment.");if(!e){if(e=document.createElement("video"),!t.resizeWidth||!t.resizeHeight)throw new Error("Please provide webcam video element, or resizeWidth and resizeHeight to create a hidden video element.");e.width=t.resizeWidth,e.height=t.resizeHeight}const a=new Oe(e,t);return await a.start(),a}async start(){this.webcamConfig.facingMode&&s.assert(this.webcamConfig.facingMode==="user"||this.webcamConfig.facingMode==="environment",()=>`Invalid webcam facing mode: ${this.webcamConfig.facingMode}. Please provide 'user' or 'environment'`);try{this.stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:this.webcamConfig.deviceId,facingMode:this.webcamConfig.facingMode?this.webcamConfig.facingMode:"user",width:this.webcamVideoElement.width,height:this.webcamVideoElement.height}})}catch(e){throw e.message=`Error thrown while initializing video stream: ${e.message}`,e}if(!this.stream)throw new Error("Could not obtain video from webcam.");try{this.webcamVideoElement.srcObject=this.stream}catch(e){console.log(e),this.webcamVideoElement.src=window.URL.createObjectURL(this.stream)}return this.webcamVideoElement.play(),this.isClosed=!1,new Promise(e=>{this.webcamVideoElement.onloadedmetadata=()=>{e()}})}async next(){if(this.isClosed)return{value:null,done:!0};let e;try{e=s.fromPixels(this.webcamVideoElement)}catch(t){throw new Error(`Error thrown converting video to pixels: ${JSON.stringify(t)}`)}if(this.resize)try{return{value:this.cropAndResizeFrame(e),done:!1}}catch(t){throw new Error(`Error thrown cropping the video: ${t.message}`)}finally{e.dispose()}else return{value:e,done:!1}}needToResize(){return!!(this.webcamConfig.resizeWidth&&this.webcamConfig.resizeHeight&&(this.webcamVideoElement.width!==this.webcamConfig.resizeWidth||this.webcamVideoElement.height!==this.webcamConfig.resizeHeight))}cropAndResizeFrame(e){return s.tidy(()=>{const t=s.expandDims(s.cast(e,"float32"),0);let a;a=s.image.cropAndResize(t,this.cropBox,this.cropBoxInd,this.cropSize,"bilinear");const n=a.shape;return s.reshape(a,n.slice(1))})}async capture(){return(await this.next()).value}stop(){this.stream.getTracks().forEach(t=>t.stop());try{this.webcamVideoElement.srcObject=null}catch(t){console.log(t),this.webcamVideoElement.src=null}this.isClosed=!0}toArray(){throw new Error("Can not convert infinite video stream to array.")}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Ht{}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Gt extends s.LazyIterator{split(e){return new Zi(this,e)}}class Zi extends Gt{constructor(e,t){super(),this.upstream=e,this.impl=new Yi(e,t)}summary(){return this.impl.summary()}async next(){return this.impl.next()}}class Yi extends s.OneToManyIterator{constructor(e,t){super(),this.upstream=e,this.separator=t,this.carryover=""}summary(){return`${this.upstream.summary()} -> Split('${this.separator}')`}async pump(){const e=await this.upstream.next();if(e.done)return this.carryover===""?!1:(this.outputQueue.push(this.carryover),this.carryover="",!0);const t=e.value.split(this.separator);t[0]=this.carryover+t[0];for(const a of t.slice(0,-1))this.outputQueue.push(a);return this.carryover=t[t.length-1],!0}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Mi extends s.LazyIterator{decodeUTF8(){return new eo(this)}}class eo extends Gt{constructor(e){super(),this.upstream=e,this.impl=new to(e)}summary(){return this.impl.summary()}async next(){return this.impl.next()}}class to extends s.OneToManyIterator{constructor(e){if(super(),this.upstream=e,s.env().get("IS_BROWSER"))this.decoder=new TextDecoder("utf-8");else{const{StringDecoder:t}=require("string_decoder");this.decoder=new t("utf8")}}summary(){return`${this.upstream.summary()} -> Utf8`}async pump(){const e=await this.upstream.next();let t;if(e.done)return!1;t=e.value;let a;return s.env().get("IS_BROWSER")?a=this.decoder.decode(t,{stream:!0}):a=this.decoder.write(Buffer.from(t.buffer)),this.outputQueue.push(a),!0}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Wt extends Mi{constructor(e,t={}){super(),this.file=e,this.options=t,s.assert(e instanceof Uint8Array||(s.env().get("IS_BROWSER")?e instanceof File||e instanceof Blob:!1),()=>"FileChunkIterator only supports File, Blob and Uint8Array right now."),this.offset=t.offset||0,this.chunkSize=t.chunkSize||1024*1024}summary(){return`FileChunks ${this.file}`}async next(){return this.offset>=(this.file instanceof Uint8Array?this.file.byteLength:this.file.size)?{value:null,done:!0}:{value:await new Promise((t,a)=>{const n=this.offset+this.chunkSize;if(this.file instanceof Uint8Array)t(new Uint8Array(this.file.slice(this.offset,n)));else{const i=new FileReader;i.onload=u=>{let p=i.result;if(p instanceof ArrayBuffer&&(p=new Uint8Array(p)),!(p instanceof Uint8Array))return a(new TypeError("FileReader returned unknown type."));t(p)},i.onabort=u=>a(new Error("Aborted")),i.onerror=u=>a(new Error(u.type));const l=this.file.slice(this.offset,n);i.readAsArrayBuffer(l)}this.offset=n}),done:!1}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */async function ro(r,e={},t){let a,n;typeof r=="string"?a=r:(a=r.url,n=so(r));const i=await(t||s.fetch)(a,n);if(i.ok){const l=new Uint8Array(await i.arrayBuffer());return new Wt(l,e)}else throw new Error(i.statusText)}const so=r=>({method:r.method,headers:r.headers,body:r.body,mode:r.mode,credentials:r.credentials,cache:r.cache,redirect:r.redirect,referrer:r.referrer,integrity:r.integrity});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */function Ut(r){return typeof r=="string"&&r.slice(0,7)==="file://"}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Kt extends Ht{constructor(e,t={}){super(),this.input=e,this.options=t}async iterator(){if(Ut(this.input)&&s.env().get("IS_NODE")){const e=require("fs");this.input=e.readFileSync(this.input.slice(7))}return new Wt(this.input,this.options)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */class Jt extends Ht{constructor(e,t={}){super(),this.url=e,this.fileOptions=t}async iterator(){return Ut(this.url)?new Kt(this.url,this.fileOptions).iterator():ro(this.url,this.fileOptions)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */function ao(r,e={}){return new qt(new Jt(r),e)}function no(r){const e=s.iteratorFromFunction(r);return s.datasetFromIteratorFn(async()=>e)}function io(r){return s.datasetFromIteratorFn(async()=>{const e=await r();return s.iteratorFromFunction(()=>e.next())})}async function oo(r,e){return Oe.create(r,e)}async function uo(r){return ve.create(r)}/** @license See the LICENSE file. */const Qt="4.10.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lo=Object.freeze(Object.defineProperty({__proto__:null,CSVDataset:qt,Dataset:s.Dataset,FileDataSource:Kt,TextLineDataset:jt,URLDataSource:Jt,array:s.array,csv:ao,func:no,generator:io,microphone:uo,version_data:Qt,webcam:oo,zip:s.zip},Symbol.toStringTag,{value:"Module"}));/** @license See the LICENSE file. */const Xt="4.10.0";/** @license See the LICENSE file. */const Zt="4.10.0";/** @license See the LICENSE file. */const po="4.10.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const co={"tfjs-core":It,"tfjs-backend-cpu":Xt,"tfjs-backend-webgl":Zt,"tfjs-data":Qt,"tfjs-layers":s.version,"tfjs-converter":Bt,tfjs:po},mo=Object.freeze(Object.defineProperty({__proto__:null,Abs:s.Abs,Acos:s.Acos,Acosh:s.Acosh,AdadeltaOptimizer:s.AdadeltaOptimizer,AdagradOptimizer:s.AdagradOptimizer,AdamOptimizer:s.AdamOptimizer,AdamaxOptimizer:s.AdamaxOptimizer,Add:s.Add$1,AddN:s.AddN,All:s.All,Any:s.Any,ArgMax:s.ArgMax,ArgMin:s.ArgMin,Asin:s.Asin,Asinh:s.Asinh,Atan:s.Atan,Atan2:s.Atan2,Atanh:s.Atanh,AvgPool:s.AvgPool,AvgPool3D:s.AvgPool3D,AvgPool3DGrad:s.AvgPool3DGrad,AvgPoolGrad:s.AvgPoolGrad,BatchMatMul:s.BatchMatMul,BatchToSpaceND:s.BatchToSpaceND,Bincount:s.Bincount,BitwiseAnd:s.BitwiseAnd,BroadcastArgs:s.BroadcastArgs,BroadcastTo:s.BroadcastTo,Callback:Lt,CallbackList:s.CallbackList,Cast:s.Cast,Ceil:s.Ceil,ClipByValue:s.ClipByValue,Complex:s.Complex,ComplexAbs:s.ComplexAbs,Concat:s.Concat,Conv2D:s.Conv2D$1,Conv2DBackpropFilter:s.Conv2DBackpropFilter,Conv2DBackpropInput:s.Conv2DBackpropInput,Conv3D:s.Conv3D$1,Conv3DBackpropFilterV2:s.Conv3DBackpropFilterV2,Conv3DBackpropInputV2:s.Conv3DBackpropInputV2,Cos:s.Cos,Cosh:s.Cosh,CropAndResize:s.CropAndResize,Cumprod:s.Cumprod,Cumsum:s.Cumsum,CustomCallback:s.CustomCallback,DataStorage:s.DataStorage,DenseBincount:s.DenseBincount,DepthToSpace:s.DepthToSpace,DepthwiseConv2dNative:s.DepthwiseConv2dNative,DepthwiseConv2dNativeBackpropFilter:s.DepthwiseConv2dNativeBackpropFilter,DepthwiseConv2dNativeBackpropInput:s.DepthwiseConv2dNativeBackpropInput,Diag:s.Diag,Dilation2D:s.Dilation2D,Dilation2DBackpropFilter:s.Dilation2DBackpropFilter,Dilation2DBackpropInput:s.Dilation2DBackpropInput,Draw:s.Draw,get ENV(){return s.ENV},EarlyStopping:Pt,Einsum:s.Einsum,Elu:s.Elu,EluGrad:s.EluGrad,Environment:s.Environment,Equal:s.Equal,Erf:s.Erf,Exp:s.Exp,ExpandDims:s.ExpandDims,Expm1:s.Expm1,FFT:s.FFT,Fill:s.Fill,FlipLeftRight:s.FlipLeftRight,Floor:s.Floor,FloorDiv:s.FloorDiv,FromPixels:s.FromPixels,FusedBatchNorm:s.FusedBatchNorm,FusedConv2D:s.FusedConv2D,FusedDepthwiseConv2D:s.FusedDepthwiseConv2D,GPGPUContext:s.GPGPUContext,GatherNd:s.GatherNd,GatherV2:s.GatherV2,GraphModel:Se,Greater:s.Greater,GreaterEqual:s.GreaterEqual,History:s.History,IFFT:s.IFFT,Identity:s.Identity$1,Imag:s.Imag,InputSpec:s.InputSpec,IsFinite:s.IsFinite,IsInf:s.IsInf,IsNan:s.IsNan,KernelBackend:s.KernelBackend,LRN:s.LRN,LRNGrad:s.LRNGrad,LayerVariable:s.LayerVariable,LayersModel:s.LayersModel,LeakyRelu:s.LeakyRelu,Less:s.Less,LessEqual:s.LessEqual,LinSpace:s.LinSpace,Log:s.Log,Log1p:s.Log1p,LogSoftmax:s.LogSoftmax,LogicalAnd:s.LogicalAnd,LogicalNot:s.LogicalNot,LogicalOr:s.LogicalOr,LogicalXor:s.LogicalXor,LowerBound:s.LowerBound,MathBackendCPU:s.MathBackendCPU,MathBackendWebGL:s.MathBackendWebGL,MatrixBandPart:s.MatrixBandPart,Max:s.Max,MaxPool:s.MaxPool,MaxPool3D:s.MaxPool3D,MaxPool3DGrad:s.MaxPool3DGrad,MaxPoolGrad:s.MaxPoolGrad,MaxPoolWithArgmax:s.MaxPoolWithArgmax,Maximum:s.Maximum$1,Mean:s.Mean,Min:s.Min,Minimum:s.Minimum$1,MirrorPad:s.MirrorPad,Mod:s.Mod,MomentumOptimizer:s.MomentumOptimizer,Multinomial:s.Multinomial,Multiply:s.Multiply$1,Neg:s.Neg,NonMaxSuppressionV3:s.NonMaxSuppressionV3,NonMaxSuppressionV4:s.NonMaxSuppressionV4,NonMaxSuppressionV5:s.NonMaxSuppressionV5,NotEqual:s.NotEqual,OP_SCOPE_SUFFIX:s.OP_SCOPE_SUFFIX,OneHot:s.OneHot,OnesLike:s.OnesLike,Optimizer:s.Optimizer,OptimizerConstructors:s.OptimizerConstructors,Pack:s.Pack,PadV2:s.PadV2,Pool:s.Pool,Pow:s.Pow,Prelu:s.Prelu,Prod:s.Prod,RMSPropOptimizer:s.RMSPropOptimizer,RNN:s.RNN,RaggedGather:s.RaggedGather,RaggedRange:s.RaggedRange,RaggedTensorToTensor:s.RaggedTensorToTensor,Range:s.Range,get Rank(){return s.Rank},Real:s.Real,RealDiv:s.RealDiv,Reciprocal:s.Reciprocal,get Reduction(){return s.Reduction},Relu:s.Relu,Relu6:s.Relu6,Reshape:s.Reshape$1,ResizeBilinear:s.ResizeBilinear,ResizeBilinearGrad:s.ResizeBilinearGrad,ResizeNearestNeighbor:s.ResizeNearestNeighbor,ResizeNearestNeighborGrad:s.ResizeNearestNeighborGrad,Reverse:s.Reverse,RotateWithOffset:s.RotateWithOffset,Round:s.Round,Rsqrt:s.Rsqrt,SGDOptimizer:s.SGDOptimizer,ScatterNd:s.ScatterNd,SearchSorted:s.SearchSorted,Select:s.Select,Selu:s.Selu,Sequential:s.Sequential,Sigmoid:s.Sigmoid,Sign:s.Sign,Sin:s.Sin,Sinh:s.Sinh,Slice:s.Slice,Softmax:s.Softmax$1,Softplus:s.Softplus,SpaceToBatchND:s.SpaceToBatchND,SparseFillEmptyRows:s.SparseFillEmptyRows,SparseReshape:s.SparseReshape,SparseSegmentMean:s.SparseSegmentMean,SparseSegmentSum:s.SparseSegmentSum,SparseToDense:s.SparseToDense,SplitV:s.SplitV,Sqrt:s.Sqrt,Square:s.Square,SquaredDifference:s.SquaredDifference,StaticRegexReplace:s.StaticRegexReplace,Step:s.Step,StridedSlice:s.StridedSlice,StringNGrams:s.StringNGrams,StringSplit:s.StringSplit,StringToHashBucketFast:s.StringToHashBucketFast,Sub:s.Sub,Sum:s.Sum,SymbolicTensor:s.SymbolicTensor,Tan:s.Tan,Tanh:s.Tanh,Tensor:s.Tensor,TensorBuffer:s.TensorBuffer,TensorScatterUpdate:s.TensorScatterUpdate,Tile:s.Tile,TopK:s.TopK,Transform:s.Transform,Transpose:s.Transpose,Unique:s.Unique,Unpack:s.Unpack,UnsortedSegmentSum:s.UnsortedSegmentSum,UpperBound:s.UpperBound,Variable:s.Variable,ZerosLike:s.ZerosLike,_FusedMatMul:s._FusedMatMul,abs:s.abs,acos:s.acos,acosh:s.acosh,add:s.add,addN:Ve,all:s.all,any:s.any,argMax:s.argMax,argMin:s.argMin,asin:s.asin,asinh:s.asinh,atan:s.atan,atan2:s.atan2,atanh:s.atanh,avgPool:s.avgPool,avgPool3d:s.avgPool3d,backend:s.backend,backend_util:s.backend_util,basicLSTMCell:Be,batchNorm:s.batchNorm,batchNorm2d:s.batchNorm2d,batchNorm3d:s.batchNorm3d,batchNorm4d:s.batchNorm4d,batchToSpaceND:s.batchToSpaceND,bincount:s.bincount,bitwiseAnd:je,booleanMaskAsync:wt,broadcastArgs:qe,broadcastTo:s.broadcastTo,broadcast_util:s.broadcast_util,browser:s.browser,buffer:s.buffer,callbacks:wn,cast:s.cast,ceil:s.ceil,clipByValue:s.clipByValue,clone:s.clone,complex:s.complex,concat:s.concat,concat1d:s.concat1d,concat2d:s.concat2d,concat3d:s.concat3d,concat4d:s.concat4d,constraints:ys,conv1d:s.conv1d,conv2d:s.conv2d$1,conv2dTranspose:s.conv2dTranspose,conv3d:s.conv3d,conv3dTranspose:s.conv3dTranspose,copyRegisteredKernels:s.copyRegisteredKernels,cos:s.cos,cosh:s.cosh,cosineWindow:s.cosineWindow,cumprod:s.cumprod,cumsum:s.cumsum,customGrad:s.customGrad,data:lo,denseBincount:s.denseBincount,deprecationWarn:s.deprecationWarn,depthToSpace:s.depthToSpace,depthwiseConv2d:s.depthwiseConv2d,deregisterOp:vn,device_util:s.device_util,diag:He,dilation2d:s.dilation2d,disableDeprecationWarnings:s.disableDeprecationWarnings,dispose:s.dispose,disposeVariables:s.disposeVariables,div:s.div,divNoNan:s.divNoNan,dot:s.dot,dropout:s.dropout,einsum:s.einsum,elu:s.elu,enableDebugMode:s.enableDebugMode,enableProdMode:s.enableProdMode,enclosingPowerOfTwo:s.enclosingPowerOfTwo,engine:s.engine,ensureShape:Ge,env:s.env,equal:s.equal,erf:s.erf,euclideanNorm:s.euclideanNorm,exp:s.exp,expandDims:s.expandDims,expm1:s.expm1,eye:s.eye,fft:s.fft,fill:s.fill,findBackend:s.findBackend,findBackendFactory:s.findBackendFactory,floor:s.floor,floorDiv:s.floorDiv,forceHalfFloat:s.forceHalfFloat,fused:_t,gather:s.gather,gatherND:Et,gather_util:s.gather_nd_util,getBackend:s.getBackend,getGradient:s.getGradient,getKernel:s.getKernel,getKernelsForBackend:s.getKernelsForBackend,gpgpu_util:s.gpgpu_util,grad:s.grad,grads:s.grads,greater:s.greater,greaterEqual:s.greaterEqual,ifft:s.ifft,imag:s.imag,image:s.image,inTopKAsync:At,initializers:$s,input:Dt,io:ye,irfft:s.irfft,isFinite:s.isFinite,isInf:s.isInf,isNaN:s.isNaN,keep:s.keep,kernel_impls:cs,layers:Za,leakyRelu:s.leakyRelu,less:s.less,lessEqual:s.lessEqual,linalg:s.linalg,linspace:We,loadGraphModel:Ji,loadGraphModelSync:Qi,loadLayersModel:s.loadLayersModel,localResponseNormalization:s.localResponseNormalization,log:s.log,log1p:s.log1p,logSigmoid:s.logSigmoid,logSoftmax:s.logSoftmax,logSumExp:s.logSumExp,logicalAnd:s.logicalAnd,logicalNot:s.logicalNot,logicalOr:s.logicalOr,logicalXor:s.logicalXor,losses:s.losses,lowerBound:Ue,matMul:s.matMul,math:ps,max:s.max,maxPool:s.maxPool,maxPool3d:s.maxPool3d,maxPoolWithArgmax:Ke,maximum:s.maximum,mean:s.mean,memory:s.memory,meshgrid:Je,metrics:hn,min:s.min,minimum:s.minimum,mirrorPad:s.mirrorPad,mod:s.mod,model:xs,models:fn,moments:s.moments,movingAverage:St,mul:s.mul,multiRNNCell:Qe,multinomial:Xe,neg:s.neg,nextFrame:s.nextFrame,norm:s.norm,notEqual:s.notEqual,oneHot:s.oneHot,ones:s.ones,onesLike:s.onesLike,op:s.op,outerProduct:Ze,pad:s.pad,pad1d:Ye,pad2d:Me,pad3d:et,pad4d:tt,pool:s.pool,pow:s.pow,prelu:s.prelu,print:s.print,prod:s.prod,profile:s.profile,raggedGather:rt,raggedRange:st,raggedTensorToTensor:at,rand:nt,randomGamma:ut,randomNormal:s.randomNormal,randomStandardNormal:lt,randomUniform:s.randomUniform,randomUniformInt:pt,range:s.range,ready:s.ready,real:s.real,reciprocal:s.reciprocal,registerBackend:s.registerBackend,registerCallbackConstructor:Ls,registerGradient:s.registerGradient,registerKernel:s.registerKernel,registerOp:Sn,regularizers:bn,relu:s.relu,relu6:s.relu6,removeBackend:s.removeBackend,reshape:s.reshape,reverse:s.reverse,reverse1d:ct,reverse2d:mt,reverse3d:dt,reverse4d:ht,rfft:s.rfft,round:s.round,rsqrt:s.rsqrt,scalar:s.scalar,scatterND:vt,scatter_util:s.scatter_nd_util,searchSorted:X,selu:s.selu,separableConv2d:s.separableConv2d,sequential:zs,serialization:s.serialization,setBackend:s.setBackend,setPlatform:s.setPlatform,setWebGLContext:s.setWebGLContext,setdiff1dAsync:ft,shared:s.shared,sigmoid:s.sigmoid,sign:s.sign,signal:s.signal,sin:s.sin,sinh:s.sinh,slice:s.slice,slice1d:s.slice1d,slice2d:s.slice2d,slice3d:s.slice3d,slice4d:s.slice4d,slice_util:s.slice_util,softmax:s.softmax,softplus:s.softplus,spaceToBatchND:s.spaceToBatchND,sparse:s.sparse,sparseToDense:Ot,spectral:s.spectral,split:s.split,sqrt:s.sqrt,square:s.square,squaredDifference:s.squaredDifference,squeeze:s.squeeze,stack:s.stack,step:s.step,stridedSlice:s.stridedSlice,string:s.string,sub:s.sub,sum:s.sum,sumOutType:s.sumOutType,tan:s.tan,tanh:s.tanh,tensor:s.tensor,tensor1d:s.tensor1d,tensor2d:s.tensor2d,tensor3d:s.tensor3d,tensor4d:yt,tensor5d:gt,tensor6d:Nt,tensorScatterUpdate:bt,tensor_util:s.tensor_util,test_util:Cr,tidy:s.tidy,tile:s.tile,time:s.time,topk:s.topk,train:s.train,transpose:s.transpose,truncatedNormal:s.truncatedNormal,unique:s.unique,unregisterGradient:s.unregisterGradient,unregisterKernel:s.unregisterKernel,unsortedSegmentSum:s.unsortedSegmentSum,unstack:s.unstack,upcastType:s.upcastType,upperBound:Tt,util:s.util,valueAndGrad:s.valueAndGrad,valueAndGrads:s.valueAndGrads,variable:s.variable,variableGrads:s.variableGrads,version:co,version_converter:Bt,version_core:It,version_cpu:Xt,version_layers:s.version,version_webgl:Zt,webgl:s.webgl,webgl_util:s.webgl_util,where:s.where,whereAsync:fe,zeros:s.zeros,zerosLike:s.zerosLike},Symbol.toStringTag,{value:"Module"})),Yt=new E.Matrix4;Yt.compose(new E.Vector3,new E.Quaternion,new E.Vector3(.001,.001,.001));const ho=new E.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1);class Mt{constructor({container:e,imageTargetSrc:t,maxTrack:a,uiLoading:n="yes",uiScanning:i="yes",uiError:l="yes",filterMinCF:u=null,filterBeta:p=null,warmupTolerance:c=null,missTolerance:m=null,userDeviceId:d=null,environmentDeviceId:h=null}){this.container=e,this.imageTargetSrc=t,this.maxTrack=a,this.filterMinCF=u,this.filterBeta=p,this.warmupTolerance=c,this.missTolerance=m,this.ui=new rr.UI({uiLoading:n,uiScanning:i,uiError:l}),this.userDeviceId=d,this.environmentDeviceId=h,this.shouldFaceUser=!1,this.scene=new E.Scene,this.cssScene=new E.Scene,this.renderer=new E.WebGLRenderer({antialias:!0,alpha:!0}),this.cssRenderer=new tr.CSS3DRenderer({antialias:!0}),this.renderer.outputEncoding=E.sRGBEncoding,this.renderer.setPixelRatio(window.devicePixelRatio),this.camera=new E.PerspectiveCamera,this.anchors=[],this.renderer.domElement.style.position="absolute",this.cssRenderer.domElement.style.position="absolute",this.container.appendChild(this.renderer.domElement),this.container.appendChild(this.cssRenderer.domElement),window.addEventListener("resize",this.resize.bind(this))}async start(){this.ui.showLoading(),await this._startVideo(),await this._startAR()}stop(){this.controller.stopProcessVideo(),this.video.srcObject.getTracks().forEach(function(t){t.stop()}),this.video.remove()}switchCamera(){this.shouldFaceUser=!this.shouldFaceUser,this.stop(),this.start()}addAnchor(e){const t=new E.Group;t.visible=!1,t.matrixAutoUpdate=!1;const a={group:t,targetIndex:e,onTargetFound:null,onTargetLost:null,onTargetUpdate:null,css:!1,visible:!1};return this.anchors.push(a),this.scene.add(t),a}addCSSAnchor(e){const t=new E.Group;t.visible=!1,t.matrixAutoUpdate=!1;const a={group:t,targetIndex:e,onTargetFound:null,onTargetLost:null,onTargetUpdate:null,css:!0,visible:!1};return this.anchors.push(a),this.cssScene.add(t),a}_startVideo(){return new Promise((e,t)=>{if(this.video=document.createElement("video"),this.video.setAttribute("autoplay",""),this.video.setAttribute("muted",""),this.video.setAttribute("playsinline",""),this.video.style.position="absolute",this.video.style.top="0px",this.video.style.left="0px",this.video.style.zIndex="-2",this.container.appendChild(this.video),!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){this.ui.showCompatibility(),t();return}const a={audio:!1,video:{}};this.shouldFaceUser?this.userDeviceId?a.video.deviceId={exact:this.userDeviceId}:a.video.facingMode="user":this.environmentDeviceId?a.video.deviceId={exact:this.environmentDeviceId}:a.video.facingMode="environment",navigator.mediaDevices.getUserMedia(a).then(n=>{this.video.addEventListener("loadedmetadata",()=>{this.video.setAttribute("width",this.video.videoWidth),this.video.setAttribute("height",this.video.videoHeight),e()}),this.video.srcObject=n}).catch(n=>{console.log("getUserMedia error",n),t()})})}_startAR(){return new Promise(async(e,t)=>{const a=this.video;this.container,this.controller=new s.Controller({inputWidth:a.videoWidth,inputHeight:a.videoHeight,filterMinCF:this.filterMinCF,filterBeta:this.filterBeta,warmupTolerance:this.warmupTolerance,missTolerance:this.missTolerance,maxTrack:this.maxTrack,onUpdate:i=>{if(i.type==="updateMatrix"){const{targetIndex:l,worldMatrix:u}=i;for(let c=0;c<this.anchors.length;c++)if(this.anchors[c].targetIndex===l){if(this.anchors[c].css?this.anchors[c].group.children.forEach(m=>{m.element.style.visibility=u===null?"hidden":"visible"}):this.anchors[c].group.visible=u!==null,u!==null){let m=new E.Matrix4;m.elements=[...u],m.multiply(this.postMatrixs[l]),this.anchors[c].css&&m.multiply(Yt),this.anchors[c].group.matrix=m}else this.anchors[c].group.matrix=ho;this.anchors[c].visible&&u===null&&(this.anchors[c].visible=!1,this.anchors[c].onTargetLost&&this.anchors[c].onTargetLost()),!this.anchors[c].visible&&u!==null&&(this.anchors[c].visible=!0,this.anchors[c].onTargetFound&&this.anchors[c].onTargetFound()),this.anchors[c].onTargetUpdate&&this.anchors[c].onTargetUpdate()}this.anchors.reduce((c,m)=>c||m.visible,!1)?this.ui.hideScanning():this.ui.showScanning()}}}),this.resize();const{dimensions:n}=await this.controller.addImageTargets(this.imageTargetSrc);this.postMatrixs=[];for(let i=0;i<n.length;i++){const l=new E.Vector3,u=new E.Quaternion,p=new E.Vector3,[c,m]=n[i];l.x=c/2,l.y=c/2+(m-c)/2,p.x=c,p.y=c,p.z=c;const d=new E.Matrix4;d.compose(l,u,p),this.postMatrixs.push(d)}await this.controller.dummyRun(this.video),this.ui.hideLoading(),this.ui.showScanning(),this.controller.processVideo(this.video),e()})}resize(){const{renderer:e,cssRenderer:t,camera:a,container:n,video:i}=this;if(!i)return;this.video.setAttribute("width",this.video.videoWidth),this.video.setAttribute("height",this.video.videoHeight);let l,u;const p=i.videoWidth/i.videoHeight,c=n.clientWidth/n.clientHeight;p>c?(u=n.clientHeight,l=u*p):(l=n.clientWidth,u=l/p);const m=this.controller.getProjectionMatrix(),d=this.controller.inputWidth/this.controller.inputHeight;let h;d>c?h=this.video.width/this.controller.inputWidth:h=this.video.height/this.controller.inputHeight;let f,b;d>c?(f=n.clientHeight,f*=h):(b=n.clientWidth,f=b/this.controller.inputWidth*this.controller.inputHeight,f*=h);let g=n.clientHeight/f;const y=2*Math.atan(1/m[5]*g)*180/Math.PI,N=m[14]/(m[10]-1),w=m[14]/(m[10]+1);m[5]/m[0],a.fov=y,a.near=N,a.far=w,a.aspect=n.clientWidth/n.clientHeight,a.updateProjectionMatrix(),i.style.top=-(u-n.clientHeight)/2+"px",i.style.left=-(l-n.clientWidth)/2+"px",i.style.width=l+"px",i.style.height=u+"px";const S=e.domElement,T=t.domElement;S.style.position="absolute",S.style.left=0,S.style.top=0,S.style.width=n.clientWidth+"px",S.style.height=n.clientHeight+"px",T.style.position="absolute",T.style.left=0,T.style.top=0,T.style.width=n.clientWidth+"px",T.style.height=n.clientHeight+"px",e.setSize(n.clientWidth,n.clientHeight),t.setSize(n.clientWidth,n.clientHeight)}}window.MINDAR||(window.MINDAR={});window.MINDAR.IMAGE||(window.MINDAR.IMAGE={});window.MINDAR.IMAGE.MindARThree=Mt;window.MINDAR.IMAGE.tf=mo;exports.MindARThree=Mt;
//# sourceMappingURL=mindar-image-three.prod.js.map
