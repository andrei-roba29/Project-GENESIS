/* Project GENESIS — WebGL globe + flat map renderer (zero dependencies).
 *
 * The globe is a textured sphere; the texture is a PNG rendered server-side
 * from the simulation grid (equirectangular). Drag rotates, wheel zooms,
 * click picks a geographic coordinate via ray-sphere intersection.
 */
(function () {
  "use strict";

  const VERT = `
    attribute vec3 aPos;
    attribute vec2 aUV;
    attribute vec3 aNormal;
    uniform mat4 uMVP;
    uniform mat4 uModel;
    varying vec2 vUV;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      vUV = aUV;
      vNormal = normalize(mat3(uModel) * aNormal);
      vec4 wp = uModel * vec4(aPos, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = uMVP * vec4(aPos, 1.0);
    }`;

  // NOTE: raw WebGL has no built-in uniforms (cameraPosition is a Three.js
  // convenience), so the camera position is passed in explicitly as uEye.
  const FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform vec3 uSunDir;
    uniform vec3 uEye;
    uniform float uGraticule;
    uniform float uShade;
    varying vec2 vUV;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      vec4 tex = texture2D(uTex, vUV);
      vec3 n = normalize(vNormal);
      vec3 viewDir = normalize(uEye - vWorldPos);
      float diff = max(dot(n, normalize(uSunDir)), 0.0);
      float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.5);
      // uShade at 0 gives a flat, fully readable map (best for inspecting
      // data layers); at 1 the planet gets full directional sunlight.
      float lit = mix(1.0, 0.30 + 0.90 * diff, uShade);
      vec3 col = tex.rgb * lit + vec3(0.15, 0.3, 0.55) * rim * 0.6 * mix(0.25, 1.0, uShade);
      if (uGraticule > 0.5) {
        vec2 g = abs(fract(vUV * vec2(24.0, 12.0)) - 0.5);
        float line = step(0.47, max(g.x, g.y));
        col = mix(col, vec3(1.0), line * 0.18);
      }
      gl_FragColor = vec4(col, 1.0);
    }`;

  /** y = rotX(rx) * rotY(ry) * p, matching draw()'s model matrix. */
  function applyModelRotation(rx, ry, p) {
    const cX = Math.cos(rx), sX = Math.sin(rx);
    const cY = Math.cos(ry), sY = Math.sin(ry);
    return [
      cY * p[0] + sY * p[2],
      sX * sY * p[0] + cX * p[1] - sX * cY * p[2],
      -cX * sY * p[0] + sX * p[1] + cX * cY * p[2],
    ];
  }

  /** Wrap an angle into (-pi, pi]. */
  function normaliseAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a <= -Math.PI) a += 2 * Math.PI;
    return a;
  }

  function mat4Identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  function mat4Multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
    }
    return o;
  }
  function mat4Perspective(fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  }
  function mat4RotX(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);
  }
  function mat4RotY(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]);
  }
  function mat4Translate(x, y, z) {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]);
  }

  class Globe {
    constructor(canvas, onPick) {
      this.canvas = canvas;
      this.onPick = onPick;
      this.gl = canvas.getContext("webgl", { antialias: true });
      if (!this.gl) { canvas.style.display = "none"; return; }
      this.rotX = 0.35; this.rotY = 0.6; this.zoom = 2.6;
      this.minZoom = 1.02; this.maxZoom = 7.0;   // 1.02 = just above the surface
      this.pickLat = 20; this.pickLon = 30;
      this._flight = null;
      this.texture = null;
      this.graticule = true;
      this._initGL();
      this._bindInput();
      this._resize();
      window.addEventListener("resize", () => this._resize());
    }

    _initGL() {
      const gl = this.gl;
      const compile = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src); gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
        return sh;
      };
      const prog = gl.createProgram();
      const vs = compile(gl.VERTEX_SHADER, VERT), fs = compile(gl.FRAGMENT_SHADER, FRAG);
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        // Surface the reason instead of silently drawing nothing.
        this.error = gl.getProgramInfoLog(prog) || "shader link failed";
        console.error("Genesis globe:", this.error);
        return;
      }
      this.prog = prog;
      gl.useProgram(prog);
      this.aPos = gl.getAttribLocation(prog, "aPos");
      this.aUV = gl.getAttribLocation(prog, "aUV");
      this.aNormal = gl.getAttribLocation(prog, "aNormal");
      this.uMVP = gl.getUniformLocation(prog, "uMVP");
      this.uModel = gl.getUniformLocation(prog, "uModel");
      this.uTex = gl.getUniformLocation(prog, "uTex");
      this.uSunDir = gl.getUniformLocation(prog, "uSunDir");
      this.uEye = gl.getUniformLocation(prog, "uEye");
      this.uGraticule = gl.getUniformLocation(prog, "uGraticule");
      this.uShade = gl.getUniformLocation(prog, "uShade");
      this.shade = 0.55;
      gl.enableVertexAttribArray(this.aPos);
      gl.enableVertexAttribArray(this.aUV);
      gl.enableVertexAttribArray(this.aNormal);

      // Sphere geometry (unit sphere, equirectangular UVs).
      const latSeg = 72, lonSeg = 144;
      const pos = [], uv = [], norm = [], idx = [];
      for (let i = 0; i <= latSeg; i++) {
        const lat = -90 + 180 * i / latSeg;
        const phi = lat * Math.PI / 180;
        for (let j = 0; j <= lonSeg; j++) {
          const lon = -180 + 360 * j / lonSeg;
          const lam = lon * Math.PI / 180;
          const x = Math.cos(phi) * Math.cos(lam), y = Math.sin(phi), z = Math.cos(phi) * Math.sin(lam);
          pos.push(x, y, z); norm.push(x, y, z);
          uv.push(j / lonSeg, (90 - lat) / 180);
        }
      }
      for (let i = 0; i < latSeg; i++) for (let j = 0; j < lonSeg; j++) {
        const a = i * (lonSeg + 1) + j, b = a + 1, c = a + lonSeg + 1, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
      const buf = (data, size) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
        return b;
      };
      this.posBuf = buf(pos); this.uvBuf = buf(uv); this.normBuf = buf(norm);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
      this.indexBuffer = ib;
      this.indexCount = idx.length;
      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([20, 70, 140, 255]));
    }

    setTextureImage(img) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    }

    _resize() {
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    }

    _bindInput() {
      const c = this.canvas;
      let dragging = false, lx = 0, ly = 0, moved = 0;
      c.addEventListener("pointerdown", e => { dragging = true; lx = e.clientX; ly = e.clientY; moved = 0; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointermove", e => {
        if (!dragging) return;
        const dx = e.clientX - lx, dy = e.clientY - ly;
        moved += Math.abs(dx) + Math.abs(dy);
        // One screen-width drag sweeps a constant arc *on the surface*, so
        // close-up navigation stays precise instead of flying past.
        const sens = 0.008 * Math.max(0.04, this.zoom - 1.0) / 1.6;
        this.rotY += dx * sens;
        this.rotX = Math.max(-1.45, Math.min(1.45, this.rotX + dy * sens));
        lx = e.clientX; ly = e.clientY;
      });
      c.addEventListener("pointerup", e => {
        dragging = false;
        if (moved < 5) this._pick(e);
      });
      c.addEventListener("wheel", e => {
        e.preventDefault();
        this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * (e.deltaY < 0 ? 1.18 : 0.85)));
      }, { passive: false });
    }

    _pick(e) {
      const rect = this.canvas.getBoundingClientRect();
      const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ny = 1 - ((e.clientY - rect.top) / rect.height) * 2;
      // Ray from the camera (origin in view space) through the NDC point.
      // Aspect must come from the drawing buffer, not CSS pixels, otherwise
      // picking drifts on high-DPI screens.
      const aspect = (this.canvas.width || rect.width) / (this.canvas.height || rect.height);
      const f = 1 / Math.tan(0.6); // fovy 1.2 rad
      const dir = [nx * aspect / f, ny / f, -1];
      const len = Math.hypot(...dir);
      const d = dir.map(v => v / len);
      // Sphere centre in view space is (0, 0, -zoom). Solve |t*d - C|^2 = 1.
      const C = [0, 0, -this.zoom];
      const b = d[0]*C[0] + d[1]*C[1] + d[2]*C[2];
      const cc = C[0]**2 + C[1]**2 + C[2]**2 - 1;
      const disc = b*b - cc;
      if (disc < 0) return;
      const t = b - Math.sqrt(disc);
      // Back to model space: undo view translation, then inverse rotations.
      let x = t*d[0], y = t*d[1], z = t*d[2] + this.zoom;
      const cx = Math.cos(-this.rotX), sx = Math.sin(-this.rotX);
      let y1 = cx*y - sx*z, z1 = sx*y + cx*z; y = y1; z = z1;
      const cy = Math.cos(-this.rotY), sy = Math.sin(-this.rotY);
      let x1 = cy*x + sy*z, z2 = -sy*x + cy*z; x = x1; z = z2;
      const lat = Math.asin(Math.max(-1, Math.min(1, y))) * 180 / Math.PI;
      const lon = Math.atan2(z, x) * 180 / Math.PI;
      this.pickLat = lat; this.pickLon = lon;
      this.onPick(lat, lon);
    }

    /* Rotate so a geographic point sits dead-centre, facing the camera.
     *
     * Derived from the model matrix M = rotX(a) * rotY(b) above:
     *   row0 = (cos b, 0, sin b)                -> puts longitude at centre
     *   row1 = (sin a sin b, cos a, -sin a cos b) -> sets pitch
     * Requiring M*p = (0, 0, 1) gives b = atan2(px, -pz) and then a from the
     * pitch residual. Solving it numerically on the pitch residual keeps this
     * correct even if the rotation convention changes.
     */
    focusOn(lat, lon) {
      // The two centring solutions are exactly pi apart, so one always lies
      // inside +-pi/2; pick that one, then cap the pitch.
      const phi = lat * Math.PI / 180, lam = lon * Math.PI / 180;
      const p = [Math.cos(phi) * Math.cos(lam), Math.sin(phi), Math.cos(phi) * Math.sin(lam)];
      this.rotY = Math.atan2(p[0], -p[2]);
      const proj = (a) => applyModelRotation(a, this.rotY, p);
      const refine = (start) => {
        let a = start;
        for (let i = 0; i < 40; i++) {
          const v = proj(a);
          if (Math.abs(v[1]) < 1e-13) break;
          const h = 1e-7, v2 = proj(a + h);
          const d = (v2[1] - v[1]) / h;
          if (Math.abs(d) < 1e-12) break;
          a -= v[1] / d;
        }
        return a;
      };
      // Two solutions differ by pi; pick the one whose pivot stays closest to
      // the vertical so the camera does not end up underneath the planet.
      const a1 = normaliseAngle(refine(Math.atan2(p[1], Math.hypot(p[0], p[2]))));
      const a2 = normaliseAngle(a1 + Math.PI);
      // Cap the pitch so the camera never rolls past the horizon; targets very
      // near a pole end up slightly off-centre, which is the correct trade.
      this.rotX = Math.max(-1.35, Math.min(1.35, Math.abs(a1) <= Math.abs(a2) ? a1 : a2));
      this.pickLat = lat; this.pickLon = lon;
    }

    draw() {
      const gl = this.gl;
      if (!gl || !this.prog) return;
      this._resize();
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.02, 0.04, 0.08, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      const aspect = this.canvas.width / this.canvas.height;
      // Near plane tracks altitude so a close-up of the surface does not clip.
      const near = Math.max(0.0005, (this.zoom - 1.0) * 0.05);
      const proj = mat4Perspective(1.2, aspect, near, this.zoom + 5.0);
      const model = mat4Multiply(mat4RotX(this.rotX), mat4RotY(this.rotY));
      const view = mat4Translate(0, 0, -this.zoom);
      const mvp = mat4Multiply(proj, mat4Multiply(view, model));
      gl.useProgram(this.prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.vertexAttribPointer(this.aUV, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.normBuf);
      gl.vertexAttribPointer(this.aNormal, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
      gl.uniformMatrix4fv(this.uMVP, false, mvp);
      gl.uniformMatrix4fv(this.uModel, false, model);
      gl.uniform3f(this.uSunDir, 0.6, 0.35, 0.7);
      gl.uniform3f(this.uEye, 0, 0, this.zoom);
      gl.uniform1f(this.uGraticule, this.graticule ? 1 : 0);
      gl.uniform1f(this.uShade, this.shade);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.uTex, 0);
      gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    }
  }

  /* ---------------- flat equirectangular map ---------------- */
  class FlatMap {
    constructor(canvas, onPick) {
      this.canvas = canvas;
      this.onPick = onPick;
      this.ctx = canvas.getContext("2d");
      this.img = null;
      this.scale = 1; this.ox = 0; this.oy = 0;
      this._bindInput();
    }
    setImage(img) { this.img = img; this.fit(); }
    fit() {
      const c = this.canvas;
      if (!this.img) return;
      this.scale = Math.min(c.width / this.img.width, c.height / this.img.height);
      this.ox = (c.width - this.img.width * this.scale) / 2;
      this.oy = (c.height - this.img.height * this.scale) / 2;
    }
    _bindInput() {
      const c = this.canvas;
      let dragging = false, lx = 0, ly = 0, moved = 0;
      c.addEventListener("pointerdown", e => { dragging = true; lx = e.clientX; ly = e.clientY; moved = 0; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointermove", e => {
        if (!dragging) return;
        const dx = e.clientX - lx, dy = e.clientY - ly;
        moved += Math.abs(dx) + Math.abs(dy);
        this.ox += dx; this.oy += dy; lx = e.clientX; ly = e.clientY;
      });
      c.addEventListener("pointerup", e => {
        dragging = false;
        if (moved < 5) {
          const rect = c.getBoundingClientRect();
          const px = (e.clientX - rect.left - this.ox) / this.scale;
          const py = (e.clientY - rect.top - this.oy) / this.scale;
          if (this.img && px >= 0 && py >= 0 && px <= this.img.width && py <= this.img.height) {
            const lon = px / this.img.width * 360 - 180;
            const lat = 90 - py / this.img.height * 180;
            this.pickLat = lat; this.pickLon = lon;
            this.onPick(lat, lon);
          }
        }
      });
      c.addEventListener("wheel", e => {
        e.preventDefault();
        const rect = c.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const f = e.deltaY < 0 ? 1.15 : 0.87;
        this.ox = mx - (mx - this.ox) * f;
        this.oy = my - (my - this.oy) * f;
        this.scale *= f;
      }, { passive: false });
    }
    draw() {
      const c = this.canvas, ctx = this.ctx;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (c.width !== c.clientWidth * dpr) { c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr; }
      ctx.fillStyle = "#05080f";
      ctx.fillRect(0, 0, c.width, c.height);
      if (this.img) {
        ctx.imageSmoothingEnabled = this.scale < 3;
        ctx.drawImage(this.img, this.ox, this.oy, this.img.width * this.scale, this.img.height * this.scale);
      }
    }
  }


  /* ---------------- 3D ground view (terrain close-up) ----------------
   * Drawn from the real simulation heightfield (bilinear-sampled and given
   * fractal micro-relief server-side), so zooming toward the surface shows an
   * actual landscape rather than a stretched texture. Drag orbits, wheel
   * dollies; the camera can descend into valleys.
   */
  /* ---------------- 3D ground view (Age of Empires III style) ----------------
   * A low, oblique camera looking across a landscape built from the real
   * simulation heightfield. Left-drag pans, right/shift-drag orbits, wheel
   * zooms. Distance fog melts the patch edge into the sky so the world reads
   * as continuing beyond the loaded tile.
   */
  const GROUND_VERT = `
    attribute vec3 aPos;
    attribute vec2 aUV;
    uniform mat4 uMVP;
    varying vec2 vUV;
    varying vec3 vPos;
    void main() {
      vUV = aUV;
      vPos = aPos;
      gl_Position = uMVP * vec4(aPos, 1.0);
    }`;

  const GROUND_FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform sampler2D uCloudTex;
    uniform vec3 uEye;
    uniform vec3 uFogColor;
    uniform vec2 uFogRange;
    uniform float uWaterZ;
    uniform float uShadowAmt;
    varying vec2 vUV;
    varying vec3 vPos;
    void main() {
      vec4 tex = texture2D(uTex, vUV);
      vec3 col = tex.rgb;
      // Terrain shade from the cloud field directly overhead.
      float cloud = texture2D(uCloudTex, vUV).a;
      col *= 1.0 - uShadowAmt * cloud;
      // Water sits below sea level: tint it and darken with depth.
      if (vPos.z < uWaterZ) {
        float depth = clamp((uWaterZ - vPos.z) * 2.5, 0.0, 0.65);
        col = mix(col, vec3(0.05, 0.24, 0.42), 0.45 + depth);
      }
      // Aerial perspective: distant ground fades into the sky colour.
      float d = distance(uEye, vPos);
      float f = clamp((d - uFogRange.x) / max(uFogRange.y - uFogRange.x, 1e-3), 0.0, 1.0);
      f = f * f;
      gl_FragColor = vec4(mix(col, uFogColor, f * 0.9), 1.0);
    }`;

  const SKY_VERT = `
    attribute vec2 aNDC;
    varying float vY;
    void main() {
      vY = aNDC.y * 0.5 + 0.5;
      gl_Position = vec4(aNDC, 0.9999, 1.0);
    }`;

  const SKY_FRAG = `
    precision mediump float;
    varying float vY;
    uniform vec3 uSkyTop;
    uniform vec3 uSkyHorizon;
    void main() {
      float t = pow(clamp(vY, 0.0, 1.0), 0.7);
      gl_FragColor = vec4(mix(uSkyHorizon, uSkyTop, t), 1.0);
    }`;

  /* Vegetation billboards: alpha-tested so leaves have shape, tinted by a
   * per-instance colour attribute so one forest is not one flat green. */
  // Real-world sizes for the plants behind the ground view's decor.
  // Trees ~20 m tall and ~7 m across, shrubs and grass scaled down from there.
  const PLANT_H_M = 20.0;
  const PLANT_W_M = 7.0;

  const DECOR_VERT = `
    attribute vec3 aPos;
    attribute vec2 aUV;
    attribute float aShade;
    uniform mat4 uMVP;
    varying vec2 vUV;
    varying float vShade;
    varying vec3 vPos;
    void main() {
      vUV = aUV;
      vShade = aShade;
      vPos = aPos;
      gl_Position = uMVP * vec4(aPos, 1.0);
    }`;

  const DECOR_FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform vec3 uEye;
    uniform vec3 uFogColor;
    uniform vec2 uFogRange;
    uniform vec3 uTint;
    varying vec2 vUV;
    varying float vShade;
    varying vec3 vPos;
    void main() {
      vec4 tex = texture2D(uTex, vUV);
      if (tex.a < 0.35) discard;
      // Per-instance shade keeps a stand from looking like one flat colour.
      vec3 col = tex.rgb * uTint * vShade;
      float d = distance(uEye, vPos);
      float f = clamp((d - uFogRange.x) / max(uFogRange.y - uFogRange.x, 1e-3), 0.0, 1.0);
      gl_FragColor = vec4(mix(col, uFogColor, f * f * 0.9), 1.0);
    }`;

  /* Cloud deck: a translucent plane lifted above the terrain, drawn from the
   * simulation's own cloud-cover field. */
  const CLOUD_VERT = `
    attribute vec3 aPos;
    attribute vec2 aUV;
    uniform mat4 uMVP;
    varying vec2 vUV;
    varying vec3 vPos;
    void main() {
      vUV = aUV;
      vPos = aPos;
      gl_Position = uMVP * vec4(aPos, 1.0);
    }`;

  const CLOUD_FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform vec3 uFogColor;
    uniform vec3 uEye;
    uniform vec2 uFogRange;
    varying vec2 vUV;
    varying vec3 vPos;
    void main() {
      float a = texture2D(uTex, vUV).r;
      if (a < 0.04) discard;
      float d = distance(uEye, vPos);
      float f = clamp((d - uFogRange.x) / max(uFogRange.y - uFogRange.x, 1e-3), 0.0, 1.0);
      vec3 col = mix(vec3(0.97, 0.98, 1.0), uFogColor, f * 0.8);
      gl_FragColor = vec4(col, a * 0.62);
    }`;

  const SKY_HORIZON = [0.72, 0.80, 0.88];
  const SKY_TOP = [0.34, 0.55, 0.80];

  class GroundView {
    constructor(canvas, onChange) {
      this.canvas = canvas;
      this.onChange = onChange || (() => {});
      this.gl = canvas.getContext("webgl", { antialias: true });
      this.lat = 20; this.lon = 30; this.span = 24;
      // Camera state. AoE3-ish: low oblique angle, telephoto-ish FOV.
      this.yaw = 0;
      this.pitch = 0.58;          // radians above the ground plane
      this.dist = 1.7;
      this.offsetX = 0; this.offsetY = 0;   // pan within the loaded patch
      this.exaggeration = 4.0;
      this.mesh = null; this.texture = null; this.loading = false;
      this.error = null;
      this._heights = null;
      if (this.gl) { this._initGL(); this._bindInput(); }
    }

    _initGL() {
      const gl = this.gl;
      const sh = (t, src, label) => {
        const s = gl.createShader(t);
        gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          this.error = `${label}: ${gl.getShaderInfoLog(s)}`;
          console.error("GroundView shader:", this.error);
          return null;
        }
        return s;
      };
      const makeProgram = (vsSrc, fsSrc, label) => {
        const vs = sh(gl.VERTEX_SHADER, vsSrc, label + " vertex");
        const fs = sh(gl.FRAGMENT_SHADER, fsSrc, label + " fragment");
        if (!vs || !fs) return null;
        const prog = gl.createProgram();
        gl.attachShader(prog, vs); gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
          this.error = `${label}: ${gl.getProgramInfoLog(prog)}`;
          console.error("GroundView link:", this.error);
          return null;
        }
        return prog;
      };
      if (!(this.prog = makeProgram(GROUND_VERT, GROUND_FRAG, "terrain"))) return;
      if (!(this.skyProg = makeProgram(SKY_VERT, SKY_FRAG, "sky"))) return;
      if (!(this.decorProg = makeProgram(DECOR_VERT, DECOR_FRAG, "vegetation"))) return;
      if (!(this.cloudProg = makeProgram(CLOUD_VERT, CLOUD_FRAG, "clouds"))) return;

      gl.useProgram(this.prog);
      this.aPos = gl.getAttribLocation(this.prog, "aPos");
      this.aUV = gl.getAttribLocation(this.prog, "aUV");
      this.uMVP = gl.getUniformLocation(this.prog, "uMVP");
      this.uTex = gl.getUniformLocation(this.prog, "uTex");
      this.uEye = gl.getUniformLocation(this.prog, "uEye");
      this.uFogColor = gl.getUniformLocation(this.prog, "uFogColor");
      this.uFogRange = gl.getUniformLocation(this.prog, "uFogRange");
      this.uWaterZ = gl.getUniformLocation(this.prog, "uWaterZ");
      this.uCloudTex = gl.getUniformLocation(this.prog, "uCloudTex");
      this.uShadowAmt = gl.getUniformLocation(this.prog, "uShadowAmt");

      gl.useProgram(this.skyProg);
      this.skyNDC = gl.getAttribLocation(this.skyProg, "aNDC");
      this.skyTopLoc = gl.getUniformLocation(this.skyProg, "uSkyTop");
      this.skyHorizonLoc = gl.getUniformLocation(this.skyProg, "uSkyHorizon");

      gl.useProgram(this.decorProg);
      this.dPos = gl.getAttribLocation(this.decorProg, "aPos");
      this.dUV = gl.getAttribLocation(this.decorProg, "aUV");
      this.dShade = gl.getAttribLocation(this.decorProg, "aShade");
      this.dTint = gl.getUniformLocation(this.decorProg, "uTint");
      this.dMVP = gl.getUniformLocation(this.decorProg, "uMVP");
      this.dTex = gl.getUniformLocation(this.decorProg, "uTex");
      this.dEye = gl.getUniformLocation(this.decorProg, "uEye");
      this.dFogColor = gl.getUniformLocation(this.decorProg, "uFogColor");
      this.dFogRange = gl.getUniformLocation(this.decorProg, "uFogRange");

      gl.useProgram(this.cloudProg);
      this.cPos = gl.getAttribLocation(this.cloudProg, "aPos");
      this.cUV = gl.getAttribLocation(this.cloudProg, "aUV");
      this.cMVP = gl.getUniformLocation(this.cloudProg, "uMVP");
      this.cTex = gl.getUniformLocation(this.cloudProg, "uTex");
      this.cEye = gl.getUniformLocation(this.cloudProg, "uEye");
      this.cFogColor = gl.getUniformLocation(this.cloudProg, "uFogColor");
      this.cFogRange = gl.getUniformLocation(this.cloudProg, "uFogRange");
      this.cloudBuf = gl.createBuffer();
      this.cloudUVBuf = gl.createBuffer();
      this.cloudTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.cloudTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                    new Uint8Array([0, 0, 0, 0]));

      // Sky is a single full-screen quad.
      this.skyBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

      gl.useProgram(this.prog);
      this.posBuf = gl.createBuffer();
      this.uvBuf = gl.createBuffer();
      this.idxBuf = gl.createBuffer();

      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                    new Uint8Array([70, 95, 70, 255]));
      this._uploadSprites();
    }

    /* Load the heightfield and texture for a region. */
    async load(lat, lon, span, layer) {
      this.layer = layer || this.layer || "elevation";
      if (lat !== undefined) this.lat = lat;
      if (lon !== undefined) this.lon = lon;
      if (span !== undefined) this.span = span;
      const key = `${this.lat.toFixed(2)},${this.lon.toFixed(2)},${this.span.toFixed(2)},${this.layer}`;
      if (key === this._loadedKey) return;
      this._loadedKey = key;
      this.loading = true;
      this.error = null;
      try {
        const size = 240;   // 240^2 = 57,600 vertices: safely inside uint16
        const mesh = await fetch(
          `/api/region?lat=${this.lat}&lon=${this.lon}&span=${this.span}&size=${size}&layer=${this.layer}`
        ).then((r) => r.json());
        this.extentKm = (mesh.span / 360) * 2 * Math.PI * mesh.radius_km;
        this._lastMesh = mesh;
        this._lastSize = size;
        this._buildMesh(mesh, size);
        await this._loadTexture();
      } catch (e) {
        this.error = String(e && e.message ? e.message : e);
        console.warn("ground load failed", e);
      }
      this.loading = false;
      this.onChange(this);
    }

    /* Upload the texture into the existing GL texture object.
     * NOTE: returns nothing — the GL texture handle is created once in
     * _initGL and must never be replaced by a promise result. */
    _loadTexture() {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const gl = this.gl;
          if (!gl) return resolve();
          gl.bindTexture(gl.TEXTURE_2D, this.texture);
          // Source rows run north -> south; with no flip, row 0 lands at t=0,
          // which the mesh maps to its northern edge (see _buildMesh UVs).
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          resolve();
        };
        img.onerror = () => reject(new Error("region texture failed"));
        img.src = `/api/region/texture?lat=${this.lat}&lon=${this.lon}` +
                  `&span=${this.span}&w=512&h=512&layer=${this.layer}`;
      });
    }

    _buildMesh(mesh, size) {
      const gl = this.gl;
      if (!gl) return;
      const raw = atob(mesh.elevation);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      // Little-endian int16 decimetres, as sent by the server.
      const dm = new Int16Array(bytes.buffer);
      const heights = new Float32Array(dm.length);
      for (let i = 0; i < dm.length; i++) heights[i] = dm[i] / 10.0;
      this._heights = heights;
      this._meshSize = size;

      const n = size;
      // The patch occupies 2x2 model units, so one unit is extentKm/2 km.
      const extentKm = (mesh.span / 360) * 2 * Math.PI * mesh.radius_km;
      const kmPerUnit = Math.max(extentKm, 1e-6) / 2.0;
      // metres -> km -> model units, then exaggerated for legibility.
      const hScale = (1 / 1000.0) / kmPerUnit * this.exaggeration;

      const pos = new Float32Array(n * n * 3);
      const uv = new Float32Array(n * n * 2);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const k = i * n + j;
          // i runs north -> south, j runs west -> east.
          pos[k * 3] = (j / (n - 1)) * 2 - 1;
          pos[k * 3 + 1] = 1 - (i / (n - 1)) * 2;
          pos[k * 3 + 2] = heights[k] * hScale;
          uv[k * 2] = j / (n - 1);
          // Texture t=0 holds the image's first row, which is the north edge.
          uv[k * 2 + 1] = i / (n - 1);
        }
      }

      const use32 = n * n > 65535;
      const idx = use32
        ? new Uint32Array((n - 1) * (n - 1) * 6)
        : new Uint16Array((n - 1) * (n - 1) * 6);
      let p = 0;
      for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - 1; j++) {
          const a = i * n + j, b = a + 1, c = a + n, d = c + 1;
          idx[p++] = a; idx[p++] = c; idx[p++] = b;
          idx[p++] = b; idx[p++] = c; idx[p++] = d;
        }
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

      // The index type is decided here and reused verbatim by draw(), so the
      // upload and the draw call can never disagree.
      this.indexType = use32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      this.mesh = { n, count: idx.length, hScale, extentKm };
      this._storeRasters(mesh, n);
      this.waterZ = 0.0;
      this.minH = Infinity; this.maxH = -Infinity;
      for (let i = 0; i < heights.length; i++) {
        if (heights[i] < this.minH) this.minH = heights[i];
        if (heights[i] > this.maxH) this.maxH = heights[i];
      }
    }

    /* Decode the per-cell vegetation / cloud rasters sent alongside the mesh.
     * These come straight from the simulation, so what is drawn here matches
     * what the flora and climate models actually say about this patch. */
    _storeRasters(mesh, n) {
      const decode = (b64, len) => {
        if (!b64) return null;
        const raw = atob(b64);
        const out = new Uint8Array(len);
        for (let i = 0; i < len; i++) out[i] = raw.charCodeAt(i);
        return out;
      };
      const cells = n * n;
      this.rasters = {
        trees: decode(mesh.veg && mesh.veg.trees, cells),
        shrubs: decode(mesh.veg && mesh.veg.shrubs, cells),
        grasses: decode(mesh.veg && mesh.veg.grasses, cells),
        ferns: decode(mesh.veg && mesh.veg.ferns, cells),
        cloud: decode(mesh.cloud, cells),
        biome: decode(mesh.biome_ids, cells),
        n,
      };
      this.biomeNames = mesh.biome_names || [];
      // A new tile means new rasters: rebuild both the vegetation and the
      // cloud deck, otherwise the previous tile's clouds would linger.
      this._cloudBuilt = false;
      this._buildDecor();
    }

    /* Sample a raster at model-space x/y (nearest cell). */
    _rasterAt(r, x, y) {
      if (!r) return 0;
      const n = this.rasters.n;
      const j = Math.min(n - 1, Math.max(0, Math.round((x + 1) / 2 * (n - 1))));
      const i = Math.min(n - 1, Math.max(0, Math.round((1 - y) / 2 * (n - 1))));
      return r[i * n + j];
    }

    /* Build billboard geometry for trees, shrubs and grass tufts.
     *
     * Vegetation is placed by sampling the simulation's own cover field on a
     * jittered grid: cells with more tree cover get more trees, bare cells get
     * none. Nothing is scattered at random over the whole map. */
    _buildDecor() {
      const r = this.rasters;
      if (!r || !r.trees) return;
      const n = r.n;
      const slots = 96;            // candidate positions per axis
      const maxTrees = 7000, maxShrubs = 7000, maxGrass = 9000;
      const trees = [], shrubs = [], grass = [];
      // Deterministic hash so the same world always grows the same forest.
      let seed = 1234567;
      const rnd = () => {
        seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
        return seed / 0x7fffffff;
      };
      for (let gy = 0; gy < slots; gy++) {
        for (let gx = 0; gx < slots; gx++) {
          const x = (gx + rnd()) / slots * 2 - 1;
          const y = (gy + rnd()) / slots * 2 - 1;
          const j = Math.round((x + 1) / 2 * (n - 1));
          const i = Math.round((1 - y) / 2 * (n - 1));
          const k = i * n + j;
          // _heights is already in metres (decimetres were divided by 10 on decode).
          const h = this._heights[k];
          if (h <= 0) continue;                        // no vegetation in water
          const t = r.trees[k] / 255, s = r.shrubs[k] / 255, g = r.grasses[k] / 255;
          // Too steep => bare rock, so skip the steepest cells.
          const e = this._heights[k], eR = this._heights[k + 1] || e;
          if (eR - e > 260) continue;
          // Denser cover reads slightly darker; rnd() breaks up clones.
          if (t > 0.08 && rnd() < Math.min(1, t * 2.6) && trees.length < maxTrees) {
            // `w` is footprint, `sh` is height, both relative to the group.
            // Tall thin conifers read as forest; equal-sided blobs read as toys.
            trees.push({ x, y, h,
                         w: 0.55 + rnd() * 0.35,
                         sh: 0.75 + t * 0.55 + rnd() * 0.45,
                         broad: rnd() < 0.35,
                         shade: 0.72 + t * 0.35 + rnd() * 0.22 });
          }
          if (s > 0.10 && rnd() < Math.min(1, s * 1.7) && shrubs.length < maxShrubs) {
            shrubs.push({ x, y, h,
                          w: 0.45 + s * 0.5 + rnd() * 0.2,
                          sh: 0.30 + s * 0.35 + rnd() * 0.15,
                          shade: 0.80 + rnd() * 0.30 });
          }
          if (g > 0.12 && rnd() < Math.min(1, g * 1.6) && grass.length < maxGrass) {
            grass.push({ x, y, h,
                         w: 0.55 + rnd() * 0.4,
                         sh: 0.22 + g * 0.25 + rnd() * 0.12,
                         shade: 0.85 + rnd() * 0.28 });
          }
        }
      }
      // Two tree forms, so a forest is not one repeated silhouette.
      const conifer = trees.filter((t) => !t.broad);
      const broad = trees.filter((t) => t.broad);
      this.decor = { conifer, broad, shrubs, grass };
      this._buildDecorBuffers();
    }

    /* Smallest billboard height, in model units, that still shows up on
     * screen. A 20 m tree at a 24 deg span is a fraction of a pixel, so at
     * wide spans the decor stands in for a stand of trees; zoom in and it
     * becomes the real thing. Derived from the actual camera, so it stays
     * legible at every zoom without a magic constant. */
    _minPlantUnits() {
      const h = (this.canvas && this.canvas.height) || 600;
      const worldH = 2 * Math.max(this.dist || 1.7, 0.05) * Math.tan(0.78 / 2);
      const px = Math.max(9, h * 0.016);        // ~9 px tall at the smallest
      return (px * worldH) / h;
    }

    _buildDecorBuffers() {
      const gl = this.gl;
      if (!gl || !this.decor) return;
      // Rebuilding on zoom changes would leak GPU buffers, so release the old set.
      if (this.decorBufs) {
        for (const k of Object.keys(this.decorBufs)) {
          const b = this.decorBufs[k];
          if (!b) continue;
          if (b.buf) gl.deleteBuffer(b.buf);
          if (b.uv) gl.deleteBuffer(b.uv);
          if (b.shade) gl.deleteBuffer(b.shade);
        }
      }
      // One quad (two crossed billboards) per instance, built as a single
      // triangle list so a whole forest costs one draw call.
      const build = (items) => {
        if (!items.length) return { buf: null, count: 0 };
        const verts = new Float32Array(items.length * 6 * 3);
        const uvs = new Float32Array(items.length * 6 * 2);
        const shades = new Float32Array(items.length * 6);
        let p = 0, q = 0, r = 0;
        for (const it of items) {
          const z = it.h * (this.mesh ? this.mesh.hScale : 1);
          const sh = it.shade === undefined ? 1 : it.shade;
          // Plants are sized in real metres, so they are honest: zoom into a
          // small span and a 20 m tree really is a 20 m tree. At a wide span a
          // true-to-scale tree is a fraction of a pixel, so a floor keeps it
          // legible — it is a symbol for a forest stand at that zoom.
          const kmPerUnit = this.mesh && this.mesh.extentKm ? this.mesh.extentKm / 2 : 20;
          const exag = this.exaggeration || 1;
          const realH = (it.sh === undefined ? 1 : it.sh) * PLANT_H_M / 1000 / kmPerUnit * exag;
          const realW = (it.w === undefined ? 1 : it.w) * PLANT_W_M / 1000 / kmPerUnit * exag;
          const floorU = this._minPlantUnits();
          const hgt = Math.max(realH, floorU);
          const w = Math.max(realW, floorU * 0.45);
          // Two crossed quads = 4 triangles.
          const quads = [[-w, 0, 1], [w, 0, 1], [-w, 0, 0], [w, 0, 0]];
          for (let qi = 0; qi < 2; qi++) {
            const ax = qi === 0 ? 1 : 0, ay = qi === 0 ? 0 : 1;
            const pts = [
              [it.x - w * ax, it.y - w * ay, z],
              [it.x + w * ax, it.y + w * ay, z],
              [it.x - w * ax, it.y - w * ay, z + hgt],
              [it.x + w * ax, it.y + w * ay, z + hgt],
            ];
            const uvq = [[0, 0], [1, 0], [0, 1], [1, 1]];
            const order = [0, 1, 2, 2, 1, 3];
            for (const oi of order) {
              verts[p++] = pts[oi][0]; verts[p++] = pts[oi][1]; verts[p++] = pts[oi][2];
              uvs[q++] = uvq[oi][0]; uvs[q++] = uvq[oi][1];
              shades[r++] = sh;
            }
          }
        }
        const vbuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
        const ubuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, ubuf);
        gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
        const sbuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, sbuf);
        gl.bufferData(gl.ARRAY_BUFFER, shades, gl.STATIC_DRAW);
        return { buf: vbuf, uv: ubuf, shade: sbuf, count: items.length * 12 };
      };
      this.decorBufs = {
        conifer: build(this.decor.conifer),
        broad: build(this.decor.broad),
        shrubs: build(this.decor.shrubs),
        grass: build(this.decor.grass),
      };
    }

    /* Bilinear terrain height (model units) at a model-space x/y. */
    heightAt(x, y) {
      if (!this._heights) return 0;
      const n = this._meshSize;
      const u = Math.max(0, Math.min(1, (x + 1) / 2));
      const v = Math.max(0, Math.min(1, (1 - y) / 2));
      const fx = u * (n - 1), fy = v * (n - 1);
      const j0 = Math.min(n - 2, Math.floor(fx)), i0 = Math.min(n - 2, Math.floor(fy));
      const tx = fx - j0, ty = fy - i0;
      const h = this._heights;
      const a = h[i0 * n + j0], b = h[i0 * n + j0 + 1];
      const c = h[(i0 + 1) * n + j0], d = h[(i0 + 1) * n + j0 + 1];
      const metres = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
      return metres * (this.mesh ? this.mesh.hScale : 1);
    }

    _bindInput() {
      const c = this.canvas;
      let dragging = false, mode = "pan", lx = 0, ly = 0;
      c.addEventListener("contextmenu", (e) => e.preventDefault());
      c.addEventListener("pointerdown", (e) => {
        dragging = true;
        mode = (e.button === 2 || e.shiftKey || e.ctrlKey) ? "orbit" : "pan";
        lx = e.clientX; ly = e.clientY;
        c.setPointerCapture(e.pointerId);
      });
      c.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lx, dy = e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        if (mode === "orbit") {
          this.yaw += dx * 0.006;
          this.pitch = Math.max(0.22, Math.min(1.32, this.pitch + dy * 0.004));
        } else {
          // Pan along the ground, scaled by zoom so the grab point stays put.
          const k = this.dist * 0.0011;
          const cs = Math.cos(this.yaw), sn = Math.sin(this.yaw);
          this.offsetX -= (dx * cs - dy * sn) * k;
          this.offsetY -= (dx * sn + dy * cs) * k;
          this._maybeRecentre();
        }
      });
      c.addEventListener("pointerup", () => { dragging = false; });
      c.addEventListener("pointerleave", () => { dragging = false; });
      c.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.dist = Math.max(0.45, Math.min(3.6, this.dist * (e.deltaY < 0 ? 0.88 : 1.14)));
      }, { passive: false });
    }

    /* When the camera wanders past the loaded tile, fetch the tile the
     * landscape actually needs. Keeps panning continuous instead of hitting
     * the edge of the world after two drags. */
    _maybeRecentre() {
      const limit = 0.42;
      if (Math.abs(this.offsetX) < limit && Math.abs(this.offsetY) < limit) return;
      const dLat = this.offsetY * (this.span / 2);
      const cosLat = Math.max(0.2, Math.cos(this.lat * Math.PI / 180));
      const dLon = this.offsetX * (this.span / 2) / cosLat;
      let lat = this.lat + dLat;
      let lon = this.lon + dLon;
      lat = Math.max(-85, Math.min(85, lat));
      while (lon > 180) lon -= 360;
      while (lon < -180) lon += 360;
      this.offsetX = 0; this.offsetY = 0;
      this.lat = lat; this.lon = lon;
      this.load();
    }

    _eye() {
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      // yaw 0 looks north (+y); the camera sits behind the target.
      return [
        this.offsetX + Math.sin(this.yaw) * this.dist * cp,
        this.offsetY - Math.cos(this.yaw) * this.dist * cp,
        this.dist * sp,
      ];
    }

    /* Procedurally drawn billboard sprites: a conifer, a round broadleaf, a
     * shrub and a grass tuft. Generated on a 2D canvas at load time so the
     * project stays dependency-free and ships no image assets. */
    _makeSprite(kind) {
      const S = 64;
      const cv = document.createElement("canvas");
      cv.width = S; cv.height = S;
      const g = cv.getContext("2d");
      const cx = S / 2, base = S - 2;
      if (kind === "tree") {
        // trunk
        g.fillStyle = "#4a3524";
        g.fillRect(cx - 2, S * 0.55, 4, S * 0.45);
        // conifer: stacked triangles
        for (let i = 0; i < 4; i++) {
          const t = i / 4;
          const y = S * (0.56 - t * 0.34);
          const w = S * (0.42 - t * 0.30);
          g.fillStyle = i % 2 ? "#2f6b32" : "#377a3a";
          g.beginPath();
          g.moveTo(cx, y - S * 0.20);
          g.lineTo(cx - w / 2, y + S * 0.05);
          g.lineTo(cx + w / 2, y + S * 0.05);
          g.closePath(); g.fill();
        }
      } else if (kind === "broadleaf") {
        g.fillStyle = "#5b4227";
        g.fillRect(cx - 2.5, S * 0.5, 5, S * 0.5);
        const blobs = [[0.50, 0.34, 0.26], [0.34, 0.46, 0.19], [0.66, 0.46, 0.19], [0.50, 0.52, 0.20]];
        for (const [bx, by, br] of blobs) {
          g.fillStyle = "#3d7d3a";
          g.beginPath(); g.arc(S * bx, S * by, S * br, 0, Math.PI * 2); g.fill();
          g.fillStyle = "rgba(120,180,90,0.45)";
          g.beginPath(); g.arc(S * bx - S * br * 0.25, S * by - S * br * 0.25, S * br * 0.55, 0, Math.PI * 2); g.fill();
        }
      } else if (kind === "shrub") {
        for (const [bx, by, br] of [[0.5, 0.62, 0.24], [0.34, 0.7, 0.17], [0.66, 0.7, 0.17]]) {
          g.fillStyle = "#4a8a44";
          g.beginPath(); g.arc(S * bx, S * by, S * br, 0, Math.PI * 2); g.fill();
        }
      } else {
        // grass tuft: a few blades
        g.strokeStyle = "#6fa84c";
        g.lineWidth = 2.2;
        for (let i = 0; i < 7; i++) {
          const t = (i / 6 - 0.5);
          g.beginPath();
          g.moveTo(cx + t * S * 0.34, base);
          g.quadraticCurveTo(cx + t * S * 0.5, S * 0.72, cx + t * S * 0.62, S * 0.40);
          g.stroke();
        }
      }
      return cv;
    }

    _uploadSprites() {
      const gl = this.gl;
      const make = (canvas) => {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
        return tex;
      };
      this.sprites = {
        tree: make(this._makeSprite("tree")),
        broadleaf: make(this._makeSprite("broadleaf")),
        shrub: make(this._makeSprite("shrub")),
        grass: make(this._makeSprite("grass")),
      };
    }

    draw() {
      const gl = this.gl;
      if (!gl || !this.prog || !this.mesh) return;
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (c.width !== Math.round(c.clientWidth * dpr) ||
          c.height !== Math.round(c.clientHeight * dpr)) {
        c.width = Math.round(c.clientWidth * dpr);
        c.height = Math.round(c.clientHeight * dpr);
      }
      gl.viewport(0, 0, c.width, c.height);
      gl.clearColor(SKY_HORIZON[0], SKY_HORIZON[1], SKY_HORIZON[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // --- sky gradient (no depth interaction) ---
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.useProgram(this.skyProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
      gl.enableVertexAttribArray(this.skyNDC);
      gl.vertexAttribPointer(this.skyNDC, 2, gl.FLOAT, false, 0, 0);
      gl.uniform3fv(this.skyTopLoc, SKY_TOP);
      gl.uniform3fv(this.skyHorizonLoc, SKY_HORIZON);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);

      // --- terrain ---
      const eye = this._eye();
      // Keep the camera above the ground it is flying over.
      if (this._heights) {
        const ground = this.heightAt(eye[0], eye[1]);
        if (eye[2] < ground + 0.06) eye[2] = ground + 0.06;
      }
      const target = [this.offsetX, this.offsetY, 0.04];
      const aspect = c.width / c.height;
      const proj = mat4Perspective(0.78, aspect, 0.01, 60);
      const view = lookAt(eye, target, [0, 0, 1]);
      const mvp = mat4Multiply(proj, view);

      gl.useProgram(this.prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.enableVertexAttribArray(this.aPos);
      gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.enableVertexAttribArray(this.aUV);
      gl.vertexAttribPointer(this.aUV, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);

      gl.uniformMatrix4fv(this.uMVP, false, mvp);
      gl.uniform3fv(this.uEye, eye);
      gl.uniform3fv(this.uFogColor, SKY_HORIZON);
      // Fog starts beyond the near field and saturates at the patch edge.
      gl.uniform2f(this.uFogRange, this.dist * 1.15, this.dist * 2.9);
      gl.uniform1f(this.uWaterZ, this.waterZ);
      // Cloud shadows: the same field that draws the deck also darkens the
      // ground beneath it, so shade on the terrain matches the sky overhead.
      gl.uniform1f(this.uShadowAmt, this.showClouds === false ? 0.0 : 0.42);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.cloudTex);
      gl.uniform1i(this.uCloudTex, 1);
      gl.activeTexture(gl.TEXTURE0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.uTex, 0);
      gl.drawElements(gl.TRIANGLES, this.mesh.count, this.indexType, 0);

      // --- vegetation, clouds ---
      if (this.showVegetation !== false) {
        // Billboard size depends on camera distance, so re-derive it when the
        // player zooms far enough for the change to matter.
        if (this._decorDist === undefined ||
            Math.abs(this.dist - this._decorDist) > this._decorDist * 0.12) {
          this._decorDist = this.dist;
          if (this.decor) this._buildDecorBuffers();
        }
        this._drawDecor(mvp, eye);
      }
      if (this.showClouds !== false) this._drawClouds(mvp, eye);
    }

    /* Billboard vegetation. Alpha-tested quads sorted back-to-front by
     * distance so overlapping trees composite correctly. */
    _drawDecor(mvp, eye) {
      const gl = this.gl;
      if (!gl || !this.decorBufs || !this.sprites) return;
      gl.useProgram(this.decorProg);
      gl.uniformMatrix4fv(this.dMVP, false, mvp);
      gl.uniform3fv(this.dEye, eye);
      gl.uniform3fv(this.dFogColor, SKY_HORIZON);
      gl.uniform2f(this.dFogRange, this.dist * 1.15, this.dist * 2.9);
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(this.dTex, 0);

      // Depth writes are off so crossing billboards blend instead of clipping
      // each other, while the depth *test* still hides decor behind terrain.
      gl.depthMask(false);
      const passes = [
        ["conifer", this.sprites.tree, [0.82, 1.0, 0.84]],
        ["broad", this.sprites.broadleaf, [1.0, 1.02, 0.82]],
        ["shrubs", this.sprites.shrub, [0.95, 1.05, 0.9]],
        ["grass", this.sprites.grass, [1.05, 1.1, 0.9]],
      ];
      for (const [kind, sprite, tint] of passes) {
        const b = this.decorBufs[kind];
        if (!b || !b.count) continue;
        gl.bindTexture(gl.TEXTURE_2D, sprite);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
        gl.enableVertexAttribArray(this.dPos);
        gl.vertexAttribPointer(this.dPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.uv);
        gl.enableVertexAttribArray(this.dUV);
        gl.vertexAttribPointer(this.dUV, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.shade);
        gl.enableVertexAttribArray(this.dShade);
        gl.vertexAttribPointer(this.dShade, 1, gl.FLOAT, false, 0, 0);
        // One base tint per group; aShade varies it per plant.
        if (this.dTint) gl.uniform3f(this.dTint, tint[0], tint[1], tint[2]);
        gl.drawArrays(gl.TRIANGLES, 0, b.count);
      }
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }

    /* Translucent cloud deck from the simulation's cloud-cover field. */
    _drawClouds(mvp, eye) {
      const gl = this.gl;
      if (!gl || !this.rasters || !this.rasters.cloud) return;
      if (!this._cloudBuilt) this._buildCloudPlane();
      if (!this._cloudBuilt) return;
      gl.useProgram(this.cloudProg);
      gl.uniformMatrix4fv(this.cMVP, false, mvp);
      gl.uniform3fv(this.cEye, eye);
      gl.uniform3fv(this.cFogColor, SKY_HORIZON);
      gl.uniform2f(this.cFogRange, this.dist * 1.15, this.dist * 2.9);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cloudBuf);
      gl.enableVertexAttribArray(this.cPos);
      gl.vertexAttribPointer(this.cPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cloudUVBuf);
      gl.enableVertexAttribArray(this.cUV);
      gl.vertexAttribPointer(this.cUV, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.cloudTex);
      gl.uniform1i(this.cTex, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    _buildCloudPlane() {
      const gl = this.gl;
      const r = this.rasters;
      if (!gl || !r || !r.cloud) return;
      const n = r.n;
      // Upload the cloud field as a luminance alpha texture.
      // Cloud cover is a continuous 0..1 field, but as a solid deck it would
      // white out any humid region (some patches average 0.74). Only the
      // thicker part of the field becomes visible cloud, so the deck reads as
      // scattered cloud rather than fog over the whole map.
      const rgba = new Uint8Array(n * n * 4);
      const LO = 0.55, HI = 0.92;
      for (let i = 0; i < n * n; i++) {
        const raw = r.cloud[i] / 255;
        const v = Math.max(0, Math.min(1, (raw - LO) / (HI - LO)));
        rgba[i * 4] = 255; rgba[i * 4 + 1] = 255; rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = Math.round(v * 255);
      }
      gl.bindTexture(gl.TEXTURE_2D, this.cloudTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      // The deck sits above the tallest terrain so it never slices a peak.
      // It is a fixed altitude: tying it to the camera distance made the clouds
      // move whenever the player zoomed, which is not how a cloud base behaves.
      const peak = Math.max(this.maxH || 0, 0);
      const z = peak * (this.mesh ? this.mesh.hScale : 1) +
                (this.mesh ? this.mesh.hScale * 900 : 0.12);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cloudBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1, z, 1, -1, z, -1, 1, z, 1, 1, z,
      ]), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cloudUVBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0, 0, 1, 0, 0, 1, 1, 1,
      ]), gl.STATIC_DRAW);
      this._cloudBuilt = true;
    }
  }

  function lookAt(eye, target, up) {
    const f = norm(sub(target, eye));
    const s = norm(cross(f, up));
    const u = cross(s, f);
    return new Float32Array([
      s[0], u[0], -f[0], 0,
      s[1], u[1], -f[1], 0,
      s[2], u[2], -f[2], 0,
      -dot(s, eye), -dot(u, eye), dot(f, eye), 1,
    ]);
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  window.GenesisGlobe = { Globe, FlatMap, GroundView };
})();
