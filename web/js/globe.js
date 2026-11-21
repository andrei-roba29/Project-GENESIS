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
  const GROUND_VERT = `
    attribute vec3 aPos;
    attribute vec2 aUV;
    varying vec2 vUV;
    varying vec3 vPos;
    uniform mat4 uMVP;
    void main() {
      vUV = aUV;
      vPos = aPos;
      gl_Position = uMVP * vec4(aPos, 1.0);
    }`;

  const GROUND_FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform vec3 uSun;
    uniform float uWaterZ;
    varying vec2 vUV;
    varying vec3 vPos;
    void main() {
      vec4 tex = texture2D(uTex, vUV);
      // Gentle slope shading from the per-vertex normals is baked into the
      // texture; keep the surface readable with a soft ambient term.
      vec3 col = tex.rgb;
      if (vPos.z < uWaterZ) {
        float depth = clamp((uWaterZ - vPos.z) * 0.08, 0.0, 0.6);
        col = mix(col, vec3(0.06, 0.26, 0.48), 0.55 + depth);
      }
      gl_FragColor = vec4(col, 1.0);
    }`;

  class GroundView {
    constructor(canvas, onChange) {
      this.canvas = canvas;
      this.onChange = onChange || (() => {});
      this.gl = canvas.getContext("webgl", { antialias: true });
      this.lat = 20; this.lon = 30; this.span = 12;
      this.yaw = Math.PI * 0.25; this.pitch = 0.62; this.dist = 2.1;
      this.texture = null; this.mesh = null;
      this.exaggeration = 3.0; this.loading = false; this._pending = null;
      this._dirtyView = false;
      if (this.gl) { this._initGL(); this._bindInput(); }
    }

    _initGL() {
      const gl = this.gl;
      const sh = (t, src) => {
        const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          this.error = gl.getShaderInfoLog(s);
          console.error("GroundView shader:", this.error);
        }
        return s;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, GROUND_VERT));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, GROUND_FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        this.error = gl.getProgramInfoLog(prog);
        return;
      }
      this.prog = prog;
      gl.useProgram(prog);
      this.aPos = gl.getAttribLocation(prog, "aPos");
      this.aUV = gl.getAttribLocation(prog, "aUV");
      this.uMVP = gl.getUniformLocation(prog, "uMVP");
      this.uTex = gl.getUniformLocation(prog, "uTex");
      this.uWaterZ = gl.getUniformLocation(prog, "uWaterZ");
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
                    new Uint8Array([60, 90, 70, 255]));
    }

    /* Move to a new region and fetch its heightfield + texture. */
    async load(lat, lon, span, layer) {
      this.layer = layer || this.layer || "elevation";
      if (lat !== undefined) this.lat = lat;
      if (lon !== undefined) this.lon = lon;
      if (span !== undefined) this.span = span;
      const key = `${this.lat.toFixed(2)},${this.lon.toFixed(2)},${this.span.toFixed(2)},${this.layer}`;
      if (key === this._loadedKey) return;
      this._loadedKey = key;
      this.loading = true;
      try {
        const size = 192;
        const [mesh, tex] = await Promise.all([
          fetch(`/api/region?lat=${this.lat}&lon=${this.lon}&span=${this.span}&size=${size}&layer=${this.layer}`).then(r => r.json()),
          this._loadTexture(),
        ]);
        this.extentKm = mesh.span / 360 * 2 * Math.PI * mesh.radius_km;
        // Keep the heightfield so the relief slider can rebuild without a
        // server round-trip; set it before building so a rebuild always has it.
        this._lastMesh = mesh;
        this._lastSize = size;
        this._buildMesh(mesh, size);
        this.texture = tex;
      } catch (e) { console.warn("ground load failed", e); }
      this.loading = false;
      this.onChange(this);
    }

    _loadTexture() {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const gl = this.gl;
          if (!gl) return resolve(null);
          gl.bindTexture(gl.TEXTURE_2D, this.texture);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          resolve(true);
        };
        img.onerror = reject;
        // Cache-bust on layer + region so the browser never shows a stale patch.
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
      const dm = new Int16Array(bytes.buffer);
      const heights = Array.from(dm, v => v / 10.0);

      // Normalise the patch into a -1..1 square: 2 model units span the whole
      // region, so one model unit = extentKm/2 kilometres. Vertical relief is
      // scaled the same way and then multiplied by the exaggeration factor.
      const n = size;
      const extentKm = (mesh.span / 360) * 2 * Math.PI * mesh.radius_km;
      const kmPerUnit = Math.max(extentKm, 1e-6) / 2.0;
      const exag = this.exaggeration;
      // height_m -> km -> model units, exaggerated.
      const hScale = (1 / 1000.0) / kmPerUnit * exag;
      const pos = new Float32Array(n * n * 3);
      const uv = new Float32Array(n * n * 2);
      let minH = Infinity, maxH = -Infinity;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const k = i * n + j;
          const h = heights[k];
          minH = Math.min(minH, h); maxH = Math.max(maxH, h);
          pos[k * 3] = (j / (n - 1)) * 2 - 1;
          pos[k * 3 + 1] = 1 - (i / (n - 1)) * 2;
          pos[k * 3 + 2] = h * hScale;
          uv[k * 2] = j / (n - 1);
          uv[k * 2 + 1] = 1 - i / (n - 1);
        }
      }
      const idx = new Uint32Array((n - 1) * (n - 1) * 6);
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
      this.mesh = { n, count: idx.length, minH, maxH, extentKm };
      this.waterZ = 0.0;   // sea level in the same model units
      // Uint32 indices need an extension on WebGL1.
      this.uintIndex = gl.getExtension("OES_element_index_uint") && n * n > 65535;
    }

    _bindInput() {
      const c = this.canvas;
      let dragging = false, lx = 0, ly = 0;
      c.addEventListener("pointerdown", e => { dragging = true; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointermove", e => {
        if (!dragging) return;
        this.yaw -= (e.clientX - lx) * 0.006;
        this.pitch = Math.max(0.12, Math.min(1.45, this.pitch + (e.clientY - ly) * 0.005));
        lx = e.clientX; ly = e.clientY;
      });
      c.addEventListener("pointerup", () => { dragging = false; });
      c.addEventListener("wheel", e => {
        e.preventDefault();
        // Dollying below 1.0 descends into the landscape.
        this.dist = Math.max(0.35, Math.min(4.0, this.dist * (e.deltaY < 0 ? 0.88 : 1.12)));
      }, { passive: false });
    }

    draw() {
      const gl = this.gl;
      if (!gl || !this.prog || !this.mesh) return;
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (c.width !== c.clientWidth * dpr) { c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr; }
      gl.viewport(0, 0, c.width, c.height);
      gl.clearColor(0.05, 0.07, 0.10, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(this.prog);

      const aspect = c.width / c.height;
      const proj = mat4Perspective(1.0, aspect, 0.01, 50);
      const eye = [Math.sin(this.yaw) * this.dist * Math.cos(this.pitch),
                   Math.cos(this.yaw) * this.dist * Math.cos(this.pitch),
                   this.dist * Math.sin(this.pitch) + 0.35];
      const view = lookAt(eye, [0, 0, 0.1], [0, 0, 1]);
      const mvp = mat4Multiply(proj, view);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(this.aPos);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.vertexAttribPointer(this.aUV, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(this.aUV);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.uniformMatrix4fv(this.uMVP, false, mvp);
      gl.uniform1f(this.uWaterZ, this.waterZ);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.uTex, 0);
      const type = (this.uintIndex && this.mesh.n * this.mesh.n > 65535) ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      if (type === gl.UNSIGNED_SHORT && this.mesh.n * this.mesh.n > 65535) return;
      gl.drawElements(gl.TRIANGLES, this.mesh.count, type, 0);
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
