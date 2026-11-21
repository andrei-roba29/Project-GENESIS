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

  const FRAG = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform vec3 uSunDir;
    uniform float uGraticule;
    varying vec2 vUV;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      vec4 tex = texture2D(uTex, vUV);
      vec3 n = normalize(vNormal);
      vec3 viewDir = normalize(cameraPosition - vWorldPos);
      float diff = max(dot(n, normalize(uSunDir)), 0.0);
      float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.5);
      vec3 col = tex.rgb * (0.25 + 0.85 * diff) + vec3(0.15, 0.3, 0.55) * rim * 0.6;
      if (uGraticule > 0.5) {
        vec2 g = abs(fract(vUV * vec2(24.0, 12.0)) - 0.5);
        float line = step(0.47, max(g.x, g.y));
        col = mix(col, vec3(1.0), line * 0.18);
      }
      gl_FragColor = vec4(col, 1.0);
    }`;

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
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      this.prog = prog;
      gl.useProgram(prog);
      this.aPos = gl.getAttribLocation(prog, "aPos");
      this.aUV = gl.getAttribLocation(prog, "aUV");
      this.aNormal = gl.getAttribLocation(prog, "aNormal");
      this.uMVP = gl.getUniformLocation(prog, "uMVP");
      this.uModel = gl.getUniformLocation(prog, "uModel");
      this.uTex = gl.getUniformLocation(prog, "uTex");
      this.uSunDir = gl.getUniformLocation(prog, "uSunDir");
      this.uGraticule = gl.getUniformLocation(prog, "uGraticule");
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
        this.rotY += dx * 0.008;
        this.rotX = Math.max(-1.4, Math.min(1.4, this.rotX + dy * 0.008));
        lx = e.clientX; ly = e.clientY;
      });
      c.addEventListener("pointerup", e => {
        dragging = false;
        if (moved < 5) this._pick(e);
      });
      c.addEventListener("wheel", e => {
        e.preventDefault();
        this.zoom = Math.max(1.4, Math.min(6.0, this.zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
      }, { passive: false });
    }

    _pick(e) {
      const rect = this.canvas.getBoundingClientRect();
      const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ny = 1 - ((e.clientY - rect.top) / rect.height) * 2;
      // Ray from the camera (origin in view space) through the NDC point.
      const aspect = rect.width / rect.height;
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
      this.onPick(lat, lon);
    }

    draw() {
      const gl = this.gl;
      if (!gl) return;
      this._resize();
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.02, 0.04, 0.08, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      const aspect = this.canvas.width / this.canvas.height;
      const proj = mat4Perspective(1.2, aspect, 0.1, 20);
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
      gl.uniform1f(this.uGraticule, this.graticule ? 1 : 0);
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

  window.GenesisGlobe = { Globe, FlatMap };
})();
