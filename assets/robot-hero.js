const POINT_FILE = new URL("./robot-particles.bin", import.meta.url);
// Each point carries a mask for exterior visibility and one for silhouette emphasis.
const VIEW_COUNT = 32;
const POINT_STRIDE = 16;
const HEADER_BYTES = 8 + VIEW_COUNT * 3 * 4;
const MODEL_FRAME = 2.76;
const TRAIL_COUNT = 9;
const TRAIL_LIFETIME = 580;

const vertexSource = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aAppearance;
layout(location = 2) in uvec2 aMasks;

uniform mat3 uRotation;
uniform vec2 uViewport;
uniform float uScale;
uniform float uPixelRatio;
uniform float uScatterRadius;
uniform uvec4 uViewIndices;
uniform vec4 uViewWeights;
uniform vec4 uTrail[9];

out float vOpacity;

float visibleWeight(uint mask) {
  return dot(vec4((uvec4(mask) >> uViewIndices) & uvec4(1u)), uViewWeights);
}

void main() {
  float visibility = visibleWeight(aMasks.x);
  if (visibility < 0.012) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    vOpacity = 0.0;
    return;
  }

  vec3 p = uRotation * (aPosition / 8192.0);
  float perspective = 1.0 / (1.0 - p.z / 6.2);
  vec2 pixel = uViewport * 0.5 + vec2(p.x, -p.y) * uScale * perspective;
  float alongEdge = visibleWeight(aMasks.y);
  float seed = aAppearance.y;

  vec2 scatter = vec2(0.0);
  float strongest = 0.0;
  for (int i = 0; i < 9; i++) {
    vec4 pointer = uTrail[i];
    if (pointer.z <= 0.0) continue;
    vec2 delta = pixel - pointer.xy;
    float distance = length(delta);
    float influence = pointer.z * (1.0 - smoothstep(uScatterRadius * 0.12, uScatterRadius, distance));
    if (influence > strongest) {
      float angle = seed * 6.2831853;
      vec2 direction = normalize(delta + vec2(cos(angle), sin(angle)) * 2.0);
      vec2 sideways = vec2(-direction.y, direction.x);
      scatter = (direction + sideways * (seed - 0.5) * 1.06)
        * uScatterRadius * influence * (0.48 + seed * 0.58);
      strongest = influence;
    }
  }
  pixel += scatter;

  float strength = aAppearance.x;
  float diameter = mix(2.55, 3.78, strength) + alongEdge * 0.24;
  float sizeOnScreen = clamp(sqrt(uScale / 220.0), 0.68, 1.0);
  gl_PointSize = diameter * sizeOnScreen * uPixelRatio * (0.94 + seed * 0.16) * sqrt(perspective);
  vOpacity = max(0.12, strength * 1.17 + alongEdge * 0.12)
    * pow(visibility, 1.15) * (1.0 - strongest * 0.18);
  gl_Position = vec4(pixel / uViewport * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
}
`;

const fragmentSource = `#version 300 es
precision mediump float;

in float vOpacity;
out vec4 color;

void main() {
  float radius = length(gl_PointCoord * 2.0 - vec2(1.0));
  float core = 1.0 - smoothstep(0.17, 0.53, radius);
  float halo = 1.0 - smoothstep(0.26, 1.0, radius);
  color = vec4(vec3(1.0), vOpacity * (0.90 * core + 0.10 * halo));
}
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createPipeline(gl, data) {
  const vertex = compile(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message);
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(data, HEADER_BYTES), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.SHORT, false, POINT_STRIDE, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.UNSIGNED_BYTE, true, POINT_STRIDE, 6);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribIPointer(2, 2, gl.UNSIGNED_INT, POINT_STRIDE, 8);
  gl.useProgram(program);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);

  const uniforms = {};
  for (const name of ["Rotation", "Viewport", "Scale", "PixelRatio", "ScatterRadius", "ViewIndices", "ViewWeights", "Trail"]) {
    uniforms[name] = gl.getUniformLocation(program, "u" + (name === "Trail" ? "Trail[0]" : name));
  }
  return uniforms;
}

function initialView(directions) {
  const [x, y, z] = directions;
  const horizontal = Math.hypot(x, z);
  const rightX = z / horizontal;
  const rightZ = -x / horizontal;
  return [rightX, y * rightZ, x, 0, horizontal, y, rightZ, -y * rightX, z];
}

function cameraMatrix(output, base, orbit) {
  const [x, y, z, w] = orbit;
  const rotation = [
    1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w),
    2 * (x*y - z*w), 1 - 2 * (x*x + z*z), 2 * (y*z + x*w),
    2 * (x*z + y*w), 2 * (y*z - x*w), 1 - 2 * (x*x + y*y)
  ];
  for (let column = 0; column < 9; column += 3) {
    for (let row = 0; row < 3; row++) {
      output[column + row] = rotation[row] * base[column]
        + rotation[row + 3] * base[column + 1] + rotation[row + 6] * base[column + 2];
    }
  }
}

function blendViews(directions, matrix, indices, weights) {
  const scores = Array.from({ length: VIEW_COUNT }, (_, i) => ({
    index: i,
    dot: matrix[2] * directions[i * 3] + matrix[5] * directions[i * 3 + 1]
      + matrix[8] * directions[i * 3 + 2]
  })).sort((left, right) => right.dot - left.dot);

  let total = 0;
  for (let i = 0; i < 4; i++) {
    indices[i] = scores[i].index;
    weights[i] = 1 / Math.pow(0.004 + Math.max(0, 1 - scores[i].dot), 1.85);
    total += weights[i];
  }
  for (let i = 0; i < 4; i++) weights[i] /= total;
}

export function mountRobotHero(stage) {
  const canvas = stage.querySelector("canvas");
  const controls = stage.querySelector(".robot-controls");
  const reset = stage.querySelector(".robot-reset");
  const gl = canvas.getContext("webgl2", {
    alpha: true, antialias: false, depth: false, preserveDrawingBuffer: true
  });
  if (!gl) return null;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const matrix = new Float32Array(9);
  const viewIndices = new Uint32Array(4);
  const viewWeights = new Float32Array(4);
  const trailUniform = new Float32Array(TRAIL_COUNT * 4);
  const trail = [];
  let active = false, onscreen = true, pendingFrame = 0;
  let request = null, data = null, count = 0, directions, base;
  let uniforms = null, width = 0, height = 0;
  let orbit = [0, 0, 0, 1], drag = null, pointer = null;

  function requestFrame() {
    if (active && onscreen && uniforms && !pendingFrame) {
      pendingFrame = requestAnimationFrame(draw);
    }
  }

  function showCanvas(ready) {
    if (stage.classList.contains("is-interactive") === ready) return;
    stage.classList.toggle("is-interactive", ready);
    canvas.classList.toggle("is-touch-enabled", ready);
    canvas.tabIndex = ready ? 0 : -1;
    controls.hidden = !ready;
  }

  function position(event) {
    const bounds = canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  function rotate(dx, dy) {
    const distance = Math.hypot(dx, dy);
    if (distance === 0) return;
    const angle = distance * 2.8 / Math.max(1, Math.min(width, height));
    const scale = Math.sin(angle) / distance;
    const ax = dy * scale, ay = dx * scale, aw = Math.cos(angle);
    const [x, y, z, w] = orbit;
    const next = [
      aw*x + ax*w + ay*z, aw*y + ay*w - ax*z,
      aw*z + ax*y - ay*x, aw*w - ax*x - ay*y
    ];
    const length = Math.hypot(...next);
    orbit = next.map(value => value / length);
    requestFrame();
  }

  function remember(point, time) {
    if (!point || reducedMotion.matches) return;
    trail.unshift({ ...point, time });
    if (trail.length >= TRAIL_COUNT) trail.pop();
  }

  function clearPointer() {
    pointer = null;
    trail.length = 0;
  }

  function draw(timestamp) {
    pendingFrame = 0;
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    if (width === 0 || height === 0 || gl.isContextLost()) return;
    const pixelRatio = Math.min(devicePixelRatio || 1, 2);
    const pixelsWide = Math.round(width * pixelRatio);
    const pixelsHigh = Math.round(height * pixelRatio);
    if (canvas.width !== pixelsWide || canvas.height !== pixelsHigh) {
      canvas.width = pixelsWide;
      canvas.height = pixelsHigh;
    }

    cameraMatrix(matrix, base, orbit);
    blendViews(directions, matrix, viewIndices, viewWeights);
    while (trail.length && timestamp - trail.at(-1).time >= TRAIL_LIFETIME) trail.pop();
    trailUniform.fill(0);
    if (pointer) trailUniform.set([pointer.x, pointer.y, 1], 0);
    trail.forEach((sample, index) => {
      const life = Math.min(1, 1 - (timestamp - sample.time) / TRAIL_LIFETIME);
      trailUniform.set([sample.x, sample.y, life * life * 0.86], (index + 1) * 4);
    });

    gl.viewport(0, 0, pixelsWide, pixelsHigh);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniformMatrix3fv(uniforms.Rotation, false, matrix);
    gl.uniform2f(uniforms.Viewport, width, height);
    gl.uniform1f(uniforms.Scale, Math.min(width, height) / MODEL_FRAME);
    gl.uniform1f(uniforms.PixelRatio, pixelRatio);
    gl.uniform1f(uniforms.ScatterRadius, Math.min(92, Math.max(43, width * 0.12)));
    gl.uniform4uiv(uniforms.ViewIndices, viewIndices);
    gl.uniform4fv(uniforms.ViewWeights, viewWeights);
    gl.uniform4fv(uniforms.Trail, trailUniform);
    gl.drawArrays(gl.POINTS, 0, count);
    showCanvas(true);
    if (trail.length) requestFrame();
  }

  function finishDrag() {
    if (!drag) return;
    const id = drag.id;
    drag = null;
    canvas.classList.remove("is-dragging");
    if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
  }

  function resetView() {
    orbit = [0, 0, 0, 1];
    clearPointer();
    requestFrame();
  }

  canvas.addEventListener("contextmenu", event => event.preventDefault());
  canvas.addEventListener("pointerdown", event => {
    if (drag || (event.pointerType === "mouse" && event.button !== 2)) return;
    event.preventDefault();
    clearPointer();
    drag = { id: event.pointerId, ...position(event) };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
    requestFrame();
  });
  canvas.addEventListener("pointermove", event => {
    const point = position(event);
    if (drag) {
      if (drag.id !== event.pointerId) return;
      rotate(point.x - drag.x, point.y - drag.y);
      Object.assign(drag, point);
    } else if (event.pointerType === "mouse") {
      const recent = trail[0];
      if (!recent || Math.hypot(recent.x - point.x, recent.y - point.y) >= 13) {
        remember(pointer, performance.now());
      }
      pointer = point;
      requestFrame();
    }
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
    canvas.addEventListener(name, event => {
      if (drag?.id === event.pointerId) finishDrag();
    });
  }
  canvas.addEventListener("pointerleave", () => {
    remember(pointer, performance.now());
    pointer = null;
    requestFrame();
  });
  canvas.addEventListener("keydown", event => {
    const step = Math.min(width, height) * 0.038;
    const arrows = {
      ArrowLeft: [-step, 0], ArrowRight: [step, 0],
      ArrowUp: [0, -step], ArrowDown: [0, step]
    };
    if (event.key === "Home") {
      event.preventDefault();
      resetView();
    } else if (Object.hasOwn(arrows, event.key)) {
      event.preventDefault();
      rotate(...arrows[event.key]);
    }
  });
  reset.addEventListener("click", resetView);
  window.addEventListener("blur", finishDrag);

  new ResizeObserver(requestFrame).observe(canvas);
  new IntersectionObserver(([entry]) => {
    onscreen = entry.isIntersecting;
    if (onscreen) requestFrame();
  }).observe(canvas);

  function initialize() {
    if (gl.isContextLost()) return;
    uniforms = createPipeline(gl, data);
    requestFrame();
  }

  canvas.addEventListener("webglcontextlost", event => {
    event.preventDefault();
    uniforms = null;
    finishDrag();
    clearPointer();
    showCanvas(false);
  });
  canvas.addEventListener("webglcontextrestored", () => {
    if (data) initialize();
  });

  return {
    setActive(next) {
      active = next;
      if (!active) {
        cancelAnimationFrame(pendingFrame);
        pendingFrame = 0;
        finishDrag();
        clearPointer();
        return;
      }
      request ??= fetch(POINT_FILE).then(async response => {
        if (!response.ok) throw new Error("Robot particle data is unavailable");
        const points = await response.arrayBuffer();
        if (points.byteLength < HEADER_BYTES) throw new Error("Incomplete robot particle data");
        const header = new DataView(points);
        count = header.getUint32(4, true);
        if (header.getUint32(0, true) !== 0x31504252 || count === 0 ||
            points.byteLength !== HEADER_BYTES + count * POINT_STRIDE) {
          throw new Error("Invalid robot particle data");
        }
        directions = new Float32Array(points, 8, VIEW_COUNT * 3);
        base = initialView(directions);
        data = points;
        initialize();
      }).catch(() => {
        request = null;
        showCanvas(false);
      });
      requestFrame();
    }
  };
}
