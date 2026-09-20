import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { groundAt, UNDER_CANOPY_LIMIT } from "./shade.js";

const TERRAIN_STRIDE = 4;

// Same march as shadeAt() in shade.js, run once per ground texel whenever the sun moves.
const SHADOW_FRAG = /* glsl */ `
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uTop, uGround, uUnder;
uniform vec3 uSun;
uniform vec2 uSize;
uniform float uMaxH;
in vec2 vUv;
out vec4 color;

ivec2 cell(vec2 p) { return ivec2(int(p.x), int(uSize.y) - 1 - int(p.y)); }

void main() {
  color = vec4(0.0, 0.0, 0.0, 1.0);
  if (uSun.z <= 0.0) return;
  vec2 p = vUv * uSize;
  float z = float(texelFetch(uGround, cell(p), 0).r) * 0.01 + 0.1;
  float horiz = length(uSun.xy);
  vec2 dir = uSun.xy / horiz;
  float rise = uSun.z / horiz, t = 0.5, beneath = 0.0;
  for (int i = 0; i < 320; i++) {
    float stride = t < 40.0 ? 1.0 : t < 120.0 ? 2.0 : 4.0;
    t += stride;
    vec2 q = p + dir * t;
    if (q.x < 0.0 || q.y < 0.0 || q.x >= uSize.x || q.y >= uSize.y) break;
    float rz = z + t * rise;
    if (rz > uMaxH) break;
    ivec2 c = cell(q);
    if (rz >= float(texelFetch(uTop, c, 0).r) * 0.01) continue;
    float g = float(texelFetch(uGround, c, 0).r) * 0.01;
    uint u = texelFetch(uUnder, c, 0).r;
    if (u == 0u || rz < g) return;
    if (rz >= g + float(u) * 0.25) { color.g = 1.0; return; }
    beneath += stride;
    if (beneath > ${UNDER_CANOPY_LIMIT.toFixed(1)}) { color.g = 1.0; return; }
  }
  color.r = 1.0;
}`;

const QUAD_VERT = /* glsl */ `
in vec3 position;
in vec2 uv;
out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const TERRAIN_VERT = /* glsl */ `
varying vec2 vUv;
varying vec3 vNormal;
void main() { vUv = uv; vNormal = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const TERRAIN_FRAG = /* glsl */ `
uniform sampler2D uBase, uShadow;
uniform vec3 uSunDir, uSunTint;
uniform float uDaylight;
uniform vec2 uTexel;
varying vec2 vUv;
varying vec3 vNormal;
void main() {
  vec3 base = texture2D(uBase, vUv).rgb;
  // 3x3 tent over the 1m shadow texels; otherwise edges stair-step when the camera is near the ground.
  vec4 s = vec4(0.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
    float wgt = (2.0 - abs(float(i))) * (2.0 - abs(float(j))) / 16.0;
    s += wgt * texture2D(uShadow, vUv + vec2(float(i), float(j)) * uTexel);
  }
  float facing = 0.8 + 0.2 * max(dot(normalize(vNormal), uSunDir), 0.0);
  // Tree shade reads a touch lighter than hard shade: canopy shadow is dappled, a wall's is not.
  float shade = mix(0.36, 0.46, s.g);
  vec3 lit = base * uSunTint * facing;
  vec3 col = mix(base * shade * vec3(0.78, 0.88, 1.15), lit, s.r);
  gl_FragColor = vec4(col * mix(0.12, 1.0, uDaylight), 1.0);
  #include <colorspace_fragment>
}`;

function intTexture(data, w, h, type, internalFormat) {
  const t = new THREE.DataTexture(data, w, h, THREE.RedIntegerFormat, type);
  t.internalFormat = internalFormat;
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

function drawBasemap(world) {
  const { w, h } = world, size = 2048, k = size / w;
  const mask = document.createElement("canvas");
  mask.width = w; mask.height = h;
  const mctx = mask.getContext("2d"), img = mctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    if (!world.under[i]) continue;
    img.data.set([122, 150, 88, 255], i * 4);
  }
  mctx.putImageData(img, 0, 0);

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const trace = (pts, close) => {
    ctx.beginPath();
    pts.forEach(([x, y], i) => ctx[i ? "lineTo" : "moveTo"](x * k, (h - y) * k));
    if (close) ctx.closePath();
  };
  ctx.fillStyle = "#b4b9a6"; ctx.fillRect(0, 0, size, size);
  trace(world.park, true); ctx.fillStyle = "#93b96c"; ctx.fill();
  ctx.drawImage(mask, 0, 0, size, size);
  ctx.fillStyle = "#6f9fc0";
  for (const poly of world.water) { trace(poly, true); ctx.fill(); }
  ctx.lineCap = ctx.lineJoin = "round";
  const styles = { road: ["#8e8e8a", 7], service: ["#a3a39c", 4], rail: ["#5d5a58", 5], path: ["#e4d9bd", 2.6] };
  for (const kind of ["road", "service", "rail", "path"]) {
    [ctx.strokeStyle, ctx.lineWidth] = [styles[kind][0], styles[kind][1] * k];
    for (const line of world.lines) if (line.k === kind) { trace(line.p); ctx.stroke(); }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export function createScene(container, world) {
  const { w, h, minH } = world;
  const toWorld = (x, y, z) => new THREE.Vector3(x - w / 2, z - minH, h / 2 - y);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 2, 12000);
  camera.position.set(-350, 650, 900);
  const controls = new MapControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 25;
  controls.maxDistance = 3000;
  controls.target.copy(toWorld(w / 2, h / 2, groundAt(world, w / 2, h / 2)));

  // --- shadow pass
  const coarse = matchMedia("(pointer: coarse)").matches;
  const shadowTarget = new THREE.WebGLRenderTarget(coarse ? 1024 : w, coarse ? 1024 : h, { depthBuffer: false });
  const shadowMaterial = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, vertexShader: QUAD_VERT, fragmentShader: SHADOW_FRAG, depthTest: false,
    uniforms: {
      uTop: { value: intTexture(world.top, w, h, THREE.UnsignedShortType, "R16UI") },
      uGround: { value: intTexture(world.ground, w, h, THREE.UnsignedShortType, "R16UI") },
      uUnder: { value: intTexture(world.under, w, h, THREE.UnsignedByteType, "R8UI") },
      uSun: { value: new THREE.Vector3(0, 0, 1) }, uSize: { value: new THREE.Vector2(w, h) }, uMaxH: { value: world.maxH },
    },
  });
  const shadowScene = new THREE.Scene();
  shadowScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), shadowMaterial));
  const shadowCamera = new THREE.Camera();

  // --- terrain
  const terrainGeo = new THREE.PlaneGeometry(w, h, w / TERRAIN_STRIDE, h / TERRAIN_STRIDE);
  const pos = terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setZ(i, groundAt(world, pos.getX(i) + w / 2, pos.getY(i) + h / 2) - minH);
  terrainGeo.rotateX(-Math.PI / 2);
  terrainGeo.computeVertexNormals();
  const terrainMaterial = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VERT, fragmentShader: TERRAIN_FRAG,
    uniforms: {
      uBase: { value: drawBasemap(world) }, uShadow: { value: shadowTarget.texture },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunTint: { value: new THREE.Color(1, 1, 1) }, uDaylight: { value: 1 }, uTexel: { value: new THREE.Vector2(0.8 / shadowTarget.width, 0.8 / shadowTarget.height) },
    },
  });
  scene.add(new THREE.Mesh(terrainGeo, terrainMaterial));

  // --- buildings
  const buildingGeos = world.buildings.map((b) => {
    const shape = new THREE.Shape(b.p.map(([x, y]) => new THREE.Vector2(x - w / 2, y - h / 2)));
    const geo = new THREE.ExtrudeGeometry(shape, { depth: b.h, bevelEnabled: false });
    geo.rotateX(-Math.PI / 2);
    geo.translate(0, b.z - minH, 0);
    return geo;
  });
  scene.add(new THREE.Mesh(mergeGeometries(buildingGeos), new THREE.MeshLambertMaterial({ color: 0xd9d2c5 })));

  // --- trees: crowns sized from LIDAR crown-top height
  const crowns = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshLambertMaterial({ flatShading: true }), world.trees.length);
  const trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.5, 0.7, 1, 5), new THREE.MeshLambertMaterial({ color: 0x5b4636 }), world.trees.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), tint = new THREE.Color();
  world.trees.forEach(([x, y, th], i) => {
    const base = toWorld(x + 0.5, y + 0.5, groundAt(world, x, y));
    const r = Math.min(7, Math.max(2, 0.2 * th + 1.2)), crownH = th * 0.62;
    m.compose(base.clone().setY(base.y + th - crownH / 2), q, new THREE.Vector3(r, crownH / 2, r));
    crowns.setMatrixAt(i, m);
    m.compose(base.clone().setY(base.y + (th - crownH) / 2), q, new THREE.Vector3(Math.max(0.5, th / 30), th - crownH, Math.max(0.5, th / 30)));
    trunks.setMatrixAt(i, m);
    const jitter = ((x * 73 + y * 151) % 100) / 100;
    crowns.setColorAt(i, tint.setHSL(0.27 + jitter * 0.06, 0.42, 0.25 + jitter * 0.1));
  });
  scene.add(crowns, trunks);

  // --- bench and picnic table furniture, turned to face the way a sitter would
  const box = (sx, sy, sz, x, y, z) => new THREE.BoxGeometry(sx, sy, sz).translate(x, y, z);
  const furniture = {
    bench: mergeGeometries([box(1.8, 0.08, 0.5, 0, 0.45, 0), box(1.8, 0.5, 0.06, 0, 0.75, -0.25), box(0.08, 0.45, 0.45, -0.8, 0.22, 0), box(0.08, 0.45, 0.45, 0.8, 0.22, 0)]),
    table: mergeGeometries([box(1.8, 0.08, 0.8, 0, 0.75, 0), box(1.8, 0.06, 0.3, 0, 0.45, 0.75), box(1.8, 0.06, 0.3, 0, 0.45, -0.75), box(0.1, 0.75, 1.6, -0.7, 0.37, 0), box(0.1, 0.75, 1.6, 0.7, 0.37, 0)]),
  };
  for (const [kind, geo] of Object.entries(furniture)) {
    const items = world.benches.filter((b) => b.kind === kind);
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial({ color: 0x7a5a3c }), items.length);
    items.forEach((b, i) => {
      q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, Math.PI - (b.f * Math.PI) / 180);
      mesh.setMatrixAt(i, m.compose(toWorld(b.x, b.y, groundAt(world, b.x, b.y)), q, new THREE.Vector3(1, 1, 1)));
    });
    scene.add(mesh);
  }
  q.identity();

  // --- light and sky
  const sunLight = new THREE.DirectionalLight(0xffffff, 2.5);
  const ambient = new THREE.HemisphereLight(0xcfe3ff, 0x6b7a55, 1.1);
  scene.add(sunLight, sunLight.target, ambient);
  const sunDisc = new THREE.Mesh(new THREE.SphereGeometry(140, 24, 12), new THREE.MeshBasicMaterial({ color: 0xfff1b8, fog: false }));
  scene.add(sunDisc);
  const SKY = { night: new THREE.Color(0x0b1020), low: new THREE.Color(0xf0b98a), day: new THREE.Color(0x9cc7ec) };
  scene.background = new THREE.Color();

  // --- bench markers
  const markers = new Map();
  const markerGroup = new THREE.Group();
  scene.add(markerGroup);
  for (const b of world.benches) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, sizeAttenuation: false, transparent: true }));
    sprite.position.copy(toWorld(b.x, b.y, groundAt(world, b.x, b.y) + 2.5));
    sprite.renderOrder = 10;
    sprite.userData.id = b.id;
    markerGroup.add(sprite);
    markers.set(b.id, { sprite, canvas, tex });
  }

  const dotCanvas = document.createElement("canvas");
  dotCanvas.width = dotCanvas.height = 64;
  const dctx = dotCanvas.getContext("2d");
  dctx.beginPath(); dctx.arc(32, 32, 30, 0, Math.PI * 2); dctx.fillStyle = "rgba(60, 140, 255, 0.28)"; dctx.fill();
  dctx.beginPath(); dctx.arc(32, 32, 13, 0, Math.PI * 2); dctx.fillStyle = "#2f7bff"; dctx.fill();
  dctx.lineWidth = 5; dctx.strokeStyle = "#ffffff"; dctx.stroke();
  const userDot = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(dotCanvas), depthTest: false, sizeAttenuation: false, transparent: true }));
  userDot.scale.setScalar(0.06);
  userDot.renderOrder = 13;
  userDot.visible = false;
  const guide = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0x5ad1ff, dashSize: 6, gapSize: 4, depthTest: false, transparent: true }));
  guide.renderOrder = 9;
  guide.visible = false;
  guide.frustumCulled = false;
  scene.add(userDot, guide);

  let dirty = true, flight = null;
  controls.addEventListener("change", () => (dirty = true));

  const api = {
    onPick: null, onHover: null, onCompass: null,

    setSun(sun) {
      const daylight = THREE.MathUtils.smoothstep(sun.altitude, -0.1, 0.12);
      const dir = new THREE.Vector3(sun.e, sun.u, -sun.n);
      shadowMaterial.uniforms.uSun.value.set(sun.e, sun.n, sun.u);
      renderer.setRenderTarget(shadowTarget);
      renderer.render(shadowScene, shadowCamera);
      renderer.setRenderTarget(null);

      const warmth = 1 - THREE.MathUtils.smoothstep(sun.altitude, 0.05, 0.45);
      terrainMaterial.uniforms.uSunDir.value.copy(dir);
      terrainMaterial.uniforms.uSunTint.value.setRGB(1.06, 1.0 - 0.1 * warmth, 0.94 - 0.24 * warmth);
      terrainMaterial.uniforms.uDaylight.value = daylight;
      sunLight.position.copy(controls.target).addScaledVector(dir, 1000);
      sunLight.target.position.copy(controls.target);
      sunLight.intensity = 2.5 * daylight * (sun.u > 0 ? 1 : 0);
      sunLight.color.setRGB(1, 1 - 0.15 * warmth, 1 - 0.35 * warmth);
      ambient.intensity = 0.15 + 0.95 * daylight;
      sunDisc.position.copy(dir).multiplyScalar(7000);
      sunDisc.visible = sun.u > -0.02;
      scene.background.copy(SKY.night).lerp(SKY.low, daylight).lerp(SKY.day, THREE.MathUtils.smoothstep(sun.altitude, 0.02, 0.35));
      dirty = true;
    },

    /** Sunlit benches glow in `fill`; shaded ones recede so the eye lands on where the sun is. */
    paintMarker(id, fill, sunlit, selected) {
      const { canvas, tex, sprite } = markers.get(id);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, 64, 64);
      if (sunlit) {
        const halo = ctx.createRadialGradient(32, 32, 10, 32, 32, 32);
        halo.addColorStop(0, "rgba(255, 200, 70, 0.75)");
        halo.addColorStop(1, "rgba(255, 200, 70, 0)");
        ctx.fillStyle = halo; ctx.fillRect(0, 0, 64, 64);
      }
      ctx.beginPath(); ctx.arc(32, 32, sunlit ? 15 : 11, 0, Math.PI * 2);
      ctx.fillStyle = sunlit ? fill : "#475063"; ctx.fill();
      ctx.lineWidth = selected ? 6 : 3.5;
      ctx.strokeStyle = selected ? "#5ad1ff" : sunlit ? "#ffffff" : "#15181d";
      ctx.stroke();
      tex.needsUpdate = true;
      sprite.scale.setScalar(selected ? 0.07 : 0.05);
      sprite.renderOrder = selected ? 12 : sunlit ? 11 : 10;
      dirty = true;
    },

    /** GPS fix in local metres, or null to hide. */
    setUser(me) {
      userDot.visible = !!me;
      if (me) userDot.position.copy(toWorld(me.x, me.y, groundAt(world, me.x, me.y) + 2));
      dirty = true;
    },

    /** Straight line from the user to a bench; null clears it. */
    setGuide(me, bench) {
      guide.visible = !!(me && bench);
      if (guide.visible) {
        guide.geometry.setFromPoints([me, bench].map((p) => toWorld(p.x, p.y, groundAt(world, p.x, p.y) + 2)));
        guide.computeLineDistances();
      }
      dirty = true;
    },

    faceNorth() {
      const offset = camera.position.clone().sub(controls.target);
      const flat = Math.hypot(offset.x, offset.z);
      flight = { target: controls.target.clone(), position: controls.target.clone().add(new THREE.Vector3(0, offset.y, flat)) };
    },

    flyTo(bench) {
      const target = toWorld(bench.x, bench.y, groundAt(world, bench.x, bench.y));
      const offset = camera.position.clone().sub(controls.target).setLength(160);
      offset.y = Math.max(offset.y, 90);
      flight = { target, position: target.clone().add(offset) };
    },

    resize() {
      const { clientWidth, clientHeight } = container;
      renderer.setSize(clientWidth, clientHeight);
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
      dirty = true;
    },
  };

  // --- picking: a tap selects, a drag pans
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  let downAt = null;
  const hit = (ev) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(markerGroup.children)[0]?.object.userData.id ?? null;
  };
  renderer.domElement.addEventListener("pointerdown", (ev) => (downAt = [ev.clientX, ev.clientY]));
  renderer.domElement.addEventListener("pointerup", (ev) => {
    if (downAt && Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) < 6) api.onPick?.(hit(ev));
    downAt = null;
  });
  renderer.domElement.addEventListener("pointermove", (ev) => {
    if (ev.pointerType === "mouse" && !ev.buttons) api.onHover?.(hit(ev), ev);
  });
  renderer.domElement.addEventListener("pointerleave", () => api.onHover?.(null));

  new ResizeObserver(api.resize).observe(container);
  renderer.setAnimationLoop(() => {
    if (flight) {
      controls.target.lerp(flight.target, 0.12);
      camera.position.lerp(flight.position, 0.12);
      if (controls.target.distanceTo(flight.target) < 0.5) flight = null;
      dirty = true;
    }
    controls.update();
    if (!dirty) return;
    dirty = false;
    renderer.render(scene, camera);
    api.onCompass?.(controls.getAzimuthalAngle());
  });
  return api;
}
