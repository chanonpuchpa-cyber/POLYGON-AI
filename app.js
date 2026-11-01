const MODEL_PATH = "./models/best.onnx";

const IMAGE_SIZE = 640;
const NUM_CLASSES = 5;
const NUM_MASKS = 32;
const PROTO_SIZE = 160;

const CONFIDENCE = 0.20;
const IOU_THRESHOLD = 0.45;       // ตัดผลซ้ำของคลาสเดียวกัน
const CROSS_CLASS_IOU = 0.65;     // ถ้าวัตถุเดียวกันถูกทายเป็นคนละคลาส ให้เก็บอันที่มั่นใจสุด

// ----- ตั้งค่าการแสดงผล -----
const DEBUG = true;               // แสดงคะแนนสูงสุดของแต่ละคลาส (ไว้ดูว่าโมเดลเห็นขวดไหม) ปิดเมื่อใช้งานจริง
const FILL_ALPHA = 0;             // 0 = เส้นขอบอย่างเดียว, 0.12-0.25 = ใส่สีจางๆ ทับวัตถุ
const SHOW_LABEL = true;          // ป้ายชื่อ + % ที่จุดบนสุดของวัตถุ

const CLASS_NAMES = [
  "Bottle",
  "Dog",
  "Glass Bottle",
  "Goat",
  "Monkey"
];

// สีประจำคลาส: Bottle=ฟ้า, Dog=ส้ม, Glass Bottle=ม่วง, Goat=เขียว, Monkey=ชมพู
// ใช้กับเส้นขอบ ป้ายชื่อ ชิป และการ์ดสรุป
const COLORS = [
  "#43e8ff",
  "#ffad66",
  "#b69aff",
  "#55f5b5",
  "#ff7fc9"
];

const $ = id => document.getElementById(id);

const fileInput = $("fileInput");
const cameraInput = $("cameraInput");
const preview = $("preview");
const detectBtn = $("detectBtn");
const canvas = $("resultCanvas");
const ctx = canvas.getContext("2d");

let model = null;
let currentImage = null;

// ======================================
// LOAD MODEL
// ======================================

async function loadModel() {
  try {
    ort.env.wasm.numThreads = 1;

    model = await ort.InferenceSession.create(
      MODEL_PATH,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      }
    );

    $("status").textContent = "โมเดลพร้อมใช้งาน";
    detectBtn.disabled = !currentImage;

    console.log("Model loaded");
    console.log("Inputs:", model.inputNames);
    console.log("Outputs:", model.outputNames);

  } catch (error) {
    console.error(error);
    $("status").textContent = "โหลดโมเดลไม่สำเร็จ";
    $("message").textContent = error.message;
  }
}

// ======================================
// IMAGE UPLOAD
// ======================================

function openImage(file) {
  if (!file || !file.type.startsWith("image/")) return;

  const image = new Image();
  const url = URL.createObjectURL(file);

  image.onload = () => {
    currentImage = image;
    preview.src = url;
    preview.hidden = false;
    detectBtn.disabled = !model;

    $("empty").hidden = false;
    $("message").textContent = "พร้อมตรวจจับภาพ";
    $("previewEmpty").hidden = true;
  };

  image.onerror = () => {
    URL.revokeObjectURL(url);
    alert("ไม่สามารถเปิดภาพได้");
  };

  image.src = url;
}

fileInput.addEventListener("change", e => {
  openImage(e.target.files[0]);
});

cameraInput.addEventListener("change", e => {
  openImage(e.target.files[0]);
});

// ======================================
// PREPROCESSING
// ======================================

function preprocess(image) {
  const scale = Math.min(
    IMAGE_SIZE / image.width,
    IMAGE_SIZE / image.height
  );

  const newWidth = Math.round(image.width * scale);
  const newHeight = Math.round(image.height * scale);

  const padX = Math.floor((IMAGE_SIZE - newWidth) / 2);
  const padY = Math.floor((IMAGE_SIZE - newHeight) / 2);

  const temp = document.createElement("canvas");

  temp.width = IMAGE_SIZE;
  temp.height = IMAGE_SIZE;

  const tctx = temp.getContext("2d");

  tctx.fillStyle = "rgb(114,114,114)";
  tctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);

  tctx.drawImage(
    image,
    padX,
    padY,
    newWidth,
    newHeight
  );

  const pixels = tctx.getImageData(
    0, 0, IMAGE_SIZE, IMAGE_SIZE
  ).data;

  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const input = new Float32Array(3 * plane);

  for (let i = 0; i < plane; i++) {
    input[i] = pixels[i * 4] / 255;
    input[plane + i] = pixels[i * 4 + 1] / 255;
    input[2 * plane + i] = pixels[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor(
      "float32",
      input,
      [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    ),
    scale,
    padX,
    padY,
    originalWidth: image.width,
    originalHeight: image.height,
    newWidth,
    newHeight
  };
}

// ======================================
// NON MAXIMUM SUPPRESSION
// ======================================

function calculateIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection =
    Math.max(0, x2 - x1) *
    Math.max(0, y2 - y1);

  const areaA =
    (a.x2 - a.x1) * (a.y2 - a.y1);

  const areaB =
    (b.x2 - b.x1) * (b.y2 - b.y1);

  return intersection / (areaA + areaB - intersection + 1e-7);
}

function nms(detections) {
  detections.sort((a, b) => b.score - a.score);

  const selected = [];

  for (const item of detections) {
    let keep = true;

    for (const previous of selected) {
      const iou = calculateIoU(item, previous);
      const limit =
        item.classId === previous.classId
          ? IOU_THRESHOLD
          : CROSS_CLASS_IOU;

      if (iou > limit) {
        keep = false;
        break;
      }
    }

    if (keep) selected.push(item);
    if (selected.length >= 100) break;
  }

  return selected;
}

// ======================================
// DECODE YOLO26 OUTPUT
// output0: [1, 41, 8400]
// ======================================

let lastClassMax = new Array(NUM_CLASSES).fill(0);

function decodeOutput(output, meta) {
  const data = output.data;
  const anchors = 8400;

  const detections = [];
  const classMax = new Array(NUM_CLASSES).fill(0);

  for (let i = 0; i < anchors; i++) {
    // เก็บทุกคลาสที่ผ่านเกณฑ์ ไม่ใช่แค่คลาสที่คะแนนสูงสุด
    // เพื่อให้ตำแหน่งเดียวกันเจอได้ทั้งลิงและขวด
    const passed = [];

    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = data[(4 + c) * anchors + i];

      if (score > classMax[c]) classMax[c] = score;
      if (score >= CONFIDENCE) passed.push({ c, score });
    }

    if (passed.length === 0) continue;

    const cx = data[i];
    const cy = data[anchors + i];
    const w = data[2 * anchors + i];
    const h = data[3 * anchors + i];

    const x1 = Math.max(0, (cx - w / 2 - meta.padX) / meta.scale);
    const y1 = Math.max(0, (cy - h / 2 - meta.padY) / meta.scale);
    const x2 = Math.min(
      meta.originalWidth,
      (cx + w / 2 - meta.padX) / meta.scale
    );
    const y2 = Math.min(
      meta.originalHeight,
      (cy + h / 2 - meta.padY) / meta.scale
    );

    if (x2 <= x1 || y2 <= y1) continue;

    const coefficients = new Float32Array(NUM_MASKS);

    for (let k = 0; k < NUM_MASKS; k++) {
      coefficients[k] = data[(9 + k) * anchors + i];
    }

    for (const { c, score } of passed) {
      detections.push({
        classId: c,
        score,
        x1, y1, x2, y2,
        coefficients
      });
    }
  }

  lastClassMax = classMax;

  return nms(detections);
}

// ======================================
// CREATE SEGMENTATION MASK
// output1: [1, 32, 160, 160]
// คืนค่า { canvas, top } เมื่อ top = จุดบนสุดของวัตถุ (ไว้วางป้ายชื่อ)
// ======================================

function createMask(item, prototypes, meta) {
  const size = PROTO_SIZE;
  const plane = size * size;

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = size;
  maskCanvas.height = size;

  const maskCtx = maskCanvas.getContext("2d");
  const imageData = maskCtx.createImageData(size, size);

  const left = Math.max(
    0,
    Math.floor((item.x1 * meta.scale + meta.padX) / IMAGE_SIZE * size)
  );
  const top = Math.max(
    0,
    Math.floor((item.y1 * meta.scale + meta.padY) / IMAGE_SIZE * size)
  );
  const right = Math.min(
    size,
    Math.ceil((item.x2 * meta.scale + meta.padX) / IMAGE_SIZE * size)
  );
  const bottom = Math.min(
    size,
    Math.ceil((item.y2 * meta.scale + meta.padY) / IMAGE_SIZE * size)
  );

  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) {
      const p = y * size + x;

      let sum = 0;

      for (let k = 0; k < NUM_MASKS; k++) {
        sum += item.coefficients[k] * prototypes[k * plane + p];
      }

      const probability = 1 / (1 + Math.exp(-sum));
      const index = p * 4;

      // เก็บความน่าจะเป็นไว้ใน alpha (ยังไม่ตัดขาด)
      // เพื่อให้ขยายภาพแล้วขอบเรียบ ไม่เป็นขั้นบันได
      imageData.data[index] = 255;
      imageData.data[index + 1] = 255;
      imageData.data[index + 2] = 255;
      imageData.data[index + 3] = Math.round(probability * 255);
    }
  }

  maskCtx.putImageData(imageData, 0, 0);

  // ขยาย 160 -> 640 แบบ smooth แล้วค่อยตัดขาดที่ 0.5
  const resized = document.createElement("canvas");
  resized.width = IMAGE_SIZE;
  resized.height = IMAGE_SIZE;

  const rz = resized.getContext("2d", { willReadFrequently: true });
  rz.imageSmoothingEnabled = true;
  rz.imageSmoothingQuality = "high";
  rz.drawImage(maskCanvas, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const rd = rz.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const px = rd.data;

  let topY = -1;
  let sumX = 0;
  let count = 0;

  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const a = (y * IMAGE_SIZE + x) * 4 + 3;

      if (px[a] >= 128) {
        px[a] = 255;

        if (topY === -1) topY = y;
        if (y === topY) {
          sumX += x;
          count++;
        }
      } else {
        px[a] = 0;
      }
    }
  }

  rz.putImageData(rd, 0, 0);

  // ย้ายกลับไปพิกัดภาพต้นฉบับ
  const result = document.createElement("canvas");
  result.width = meta.originalWidth;
  result.height = meta.originalHeight;

  result.getContext("2d").drawImage(
    resized,
    meta.padX,
    meta.padY,
    meta.newWidth,
    meta.newHeight,
    0,
    0,
    meta.originalWidth,
    meta.originalHeight
  );

  const topPoint =
    topY >= 0
      ? {
          x: (sumX / count - meta.padX) / meta.scale,
          y: (topY - meta.padY) / meta.scale
        }
      : { x: item.x1, y: item.y1 };

  return { canvas: result, top: topPoint };
}

// ======================================
// DRAW OUTLINE (เส้นขอบตามตัววัตถุ ไม่มีกรอบสี่เหลี่ยม)
// ======================================

function tintMask(maskCanvas, color) {
  const tinted = document.createElement("canvas");
  tinted.width = maskCanvas.width;
  tinted.height = maskCanvas.height;

  const t = tinted.getContext("2d");
  t.drawImage(maskCanvas, 0, 0);
  t.globalCompositeOperation = "source-in";
  t.fillStyle = color;
  t.fillRect(0, 0, tinted.width, tinted.height);

  return tinted;
}

// สร้างเส้นรอบวัตถุ (วงแหวนด้านนอก) หนา thickness พิกเซล
function makeRing(maskCanvas, color, thickness) {
  const tinted = tintMask(maskCanvas, color);

  const ring = document.createElement("canvas");
  ring.width = maskCanvas.width;
  ring.height = maskCanvas.height;

  const r = ring.getContext("2d");
  const steps = 24;

  for (let s = 0; s < steps; s++) {
    const angle = (s / steps) * Math.PI * 2;
    r.drawImage(
      tinted,
      Math.cos(angle) * thickness,
      Math.sin(angle) * thickness
    );
  }

  // เจาะส่วนในออก เหลือแต่เส้นขอบ
  r.globalCompositeOperation = "destination-out";
  r.drawImage(maskCanvas, 0, 0);

  return ring;
}

function drawInstance(mask, color) {
  const thickness = Math.max(
    2,
    Math.round(Math.min(canvas.width, canvas.height) / 220)
  );

  // สีจางๆ ทับพื้นที่วัตถุ (ถ้าเปิดใช้)
  if (FILL_ALPHA > 0) {
    ctx.save();
    ctx.globalAlpha = FILL_ALPHA;
    ctx.drawImage(tintMask(mask.canvas, color), 0, 0);
    ctx.restore();
  }

  // เงาดำบางๆ ใต้เส้น ให้เห็นชัดบนพื้นหลังสว่าง
  ctx.save();
  ctx.globalAlpha = 0.45;
  ctx.drawImage(makeRing(mask.canvas, "#000", thickness + 1.5), 0, 0);
  ctx.restore();

  // เส้นสีประจำวัตถุชิ้นนี้
  ctx.drawImage(makeRing(mask.canvas, color, thickness), 0, 0);
}

function drawLabel(item, point, color) {
  const label = `${CLASS_NAMES[item.classId]} ${(item.score * 100).toFixed(1)}%`;
  const fontSize = Math.max(
    13,
    Math.min(24, Math.min(canvas.width, canvas.height) / 30)
  );
  const padding = Math.round(fontSize * 0.4);

  ctx.save();
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.textBaseline = "middle";

  const boxW = ctx.measureText(label).width + padding * 2;
  const boxH = fontSize + padding * 2;

  const x = Math.min(
    Math.max(0, point.x - boxW / 2),
    Math.max(0, canvas.width - boxW)
  );
  const y = Math.max(0, point.y - boxH - 8);

  ctx.fillStyle = color;
  ctx.fillRect(x, y, boxW, boxH);
  ctx.fillStyle = "#07111f";
  ctx.fillText(label, x + padding, y + boxH / 2);
  ctx.restore();
}

// ======================================
// DISPLAY RESULTS
// ======================================

function showResults(counts, detections) {
  $("classCounts").innerHTML = CLASS_NAMES.map((name, i) => `
    <div class="class-card">
      <b>${counts[i]}</b>
      <span style="color:${COLORS[i]}">${name}</span>
    </div>
  `).join("");

  // ชิปแต่ละอันใช้สีตามคลาส ตรงกับเส้นบนภาพ
  $("detections").innerHTML = detections.map(item => `
    <span class="det-item" style="color:${item.color} !important; border-color:${item.color} !important; background:${item.color}22 !important">
      ${CLASS_NAMES[item.classId]}
      ${(item.score * 100).toFixed(1)}%
    </span>
  `).join("");

  $("message").textContent =
    `พบวัตถุทั้งหมด ${detections.length} รายการ`;
}

// ======================================
// MAIN PREDICTION
// ======================================

async function detectObjects() {
  if (!model || !currentImage) return;

  detectBtn.disabled = true;
  $("status").textContent = "กำลังประมวลผล...";

  const start = performance.now();

  try {
    const meta = preprocess(currentImage);

    const feeds = {
      [model.inputNames[0]]: meta.tensor
    };

    const outputs = await model.run(feeds);

    const output0 = outputs.output0;
    const output1 = outputs.output1;

    if (!output0 || !output1) {
      throw new Error("ไม่พบ Output ของโมเดล");
    }

    const detections = decodeOutput(output0, meta);

    canvas.width = meta.originalWidth;
    canvas.height = meta.originalHeight;

    ctx.drawImage(currentImage, 0, 0);

    const counts = [0, 0, 0, 0, 0];
    const masks = [];

    // วาดเส้นขอบของวัตถุแต่ละชิ้น ใช้สีตามคลาส
    detections.forEach((item, index) => {
      counts[item.classId]++;
      item.color = COLORS[item.classId];  // สีตามคลาส

      const mask = createMask(item, output1.data, meta);
      masks.push(mask);
      drawInstance(mask, item.color);
    });

    // วาดป้ายชื่อไว้บนสุด
    if (SHOW_LABEL) {
      detections.forEach((item, index) => {
        drawLabel(item, masks[index].top, item.color);
      });
    }

    showResults(counts, detections);

    $("empty").hidden = true;

    const elapsed = (
      (performance.now() - start) / 1000
    ).toFixed(2);

    $("message").textContent +=
      ` | เวลา ${elapsed} วินาที`;

    if (DEBUG) {
      const summary = CLASS_NAMES
        .map((n, i) => `${n} ${(lastClassMax[i] * 100).toFixed(0)}%`)
        .join(" · ");
      $("message").textContent += ` | คะแนนสูงสุด: ${summary}`;
      console.log("Max score per class:", summary);
    }

    $("status").textContent = "โมเดลพร้อมใช้งาน";

  } catch (error) {
    console.error(error);
    $("message").textContent =
      "เกิดข้อผิดพลาด: " + error.message;

    $("status").textContent = "ประมวลผลไม่สำเร็จ";

  } finally {
    detectBtn.disabled = false;
  }
}

detectBtn.addEventListener("click", detectObjects);

// ======================================
// START
// ======================================

showResults([0, 0, 0, 0, 0], []);
loadModel();