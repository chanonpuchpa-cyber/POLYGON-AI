
const MODEL_PATH = "./models/best.onnx";

const IMAGE_SIZE = 640;
const NUM_CLASSES = 5;
const NUM_MASKS = 32;
const PROTO_SIZE = 160;

const CONFIDENCE = 0.25;
const IOU_THRESHOLD = 0.45;

const CLASS_NAMES = [
  "Bottle",
  "Dog",
  "Glass Bottle",
  "Goat",
  "Monkey"
];

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
      if (
        item.classId === previous.classId &&
        calculateIoU(item, previous) > IOU_THRESHOLD
      ) {
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

function decodeOutput(output, meta) {
  const data = output.data;
  const anchors = 8400;

  const detections = [];

  for (let i = 0; i < anchors; i++) {
    let bestClass = -1;
    let bestScore = 0;

    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = data[(4 + c) * anchors + i];

      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < CONFIDENCE) continue;

    const cx = data[i];
    const cy = data[anchors + i];
    const w = data[2 * anchors + i];
    const h = data[3 * anchors + i];

    const x1 = Math.max(
      0, (cx - w / 2 - meta.padX) / meta.scale
    );

    const y1 = Math.max(
      0, (cy - h / 2 - meta.padY) / meta.scale
    );

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

    detections.push({
      classId: bestClass,
      score: bestScore,
      x1, y1, x2, y2,
      coefficients
    });
  }

  return nms(detections);
}

// ======================================
// CREATE SEGMENTATION MASK
// output1: [1, 32, 160, 160]
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
    Math.floor(
      (item.x1 * meta.scale + meta.padX) /
      IMAGE_SIZE * size
    )
  );

  const top = Math.max(
    0,
    Math.floor(
      (item.y1 * meta.scale + meta.padY) /
      IMAGE_SIZE * size
    )
  );

  const right = Math.min(
    size,
    Math.ceil(
      (item.x2 * meta.scale + meta.padX) /
      IMAGE_SIZE * size
    )
  );

  const bottom = Math.min(
    size,
    Math.ceil(
      (item.y2 * meta.scale + meta.padY) /
      IMAGE_SIZE * size
    )
  );

  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) {
      const p = y * size + x;

      let sum = 0;

      for (let k = 0; k < NUM_MASKS; k++) {
        sum +=
          item.coefficients[k] *
          prototypes[k * plane + p];
      }

      const probability = 1 / (1 + Math.exp(-sum));
      const index = p * 4;

      imageData.data[index] = 255;
      imageData.data[index + 1] = 255;
      imageData.data[index + 2] = 255;
      imageData.data[index + 3] =
        probability > 0.5 ? 255 : 0;
    }
  }

  maskCtx.putImageData(imageData, 0, 0);

  // Resize mask to original image coordinates.
  const result = document.createElement("canvas");
  result.width = meta.originalWidth;
  result.height = meta.originalHeight;

  const rctx = result.getContext("2d");

  const resized = document.createElement("canvas");
  resized.width = IMAGE_SIZE;
  resized.height = IMAGE_SIZE;

  resized.getContext("2d").drawImage(
    maskCanvas, 0, 0, IMAGE_SIZE, IMAGE_SIZE
  );

  rctx.drawImage(
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

  return result;
}

// ======================================
// DRAW MASK AND OUTLINE
// ======================================

function drawMask(maskCanvas, color) {
  const overlay = document.createElement("canvas");

  overlay.width = canvas.width;
  overlay.height = canvas.height;

  const octx = overlay.getContext("2d");
  octx.drawImage(maskCanvas, 0, 0);

  const imageData = octx.getImageData(
    0, 0, overlay.width, overlay.height
  );

  const rgb = color.match(/\w\w/g).map(v => parseInt(v, 16));

  for (let i = 0; i < imageData.data.length; i += 4) {
    if (imageData.data[i + 3] > 0) {
      imageData.data[i] = rgb[0];
      imageData.data[i + 1] = rgb[1];
      imageData.data[i + 2] = rgb[2];
      imageData.data[i + 3] = 75;
    }
  }

  octx.putImageData(imageData, 0, 0);
  ctx.drawImage(overlay, 0, 0);

  // Draw boundary pixels around the mask.
  const maskData = maskCanvas.getContext("2d").getImageData(
    0, 0, canvas.width, canvas.height
  ).data;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) / 400);
  ctx.beginPath();

  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < canvas.width - 1; x++) {
      const p = (y * canvas.width + x) * 4;

      if (maskData[p + 3] < 128) continue;

      const top = maskData[((y - 1) * canvas.width + x) * 4 + 3] < 128;
      const bottom = maskData[((y + 1) * canvas.width + x) * 4 + 3] < 128;
      const left = maskData[(y * canvas.width + x - 1) * 4 + 3] < 128;
      const right = maskData[(y * canvas.width + x + 1) * 4 + 3] < 128;

      if (top || bottom || left || right) {
        ctx.fillStyle = color;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  ctx.stroke();
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

  $("detections").innerHTML = detections.map(item => `
    <span class="det-item">
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

    for (const item of detections) {
      counts[item.classId]++;

      const mask = createMask(
        item,
        output1.data,
        meta
      );

      drawMask(mask, COLORS[item.classId]);
    }

    showResults(counts, detections);

    $("empty").hidden = true;

    const elapsed = (
      (performance.now() - start) / 1000
    ).toFixed(2);

    $("message").textContent +=
      ` | เวลา ${elapsed} วินาที`;

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
