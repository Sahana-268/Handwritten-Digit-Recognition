const canvas = document.getElementById("digitCanvas");
const context = canvas.getContext("2d");
const clearButton = document.getElementById("clearButton");
const predictButton = document.getElementById("predictButton");
const predictionValue = document.getElementById("predictionValue");
const confidenceValue = document.getElementById("confidenceValue");
const probabilities = document.getElementById("probabilities");
const statusValue = document.getElementById("status");
const sizeOptions = document.querySelectorAll(".size-option");
const brushOptions = document.querySelectorAll(".brush-option");
const swatches = document.querySelectorAll(".swatch");

let drawing = false;
let lastPoint = null;
let predictTimer = null;
let inkColor = localStorage.getItem("digitInkColor") || "#000000";
let brushWidth = Number(localStorage.getItem("digitBrushWidth") || 26);

const brushWidths = {
  small: 12,
  medium: 26,
  large: 42,
};

function resetCanvas() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = inkColor;
  context.lineWidth = brushWidth;
}

function setFontSize(size) {
  document.documentElement.dataset.fontSize = size;
  localStorage.setItem("digitFontSize", size);
  for (const option of sizeOptions) {
    option.setAttribute("aria-pressed", String(option.dataset.size === size));
  }
}

function setInkColor(color) {
  inkColor = color;
  context.strokeStyle = inkColor;
  localStorage.setItem("digitInkColor", color);
  for (const swatch of swatches) {
    const selected = swatch.dataset.color === color;
    swatch.classList.toggle("is-selected", selected);
    swatch.setAttribute("aria-pressed", String(selected));
  }
}

function setBrushSize(size) {
  brushWidth = brushWidths[size] || brushWidths.medium;
  context.lineWidth = brushWidth;
  localStorage.setItem("digitBrushSize", size);
  localStorage.setItem("digitBrushWidth", String(brushWidth));
  for (const option of brushOptions) {
    option.setAttribute("aria-pressed", String(option.dataset.brush === size));
  }
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function drawLine(from, to) {
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
}

function schedulePrediction() {
  window.clearTimeout(predictTimer);
  predictTimer = window.setTimeout(predictDigit, 320);
}

async function predictDigit() {
  statusValue.textContent = "Predicting";
  statusValue.classList.remove("error");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Prediction failed");
    }

    predictionValue.textContent = result.digit;
    confidenceValue.textContent = `${(result.confidence * 100).toFixed(2)}%`;
    renderProbabilities(result.top);
    statusValue.textContent = "Ready";
  } catch (error) {
    statusValue.textContent = error.message;
    statusValue.classList.add("error");
  }
}

function renderProbabilities(top) {
  probabilities.innerHTML = "";
  for (const item of top) {
    const row = document.createElement("div");
    row.className = "probability-row";

    const digit = document.createElement("span");
    digit.textContent = item.digit;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(1, item.probability * 100)}%`;
    track.appendChild(fill);

    const probability = document.createElement("span");
    probability.textContent = `${(item.probability * 100).toFixed(1)}%`;

    row.append(digit, track, probability);
    probabilities.appendChild(row);
  }
}

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  drawing = true;
  lastPoint = pointFromEvent(event);
  drawLine(lastPoint, lastPoint);
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing || lastPoint === null) {
    return;
  }
  const nextPoint = pointFromEvent(event);
  drawLine(lastPoint, nextPoint);
  lastPoint = nextPoint;
  schedulePrediction();
});

canvas.addEventListener("pointerup", () => {
  drawing = false;
  lastPoint = null;
  schedulePrediction();
});

canvas.addEventListener("pointercancel", () => {
  drawing = false;
  lastPoint = null;
});

clearButton.addEventListener("click", () => {
  resetCanvas();
  predictionValue.textContent = "-";
  confidenceValue.textContent = "0.00%";
  probabilities.innerHTML = "";
  statusValue.textContent = "Ready";
  statusValue.classList.remove("error");
});

predictButton.addEventListener("click", predictDigit);

for (const option of sizeOptions) {
  option.addEventListener("click", () => {
    setFontSize(option.dataset.size || "medium");
  });
}

for (const swatch of swatches) {
  swatch.addEventListener("click", () => {
    setInkColor(swatch.dataset.color || "#000000");
  });
}

for (const option of brushOptions) {
  option.addEventListener("click", () => {
    setBrushSize(option.dataset.brush || "medium");
  });
}

setFontSize(localStorage.getItem("digitFontSize") || "medium");
setBrushSize(localStorage.getItem("digitBrushSize") || "medium");
setInkColor(inkColor);
resetCanvas();
