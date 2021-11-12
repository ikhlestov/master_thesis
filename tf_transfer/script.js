
async function getInit() {
    initializer_model = await tf.loadGraphModel('http://0.0.0.0:8000/models/converted/initializer/model.json');
    return initializer_model
}

async function getPrePlot() {
    pre_plotter_model = await tf.loadGraphModel('http://0.0.0.0:8000/models/converted/pre_plotter/model.json');
    return pre_plotter_model
}

initializer_model = getInit();
pre_plotter_model = getPrePlot();
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');

// https://developer.mozilla.org/en-US/docs/Web/API/ImageData/ImageData
// https://stackoverflow.com/questions/55059475/how-to-convert-a-tensor-to-a-uint8array-array-in-tensorflowjs - getting data from array
function Initialize() {
    F = initializer_model.predict(42);
    speed = pre_plotter_model.predict(F);
    const arr = new Uint8ClampedArray(speed.dataSync());
    
    let imageData = new ImageData(arr, 400, 100);
    ctx.putImageData(imageData, 10, 10);

    ctx.putImageData(imageData, 10, 120);
};
