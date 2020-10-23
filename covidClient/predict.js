// https://deeplizard.com/learn/video/nnxJyxtIuFM

// selecting an image
const IMAGENET_CLASSES = {
    0: 'Covid',
	1: 'Normal'
}
let model;
document.getElementById("predict-button").addEventListener("click",clickProdict);
(async function () {
	//model = await tf.loadLayersModel("model/model.json");
	model = await tf.loadGraphModel("model/model.json");
    $(".progress-bar").hide();
})();

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
}); 

// Loading the model 
// Attention: change Port appropriate

async function clickProdict(ev){

	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224,224]) // change the image size here
		.toFloat()
		.div(tf.scalar(255.0))
		.expandDims();

	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: IMAGENET_CLASSES[i] // we are selecting the value from the obj
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 2);

	$("#prediction-list").empty();
	top5.forEach(function (p) {
		$("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
		});
}
