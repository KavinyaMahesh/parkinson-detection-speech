function predict() {
    let fileInput = document.getElementById("audioFile").files[0];
    let formData = new FormData();
    formData.append("file", fileInput);

    fetch("https://parkinson-detection-speech.onrender.com/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Prediction: " + data.prediction;
    })
    .catch(error => console.error("Error:", error));
}
