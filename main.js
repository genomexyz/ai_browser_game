// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-web');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        let session = await ort.InferenceSession.create('./model.onnx');

        let data = Float32Array.from({ length: 625 }, () => Math.random());
        let data_tensor = new ort.Tensor('float32', data, [1, 625]);
        
        // prepare feeds. use model input names as keys.
        console.log("hello, this is the change")
        let feeds = { 'input': data_tensor};


        // feed inputs and run
        let results = await session.run(feeds);
        let results_data = results.action.data

        // put to the html
        let res_div = document.getElementById('result')
        res_div.innerHTML = `data of result tensor 'c': ${results_data}`

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

main();
