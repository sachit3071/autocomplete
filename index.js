const vectorLength = 29
const MODEL_URL = "./model.json";
var write = document.getElementById("write")
var suggestion = document.getElementById("suggestion")
// const clearSuggestion = ()=>{   suggestion.innerHTML = "" }


// Create an asynchronous function to load model
async function loadModel() {
    const model = await tf.loadLayersModel(MODEL_URL);
    return model
}

// Create an asynchronous function to load tokenizer
async function loadTokenizer() {
    tokenizer = JSON.parse(await fetch("./tokenizer.json").then((response) => { return response.json() }))
    word2Index = JSON.parse(await tokenizer.config.word_index)
    index2Word = JSON.parse(await tokenizer.config.index_word)
}

async function predictSentence(text) {

    model = await loadModel()
    await loadTokenizer()
    let vector = text.toLowerCase().split(" ")
    if (vector.length > 29) {
        vector = vector.slice(vector.length - vectorLength)
    }
    tokenizedVector = vector.map((word) => { return word2Index[word] })
    tokenizedTensor = tf.tensor(tokenizedVector)
    paddedTokenizedTensor = tokenizedTensor.pad([[vectorLength - tokenizedVector.length, 0]])
    paddedTokenizedTensor = paddedTokenizedTensor.reshape([1, vectorLength])
    predictedTensor = await model.predict(paddedTokenizedTensor)
    const { values, indices } = await tf.topk(predictedTensor)
    index = await indices.data().then(data => data[0])
    let prediction = index2Word[index]
    return prediction

}
function predictWord(word) {

}

async function generate() {
    // write.innerHTMl = write.innerText + "<span style='color:gray' id='suggestion'></span>";
    setTimeout(async () => {
        let text = write.innerText;
        let prediction = ""
        for (let i = 0; i < 5; i++) {
            let word = await predictSentence(text);
            prediction = prediction + " " + word;
            text = text + " " + prediction;
        }
        // var curPos = write.selectionStart;
        suggestion.innerText = prediction;
    }, 100);
}

// write.addEventListener("keydown", (e)=>{
//     // console.log(e.code);
//     if(e.code == "Tab")
//     {
//         prediction = suggestion.innerText;
//         suggestion.innerText = "";
//         let text = write.innerText;
//         if(text.charAt(text.length-1) != ' ')
//         {   prediction = " "+prediction.trim(); }
//         write.innerText = text+prediction;
//         write.setSelectionRange(write.value.length, write.value.length);

//     }
// })