import React, {useRef, useEffect} from "react";
import "./App.css";
import logo from './logo.svg';

import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import {drawMesh} from "./utilities";


let infer_every_ms = 1000;

function App() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const blazeface = require('@tensorflow-models/blazeface')
    const canvas = document.createElement('canvas');
    canvas.style.display = 'none';
    let context = canvas.getContext('2d');
    let visibleContext = null;
    let socket = null;

    const emotions = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'];
    const emotionEls = [];
    let emotionEl = null;
    let videoWidth = null;
    let videoHeight = null;

    const q = window.location.search;
    try {
        const new_infer_every_ms = parseInt(q.split('=')[1]);
        if (!isNaN(new_infer_every_ms)) {
            infer_every_ms = Math.min(Math.max(10, new_infer_every_ms), 1000);
        }
    } catch (Error) {
        // do nothing
    }

    //  Load blazeface
    const runFaceDetectorModel = async () => {

        const model = await blazeface.load()
        console.log("FaceDetection Model is Loaded..")
        setInterval(() => {
            detect(model);
        }, infer_every_ms);

    }

    const detect = async (net) => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video.readyState === 4
        ) {
            // Get Video Properties
            const video = webcamRef.current.video;
            const newVideoWidth = webcamRef.current.video.videoWidth;
            const newVideoHeight = webcamRef.current.video.videoHeight;

            // Set video width
            webcamRef.current.video.width = newVideoWidth;
            webcamRef.current.video.height = newVideoHeight;

            // Set canvas width
            if (newVideoWidth !== videoWidth || newVideoHeight !== videoHeight) {
                canvasRef.current.width = newVideoWidth;
                canvasRef.current.height = newVideoHeight;
            }
            videoWidth = newVideoWidth;
            videoHeight = newVideoHeight;

            // Make Detections
            const face = await net.estimateFaces(video);

            const additionalWidth = 0;
            const additionalTop = 1 / 2;
            const additionalBottom = 0;

            let [cropX, cropY] = face[0].topLeft;
            const [X, Y] = face[0].bottomRight;
            let cropWidth = X - cropX;
            let cropHeight = Y - cropY;

            cropY = Math.max(0, Math.round(cropY - cropHeight * additionalTop));
            cropHeight = Y - cropY + Math.round(cropHeight * additionalBottom);
            cropX = Math.max(0, Math.round(cropX - cropWidth * additionalWidth));
            cropWidth = X - cropX + Math.round(cropWidth * additionalWidth);

            face[0].topLeft = [cropX, cropY];
            face[0].bottomRight = [cropX + cropWidth, cropY + cropHeight];

            // Websocket
            const promise = new Promise(resolve => {
                let imageSrc = webcamRef.current.getScreenshot();
                const imageObj = new Image();
                imageObj.onload = function () {
                    //resize our canvas to match the size of the cropped area
                    canvas.style.width = cropWidth;
                    canvas.style.height = cropHeight;
                    //fill canvas with cropped image
                    context.drawImage(imageObj, cropX, cropY, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
                    imageSrc = canvas.toDataURL();
                    const apiCall = {
                        event: "localhost:subscribe",
                        data: {
                            image: imageSrc
                        },
                    };

                    if (socket === null || socket.readyState === WebSocket.CLOSED) {
                        socket = new WebSocket('ws://localhost:8000');
                    }

                    // Find emotion meter and text els
                    if (!emotionEl) {
                        emotionEl = document.getElementById("emotion_text");
                        for (const i of emotions) {
                            const el = document.getElementById(i);
                            if (!el) {
                                console.error(`FAILED to find element of emotion ${i}!`);
                            }
                            emotionEls.push(el);
                        }
                    }

                    if (socket.readyState === WebSocket.OPEN) {
                        console.debug('sending api call');
                        socket.send(JSON.stringify(apiCall));
                        socket.onmessage = function (event) {
                            const pred_log = JSON.parse(event.data);

                            // Update emotion meters and emotion text
                            let emotion = '';
                            let maxVal = 0;
                            for (let i = 0; i < emotionEls.length; i++) {
                                const strength = pred_log['predictions'][i];
                                emotionEls[i].value = strength * 100;
                                if (strength > maxVal) {
                                    maxVal = strength;
                                    emotion = emotions[i];
                                }
                            }
                            emotionEl.value = emotion;
                            console.debug('res from serv:', pred_log, 'emotion:', emotion);

                            // Get canvas context
                            if (!visibleContext) {
                                visibleContext = canvasRef.current.getContext("2d");
                            }
                            drawMesh(face, emotion, visibleContext, canvasRef.current);
                            resolve();
                        }
                    } else {
                        emotionEl.value = 'Reconnecting...';
                        console.debug('connecting to backend...');
                    }
                };
                imageObj.src = imageSrc;
            });
            await promise;
        }
    };

    useEffect(() => {
        runFaceDetectorModel()
    }, []);

    return (
        <div className="App">
            <Webcam
                ref={webcamRef}
                style={{
                    position: "absolute",
                    marginLeft: "auto",
                    marginRight: "auto",
                    left: 0,
                    right: 600,
                    top: 20,
                    textAlign: "center",
                    zindex: 9,
                    width: 640,
                    height: 480,
                }}
            />

            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    marginLeft: "auto",
                    marginRight: "auto",
                    left: 0,
                    right: 600,
                    top: 20,
                    textAlign: "center",
                    zindex: 9,
                    width: 640,
                    height: 480,
                }}
            />
            <header className="App-header">
                <img src={logo}
                     className="App-logo"
                     alt="logo"
                     style={{
                         position: "absolute",
                         marginLeft: "auto",
                         marginRight: "auto",
                         bottom: 10,
                         left: 0,
                         right: 0,
                         width: 150,
                         height: 150,
                     }}
                />
                <div className="Prediction" style={{
                    position: "absolute",
                    right: 100,
                    width: 500,
                    top: 60
                }}>
                    <label forhtml="Angry" style={{color: 'red'}}>Angry </label>
                    <progress id="Angry" value="0" max="100">10%</progress>
                    <br></br>
                    <br></br>
                    <label forhtml="Neutral" style={{color: 'green'}}>Neutral </label>
                    <progress id="Neutral" value="0" max="100">10%</progress>
                    <br></br>
                    <br></br>
                    <label forhtml="Happy" style={{color: 'orange'}}>Happy </label>
                    <progress id="Happy" value="0" max="100">10%</progress>
                    <br></br>
                    <br></br>
                    <label forhtml="Surprise" style={{color: 'cyan'}}>Surprised </label>
                    <progress id="Surprise" value="0" max="100">10%</progress>
                    <br></br>
                    <br></br>
                    <label forhtml="Sad" style={{color: 'blue'}}>Sad </label>
                    <progress id="Sad" value="0" max="100">10%</progress>
                    <br></br>
                    <br></br>
                    <label forhtml="Ahegao" style={{color: 'magenta'}}>Ahegao </label>
                    <progress id="Ahegao" value="0" max="100">10%</progress>
                </div>
                <input id="emotion_text" name="emotion_text" vale="Neutral"
                       style={{
                           position: "absolute",
                           width: 200,
                           height: 50,
                           bottom: 60,
                           left: 300,
                           fontSize: "30px",
                       }}></input>
            </header>
        </div>
    );
}

export default App;