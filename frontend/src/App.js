import React, { useRef, useEffect } from "react";
import "./App.css";

import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import { drawMesh } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const blazeface = require("@tensorflow-models/blazeface");

  //  Load blazeface
  const runFaceDetectorModel = async () => {
    const model = await blazeface.load();
    console.log("FaceDetection Model is Loaded..");
    setInterval(() => {
      detect(model);
    }, 100);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      const face = await net.estimateFaces(video);
      //console.log(face);

      // Websocket
      var socket = new WebSocket("ws://localhost:8000");
      var imageSrc = webcamRef.current.getScreenshot();
      var apiCall = {
        event: "localhost:subscribe",
        data: {
          image: imageSrc,
        },
      };
      socket.onopen = () => socket.send(JSON.stringify(apiCall));
      socket.onmessage = function (event) {
        var pred_log = JSON.parse(event.data);
        document.getElementById("Angry").value = Math.round(
          pred_log["predictions"]["angry"] * 100
        );
        document.getElementById("Neutral").value = Math.round(
          pred_log["predictions"]["neutral"] * 100
        );
        document.getElementById("Happy").value = Math.round(
          pred_log["predictions"]["happy"] * 100
        );
        document.getElementById("Fear").value = Math.round(
          pred_log["predictions"]["fear"] * 100
        );
        document.getElementById("Surprise").value = Math.round(
          pred_log["predictions"]["surprise"] * 100
        );
        document.getElementById("Sad").value = Math.round(
          pred_log["predictions"]["sad"] * 100
        );
        document.getElementById("Disgust").value = Math.round(
          pred_log["predictions"]["disgust"] * 100
        );

        document.getElementById("emotion_text").value = pred_log["emotion"];

        // Get canvas context
        const ctx = canvasRef.current.getContext("2d");
        requestAnimationFrame(() => {
          drawMesh(face, pred_log, ctx);
        });
      };
    }
  };

  useEffect(() => {
    runFaceDetectorModel();
  }, []);
  return (
    <div className="App">
      <header className="App-header"></header>
      <div>
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            left: 100,
            top: 60,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </div>
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          left: 100,
          top: 60,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      ></canvas>

      <div>
        <div
          className="Prediction"
          style={{
            position: "absolute",
            left: 850,
            width: 500,
            top: 60,
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
          }}
        >
          <label forhtml="Angry" style={{ color: "red" }}>
            Злость{" "}
          </label>
          <progress id="Angry" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Neutral" style={{ color: "lightgreen" }}>
            Pockerface{" "}
          </label>
          <progress id="Neutral" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Happy" style={{ color: "orange" }}>
            Счастье{" "}
          </label>
          <progress id="Happy" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Fear" style={{ color: "lightblue" }}>
            Страх{" "}
          </label>
          <progress id="Fear" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Surprise" style={{ color: "yellow" }}>
            Удивление{" "}
          </label>
          <progress id="Surprise" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Sad" style={{ color: "gray" }}>
            Грусть{" "}
          </label>
          <progress id="Sad" value="0" max="100">
            10%
          </progress>

          <br></br>
          <label forhtml="Disgust" style={{ color: "pink" }}>
            Отвращение{" "}
          </label>
          <progress id="Disgust" value="0" max="100">
            10%
          </progress>
        </div>
      </div>
    </div>
  );
}

export default App;
