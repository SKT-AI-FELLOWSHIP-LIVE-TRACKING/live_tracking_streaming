import { useParams } from "react-router-dom";
import { Link } from "react-router-dom";
import socketio from "socket.io-client";
import '../TrackScreen.css'

function TrackScreen() {
  const params = useParams();
  const localUsername = params.username;
  const roomName = params.room;
  const deviceId = params.deviceId;

  const socket = socketio("https://signaling-server-flask.herokuapp.com/", {
    autoConnect: false,
  });

  const sendData = (data) => {
    socket.emit("data", {
      username: localUsername,
      room: roomName,
      //data: data,
    });
  };

  //let myStream;
  let pc;
  //const camerasSelect = document.getElementById("cameras");
//  camerasSelect.addEventListener("input", handleCameraChange)

  //console.log(camerasSelect);

  return (
    <div className="pl-6 ... pt-5">
      <h2 className="text-lg font-semibold text-indigo-600">Welcome, {"User " + localUsername}! You're now at.. 
      </h2>
      <p className="mt-2 text-3xl font-bold leading-8 tracking-tight text-gray-900 sm:text-4xl">
        {"Room Id: " + roomName}       
        <Link to={`/preview/${localUsername}/${roomName}/${deviceId}`} target="_blank"> 
            <input class="bg-red-500 hover:bg-red-700 
            text-white font-bold py-1 px-2 rounded 
            focus:outline-none focus:shadow-outline ml-4" id="preview" type="submit" name="submit" value="preview" />
        </Link>
       </p>   
      <p className="mt-4 mb-4 max-w-2xl text-xl text-gray-500">
            Show us your Content! Have Fun!!
      </p>
      <div id="options">
        <h2 className="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">Options</h2>
        <div className="pt-2" >
            <input id="use-datachannel" type="checkbox"/>
            <label htmlFor="use-datachannel" class="px-2">Use datachannel</label>
            <select id="datachannel-parameters" class="border-0 cursor-pointer rounded-full drop-shadow-md bg-sky-200 w-72 duration-300 hover:bg-sky-400 focus:bg-gray-300">
                <option value='{"ordered": true}'>Ordered, reliable</option>
                <option value='{"ordered": false, "maxRetransmits": 0}'>Unordered, no retransmissions</option>
                <option value='{"ordered": false, "maxPacketLifetime": 500}'>Unordered, 500ms lifetime</option>
            </select>
        </div>
        <div class="option">
            <input id="use-audio"  type="checkbox"/>
            <label htmlFor="use-audio" class="px-2">Use audio</label>
            <select id="audio-codec" class="border-0 cursor-pointer rounded-full drop-shadow-md bg-sky-200 w-72 duration-300 hover:bg-sky-400 focus:bg-gray-300">
                <option value="default">Default codecs</option>
                <option value="opus/48000/2">Opus</option>
                <option value="PCMU/8000">PCMU</option>
                <option value="PCMA/8000">PCMA</option>
            </select>
        </div>
        <div class="option">
            <input id="use-video" type="checkbox"/>
            <label htmlFor="use-video" class="px-2">Use video</label>
            <select id="video-resolution" class="border-0 cursor-pointer rounded-full drop-shadow-md bg-sky-200 w-72 duration-300 hover:bg-sky-400 focus:bg-gray-300">
                <option value="">Default resolution</option>
                <option value="320x240">320x240</option>
                <option value="640x480">640x480</option>
                <option value="960x540">960x540</option>
                <option value="1280x720">1280x720</option>
            </select>
            <select id="video-transform" class="border-0 cursor-pointer rounded-full drop-shadow-md bg-sky-200 w-72 duration-300 hover:bg-sky-400 focus:bg-gray-300">
                <option value="tracking">Default</option>
                <option value="none">Tracking</option>              
                <option value="edges">Edge detection</option>
                <option value="cartoon">Cartoon effect</option>
                <option value="rotate">Rotate</option>
            </select>
            <select id="video-codec" class="border-0 cursor-pointer rounded-full drop-shadow-md bg-sky-200 w-72 duration-300 hover:bg-sky-400 focus:bg-gray-300">
                <option value="default">Default codecs</option>
                <option value="VP8/90000">VP8</option>
                <option value="H264/90000">H264</option>
            </select>
        </div>
        <div class="option">
            <input id="use-stun" type="checkbox"/>
            <label htmlFor="use-stun" class="px-2">Use STUN server</label>
        </div>

        <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4" 
                id="start" 
                onClick={start}
        >Start Tracking</button>
        <button class="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4" 
                id="stop" 
                onClick={stop}
        >Stop</button>
      </div>
      
      <div class="pt-3">
        <h2 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">State</h2>
        <p>
            ICE gathering state: <span id="ice-gathering-state"></span>
        </p>
        <p>
            ICE connection state: <span id="ice-connection-state"></span>
        </p>
        <p>
            Signaling state: <span id="signaling-state"></span>
        </p>
      </div>

      <div id="media" class="pt-3">
          <h2 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">Media</h2>
      
          <audio id="audio" autoPlay={true}></audio>
          <video id="video" 
          autoPlay={true} 
          playsInline={true} 
          style={{
            width:'30%'
        }} ></video>
      </div>
      
      <div class="pt-3">
        <h2 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">Data channel</h2>
        <pre id="data-channel"></pre>
        
        <h2 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">SDP</h2>
        
        <h3 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">Offer</h3>
        <pre id="offer-sdp"></pre>
        
        <h3 class="text-xl font-bold text-gray-900 md:text-2xl dark:text-white">Answer</h3>
        <pre id="answer-sdp"></pre>
      </div>
    </div>
  );


  // get DOM elements
  var dataChannelLog = document.getElementById('data-channel'),
  iceConnectionLog = document.getElementById('ice-connection-state'),
  iceGatheringLog = document.getElementById('ice-gathering-state'),
  signalingLog = document.getElementById('signaling-state');

 
  // data channel
  var dc = null, dcInterval = null;

  async function getCameras() {
    try {
      const camerasSelect = document.getElementById("cameras");
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      cameras.forEach((camera) => {
        const option = document.createElement("option");
        option.value = camera.deviceId;
        option.innerText = camera.label;
        camerasSelect.appendChild(option);
      });
      console.log(camerasSelect);
    } catch (e) {
      console.log(e);
    }
  }
 

  async function getMedia2 () {
    //console.log(deviceId);
    
    /*
    navigator.mediaDevices.enumerateDevices()
    .then(function(devices) {
        devices.forEach(function(device) {
        console.log(device);
        console.log(device.kind + ": " + device.label +
                    " id = " + device.deviceId);
       if (deviceId =='562e5d2b628835a066f3b90307ef6dce5c2f25112b3ab168b4ecfc6e9c29e35e')
            console.log("Gopro");
    });
    })
    .catch(function(err) {
    console.log(err.name + ": " + err.message);
    });
*/  //getCameras();
    //const deviceId = '562e5d2b628835a066f3b90307ef6dce5c2f25112b3ab168b4ecfc6e9c29e35e'

    var constraints = {
        audio: false,
        video: { deviceId: { exact: deviceId } },
    };
    /*
    const cameraConstraints = {
        audio: true,
        video: { deviceId: { exact: deviceId } },
    };
    */
    //console.log(cameraConstraints);

    if (document.getElementById('use-video').checked) {
        var resolution = document.getElementById('video-resolution').value;
        if (resolution) {
            resolution = resolution.split('x');
            constraints.video = {
                width: parseInt(resolution[0], 0),
                height: parseInt(resolution[1], 0)
            };
        } else {
            constraints.video = { deviceId: { exact: deviceId } };
        }
    }

    if (constraints.audio || constraints.video) {
        if (constraints.video) {
            document.getElementById('media').style.display = 'block';
        }
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            console.log(stream.getTracks());
            stream.getTracks().forEach(function(track) {
                pc.addTrack(track, stream);
            });
            //myFace.srcObject = myStream; //new
            //if (!deviceId) { //new
            //    getCameras(); //new
            //  } //new
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }
  }
  
  function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
    }

    pc = new RTCPeerConnection(config);
    //console.log("log: ",iceConnectionLog);

    // register some listeners to help debugging
    /*
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    signalingLog.textContent = pc.signalingState;
    */
    // connect audio / video
    pc.addEventListener('track', function(evt) {
        if (evt.track.kind === 'video'){
            //console.log("evt:",evt)

            document.getElementById('video').srcObject = evt.streams[0];
        }
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
  }

  function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        codec = document.getElementById('audio-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        codec = document.getElementById('video-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        }

        //document.getElementById('offer-sdp').textContent = offer.sdp;
        return fetch('http://localhost:8080/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: document.getElementById('video-transform').value
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        //document.getElementById('answer-sdp').textContent = answer.sdp;
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
  });
  }

  function start() {
    document.getElementById('start').style.display = 'none';

    pc = createPeerConnection();

    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }

    if (document.getElementById('use-datachannel').checked) {
        var parameters = JSON.parse(document.getElementById('datachannel-parameters').value);

        dc = pc.createDataChannel('chat', parameters);
        dc.onclose = function() {
            clearInterval(dcInterval);
            //dataChannelLog.textContent += '- close\n';
        };
        dc.onopen = function() {
            //dataChannelLog.textContent += '- open\n';
            dcInterval = setInterval(function() {
                var message = 'ping ' + current_stamp();
                //dataChannelLog.textContent += '> ' + message + '\n';
                dc.send(message);
            }, 1000);
        };
        dc.onmessage = function(evt) {
            //dataChannelLog.textContent += '< ' + evt.data + '\n';

            if (evt.data.substring(0, 4) === 'pong') {
                var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
                //dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
            }
        };
    }
    //getUserMedia
    getMedia2()

    document.getElementById('stop').style.display = 'inline-block';
  }

  function stop() {
    document.getElementById('stop').style.display = 'none';

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
  }

  function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')

    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
  }

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
  }
}

export default TrackScreen;