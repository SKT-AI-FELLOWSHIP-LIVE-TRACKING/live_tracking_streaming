function SelectCamera()  {
    return (
        <div class="option">
            <label id="cameras">Select Device</label>
            <select class="form-select" id="cameras"></select>
        </div>
    )

    async function getCameras() {
        try {
          const camerasSelect = document.getElementById("cameras");
          console.log(camerasSelect);
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

    getCameras();

}



export default SelectCamera;