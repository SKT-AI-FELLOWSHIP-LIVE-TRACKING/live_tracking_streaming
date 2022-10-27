import { Routes, Route } from "react-router-dom";
import CallScreen from "./CallScreen";
import HomeScreen from "./HomeScreen";
import TrackScreen from "./TrackScreen";
import WatchScreen from "./WatchScreen";
import Introduction from "./Introduction";

function RouteList() {
  return (
    <Routes>
      <Route path="/intro" element={<Introduction />} />
      <Route path="/" element={<HomeScreen />} />
      <Route path="/call/:username/:room" element={<CallScreen />} />
      <Route path="/tracking/:username/:room/:deviceId" element={<TrackScreen />} />
      <Route path="/preview/:username/:room/:deviceId" element={<WatchScreen />} />    
        
    </Routes>
  );
}

export default RouteList;
