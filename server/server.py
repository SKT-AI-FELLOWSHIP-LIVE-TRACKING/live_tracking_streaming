import math
import argparse
import json
import logging
from logging import raiseExceptions
import os
import ssl
import uuid

import cv2
from aiohttp import web
import aiohttp_cors
from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

#medaipipe!!
import timeit
import asyncio
import numpy as np
import mediapipe as mp
from urllib import request

from face_detection import face_detection
from object_det_v2 import object_detection
from detection_processing import detection_processing
from reid.torchreid.utils import FeatureExtractor
from dtos import DetectionRegions
from dtos import FaceRegions
from dtos import FMOT_TrackingRegions
from face_detection import face_detection
from object_det_v2 import object_detection
from detection_processing import detection_processing
from FastMOT.fastmot.tracker import MultiTracker
from FastMOT.fastmot.utils import ConfigDecoder
from types import SimpleNamespace
#from utils import get_ratio, round_to_even, cos_sim, visualize_faces, visualize_objects, binary_search, decide_target_size, show_fps, float_frame_imshow, piecewise_function
from utils import *
from web_demo import x_processing, y_processing

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
local_video = None

# initialize deep sort
model_name = "osnet_x0_25"
model_weights = "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth"
feature_extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_weights,
            device='cpu'
)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
    
    
    async def recv(self):
        # detections type
        DET_DTYPE = np.dtype(
            [('tlbr', float, 4),
            ('label', int),
            ('conf', float)],
            align=True
        )

        #last_detection = 0
        mot_json = "FastMOT/cfg/mot.json"
        with open(mot_json) as cfg_file:
            mot_json = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

        def regions_to_detections(all_regions):
            boxes = []
            d = DetectionRegions(0,0,0,0,0,-1)
            f = FaceRegions(0,0,0,0,'tmp',0)
            t = FMOT_TrackingRegions(0,0,0,0,-1)
            for i, region in enumerate(all_regions):
                y1 = int(region.y * image_height)
                x1 = int(region.x * image_width)
                y2 = int((region.y + region.h) * image_height)
                x2 = int((region.x + region.w) * image_width)
                
                if (type(region) == type(f)):
                    class_id = 1
                elif (type(region)== type(d)):
                    class_id = int(region.class_id)
                else:
                    raiseExceptions("data type을 확인할 수 없습니다.")
                score = region.score
                # boxes.append(([top, left, bottom, right], class_id, score))
                boxes.append(([x1, y1, x2, y2], class_id, score))

            return np.array(boxes, DET_DTYPE).view(np.recarray)

        def detect_objects(image):
            # face detection
            fd = face_detection(image)
            fd.detect_faces()
            regions = fd.localization_to_region()
            #visualize_faces(image, regions, image_width, image_height)

            # object detection
            od = object_detection(image)
            output_dict, category_index = od.detect_objects()
            boxes = output_dict['detection_boxes']
            classes = output_dict['detection_classes']
            scores = output_dict['detection_scores']
            #visualize_objects(image, boxes, classes, scores, category_index, image_width, image_height)     


            # detection processing
            dp = detection_processing(boxes, classes, scores, regions[0])
            dp.tensors_to_regions()
            all_regions = dp.sort_detection()

            return all_regions



        global image_width, image_height

        frame = await self.track.recv()

        image = frame.to_ndarray(format="bgr24")
        image_width = image.shape[1]
        image_height = image.shape[0]
        # 비율 정하기
        original_ratio = get_ratio(image_width, image_height)
         # requested_ratio = 9 / 16 # arg 인자로 받아오기 !
        requested_ratio =  9 / 16     
        target_width, target_height, scaled_target, wh_flag = decide_target_size(original_ratio, requested_ratio, image_width, image_height)
        pre_x_center = 0.5
        pre_y_center = 0.5
        fps = 0
        frame_id = 0

        tracker = MultiTracker((image_width, image_height), 'cosine', **vars(mot_json.mot_cfg.tracker_cfg))
        frame_rate = 30
        cap_dt = 1. / frame_rate
        tracker.reset(cap_dt)

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
 
        elif self.transform == "tracking":
        
            start_t = timeit.default_timer()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if (frame_id == 0):
                # detect objects
                all_regions = detect_objects(image)

                ### image color transition
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #### detections 처리
                detections = regions_to_detections(all_regions)

                # tracker initiation
                tracker.init(image, detections)

                ### image color transition
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    

            elif (frame_id % 5 == 0): # detection per 5 frames # % 5 == 0
                # detect objects
                all_regions = detect_objects(image)

                ### image color transition
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                tracker.compute_flow(image)

                # track
                tracker.apply_kalman()

                ############################
                detections = regions_to_detections(all_regions)

                features = get_features(detections.tlbr, image, feature_extractor)

                if (len(features)):
                    embeddings = features.numpy()
                    ### 디텍션 처리
                    tracker.update(frame_id, detections, embeddings)

                    results = []
                    track_lists = list(track for track in tracker.tracks.values()
                            if track.confirmed and track.active)
                    
                    for track in track_lists:
                        bbox = track.tlbr
                        xmin = bbox[0] / image_width
                        ymin = bbox[1] / image_height
                        w = (bbox[2] - bbox[0]) / image_width
                        h = (bbox[3] - bbox[1]) / image_height

                        try:
                            results.append(FMOT_TrackingRegions(xmin, ymin, w, h, track.trk_id))
                        except:
                            print("Failed to append Tracking Regions")
                    
                    if (len(results) == 0):
                        frame_id = 4
                    else:
                        all_regions = results                  

                # no detection
                else:
                    all_regions = []


                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            else:
                ### image color transition
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # tracking
                tracker.track(image)

                track_lists = list(track for track in tracker.tracks.values()
                        if track.confirmed and track.active)
                results = []
                for track in track_lists:
                    if not (track.confirmed and track.active):
                        continue
                    bbox = track.tlbr

                    xmin = bbox[0] / image_width
                    ymin = bbox[1] / image_height
                    w = (bbox[2] - bbox[0]) / image_width
                    h = (bbox[3] - bbox[1]) / image_height
                    
                    try:
                        results.append(FMOT_TrackingRegions(xmin, ymin, w, h, track.trk_id))
                    except:
                        print("Failed to update TrackingRegions")
                if (len(results) == 0):
                    frame_id = 4
                else:
                    all_regions = results

                ### image color transition
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_id += 1

            # terminate_t = timeit.default_timer()
            if (wh_flag == 0):
                optimal_x_center = x_processing(all_regions, scaled_target)

                terminate_t = timeit.default_timer()
                if (abs(pre_x_center - optimal_x_center) * image_width < 10): # 가만히 있을 때 흔들리는 이슈 해결하기 위해 사용.... 일시적인 방법이므로 이보다 더 좋은 방법이 있을 시 대체하는 것이 좋을듯. 현재 10으로 흔들리는 것을 방지해놨는데 이보다 크게 파라미터를 설정하면 interpolation 과정에서 프레임이 불안정하게 보간됨.
                    optimal_x_center = pre_x_center
                await real_time_interpolate_x(pre_x_center, optimal_x_center, image_width, target_width, image)
        # terminate_t = timeit.default_timer()

                left = int(optimal_x_center * image_width - target_width / 2)
                if (left < 0):
                    left = 0
                elif (left > image_width - target_width):
                    left = image_width - target_width

                pre_x_center = optimal_x_center
                img = image[:, left:left+target_width]

            else: # 세로가 crop될 때
                optimal_y_center = y_processing(all_regions, scaled_target)

                terminate_t = timeit.default_timer()
                if (abs(pre_y_center - optimal_y_center) * image_height < 10): # 가만히 있을 때 흔들리는 이슈 해결하기 위해 사용.... 일시적인 방법이므로 이보다 더 좋은 방법이 있을 시 대체하는 것이 좋을듯. 현재 10으로 흔들리는 것을 방지해놨는데 이보다 크게 파라미터를 설정하면 interpolation 과정에서 프레임이 불안정하게 보간됨.
                    optimal_y_center = pre_y_center
                await real_time_interpolate_y(pre_y_center, optimal_y_center, image_height, target_height, image)
        # terminate_t = timeit.default_timer()

                top = int(optimal_y_center * image_height - target_height / 2)
                if (top < 0):
                        top = 0
                elif (top > image_height - target_height):
                        top = image_height - target_height
            

                pre_y_center = optimal_y_center            
                img = image[:, left:left+target_width]

            # fps 계산
            # terminate_t = timeit.default_timer()
            fps += int(1.0 / (terminate_t - start_t))
            '''
            cv2.putText(img,
                        "FPS:" + str(int(fps / (frame_id+1))),
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        2)
            '''
             # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame  

        else:
            return frame


async def home(request):
    return await javascript(request, 'HomeScreen.js')
   
async def track(request):
    return await javascript(request, 'TrackScreen.js')
  
async def javascript(request, file):
    content = open(os.path.join(ROOT,"src/components", file), "r").read()
    return web.Response(content_type="application/javascript", text=content)

'''
async def javascriptn(request):
    content = open(os.path.join(ROOT, "nclient.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)



async def App(request):
    content = open(os.path.join(ROOT, "src/App.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def Route(request):
    content = open(os.path.join(ROOT, "src/components/RouteList.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def indexJS(request):
    content = open(os.path.join(ROOT, "src/index.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

'''
async def receive(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()


    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Receiving for %s", request.remote)

    # handle offer
    await pc.setRemoteDescription(offer)


    for t in pc.getTransceivers():
        # if t.kind == "audio" and player.audio:
        #     pc.addTrack(player.audio)
        print(local_video)
        if t.kind == "video":
            pc.addTrack(local_video)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
            }
        ),
    )


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)


    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global local_video
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            print(local_video)
            print("done")
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
            }
        ),
        headers={
            "X-Custom-Server-Header": "Custom data",    
        }
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    #parser.add_argument('--mode', type=str, default="default", help='If u want sport tracking, set "--mode sport".')
    #parser.add_argument('--w', type=int, default=1, help='Ratio of Frame Width')
    #parser.add_argument('--h', type=int, default=1, help='Ratio of Frame Height')
    ##p#rint(config)
    #asyncio.run((args))

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    
    @asyncio.coroutine
    def handler(request):
        return web.Response(
            text="Hello!",
            headers={
                "X-Custom-Server-Header": "Custom data",
            })

    app = web.Application()
    # `aiohttp_cors.setup` returns `aiohttp_cors.CorsConfig` instance.
    # The `cors` instance will store CORS configuration for the
    # application.
    cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", home)
    #app.router.add_get("/preview", preview)
    app.router.add_get("/track", track)
    #app.router.add_get("/client.js", javascript)
    #app.router.add_get("/nclient.js", javascriptn)
    
    #app.router.add_post("/offer", offer)
    # Add all resources to `CorsConfig`.
    resource = cors.add(app.router.add_resource("/offer"))
    cors.add(resource.add_route("POST", offer))
    
    resource = cors.add(app.router.add_resource("/receive"))
    cors.add(resource.add_route("POST", receive))

    #app.router.add_post("/receive", receive)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
