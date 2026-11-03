# Live Camera Attendance (RTSP / Webcam)

Author: Divyanshi Bajpai (AI Search Engineer)

`run_attendance.py` takes attendance from a live classroom camera as well as a
recorded video. `--source` decides which:

| `--source` | Input |
|---|---|
| `data/raw/videos/classroom/class.mp4` | Recorded video file (as before) |
| `camera` | The IP camera set in `.env` |
| `rtsp://...` | Any RTSP URL (avoid this on shared machines: the password shows in the shell history) |
| `0`, `1`, ... | Laptop / USB webcam |

## 1. Set up the camera (Imou / Dahua)

1. Add the camera in the **Imou Life** app, on the same 2.4 GHz Wi-Fi as the laptop.
2. Change the camera's default password in the app (device settings → device password).
3. Find its IP address (Imou Life → device → device info / network, or your router's device list).
4. Check the stream in **VLC** → Media → Open Network Stream:
   `rtsp://admin:PASSWORD@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0`
   (`subtype=1` is the lighter sub stream, often smoother on a laptop CPU).

If VLC cannot open it, the pipeline cannot either. Fix that first.

## 2. Put the camera settings in `.env`

```bash
cp .env.example .env
```

Fill in `CAMERA_IP` and `CAMERA_PASSWORD` (and `CAMERA_SUBTYPE=1` for the sub
stream). `.env` is git-ignored, so the password stays on this laptop. Never put
it in code, commands you share, screenshots or GitHub. Special characters in the
password are handled. Logs show it as `****`.

## 3. Run

```bash
python scripts/run_attendance.py --source camera --registry data/enrollment/registry \
    --show --save-video output/attendance/live_view.mp4 --verify --duration 15
```

- The script checks the camera first and stops with a clear message if it cannot connect.
- `--duration 15` stops after 15 minutes. Without it the run lasts until **Q**
  (in the `--show` window) or **Ctrl+C**. Either way the attendance and
  verification reports are written for everything seen so far.
- If the camera drops out, the stream reconnects by itself. The run ends only
  after 30 s with no frames.
- First / Last seen in the attendance report are wall-clock times. Time in
  class and the live view count from the first frame.
- Verification works as for videos, except `clip.mp4`: a live run keeps no
  recording to cut clips from. Use `--save-video` to keep one.
- On a laptop CPU expect about 4–5 processed frames per second. The newest frame
  is always used, so the run never falls behind the camera.
