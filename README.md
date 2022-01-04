# Simple PyTorch Object Tracking

Track objects in a video in 75 lines of code. Here are commands to track cars in part of a [test video of traffic](https://www.youtube.com/watch?v=MNn9qKG2UFI) that's available on YouTube if you want to try it yourself.

```
wget https://silly-noyce-82730b.netlify.app/traffic.mp4
poetry install
python simple-track.py traffic.mp4 --class-index 3
```

You can see a visualization of this algorithm in action [here](https://silly-noyce-82730b.netlify.app/).
