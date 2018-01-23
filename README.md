# Planetside Stalker
A basic Open CV Python app that can track moving objects (Planetmans) in footage from a static camera (the stalkiest Stalker Cloaker). Video game domain because, well, I'm me.

## Execution

- ```motion_detector.py``` is the executable. Try the ``--help`` flag.
- ```demo.sh``` to see a well-performing example

## Dependencies
- Developed on OpenCV 3.1.0, Python 3 API

## Domain
I was an avid player of Planetside 2, a massive first-person shooter. There are lots of humans running around to track; with the wrinkle that they're science fiction soldiers - covered in armor, wearing helmets and backpacks. I wanted to train a unique computer vision system: a human detector that could distinguish the three main factions in the game's story. It's difficult because they're often wearing camouflage, easy when they're wearing their signature colors (red/blue/purple), and I think, tractable due to stylistic differences (sleek, blocky, alien). Thus, blob detection is a no-go, I would have to learn more advanced techniques.

## Approach
Currently, it fuses:

- Difference map to find blobs - good for camouflaged, moving objects
- Mean-shift tracking
- Naive Kalman filter tracking. It uses OpenCV's with a few tweaks; a HUGe amount of work is needed

## Limitations
- Single-target tracking. Sometimes there are many, sometimes none; it handles all poorly.
- Blip handling. The sensor model puts too much weight on intermittent tiny visual distortions and particle effects (like smoke or explosions).
- Classification. The heurestic for humans is aspect ratio and error in mean-shift map. 

## Improvements
- More objective metrics
- Better tuning
- Custom Kalman filter. I think I could train an HMM to model Planetman movement and the sensor model to reject screen-shakes, visual distortions, and particle effects.
- Multi-target tracking. Was in progress. First pass would be based on my radar contacts work. Second pass would be the Hungarian algorithm.
- Classification. First, of humans v.s. background. Masking would be simplest (aka reverse sprites). Then, I'd use a CNN people detector that I can tinker with - e.g. trained on INRIA data set - with transfer learning to my game character dataset

## History
I had the ambitious goal of making a bot that could play Planetside! It's a very complex and layered game, but I figured I could simplify the domain enough to be tractable. It would be more impressive than a Doom bot and unique: I thought I might be the only Planetside player who also did AI! The first step was basic vision.

My first attempt as pure frustration: my HOG-SVM did not work **at all** outside the training set. I learned first-hand why convolutional neural nets are so important! I decided to switch to video and use classical CV to track moving objects, which would then be fed to modern deep neural networks. Then I got a dream job and started doing AI at work. Now it looks like I will have use for this code, so I'm putting it on Github under a permissive license.
