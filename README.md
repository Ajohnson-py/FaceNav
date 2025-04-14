# FaceNav


> A program that translates facial expressions into macOS mouse inputs.

---

### Table of Contents

- [Description](#description)
- [Installation and Running](#installation-and-running)
- [How To Use FaceNav](#how-to-use-facenav)
- [License](#license)
- [Author Info](#author-info)

---

## Description

For individuals with upper-body impairments, conventional computer input systems can be fatiguing or outright unusable. FaceNav was created to address this issue specifically for macOS. By utilizing **MediaPipe** and **OpenCV**, FaceNav translates facial expressions into mouse inputs. 

Its current functionality includes mouse movement (up, down, left, and right), left-click, right-click, and drag and drop. It also has a menu bar icon to pause the program.


#### Technologies

- Python and PyObjC
- MediaPipe
- OpenCV

---

## Installation and Running

### 1. Clone the repo

```
git clone https://github.com/Ajohnson-py/FaceNav.git
cd FaceNav
```

### 2. Set up a virtual environment (recommended)

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run FaceNav

```
python main.py
```

---

## How To Use FaceNav

Once you’ve installed and launched FaceNav (python main.py), your webcam will activate and begin analyzing your facial expressions in real-time.

Basic Usage:
- Move your mouth **left/right** to move the mouse in the corresponding direction. For **upward movement**, push your bottom lip into your top lip, and for **downward movement**, suck your top lip into your mouth.
- To **left-click**, move your eyebrows up and then back down (dragging can be done by keeping them up).
- To **right-click**, wink with your left eye for about 1 second.

> Tip: try not to move your head much during usage, and use good lighting.

[Back To The Top](#FaceNav)

---

## License

MIT License

Copyright (c) 2025 August Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- LinkedIn - [August Johnson](www.linkedin.com/in/aug-johnson)


[Back To The Top](#FaceNav)
