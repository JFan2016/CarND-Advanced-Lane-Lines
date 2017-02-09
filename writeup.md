<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
<title>Advanced Lane Lines</title>
<!-- 2017-02-09 Thu 15:31 -->
<meta  http-equiv="Content-Type" content="text/html;charset=utf-8" />
<meta  name="generator" content="Org-mode" />
<meta  name="author" content="David A. Ventimiglia" />
<style type="text/css">
 <!--/*--><![CDATA[/*><!--*/
  .title  { text-align: center; }
  .todo   { font-family: monospace; color: red; }
  .done   { color: green; }
  .tag    { background-color: #eee; font-family: monospace;
            padding: 2px; font-size: 80%; font-weight: normal; }
  .timestamp { color: #bebebe; }
  .timestamp-kwd { color: #5f9ea0; }
  .right  { margin-left: auto; margin-right: 0px;  text-align: right; }
  .left   { margin-left: 0px;  margin-right: auto; text-align: left; }
  .center { margin-left: auto; margin-right: auto; text-align: center; }
  .underline { text-decoration: underline; }
  #postamble p, #preamble p { font-size: 90%; margin: .2em; }
  p.verse { margin-left: 3%; }
  pre {
    border: 1px solid #ccc;
    box-shadow: 3px 3px 3px #eee;
    padding: 8pt;
    font-family: monospace;
    overflow: auto;
    margin: 1.2em;
  }
  pre.src {
    position: relative;
    overflow: visible;
    padding-top: 1.2em;
  }
  pre.src:before {
    display: none;
    position: absolute;
    background-color: white;
    top: -10px;
    right: 10px;
    padding: 3px;
    border: 1px solid black;
  }
  pre.src:hover:before { display: inline;}
  pre.src-sh:before    { content: 'sh'; }
  pre.src-bash:before  { content: 'sh'; }
  pre.src-emacs-lisp:before { content: 'Emacs Lisp'; }
  pre.src-R:before     { content: 'R'; }
  pre.src-perl:before  { content: 'Perl'; }
  pre.src-java:before  { content: 'Java'; }
  pre.src-sql:before   { content: 'SQL'; }

  table { border-collapse:collapse; }
  caption.t-above { caption-side: top; }
  caption.t-bottom { caption-side: bottom; }
  td, th { vertical-align:top;  }
  th.right  { text-align: center;  }
  th.left   { text-align: center;   }
  th.center { text-align: center; }
  td.right  { text-align: right;  }
  td.left   { text-align: left;   }
  td.center { text-align: center; }
  dt { font-weight: bold; }
  .footpara:nth-child(2) { display: inline; }
  .footpara { display: block; }
  .footdef  { margin-bottom: 1em; }
  .figure { padding: 1em; }
  .figure p { text-align: center; }
  .inlinetask {
    padding: 10px;
    border: 2px solid gray;
    margin: 10px;
    background: #ffffcc;
  }
  #org-div-home-and-up
   { text-align: right; font-size: 70%; white-space: nowrap; }
  textarea { overflow-x: auto; }
  .linenr { font-size: smaller }
  .code-highlighted { background-color: #ffff00; }
  .org-info-js_info-navigation { border-style: none; }
  #org-info-js_console-label
    { font-size: 10px; font-weight: bold; white-space: nowrap; }
  .org-info-js_search-highlight
    { background-color: #ffff00; color: #000000; font-weight: bold; }
  /*]]>*/-->
</style>
<style>@import 'https://fonts.googleapis.com/css?family=Quattrocento';</style>
<link rel="stylesheet" type="text/css" href="base.css"/>
<script type="text/javascript">
/*
@licstart  The following is the entire license notice for the
JavaScript code in this tag.

Copyright (C) 2012-2013 Free Software Foundation, Inc.

The JavaScript code in this tag is free software: you can
redistribute it and/or modify it under the terms of the GNU
General Public License (GNU GPL) as published by the Free Software
Foundation, either version 3 of the License, or (at your option)
any later version.  The code is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU GPL for more details.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.


@licend  The above is the entire license notice
for the JavaScript code in this tag.
*/
<!--/*--><![CDATA[/*><!--*/
 function CodeHighlightOn(elem, id)
 {
   var target = document.getElementById(id);
   if(null != target) {
     elem.cacheClassElem = elem.className;
     elem.cacheClassTarget = target.className;
     target.className = "code-highlighted";
     elem.className   = "code-highlighted";
   }
 }
 function CodeHighlightOff(elem, id)
 {
   var target = document.getElementById(id);
   if(elem.cacheClassElem)
     elem.className = elem.cacheClassElem;
   if(elem.cacheClassTarget)
     target.className = elem.cacheClassTarget;
 }
/*]]>*///-->
</script>
<script type="text/javascript" src="http://orgmode.org/mathjax/MathJax.js"></script>
<script type="text/javascript">
<!--/*--><![CDATA[/*><!--*/
    MathJax.Hub.Config({
        // Only one of the two following lines, depending on user settings
        // First allows browser-native MathML display, second forces HTML/CSS
        //  config: ["MMLorHTML.js"], jax: ["input/TeX"],
            jax: ["input/TeX", "output/HTML-CSS"],
        extensions: ["tex2jax.js","TeX/AMSmath.js","TeX/AMSsymbols.js",
                     "TeX/noUndefined.js"],
        tex2jax: {
            inlineMath: [ ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"], ["\\begin{displaymath}","\\end{displaymath}"] ],
            skipTags: ["script","noscript","style","textarea","pre","code"],
            ignoreClass: "tex2jax_ignore",
            processEscapes: false,
            processEnvironments: true,
            preview: "TeX"
        },
        showProcessingMessages: true,
        displayAlign: "center",
        displayIndent: "2em",

        "HTML-CSS": {
             scale: 100,
             availableFonts: ["STIX","TeX"],
             preferredFont: "TeX",
             webFont: "TeX",
             imageFont: "TeX",
             showMathMenu: true,
        },
        MMLorHTML: {
             prefer: {
                 MSIE:    "MML",
                 Firefox: "MML",
                 Opera:   "HTML",
                 other:   "HTML"
             }
        }
    });
/*]]>*///-->
</script>
</head>
<body>
<div id="content">
<h1 class="title">Advanced Lane Lines</h1>

<div id="outline-container-sec-1" class="outline-2">
<h2 id="sec-1">Introduction</h2>
<div class="outline-text-2" id="text-1">
<p>
The goals / steps of this project are the following:
</p>

<ul class="org-ul">
<li>Compute the camera calibration matrix and distortion coefficients
given a set of chessboard images.
</li>
<li>Apply a distortion correction to raw images.
</li>
<li>Use color transforms, gradients, etc., to create a thresholded
binary image.
</li>
<li>Apply a perspective transform to rectify binary image ("birds-eye
view").
</li>
<li>Detect lane pixels and fit to find the lane boundary.
</li>
<li>Determine the curvature of the lane and vehicle position with
respect to center.
</li>
<li>Warp the detected lane boundaries back onto the original image.
</li>
<li>Output visual display of the lane boundaries and numerical
estimation of lane curvature and vehicle position.
</li>
</ul>
</div>

<div id="outline-container-sec-1-1" class="outline-3">
<h3 id="sec-1-1">Setup</h3>
<div class="outline-text-3" id="text-1-1">
<p>
The initial setup includes creating the <a href="https://www.python.org/">Python</a> environment with
the packages that the project needs and uses.
</p>

<dl class="org-dl">
<dt> <a href="http://matplotlib.org/">matplotlib</a> </dt><dd>plotting and image processing tools
</dd>
<dt> <a href="http://www.numpy.org/">NumPy</a> </dt><dd>foundational scientific computing library
</dd>
<dt> <a href="http://zulko.github.io/moviepy/">MoviePy</a> </dt><dd>video processing tools
</dd>
<dt> <a href="http://opencv.org/">OpenCV</a> </dt><dd>computer vision library
</dd>
</dl>

<p>
The <a href="https://github.com/">GitHub</a> <a href="https://github.com/dventimi/CarND-Advanced-Lane-Lines">repository</a> for this project contains an <a href="environment.yml">environment.yml</a>
file that can be used to create and activate a <a href="https://conda.io/docs/">Conda</a> environment
with these commands.
</p>

<div class="org-src-container">

<pre class="src src-sh">conda env create --file environment.yml --name CarND-Advanced-Lane-Lines
<span style="color: #e090d7;">source</span> activate CarND-Advanced-Lane-Lines
</pre>
</div>

<p>
Once activated this environment is used to launch Python in
whatever way one likes, such as a <a href="https://www.python.org/shell/">Python shell</a>, a <a href="https://ipython.org/">IPython shell</a>,
or a <a href="http://jupyter.org/">jupyter notebook</a>.  Having done that, the usual first step is
to import the packages that are used.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-deque" class="coderef-off"><span style="color: #b4fa70;">from</span> collections <span style="color: #b4fa70;">import</span> deque</span>
<span id="coderef-itertools" class="coderef-off"><span style="color: #b4fa70;">from</span> itertools <span style="color: #b4fa70;">import</span> groupby, islice, zip_longest, cycle, filterfalse</span>
<span style="color: #b4fa70;">from</span> moviepy.editor <span style="color: #b4fa70;">import</span> VideoFileClip
<span style="color: #b4fa70;">from</span> mpl_toolkits.axes_grid1 <span style="color: #b4fa70;">import</span> ImageGrid
<span id="coderef-profiler" class="coderef-off"><span style="color: #b4fa70;">import</span> cProfile</span>
<span style="color: #b4fa70;">import</span> cv2
<span style="color: #b4fa70;">import</span> glob
<span style="color: #b4fa70;">import</span> matplotlib
<span style="color: #b4fa70;">import</span> matplotlib.image <span style="color: #b4fa70;">as</span> mpimg
<span style="color: #b4fa70;">import</span> matplotlib.pyplot <span style="color: #b4fa70;">as</span> plt
<span style="color: #b4fa70;">import</span> numpy <span style="color: #b4fa70;">as</span> np
<span id="coderef-debug" class="coderef-off"><span style="color: #b4fa70;">import</span> pdb</span>
</pre>
</div>

<p>
Besides the third-party packages listed above, the project also
makes use of these standard-library library packages.
</p>

<dl class="org-dl">
<dt> <a href="#coderef-deque"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-deque');" onmouseout="CodeHighlightOff(this, 'coderef-deque');">deque</a> </dt><dd><a href="https://en.wikipedia.org/wiki/Circular_buffer">ring buffers</a> for moving averages
</dd>
<dt> <a href="#coderef-itertools"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-itertools');" onmouseout="CodeHighlightOff(this, 'coderef-itertools');">itertools</a> </dt><dd>handy for <a href="http://davidaventimiglia.com/python_generators.html">Python generators</a>
</dd>
<dt> <a href="#coderef-profiler"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-profiler');" onmouseout="CodeHighlightOff(this, 'coderef-profiler');">cProfile</a> </dt><dd>run-time <a href="https://docs.python.org/2/library/profile.html">optimization</a>
</dd>
<dt> <a href="#coderef-debug"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-debug');" onmouseout="CodeHighlightOff(this, 'coderef-debug');">pdb</a> </dt><dd>Python <a href="https://docs.python.org/3/library/pdb.html">debugger</a>
</dd>
</dl>
</div>
</div>


<div id="outline-container-sec-1-2" class="outline-3">
<h3 id="sec-1-2">Processing Pipeline</h3>
<div class="outline-text-3" id="text-1-2">
<p>
In order to detect lane lines in a video of a car driving on a
road, and then generate an annotated video with the detected lane
overlaid, we need an image processor that performs these two
tasks&#x2013;detection and annotation&#x2013;on every frame of the video.
That image processor encompasses a "processing pipeline."  
</p>

<p>
The pipeline depends on these preliminary tasks.
</p>

<ol class="org-ol">
<li>Camera Calibration
</li>
<li>Perspective Measurement
</li>
</ol>

<p>
Then, the pipeline applies these stages.
</p>

<ol class="org-ol">
<li>Distortion Correction
</li>
<li>Gradient and Color Thresholds
</li>
<li>Perspective Transform
</li>
<li>Lane-line Detection
</li>
</ol>

<p>
Let us examine these preliminary tasks and pipeline stages in
greater detail.
</p>
</div>

<div id="outline-container-sec-1-2-1" class="outline-4">
<h4 id="sec-1-2-1">Camera Calibration</h4>
<div class="outline-text-4" id="text-1-2-1">
<p>
<a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">Camera calibration</a> measures the distortion inherent in cameras
that utilize lenses so that the images taken with the camera can
be corrected by removing the distortion.  A standard way to do
this is to measure the distortion the camera imposes on standard
images of known geometry.  Checkerboard patterns are useful for
this tasks because of their high contrast, known geometry, and
regular pattern.
</p>

<p>
The <a href="#coderef-measure_distortion"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-measure_distortion');" onmouseout="CodeHighlightOff(this, 'coderef-measure_distortion');"><code>measure_distortion</code></a> function takes a Python <a href="https://docs.python.org/2/library/stdtypes.html#sequence-types-str-unicode-list-tuple-bytearray-buffer-xrange">sequence</a> of
checkerboard image filenames taken at different distances,
center-offsets, and orientations and applies the OpenCV
functions <a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findchessboardcorners"><code>findChessboardCorners</code></a> and <a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners"><code>drawChessboardCorners</code></a> to
identify corners in the images and highlight the corners.  Then,
the <a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera"><code>calibrateCamera</code></a> function measures the distortion.  This
function <a href="#coderef-measure_distortion_reval"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-measure_distortion_reval');" onmouseout="CodeHighlightOff(this, 'coderef-measure_distortion_reval');">returns</a> the distortion parameters and matrix, along
with a sequence of tuples with the original filenames and the
annotated images.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-measure_distortion" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">measure_distortion</span>(calibration_files):</span>
    <span style="color: #fcaf3e;">files</span> = calibration_files
    <span style="color: #fcaf3e;">objp</span> = np.zeros((9*6,3), np.float32)
    <span style="color: #fcaf3e;">objp</span>[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    <span style="color: #fcaf3e;">stage1</span> = <span style="color: #e090d7;">map</span>(<span style="color: #b4fa70;">lambda</span> x: (x,), cycle(files))
    <span style="color: #fcaf3e;">stage2</span> = <span style="color: #e090d7;">map</span>(<span style="color: #b4fa70;">lambda</span> x: x + (mpimg.imread(x[0]),), stage1)
    <span style="color: #fcaf3e;">stage3</span> = <span style="color: #e090d7;">map</span>(<span style="color: #b4fa70;">lambda</span> x: x + (cv2.findChessboardCorners(cv2.cvtColor(x[1], cv2.COLOR_RGB2GRAY), (9,6)),), stage2)
    <span style="color: #fcaf3e;">stage4</span> = <span style="color: #e090d7;">map</span>(<span style="color: #b4fa70;">lambda</span> x: x + (cv2.drawChessboardCorners(np.copy(x[1]), (9,6), *(x[2][::-1])),), stage3)
    <span style="color: #fcaf3e;">filenames</span>,<span style="color: #fcaf3e;">images</span>,<span style="color: #fcaf3e;">corners</span>,<span style="color: #fcaf3e;">annotated_images</span> = <span style="color: #e090d7;">zip</span>(*<span style="color: #e090d7;">filter</span>(<span style="color: #b4fa70;">lambda</span> x: x[2][0], islice(stage4, <span style="color: #e090d7;">len</span>(files))))
    <span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">imgpoints</span> = <span style="color: #e090d7;">zip</span>(*corners)
    <span style="color: #fcaf3e;">objpoints</span> = [objp <span style="color: #b4fa70;">for</span> i <span style="color: #b4fa70;">in</span> <span style="color: #e090d7;">range</span>(<span style="color: #e090d7;">len</span>(imgpoints))]
    <span style="color: #fcaf3e;">ret</span>, <span style="color: #fcaf3e;">mtx</span>, <span style="color: #fcaf3e;">dist</span>, <span style="color: #fcaf3e;">rvecs</span>, <span style="color: #fcaf3e;">tvecs</span> = cv2.calibrateCamera(objpoints, imgpoints, <span style="color: #e090d7;">list</span>(islice(stage2,1))[0][1].shape[:2:][::-1], <span style="color: #e9b2e3;">None</span>, <span style="color: #e9b2e3;">None</span>)
<span id="coderef-measure_distortion_reval" class="coderef-off">    <span style="color: #b4fa70;">return</span> mtx, dist, <span style="color: #e090d7;">zip</span>(filenames, annotated_images)</span>
</pre>
</div>

<p>
This function is used in subsequent distortion corrections.
</p>
</div>
</div>

<div id="outline-container-sec-1-2-2" class="outline-4">
<h4 id="sec-1-2-2">Distortion Correction</h4>
<div class="outline-text-4" id="text-1-2-2">
<p>
The <a href="#coderef-get_undistorter"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_undistorter');" onmouseout="CodeHighlightOff(this, 'coderef-get_undistorter');"><code>get_undistorter</code></a> function takes a sequence of calibration
checkerboard image filenames, applies the <code>measure_distortion</code>
function, and <a href="#coderef-get_undistorter_retval"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_undistorter_retval');" onmouseout="CodeHighlightOff(this, 'coderef-get_undistorter_retval');">returns</a> a new function.  The new function function
uses the OpenCV <a href="http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20undistort(InputArray%20src,%20OutputArray%20dst,%20InputArray%20cameraMatrix,%20InputArray%20distCoeffs,%20InputArray%20newCameraMatrix)"><code>undistort</code></a> function to remove distortion from
images taken with the same camera.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-get_undistorter" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">get_undistorter</span>(calibration_files):</span>
    <span style="color: #fcaf3e;">mtx</span>,<span style="color: #fcaf3e;">dist</span>,<span style="color: #fcaf3e;">annotated_images</span> = measure_distortion(calibration_files)
<span id="coderef-get_undistorter_retval" class="coderef-off">    <span style="color: #b4fa70;">return</span> <span style="color: #b4fa70;">lambda</span> x: cv2.undistort(x, mtx, dist, <span style="color: #e9b2e3;">None</span>, mtx), annotated_images</span>
</pre>
</div>

<p>
In the example shown below, we <a href="#coderef-get_fn"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_fn');" onmouseout="CodeHighlightOff(this, 'coderef-get_fn');">get</a> an "image undistorter"
function for a set of calibration images.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-get_fn" class="coderef-off"><span style="color: #fcaf3e;">undistort</span>,<span style="color: #fcaf3e;">annotated_images</span> = get_undistorter(glob.glob(<span style="color: #e9b96e;">"camera_cal/*.jpg"</span>))</span>
<span style="color: #fcaf3e;">fig</span> = plt.figure()
<span style="color: #fcaf3e;">grid</span> = ImageGrid(fig, 111, nrows_ncols=(4,4), axes_pad=0.0)

<span id="coderef-apply_fn" class="coderef-off"><span style="color: #b4fa70;">for</span> p <span style="color: #b4fa70;">in</span> <span style="color: #e090d7;">zip</span>(annotated_images, grid):</span>
    p[1].imshow(p[0][1])

fig.savefig(<span style="color: #e9b96e;">"output_images/annotated_calibration_images.jpg"</span>)
</pre>
</div>

<p>
The annotated calibration images are shown in the figure below.
</p>


<div class="figure">
<p><img src="output_images/annotated_calibration_images.jpg" alt="annotated_calibration_images.jpg" width="800px" />
</p>
</div>

<p>
As discussed shortly, the effects of image distortion can be
subtle and difficult notice with the naked eye.  It helps
therefore to apply it to examples where the effect will be more
vivid.  The first of the camera calibration images that we
recently used to <i>measure</i> the camera distortion is a good
candidate for <i>correcting</i> distortion.  The following figure has
the original, distorted image.
</p>


<div class="figure">
<p><img src="camera_cal/calibration1.jpg" alt="calibration1.jpg" width="800px" />
</p>
</div>

<p>
It should be evident at a minimum that there is radial
distortion as the horizontal and vertical lines&#x2014;which should
be straight&#x2014;are curved outward from the center.
</p>

<p>
Next we use the camera matrix and distortion coefficients
embedded with in the <code>undistort</code> function that we obtained in
order to correct for these effects.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span style="color: #fcaf3e;">fig</span> = plt.figure()
plt.imshow(undistort(mpimg.imread(<span style="color: #e9b96e;">"camera_cal/calibration1.jpg"</span>)))
fig.savefig(<span style="color: #e9b96e;">"output_images/undistorted_calibration1.jpg"</span>)
</pre>
</div>


<div class="figure">
<p><img src="output_images/undistorted_calibration1.jpg" alt="undistorted_calibration1.jpg" width="800px" />
</p>
</div>

<p>
Next, we show the effects of applying the image undistorter to a
sequence of 6 road images taken with this same camera.  These 6
images are a test sequence that will reappear many times through
the remainder of this discussion as other image processing steps
are taken up.
</p>

<p>
The <a href="#coderef-visualize"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-visualize');" onmouseout="CodeHighlightOff(this, 'coderef-visualize');"><code>visualize</code></a> function helps us view a gallery of test images
in "ganged up" layout, and this is helpful as we develop the
processing pipeline stages.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-visualize" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">visualize</span>(filename, a):</span>
    <span style="color: #fcaf3e;">fig</span>, <span style="color: #fcaf3e;">axes</span> = plt.subplots(2,3,figsize=(24,12),subplot_kw={<span style="color: #e9b96e;">'xticks'</span>:[],<span style="color: #e9b96e;">'yticks'</span>:[]})
    fig.subplots_adjust(hspace=0.03, wspace=0.05)
    <span style="color: #b4fa70;">for</span> p <span style="color: #b4fa70;">in</span> <span style="color: #e090d7;">zip</span>(<span style="color: #e090d7;">sum</span>(axes.tolist(),[]), a):
        p[0].imshow(p[1],cmap=<span style="color: #e9b96e;">'gray'</span>)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close()
</pre>
</div>

<p>
The 6 test images that we use repeatedly are shown in the figure
below, without any image processing at all.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/test_images.jpg"</span>,
          (mpimg.imread(f) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/test_images.jpg" alt="test_images.jpg" width="800px" />
</p>
</div>

<p>
These test images are shown again, only this time the image
undistorter that we acquired above now is used to remove
distortion introduced by the camera.  The effect is subtle and
difficult to notice, but close inspection shows that at least a
small amount of radial distortion is removed by this process.  
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/undistorted_test_images.jpg"</span>,
          (undistort(mpimg.imread(f)) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/undistorted_test_images.jpg" alt="undistorted_test_images.jpg" width="800px" />
</p>
</div>

<p>
Next, we move on to perspective measurement.
</p>
</div>
</div>

<div id="outline-container-sec-1-2-3" class="outline-4">
<h4 id="sec-1-2-3">Perspective Measurement</h4>
<div class="outline-text-4" id="text-1-2-3">
<p>
Perspective measurement applies to two-dimensional images taken
of three-dimensional scenes wherein objects of
interest&#x2013;typically planar objects like roads&#x2013;are oriented such
that their <a href="http://mathworld.wolfram.com/NormalVector.html">normal vector</a> is not parallel with the camera's line
of site.  Another way to put it is that the planar object is not
parallel with the <a href="https://en.wikipedia.org/wiki/Image_plane">image plane</a>.  While there undoubtedly are more
sophisticated, perhaps automated or semi-automated ways of doing
this, a tried-and-true method is to identify a non-rectilinear
region in the image that corresponds to the planar object of
interest (the road) and then map those to a corresponding
rectilinear region on the <a href="https://en.wikipedia.org/wiki/Image_plane">image plane</a>.  
</p>

<p>
The <a href="#coderef-measure_warp"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-measure_warp');" onmouseout="CodeHighlightOff(this, 'coderef-measure_warp');"><code>measure_warp</code></a> function helps measure perspective.  It takes
an image as a <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html">NumPy array</a> and displays the image to the user in
an interactive window.  The user only has to click four corners
in sequence for the source region and then close the interactive
window.  The <a href="#coderef-dst_region"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-dst_region');" onmouseout="CodeHighlightOff(this, 'coderef-dst_region');">destination region</a> on the <a href="https://en.wikipedia.org/wiki/Image_plane">image plane</a> for now is
<a href="#coderef-set_dst"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-set_dst');" onmouseout="CodeHighlightOff(this, 'coderef-set_dst');">hard-code</a> to a bounding box between the top and bottom of the
image and 300 pixels from the left edge and 300 pixels from the
right edge.  These values were obtained through experimentation,
and while they are not as sophisticated as giving the user
interactive control, they do have the virtue of being perfectly
rectilinear.  This is something that is difficult to achieve
manually.  Setting the src region coordinates, along with
drawing guidelines to aid the eye, is accomplished in an
<a href="#coderef-event_handler"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-event_handler');" onmouseout="CodeHighlightOff(this, 'coderef-event_handler');">event handler</a> function for mouse-click events.  The function
<a href="#coderef-measure_warp_retval"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-measure_warp_retval');" onmouseout="CodeHighlightOff(this, 'coderef-measure_warp_retval');">returns</a> the transformation matrix \(M\) and the inverse
transformation matrix \(M_{inv}\).  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-measure_warp" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">measure_warp</span>(img):</span>
    <span style="color: #fcaf3e;">top</span> = 0
    <span style="color: #fcaf3e;">bottom</span> = img.shape[0]
<span id="coderef-event_handler" class="coderef-off">    <span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">handler</span>(e):</span>
        <span style="color: #b4fa70;">if</span> <span style="color: #e090d7;">len</span>(src)&lt;4:
            plt.axhline(<span style="color: #e090d7;">int</span>(e.ydata), linewidth=2, color=<span style="color: #e9b96e;">'r'</span>)
            plt.axvline(<span style="color: #e090d7;">int</span>(e.xdata), linewidth=2, color=<span style="color: #e9b96e;">'r'</span>)
<span id="coderef-set_src" class="coderef-off">            src.append((<span style="color: #e090d7;">int</span>(e.xdata),<span style="color: #e090d7;">int</span>(e.ydata)))</span>
        <span style="color: #b4fa70;">if</span> <span style="color: #e090d7;">len</span>(src)==4:
<span id="coderef-set_dst" class="coderef-off">            dst.extend([(300,bottom),(300,top),(980,top),(980,bottom)])</span>
    <span style="color: #fcaf3e;">was_interactive</span> = matplotlib.is_interactive()
    <span style="color: #b4fa70;">if</span> <span style="color: #b4fa70;">not</span> matplotlib.is_interactive():
        plt.ion()
    <span style="color: #fcaf3e;">fig</span> = plt.figure()
    plt.imshow(img)
    <span style="color: #b4fa70;">global</span> src                                                            
    <span style="color: #b4fa70;">global</span> dst                                                            
<span id="coderef-src_region" class="coderef-off">    <span style="color: #fcaf3e;">src</span> = []</span>
<span id="coderef-dst_region" class="coderef-off">    <span style="color: #fcaf3e;">dst</span> = []</span>
    <span style="color: #fcaf3e;">cid1</span> = fig.canvas.mpl_connect(<span style="color: #e9b96e;">'button_press_event'</span>, handler)
    <span style="color: #fcaf3e;">cid2</span> = fig.canvas.mpl_connect(<span style="color: #e9b96e;">'close_event'</span>, <span style="color: #b4fa70;">lambda</span> e: e.canvas.stop_event_loop())
    fig.canvas.start_event_loop(timeout=-1)
<span id="coderef-getperspectivetransform" class="coderef-off">    <span style="color: #fcaf3e;">M</span> = cv2.getPerspectiveTransform(np.asfarray(src, np.float32), np.asfarray(dst, np.float32))</span>
    <span style="color: #fcaf3e;">Minv</span> = cv2.getPerspectiveTransform(np.asfarray(dst, np.float32), np.asfarray(src, np.float32))
    matplotlib.interactive(was_interactive)
<span id="coderef-measure_warp_retval" class="coderef-off">    <span style="color: #b4fa70;">return</span> M, Minv</span>
</pre>
</div>

<p>
Like with the <code>get_undistorter</code> function described above, we use
<a href="https://www.programiz.com/python-programming/closure">Python closures</a> to create a function generator called
<a href="#coderef-get_warpers"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_warpers');" onmouseout="CodeHighlightOff(this, 'coderef-get_warpers');"><code>get_warpers</code></a>, which measures the perspective, remembers the
transformation matrices, and then generate a new function that
uses OpenCV <a href="http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective"><code>warpPerspective</code></a> to transform a target image.  Note
that it actually <a href="#coderef-get_warpers_retval"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_warpers_retval');" onmouseout="CodeHighlightOff(this, 'coderef-get_warpers_retval');">generates</a> two functions, both to "warp" and
"unwarp" images.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-get_warpers" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">get_warpers</span>(corrected_image):</span>
    <span style="color: #fcaf3e;">M</span>, <span style="color: #fcaf3e;">Minv</span> = measure_warp(corrected_image)
    <span style="color: #b4fa70;">return</span> <span style="color: #b4fa70;">lambda</span> x: cv2.warpPerspective(x,
                                         M,
                                         x.shape[:2][::-1],
                                         flags=cv2.INTER_LINEAR), \
           <span style="color: #b4fa70;">lambda</span> x: cv2.warpPerspective(x,
                                         Minv,
                                         x.shape[:2][::-1],
<span id="coderef-get_warpers_retval" class="coderef-off">                                         flags=cv2.INTER_LINEAR), M, Minv</span>
</pre>
</div>

<p>
The following code illustrates how this is put into practice.
We get an image with the matplotlib <a href="http://matplotlib.org/api/image_api.html#matplotlib.image.imread"><code>imread</code></a> function, correct
for camera distortion using the <code>undistort</code> function we
generated with the <code>undistorter</code> function created above (after
camera calibration on checkerboard images), then use
<code>get_warpers</code> to generate both the <code>warp</code> and <code>unwarp</code>
functions.  It also returns the \(M\) and \(M_{inv}\) matrices as
<code>M</code> and <code>Minv</code> for good measure.
</p>

<div class="org-src-container">

<pre class="src src-python"><span style="color: #fcaf3e;">warp</span>,<span style="color: #fcaf3e;">unwarp</span>,<span style="color: #fcaf3e;">M</span>,<span style="color: #fcaf3e;">Minv</span> = get_warpers(undistort(mpimg.imread(<span style="color: #e9b96e;">"test_images/straight_lines2.jpg"</span>)))
</pre>
</div>

<p>
The next sequence of four figures illustrates the interactive
experience the user has in this operation, showing step-by-step
the orthogonal guidelines that appear.  The trapezoidal area
formed bout the outside bottom two corners and the inside top
two corners of the last figure defines the source region that is
then mapped to the target region.  Again, as discussed above the
target region is a rectangle running from the bottom of the
image to the top, 300 pixels in from the left edge and 300
pixels in from the right edge.
</p>


<div class="figure">
<p><img src="output_images/figure_3-1.png" alt="figure_3-1.png" width="800px" />
</p>
</div>


<div class="figure">
<p><img src="output_images/figure_3-2.png" alt="figure_3-2.png" width="800px" />
</p>
</div>


<div class="figure">
<p><img src="output_images/figure_3-3.png" alt="figure_3-3.png" width="800px" />
</p>
</div>


<div class="figure">
<p><img src="output_images/figure_3-4.png" alt="figure_3-4.png" width="800px" />
</p>
</div>

<p>
Equipped not just with an <code>undistort</code> function (obtained via
camera calibration) but also a <code>warp</code> (obtained via
perspective measurement) function, we can compose both functions
in the proper sequence (<code>undistort</code> then <code>warp</code>) and apply it to
our 6 test images.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/warped_undistorted_test_images.jpg"</span>,
          (warp(undistort(mpimg.imread(f))) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>

<p>
As you can see in the following gallery we now have a
"birds-eye" (i.e. top-down) view of the road for these 6 test
images.  Note also that the perspective transform has also had
the effect of shoving out of the frame much of the extraneous
details (sky, trees, guardrails, other cars).  This is
serendipitous as it saves us from having to apply a mask just to
the lane region.  
</p>


<div class="figure">
<p><img src="output_images/warped_undistorted_test_images.jpg" alt="warped_undistorted_test_images.jpg" width="800px" />
</p>
</div>

<p>
Camera calibration and perspective measurement are preliminary
steps that occur before applying the processing pipeline to
images taken from the video stream.  However, they are essential
and they enable the distortion correction and perspective
transformation steps which <i>are</i> part of the processing
pipeline.  Another set of essential pipeline steps involve
gradient ant color thresholds, discussed in the next sections.  
</p>
</div>
</div>

<div id="outline-container-sec-1-2-4" class="outline-4">
<h4 id="sec-1-2-4">Gradient and Color Thresholds</h4>
<div class="outline-text-4" id="text-1-2-4">
<p>
Next we develop a set of useful utility functions for scaling
images, taking gradients across them, isolating different color
channels, and generating binary images.
</p>

<p>
The <a href="#coderef-scale"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-scale');" onmouseout="CodeHighlightOff(this, 'coderef-scale');"><code>scale</code></a> function scales the values of NumPy image arrays to
arbitrary ranges (e.g., [0,1] or [0,255]).  The default range is
[0,255], and this is useful in order to give all images the same
scale.  Different operations (e.g., taking gradients, producing
binary images) can introduce different scales and it eases
combining and comparing images when they have the same scale.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-scale" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">scale</span>(img, factor=255.0):</span>
    <span style="color: #fcaf3e;">scale_factor</span> = np.<span style="color: #e090d7;">max</span>(img)/factor
    <span style="color: #b4fa70;">return</span> (img/scale_factor).astype(np.uint8)
</pre>
</div>

<p>
The <a href="#coderef-derivative"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-derivative');" onmouseout="CodeHighlightOff(this, 'coderef-derivative');"><code>derivative</code></a> function uses the OpenCV <a href="http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#sobel"><code>sobel</code></a> function to
apply the <a href="https://en.wikipedia.org/wiki/Sobel_operator">Sobel operator</a> in order to estimate derivatives in the
\(x\) and \(y\) directions across the image.  For good measure, it
also <a href="#coderef-derivative_retval"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-derivative_retval');" onmouseout="CodeHighlightOff(this, 'coderef-derivative_retval');">returns</a> both the <i>magnitude</i> and the <i>direction</i> of the
<a href="https://en.wikipedia.org/wiki/Gradient">gradient</a> computed from these derivative estimates.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-derivative" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">derivative</span>(img, sobel_kernel=3):</span>
    <span style="color: #fcaf3e;">derivx</span> = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    <span style="color: #fcaf3e;">derivy</span> = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    <span style="color: #fcaf3e;">gradmag</span> = np.sqrt(derivx**2 + derivy**2)
    <span style="color: #fcaf3e;">absgraddir</span> = np.arctan2(derivy, derivx)
<span id="coderef-derivative_retval" class="coderef-off">    <span style="color: #b4fa70;">return</span> scale(derivx), scale(derivy), scale(gradmag), absgraddir</span>
</pre>
</div>

<p>
The <a href="#coderef-grad"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-grad');" onmouseout="CodeHighlightOff(this, 'coderef-grad');"><code>grad</code></a> function adapts the <code>derivative</code> function to return
both the gradient <i>magnitude</i> and <i>direction</i>.  You might wonder
what this function adds to the <code>derivative</code> function, and that
is a valid consideration.  Largely it exists because the lecture
notes seemed to suggest that it's worthwhile to use different
kernel sizes for the Sobel operator when computing the gradient
direction.  In hindsight it's not clear this function really is
adding value and it may be removed in future versions.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-grad" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">grad</span>(img, k1=3, k2=15):</span>
<span id="coderef-grad_m" class="coderef-off">    <span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">g</span>,<span style="color: #fcaf3e;">_</span> = derivative(img, sobel_kernel=k1)</span>
<span id="coderef-grad_p" class="coderef-off">    <span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">p</span> = derivative(img, sobel_kernel=k2)</span>
    <span style="color: #b4fa70;">return</span> g,p
</pre>
</div>

<p>
The <a href="#coderef-hls_select"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-hls_select');" onmouseout="CodeHighlightOff(this, 'coderef-hls_select');"><code>hls_select</code></a> function is a convenience that fans out the
three channels of the <a href="https://en.wikipedia.org/wiki/HSL_and_HSV">HLS color-space</a> into separate NumPy
arrays.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-hls_select" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">hls_select</span>(img):</span>
    <span style="color: #fcaf3e;">hsv</span> = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.<span style="color: #e090d7;">float</span>)
    <span style="color: #fcaf3e;">h</span> = hsv[:,:,0]
    <span style="color: #fcaf3e;">l</span> = hsv[:,:,1]
    <span style="color: #fcaf3e;">s</span> = hsv[:,:,2]
    <span style="color: #b4fa70;">return</span> h,l,s
</pre>
</div>

<p>
The <a href="#coderef-rgb_select"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-rgb_select');" onmouseout="CodeHighlightOff(this, 'coderef-rgb_select');"><code>rgb_select</code></a> function is another convenience that returns
the three channels of the <a href="https://en.wikipedia.org/wiki/RGB_color_space">RGB color-space</a>.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-rgb_select" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">rgb_select</span>(img):</span>
    <span style="color: #fcaf3e;">rgb</span> = img
    <span style="color: #fcaf3e;">r</span> = rgb[:,:,0]
    <span style="color: #fcaf3e;">g</span> = rgb[:,:,1]
    <span style="color: #fcaf3e;">b</span> = rgb[:,:,2]
    <span style="color: #b4fa70;">return</span> r,g,b
</pre>
</div>

<p>
The <a href="#coderef-threshold"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-threshold');" onmouseout="CodeHighlightOff(this, 'coderef-threshold');"><code>threshold</code></a> function is a convenience that applies
<code>thresh_min</code> and <code>thresh_max</code> <i>min-max</i> values and logical
operations in order to obtain "binary" images.  Binary images
have activated pixels (non-zero values) for desired features.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-threshold" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">threshold</span>(img, thresh_min=0, thresh_max=255):</span>
    <span style="color: #fcaf3e;">binary_output</span> = np.zeros_like(img)
    <span style="color: #fcaf3e;">binary_output</span>[(img &gt;= thresh_min) &amp; (img &lt;= thresh_max)] = 1
    <span style="color: #b4fa70;">return</span> binary_output
</pre>
</div>

<p>
The <a href="#coderef-land_lor"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-land_lor');" onmouseout="CodeHighlightOff(this, 'coderef-land_lor');"><code>land</code></a> and <a href="#coderef-land_lor"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-land_lor');" onmouseout="CodeHighlightOff(this, 'coderef-land_lor');"><code>lor</code></a> functions are conveniences for combining
binary images, either with logical <a href="https://en.wikipedia.org/wiki/Logical_conjunction">conjunction</a> or <a href="https://en.wikipedia.org/wiki/Logical_disjunction">disjunction</a>,
respectively.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-land_lor" class="coderef-off"><span style="color: #fcaf3e;">land</span> = <span style="color: #b4fa70;">lambda</span> *x: np.logical_and.<span style="color: #e090d7;">reduce</span>(x)</span>
<span style="color: #fcaf3e;">lor</span> = <span style="color: #b4fa70;">lambda</span> *x: np.logical_or.<span style="color: #e090d7;">reduce</span>(x)
</pre>
</div>

<p>
There are various ways of doing this.  Another way is to stack
binary image arrays using the NumPy <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html"><code>stack</code></a> function and then
interleave various combinations of such interleavings along with
the NumPy <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.any.html#numpy-any"><code>any</code></a> function and <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html#numpy-all"><code>all</code></a> function.  It's a clever
approach, but I find that applying the NumPy <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_and.html#numpy-logical-and"><code>logical_and</code></a> and
<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_or.html#numpy-logical-or"><code>logical_or</code></a> functions as above leads to less typing.  
</p>

<p>
The <a href="#coderef-highlight"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-highlight');" onmouseout="CodeHighlightOff(this, 'coderef-highlight');"><code>highlight</code></a> function composes the color channel selection,
gradient estimation, binary threshold, logical composition, and
scaling operations to an input image in order to "highlight" the
desired features, such as lane lines.  Note that distortion
correction and perspective transformation are considered outside
the scope of this function.  In a real pipeline, those two
operations almost certainly should be applied to an image before
presenting it to the <a href="#coderef-highlight"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-highlight');" onmouseout="CodeHighlightOff(this, 'coderef-highlight');"><code>highlight</code></a> function.  In general, they
need not be, which can be useful during the exploratory phase of
pipeline development.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-highlight" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">highlight</span>(img):</span>
    <span style="color: #fcaf3e;">r</span>,<span style="color: #fcaf3e;">g</span>,<span style="color: #fcaf3e;">b</span> = rgb_select(img)
    <span style="color: #fcaf3e;">h</span>,<span style="color: #fcaf3e;">l</span>,<span style="color: #fcaf3e;">s</span> = hls_select(img)
    <span style="color: #fcaf3e;">o01</span> = threshold(r, 200, 255)
    <span style="color: #fcaf3e;">o02</span> = threshold(g, 200, 255)
    <span style="color: #fcaf3e;">o03</span> = threshold(s, 200, 255)
    <span style="color: #b4fa70;">return</span> scale(lor(land(o01,o02),o03))
</pre>
</div>

<p>
In fact, the highlight and undistort operations are combined
<i>without</i> perspective transform in the next gallery of 6 test
images.  This is an example of a common iteration pattern while
exploring pipeline options.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/binary_undistorted_test_images.jpg"</span>,
          (highlight(undistort(mpimg.imread(f))) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/binary_undistorted_test_images.jpg" alt="binary_undistorted_test_images.jpg" width="800px" />
</p>
</div>
</div>
</div>

<div id="outline-container-sec-1-2-5" class="outline-4">
<h4 id="sec-1-2-5">Perspective Transform</h4>
<div class="outline-text-4" id="text-1-2-5">
<p>
Armed with a pipeline which, based on the 6 test images, we
believe may be a good candidate for detecting lane lines, we
then see what the pipeline-processed test images look like after
transforming them to a "bird's-eye" view.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/warped_binary_undistorted_images.jpg"</span>,
          (warp(highlight(undistort(mpimg.imread(f)))) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/warped_binary_undistorted_images.jpg" alt="warped_binary_undistorted_images.jpg" width="800px" />
</p>
</div>
</div>
</div>

<div id="outline-container-sec-1-2-6" class="outline-4">
<h4 id="sec-1-2-6">Lane-Finding</h4>
<div class="outline-text-4" id="text-1-2-6">
<p>
Lane-line detection can be done somewhat laboriously&#x2013;but
perhaps more accurately&#x2013;using a "sliding window" technique.
Roughly, the algorithm implemented in
<a href="#coderef-detect_lines_sliding_window"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-detect_lines_sliding_window');" onmouseout="CodeHighlightOff(this, 'coderef-detect_lines_sliding_window');"><code>detect_lines_sliding_window</code></a> below has these steps, also
discussed in the code comments.
</p>

<ol class="org-ol">
<li>Take a histogram across the bottom of the image.
</li>
<li>Find the histogram peaks to identify the lane lines at the
bottom of the image.
</li>
<li>Divide the image into a vertical stack of narrow horizontal
slices.
</li>
<li>Select activated pixels (remember, the input is a binary
image) only in a "neighborhood" of our current estimate of
the lane position.  This neighborhood is the "sliding
window."  To bootstrap the process, our initial estimate of
the lane line location is taken from the histogram peak steps
listed above.  Essentially, we are removing "outliers"
</li>
<li>Estimate the new lane-line location for this window from the
mean of the pixels falling within the sliding window.
</li>
<li>March vertically up through the stack, repeating this process.
</li>
<li>Select all activated pixels within all of our sliding windows.
</li>
<li>Fit a quadratic function to these selected pixels, obtaining
model parameters.
</li>
</ol>

<p>
The model parameters essentially represent the detected
lane-line.  We do this both for the left and right lines.
Moreover, we also perform a few somewhat ancillary operations
while we're at it.
</p>

<ol class="org-ol">
<li>Draw the sliding windows, the selected pixels, and the
modeled quadratic curve onto a copy of the image.
</li>
<li>Recompute the function fit after scaling the pixel locations
to real world values, then use these model fit parameters to
compute a real-world radius of curvature for both lanes.
</li>
</ol>

<p>
The function <a href="#coderef-detect_lines_sliding_window"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-detect_lines_sliding_window');" onmouseout="CodeHighlightOff(this, 'coderef-detect_lines_sliding_window');"><code>detect_lines_sliding_window</code></a> returns quite a few values:
</p>

<ol class="org-ol">
<li>left lane fit parameters
</li>
<li>right lane fit parameters
</li>
<li>left lane fit residuals
</li>
<li>right lane fit residuals
</li>
<li>left lane real-world radius (in meters)
</li>
<li>right lane real-world radius (in meters)
</li>
<li>annotated image, with sliding windows, selected pixels, and
modeled curves
</li>
</ol>

<p>
The code for this function is shown here. 
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-detect_lines_sliding_window" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">detect_lines_sliding_window</span>(warped_binary):</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Assuming you have created a warped binary image called "warped_binary"</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Take a histogram of the bottom half of the image</span>
    <span style="color: #fcaf3e;">histogram</span> = np.<span style="color: #e090d7;">sum</span>(warped_binary[warped_binary.shape[0]/2:,:], axis=0)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Create an output image to draw on and  visualize the result</span>
    <span style="color: #fcaf3e;">out_img</span> = np.dstack((warped_binary, warped_binary, warped_binary))*255
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Find the peak of the left and right halves of the histogram</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">These will be the starting point for the left and right lines</span>
    <span style="color: #fcaf3e;">midpoint</span> = np.<span style="color: #e090d7;">int</span>(histogram.shape[0]/2)
    <span style="color: #fcaf3e;">leftx_base</span> = np.argmax(histogram[:midpoint])
    <span style="color: #fcaf3e;">rightx_base</span> = np.argmax(histogram[midpoint:]) + midpoint
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Choose the number of sliding windows</span>
    <span style="color: #fcaf3e;">nwindows</span> = 9
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Set height of windows</span>
    <span style="color: #fcaf3e;">window_height</span> = np.<span style="color: #e090d7;">int</span>(warped_binary.shape[0]/nwindows)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Identify the x and y positions of all nonzero pixels in the image</span>
    <span style="color: #fcaf3e;">nonzero</span> = warped_binary.nonzero()
    <span style="color: #fcaf3e;">nonzeroy</span> = np.array(nonzero[0])
    <span style="color: #fcaf3e;">nonzerox</span> = np.array(nonzero[1])
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Current positions to be updated for each window</span>
    <span style="color: #fcaf3e;">leftx_current</span> = leftx_base
    <span style="color: #fcaf3e;">rightx_current</span> = rightx_base
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Set the width of the windows +/- margin</span>
    <span style="color: #fcaf3e;">margin</span> = 100
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Set minimum number of pixels found to recenter window</span>
    <span style="color: #fcaf3e;">minpix</span> = 50
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Create empty lists to receive left and right lane pixel indices</span>
    <span style="color: #fcaf3e;">left_lane_inds</span> = []
    <span style="color: #fcaf3e;">right_lane_inds</span> = []
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Step through the windows one by one</span>
    <span style="color: #b4fa70;">for</span> window <span style="color: #b4fa70;">in</span> <span style="color: #e090d7;">range</span>(nwindows):
        <span style="color: #73d216;"># </span><span style="color: #73d216;">Identify window boundaries in x and y (and right and left)</span>
        <span style="color: #fcaf3e;">win_y_low</span> = warped_binary.shape[0] - (window+1)*window_height
        <span style="color: #fcaf3e;">win_y_high</span> = warped_binary.shape[0] - window*window_height
        <span style="color: #fcaf3e;">win_xleft_low</span> = leftx_current - margin
        <span style="color: #fcaf3e;">win_xleft_high</span> = leftx_current + margin
        <span style="color: #fcaf3e;">win_xright_low</span> = rightx_current - margin
        <span style="color: #fcaf3e;">win_xright_high</span> = rightx_current + margin
        <span style="color: #73d216;"># </span><span style="color: #73d216;">Draw the windows on the visualization image</span>
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        <span style="color: #73d216;"># </span><span style="color: #73d216;">Identify the nonzero pixels in x and y within the window</span>
        <span style="color: #fcaf3e;">good_left_inds</span> = ((nonzeroy &gt;= win_y_low) &amp; (nonzeroy &lt; win_y_high) &amp; (nonzerox &gt;= win_xleft_low) &amp; (nonzerox &lt; win_xleft_high)).nonzero()[0]
        <span style="color: #fcaf3e;">good_right_inds</span> = ((nonzeroy &gt;= win_y_low) &amp; (nonzeroy &lt; win_y_high) &amp; (nonzerox &gt;= win_xright_low) &amp; (nonzerox &lt; win_xright_high)).nonzero()[0]
        <span style="color: #73d216;"># </span><span style="color: #73d216;">Append these indices to the lists</span>
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        <span style="color: #73d216;"># </span><span style="color: #73d216;">If you found &gt; minpix pixels, recenter next window on their mean position</span>
        <span style="color: #b4fa70;">if</span> <span style="color: #e090d7;">len</span>(good_left_inds) &gt; minpix:
            <span style="color: #fcaf3e;">leftx_current</span> = np.<span style="color: #e090d7;">int</span>(np.mean(nonzerox[good_left_inds]))
        <span style="color: #b4fa70;">if</span> <span style="color: #e090d7;">len</span>(good_right_inds) &gt; minpix:        
            <span style="color: #fcaf3e;">rightx_current</span> = np.<span style="color: #e090d7;">int</span>(np.mean(nonzerox[good_right_inds]))
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Concatenate the arrays of indices</span>
    <span style="color: #fcaf3e;">left_lane_inds</span> = np.concatenate(left_lane_inds)
    <span style="color: #fcaf3e;">right_lane_inds</span> = np.concatenate(right_lane_inds)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Extract left and right line pixel positions</span>
    <span style="color: #fcaf3e;">leftx</span> = nonzerox[left_lane_inds]
    <span style="color: #fcaf3e;">lefty</span> = nonzeroy[left_lane_inds] 
    <span style="color: #fcaf3e;">rightx</span> = nonzerox[right_lane_inds]
    <span style="color: #fcaf3e;">righty</span> = nonzeroy[right_lane_inds] 
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Fit a second order polynomial to each</span>
    <span style="color: #fcaf3e;">left_fit</span>,<span style="color: #fcaf3e;">left_res</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span> = np.polyfit(lefty, leftx, 2, full=<span style="color: #e9b2e3;">True</span>)
    <span style="color: #fcaf3e;">right_fit</span>,<span style="color: #fcaf3e;">right_res</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span> = np.polyfit(righty, rightx, 2, full=<span style="color: #e9b2e3;">True</span>)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Generate x and y values for plotting</span>
    <span style="color: #fcaf3e;">ploty</span> = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
    <span style="color: #fcaf3e;">left_fitx</span> = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    <span style="color: #fcaf3e;">right_fitx</span> = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    <span style="color: #fcaf3e;">out_img</span>[ploty.astype(<span style="color: #e9b96e;">'int'</span>),left_fitx.astype(<span style="color: #e9b96e;">'int'</span>)] = [0, 255, 255]
    <span style="color: #fcaf3e;">out_img</span>[ploty.astype(<span style="color: #e9b96e;">'int'</span>),right_fitx.astype(<span style="color: #e9b96e;">'int'</span>)] = [0, 255, 255]
    <span style="color: #fcaf3e;">y_eval</span> = warped_binary.shape[0]
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Define conversions in x and y from pixels space to meters</span>
    <span style="color: #fcaf3e;">ym_per_pix</span> = 30/720 <span style="color: #73d216;"># </span><span style="color: #73d216;">meters per pixel in y dimension</span>
    <span style="color: #fcaf3e;">xm_per_pix</span> = 3.7/700 <span style="color: #73d216;"># </span><span style="color: #73d216;">meters per pixel in x dimension</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Fit new polynomials to x,y in world space</span>
    <span style="color: #fcaf3e;">left_fit_cr</span> = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    <span style="color: #fcaf3e;">right_fit_cr</span> = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Calculate the new radii of curvature</span>
    <span style="color: #fcaf3e;">left_curverad</span> = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    <span style="color: #fcaf3e;">right_curverad</span> = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
<span id="coderef-sliding_window_retval" class="coderef-off">    <span style="color: #b4fa70;">return</span> left_fit, right_fit, np.sqrt(left_fit[1]/<span style="color: #e090d7;">len</span>(leftx)), np.sqrt(right_fit[1]/<span style="color: #e090d7;">len</span>(rightx)), left_curverad, right_curverad, out_img</span>
</pre>
</div>

<p>
The following figures shows the annotated image resulting from
applying this particular lane-finding algorithm to our 6 test
images, after distortion correction, highlighting, and
perspective transformation.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/detected_lines_test_images.jpg"</span>,
          (detect_lines_sliding_window(warp(highlight(undistort(mpimg.imread(f)))))[6] <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/detected_lines_test_images.jpg" alt="detected_lines_test_images.jpg" width="800px" />
</p>
</div>

<p>
Armed with a good estimate for the current lane-line locations
and with the observation that the lanes do not change
dramatically from one frame to the next, we can implement an
optimization.  Recall that the <i>only reason</i> for the sliding
window algorithm is to remove outliers.  If we were content just
to fit all of the pixels, good or bad, we would only need to
divide the frame into a left half and a right half and then fit
the quadratic curves straight away.  However, guided by the
lecture we chose to remove outliers.  That requires a good guess
for where the lane line is, which almost inevitably leads us to
the sliding window technique.
</p>

<p>
The <a href="#coderef-detect_lines"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-detect_lines');" onmouseout="CodeHighlightOff(this, 'coderef-detect_lines');"><code>detect_lines</code></a> function takes <code>left_fit</code> and <code>right_fit</code>
arguments, which are good estimates of the model fit parameters
obtained from the previous video frame.  It then selects pixels
in the neighborhood of the curve computed for these parameters,
and fits new parameters for the current frame from the selected
pixels.  Thus, it avoids the labor of the sliding window
technique so long as one already has a good estimate of the
model fit parameters.  Note that, because this function does
<i>not</i> apply the sliding window technique, it cannot draw the
sliding windows.  Therefore, the last parameter returned is
<code>None</code>.  
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-detect_lines" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">detect_lines</span>(warped_binary, left_fit, right_fit):</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">from the next frame of video (also called "binary_warped")</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">It's now much easier to find line pixels!</span>
    <span style="color: #fcaf3e;">nonzero</span> = warped_binary.nonzero()
    <span style="color: #fcaf3e;">nonzeroy</span> = np.array(nonzero[0])
    <span style="color: #fcaf3e;">nonzerox</span> = np.array(nonzero[1])
    <span style="color: #fcaf3e;">margin</span> = 100
    <span style="color: #fcaf3e;">left_lane_inds</span> = ((nonzerox &gt; (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &amp; (nonzerox &lt; (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    <span style="color: #fcaf3e;">right_lane_inds</span> = ((nonzerox &gt; (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &amp; (nonzerox &lt; (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Again, extract left and right line pixel positions</span>
    <span style="color: #fcaf3e;">leftx</span> = nonzerox[left_lane_inds]
    <span style="color: #fcaf3e;">lefty</span> = nonzeroy[left_lane_inds] 
    <span style="color: #fcaf3e;">rightx</span> = nonzerox[right_lane_inds]
    <span style="color: #fcaf3e;">righty</span> = nonzeroy[right_lane_inds]
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Fit a second order polynomial to each</span>
    <span style="color: #fcaf3e;">left_fit</span>,<span style="color: #fcaf3e;">left_res</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span> = np.polyfit(lefty, leftx, 2, full=<span style="color: #e9b2e3;">True</span>)
    <span style="color: #fcaf3e;">right_fit</span>,<span style="color: #fcaf3e;">right_res</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span>,<span style="color: #fcaf3e;">_</span> = np.polyfit(righty, rightx, 2, full=<span style="color: #e9b2e3;">True</span>)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Generate x and y values for plotting</span>
    <span style="color: #fcaf3e;">ploty</span> = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
    <span style="color: #fcaf3e;">left_fitx</span> = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    <span style="color: #fcaf3e;">right_fitx</span> = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    <span style="color: #fcaf3e;">y_eval</span> = warped_binary.shape[0]
<span id="coderef-convert" class="coderef-off">    <span style="color: #73d216;"># </span><span style="color: #73d216;">Define conversions in x and y from pixels space to meters</span></span>
    <span style="color: #fcaf3e;">ym_per_pix</span> = 30/720 <span style="color: #73d216;"># </span><span style="color: #73d216;">meters per pixel in y dimension</span>
    <span style="color: #fcaf3e;">xm_per_pix</span> = 3.7/700 <span style="color: #73d216;"># </span><span style="color: #73d216;">meters per pixel in x dimension</span>
<span id="coderef-newfit" class="coderef-off">    <span style="color: #73d216;"># </span><span style="color: #73d216;">Fit new polynomials to x,y in world space</span></span>
<span id="coderef-radisfit" class="coderef-off">    <span style="color: #fcaf3e;">left_fit_cr</span> = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)</span>
    <span style="color: #fcaf3e;">right_fit_cr</span> = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
<span id="coderef-curvecalc" class="coderef-off">    <span style="color: #73d216;"># </span><span style="color: #73d216;">Calculate the new radii of curvature</span></span>
    <span style="color: #fcaf3e;">left_curverad</span> = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    <span style="color: #fcaf3e;">right_curverad</span> = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    <span style="color: #b4fa70;">return</span> left_fit, right_fit, np.sqrt(left_fit[1]/<span style="color: #e090d7;">len</span>(leftx)), np.sqrt(right_fit[1]/<span style="color: #e090d7;">len</span>(rightx)), left_curverad, right_curverad, <span style="color: #e9b2e3;">None</span>
</pre>
</div>

<p>
Note in the function above how the radius of curvature is
calculated for the two lanes.  <a href="#coderef-convert"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-convert');" onmouseout="CodeHighlightOff(this, 'coderef-convert');">First</a>, constants establish a
conversion between pixel coordinates in the \(x\) and \(y\)
directions and corresponding real-world coordinates (in meters)
in the \(x\) and \(z\) direction.  By \(z\) direction I mean depth
into the frame.  This is an important point, because we must
account for the fact that the three-dimensional real-world image
has been warped by the perspective transform into a
two-dimensional pixel-space image.  <a href="#coderef-newfit"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-newfit');" onmouseout="CodeHighlightOff(this, 'coderef-newfit');">Second</a>, we fit our model
again, this time after converting our pixel coordinates into
real-world values.  This is important!  A simple conversion of
radius-of-curvature estimates taken from our original fit would
not be correct, because that fit does not account for the
warping between the three-dimensional real world and the
two-dimensional pixel-space of the image plane.  <a href="#coderef-curvecal"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-curvecal');" onmouseout="CodeHighlightOff(this, 'coderef-curvecal');">Third</a>, for the
left and right lanes we calculate the radius of curvature using
the model fit parameters, according to this formula, where \(A\)
and \(B\) are fit parameters.
</p>

<p>
\[ R_{curve} = \frac{\left(1 + \left(2 A y +
      B\right)^2\right)^{3/2}}{\left| 2 A \right|} \]
</p>

<p>
The <a href="#coderef-draw_lane"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-draw_lane');" onmouseout="CodeHighlightOff(this, 'coderef-draw_lane');"><code>draw_lane</code></a> function takes a distortion-corrected unwarped
image, a warped binary image like, model fit parameters,
real-world lane-curvature estimates in meters, and an image
unwarping function.  It uses these to annotate the undistorted
image with a depiction of the lane, along with vital statistics
on the left and right lane curvature, and the position of the
camera with respect to the center of the lane (taken as the mean
of the two lane locations).
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-draw_lane" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">draw_lane</span>(undistorted, warped_binary, l_fit, r_fit, l_rad, r_rad, unwarp):</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Create an image to draw the lines on</span>
    <span style="color: #fcaf3e;">warp_zero</span> = np.zeros_like(warped_binary).astype(np.uint8)
    <span style="color: #fcaf3e;">color_warp</span> = np.dstack((warp_zero, warp_zero, warp_zero))
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Generate x and y values for plotting</span>
    <span style="color: #fcaf3e;">ploty</span> = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
    <span style="color: #fcaf3e;">l_fitx</span> = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    <span style="color: #fcaf3e;">r_fitx</span> = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Recast the x and y points into usable format for cv2.fillPoly()</span>
    <span style="color: #fcaf3e;">pts_left</span> = np.array([np.transpose(np.vstack([l_fitx, ploty]))])
    <span style="color: #fcaf3e;">pts_right</span> = np.array([np.flipud(np.transpose(np.vstack([r_fitx, ploty])))])
    <span style="color: #fcaf3e;">pts</span> = np.hstack((pts_left, pts_right))
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Draw the lane onto the warped_binary blank image</span>
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Warp the blank back to original image space using inverse perspective matrix (Minv)</span>
    <span style="color: #73d216;"># </span><span style="color: #73d216;">newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) </span>
    <span style="color: #fcaf3e;">newwarp</span> = unwarp(color_warp)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Combine the result with the original image</span>
    <span style="color: #fcaf3e;">result</span> = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    <span style="color: #73d216;"># </span><span style="color: #73d216;">Annotate image with lane curvature estimates</span>
    cv2.putText(result, <span style="color: #e9b96e;">"L. Curvature: %.2f km"</span> % (l_rad/1000), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.putText(result, <span style="color: #e9b96e;">"R. Curvature: %.2f km"</span> % (r_rad/1000), (50,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
<span id="coderef-roadpos" class="coderef-off">    <span style="color: #73d216;"># </span><span style="color: #73d216;">Annotate image with position estimate</span></span>
    cv2.putText(result, <span style="color: #e9b96e;">"C. Position: %.2f m"</span> % ((np.average((l_fitx + r_fitx)/2) - warped_binary.shape[1]//2)*3.7/700), (50,110), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    <span style="color: #b4fa70;">return</span> result
</pre>
</div>

<p>
Note in the function above how we <a href="#coderef-roadpos"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-roadpos');" onmouseout="CodeHighlightOff(this, 'coderef-roadpos');">annotate</a> the image with an
estimate of the position of the car with respect to the center
of the road.  It is a simple average of the pixel coordinates of
the two lanes at the bottom of the image, minus the pixel
coordinate of the image center, then scaled to a real-world
value (meters).  Note that we do <i>not</i> need the second curve fit
in real-world coordinates that was done in the two
lane-detecting functions to do this.  Because we are estimating
the position at the <i>bottom</i> of the image frame, the horizontal
direction only comes into play and we only need account for \(x\)
coordinates.  We had to perform the second fit for the radius of
curvature calculation to compensate for the warping of the
image, but that warping <i>only</i> relates the \(z\) direction in the
three-dimensional world and the \(y\) direction in the image
plane.  It plays no role in calculating the car position, but
<i>only</i> if we assume that position is to be taken at the bottom
of the image.
</p>

<p>
Note also that as we annotate the image with the radius of
curvature for the left and right lanes, we divide the
distances, which were calculated in meters, by a factor of 1000
in order to present them in kilometers.  Given the geometry of
the problem and the distances involved, I argue that kilometers
and not meters are the natural scale length.  Distances in
meters can be provided upon request, or simply calculated in the
reader's head.
</p>

<p>
With those notes, finally we can move on to the full processing
pipeline.  
</p>

<p>
The <a href="#coderef-get_processor"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-get_processor');" onmouseout="CodeHighlightOff(this, 'coderef-get_processor');"><code>get_processor</code></a> function returns a "processor" function.  A
processor function embodies <i>all</i> of the steps of the pipeline
outlined above:
</p>

<ol class="org-ol">
<li>Distortion Correction
</li>
<li>Perspective Transformation
</li>
<li>Lane-line detection <i>with</i> bootstrapping
</li>
<li>Radius of curvature and vehicle position calculations
</li>
<li>Image annotation with drawn lane lines and vital statistics
</li>
</ol>

<p>
One other thing that this function does is this.  It takes a
weighted average of some number of recent frames, along with the
current frame.  This removes "jitter" from the lanes and values
on the video streams, and adds robustness against bad detections
on individual frames.  It uses <code>dequeue</code> to create "ring
buffers" for the <a href="#coderef-buffer_1"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-buffer_1');" onmouseout="CodeHighlightOff(this, 'coderef-buffer_1');">left lane parameters</a>, <a href="#coderef-buffer_2"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-buffer_2');" onmouseout="CodeHighlightOff(this, 'coderef-buffer_2');">right lane parameters</a>,
<a href="#coderef-buffer_3"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-buffer_3');" onmouseout="CodeHighlightOff(this, 'coderef-buffer_3');">left lane radius</a>, and <a href="#coderef-buffer_4"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-buffer_4');" onmouseout="CodeHighlightOff(this, 'coderef-buffer_4');">right lane radius</a>.  The buffers can be of
any size, though the default has 10 slots.  Note that a buffer
size of 1 essentially computes no average at all.  Weighted
averages are taken across these buffers.  The weights could be
taken from any function, simple or complex, that is appropriate
for the situation.  In practice I did not try for anything
complicated, and used a simple <a href="#coderef-weights"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-weights');" onmouseout="CodeHighlightOff(this, 'coderef-weights');">linear</a> weighting scheme:  older
frames have strictly linearly less weight.
</p>

<div class="org-src-container">

<pre class="src src-python"><span id="coderef-get_processor" class="coderef-off"><span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">get_processor</span>(nbins=10):</span>
    <span style="color: #fcaf3e;">bins</span> = nbins
<span id="coderef-buffer_1" class="coderef-off">    <span style="color: #fcaf3e;">l_params</span> = deque(maxlen=bins)</span>
<span id="coderef-buffer_2" class="coderef-off">    <span style="color: #fcaf3e;">r_params</span> = deque(maxlen=bins)</span>
<span id="coderef-buffer_3" class="coderef-off">    <span style="color: #fcaf3e;">l_radius</span> = deque(maxlen=bins)</span>
<span id="coderef-buffer_4" class="coderef-off">    <span style="color: #fcaf3e;">r_radius</span> = deque(maxlen=bins)</span>
<span id="coderef-weights" class="coderef-off">    <span style="color: #fcaf3e;">weights</span> = np.arange(1,bins+1)/bins</span>
    <span style="color: #b4fa70;">def</span> <span style="color: #fce94f;">process_image</span>(img0):
        <span style="color: #fcaf3e;">undistorted</span> = undistort(img0)
        <span style="color: #fcaf3e;">warped_binary</span> = warp(highlight(undistorted))
        <span style="color: #fcaf3e;">l_fit</span>, <span style="color: #fcaf3e;">r_fit</span>, <span style="color: #fcaf3e;">l_res</span>, <span style="color: #fcaf3e;">r_res</span>, <span style="color: #fcaf3e;">l_curverad</span>, <span style="color: #fcaf3e;">r_curverad</span>, <span style="color: #fcaf3e;">_</span> = detect_lines_sliding_window(warped_binary) <span style="color: #b4fa70;">if</span> <span style="color: #e090d7;">len</span>(l_params)==0 <span style="color: #b4fa70;">else</span> detect_lines(warped_binary,np.average(l_params,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]), np.average(r_params,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]))
        l_params.append(l_fit)
        r_params.append(r_fit)
        l_radius.append(l_curverad)
        r_radius.append(r_curverad)
        <span style="color: #fcaf3e;">annotated_image</span> = draw_lane(undistorted,
                                    warped_binary,
                                    np.average(l_params,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]),
                                    np.average(r_params,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]),
                                    np.average(l_radius,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]),
                                    np.average(r_radius,0,weights[-<span style="color: #e090d7;">len</span>(l_params):]),
                                    unwarp)
        <span style="color: #b4fa70;">return</span> annotated_image
    <span style="color: #b4fa70;">return</span> process_image
</pre>
</div>

<p>
Equipped with a bona-fide image processor, the very one we use
on the video stream we can examine its effect on our 6 test images.
</p>

<div class="org-src-container">

<pre class="src src-python">visualize(<span style="color: #e9b96e;">"output_images/drawn_lanes_test_images.jpg"</span>, 
          (get_processor(1)(mpimg.imread(f)) <span style="color: #b4fa70;">for</span> f <span style="color: #b4fa70;">in</span> cycle(glob.glob(<span style="color: #e9b96e;">"test_images/test*.jpg"</span>))))
</pre>
</div>


<div class="figure">
<p><img src="output_images/drawn_lanes_test_images.jpg" alt="drawn_lanes_test_images.jpg" width="800px" />
</p>
</div>

<p>
Finally, generate a new processor and apply it to the video
stream.  We generate a new processor in order to give it a
different buffer size for the ring buffers supporting the
weighted averages.  For the video stream, the ring buffers have
50 slots, not 10.  Since the video stream is at 25 frames per
second, this constitutes a full 2 second window for the weighted
average.  That may seem like a lot, and we <i>do</i> have to be
careful not to push it too far.  There is a trade-off between
the smoothness and robustness added by the weighted average, and
a stiffness to the model that may cause it to lag on sharp
turns.  In practice, however, the weighted average quickly
deweights older frames, and in experimentation no deleterious
effects were noticed with a set of 50-slot ring buffers.
</p>

<div class="org-src-container">

<pre class="src src-python"><span style="color: #fcaf3e;">in_clip</span> = VideoFileClip(<span style="color: #e9b96e;">"project_video.mp4"</span>)
<span style="color: #fcaf3e;">out_clip</span> = in_clip.fl_image(get_processor(50))
cProfile.run(<span style="color: #e9b96e;">'out_clip.write_videofile("output_images/project_output.mp4", audio=False)'</span>, <span style="color: #e9b96e;">'restats'</span>)
</pre>
</div>

<p>
We can see the result for the project video in the following
video clip.
</p>

<iframe width="800" height="450" src="https://www.youtube.com/embed/xuDNjYzcjzs" frameborder="0" allowfullscreen></iframe>
</div>
</div>
</div>

<div id="outline-container-sec-1-3" class="outline-3">
<h3 id="sec-1-3">Discussion</h3>
<div class="outline-text-3" id="text-1-3">
<p>
This was a <i>very</i> challenging project, perhaps the most
challenging so far in this course.  
</p>
</div>

<div id="outline-container-sec-1-3-1" class="outline-4">
<h4 id="sec-1-3-1">What Worked Well</h4>
<div class="outline-text-4" id="text-1-3-1">
</div><ul class="org-ul"><li><a id="sec-1-3-1-1" name="sec-1-3-1-1"></a>Alternate Color-Spaces<br  /><div class="outline-text-5" id="text-1-3-1-1">
<p>
If the reader refers back to the <a href="#coderef-highlight"class="coderef" onmouseover="CodeHighlightOn(this, 'coderef-highlight');" onmouseout="CodeHighlightOff(this, 'coderef-highlight');"><code>highlight</code></a> function
described above, and which is a key function that combines
various aspects of image analysis together in order to
highlight the lane lines, he or she should notice certain
things.  In particular, it only uses color-spaces:  RBG and
HLS, and within those, only certain channels.  
</p>

<p>
In the exploratory phase of this project, it seemed that in
the RGB color-space, the Red (R) and Green (G) colors
independently were somewhat effective in picking out lane
lines and better when combined with an <i>AND</i> operation.  This
surprised me somewhat, and still warrants further
investigation.  The drawback was that while these channels
worked well in good lighting conditions, they performed poorly
in shadows.  
</p>

<p>
Moreover, the Saturation (S) channel in the HLS color-space
also was <i>very</i> effective in highlighting lines under various
lighting conditions.  Its drawback is that it highlights too
many other features as well, like other cars and
discolorations on the road.  
</p>

<p>
Finally, slicing out and applying thresholds to color-spaces
seems to be a relatively inexpensive operation
computationally, which is important for rapid iteration.
</p>
</div>
</li>

<li><a id="sec-1-3-1-2" name="sec-1-3-1-2"></a>Color Thresholding<br  /><div class="outline-text-5" id="text-1-3-1-2">
<p>
Naturally, along with both color and gradient computation one
typically will apply a threshold in order to obtain a binary
image with "activated" pixels associated with lane-lines.
This worked well, of course, but more important judicious use
of thresholds was somewhat effective in mitigating the
spurious features that the color-spaces brought in, such as
road discolorations.
</p>
</div>
</li>

<li><a id="sec-1-3-1-3" name="sec-1-3-1-3"></a>Perspective Transform<br  /><div class="outline-text-5" id="text-1-3-1-3">
<p>
Of course, performing a perspective transform to a bird's-eye
view is almost a necessary component of a project like this.
However, it also had another unexpected benefit.  As alluded
to above, it naturally shoves portions of the image outside of
the trapezoidal source region <i>outside</i> the frame when the
transform is applied.  I had anticipated a need for a masking
operation on the image, but found that I did not need it as
the perspective transform naturally did most or all of the
masking for me.  
</p>
</div>
</li>

<li><a id="sec-1-3-1-4" name="sec-1-3-1-4"></a>Lane Detection<br  /><div class="outline-text-5" id="text-1-3-1-4">
<p>
I adapted both the sliding window and non-sliding window lane
detection algorithms almost exactly as they were presented in
the lecture notes, and they worked perfectly, without a
hitch.  
</p>
</div>
</li>

<li><a id="sec-1-3-1-5" name="sec-1-3-1-5"></a>Radius-of-Curvature and Car Position Calculation<br  /><div class="outline-text-5" id="text-1-3-1-5">
<p>
Likewise, I applied the radius-of-curvature calculation almost
exactly as presented in the lecture material, and it also
worked well.  As for the car position calculation, it turned
out to be quite trivial. 
</p>
</div>
</li>

<li><a id="sec-1-3-1-6" name="sec-1-3-1-6"></a>Buffering<br  /><div class="outline-text-5" id="text-1-3-1-6">
<p>
Using a ring-buffer with the Python <a href="https://docs.python.org/2/library/collections.html#collections.deque"><code>deque</code></a> data structure
along with the Numpy <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.average.html#numpy-average"><code>average</code></a> function made it <i>very</i> easy to
implement a weighted average over some number of previous
frames.  Not only did this smooth out the line detections,
lane drawings, and distance calculations, it also had the
added benefit of significantly increasing the robustness of
the whole pipeline.  Without buffering&#x2014;and without a
mechanism for identifying and discarding bad detections&#x2014;the
lane would often bend and swirl in odd directions as it became
confused by spurious data from shadows, road discolorations,
etc.  With buffering <b>almost</b> all of that went away, even
without discarding bad detections.  If you pay close attention
to the video, near the very end at around the 48s mark, the
drawn lane is slightly attracted to and bends slightly toward
the black car that is passing on the right.  Without
buffering, this was a significant problem.  With more work on
the combination of gradient and color thresholds and perhaps
by discarding bad detections this problem would have been
eliminated.  However, I found that most of it could be
banished simply with buffering.  
</p>
</div>
</li>

<li><a id="sec-1-3-1-7" name="sec-1-3-1-7"></a>Python Generators<br  /><div class="outline-text-5" id="text-1-3-1-7">
<p>
I continue to be pleased with the ease of composition in a
functional style that is enabled by use of <a href="http://davidaventimiglia.com/python_generators.html">Python generators</a>.
Wrapping generators for filenames, images, and the output of
other functions in the <a href="https://docs.python.org/3/library/itertools.html#itertools.cycle"><code>cycle</code></a> generator from <a href="https://docs.python.org/3/library/itertools.html">itertools</a> was a
mainstay, especially for the 6 test images.  This was because
I could cycle through the processed images either one by one,
or in batches of 6, right in the Python interpreter.  It was
very effective for debugging. 
</p>
</div>
</li></ul>
</div>

<div id="outline-container-sec-1-3-2" class="outline-4">
<h4 id="sec-1-3-2">What Did Not Work Well</h4>
<div class="outline-text-4" id="text-1-3-2">
</div><ul class="org-ul"><li><a id="sec-1-3-2-1" name="sec-1-3-2-1"></a>Gradient Thresholding<br  /><div class="outline-text-5" id="text-1-3-2-1">
<p>
I found it very difficult to coax much usable signal out of
the gradient calculations and was grateful that I could get by
without them.
</p>

<p>
Moreover, the gradient calculations I was performing added
<i>significant</i> computational overhead.  With gradient
thresholding and color thresholding it took approximately 15
minutes to process the project video.  With just color
thresholding I cut that time by a third, down to just 5
minutes.  No doubt some of this is do to the <code>arctan2</code>
function that computes the gradient direction, since <code>arctan2</code>
is known to be an expensive operation.  Nevertheless, the
profiler that I used did show significant time spent just in
the <code>sobel</code> operation as well.
</p>
</div>
</li></ul>
</div>

<div id="outline-container-sec-1-3-3" class="outline-4">
<h4 id="sec-1-3-3">What Could Be Improved</h4>
<div class="outline-text-4" id="text-1-3-3">
</div><ul class="org-ul"><li><a id="sec-1-3-3-1" name="sec-1-3-3-1"></a>Gradient and Color Thresholding<br  /><div class="outline-text-5" id="text-1-3-3-1">
<p>
There is almost as much art as there is science in
highlighting the lane lines (and <i>just</i> the lane lines)
robustly, in a wide range of conditions.  There are many
hyper-parameters and many many ways to combine these
operations.  I spent considerable time on this aspect of the
project yet never stumbled upon a "magic" combination that
worked very well in all conditions.  At present I have settled
for simple combination of color thresholds with no gradients
and only middling performance.  I'm sure I can do better.
</p>
</div>
</li>

<li><a id="sec-1-3-3-2" name="sec-1-3-3-2"></a>Discarding Bad Detections<br  /><div class="outline-text-5" id="text-1-3-3-2">
<p>
I started down the path of discarding bad line detections,
which is why I adapted the Numpy <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html#numpy-polyfit"><code>polyfit</code></a> function to return
residuals, but on the project video at least I found that with
buffering I did not need to do this.  Nevertheless, I think it
would be prudent to add it in order to make the pipeline more
robust.  
</p>
</div>
</li>

<li><a id="sec-1-3-3-3" name="sec-1-3-3-3"></a>Code Refactoring<br  /><div class="outline-text-5" id="text-1-3-3-3">
<p>
There is a great deal of code duplication, especially between
the <code>detect_lines_sliding_window</code> and <code>detect_lines</code>
functions.  Also, the car position calculation probably should
not be performed in the <code>draw_lane</code> function.  These blemishes
are far from fatal and removing them is not part of the
project, but they make that part of the code unwieldy,
difficult to maintain, and somewhat difficult to read.  I
would definitely refactor this portion of the code in
subsequent revisions.  
</p>
</div>
</li>

<li><a id="sec-1-3-3-4" name="sec-1-3-3-4"></a>Measuring Perspective<br  /><div class="outline-text-5" id="text-1-3-3-4">
<p>
As discussed above, the target region for the perspective
transform is hard-coded to be a rectangle from the bottom of
the image to the top, 300 pixels in from the left edge and 300
pixels in from the right edge.  While this worked well in the
end, it only was brought about by trial-and-error, and is not
very flexible.  It would be better to adapt the <code>measure_warp</code>
function so that the user has more freedom in specifying this
region.  
</p>
</div>
</li></ul>
</div>
</div>
</div>
</div>
<div id="postamble" class="status">
<p class="author">Author: David A. Ventimiglia (<a href="mailto:dventimi@gmail.com">dventimi@gmail.com</a>)</p>
<p class="date">Date: <span class="timestamp-wrapper"><span class="timestamp">&lt;2017-02-08&gt;</span></span></p>
<p class="creator"><a href="http://www.gnu.org/software/emacs/">Emacs</a> 24.5.1 (<a href="http://orgmode.org">Org</a> mode 8.2.10)</p>
<p class="validation"><a href="http://validator.w3.org/check?uri=referer">Validate</a></p>
</div>
</body>
</html>