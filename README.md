
<img width="600" height="262" alt="image" src="https://github.com/user-attachments/assets/b5c0f96a-290e-4f69-94a7-5d5493089d9a" />

Introduction
------------

This repository implements the algorithm that improves performance of the original algorithm for recreating game world maps. Original implementations is located in [this repository](https://github.com/kataklinger/rgc).

Improvements are in made in several segments of the algorithm:

1.  run-time performance
2.  reduction of frame "misplacement"
3.  artifact filtering

To achieve these goals there were several changes in approach to the problem. Changes range from implementation details to more fundamental modifications of the algorithm.

First and the most obvious change is switch from managed F# code to native C++ code. In addition to that, AVX2 is also employed to improve run-time performance of low-level operations on images.

The most fundamental change in the approach is that the algorithm no longer keeps a global view (keypoints database) of the world, nor it tries to match frames against it. Instead each successive frame is compared only to its predecessor and only relative change in position is calculated. Stream of consecutive frames that can be matched form a fragment, or part of the map. If a new frame cannot match the previous one, a new fragment is started. At the end, fragments are matched against each other and spliced together. This solves some issues when it comes to run-time performance, fluctuating keypoints and erratic placement of frames, but it also introduces new problems which are going to be discussed later in the article.

Additional improvements are also made to foreground extraction and artifact filtering to further reduce random pixels left from moving objects. To improve foreground extraction, the map building is split in two phases. The first one builds a map and applies only the most basic filtering. More involved filtering is left for the second phase that goes over the all frames again and tries to identify and remove all moving objects. Finally simple statistical analysis is applied to remove remaining artifacts - random pixels were not correctly identified as foreground.

Building Fragments
------------------

This version of the algorithm does not keep a single global map of the world, instead it tries to construct a map of local regions called fragments. Current position in the fragment is kept and updated with each new frame. To get the position of the next frame, the relative displacement (or offset) from the previous frame is calculated and the current position is updated by applying this. Content of the frame is then written to the fragment at the calculated position. If offset cannot be calculated, the current fragment is sealed and the frame is moved to a new fragment.

To calculate offset between two frames, keypoints are extracted from a frame and then they are matched against keypoints of the previous frame. Matching algorithm tries to ensure that two frames are actually next to each other and only then calculates the offset between them.

Flow of fragment collection/build is illustrated in the following diagram:

<img width="518" height="591" alt="image" src="https://github.com/user-attachments/assets/75c5e341-57a1-42d2-91a2-b78f79910c71" />

Fragment is a sequence of consecutive frames that are successfully matched and forms one piece of the world map. It is represented by matrix that stores history of each pixel. Every element of the matrix is a "dot" that tracks how many times each color appeared at the location. Adding frame (or "blitting") into the region just increases the count for the color that frame had for the pixel. Fragments can be "blended" into images. The process simply takes the most frequent color from each dot as the pixel value of the output image. Blending also outputs a bit mask that tells which dots actually observed frames/color data and which are "empty" and can be ignored/filtered. Each frame added to the fragment is also preserved in a compressed format with its location information, since this data will be needed again in later phases of the algorithm.

<img width="489" height="571" alt="image" src="https://github.com/user-attachments/assets/7bf4f91a-394e-4af7-80f1-8946535f94bd" />

`fgm` namespace wraps code dealing with fragment representation and manipulation. `fragment` class is the most important one.

`collector` from `frc` namespace represents the loop that collects and builds fragments from a stream of frames. Collector takes a feeder object that is producing frames. In the example application, the feeder produces frames by reading them from directory.

Keypoint Extraction
-------------------

Keypoint extraction has been changed significantly from the first version of the algorithm. To identify keypoints, the median filter is applied twice to the frame with two different window sizes: 3x3 and 5x5, then the two resulting images and the original image are compared.

We define _keypoint(i,j)_ as a predicate that is true if pixel is identified as keypoint (_originalij_ stands for pixel of the original image, while _medianij_ represents pixel of image that went through median filter):

$keypoint(i,j) \\Rightarrow original\_{ij} \\neq \\overset{3x3}{median}\_{ij} \\wedge \\overset{3x3}{median}\_{ij} \\neq \\overset{5x5}{median}\_{ij}$

_weight(i,j)_ is a function that maps keypoint to its weight for pixel _(i,j)_:

$$
\mathrm{weight}(i,j) =
\begin{cases}
2,\text{ if } keypoint(i,j) \land original_{ij} \neq \overset{5x5}{median}_{ij} \\
1,\text{ if } keypoint(i,j) \land original_{ij} = \overset{5x5}{median}_{ij} \\
0,\text{ otherwise}
\end{cases}
$$


| input image                                                                                                                        |                                                                                                                                    |
|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/51a97a2e-88ff-4c97-b3fc-43b7d0af0c66" /> |                                                                                                                                    |
| keypoints with weight = 1                                                                                                          | keypoints with weight = 2                                                                                                          |
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/a0884788-2ca6-4f9d-906f-414282374039" /> | <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/d1460231-e0a3-4759-9718-9e66a3b7fcf5" /> |
| keypoints combined                                                                                                                                                                                                                                                      |
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/a82c3e2d-949d-4aba-97e1-1ec0a0c4669c" />                                                                                                                                      |

Usually only keypoints with higher _weight_ value are used for matching and offset calculation, but in regions of frame sparsely populated with those, the algorithm falls back and uses keypoints with lower _weight_ as well. Once a keypoint is identified, its description is encoded by extracting pixels from the surrounding 5x5 window and concatenating weight to the resulting byte array to generate 104-bit code.

When extracting keypoints, the frame is "split" by a grid of a certain configuration. Each extracted keypoint is assigned to one or more regions of the grid. Later when the algorithm tries to match two frames it will only try to compare keypoints belonging to the same regions.

Median Filter for Native Images
-------------------------------

Native images are using native color codes for pixels. Actual numeric value of the color code does not correspond to its real RGB color or its intensity or anything "physical", so they are not suitable for doing filtering of any kind. To make images suitable for processing, color codes of pixels are converted to values that correspond to brightness of their native color (the format is called native ordered values). The conversion is done in the following way:

1.  native color code acts as input format
2.  RGB conversion is done using lookup table
3.  intensity of RGB color is calculated based on standard formula
4.  ordered value is equal to index of the color in the array of all colors sorted by their intensity

Before the median filter is applied, native image is converted to this "ordered value" format, then the filter is applied. This format can also be converted to the native one if needed.

Matching Frames
---------------

Keypoint matching works by calculating all offsets between each keypoint in the current frame and corresponding keypoints in the previous frame. Corresponding keypoints are the ones with the same code residing in the same regions of their frames. Process keeps the count of each distinct offset that appears.

If we let _KP_ be set of all keypoints of the previous frame and we define functions _code_ and _region_ that maps keypoint to its code and region to which it belongs, we can identify set _CP_ that contains all matched keypoints between two frames:

$C\_P(c) \\in \\{p \\in K\_P \\vert \\forall p\[\\mathrm{code}(p) = \\mathrm{code}(c) \\wedge \\mathrm{region}(p) = \\mathrm{region}(c)\]\\}$

We also define function _distθ_ that maps pair of keypoints to their in specified dimension as (_posθ_ is a function that maps keypoint to one coordinate of its location):

$\\mathrm{dist}\_{\\theta}(c,p) = \\mathrm{pos\_{\\theta}}(c) - \\mathrm{pos\_{\\theta}}(p) \\text{ where } \\theta \\in \\{x, y\\}$

Now we can build a sequence _MP(c)_ of all offsets which can be used for matching a single keypoint _c_ of the current frame with corresponding keypoints from the previous frame:

$M\_P(c)=\\langle \\langle \\Delta x, \\Delta y \\rangle \\in \\mathbb{Z}^2\\vert (\\forall p \\in C\_P(c))\[\\Delta x = \\mathrm{dist\_x}(c, p) \\wedge \\Delta y = \\mathrm{dist\_y}(c,p)\] \\rangle$

Finally the sequence _M_ of all possible offsets can be defined as:

$M = M\_P(c\_1) ^\\frown M\_P(c\_2) ^\\frown... ^\\frown M\_P(c\_n) \\text { where } c \\in K\_C$

Once all possible offset are defined, we can assign scores to them using _score_ function:

$\\mathrm{score}(\\Delta x, \\Delta y) = \\sum\_{i=1}^{|M|}\[M\_i=(\\Delta x, \\Delta y)\]$

<img width="462" height="278" alt="image" src="https://github.com/user-attachments/assets/717ced3c-5ed9-44c8-b260-b5512c7e44be" />

When matching two frames, keypoints are grouped in 8 regions. These regions are divided by a 4x2 grid. Each region has 16 pixels overlap with neighboring regions. Since keypoints matching is local to the frame region, overlapping is necessary to avoid exclusion of keypoints located near borders of the regions. If there are enough keypoints of weight 2 in the region, only those will be used. If a region is sparsely populated with such keypoints, matching will switch to using keypoints of weight 1.

<img width="387" height="304" alt="image" src="https://github.com/user-attachments/assets/275daf31-9bd7-4d8d-9908-66a47cdafd43" />

Once scores of all offsets for the regions are obtained, vote power is assigned to them. Top three offsets in each region are selected, based on their score. The offset with the highest score gets vote power of 3, while the 3rd ranked offset gets vote power of 1. Finally vote power of each offset is summed across the regions. Offset with greatest total power is declared as the offset of the whole frame, but only if its total power value is larger than then the next smaller one by at least 4 points (half the number of regions). This structure of voting power is established to prevent a single keypoint-rich region from overpowering all other sparsely populated regions and to remove possible ambiguity if there are multiple offsets with close value of vote power. This way multiple regions must agree on the actual offset of the whole frame and if they fail to do so, the frame will be declared as non-matching and a new fragment will be spawned.

<img width="387" height="243" alt="image" src="https://github.com/user-attachments/assets/87d8d4e5-c396-4fd1-af3e-92297719f97f" />

Fragment Splicing
-----------------

Output of the algorithm's first phase is a set of fragments. Each fragment represents part of the game world map made from consecutive frames that could be matched. These fragments have to be spliced together in the next step.

Process of splicing fragments into a map starts by matching each fragment against all others. Fragment matching works in a similar fashion to frame matching. Keypoint identification and extraction are the same, but there is no grid splitting fragment into regions that group keypoints (or stated differently grid is of size 1x1). Matching keypoints is also similar, but it has different policies to ensure that false matches are filtered. This process outputs a queue containing pairs of fragments that could be matched along with calculated offsets and matching scores.

Once the queue is built, the pair with the highest score is selected and its fragments are spliced together. Splicing is done by blitting one fragment into the other (blitting fragments preserve all pixel history information). Fusion produces a new fragment and the two source fragments have to be retired. It means the queue has to be purged of all pairs referencing one of the source fragments. After queue purge, this new fragment is matched against remaining fragments and matching pairs are added back to the queue. The splicing process loops back to select the next highest scoring pair and the process stops once the queue is empty (no pair of fragments has left that could be matched).

<img width="449" height="351" alt="image" src="https://github.com/user-attachments/assets/8affb323-f327-4146-8297-e7cdfbc613b1" />

### Matching Fragments

Keypoints are extracted from the fragment in the same way they are extracted from the frames. To extract keypoints, a fragment is blended to get the native image from the "dot" matrix. Keypoints are not grouped into regions like in case of frame matching, which means all keypoints in the fragment are compared against all keypoints in another fragment. Another difference to frame matching is that keypoints of all weights are used.

Each pair of matching keypoints produces a possible matching offset. Offset with the greatest count of matching keypoints is declared as a winner. Once the offset is selected, it has to be verified. To verify it, an overlapping region of the two fragments has to be calculated. This region of interest then is split into cells of fixed size and each keypoint from both fragments is assigned to a cell which encompasses the position of that keypoint. Matching is valid if at least 2/3 of cells in the regions of interest are populated by at least one matching keypoint. Cells not populated by any keypoints from both fragments are ignored from the total count of cells in the region of interest.

We choose a pair of keypoints that are matching, one from each fragment and label them as _l0_ and _r0_. Then offset of the frames can be defined as:

$\\Delta p\_x = \\mathrm{pos\_x}(l\_0) - \\mathrm{pos\_x}(r\_0), \\, \\Delta p\_y = \\mathrm{pos\_y}(l\_0) - \\mathrm{pos\_y}(r\_0)$ 

Next we define predicate _match(l,r)_ that is true if keypoint _l_ from the first fragment is matching keypoint _r_ from the second, under assumption that _l0_ and _r0_ are matching:

$loc\_\\theta(l, r) \\Leftrightarrow \\mathrm{pos\_\\theta}(l) - \\mathrm{pos\_\\theta}(r) = \\Delta p\_\\theta \\text{, where } \\theta \\text{ is } x, y \\\\ match(l,r) \\Leftrightarrow \\mathrm{code}(l) = \\mathrm{code}(r) \\land loc\_x(l, r) \\land loc\_y(l, r)$ 

We also need to define functions _celll(l)_ and _cellr(r)_ that maps a keypoint to a cell of region of interest to which it belongs (_λ_ represents the cell size):

$\\mathrm{cell\_l}(l) = \\left \\langle \\left \\lfloor \\frac{\\mathrm{pos\_x}(l) + \\Delta {p\_x}^-}{\\lambda} \\right \\rfloor, \\left \\lfloor \\frac{\\mathrm{pos\_y}(l) + \\Delta {p\_y}^-}{\\lambda} \\right \\rfloor \\right \\rangle \\\\ \\mathrm{cell\_r}(r) = \\left \\langle \\left \\lfloor \\frac{\\mathrm{pos\_x}(r) + \\Delta {p\_x}^+}{\\lambda} \\right \\rfloor, \\left \\lfloor \\frac{\\mathrm{pos\_y}(r) + \\Delta {p\_y}^+}{\\lambda} \\right \\rfloor \\right \\rangle$ 

We use _dx_ and _dy_ that represent height and width of region of interest (while width and height of fragments are represented by _widthl_, _widthr_, _heightl_ and _heightr_)

$d\_x = \\mathrm{min}(width\_l, width\_r - \\Delta p\_x) - \\Delta {p\_x}^+ \\\\ d\_y = \\mathrm{min}(height\_l, height\_r -\\Delta p\_y) - \\Delta {p\_y}^+$ 

Set _CO_ of all cells in the region of interest can be defined as:

$C\_O \\in \\left \\{\\langle x, y \\rangle \\in \\mathbb{N} \\times \\mathbb{N} \\vert x < \\left \\lceil \\frac{d\_x}{\\lambda} \\right \\rceil \\wedge y < \\left \\lceil \\frac{d\_y}{\\lambda} \\right \\rceil \\right \\}$ 

If we have _KL_ and _KR_ as sets of all keypoints belonging to these two fragments that we are matching, then set _CE_ of ignored cells is can be defined:

$C\_E \\in \\{c \\in C\_O \\vert \\neg \[\\exists l \\in K\_L, \\exists r \\in K\_R, (\\mathrm{cell\_l}(l) = c \\lor \\mathrm{cell\_r}(r) = c)\] \\}$ 

_CM_ is a set of all the cells that contains at least one pair of matching keypoints:

$C\_M \\in \\{c \\in C\_O \\vert \\exists l \\in K\_L, \\exists r \\in K\_R, \\mathrm{match}(l,r) \\wedge \\mathrm{cell\_l}(l) = c) \\}$ 

Now that we have all these cell sets defined we can define predicate _valid_ that is true if proposed offset (_Δpx, Δpy_) is valid:

$valid \\Rightarrow \\frac{\\vert C\_M \\vert}{\\vert C\_O \\setminus C\_E \\vert } \\geq \\frac{2}{3}$

<img width="596" height="424" alt="image" src="https://github.com/user-attachments/assets/1d14b2c0-5dee-4873-a98b-3c2357a7c7fb" />

If this offset cannot be validated, fragments are declared as non-matching and they are not considered as candidates for splicing.

`kpm` namespace is also home for the fragment matching algorithm which is implemented by another overload of `match` function.

Foreground Filtering
--------------------

Moving objects leave all kinds of artifacts over the map image, so in order to have a clean map they have to be filtered out.

Simplest form of foreground filtering is implemented by the fragment blending process. It takes the most frequent color of each pixel in a fragment and writes it to image output. The process removes most artifacts created by moving objects from the output, but there are still some remaining from objects that are moving at a slower pace, that are more static or that are moving in more confined spaces.

After splicing fragments, all historic data about each pixel will be captured in a single fragment. Once all the data is available, fragments can be blended to get a preliminary background that has most of the moving object removed. This background image will be used for identifying contours of foreground objects in the source frames.

The sequence of frames for each fragment is replayed one more time and frames are blitted into the new fragment. This time, frames are compared against the preliminary background of the original fragment. Each mismatched pixel is marked and contours to which such pixels belong are extracted and filtered out. The whole rectangle enclosing the foreground contour is removed from the frame and is not blitted into the target fragment.

This process will also exclude some contours belonging to the background. To limit consequences of this unwanted effect, only the contours whose sizes are comparable to native sprite sizes are candidates for filtering, all larger contours having are still blitted, even if they have mismatched pixels.

| background                                                                                                                         |                                                                                                                                    |
|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/069f4275-ef91-45e4-9496-b159074fd9c1" /> |                                                                                                                                    |
| input frame                                                                                                                        |                                                                                                                                    |
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/f9ab27db-62b7-4fa7-bc5f-3992a317f3e3" /> |                                                                                                                                    |
| extracted foreground                                                                                                               | foreground mask                                                                                                                    |
| <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/9700c8b8-e3a0-4a1b-a177-a673112c4325" /> | <img width="301" height="150" alt="image" src="https://github.com/user-attachments/assets/f9a96022-0b5f-4b52-8e1e-6f4728fd374f" /> |

This process is the reason (compressed) copies of source frames are kept in memory: so sequence could be replied. To save processing time , the position of each frame within the fragment is preserved. Fragment splicing affects position of these frames relative to fragment origin, but each time fragments are spliced, relative frame positions are adjusted. This is computationally much cheaper than trying to read and match frames against fragments again, but it consumes more memory since content of the frames have to be preserved.

Namespace `fde` contains `extractor` class that is responsible for extracting foreground contours from frame. `mask` function will generate a blit mask from extracted contours, or rather their enclosing rectangles.

`fdf` namespace has `filter` function that will do the filtering of the foreground.

Artifact Filtering
------------------

After removing the foreground with the process described in the previous section there is still going to be some artifact remaining. The remaining artifacts are identified by determining the frequency of each pixel sequence of a certain size . For each pixel in the fragment's image, a sequence of neighboring pixels is created and a count for each unique sequence is kept. Only pixels belonging to the same axis are used for forming the sequences. Sequence counting is done for both axes and the counts are kept separately. Then the number is added together for each pixel and those pixels with lower numbers are marked as potential artifacts.

We define _H<x, y>_ and _V<x, y>_ as horizontal and vertical sequences of colors assigned to pixels neighboring to _x,y_ like this (we use _n_ to represent size of the filter and _k_ is the number of preceding or succeeding pixels that are taken into the account by filter):

$k = \\left \\lfloor \\frac{n}{2} \\right \\rfloor \\\\H\_{\\langle x,y \\rangle} = \\left \\langle p\_{\\left \\langle x-k,y \\right \\rangle},p\_{\\left \\langle x-k+1, y \\right \\rangle}, ..., p\_{\\left \\langle x, y \\right \\rangle }, ..., p\_{\\left \\langle x+k-1, y \\right \\rangle }, p\_{\\left \\langle x+k, y \\right \\rangle } \\right \\rangle \\\\ V\_{\\langle x,y \\rangle} = \\left \\langle p\_{\\left \\langle x,y-k \\right \\rangle},p\_{\\left \\langle x, y-k+1 \\right \\rangle}, ..., p\_{\\left \\langle x, y \\right \\rangle }, ..., p\_{\\left \\langle x, y+k-1 \\right \\rangle }, p\_{\\left \\langle x, y+k \\right \\rangle } \\right \\rangle$ 

Then we define _P_ as set of all coordinates in the image (_width_ and _height_ as size of the fragment):

$P \\in \\{ \\langle x, y \\rangle \\in \\mathbb{N}^2 \\vert ( k \\leq x < width - k ) \\land ( k \\leq y < height - k )\\}$ 

Now we can define function _heat(x,y)_ that maps pixel _x,y_ to its heat value:

Finally we define predicate _artifact(x,y)_ that is true if pixel is potential artifact:

$\\mathrm{heat}(x,y) = \\frac{\\sum\_{p \\in P} \\left ( \\left \[H\_p = H\_{\\langle x,y \\rangle} \\right \] + \\left \[V\_p = V\_{\\langle x,y \\rangle} \\right \] \\right )}{2}$ 
$artifact(x,y) \\Rightarrow \\sqrt{\\mathrm{heat}(x,y)} > 4$

| <img width="592" height="138" alt="image" src="https://github.com/user-attachments/assets/e227b6a5-151a-4528-9d8c-9a54cf8182cd" /> |
|------------------------------------------------------------------------------------------------------------------------------------|
| input image                                                                                                                        |
| <img width="592" height="138" alt="image" src="https://github.com/user-attachments/assets/86e379e0-a58f-46fe-961e-e454ac364877" /> |
| heatmap                                                                                                                            |
| <img width="592" height="138" alt="image" src="https://github.com/user-attachments/assets/3da5e0a7-b0ae-4f3f-bf77-42b0a6c18b52" /> |
| identified artifacts                                                                                                               |

For each pixel that is a potential artifact, Gaussian blur-like filter is applied. This filter performs blurring on all historical data available for the pixel and its neighbors in the fragment. Each color is separated in its own plane and then filter is applied plane by plane. Once all plains are filtered, the color represented by the plane which has the greatest value is selected as pixel color. Planes representing colors which are not present in the history of the targeted pixel are ignored.

Standard Gaussian function _G(x,y)_ is defined as:

$G(x,y) = \\frac{1}{2 \\pi \\sigma ^ 2}e^{- \\frac {x^2 + y^2}{2 \\sigma^2}}$

We need auxiliary functions _count(x,y,z)_ and _total(x,y)_ that are mapping pixel's position to its history: number of samples in which specified color appeared as value of the pixel and total number of samples captured for the pixel. Value of each plane _z_ for pixel with coordinates _(x0, y0)_ is calculated as:

$B(z) = \\sum\_{x=x\_0-6\\sigma }^{x\_0+6\\sigma}\\sum\_{y=y\_0-6\\sigma }^{y\_0+6\\sigma} \\left \[ \\frac{\\mathrm{count}(x,y,z) \\ast G(x, y)}{\\mathrm{total}(x,y)} \\right \]$

With this we can calculate values of all planes in fragment's "dot" as (_N_ is total number of colors in the pallet):

$D = \\langle B(z)\\rangle\_{z=1}^{N}$

We select an element from _D_ such that _Di_ has the greatest value and we use the index of that element as the color value of the pixel.

`arf` namespace hosts `filter` function that performs filtering on the fragment and produces native image.

Contour Extraction
------------------

Process of extracting contours remains the same as in the original algorithm, so it will not be discussed further.

`ctr` and `cte` namespaces wrap code dealing with representation of contour and algorithm for their extraction. `ctr::contour` class is representation of a contour which is stored as a sequence of its edge pixels. `cte::extractor` class wraps contour extraction algorithm.

Active Window Scanning
----------------------

Active window scanning process remains largely unchanged. Each successive frame is scanned for changes from the previous one and each changed pixel is marked in a dedicated bitmap. This way the bitmap accumulates all the changes across time. With each new frame algorithm locates the largest contour in the bitmap. Once the largest contour's area becomes stable, its enclosing rectangle is declared as the active window.

`aws` namespace wraps active window scanning functionality. `scan` function goes over frames in the provided feed and tries to identify the window.

Supporting Stuff
----------------

Following sections are describing supporting code that is (re)used by higher-level algorithm code.

### Memory Allocation

For parts of the algorithm that operate on data local to the current frame, arena allocator is employed as another low-level optimization. For each frame a new arena is allocated that is slightly larger than the highest watermark of any of the previous frames. Memory arena is discarded once it is no longer needed. In some cases it is discarded when processing has finished for the current frame. In other cases, the arena has to be kept until the end of the successive frame (for example when the algorithm processes a stream of frames and has to match two frames to determine their offset).

<img width="402" height="161" alt="image" src="https://github.com/user-attachments/assets/fc9bf673-04bb-4be1-b56a-83be145680d7" />

Code dealing with memory management and allocators is located in `all` namespace. `frame_allocator` and `memory_pool` classes implement arena allocator pattern. `memory_stack` and `memory_swing` can be used for managing arenas that are overlapping in time.

### Common Datatypes

`cdt` namespace contains all common data types that are representing spatial information used in various parts of the algorithm.

Following types are defined:

*   `point` - class templates that represents coordinate in two dimensions space (usually represents location of a pixel in the image)
*   `offset_t` - alias that represent signed coordinate point type or a delta (usually represents relative location of two frames)
*   `dimensions` - class template that represent two dimensional size (usually represents size of an image/fragment)
*   `limits` - class template that represent single dimensional limits
*   `region` - class template that represent two dimensional limits (usually represents margins of a frame/fragment or region of overlap between frame/fragment)

### Matrix Representation and Manipulation

Code dealing with matrices is located in `mrl` namespace. `matrix` is class that encapsulates a matrix and provides some basic operation. It is used for representing images in different formats. This type is allocator aware.

### Image and Pixel Types

`cpl` namespace defines all necessary pixel types and `sid` namespace contains some standard image types based on those pixel types.

Several pixel formats are provided:

*   `cpl::rgb_cc` - (RGB, component color) pixel of a single RGB color component
*   `cpl::rgb_bc` - (RGB, blended color) RGB24 pixel
*   `cpl::rgb_pc` - (RGB, packed color) pixel represented by tuple of three RGB components
*   `cpl::nat_cc` - (native, color coded) pixel as color code encoded by the value on the native platform
*   `cpl::nat_ov` - (native, ordered value) pixel that stores index of color in native color intensity table
*   `cpl::grs_iv` - (greyscale, intensity value) pixel stores intensity as floating point value
*   `cpl::mon_bv` - (monochrome, bit value) monochrome pixel

There are also utilities that perform conversions between pixel formats (names are self-explanatory):

|                        |                      |                      |
|------------------------|----------------------|----------------------|
| `native_to_blend`      | `pack_to_blend`      | `intensity_to_pack`  |
| `native_to_ordered`    | `pack_to_intentisy`  | `intensity_to_blend` |
| `native_to_intensity`  | `pack_to_intensity`  |                      |
| `ordered_to_native`    | `blend_to_pack`      |                      |
| `ordered_to_blend`     | `blend_to_intensity` |                      |
| `ordered_to_intensity` |                      |                      |

Image is nothing more than a matrix of specific types of pixels. `sid` has two namespaces: `mon` and `nat`. They define two most commonly used image formats: monochromatic and native. Each name space has `dimg_t` alias that defines image using default allocator and `aimg_t` templated alias which takes allocator type as its parameter.

Image Compression
-----------------

`icd` namespace has definitions of interface for services that provides image (de)compression needed by the algorithm and `nic` namespace contains implementation of these interfaces for (de)compression of images in native image format.

`nic::compress` and `nic::decompress` offers a rather simple and not so efficient way to compress native images.

`icd::compressor` and `icd::decompressor` concepts are provided as interface for (de)compression algorithm so more efficient implementation (in terms of size at least) can be provided.

Frame Feed and Map Builder
--------------------------

Interface of frame feed, on which the rest of the algorithm relies as source of frames is defined in `ifd` namespace. Namespace `mpb` contains map builder.

`ifd::feeder` is a concept that defines the interface of the feeder that is used by active window scanner or frame collector to read a sequence of frames. Feeder returns `ifd::frame` which contains native image and frame order number.

`mpb::builder` class sits at the top and unifies all the steps of the algorithm and support classes. It takes an instance of an adapter object which has to be provided by the application. It has to provide frame feed, image compression callback and other settings to the algorithm. Adapter also offers callbacks for the application to hook into the execution of the algorithm.

Utilities
---------

`nil` namespace has utilities for storing, loading and converting images in native format. On the other hand `ful` namespace contains utilities to deal with fragments.

`nil::read_raw` function loads native image from the file and `nil::write_png` writes native image in PNG format. `ful::read` loads from, while `ful::write` stores collection of fragments to the disk.
