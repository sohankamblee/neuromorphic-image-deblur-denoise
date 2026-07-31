# IMPLEMENTATION_SPEC.md

**Project:** Neuromorphic Imaging With Joint Image Deblurring and Event
Denoising\
**Paper:** IEEE TIP 2024\
**Implementation Language:** Python 3.x

------------------------------------------------------------------------

# 1. Objective

Implement the optimization-based Algorithm 1 from the paper in Python,
reproduce representative results, and prepare material for MTP
supervisor approval.

**Current Priority** - Working implementation - Correct mathematical
mapping - Modular code - Reproducible results

**Not a Priority (for now)** - Perfect documentation -
Production-quality software - Extensive GitHub polishing

------------------------------------------------------------------------

# 2. Inputs

### Image

-   Blurry grayscale APS image `B`

### Event Stream

Each event:

    (x, t, p)

where

-   `x` : pixel location
-   `t` : timestamp
-   `p` : polarity (+1/-1)

### Hyperparameters

-   τ
-   α
-   β
-   γ
-   σ
-   ω
-   μ
-   ν
-   l_max
-   γ_max

------------------------------------------------------------------------

# 3. Outputs

-   Deblurred image `S`
-   Blur kernel `k`
-   Denoised event stream `Ė`

------------------------------------------------------------------------

# 4. Variable Dictionary

  Paper   Meaning              Python Variable
  ------- -------------------- -----------------
  B       Blurry image         blurry_img
  S       Latent sharp image   latent_img
  k       Blur kernel          blur_kernel
  E       Raw events           raw_events
  Iτ      Event prior          event_prior
  z       Auxiliary variable   aux_grad
  g       Gradient mask        gradient_mask
  Ė       Denoised events      denoised_events

------------------------------------------------------------------------

# 5. Computational Pipeline

    Load Image
          │
    Load Events
          │
    Compute Event Prior
          │
    Initialize S and k
          │
    Repeat
        ├── Update auxiliary variable z
        ├── Update latent image S
        ├── Update gradient mask g
        ├── Denoise events
        ├── Recover neighbouring events
        └── Update blur kernel k
          │
    Save Results

------------------------------------------------------------------------

# 6. Equation Mapping

  Paper Equation   Purpose                              Python Module
  ---------------- ------------------------------------ ----------------------
  (1)-(9)          Event representation & event prior   event_processing.py
  (15)-(21)        Latent image optimization            latent_image.py
  (22)-(23)        Blur kernel estimation               kernel_estimation.py
  (24)-(26)        Event denoising                      event_denoising.py
  Algorithm 1      Overall optimization loop            main.py

------------------------------------------------------------------------

# 7. Initial Project Structure

    neuromorphic_deblur/

    docs/
        IMPLEMENTATION_SPEC.md

    data/

    results/

    src/
        event_processing.py
        latent_image.py
        kernel_estimation.py
        event_denoising.py
        utils.py
        main.py

    requirements.txt
    README.md

------------------------------------------------------------------------

# 8. Required Libraries

-   numpy
-   scipy
-   opencv-python
-   matplotlib
-   tqdm

(Optional) - scikit-image

------------------------------------------------------------------------

# 9. Implementation Order

1.  Repository setup
2.  Event processing
3.  Latent image update
4.  Auxiliary variable update
5.  Blur kernel estimation
6.  Event denoising
7.  Main optimization loop
8.  Run experiments
9.  Compare with paper

------------------------------------------------------------------------

# 10. Progress Tracker

-   [x] Paper studied
-   [x] Mathematical understanding
-   [x] Implementation strategy
-   [x] Initial specification

Next milestone:

**Create repository skeleton and implement `event_processing.py`.**

------------------------------------------------------------------------

This document is intentionally concise. It exists to guide
implementation quickly. A detailed engineering design document will be
created after a working implementation is achieved.
